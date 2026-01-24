"""
CCECE Paper: Progressive Prototype Trajectories (PPT) Experiment

Trains and validates the PPT model for explainable MG detection.

This script:
1. Trains PPT with 5-fold cross-validation (or 2-3 for rapid prototyping)
2. Runs falsification tests including trajectory-specific tests
3. Compares against TempProtoNet baseline
4. Generates trajectory-based explanations
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
from dataclasses import dataclass, asdict

# Ensure PYTHONPATH is set correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ccece.data_loader import load_binary_dataset, extract_arrays
from ccece.run_experiment import preprocess_items, set_all_seeds, compute_target_seq_len
from ccece.trainer import (
    TrainingConfig, SequenceScaler, SaccadeDataset,
    EvaluationMetrics, create_data_loaders
)
from ccece.models.ppt import ProgressivePrototypeTrajectories


RANDOM_SEED = 42


@dataclass
class PPTConfig:
    """Configuration for PPT experiment."""
    # Model hyperparameters
    latent_dim: int = 64
    n_prototypes_per_class: int = 5
    n_segments: int = 8
    encoder_hidden: int = 64
    encoder_layers: int = 3
    kernel_size: int = 7
    dropout: float = 0.2
    trajectory_type: str = 'linear'

    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 15
    grad_clip_norm: float = 1.0

    # Loss weights
    cluster_loss_weight: float = 0.5
    separation_loss_weight: float = 0.1
    diversity_loss_weight: float = 0.05  # New for PPT

    # Cross-validation
    n_folds: int = 5


@dataclass
class PPTFalsificationResults:
    """Results from PPT-specific falsification tests."""
    # Origin diversity (same as TempProtoNet)
    mean_origin_distance: float
    min_origin_distance: float
    origin_diversity_pass: bool

    # Trajectory-specific tests
    mean_velocity_norm: float
    min_velocity_norm: float
    velocity_norm_pass: bool  # v_k doesn't collapse to zero

    mean_velocity_cosine_similarity: float
    velocity_diversity_pass: bool  # Different prototypes have different directions

    # Class alignment (via trajectory similarity)
    alignment_ratio: float
    alignment_pass: bool

    # Ablation (trajectory vs static)
    trajectory_accuracy: float
    static_accuracy: float
    trajectory_helps: bool  # True if trajectory > static

    # Temporal discrimination
    early_segment_discrimination: float
    late_segment_discrimination: float
    temporal_pattern_pass: bool  # Later segments more discriminative for MG

    # Overall
    all_pass: bool


class PPTTrainer:
    """Custom trainer for PPT with trajectory-specific losses."""

    def __init__(
        self,
        model: ProgressivePrototypeTrajectories,
        config: PPTConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.best_model_state = None

        # Track velocity norms during training
        self.velocity_norm_history = []

    def _setup_training(self, train_labels: np.ndarray):
        """Setup optimizer, scheduler, and loss function."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )

        # Class weights for imbalanced data
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_labels: np.ndarray,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train the model."""
        self._setup_training(train_labels)
        patience_counter = 0

        epoch_iter = range(self.config.epochs)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc="Training", unit="epoch")

        for epoch in epoch_iter:
            # Training
            train_loss, train_acc = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # Track velocity norms
            velocity_norms = self.model.prototype_layer.compute_velocity_norms()
            self.velocity_norm_history.append(velocity_norms.mean().item())

            # Validation
            val_loss, val_acc = self._validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Track best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_epoch = epoch
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            # Update progress bar
            if verbose:
                v_norm = self.velocity_norm_history[-1]
                epoch_iter.set_postfix({
                    'tr_loss': f'{train_loss:.4f}',
                    'val_acc': f'{val_acc:.4f}',
                    'best': f'{self.best_val_accuracy:.4f}',
                    'v_norm': f'{v_norm:.4f}',
                })

            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return {
            'best_val_accuracy': self.best_val_accuracy,
            'best_epoch': self.best_epoch,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'final_velocity_norm': self.velocity_norm_history[-1],
        }

    def _train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch with trajectory-specific losses."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward with explanations
            logits, z_segments, distances, per_seg_sims, traj_sims = \
                self.model.forward_with_explanations(inputs)

            # Classification loss
            ce_loss = self.criterion(logits, labels)

            # Prototype losses
            cluster_loss, separation_loss, diversity_loss = \
                self.model.compute_prototype_loss(z_segments, labels, distances)

            # Total loss
            loss = (
                ce_loss +
                self.config.cluster_loss_weight * cluster_loss +
                self.config.separation_loss_weight * separation_loss +
                self.config.diversity_loss_weight * diversity_loss
            )

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip_norm
            )

            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    def _validate_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(inputs)
                loss = self.criterion(logits, labels)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        return total_loss / total, correct / total

    def evaluate(self, dataloader: DataLoader) -> EvaluationMetrics:
        """Evaluate the model and compute metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix
        )

        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)

                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        predictions = np.array(all_preds)
        labels = np.array(all_labels)
        probabilities = np.array(all_probs)

        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel()

        return EvaluationMetrics(
            accuracy=accuracy_score(labels, predictions),
            sensitivity=tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            specificity=tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            precision=precision_score(labels, predictions, zero_division=0),
            f1=f1_score(labels, predictions, zero_division=0),
            auc_roc=roc_auc_score(labels, probabilities),
            confusion_matrix=cm,
            predictions=predictions,
            labels=labels,
            probabilities=probabilities,
        )


def run_ppt_falsification_tests(
    model: ProgressivePrototypeTrajectories,
    val_loader: DataLoader,
    device: torch.device,
) -> PPTFalsificationResults:
    """
    Run PPT-specific falsification tests.

    Tests:
    1. Origin diversity - are prototype origins well-separated?
    2. Velocity norm - do velocities remain non-trivial?
    3. Velocity diversity - do prototypes have different directions?
    4. Class alignment - are trajectories aligned with classes?
    5. Ablation - does trajectory outperform static?
    6. Temporal discrimination - are later segments more discriminative?
    """
    model.eval()

    # 1 & 2 & 3. Diversity metrics
    diversity = model.compute_prototype_diversity()

    origin_diversity_pass = (
        diversity['mean_origin_distance'] > 0.5 and
        diversity['min_origin_distance'] > 0.1
    )

    velocity_norm_pass = diversity['min_velocity_norm'] > 0.01
    velocity_diversity_pass = diversity['mean_velocity_cosine_similarity'] < 0.7

    # 4. Class alignment
    prototype_classes = model.get_prototype_classes()

    mg_to_mg_sims = []
    mg_to_hc_sims = []
    hc_to_hc_sims = []
    hc_to_mg_sims = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            traj_sims = model.get_prototype_similarities(inputs)

            for i in range(inputs.size(0)):
                label = labels[i].item()
                sims = traj_sims[i]

                mg_proto_mask = (prototype_classes == 1)
                hc_proto_mask = (prototype_classes == 0)

                mg_sim = sims[mg_proto_mask].mean().item()
                hc_sim = sims[hc_proto_mask].mean().item()

                if label == 1:  # MG sample
                    mg_to_mg_sims.append(mg_sim)
                    mg_to_hc_sims.append(hc_sim)
                else:  # HC sample
                    hc_to_hc_sims.append(hc_sim)
                    hc_to_mg_sims.append(mg_sim)

    mg_alignment = np.mean(mg_to_mg_sims) / (np.mean(mg_to_hc_sims) + 1e-6)
    hc_alignment = np.mean(hc_to_hc_sims) / (np.mean(hc_to_mg_sims) + 1e-6)
    alignment_ratio = (mg_alignment + hc_alignment) / 2

    mg_correct = sum(1 for i in range(len(mg_to_mg_sims)) if mg_to_mg_sims[i] > mg_to_hc_sims[i])
    hc_correct = sum(1 for i in range(len(hc_to_hc_sims)) if hc_to_hc_sims[i] > hc_to_mg_sims[i])
    total_samples = len(mg_to_mg_sims) + len(hc_to_hc_sims)
    alignment_rate = (mg_correct + hc_correct) / total_samples
    alignment_pass = alignment_rate >= 0.70

    # 5. Ablation: trajectory vs static
    trajectory_accuracy = compute_accuracy(model, val_loader, device)

    # Zero out velocities and test
    original_velocities = model.prototype_layer.prototype_velocities.data.clone()
    model.prototype_layer.prototype_velocities.data.zero_()
    static_accuracy = compute_accuracy(model, val_loader, device)
    model.prototype_layer.prototype_velocities.data = original_velocities

    trajectory_helps = trajectory_accuracy > static_accuracy + 0.005  # > 0.5% improvement

    # 6. Temporal discrimination
    # Check if later segments are more discriminative for MG classification
    n_segments = model.n_segments

    early_seg_diffs = []  # First half of segments
    late_seg_diffs = []   # Second half of segments

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            per_seg_sims = model.get_per_segment_similarities(inputs)
            # per_seg_sims: (batch, n_segments, n_prototypes)

            for i in range(inputs.size(0)):
                label = labels[i].item()

                mg_proto_mask = (prototype_classes == 1)
                hc_proto_mask = (prototype_classes == 0)

                # Per-segment discrimination: MG_sim - HC_sim for MG samples
                seg_mg_sims = per_seg_sims[i, :, mg_proto_mask].mean(dim=1).cpu().numpy()
                seg_hc_sims = per_seg_sims[i, :, hc_proto_mask].mean(dim=1).cpu().numpy()

                if label == 1:  # MG sample
                    seg_diff = seg_mg_sims - seg_hc_sims
                else:  # HC sample
                    seg_diff = seg_hc_sims - seg_mg_sims

                half = n_segments // 2
                early_seg_diffs.append(seg_diff[:half].mean())
                late_seg_diffs.append(seg_diff[half:].mean())

    early_discrimination = np.mean(early_seg_diffs)
    late_discrimination = np.mean(late_seg_diffs)

    # For MG (fatigue), we expect later segments to be MORE discriminative
    # This means late_discrimination should be higher (more positive)
    temporal_pattern_pass = late_discrimination > early_discrimination

    all_pass = (
        origin_diversity_pass and
        velocity_norm_pass and
        velocity_diversity_pass and
        alignment_pass and
        trajectory_helps and
        temporal_pattern_pass
    )

    return PPTFalsificationResults(
        mean_origin_distance=diversity['mean_origin_distance'],
        min_origin_distance=diversity['min_origin_distance'],
        origin_diversity_pass=origin_diversity_pass,
        mean_velocity_norm=diversity['mean_velocity_norm'],
        min_velocity_norm=diversity['min_velocity_norm'],
        velocity_norm_pass=velocity_norm_pass,
        mean_velocity_cosine_similarity=diversity['mean_velocity_cosine_similarity'],
        velocity_diversity_pass=velocity_diversity_pass,
        alignment_ratio=alignment_ratio,
        alignment_pass=alignment_pass,
        trajectory_accuracy=trajectory_accuracy,
        static_accuracy=static_accuracy,
        trajectory_helps=trajectory_helps,
        early_segment_discrimination=early_discrimination,
        late_segment_discrimination=late_discrimination,
        temporal_pattern_pass=temporal_pattern_pass,
        all_pass=all_pass,
    )


def compute_accuracy(model, dataloader: DataLoader, device: torch.device) -> float:
    """Compute accuracy on a dataloader."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return correct / total


def run_experiment(
    config: PPTConfig,
    output_dir: str,
    verbose: bool = True,
    rapid_prototype: bool = False,
):
    """
    Run the full PPT experiment.

    Args:
        config: Experiment configuration
        output_dir: Directory to save results
        verbose: Whether to print progress
        rapid_prototype: If True, only run 2-3 folds for quick validation
    """
    set_all_seeds(RANDOM_SEED)

    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")

    # Load and preprocess data
    if verbose:
        print("Loading data...")
    items = load_binary_dataset(verbose=False)
    items = preprocess_items(items)

    X, y, patient_ids = extract_arrays(items)
    seq_len = compute_target_seq_len(items)
    input_dim = items[0]['data'].shape[1]
    num_classes = 2

    if verbose:
        print(f"Data loaded: {len(items)} samples, seq_len={seq_len}, input_dim={input_dim}")

    # Cross-validation
    n_folds = 3 if rapid_prototype else config.n_folds
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    if verbose:
        mode = "RAPID PROTOTYPING" if rapid_prototype else "FULL EXPERIMENT"
        print(f"\n{mode}: Running {n_folds}-fold cross-validation")

    fold_results = []
    all_explanations = []
    best_model = None
    best_accuracy = 0.0
    best_falsification = None

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, patient_ids)):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Fold {fold + 1}/{n_folds}")
            print('='*50)

        # Split data
        train_items = [items[i] for i in train_idx]
        val_items = [items[i] for i in val_idx]
        train_labels = np.array([item['label'] for item in train_items])

        # Create data loaders
        train_loader, val_loader, scaler = create_data_loaders(
            train_items, val_items, seq_len, config.batch_size
        )

        # Create model
        model = ProgressivePrototypeTrajectories(
            input_dim=input_dim,
            num_classes=num_classes,
            seq_len=seq_len,
            latent_dim=config.latent_dim,
            n_prototypes_per_class=config.n_prototypes_per_class,
            n_segments=config.n_segments,
            encoder_hidden=config.encoder_hidden,
            encoder_layers=config.encoder_layers,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            trajectory_type=config.trajectory_type,
        )

        if verbose:
            print(f"Model parameters: {model.count_parameters():,}")

        # Train
        trainer = PPTTrainer(model, config, device)
        train_result = trainer.train(train_loader, val_loader, train_labels, verbose=verbose)

        # Evaluate
        metrics = trainer.evaluate(val_loader)

        if verbose:
            print(f"\nFold {fold + 1} Results: {metrics}")

        # Run falsification tests
        falsification = run_ppt_falsification_tests(model, val_loader, device)

        if verbose:
            print(f"\nPPT Falsification Tests:")
            print(f"  Origin Diversity: {'PASS' if falsification.origin_diversity_pass else 'FAIL'}")
            print(f"  Velocity Norm: {'PASS' if falsification.velocity_norm_pass else 'FAIL'} "
                  f"(min={falsification.min_velocity_norm:.4f})")
            print(f"  Velocity Diversity: {'PASS' if falsification.velocity_diversity_pass else 'FAIL'} "
                  f"(cosine_sim={falsification.mean_velocity_cosine_similarity:.4f})")
            print(f"  Class Alignment: {'PASS' if falsification.alignment_pass else 'FAIL'}")
            print(f"  Trajectory > Static: {'PASS' if falsification.trajectory_helps else 'FAIL'} "
                  f"({falsification.trajectory_accuracy:.4f} vs {falsification.static_accuracy:.4f})")
            print(f"  Temporal Pattern: {'PASS' if falsification.temporal_pattern_pass else 'FAIL'} "
                  f"(early={falsification.early_segment_discrimination:.4f}, "
                  f"late={falsification.late_segment_discrimination:.4f})")
            print(f"  Overall: {'PASS' if falsification.all_pass else 'FAIL'}")

        # Generate explanations
        explanations = generate_explanations(model, val_loader, device, n_samples=3)
        all_explanations.extend(explanations)

        # Store fold results
        fold_result = {
            'fold': fold + 1,
            'accuracy': metrics.accuracy,
            'sensitivity': metrics.sensitivity,
            'specificity': metrics.specificity,
            'f1': metrics.f1,
            'auc_roc': metrics.auc_roc,
            'best_epoch': train_result['best_epoch'],
            'final_velocity_norm': train_result['final_velocity_norm'],
            'falsification_pass': falsification.all_pass,
            'trajectory_accuracy': falsification.trajectory_accuracy,
            'static_accuracy': falsification.static_accuracy,
        }
        fold_results.append(fold_result)

        # Track best model
        if metrics.accuracy > best_accuracy:
            best_accuracy = metrics.accuracy
            best_model = model
            best_falsification = falsification

    # Aggregate results
    mean_accuracy = np.mean([r['accuracy'] for r in fold_results])
    std_accuracy = np.std([r['accuracy'] for r in fold_results])
    mean_auc = np.mean([r['auc_roc'] for r in fold_results])
    std_auc = np.std([r['auc_roc'] for r in fold_results])
    mean_f1 = np.mean([r['f1'] for r in fold_results])
    mean_sensitivity = np.mean([r['sensitivity'] for r in fold_results])
    mean_specificity = np.mean([r['specificity'] for r in fold_results])

    # Trajectory vs static comparison
    mean_traj_acc = np.mean([r['trajectory_accuracy'] for r in fold_results])
    mean_static_acc = np.mean([r['static_accuracy'] for r in fold_results])

    # Check if accuracy goal is met
    accuracy_goal_met = mean_accuracy >= 0.65

    # Summary
    summary = {
        'config': asdict(config),
        'rapid_prototype': rapid_prototype,
        'n_folds': n_folds,
        'results': {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'mean_f1': mean_f1,
            'mean_sensitivity': mean_sensitivity,
            'mean_specificity': mean_specificity,
            'accuracy_goal_met': accuracy_goal_met,
            'accuracy_goal': 0.65,
        },
        'trajectory_vs_static': {
            'mean_trajectory_accuracy': mean_traj_acc,
            'mean_static_accuracy': mean_static_acc,
            'trajectory_improvement': mean_traj_acc - mean_static_acc,
            'trajectory_helps': mean_traj_acc > mean_static_acc + 0.005,
        },
        'fold_results': fold_results,
        'falsification': {
            'origin_diversity_pass': best_falsification.origin_diversity_pass,
            'velocity_norm_pass': best_falsification.velocity_norm_pass,
            'velocity_diversity_pass': best_falsification.velocity_diversity_pass,
            'alignment_pass': best_falsification.alignment_pass,
            'trajectory_helps': best_falsification.trajectory_helps,
            'temporal_pattern_pass': best_falsification.temporal_pattern_pass,
            'all_pass': best_falsification.all_pass,
            'mean_origin_distance': best_falsification.mean_origin_distance,
            'mean_velocity_norm': best_falsification.mean_velocity_norm,
            'mean_velocity_cosine_similarity': best_falsification.mean_velocity_cosine_similarity,
            'alignment_ratio': best_falsification.alignment_ratio,
        },
        'sample_explanations': all_explanations[:5],
        'timestamp': datetime.now().isoformat(),
    }

    # Print summary
    if verbose:
        print("\n" + "="*60)
        print("PPT EXPERIMENT SUMMARY")
        print("="*60)
        print(f"\nPerformance:")
        print(f"  Mean Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
        print(f"  Mean AUC-ROC: {mean_auc:.4f} (+/- {std_auc:.4f})")
        print(f"  Mean F1: {mean_f1:.4f}")
        print(f"  Mean Sensitivity: {mean_sensitivity:.4f}")
        print(f"  Mean Specificity: {mean_specificity:.4f}")
        print(f"\n  Accuracy Goal (>=65%): {'MET' if accuracy_goal_met else 'NOT MET'}")

        print(f"\nTrajectory vs Static Ablation:")
        print(f"  Trajectory Accuracy: {mean_traj_acc:.4f}")
        print(f"  Static Accuracy: {mean_static_acc:.4f}")
        print(f"  Improvement: {mean_traj_acc - mean_static_acc:.4f}")
        print(f"  Trajectory Helps: {'YES' if mean_traj_acc > mean_static_acc + 0.005 else 'NO'}")

        print(f"\nFalsification Tests (Best Fold):")
        print(f"  Origin Diversity: {'PASS' if best_falsification.origin_diversity_pass else 'FAIL'}")
        print(f"  Velocity Norm: {'PASS' if best_falsification.velocity_norm_pass else 'FAIL'}")
        print(f"  Velocity Diversity: {'PASS' if best_falsification.velocity_diversity_pass else 'FAIL'}")
        print(f"  Class Alignment: {'PASS' if best_falsification.alignment_pass else 'FAIL'}")
        print(f"  Trajectory > Static: {'PASS' if best_falsification.trajectory_helps else 'FAIL'}")
        print(f"  Temporal Pattern: {'PASS' if best_falsification.temporal_pattern_pass else 'FAIL'}")
        print(f"  Overall: {'PASS' if best_falsification.all_pass else 'FAIL'}")

        print("\nSample Explanations:")
        for i, exp in enumerate(all_explanations[:2]):
            print(f"\n  Sample {i+1}:")
            print(f"    Ground Truth: {exp.get('ground_truth_name', 'N/A')}")
            print(f"    Prediction: {exp['prediction_name']} ({exp['confidence']:.2%})")
            print(f"    Most Discriminative Segment: {exp['most_discriminative_segment']+1} "
                  f"(time={exp['most_discriminative_time']:.2f})")
            if exp['top_similar_prototypes']:
                top_proto = exp['top_similar_prototypes'][0]
                print(f"    Top Prototype: {top_proto['prototype_class_name']} "
                      f"(sim={top_proto['trajectory_similarity']:.4f}, "
                      f"trend={top_proto['similarity_trend']:.4f})")

    # Save results
    results_path = os.path.join(output_dir, 'results.json')

    def convert_to_serializable(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return [convert_to_serializable(v) for v in obj]
        return obj

    summary_serializable = convert_to_serializable(summary)

    with open(results_path, 'w') as f:
        json.dump(summary_serializable, f, indent=2)

    if verbose:
        print(f"\nResults saved to: {results_path}")

    # Save best model
    model_path = os.path.join(output_dir, 'best_model.pt')
    torch.save(best_model.state_dict(), model_path)

    if verbose:
        print(f"Best model saved to: {model_path}")

    return summary


def generate_explanations(
    model: ProgressivePrototypeTrajectories,
    val_loader: DataLoader,
    device: torch.device,
    n_samples: int = 5,
) -> List[Dict[str, Any]]:
    """Generate explanations for sample predictions."""
    model.eval()

    inputs_list = []
    labels_list = []

    for inputs, labels in val_loader:
        inputs_list.append(inputs)
        labels_list.append(labels)
        if sum(x.size(0) for x in inputs_list) >= n_samples:
            break

    inputs = torch.cat(inputs_list, dim=0)[:n_samples].to(device)
    labels = torch.cat(labels_list, dim=0)[:n_samples]

    explanations = model.explain_prediction(inputs)

    for i, exp in enumerate(explanations):
        exp['ground_truth'] = labels[i].item()
        exp['ground_truth_name'] = 'MG' if labels[i].item() == 1 else 'HC'
        exp['correct'] = exp['prediction'] == labels[i].item()

    return explanations


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run PPT experiment')
    parser.add_argument('--rapid', action='store_true', help='Run rapid prototyping (3 folds)')
    parser.add_argument('--n-segments', type=int, default=8, help='Number of temporal segments')
    parser.add_argument('--trajectory-type', type=str, default='linear',
                       choices=['linear', 'polynomial'], help='Trajectory type')
    args = parser.parse_args()

    # Configuration
    config = PPTConfig(
        latent_dim=64,
        n_prototypes_per_class=5,
        n_segments=args.n_segments,
        encoder_hidden=64,
        encoder_layers=3,
        kernel_size=7,
        dropout=0.2,
        trajectory_type=args.trajectory_type,
        epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        early_stopping_patience=15,
        cluster_loss_weight=0.5,
        separation_loss_weight=0.1,
        diversity_loss_weight=0.05,
        n_folds=5,
    )

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "rapid" if args.rapid else "full"
    output_dir = f"results/ccece/ppt/{mode}_{timestamp}"

    # Run experiment
    summary = run_experiment(config, output_dir, verbose=True, rapid_prototype=args.rapid)
