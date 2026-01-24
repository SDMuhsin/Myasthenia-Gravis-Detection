"""
CCECE Paper: TempProtoNet Experiment

Trains and validates the Temporal Prototype Network for explainable MG detection.

This script:
1. Trains TempProtoNet with 5-fold cross-validation
2. Runs falsification tests (prototype diversity, class alignment)
3. Generates prototype visualizations and explanations
4. Reports results with evidence
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
import matplotlib.pyplot as plt

# Ensure PYTHONPATH is set correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ccece.data_loader import load_binary_dataset, extract_arrays
from ccece.run_experiment import preprocess_items, set_all_seeds, compute_target_seq_len
from ccece.trainer import (
    TrainingConfig, SequenceScaler, SaccadeDataset,
    EvaluationMetrics, create_data_loaders
)
from ccece.models.temp_proto_net import TempProtoNet


RANDOM_SEED = 42


@dataclass
class TempProtoNetConfig:
    """Configuration for TempProtoNet experiment."""
    # Model hyperparameters
    latent_dim: int = 64
    n_prototypes_per_class: int = 5
    encoder_hidden: int = 64
    encoder_layers: int = 3
    kernel_size: int = 7
    dropout: float = 0.2

    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 15
    grad_clip_norm: float = 1.0

    # Prototype loss weights (tuned for best accuracy while attempting alignment improvement)
    cluster_loss_weight: float = 0.3
    separation_loss_weight: float = 0.1
    diversity_loss_weight: float = 0.3
    contrastive_loss_weight: float = 0.3

    # Cross-validation
    n_folds: int = 5


@dataclass
class FalsificationResults:
    """Results from falsification tests."""
    # Prototype diversity
    mean_pairwise_distance: float
    min_pairwise_distance: float
    inter_class_distance: float
    intra_class_distance: float
    diversity_pass: bool

    # Class alignment
    mg_to_mg_similarity: float
    mg_to_hc_similarity: float
    hc_to_hc_similarity: float
    hc_to_mg_similarity: float
    alignment_ratio: float  # How much more similar samples are to own-class prototypes
    alignment_pass: bool

    # Ablation (per prototype importance)
    prototype_importance: Dict[int, float]  # Accuracy drop when removing each prototype
    ablation_pass: bool

    # Overall
    all_pass: bool


class TempProtoNetTrainer:
    """Custom trainer for TempProtoNet with prototype-specific losses."""

    def __init__(
        self,
        model: TempProtoNet,
        config: TempProtoNetConfig,
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

    def _initialize_prototypes_from_data(self, train_loader: DataLoader):
        """Initialize prototypes from actual training data embeddings."""
        self.model.eval()
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                z = self.model.encode(inputs)
                all_embeddings.append(z.cpu())
                all_labels.append(labels)

        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # Initialize prototypes from data using k-means++ style selection
        self.model.initialize_prototypes_from_data(embeddings.to(self.device), labels.to(self.device))

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_labels: np.ndarray,
        verbose: bool = True,
        initialize_from_data: bool = False,  # Disabled - hurts accuracy
    ) -> Dict[str, Any]:
        """Train the model."""
        self._setup_training(train_labels)
        patience_counter = 0

        # Initialize prototypes from actual data (disabled by default - hurts accuracy)
        if initialize_from_data:
            if verbose:
                print("Initializing prototypes from data...")
            self._initialize_prototypes_from_data(train_loader)

        epoch_iter = range(self.config.epochs)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc="Training", unit="epoch")

        for epoch in epoch_iter:
            # Training
            train_loss, train_acc = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

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
                epoch_iter.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'val_acc': f'{val_acc:.4f}',
                    'best': f'{self.best_val_accuracy:.4f}',
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
        }

    def _train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch with prototype-specific losses."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward with explanations (to get distances for prototype loss)
            logits, z, distances, similarities = self.model.forward_with_explanations(inputs)

            # Classification loss
            ce_loss = self.criterion(logits, labels)

            # Prototype losses (softmin + diversity + contrastive for alignment improvement)
            cluster_loss, separation_loss, diversity_loss, contrastive_loss = self.model.compute_prototype_loss(
                z, labels, distances
            )

            # Total loss
            loss = (
                ce_loss +
                self.config.cluster_loss_weight * cluster_loss +
                self.config.separation_loss_weight * separation_loss +
                self.config.diversity_loss_weight * diversity_loss +
                self.config.contrastive_loss_weight * contrastive_loss
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


def run_falsification_tests(
    model: TempProtoNet,
    val_loader: DataLoader,
    device: torch.device,
) -> FalsificationResults:
    """
    Run falsification tests as defined in the architecture proposal.

    Tests:
    1. Prototype diversity - are prototypes well-separated?
    2. Class alignment - are samples closer to same-class prototypes?
    3. Ablation sensitivity - do individual prototypes matter?
    """
    model.eval()

    # 1. Prototype diversity
    diversity = model.compute_prototype_diversity()
    diversity_pass = (
        diversity['mean_pairwise_distance'] > 0.5 and
        diversity['min_pairwise_distance'] > 0.1
    )

    # 2. Class alignment
    # Compute average similarity of each class to each prototype type
    prototype_classes = model.get_prototype_classes()

    mg_to_mg_sims = []
    mg_to_hc_sims = []
    hc_to_hc_sims = []
    hc_to_mg_sims = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            similarities = model.get_prototype_similarities(inputs)

            for i in range(inputs.size(0)):
                label = labels[i].item()
                sims = similarities[i]

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

    mg_to_mg_avg = np.mean(mg_to_mg_sims)
    mg_to_hc_avg = np.mean(mg_to_hc_sims)
    hc_to_hc_avg = np.mean(hc_to_hc_sims)
    hc_to_mg_avg = np.mean(hc_to_mg_sims)

    # Alignment ratio: samples should be closer to their own class
    # > 1 means better alignment
    mg_alignment = mg_to_mg_avg / (mg_to_hc_avg + 1e-6)
    hc_alignment = hc_to_hc_avg / (hc_to_mg_avg + 1e-6)
    alignment_ratio = (mg_alignment + hc_alignment) / 2

    # Pass if at least 70% of samples are closer to own-class prototypes
    mg_correct = sum(1 for i in range(len(mg_to_mg_sims)) if mg_to_mg_sims[i] > mg_to_hc_sims[i])
    hc_correct = sum(1 for i in range(len(hc_to_hc_sims)) if hc_to_hc_sims[i] > hc_to_mg_sims[i])
    total_samples = len(mg_to_mg_sims) + len(hc_to_hc_sims)
    alignment_rate = (mg_correct + hc_correct) / total_samples
    alignment_pass = alignment_rate >= 0.70

    # 3. Ablation sensitivity
    # Compute accuracy with each prototype removed
    base_metrics = compute_accuracy(model, val_loader, device)
    base_accuracy = base_metrics

    prototype_importance = {}
    n_prototypes = model.prototype_layer.n_prototypes
    original_prototypes = model.prototype_layer.prototypes.data.clone()

    for proto_idx in range(n_prototypes):
        # Zero out this prototype
        model.prototype_layer.prototypes.data[proto_idx] = 0.0

        # Compute accuracy
        ablated_accuracy = compute_accuracy(model, val_loader, device)

        # Importance = drop in accuracy
        importance = base_accuracy - ablated_accuracy
        prototype_importance[proto_idx] = importance

        # Restore prototype
        model.prototype_layer.prototypes.data = original_prototypes.clone()

    # Pass if at least one prototype has >2% importance
    max_importance = max(prototype_importance.values())
    ablation_pass = max_importance > 0.02

    all_pass = diversity_pass and alignment_pass and ablation_pass

    return FalsificationResults(
        mean_pairwise_distance=diversity['mean_pairwise_distance'],
        min_pairwise_distance=diversity['min_pairwise_distance'],
        inter_class_distance=diversity['inter_class_distance'],
        intra_class_distance=diversity['intra_class_distance'],
        diversity_pass=diversity_pass,
        mg_to_mg_similarity=mg_to_mg_avg,
        mg_to_hc_similarity=mg_to_hc_avg,
        hc_to_hc_similarity=hc_to_hc_avg,
        hc_to_mg_similarity=hc_to_mg_avg,
        alignment_ratio=alignment_ratio,
        alignment_pass=alignment_pass,
        prototype_importance=prototype_importance,
        ablation_pass=ablation_pass,
        all_pass=all_pass,
    )


def compute_accuracy(model: TempProtoNet, dataloader: DataLoader, device: torch.device) -> float:
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


def store_training_embeddings(
    model: TempProtoNet,
    train_loader: DataLoader,
    device: torch.device,
):
    """Store training embeddings for prototype projection."""
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            z = model.encode(inputs)
            all_embeddings.append(z.cpu())
            all_labels.append(labels)

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)

    model.store_training_embeddings(embeddings, labels)


def generate_explanations(
    model: TempProtoNet,
    val_loader: DataLoader,
    device: torch.device,
    n_samples: int = 5,
) -> List[Dict[str, Any]]:
    """Generate explanations for sample predictions."""
    model.eval()

    # Get a few samples
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

    # Add ground truth
    for i, exp in enumerate(explanations):
        exp['ground_truth'] = labels[i].item()
        exp['ground_truth_name'] = 'MG' if labels[i].item() == 1 else 'HC'
        exp['correct'] = exp['prediction'] == labels[i].item()

    return explanations


def run_experiment(config: TempProtoNetConfig, output_dir: str, verbose: bool = True):
    """
    Run the full TempProtoNet experiment.

    Args:
        config: Experiment configuration
        output_dir: Directory to save results
        verbose: Whether to print progress
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
    cv = StratifiedGroupKFold(n_splits=config.n_folds, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []
    all_explanations = []
    best_model = None
    best_accuracy = 0.0
    best_falsification = None

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, patient_ids)):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Fold {fold + 1}/{config.n_folds}")
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
        model = TempProtoNet(
            input_dim=input_dim,
            num_classes=num_classes,
            seq_len=seq_len,
            latent_dim=config.latent_dim,
            n_prototypes_per_class=config.n_prototypes_per_class,
            encoder_hidden=config.encoder_hidden,
            encoder_layers=config.encoder_layers,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )

        if verbose:
            print(f"Model parameters: {model.count_parameters():,}")

        # Train
        trainer = TempProtoNetTrainer(model, config, device)
        train_result = trainer.train(train_loader, val_loader, train_labels, verbose=verbose)

        # Evaluate
        metrics = trainer.evaluate(val_loader)

        if verbose:
            print(f"\nFold {fold + 1} Results: {metrics}")

        # Store training embeddings for prototype projection
        store_training_embeddings(model, train_loader, device)

        # Run falsification tests
        falsification = run_falsification_tests(model, val_loader, device)

        if verbose:
            print(f"\nFalsification Tests:")
            print(f"  Diversity: {'PASS' if falsification.diversity_pass else 'FAIL'} "
                  f"(mean_dist={falsification.mean_pairwise_distance:.4f})")
            print(f"  Alignment: {'PASS' if falsification.alignment_pass else 'FAIL'} "
                  f"(ratio={falsification.alignment_ratio:.4f})")
            print(f"  Ablation: {'PASS' if falsification.ablation_pass else 'FAIL'} "
                  f"(max_importance={max(falsification.prototype_importance.values()):.4f})")
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
            'falsification_pass': falsification.all_pass,
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

    # Check if accuracy goal is met
    accuracy_goal_met = mean_accuracy >= 0.65

    # Summary
    summary = {
        'config': asdict(config),
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
        'fold_results': fold_results,
        'falsification': {
            'diversity_pass': best_falsification.diversity_pass,
            'alignment_pass': best_falsification.alignment_pass,
            'ablation_pass': best_falsification.ablation_pass,
            'all_pass': best_falsification.all_pass,
            'mean_pairwise_distance': best_falsification.mean_pairwise_distance,
            'min_pairwise_distance': best_falsification.min_pairwise_distance,
            'inter_class_distance': best_falsification.inter_class_distance,
            'intra_class_distance': best_falsification.intra_class_distance,
            'alignment_ratio': best_falsification.alignment_ratio,
            'mg_to_mg_similarity': best_falsification.mg_to_mg_similarity,
            'mg_to_hc_similarity': best_falsification.mg_to_hc_similarity,
            'hc_to_hc_similarity': best_falsification.hc_to_hc_similarity,
            'hc_to_mg_similarity': best_falsification.hc_to_mg_similarity,
        },
        'sample_explanations': all_explanations[:5],  # First 5 explanations
        'timestamp': datetime.now().isoformat(),
    }

    # Print summary
    if verbose:
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(f"\nPerformance:")
        print(f"  Mean Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
        print(f"  Mean AUC-ROC: {mean_auc:.4f} (+/- {std_auc:.4f})")
        print(f"  Mean F1: {mean_f1:.4f}")
        print(f"  Mean Sensitivity: {mean_sensitivity:.4f}")
        print(f"  Mean Specificity: {mean_specificity:.4f}")
        print(f"\n  Accuracy Goal (>=65%): {'MET' if accuracy_goal_met else 'NOT MET'}")

        print(f"\nFalsification Tests (Best Fold):")
        print(f"  Diversity: {'PASS' if best_falsification.diversity_pass else 'FAIL'}")
        print(f"  Alignment: {'PASS' if best_falsification.alignment_pass else 'FAIL'}")
        print(f"  Ablation: {'PASS' if best_falsification.ablation_pass else 'FAIL'}")
        print(f"  Overall: {'PASS' if best_falsification.all_pass else 'FAIL'}")

        print("\nSample Explanations:")
        for i, exp in enumerate(all_explanations[:3]):
            print(f"\n  Sample {i+1}:")
            print(f"    Ground Truth: {exp['ground_truth_name']}")
            print(f"    Prediction: {exp['prediction_name']} ({exp['confidence']:.2%})")
            print(f"    Correct: {exp['correct']}")
            print(f"    Top Prototype: {exp['top_similar_prototypes'][0]['prototype_class_name']} "
                  f"(similarity={exp['top_similar_prototypes'][0]['similarity']:.4f})")

    # Save results
    results_path = os.path.join(output_dir, 'results.json')

    # Convert numpy types for JSON serialization
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


if __name__ == '__main__':
    # Configuration
    config = TempProtoNetConfig(
        latent_dim=64,
        n_prototypes_per_class=5,
        encoder_hidden=64,
        encoder_layers=3,
        kernel_size=7,
        dropout=0.2,
        epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        early_stopping_patience=15,
        cluster_loss_weight=0.5,
        separation_loss_weight=0.1,
        n_folds=5,
    )

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/ccece/temp_proto_net/{timestamp}"

    # Run experiment
    summary = run_experiment(config, output_dir, verbose=True)
