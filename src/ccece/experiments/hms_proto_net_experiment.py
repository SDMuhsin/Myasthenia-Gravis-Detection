"""
CCECE Paper: HMSProtoNet Experiment

Rapid prototype experiment for Hierarchical Multi-Scale Prototype Network.

PRIMARY SUCCESS CRITERION (IMMUTABLE): Accuracy > 71.70% (TempProtoNet baseline)

This script:
1. Trains HMSProtoNet with N-fold cross-validation (default 3 for rapid prototype)
2. Evaluates against PRIMARY criterion
3. If PRIMARY criterion FAILS -> STOP (no investigation)
4. If PRIMARY criterion PASSES -> Proceed to full evaluation
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
from typing import Dict, List, Tuple, Any
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
from ccece.models.hms_proto_net import HMSProtoNet


RANDOM_SEED = 42

# PRIMARY CRITERION (IMMUTABLE)
BASELINE_ACCURACY = 0.7170


@dataclass
class HMSProtoNetConfig:
    """Configuration for HMSProtoNet experiment."""
    # Model hyperparameters
    latent_dim: int = 64
    n_prototypes_per_class: int = 3  # Per scale
    encoder_hidden: int = 64
    encoder_layers: int = 3
    kernel_size: int = 7
    dropout: float = 0.2
    macro_positions: int = 12
    meso_positions: int = 48
    micro_positions: int = 96
    use_diversity_loss: bool = False
    diversity_weight: float = 0.01

    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 15
    grad_clip_norm: float = 1.0

    # Prototype loss weights
    cluster_loss_weight: float = 0.5
    separation_loss_weight: float = 0.1

    # Cross-validation
    n_folds: int = 3  # Rapid prototype uses 3 folds


class HMSProtoNetTrainer:
    """Custom trainer for HMSProtoNet with multi-scale prototype losses."""

    def __init__(
        self,
        model: HMSProtoNet,
        config: HMSProtoNetConfig,
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
        """Train for one epoch with multi-scale prototype losses."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward with explanations
            outputs = self.model.forward_with_explanations(inputs)
            logits = outputs['logits']

            # Classification loss
            ce_loss = self.criterion(logits, labels)

            # Prototype losses
            cluster_loss, separation_loss, diversity_loss = self.model.compute_prototype_loss(
                outputs, labels
            )

            # Total loss
            loss = (
                ce_loss +
                self.config.cluster_loss_weight * cluster_loss +
                self.config.separation_loss_weight * separation_loss +
                diversity_loss  # Already weighted in model
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


def run_rapid_prototype(config: HMSProtoNetConfig, output_dir: str, verbose: bool = True):
    """
    Run rapid prototype experiment (2-3 folds).

    PRIMARY CRITERION CHECK: If mean accuracy <= 71.70%, report FAILURE.
    """
    set_all_seeds(RANDOM_SEED)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
        print(f"\n{'='*60}")
        print("PRIMARY CRITERION: Accuracy > 71.70% (TempProtoNet baseline)")
        print(f"{'='*60}\n")

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
        model = HMSProtoNet(
            input_dim=input_dim,
            num_classes=num_classes,
            seq_len=seq_len,
            latent_dim=config.latent_dim,
            n_prototypes_per_class=config.n_prototypes_per_class,
            encoder_hidden=config.encoder_hidden,
            encoder_layers=config.encoder_layers,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            macro_positions=config.macro_positions,
            meso_positions=config.meso_positions,
            micro_positions=config.micro_positions,
            use_diversity_loss=config.use_diversity_loss,
            diversity_weight=config.diversity_weight,
        )

        if verbose:
            print(f"Model parameters: {model.count_parameters():,}")

        # Train
        trainer = HMSProtoNetTrainer(model, config, device)
        train_result = trainer.train(train_loader, val_loader, train_labels, verbose=verbose)

        # Evaluate
        metrics = trainer.evaluate(val_loader)

        if verbose:
            print(f"\nFold {fold + 1} Results:")
            print(f"  Accuracy: {metrics.accuracy:.4f}")
            print(f"  AUC-ROC: {metrics.auc_roc:.4f}")
            print(f"  F1: {metrics.f1:.4f}")

        # Get scale weights
        scale_weights = model.get_scale_weights()
        if verbose:
            print(f"  Scale weights: macro={scale_weights['macro']:.3f}, "
                  f"meso={scale_weights['meso']:.3f}, micro={scale_weights['micro']:.3f}")

        fold_result = {
            'fold': fold + 1,
            'accuracy': metrics.accuracy,
            'sensitivity': metrics.sensitivity,
            'specificity': metrics.specificity,
            'f1': metrics.f1,
            'auc_roc': metrics.auc_roc,
            'best_epoch': train_result['best_epoch'],
            'scale_weights': scale_weights,
        }
        fold_results.append(fold_result)

    # Aggregate results
    mean_accuracy = np.mean([r['accuracy'] for r in fold_results])
    std_accuracy = np.std([r['accuracy'] for r in fold_results])
    mean_auc = np.mean([r['auc_roc'] for r in fold_results])
    std_auc = np.std([r['auc_roc'] for r in fold_results])
    mean_f1 = np.mean([r['f1'] for r in fold_results])

    # PRIMARY CRITERION CHECK
    primary_criterion_passed = mean_accuracy > BASELINE_ACCURACY

    # Summary
    summary = {
        'config': asdict(config),
        'results': {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'mean_f1': mean_f1,
        },
        'fold_results': fold_results,
        'primary_criterion': {
            'baseline': BASELINE_ACCURACY,
            'achieved': mean_accuracy,
            'passed': primary_criterion_passed,
        },
        'timestamp': datetime.now().isoformat(),
    }

    # Print PRIMARY CRITERION RESULT
    print("\n" + "="*60)
    print("PRIMARY CRITERION EVALUATION")
    print("="*60)
    print(f"\nBaseline (TempProtoNet): {BASELINE_ACCURACY:.4f} (71.70%)")
    print(f"HMSProtoNet achieved:    {mean_accuracy:.4f} ({mean_accuracy*100:.2f}%)")
    print(f"Standard deviation:      {std_accuracy:.4f}")
    print()

    if primary_criterion_passed:
        print(">>> PRIMARY CRITERION: PASSED <<<")
        print("HMSProtoNet outperforms baseline. Proceeding to full evaluation.")
    else:
        print(">>> PRIMARY CRITERION: FAILED <<<")
        print("HMSProtoNet does NOT outperform baseline.")
        print("Per protocol: STOP. Do not investigate. PIVOT to next approach.")

    print("="*60)

    # Save results
    results_path = os.path.join(output_dir, 'rapid_prototype_results.json')

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
        return obj

    summary_serializable = convert_to_serializable(summary)

    with open(results_path, 'w') as f:
        json.dump(summary_serializable, f, indent=2)

    if verbose:
        print(f"\nResults saved to: {results_path}")

    return summary


if __name__ == '__main__':
    # Rapid prototype configuration (3 folds)
    config = HMSProtoNetConfig(
        latent_dim=64,
        n_prototypes_per_class=3,
        encoder_hidden=64,
        encoder_layers=3,
        kernel_size=7,
        dropout=0.2,
        macro_positions=12,
        meso_positions=48,
        micro_positions=96,
        use_diversity_loss=False,  # Test without first
        diversity_weight=0.01,
        epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        early_stopping_patience=15,
        cluster_loss_weight=0.5,
        separation_loss_weight=0.1,
        n_folds=3,  # Rapid prototype
    )

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/ccece/hms_proto_net/rapid_prototype_{timestamp}"

    # Run experiment
    summary = run_rapid_prototype(config, output_dir, verbose=True)
