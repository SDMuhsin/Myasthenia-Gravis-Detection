"""
Multi-Head Prototype Network Experiment

Tests the multi-head architecture that addresses prototype collapse through structural constraints.

Key hypothesis:
- With 5 heads × 2 prototypes (1 per class each), collapse is structurally impossible
- Alignment ≈ per-head accuracy (by construction)
- Multiple heads averaging should maintain or improve accuracy

Success criteria (from CONTEXT.md):
- Prototype alignment: >65% (vs current 50.4%)
- Model accuracy: ≥71% (current is 72.0%)
- Active prototypes: >4/10 (vs current 2/10) - N/A for multi-head (all are active)
- Reproducibility: Works across 5 folds
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ccece.data_loader import load_binary_dataset, extract_arrays
from ccece.run_experiment import preprocess_items, set_all_seeds, compute_target_seq_len
from ccece.trainer import (
    TrainingConfig, EvaluationMetrics, create_data_loaders
)
from ccece.models.multi_head_proto_net import MultiHeadProtoNet


RANDOM_SEED = 42


@dataclass
class MultiHeadConfig:
    """Configuration for MultiHeadProtoNet experiment."""
    # Model hyperparameters
    latent_dim: int = 64
    n_heads: int = 5
    head_dim: int = 32
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

    # Loss weights
    cluster_loss_weight: float = 0.3
    separation_loss_weight: float = 0.1
    per_head_ce_weight: float = 0.2  # Additional CE loss per head (encourages head specialization)

    # Cross-validation
    n_folds: int = 5


class MultiHeadTrainer:
    """Trainer for MultiHeadProtoNet."""

    def __init__(
        self,
        model: MultiHeadProtoNet,
        config: MultiHeadConfig,
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
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)

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
                    'loss': f'{train_loss:.4f}',
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
        }

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward with explanations
            logits, z, all_distances, all_similarities = self.model.forward_with_explanations(inputs)

            # Main classification loss (on averaged logits)
            ce_loss = self.criterion(logits, labels)

            # Per-head classification loss (encourages each head to be discriminative)
            per_head_ce_loss = torch.tensor(0.0, device=self.device)
            for h_idx in range(self.model.n_heads):
                head_logits = all_similarities[h_idx]  # (batch, 2)
                head_ce = self.criterion(head_logits, labels)
                per_head_ce_loss = per_head_ce_loss + head_ce
            per_head_ce_loss = per_head_ce_loss / self.model.n_heads

            # Prototype losses
            cluster_loss, separation_loss = self.model.compute_prototype_loss(z, labels)

            # Total loss
            loss = (
                ce_loss +
                self.config.per_head_ce_weight * per_head_ce_loss +
                self.config.cluster_loss_weight * cluster_loss +
                self.config.separation_loss_weight * separation_loss
            )

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip_norm
            )

            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        return total_loss / total_samples

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


def store_training_embeddings(
    model: MultiHeadProtoNet,
    train_loader: DataLoader,
    device: torch.device,
):
    """Store training embeddings for alignment computation."""
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


def compute_alignment_metrics(
    model: MultiHeadProtoNet,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Compute detailed alignment metrics for the multi-head model.

    Returns:
        Dict with per-head and overall alignment metrics
    """
    model.eval()

    # Get per-head alignment
    per_head = model.compute_alignment_per_head()

    # Compute aggregate stats
    alignments = [per_head[i]['alignment'] for i in range(model.n_heads)]
    hc_alignments = [per_head[i]['hc_alignment'] for i in range(model.n_heads)]
    mg_alignments = [per_head[i]['mg_alignment'] for i in range(model.n_heads)]
    proto_distances = [per_head[i]['proto_distance'] for i in range(model.n_heads)]

    return {
        'per_head': per_head,
        'overall_alignment': np.mean(alignments),
        'std_alignment': np.std(alignments),
        'min_alignment': np.min(alignments),
        'max_alignment': np.max(alignments),
        'mean_hc_alignment': np.mean(hc_alignments),
        'mean_mg_alignment': np.mean(mg_alignments),
        'mean_proto_distance': np.mean(proto_distances),
        'n_heads_above_65': sum(1 for a in alignments if a >= 0.65),
        'n_heads_above_70': sum(1 for a in alignments if a >= 0.70),
    }


def run_experiment(config: MultiHeadConfig, output_dir: str, verbose: bool = True):
    """
    Run the MultiHeadProtoNet experiment with 5-fold cross-validation.
    """
    set_all_seeds(RANDOM_SEED)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
        print("\n" + "="*70)
        print("MULTI-HEAD PROTOTYPE NETWORK EXPERIMENT")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  n_heads: {config.n_heads}")
        print(f"  head_dim: {config.head_dim}")
        print(f"  Total prototypes: {config.n_heads * 2}")

    # Load data
    if verbose:
        print("\nLoading data...")
    items = load_binary_dataset(verbose=False)
    items = preprocess_items(items)

    X, y, patient_ids = extract_arrays(items)
    seq_len = compute_target_seq_len(items)
    input_dim = items[0]['data'].shape[1]

    if verbose:
        print(f"Data: {len(items)} samples, seq_len={seq_len}, input_dim={input_dim}")
        hc_count = sum(y == 0)
        mg_count = sum(y == 1)
        print(f"Class distribution: HC={hc_count} ({hc_count/len(y):.1%}), MG={mg_count} ({mg_count/len(y):.1%})")

    # Cross-validation
    cv = StratifiedGroupKFold(n_splits=config.n_folds, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, patient_ids)):
        if verbose:
            print(f"\n{'='*60}")
            print(f"FOLD {fold + 1}/{config.n_folds}")
            print('='*60)

        # Split data
        train_items = [items[i] for i in train_idx]
        val_items = [items[i] for i in val_idx]
        train_labels = np.array([item['label'] for item in train_items])

        # Create data loaders
        train_loader, val_loader, scaler = create_data_loaders(
            train_items, val_items, seq_len, config.batch_size
        )

        # Create model
        model = MultiHeadProtoNet(
            input_dim=input_dim,
            num_classes=2,
            seq_len=seq_len,
            latent_dim=config.latent_dim,
            n_heads=config.n_heads,
            head_dim=config.head_dim,
            encoder_hidden=config.encoder_hidden,
            encoder_layers=config.encoder_layers,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )

        if verbose:
            print(f"Model parameters: {model.count_parameters():,}")

        # Train
        trainer = MultiHeadTrainer(model, config, device)
        train_result = trainer.train(train_loader, val_loader, train_labels, verbose=verbose)

        # Evaluate accuracy
        metrics = trainer.evaluate(val_loader)

        if verbose:
            print(f"\nFold {fold + 1} Accuracy: {metrics.accuracy:.1%}")

        # Store training embeddings and compute alignment
        store_training_embeddings(model, train_loader, device)
        alignment_metrics = compute_alignment_metrics(model, device)

        if verbose:
            print(f"\nAlignment Metrics:")
            print(f"  Overall alignment: {alignment_metrics['overall_alignment']:.1%}")
            print(f"  Per-head alignments: {[f'{a:.1%}' for a in [alignment_metrics['per_head'][i]['alignment'] for i in range(config.n_heads)]]}")
            print(f"  HC alignment: {alignment_metrics['mean_hc_alignment']:.1%}")
            print(f"  MG alignment: {alignment_metrics['mean_mg_alignment']:.1%}")
            print(f"  Heads above 65%: {alignment_metrics['n_heads_above_65']}/{config.n_heads}")

        # Store fold result
        fold_result = {
            'fold': fold + 1,
            'accuracy': metrics.accuracy,
            'sensitivity': metrics.sensitivity,
            'specificity': metrics.specificity,
            'f1': metrics.f1,
            'auc_roc': metrics.auc_roc,
            'best_epoch': train_result['best_epoch'],
            'alignment': alignment_metrics,
        }
        fold_results.append(fold_result)

    # Aggregate results
    mean_accuracy = np.mean([r['accuracy'] for r in fold_results])
    std_accuracy = np.std([r['accuracy'] for r in fold_results])
    mean_alignment = np.mean([r['alignment']['overall_alignment'] for r in fold_results])
    std_alignment = np.std([r['alignment']['overall_alignment'] for r in fold_results])
    mean_hc_alignment = np.mean([r['alignment']['mean_hc_alignment'] for r in fold_results])
    mean_mg_alignment = np.mean([r['alignment']['mean_mg_alignment'] for r in fold_results])

    # Check success criteria
    accuracy_pass = mean_accuracy >= 0.71
    alignment_pass = mean_alignment >= 0.65

    summary = {
        'config': asdict(config),
        'results': {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'mean_alignment': mean_alignment,
            'std_alignment': std_alignment,
            'mean_hc_alignment': mean_hc_alignment,
            'mean_mg_alignment': mean_mg_alignment,
            'accuracy_pass': accuracy_pass,
            'alignment_pass': alignment_pass,
            'both_pass': accuracy_pass and alignment_pass,
        },
        'fold_results': fold_results,
        'timestamp': datetime.now().isoformat(),
    }

    # Print summary
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)

        print(f"\n  PRIMARY CRITERIA:")
        print(f"  {'='*40}")
        print(f"  Accuracy:  {mean_accuracy:.1%} +/- {std_accuracy:.1%}  (threshold: >=71%)  {'PASS' if accuracy_pass else 'FAIL'}")
        print(f"  Alignment: {mean_alignment:.1%} +/- {std_alignment:.1%}  (threshold: >=65%)  {'PASS' if alignment_pass else 'FAIL'}")
        print(f"  {'='*40}")
        print(f"  OVERALL: {'SUCCESS' if accuracy_pass and alignment_pass else 'FAILURE'}")

        print(f"\n  Per-class alignment:")
        print(f"    HC: {mean_hc_alignment:.1%}")
        print(f"    MG: {mean_mg_alignment:.1%}")

        print(f"\n  Per-fold results:")
        for r in fold_results:
            print(f"    Fold {r['fold']}: acc={r['accuracy']:.1%}, align={r['alignment']['overall_alignment']:.1%}")

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
        return obj

    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(summary), f, indent=2)

    if verbose:
        print(f"\nResults saved to: {results_path}")

    return summary


if __name__ == '__main__':
    config = MultiHeadConfig(
        latent_dim=64,
        n_heads=5,
        head_dim=64,  # Increased from 32 for more capacity
        encoder_hidden=64,
        encoder_layers=3,
        kernel_size=7,
        dropout=0.2,
        epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        early_stopping_patience=20,  # Increased from 15 to allow more training
        cluster_loss_weight=0.3,
        separation_loss_weight=0.1,
        per_head_ce_weight=0.0,  # Disabled - was causing accuracy drop
        n_folds=5,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/ccece/multi_head_proto/{timestamp}"

    summary = run_experiment(config, output_dir, verbose=True)
