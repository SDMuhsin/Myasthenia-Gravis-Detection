"""
External Validation: GazeBase Session Classification

Validates MHTPN on GazeBase dataset for Session 1 vs Session 2 classification.
This tests whether MHTPN can capture temporal dynamics where fatigue
may cause later portions (Session 2) to differ from earlier portions (Session 1).

SUCCESS CRITERIA:
1. MHTPN must outperform random baseline (50% for balanced classes)
2. MHTPN should show comparable or better performance than simple baselines
3. Late segments should be more discriminative than early segments (temporal pattern)
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, confusion_matrix, precision_score, recall_score
)
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[3]))

from ccece.models.multi_head_trajectory_proto_net import MultiHeadTrajectoryProtoNet

RANDOM_SEED = 42


@dataclass
class GazeBaseConfig:
    """Configuration for GazeBase validation experiment."""
    # Data
    data_path: str = "/workspace/Myasthenia-Gravis-Detection/data/external/gazebase/processed/gazebase_tex_session_clf.npz"

    # Model hyperparameters (adapted for GazeBase)
    latent_dim: int = 64
    n_heads: int = 5
    head_dim: int = 32
    n_segments: int = 8  # Segment 30s into 8 parts
    encoder_hidden: int = 64
    encoder_layers: int = 3
    kernel_size: int = 7
    dropout: float = 0.2

    # Training hyperparameters
    epochs: int = 50  # Shorter for validation
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 15
    grad_clip_norm: float = 1.0

    # Loss weights
    cluster_loss_weight: float = 0.3
    separation_loss_weight: float = 0.1
    diversity_loss_weight: float = 0.05

    # Cross-validation
    n_folds: int = 5


def set_all_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_gazebase_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed GazeBase data."""
    data = np.load(data_path)
    X = data['X']
    y = data['y']
    groups = data['groups']
    print(f"Loaded data: X={X.shape}, y={y.shape}, groups={groups.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Unique subjects: {len(np.unique(groups))}")
    return X, y, groups


def standardize_data(
    train_X: np.ndarray,
    val_X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardize data using training set statistics.

    This is CRITICAL - the MG experiment uses SequenceScaler for normalization.
    Without this, features on different scales (e.g., pupil ~2000 vs x ~1)
    will cause the model to fail to learn.
    """
    # Compute mean and std from training data
    # Shape: (n_samples, seq_len, n_features) -> compute per-feature stats
    train_flat = train_X.reshape(-1, train_X.shape[-1])  # (n_samples * seq_len, n_features)
    mean = train_flat.mean(axis=0, keepdims=True)  # (1, n_features)
    std = train_flat.std(axis=0, keepdims=True) + 1e-8  # (1, n_features)

    # Apply standardization
    train_X_norm = (train_X - mean) / std
    val_X_norm = (val_X - mean) / std

    return train_X_norm, val_X_norm


def compute_segment_length(seq_len: int, n_segments: int) -> int:
    """Compute segment length that divides evenly."""
    segment_len = seq_len // n_segments
    return segment_len * n_segments


class GazeBaseTrainer:
    """Trainer for MHTPN on GazeBase."""

    def __init__(
        self,
        model: MultiHeadTrajectoryProtoNet,
        config: GazeBaseConfig,
        device: torch.device
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.best_model_state = None

    def _setup_training(self, train_labels: np.ndarray):
        """Setup optimizer, scheduler, and criterion."""
        # Class weights for imbalanced data
        class_counts = np.bincount(train_labels)
        total = len(train_labels)
        class_weights = torch.tensor(
            [total / (2 * c) for c in class_counts],
            dtype=torch.float32
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with explanations for auxiliary losses
            logits, z_segments, segment_weights, all_per_seg_sims, all_traj_sims = \
                self.model.forward_with_explanations(batch_X)

            # Build trajectory_info dict for compatibility
            if all_per_seg_sims:
                # Stack per-segment similarities: (n_heads, batch, n_segments, n_classes)
                # -> (batch, n_heads, n_segments, n_classes)
                sims = torch.stack(all_per_seg_sims, dim=0).permute(1, 0, 2, 3)
                trajectory_info = {'prototype_similarities': sims}
            else:
                trajectory_info = {}

            # Classification loss
            loss = self.criterion(logits, batch_y)

            # Auxiliary losses
            if 'prototype_similarities' in trajectory_info:
                sims = trajectory_info['prototype_similarities']

                # Cluster loss
                if self.config.cluster_loss_weight > 0:
                    cluster_loss = -torch.mean(torch.max(sims, dim=-1)[0])
                    loss = loss + self.config.cluster_loss_weight * cluster_loss

                # Separation loss
                if self.config.separation_loss_weight > 0:
                    batch_size = sims.shape[0]
                    n_heads = sims.shape[1]
                    n_segments = sims.shape[2]
                    n_protos = sims.shape[3]

                    sims_flat = sims.view(batch_size, n_heads, n_segments, n_protos)
                    proto_diffs = []
                    for h in range(n_heads):
                        for p1 in range(n_protos):
                            for p2 in range(p1+1, n_protos):
                                diff = torch.abs(sims_flat[:, h, :, p1] - sims_flat[:, h, :, p2])
                                proto_diffs.append(diff.mean())
                    if proto_diffs:
                        separation_loss = -torch.stack(proto_diffs).mean()
                        loss = loss + self.config.separation_loss_weight * separation_loss

            # Backward pass
            loss.backward()
            if self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip_norm
                )
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, val_loader: DataLoader) -> Dict:
        """Evaluate model on validation set."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        all_segment_sims = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)

                logits, z_segments, segment_weights, all_per_seg_sims, all_traj_sims = \
                    self.model.forward_with_explanations(batch_X)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch_y.numpy())
                all_probs.append(probs.cpu().numpy())

                if all_per_seg_sims:
                    # Stack per-segment similarities: (n_heads, batch, n_segments, n_classes)
                    # -> (batch, n_heads, n_segments, n_classes)
                    sims = torch.stack(all_per_seg_sims, dim=0).permute(1, 0, 2, 3)
                    all_segment_sims.append(sims.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)

        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='macro'),
            'precision': precision_score(all_labels, all_preds, average='macro'),
            'recall': recall_score(all_labels, all_preds, average='macro'),
            'auc': roc_auc_score(all_labels, all_probs[:, 1]) if all_probs.shape[1] == 2 else 0.0
        }

        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Compute segment discrimination
        if all_segment_sims:
            segment_sims = np.concatenate(all_segment_sims, axis=0)
            n_segments = segment_sims.shape[2]
            early_idx = n_segments // 2
            late_idx = n_segments

            # Average over heads and prototypes
            sims_avg = segment_sims.mean(axis=(1, 3))  # (batch, n_segments)

            early_sims = sims_avg[:, :early_idx].mean(axis=1)
            late_sims = sims_avg[:, early_idx:late_idx].mean(axis=1)

            # Check if late segments are more discriminative
            session1_mask = all_labels == 0
            session2_mask = all_labels == 1

            early_diff = early_sims[session2_mask].mean() - early_sims[session1_mask].mean()
            late_diff = late_sims[session2_mask].mean() - late_sims[session1_mask].mean()

            metrics['early_segment_discrimination'] = float(early_diff)
            metrics['late_segment_discrimination'] = float(late_diff)
            metrics['temporal_pattern_passes'] = abs(late_diff) > abs(early_diff)

        return metrics

    def train(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        val_X: np.ndarray,
        val_y: np.ndarray
    ) -> Dict:
        """Full training loop."""
        self._setup_training(train_y)

        # CRITICAL FIX: Standardize data using training set statistics
        # Without this, features on different scales cause model to fail
        train_X_norm, val_X_norm = standardize_data(train_X, val_X)

        # Create data loaders
        train_dataset = TensorDataset(
            torch.tensor(train_X_norm, dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(val_X_norm, dtype=torch.float32),
            torch.tensor(val_y, dtype=torch.long)
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )

        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0

        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            self.scheduler.step(val_metrics['accuracy'])

            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_epoch = epoch
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        # Final evaluation
        final_metrics = self.evaluate(val_loader)
        final_metrics['best_epoch'] = best_epoch

        return final_metrics


def run_cross_validation(config: GazeBaseConfig, output_dir: str) -> Dict:
    """Run full 5-fold cross-validation."""
    set_all_seeds(RANDOM_SEED)

    # Load data
    X, y, groups = load_gazebase_data(config.data_path)

    # Compute target sequence length (divisible by n_segments)
    original_seq_len = X.shape[1]
    target_seq_len = compute_segment_length(original_seq_len, config.n_segments)

    if target_seq_len != original_seq_len:
        print(f"Truncating sequences from {original_seq_len} to {target_seq_len}")
        X = X[:, :target_seq_len, :]

    n_features = X.shape[2]
    print(f"Data shape: {X.shape}, Features: {n_features}")

    # Setup cross-validation
    cv = StratifiedGroupKFold(n_splits=config.n_folds, shuffle=True, random_state=RANDOM_SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    all_fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx + 1}/{config.n_folds}")
        print(f"{'='*50}")

        train_X, train_y = X[train_idx], y[train_idx]
        val_X, val_y = X[val_idx], y[val_idx]

        print(f"Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")
        print(f"Train class dist: {np.bincount(train_y)}, Val class dist: {np.bincount(val_y)}")

        # Create model
        model = MultiHeadTrajectoryProtoNet(
            input_dim=n_features,
            num_classes=2,
            seq_len=target_seq_len,
            latent_dim=config.latent_dim,
            n_heads=config.n_heads,
            head_dim=config.head_dim,
            n_segments=config.n_segments,
            encoder_hidden=config.encoder_hidden,
            encoder_layers=config.encoder_layers,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )

        # Train
        trainer = GazeBaseTrainer(model, config, device)
        fold_metrics = trainer.train(train_X, train_y, val_X, val_y)

        fold_metrics['fold'] = fold_idx + 1
        all_fold_results.append(fold_metrics)

        print(f"\nFold {fold_idx + 1} Results:")
        print(f"  Accuracy: {fold_metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {fold_metrics['balanced_accuracy']:.4f}")
        print(f"  F1: {fold_metrics['f1']:.4f}")
        print(f"  AUC: {fold_metrics['auc']:.4f}")
        if 'temporal_pattern_passes' in fold_metrics:
            print(f"  Temporal Pattern Passes: {fold_metrics['temporal_pattern_passes']}")

        # Save fold result incrementally
        fold_path = Path(output_dir) / f"fold_{fold_idx + 1}_results.json"
        with open(fold_path, 'w') as f:
            json.dump(fold_metrics, f, indent=2, default=float)

    # Aggregate results
    metric_keys = ['accuracy', 'balanced_accuracy', 'f1', 'auc', 'sensitivity', 'specificity']
    aggregated = {}
    for key in metric_keys:
        values = [r[key] for r in all_fold_results if key in r]
        if values:
            aggregated[f'{key}_mean'] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values))

    # Temporal pattern analysis
    temporal_passes = [
        r.get('temporal_pattern_passes', False) for r in all_fold_results
    ]
    aggregated['temporal_pattern_passes_count'] = sum(temporal_passes)
    aggregated['temporal_pattern_passes_folds'] = f"{sum(temporal_passes)}/{len(temporal_passes)}"

    # Summary
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {aggregated['accuracy_mean']:.4f} ± {aggregated['accuracy_std']:.4f}")
    print(f"Balanced Accuracy: {aggregated['balanced_accuracy_mean']:.4f} ± {aggregated['balanced_accuracy_std']:.4f}")
    print(f"F1: {aggregated['f1_mean']:.4f} ± {aggregated['f1_std']:.4f}")
    print(f"AUC: {aggregated['auc_mean']:.4f} ± {aggregated['auc_std']:.4f}")
    print(f"Temporal Pattern: {aggregated['temporal_pattern_passes_folds']} folds passed")

    # Save full results
    full_results = {
        'config': asdict(config),
        'fold_results': all_fold_results,
        'aggregated': aggregated,
        'timestamp': datetime.now().isoformat()
    }

    results_path = Path(output_dir) / "full_results.json"
    with open(results_path, 'w') as f:
        json.dump(full_results, f, indent=2, default=float)

    print(f"\nResults saved to {results_path}")

    return full_results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Run GazeBase external validation')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')

    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path('/workspace/Myasthenia-Gravis-Detection/results/ccece/external_validation/gazebase') / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Create config
    config = GazeBaseConfig(
        epochs=args.epochs,
        n_folds=args.n_folds
    )

    # Run experiment
    results = run_cross_validation(config, str(output_dir))

    # Print final summary
    print("\n" + "="*60)
    print("EXTERNAL VALIDATION COMPLETE")
    print("="*60)
    acc = results['aggregated']['accuracy_mean']
    print(f"MHTPN Accuracy on GazeBase: {acc:.4f}")
    print(f"Random Baseline: 0.5000")
    print(f"Improvement over random: {(acc - 0.5) * 100:.1f}%")

    return results


if __name__ == '__main__':
    main()
