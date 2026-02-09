"""
MHTPN Best Configuration Analysis

Analyze the time_only configuration (n_segments=1 + L2 norm) that achieves 54.8%.
Get per-class accuracy to understand remaining gap.

Usage:
    python3 -m ccece.experiments.sota_comparison.mhtpn_best_analysis \
        --output results/ccece/sota_comparison/lsst_best_analysis
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from ccece.run_experiment import set_all_seeds

try:
    from .datasets import load_dataset, get_cv_strategy, standardize_data
except ImportError:
    from ccece.experiments.sota_comparison.datasets import load_dataset, get_cv_strategy, standardize_data


RANDOM_SEED = 42


class MHTPNBest(nn.Module):
    """
    Best MHTPN configuration: n_segments=1 with L2 normalization.
    This is the exact configuration that achieves 54.8%.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        latent_dim: int = 64,
        n_heads: int = 5,
        head_dim: int = 32,
        encoder_hidden: int = 64,
        encoder_layers: int = 3,
        kernel_size: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # Build encoder
        layers = []
        in_channels = input_dim
        for i in range(encoder_layers):
            out_channels = encoder_hidden * (2 ** min(i, 2))
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers)
        self.encoder_output_dim = in_channels

        # Projection head
        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )

        # Trajectory prototype heads
        from ccece.models.multi_head_trajectory_proto_net import TrajectoryPrototypeHead
        self.heads = nn.ModuleList([
            TrajectoryPrototypeHead(latent_dim, head_dim, num_classes)
            for _ in range(n_heads)
        ])

        # Single segment (n_segments=1)
        self.register_buffer('t_default', torch.tensor([0.5]))

    def encode(self, x):
        """Encode full sequence (no segmentation)."""
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        features = self.encoder(x)
        z = self.projection_head(features)
        z = F.normalize(z, p=2, dim=1)  # CRITICAL: L2 normalization
        return z.unsqueeze(1)  # (batch, 1, latent_dim) - single segment

    def forward(self, x, lengths=None):
        z_segments = self.encode(x)  # (batch, 1, latent_dim)
        segment_weights = torch.ones(x.size(0), 1, device=x.device)

        all_logits = []
        for head in self.heads:
            _, _, _, head_logits = head(z_segments, segment_weights, self.t_default)
            all_logits.append(head_logits)

        logits = torch.stack(all_logits, dim=0).mean(dim=0)
        return logits


def train_model(model, X_train, y_train, device, epochs=150, patience=30):
    """Train model and return trained model."""
    model = model.to(device)
    num_classes = len(np.unique(y_train))

    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    class_counts = np.bincount(y_train, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts.astype(np.float32)
    class_weights = class_weights / class_weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).float().to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    return model


def evaluate_model(model, X_val, y_val, device, class_names=None):
    """Evaluate model and return detailed metrics."""
    model.eval()
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    # Per-class metrics
    unique_classes = np.unique(all_labels)
    per_class_acc = {}
    per_class_counts = {}
    per_class_correct = {}

    for cls in unique_classes:
        mask = (all_labels == cls)
        cls_preds = all_preds[mask]
        cls_labels = all_labels[mask]
        cls_acc = (cls_preds == cls_labels).mean() if len(cls_labels) > 0 else 0.0
        per_class_acc[int(cls)] = cls_acc
        per_class_counts[int(cls)] = int(mask.sum())
        per_class_correct[int(cls)] = int((cls_preds == cls_labels).sum())

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'per_class_accuracy': per_class_acc,
        'per_class_counts': per_class_counts,
        'per_class_correct': per_class_correct,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
    }


def run_analysis(
    dataset_name: str = 'LSST',
    output_dir: str = None,
    n_folds: int = 5,
):
    """Run detailed analysis of best MHTPN configuration."""
    set_all_seeds(RANDOM_SEED)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/ccece/sota_comparison/lsst_best_analysis_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("MHTPN BEST CONFIGURATION ANALYSIS")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print()

    # Load dataset
    print("Loading data...")
    X, y, groups, dataset_config = load_dataset(dataset_name, verbose=False)

    print(f"Dataset: {dataset_name}")
    print(f"  Shape: {X.shape}")
    print(f"  Classes: {dataset_config.n_classes}")

    # Get class names if available
    class_names = getattr(dataset_config, 'class_names', None)
    if class_names:
        print(f"  Class names: {class_names}")

    # Cross-validation
    cv = get_cv_strategy(dataset_config, n_splits=n_folds, random_state=RANDOM_SEED)
    if dataset_config.has_groups:
        splits = list(cv.split(X, y, groups))
    else:
        splits = list(cv.split(X, y))

    fold_results = []
    all_per_class_acc = defaultdict(list)
    all_per_class_counts = defaultdict(int)

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print('=' * 60)

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        X_train, X_val = standardize_data(X_train, X_val)

        print(f"  Training: {len(y_train)} samples")
        print(f"  Validation: {len(y_val)} samples")

        # Train model
        print("  Training model...", end=' ', flush=True)
        torch.manual_seed(RANDOM_SEED)
        model = MHTPNBest(
            input_dim=X_train.shape[2],
            num_classes=dataset_config.n_classes,
            seq_len=X_train.shape[1],
            latent_dim=64,
            n_heads=5,
            head_dim=32,
            encoder_hidden=64,
            encoder_layers=3,
            kernel_size=7,
            dropout=0.2,
        )

        model = train_model(model, X_train, y_train, device)
        print("done")

        # Evaluate
        print("  Evaluating...", end=' ', flush=True)
        metrics = evaluate_model(model, X_val, y_val, device, class_names)
        print(f"Acc: {metrics['accuracy']*100:.1f}%, Bal: {metrics['balanced_accuracy']*100:.1f}%")

        fold_results.append(metrics)

        # Accumulate per-class stats
        for cls, acc in metrics['per_class_accuracy'].items():
            all_per_class_acc[cls].append(acc)
        for cls, count in metrics['per_class_counts'].items():
            all_per_class_counts[cls] += count

        torch.cuda.empty_cache()

    # Aggregate results
    print(f"\n{'=' * 70}")
    print("AGGREGATED RESULTS")
    print('=' * 70)

    accuracies = [r['accuracy'] for r in fold_results]
    balanced_accs = [r['balanced_accuracy'] for r in fold_results]

    print(f"\nOverall Accuracy: {np.mean(accuracies)*100:.1f}% +/- {np.std(accuracies)*100:.1f}%")
    print(f"Balanced Accuracy: {np.mean(balanced_accs)*100:.1f}% +/- {np.std(balanced_accs)*100:.1f}%")

    # Per-class analysis
    print(f"\n{'Class':<10} {'Samples':<10} {'Pct':<10} {'Accuracy':<15} {'Status'}")
    print("-" * 60)

    # Sort classes by accuracy (ascending - worst first)
    class_mean_acc = {cls: np.mean(accs) for cls, accs in all_per_class_acc.items()}
    total_samples = sum(all_per_class_counts.values())

    critical_classes = []
    for cls in sorted(class_mean_acc.keys(), key=lambda x: class_mean_acc[x]):
        acc = class_mean_acc[cls]
        count = all_per_class_counts[cls]
        pct = count / total_samples * 100

        if acc < 0.2:
            status = "CRITICAL"
            critical_classes.append((cls, count, pct, acc))
        elif acc < 0.4:
            status = "LOW"
        else:
            status = "OK"

        print(f"{cls:<10} {count:<10} {pct:<10.1f}% {acc*100:<15.1f}% {status}")

    # Summary of critical classes
    print(f"\n{'=' * 70}")
    print("CRITICAL CLASSES (< 20% accuracy)")
    print('=' * 70)

    if critical_classes:
        total_critical_samples = sum(c[1] for c in critical_classes)
        total_critical_pct = sum(c[2] for c in critical_classes)
        print(f"\nTotal critical samples: {total_critical_samples} ({total_critical_pct:.1f}% of data)")
        print("\nThese classes are responsible for most of the accuracy gap.")
        for cls, count, pct, acc in critical_classes:
            print(f"  Class {cls}: {count} samples ({pct:.1f}%), accuracy: {acc*100:.1f}%")
    else:
        print("No critical classes found.")

    # Gap analysis
    target = 0.6162
    current_acc = np.mean(accuracies)
    gap = target - current_acc

    print(f"\n{'=' * 70}")
    print("GAP ANALYSIS")
    print('=' * 70)
    print(f"Current accuracy: {current_acc*100:.1f}%")
    print(f"Target (MUSE):    {target*100:.1f}%")
    print(f"Gap:              {gap*100:.1f}%")

    if critical_classes:
        # Estimate potential gain from fixing critical classes
        potential_gain = 0
        for cls, count, pct, acc in critical_classes:
            # If we could get 50% accuracy on these classes
            potential_acc_gain = (0.5 - acc) * (pct / 100)
            potential_gain += potential_acc_gain
        print(f"\nPotential gain if critical classes reached 50% accuracy: {potential_gain*100:.1f}%")

    # Save results
    full_results = {
        'fold_results': fold_results,
        'summary': {
            'accuracy_mean': float(np.mean(accuracies)),
            'accuracy_std': float(np.std(accuracies)),
            'balanced_accuracy_mean': float(np.mean(balanced_accs)),
            'balanced_accuracy_std': float(np.std(balanced_accs)),
        },
        'per_class_accuracy': {str(k): float(np.mean(v)) for k, v in all_per_class_acc.items()},
        'per_class_counts': {str(k): v for k, v in all_per_class_counts.items()},
        'critical_classes': [{'class': c[0], 'samples': c[1], 'pct': c[2], 'accuracy': c[3]} for c in critical_classes],
        'gap_to_target': gap,
    }

    with open(os.path.join(output_dir, 'best_analysis.json'), 'w') as f:
        json.dump(full_results, f, indent=2, default=float)

    print(f"\nResults saved to: {output_dir}")

    return full_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MHTPN Best Configuration Analysis')
    parser.add_argument('--dataset', type=str, default='LSST', help='Dataset name')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')

    args = parser.parse_args()

    run_analysis(
        dataset_name=args.dataset,
        output_dir=args.output,
        n_folds=args.folds,
    )
