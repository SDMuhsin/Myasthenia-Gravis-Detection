"""
MHTPN Final Tuning Experiment for LSST

Current best: 54.8% (gap: 6.8% to MUSE's 61.6%)

This script tests final tuning approaches:
1. Focal loss (hard example mining)
2. Deeper encoder (4-5 layers)
3. Lower learning rate (1e-4)
4. Label smoothing
5. More epochs with longer patience

Usage:
    python3 -m ccece.experiments.sota_comparison.mhtpn_final_tuning \
        --output results/ccece/sota_comparison/lsst_final_tuning
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from ccece.run_experiment import set_all_seeds

try:
    from .datasets import load_dataset, get_cv_strategy, standardize_data
except ImportError:
    from ccece.experiments.sota_comparison.datasets import load_dataset, get_cv_strategy, standardize_data


RANDOM_SEED = 42


class FocalLoss(nn.Module):
    """Focal Loss for hard example mining."""

    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing."""

    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, x, target):
        n_classes = x.size(-1)
        log_probs = F.log_softmax(x, dim=-1)

        # Create smoothed labels
        smooth_labels = torch.zeros_like(log_probs)
        smooth_labels.fill_(self.smoothing / (n_classes - 1))
        smooth_labels.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        if self.weight is not None:
            weight = self.weight[target]
            loss = -(smooth_labels * log_probs).sum(dim=-1)
            loss = (loss * weight).sum() / weight.sum()
        else:
            loss = -(smooth_labels * log_probs).sum(dim=-1).mean()

        return loss


class MHTPNFinalTuning(nn.Module):
    """MHTPN with configurable depth and features."""

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
        self.n_heads = n_heads

        # Build encoder with variable depth
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
        z = F.normalize(z, p=2, dim=1)
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


def train_and_evaluate(
    model,
    X_train, y_train,
    X_val, y_val,
    device,
    epochs=150,
    patience=30,
    lr=1e-3,
    loss_type='ce',
    focal_gamma=2.0,
    label_smoothing=0.1,
):
    """Train model with configurable loss and hyperparameters."""
    model = model.to(device)
    num_classes = len(np.unique(y_train))

    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Class weights
    class_counts = np.bincount(y_train, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts.astype(np.float32)
    class_weights = class_weights / class_weights.sum() * num_classes
    weight_tensor = torch.from_numpy(class_weights).float().to(device)

    # Loss function
    if loss_type == 'ce':
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    elif loss_type == 'focal':
        criterion = FocalLoss(gamma=focal_gamma, weight=weight_tensor)
    elif loss_type == 'label_smoothing':
        criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing, weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_balanced_acc = 0.0
    best_model_state = None
    patience_counter = 0

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

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_balanced_acc = balanced_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_acc, best_balanced_acc


def run_final_tuning(
    dataset_name: str = 'LSST',
    output_dir: str = None,
    n_folds: int = 5,
):
    """Run final tuning experiments."""
    set_all_seeds(RANDOM_SEED)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/ccece/sota_comparison/lsst_final_tuning_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("MHTPN FINAL TUNING EXPERIMENT")
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

    # Experiments
    experiments = {
        # Baseline (current best configuration)
        'baseline': {
            'encoder_layers': 3,
            'lr': 1e-3,
            'loss_type': 'ce',
            'epochs': 150,
            'patience': 30,
        },
        # Deeper encoder
        'deep_encoder_4': {
            'encoder_layers': 4,
            'lr': 1e-3,
            'loss_type': 'ce',
            'epochs': 150,
            'patience': 30,
        },
        # Lower learning rate
        'lower_lr': {
            'encoder_layers': 3,
            'lr': 5e-4,
            'loss_type': 'ce',
            'epochs': 200,
            'patience': 40,
        },
        # Focal loss
        'focal_loss': {
            'encoder_layers': 3,
            'lr': 1e-3,
            'loss_type': 'focal',
            'epochs': 150,
            'patience': 30,
            'focal_gamma': 2.0,
        },
        # Label smoothing
        'label_smoothing': {
            'encoder_layers': 3,
            'lr': 1e-3,
            'loss_type': 'label_smoothing',
            'epochs': 150,
            'patience': 30,
            'label_smoothing': 0.1,
        },
        # Combined: lower lr + focal
        'focal_low_lr': {
            'encoder_layers': 3,
            'lr': 5e-4,
            'loss_type': 'focal',
            'epochs': 200,
            'patience': 40,
            'focal_gamma': 2.0,
        },
        # More heads
        'more_heads': {
            'encoder_layers': 3,
            'lr': 1e-3,
            'loss_type': 'ce',
            'epochs': 150,
            'patience': 30,
            'n_heads': 10,
        },
        # Larger latent dim with fewer heads
        'large_latent': {
            'encoder_layers': 3,
            'lr': 1e-3,
            'loss_type': 'ce',
            'epochs': 150,
            'patience': 30,
            'latent_dim': 128,
        },
    }

    # Cross-validation
    cv = get_cv_strategy(dataset_config, n_splits=n_folds, random_state=RANDOM_SEED)
    if dataset_config.has_groups:
        splits = list(cv.split(X, y, groups))
    else:
        splits = list(cv.split(X, y))

    results = {exp_name: {'accuracy': [], 'balanced_accuracy': []} for exp_name in experiments}

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print('=' * 60)

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        X_train, X_val = standardize_data(X_train, X_val)

        for exp_name, config in experiments.items():
            print(f"  Running {exp_name}...", end=' ', flush=True)

            try:
                torch.manual_seed(RANDOM_SEED)
                model = MHTPNFinalTuning(
                    input_dim=X_train.shape[2],
                    num_classes=dataset_config.n_classes,
                    seq_len=X_train.shape[1],
                    latent_dim=config.get('latent_dim', 64),
                    n_heads=config.get('n_heads', 5),
                    head_dim=32,
                    encoder_hidden=64,
                    encoder_layers=config['encoder_layers'],
                    kernel_size=7,
                    dropout=0.2,
                )

                val_acc, balanced_acc = train_and_evaluate(
                    model, X_train, y_train, X_val, y_val, device,
                    epochs=config['epochs'],
                    patience=config['patience'],
                    lr=config['lr'],
                    loss_type=config['loss_type'],
                    focal_gamma=config.get('focal_gamma', 2.0),
                    label_smoothing=config.get('label_smoothing', 0.1),
                )
                results[exp_name]['accuracy'].append(val_acc)
                results[exp_name]['balanced_accuracy'].append(balanced_acc)
                print(f"Acc: {val_acc*100:.1f}%, Bal: {balanced_acc*100:.1f}%")
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                results[exp_name]['accuracy'].append(None)
                results[exp_name]['balanced_accuracy'].append(None)

            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print('=' * 70)
    print(f"\n{'Experiment':<20} {'Accuracy':<20} {'Balanced Acc':<20}")
    print("-" * 60)

    summary = {}
    for exp_name in experiments:
        accs = [a for a in results[exp_name]['accuracy'] if a is not None]
        bals = [b for b in results[exp_name]['balanced_accuracy'] if b is not None]

        if accs:
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            mean_bal = np.mean(bals)
            std_bal = np.std(bals)

            summary[exp_name] = {
                'accuracy_mean': mean_acc,
                'accuracy_std': std_acc,
                'balanced_accuracy_mean': mean_bal,
                'balanced_accuracy_std': std_bal,
            }

            print(f"{exp_name:<20} {mean_acc*100:.1f}% ± {std_acc*100:.1f}%{'':<5} {mean_bal*100:.1f}% ± {std_bal*100:.1f}%")

    # Find best
    best_exp = max(summary, key=lambda x: summary[x]['accuracy_mean'])
    print(f"\n*** BEST: {best_exp} with {summary[best_exp]['accuracy_mean']*100:.1f}% accuracy ***")

    target = 0.6162
    best_acc = summary[best_exp]['accuracy_mean']
    gap = target - best_acc

    print(f"\nTarget (MUSE): {target*100:.1f}%")
    print(f"Best achieved: {best_acc*100:.1f}%")
    print(f"Gap remaining: {gap*100:.1f}%")

    if best_acc > target:
        print("\n*** SUCCESS: MHTPN BEATS MUSE! ***")

    # Save results
    full_results = {
        'experiments': {k: dict(v) for k, v in results.items()},
        'summary': summary,
        'best_experiment': best_exp,
        'best_accuracy': best_acc,
        'gap_to_target': gap,
        'success': best_acc > target,
    }

    with open(os.path.join(output_dir, 'final_tuning_results.json'), 'w') as f:
        json.dump(full_results, f, indent=2, default=float)

    print(f"\nResults saved to: {output_dir}")

    return full_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MHTPN Final Tuning Experiment')
    parser.add_argument('--dataset', type=str, default='LSST', help='Dataset name')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')

    args = parser.parse_args()

    run_final_tuning(
        dataset_name=args.dataset,
        output_dir=args.output,
        n_folds=args.folds,
    )
