"""
MHTPN with Delta (First-Order Difference) Features

Key insight from MUSE analysis:
- MUSE uses `use_first_order_differences=True` which captures rate of change
- This is likely a key feature for LSST astronomical data
- Delta features capture the slope/derivative of light curves

Experiments:
1. time_only - baseline (54.8%)
2. delta_only - first-order differences only
3. time_delta - concatenate time + delta features
4. time_delta_normed - normalize delta features before concatenating

Usage:
    python3 -m ccece.experiments.sota_comparison.mhtpn_delta_experiment \
        --output results/ccece/sota_comparison/lsst_delta_experiment
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


def compute_delta_features(X):
    """
    Compute first-order differences (delta features).

    This is similar to what MUSE does with `use_first_order_differences=True`.

    Args:
        X: (batch, seq_len, n_features)

    Returns:
        delta: (batch, seq_len, n_features) - padded to match original length
    """
    # First-order difference: x[t] - x[t-1]
    delta = np.diff(X, axis=1)

    # Pad with zeros at the beginning to match original length
    pad = np.zeros((X.shape[0], 1, X.shape[2]))
    delta = np.concatenate([pad, delta], axis=1)

    return delta.astype(np.float32)


class MHTPNWithDelta(nn.Module):
    """
    MHTPN with optional delta (first-order difference) features.

    Modes:
    - 'time_only': Original time-domain input
    - 'delta_only': First-order differences only
    - 'time_delta': Concatenate time + delta features
    - 'time_delta_second': Time + delta + second-order delta
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
        mode: str = 'time_delta',
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.mode = mode
        self.latent_dim = latent_dim

        # Compute effective input dim based on mode
        if mode == 'time_only':
            effective_input_dim = input_dim
        elif mode == 'delta_only':
            effective_input_dim = input_dim
        elif mode == 'time_delta':
            effective_input_dim = input_dim * 2
        elif mode == 'time_delta_second':
            effective_input_dim = input_dim * 3
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.effective_input_dim = effective_input_dim

        # Build encoder
        layers = []
        in_channels = effective_input_dim
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

        # Build trajectory prototype heads
        from ccece.models.multi_head_trajectory_proto_net import TrajectoryPrototypeHead
        self.heads = nn.ModuleList([
            TrajectoryPrototypeHead(latent_dim, head_dim, num_classes)
            for _ in range(n_heads)
        ])

        # Single segment
        self.register_buffer('t_default', torch.tensor([0.5]))

    def augment_with_delta(self, x):
        """
        Augment input with delta features based on mode.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            augmented_x: (batch, seq_len, effective_input_dim)
        """
        if self.mode == 'time_only':
            return x

        # Compute first-order differences
        delta = x[:, 1:, :] - x[:, :-1, :]
        # Pad with zeros at beginning
        pad = torch.zeros(x.size(0), 1, x.size(2), device=x.device, dtype=x.dtype)
        delta = torch.cat([pad, delta], dim=1)

        if self.mode == 'delta_only':
            return delta

        elif self.mode == 'time_delta':
            return torch.cat([x, delta], dim=2)

        elif self.mode == 'time_delta_second':
            # Second-order differences
            delta2 = delta[:, 1:, :] - delta[:, :-1, :]
            pad2 = torch.zeros(x.size(0), 1, x.size(2), device=x.device, dtype=x.dtype)
            delta2 = torch.cat([pad2, delta2], dim=1)
            return torch.cat([x, delta, delta2], dim=2)

        return x

    def encode(self, x):
        """Encode full sequence with delta augmentation."""
        # Augment with delta features
        x_augmented = self.augment_with_delta(x)

        x_augmented = x_augmented.transpose(1, 2)  # (batch, features, seq_len)
        features = self.encoder(x_augmented)
        z = self.projection_head(features)
        z = F.normalize(z, p=2, dim=1)  # L2 normalization
        return z.unsqueeze(1)  # (batch, 1, latent_dim)

    def forward(self, x, lengths=None):
        z_segments = self.encode(x)  # (batch, 1, latent_dim)
        segment_weights = torch.ones(x.size(0), 1, device=x.device)

        all_logits = []
        for head in self.heads:
            _, _, _, head_logits = head(z_segments, segment_weights, self.t_default)
            all_logits.append(head_logits)

        logits = torch.stack(all_logits, dim=0).mean(dim=0)
        return logits


def train_and_evaluate(model, X_train, y_train, X_val, y_val, device, epochs=150, patience=30):
    """Train model and return validation accuracy."""
    model = model.to(device)
    num_classes = len(np.unique(y_train))

    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    class_counts = np.bincount(y_train, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts.astype(np.float32)
    class_weights = class_weights / class_weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).float().to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_balanced_acc = 0.0
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
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_acc, best_balanced_acc


def run_delta_experiments(
    dataset_name: str = 'LSST',
    output_dir: str = None,
    n_folds: int = 5,
):
    """Run delta feature experiments."""
    set_all_seeds(RANDOM_SEED)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/ccece/sota_comparison/lsst_delta_experiment_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("MHTPN DELTA FEATURE EXPERIMENT")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print()
    print("Hypothesis: MUSE uses first-order differences which capture rate of change.")
    print("This experiment tests if delta features help MHTPN.")
    print()

    # Load dataset
    print("Loading data...")
    X, y, groups, dataset_config = load_dataset(dataset_name, verbose=False)

    print(f"Dataset: {dataset_name}")
    print(f"  Shape: {X.shape}")
    print(f"  Classes: {dataset_config.n_classes}")

    # Experiments
    experiments = {
        'time_only': {'mode': 'time_only'},
        'delta_only': {'mode': 'delta_only'},
        'time_delta': {'mode': 'time_delta'},
        'time_delta_second': {'mode': 'time_delta_second'},
    }

    # Cross-validation
    cv = get_cv_strategy(dataset_config, n_splits=n_folds, random_state=RANDOM_SEED)
    if dataset_config.has_groups:
        splits = list(cv.split(X, y, groups))
    else:
        splits = list(cv.split(X, y))

    results = {exp_name: {'accuracy': [], 'balanced_accuracy': []} for exp_name in experiments}

    # Save intermediate results for crash protection
    intermediate_path = os.path.join(output_dir, 'intermediate_results.json')

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
                model = MHTPNWithDelta(
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
                    mode=config['mode'],
                )

                val_acc, balanced_acc = train_and_evaluate(
                    model, X_train, y_train, X_val, y_val, device
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

        # Save intermediate results after each fold
        intermediate_results = {
            'fold': fold_idx + 1,
            'experiments': {k: dict(v) for k, v in results.items()},
        }
        with open(intermediate_path, 'w') as f:
            json.dump(intermediate_results, f, indent=2, default=float)

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

    # Compare to baseline and target
    baseline_acc = summary.get('time_only', {}).get('accuracy_mean', 0.548)
    best_acc = summary[best_exp]['accuracy_mean']
    improvement = (best_acc - baseline_acc) * 100

    print(f"\nBaseline (time_only): {baseline_acc*100:.1f}%")
    print(f"Best:                 {best_acc*100:.1f}%")
    print(f"Improvement:          {improvement:+.1f}%")

    target = 0.6162
    gap = target - best_acc
    print(f"\nTarget (MUSE): {target*100:.1f}%")
    print(f"Gap remaining: {gap*100:.1f}%")

    if best_acc > target:
        print("\n*** SUCCESS: MHTPN BEATS MUSE! ***")

    # Save results
    full_results = {
        'experiments': {k: dict(v) for k, v in results.items()},
        'summary': summary,
        'best_experiment': best_exp,
        'improvement_over_baseline': improvement,
        'gap_to_target': gap,
        'success': best_acc > target,
    }

    with open(os.path.join(output_dir, 'delta_results.json'), 'w') as f:
        json.dump(full_results, f, indent=2, default=float)

    print(f"\nResults saved to: {output_dir}")

    return full_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MHTPN Delta Feature Experiment')
    parser.add_argument('--dataset', type=str, default='LSST', help='Dataset name')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')

    args = parser.parse_args()

    run_delta_experiments(
        dataset_name=args.dataset,
        output_dir=args.output,
        n_folds=args.folds,
    )
