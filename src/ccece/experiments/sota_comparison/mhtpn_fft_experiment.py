"""
MHTPN with FFT Features Experiment

Based on evidence that:
- MHTPN (time-domain CNN): 47.1% on LSST
- MUSE (frequency-domain SFA): 61.6% on LSST
- Gap: 14.5%

Hypothesis: Adding frequency-domain features (FFT) may help MHTPN capture
patterns that MUSE captures with SFA.

This script tests:
1. MHTPN with FFT magnitude features concatenated to input
2. MHTPN with FFT features only (no time-domain)
3. MHTPN with FFT phase + magnitude

Usage:
    python3 -m ccece.experiments.sota_comparison.mhtpn_fft_experiment \
        --output results/ccece/sota_comparison/lsst_fft_experiment
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
    from .mhtpn_configs import get_mhtpn_model_config, get_mhtpn_training_config
except ImportError:
    from ccece.experiments.sota_comparison.datasets import load_dataset, get_cv_strategy, standardize_data
    from ccece.experiments.sota_comparison.mhtpn_configs import get_mhtpn_model_config, get_mhtpn_training_config


RANDOM_SEED = 42


def compute_fft_features(X):
    """
    Compute FFT features for time series data.

    Args:
        X: (batch, seq_len, n_features)

    Returns:
        fft_features: (batch, seq_len, n_features) - FFT magnitude
    """
    # FFT along time dimension (axis=1)
    fft = np.fft.fft(X, axis=1)
    fft_magnitude = np.abs(fft)
    fft_phase = np.angle(fft)

    # Only keep first half (symmetric)
    half_len = X.shape[1] // 2 + 1
    fft_magnitude = fft_magnitude[:, :half_len, :]
    fft_phase = fft_phase[:, :half_len, :]

    # Pad to original length
    pad_len = X.shape[1] - half_len
    fft_magnitude = np.concatenate([
        fft_magnitude,
        np.zeros((X.shape[0], pad_len, X.shape[2]))
    ], axis=1)
    fft_phase = np.concatenate([
        fft_phase,
        np.zeros((X.shape[0], pad_len, X.shape[2]))
    ], axis=1)

    return fft_magnitude, fft_phase


class MHTPNWithFFT(nn.Module):
    """
    MHTPN variant with optional FFT features.

    Modes:
    - 'time_only': Original time-domain input
    - 'fft_only': FFT magnitude only
    - 'time_fft': Concatenate time + FFT magnitude
    - 'time_fft_phase': Concatenate time + FFT magnitude + FFT phase
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        latent_dim: int = 64,
        n_heads: int = 5,
        head_dim: int = 32,
        n_segments: int = 1,
        encoder_hidden: int = 64,
        encoder_layers: int = 3,
        kernel_size: int = 7,
        dropout: float = 0.2,
        mode: str = 'time_fft',
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.mode = mode
        self.n_segments = n_segments
        self.latent_dim = latent_dim

        # Compute effective input dim based on mode
        if mode == 'time_only':
            effective_input_dim = input_dim
        elif mode == 'fft_only':
            effective_input_dim = input_dim
        elif mode == 'time_fft':
            effective_input_dim = input_dim * 2
        elif mode == 'time_fft_phase':
            effective_input_dim = input_dim * 3
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.effective_input_dim = effective_input_dim

        # Segment boundaries
        segment_size = seq_len // n_segments
        self.segment_boundaries = []
        for i in range(n_segments):
            start = i * segment_size
            end = (i + 1) * segment_size if i < n_segments - 1 else seq_len
            self.segment_boundaries.append((start, end))

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

        # Build trajectory prototype heads (using original implementation)
        from ccece.models.multi_head_trajectory_proto_net import TrajectoryPrototypeHead
        self.heads = nn.ModuleList([
            TrajectoryPrototypeHead(latent_dim, head_dim, num_classes)
            for _ in range(n_heads)
        ])

        t_default = torch.linspace(0, 1, n_segments + 1)
        t_midpoints = (t_default[:-1] + t_default[1:]) / 2
        self.register_buffer('t_default', t_midpoints)

    def augment_with_fft(self, x):
        """
        Augment input with FFT features based on mode.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            augmented_x: (batch, seq_len, effective_input_dim)
        """
        if self.mode == 'time_only':
            return x

        # Compute FFT
        fft = torch.fft.fft(x, dim=1)
        fft_magnitude = torch.abs(fft)

        if self.mode == 'fft_only':
            return fft_magnitude

        elif self.mode == 'time_fft':
            return torch.cat([x, fft_magnitude], dim=2)

        elif self.mode == 'time_fft_phase':
            fft_phase = torch.angle(fft)
            return torch.cat([x, fft_magnitude, fft_phase], dim=2)

        return x

    def encode_segments(self, x):
        """Encode segments with FFT augmentation."""
        batch_size = x.size(0)
        segment_encodings = []

        # Augment with FFT
        x_augmented = self.augment_with_fft(x)

        for start, end in self.segment_boundaries:
            segment = x_augmented[:, start:end, :]
            segment = segment.transpose(1, 2)  # (batch, features, seq_len)
            features = self.encoder(segment)
            z = self.projection_head(features)
            z = F.normalize(z, p=2, dim=1)
            segment_encodings.append(z)

        return torch.stack(segment_encodings, dim=1)

    def compute_segment_weights(self, lengths):
        batch_size = lengths.size(0)
        device = lengths.device
        return torch.ones(batch_size, self.n_segments, device=device)

    def forward(self, x, lengths=None):
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), device=x.device, dtype=torch.long)

        z_segments = self.encode_segments(x)
        segment_weights = self.compute_segment_weights(lengths)

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


def run_fft_experiments(
    dataset_name: str = 'LSST',
    output_dir: str = None,
    n_folds: int = 5,
):
    """Run FFT feature experiments."""
    set_all_seeds(RANDOM_SEED)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/ccece/sota_comparison/lsst_fft_experiment_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("MHTPN FFT FEATURE EXPERIMENT")
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
        'time_only': {'mode': 'time_only', 'n_segments': 1},  # Best from previous tuning
        'fft_only': {'mode': 'fft_only', 'n_segments': 1},
        'time_fft': {'mode': 'time_fft', 'n_segments': 1},
        'time_fft_phase': {'mode': 'time_fft_phase', 'n_segments': 1},
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
                model = MHTPNWithFFT(
                    input_dim=X_train.shape[2],
                    num_classes=dataset_config.n_classes,
                    seq_len=X_train.shape[1],
                    latent_dim=64,
                    n_heads=5,
                    head_dim=32,
                    n_segments=config['n_segments'],
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
    baseline_acc = summary.get('time_only', {}).get('accuracy_mean', 0.45)
    best_acc = summary[best_exp]['accuracy_mean']
    improvement = (best_acc - baseline_acc) * 100

    print(f"\nBaseline (time_only): {baseline_acc*100:.1f}%")
    print(f"Best:                 {best_acc*100:.1f}%")
    print(f"Improvement:          {improvement:+.1f}%")

    target = 0.6162
    gap = target - best_acc
    print(f"\nTarget (MUSE): {target*100:.1f}%")
    print(f"Gap remaining: {gap*100:.1f}%")

    # Save results
    full_results = {
        'experiments': {k: dict(v) for k, v in results.items()},
        'summary': summary,
        'best_experiment': best_exp,
        'improvement_over_baseline': improvement,
        'gap_to_target': gap,
    }

    with open(os.path.join(output_dir, 'fft_results.json'), 'w') as f:
        json.dump(full_results, f, indent=2, default=float)

    print(f"\nResults saved to: {output_dir}")

    return full_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MHTPN FFT Feature Experiment')
    parser.add_argument('--dataset', type=str, default='LSST', help='Dataset name')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')

    args = parser.parse_args()

    run_fft_experiments(
        dataset_name=args.dataset,
        output_dir=args.output,
        n_folds=args.folds,
    )
