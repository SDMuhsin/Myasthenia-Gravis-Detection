"""
MHTPN LSST Tuning Script

Based on evidence from deep diagnostics:
- MHTPN segment embeddings are 14.6% LESS separable than simple encoder
- L2 normalization in TrajectoryPrototypeHead loses discriminatory information
- The trajectory mechanism doesn't help for LSST's 14-class spectral data

Experiments to run:
1. Remove L2 normalization (restore magnitude information)
2. Increase latent_dim (64 → 128, 256)
3. Different n_segments (1, 2, 4, 6)
4. Replace prototype mechanism with direct classifier

Usage:
    python3 -m ccece.experiments.sota_comparison.mhtpn_lsst_tuning \
        --output results/ccece/sota_comparison/lsst_tuning_experiments
"""

import os
import sys
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tqdm import tqdm
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


# =============================================================================
# Modified MHTPN Components for Ablation
# =============================================================================

class TrajectoryPrototypeHeadNoNorm(nn.Module):
    """TrajectoryPrototypeHead WITHOUT L2 normalization."""

    def __init__(self, latent_dim: int, head_dim: int, n_classes: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.head_dim = head_dim
        self.n_classes = n_classes

        self.projection = nn.Sequential(
            nn.Linear(latent_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, head_dim),
        )

        self.prototype_origins = nn.Parameter(torch.randn(n_classes, head_dim) * 0.1)
        self.prototype_velocities = nn.Parameter(torch.randn(n_classes, head_dim) * 0.05)
        prototype_class = torch.arange(n_classes, dtype=torch.long)
        self.register_buffer('prototype_class', prototype_class)

    def get_prototype_at_time(self, t):
        t = t.view(-1, 1, 1)
        origins = self.prototype_origins.unsqueeze(0)
        velocities = self.prototype_velocities.unsqueeze(0)
        return origins + t * velocities

    def forward(self, z_segments, segment_weights, t_values):
        batch_size, n_segments, latent_dim = z_segments.shape

        z_flat = z_segments.view(-1, latent_dim)
        h_flat = self.projection(z_flat)
        # NO L2 NORMALIZATION - key change
        h = h_flat.view(batch_size, n_segments, self.head_dim)

        prototypes = self.get_prototype_at_time(t_values)
        h_expanded = h.unsqueeze(2)
        proto_expanded = prototypes.unsqueeze(0)
        distances = torch.sum((h_expanded - proto_expanded) ** 2, dim=-1)

        similarities = torch.log(1 + 1 / (distances + 1e-6))

        weights_sum = segment_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        normalized_weights = segment_weights / weights_sum

        weighted_sims = similarities * normalized_weights.unsqueeze(-1)
        trajectory_similarities = weighted_sims.sum(dim=1)
        logits = trajectory_similarities

        return distances, similarities, trajectory_similarities, logits


class DirectClassifierHead(nn.Module):
    """Simple classifier head - no prototypes, just MLP."""

    def __init__(self, latent_dim: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, z_segments, segment_weights, t_values):
        # Average segments
        weights_sum = segment_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        normalized_weights = segment_weights / weights_sum  # (batch, n_segments)
        weighted_z = (z_segments * normalized_weights.unsqueeze(-1)).sum(dim=1)  # (batch, latent_dim)
        logits = self.classifier(weighted_z)
        # Return dummy values for compatibility
        return None, None, None, logits


class MHTPNVariant(nn.Module):
    """
    MHTPN variant for ablation studies.

    Options:
    - use_l2_norm: whether to L2-normalize embeddings in heads
    - use_trajectory: whether to use trajectory prototypes or direct classifier
    - latent_dim: embedding dimension
    - n_segments: number of temporal segments
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        latent_dim: int = 64,
        n_heads: int = 5,
        head_dim: int = 32,
        n_segments: int = 2,
        encoder_hidden: int = 64,
        encoder_layers: int = 3,
        kernel_size: int = 7,
        dropout: float = 0.2,
        use_l2_norm: bool = True,
        use_trajectory: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_segments = n_segments
        self.use_l2_norm = use_l2_norm
        self.use_trajectory = use_trajectory

        # Segment boundaries
        segment_size = seq_len // n_segments
        self.segment_boundaries = []
        for i in range(n_segments):
            start = i * segment_size
            end = (i + 1) * segment_size if i < n_segments - 1 else seq_len
            self.segment_boundaries.append((start, end))

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

        # Build heads based on configuration
        if use_trajectory:
            if use_l2_norm:
                from ccece.models.multi_head_trajectory_proto_net import TrajectoryPrototypeHead
                self.heads = nn.ModuleList([
                    TrajectoryPrototypeHead(latent_dim, head_dim, num_classes)
                    for _ in range(n_heads)
                ])
            else:
                self.heads = nn.ModuleList([
                    TrajectoryPrototypeHeadNoNorm(latent_dim, head_dim, num_classes)
                    for _ in range(n_heads)
                ])
        else:
            # Single direct classifier
            self.heads = nn.ModuleList([
                DirectClassifierHead(latent_dim, head_dim * 2, num_classes)
            ])
            self.n_heads = 1

        t_default = torch.linspace(0, 1, n_segments + 1)
        t_midpoints = (t_default[:-1] + t_default[1:]) / 2
        self.register_buffer('t_default', t_midpoints)

    def encode_segments(self, x):
        batch_size = x.size(0)
        segment_encodings = []

        for start, end in self.segment_boundaries:
            segment = x[:, start:end, :]
            segment = segment.transpose(1, 2)
            features = self.encoder(segment)
            z = self.projection_head(features)
            if self.use_l2_norm and self.use_trajectory:
                z = F.normalize(z, p=2, dim=1)
            segment_encodings.append(z)

        return torch.stack(segment_encodings, dim=1)

    def compute_segment_weights(self, lengths):
        batch_size = lengths.size(0)
        device = lengths.device
        weights = torch.ones(batch_size, self.n_segments, device=device)
        return weights

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


def run_experiment(config, X_train, y_train, X_val, y_val, dataset_config, device):
    """Run a single experiment configuration."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model = MHTPNVariant(
        input_dim=X_train.shape[2],
        num_classes=dataset_config.n_classes,
        seq_len=X_train.shape[1],
        latent_dim=config['latent_dim'],
        n_heads=config['n_heads'],
        head_dim=config['head_dim'],
        n_segments=config['n_segments'],
        encoder_hidden=config['encoder_hidden'],
        encoder_layers=config['encoder_layers'],
        kernel_size=config['kernel_size'],
        dropout=config['dropout'],
        use_l2_norm=config['use_l2_norm'],
        use_trajectory=config['use_trajectory'],
    )

    val_acc, balanced_acc = train_and_evaluate(model, X_train, y_train, X_val, y_val, device)
    return val_acc, balanced_acc


def run_tuning_experiments(
    dataset_name: str = 'LSST',
    output_dir: str = None,
    n_folds: int = 3,
):
    """Run all tuning experiments."""
    set_all_seeds(RANDOM_SEED)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/ccece/sota_comparison/lsst_tuning_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("MHTPN LSST TUNING EXPERIMENTS")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print()

    # Load dataset
    print("Loading data...")
    X, y, groups, dataset_config = load_dataset(dataset_name, verbose=False)

    # Define experiment configurations
    base_config = {
        'latent_dim': 64,
        'n_heads': 5,
        'head_dim': 32,
        'n_segments': 2,
        'encoder_hidden': 64,
        'encoder_layers': 3,
        'kernel_size': 7,
        'dropout': 0.2,
        'use_l2_norm': True,
        'use_trajectory': True,
    }

    experiments = {
        # Baseline
        'baseline': base_config.copy(),

        # Ablation 1: Remove L2 normalization
        'no_l2_norm': {**base_config, 'use_l2_norm': False},

        # Ablation 2: Direct classifier (no prototypes)
        'direct_classifier': {**base_config, 'use_trajectory': False, 'use_l2_norm': False},

        # Ablation 3: Increase latent_dim
        'latent_128': {**base_config, 'latent_dim': 128, 'use_l2_norm': False},
        'latent_256': {**base_config, 'latent_dim': 256, 'use_l2_norm': False},

        # Ablation 4: Different n_segments
        'segments_1': {**base_config, 'n_segments': 1, 'use_l2_norm': False},
        'segments_4': {**base_config, 'n_segments': 4, 'use_l2_norm': False},
        'segments_6': {**base_config, 'n_segments': 6, 'use_l2_norm': False},

        # Ablation 5: More heads
        'heads_10': {**base_config, 'n_heads': 10, 'use_l2_norm': False},

        # Combined: larger latent + no L2 + more segments
        'best_combo': {
            **base_config,
            'latent_dim': 128,
            'n_segments': 4,
            'n_heads': 8,
            'use_l2_norm': False,
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
                val_acc, balanced_acc = run_experiment(
                    config, X_train, y_train, X_val, y_val, dataset_config, device
                )
                results[exp_name]['accuracy'].append(val_acc)
                results[exp_name]['balanced_accuracy'].append(balanced_acc)
                print(f"Acc: {val_acc*100:.1f}%, Bal: {balanced_acc*100:.1f}%")
            except Exception as e:
                print(f"ERROR: {e}")
                results[exp_name]['accuracy'].append(None)
                results[exp_name]['balanced_accuracy'].append(None)

            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print('=' * 70)
    print(f"\n{'Experiment':<25} {'Accuracy':<20} {'Balanced Acc':<20}")
    print("-" * 65)

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

            print(f"{exp_name:<25} {mean_acc*100:.1f}% ± {std_acc*100:.1f}%{'':<5} {mean_bal*100:.1f}% ± {std_bal*100:.1f}%")

    # Find best
    best_exp = max(summary, key=lambda x: summary[x]['accuracy_mean'])
    print(f"\n*** BEST: {best_exp} with {summary[best_exp]['accuracy_mean']*100:.1f}% accuracy ***")

    # Compare to baseline
    baseline_acc = summary['baseline']['accuracy_mean']
    best_acc = summary[best_exp]['accuracy_mean']
    improvement = (best_acc - baseline_acc) * 100

    print(f"\nBaseline: {baseline_acc*100:.1f}%")
    print(f"Best:     {best_acc*100:.1f}%")
    print(f"Improvement: {improvement:+.1f}%")

    # Target comparison
    target = 0.6162  # MUSE baseline
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

    with open(os.path.join(output_dir, 'tuning_results.json'), 'w') as f:
        json.dump(full_results, f, indent=2, default=float)

    print(f"\nResults saved to: {output_dir}")

    return full_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MHTPN LSST Tuning Experiments')
    parser.add_argument('--dataset', type=str, default='LSST', help='Dataset name')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--folds', type=int, default=3, help='Number of CV folds')

    args = parser.parse_args()

    run_tuning_experiments(
        dataset_name=args.dataset,
        output_dir=args.output,
        n_folds=args.folds,
    )
