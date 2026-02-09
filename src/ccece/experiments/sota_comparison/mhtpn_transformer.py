"""
MHTPN with Transformer Encoder - Deep Architectural Enhancement

Based on evidence from enhanced experiment:
- full_enhanced (attention + delta + multiscale) achieves 57.4% ± 1.9%
- Fold 5 achieved 60.7% - very close to MUSE (61.62%)
- Attention is the key beneficial component

This experiment uses a proper Transformer encoder with:
1. Stacked self-attention layers (not just single attention)
2. Learned positional embeddings for temporal awareness
3. Pre-layer normalization for training stability
4. Integration with trajectory prototype mechanism

Target: Beat MUSE (61.62%) on LSST

Usage:
    python3 -m ccece.experiments.sota_comparison.mhtpn_transformer \
        --output results/ccece/sota_comparison/lsst_transformer
"""

import os
import sys
import json
import math
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


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for temporal awareness."""

    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    Pre-layer norm Transformer encoder layer.
    Pre-norm is more stable for training.
    """

    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, src_mask=None):
        # Pre-norm attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=src_mask)
        x = x + attn_out

        # Pre-norm FFN
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)

        return x


class TransformerEncoder(nn.Module):
    """Stack of Transformer encoder layers."""

    def __init__(self, d_model, n_heads, n_layers, dim_feedforward, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class TrajectoryPrototypeHead(nn.Module):
    """
    Trajectory prototype head with temperature scaling.
    Prototype evolves: p(t) = origin + t * velocity
    """

    def __init__(self, latent_dim, head_dim, n_classes):
        super().__init__()
        self.latent_dim = latent_dim
        self.head_dim = head_dim
        self.n_classes = n_classes

        # Projection to head space
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, head_dim),
        )

        # Trajectory prototypes
        self.prototype_origins = nn.Parameter(
            torch.randn(n_classes, head_dim) * 0.1
        )
        self.prototype_velocities = nn.Parameter(
            torch.randn(n_classes, head_dim) * 0.05
        )

        # Learnable temperature
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, z, t_value=0.5):
        # z: (batch, latent_dim)
        batch_size = z.size(0)

        # Project and normalize
        h = self.projection(z)
        h = F.normalize(h, p=2, dim=1)  # (batch, head_dim)

        # Get prototype at time t
        t = torch.tensor(t_value, device=z.device)
        prototypes = self.prototype_origins + t * self.prototype_velocities  # (n_classes, head_dim)

        # Compute distances
        h_expanded = h.unsqueeze(1)  # (batch, 1, head_dim)
        proto_expanded = prototypes.unsqueeze(0)  # (1, n_classes, head_dim)
        distances = torch.sum((h_expanded - proto_expanded) ** 2, dim=-1)  # (batch, n_classes)

        # Convert to similarities with temperature
        similarities = torch.log(1 + 1 / (distances + 1e-6))
        temp = F.softplus(self.temperature) + 0.1
        logits = similarities / temp

        return logits


class MHTPNTransformer(nn.Module):
    """
    Multi-Head Trajectory Prototype Network with Transformer Encoder.

    Architecture:
        Input (batch, seq_len=36, input_dim=6)
            |
            v
        Optional Delta Features -> (batch, seq_len, input_dim * 2)
            |
            v
        Linear Projection -> (batch, seq_len, d_model)
            |
            v
        Positional Encoding
            |
            v
        Transformer Encoder (n_layers attention blocks)
            |
            v
        Global Average Pooling -> (batch, d_model)
            |
            v
        K Trajectory Prototype Heads
            |
            v
        Average logits -> Final prediction
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        dim_feedforward: int = 128,
        n_proto_heads: int = 5,
        head_dim: int = 32,
        dropout: float = 0.2,
        use_delta: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.d_model = d_model
        self.use_delta = use_delta

        # Effective input dim with delta features
        effective_input = input_dim * 2 if use_delta else input_dim

        # Input projection
        self.input_proj = nn.Linear(effective_input, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        # Transformer encoder
        self.transformer = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Projection to latent space
        self.latent_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Trajectory prototype heads
        self.heads = nn.ModuleList([
            TrajectoryPrototypeHead(d_model, head_dim, num_classes)
            for _ in range(n_proto_heads)
        ])

    def compute_delta(self, x):
        """Compute first-order differences."""
        delta = x[:, 1:, :] - x[:, :-1, :]
        pad = torch.zeros(x.size(0), 1, x.size(2), device=x.device, dtype=x.dtype)
        return torch.cat([pad, delta], dim=1)

    def forward(self, x, lengths=None):
        # x: (batch, seq_len, input_dim)

        # Add delta features
        if self.use_delta:
            delta = self.compute_delta(x)
            x = torch.cat([x, delta], dim=2)

        # Project to d_model dimensions
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through transformer
        x = self.transformer(x)  # (batch, seq_len, d_model)

        # Global average pooling
        z = x.mean(dim=1)  # (batch, d_model)

        # Project to latent space
        z = self.latent_proj(z)
        z = F.normalize(z, p=2, dim=1)

        # Get logits from all prototype heads
        all_logits = []
        for head in self.heads:
            head_logits = head(z, t_value=0.5)
            all_logits.append(head_logits)

        # Average logits
        logits = torch.stack(all_logits, dim=0).mean(dim=0)

        return logits


def train_and_evaluate(
    model, X_train, y_train, X_val, y_val, device,
    epochs=150, patience=30, lr=5e-4
):
    """Train model and return validation accuracy."""
    model = model.to(device)
    num_classes = len(np.unique(y_train))

    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long()
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Class-weighted loss
    class_counts = np.bincount(y_train, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts.astype(np.float32)
    class_weights = class_weights / class_weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(class_weights).float().to(device)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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

        # Validation
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


def run_transformer_experiments(
    dataset_name: str = 'LSST',
    output_dir: str = None,
    n_folds: int = 5,
):
    """Run Transformer architecture experiments."""
    set_all_seeds(RANDOM_SEED)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/ccece/sota_comparison/lsst_transformer_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("MHTPN TRANSFORMER ENCODER EXPERIMENT")
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

    # Architecture variants to test
    experiments = {
        # Baseline transformer with delta
        'transformer_1layer': {
            'd_model': 64, 'n_heads': 4, 'n_transformer_layers': 1,
            'dim_feedforward': 128, 'use_delta': True, 'lr': 5e-4
        },
        'transformer_2layer': {
            'd_model': 64, 'n_heads': 4, 'n_transformer_layers': 2,
            'dim_feedforward': 128, 'use_delta': True, 'lr': 5e-4
        },
        'transformer_3layer': {
            'd_model': 64, 'n_heads': 4, 'n_transformer_layers': 3,
            'dim_feedforward': 128, 'use_delta': True, 'lr': 5e-4
        },
        # Wider models
        'transformer_wide': {
            'd_model': 128, 'n_heads': 8, 'n_transformer_layers': 2,
            'dim_feedforward': 256, 'use_delta': True, 'lr': 3e-4
        },
        # No delta for comparison
        'transformer_no_delta': {
            'd_model': 64, 'n_heads': 4, 'n_transformer_layers': 2,
            'dim_feedforward': 128, 'use_delta': False, 'lr': 5e-4
        },
        # More prototype heads
        'transformer_more_heads': {
            'd_model': 64, 'n_heads': 4, 'n_transformer_layers': 2,
            'dim_feedforward': 128, 'use_delta': True, 'lr': 5e-4, 'n_proto_heads': 10
        },
    }

    # Cross-validation
    cv = get_cv_strategy(dataset_config, n_splits=n_folds, random_state=RANDOM_SEED)
    if dataset_config.has_groups:
        splits = list(cv.split(X, y, groups))
    else:
        splits = list(cv.split(X, y))

    results = {exp_name: {'accuracy': [], 'balanced_accuracy': []} for exp_name in experiments}

    # Save intermediate results
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

                lr = config.pop('lr', 5e-4)
                n_proto_heads = config.pop('n_proto_heads', 5)

                model = MHTPNTransformer(
                    input_dim=X_train.shape[2],
                    num_classes=dataset_config.n_classes,
                    seq_len=X_train.shape[1],
                    n_proto_heads=n_proto_heads,
                    head_dim=32,
                    dropout=0.2,
                    **config,
                )

                # Restore popped values for next fold
                config['lr'] = lr
                config['n_proto_heads'] = n_proto_heads

                val_acc, balanced_acc = train_and_evaluate(
                    model, X_train, y_train, X_val, y_val, device, lr=lr
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

        # Save intermediate results
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
                'accuracy_folds': accs,
                'balanced_accuracy_mean': mean_bal,
                'balanced_accuracy_std': std_bal,
            }

            print(f"{exp_name:<25} {mean_acc*100:.1f}% ± {std_acc*100:.1f}%{'':<3} {mean_bal*100:.1f}% ± {std_bal*100:.1f}%")

    # Find best
    best_exp = max(summary, key=lambda x: summary[x]['accuracy_mean'])
    print(f"\n*** BEST: {best_exp} with {summary[best_exp]['accuracy_mean']*100:.1f}% accuracy ***")

    # Compare to targets
    prev_best = 0.5736  # full_enhanced from previous experiment
    muse_target = 0.6162
    best_acc = summary[best_exp]['accuracy_mean']

    print(f"\nPrevious best (full_enhanced): {prev_best*100:.1f}%")
    print(f"Current best: {best_acc*100:.1f}%")
    print(f"Improvement: {(best_acc - prev_best)*100:+.1f}%")
    print(f"\nTarget (MUSE): {muse_target*100:.1f}%")
    print(f"Gap remaining: {(muse_target - best_acc)*100:.1f}%")

    if best_acc > muse_target:
        print("\n*** SUCCESS: MHTPN BEATS MUSE! ***")

    # Save results
    full_results = {
        'experiments': {k: dict(v) for k, v in results.items()},
        'summary': summary,
        'best_experiment': best_exp,
        'previous_best': prev_best,
        'gap_to_muse': muse_target - best_acc,
        'success': best_acc > muse_target,
    }

    with open(os.path.join(output_dir, 'transformer_results.json'), 'w') as f:
        json.dump(full_results, f, indent=2, default=float)

    print(f"\nResults saved to: {output_dir}")

    return full_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MHTPN Transformer Experiment')
    parser.add_argument('--dataset', type=str, default='LSST', help='Dataset name')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')

    args = parser.parse_args()

    run_transformer_experiments(
        dataset_name=args.dataset,
        output_dir=args.output,
        n_folds=args.folds,
    )
