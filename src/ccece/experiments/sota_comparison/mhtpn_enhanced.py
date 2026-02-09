"""
Enhanced MHTPN Architecture for LSST

Based on evidence:
- n_segments=1 + L2 norm achieves 54.8%
- FFT features hurt performance
- MUSE uses first-order differences (delta features)
- Classes 42 and 65 (large classes) have near-zero accuracy

Architectural enhancements:
1. Multi-scale convolutional encoding (different kernel sizes)
2. Self-attention over temporal positions
3. Delta features (first-order differences)
4. Deeper but narrower encoder with residual connections
5. Temperature scaling for prototype logits

Usage:
    python3 -m ccece.experiments.sota_comparison.mhtpn_enhanced \
        --output results/ccece/sota_comparison/lsst_enhanced
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
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from ccece.run_experiment import set_all_seeds

try:
    from .datasets import load_dataset, get_cv_strategy, standardize_data
except ImportError:
    from ccece.experiments.sota_comparison.datasets import load_dataset, get_cv_strategy, standardize_data


RANDOM_SEED = 42


class MultiScaleConvBlock(nn.Module):
    """
    Multi-scale convolutional block that processes input at multiple kernel sizes
    and concatenates the results. This helps capture patterns at different scales.
    """

    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], dropout=0.2):
        super().__init__()
        self.branches = nn.ModuleList()

        # Each kernel size gets a portion of the output channels
        branch_channels = out_channels // len(kernel_sizes)

        for ks in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(in_channels, branch_channels, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(branch_channels),
                nn.ReLU(),
            )
            self.branches.append(branch)

        # Adjust if division wasn't exact
        self.final_channels = branch_channels * len(kernel_sizes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        outputs = [branch(x) for branch in self.branches]
        out = torch.cat(outputs, dim=1)
        return self.dropout(out)


class TemporalAttention(nn.Module):
    """
    Self-attention over temporal positions.
    Helps identify which time points are most discriminative.
    """

    def __init__(self, channels, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        assert channels % n_heads == 0, "channels must be divisible by n_heads"

        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        # x: (batch, channels, seq_len)
        batch, channels, seq_len = x.shape

        # Transpose to (batch, seq_len, channels)
        x = x.transpose(1, 2)

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(batch, seq_len, channels)
        out = self.proj(out)

        # Transpose back to (batch, channels, seq_len)
        return out.transpose(1, 2)


class ResidualBlock(nn.Module):
    """Residual convolutional block with batch norm."""

    def __init__(self, channels, kernel_size=3, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class EnhancedTrajectoryHead(nn.Module):
    """
    Enhanced trajectory prototype head with learnable temperature scaling.
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

        # Prototype representations
        self.prototype_origins = nn.Parameter(
            torch.randn(n_classes, head_dim) * 0.1
        )
        self.prototype_velocities = nn.Parameter(
            torch.randn(n_classes, head_dim) * 0.05
        )

        # Learnable temperature for sharper/softer distributions
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Class weights for prototype (learnable)
        self.class_weights = nn.Parameter(torch.ones(n_classes))

    def forward(self, z_segments, segment_weights, t_values):
        batch_size, n_segments, latent_dim = z_segments.shape

        # Project to head space
        z_flat = z_segments.view(-1, latent_dim)
        h_flat = self.projection(z_flat)
        h_flat = F.normalize(h_flat, p=2, dim=1)
        h = h_flat.view(batch_size, n_segments, self.head_dim)

        # Get prototypes at time points
        t = t_values.view(-1, 1, 1)
        origins = self.prototype_origins.unsqueeze(0)
        velocities = self.prototype_velocities.unsqueeze(0)
        prototypes = origins + t * velocities  # (n_segments, n_classes, head_dim)

        # Compute distances
        h_expanded = h.unsqueeze(2)  # (batch, n_seg, 1, head_dim)
        proto_expanded = prototypes.unsqueeze(0)  # (1, n_seg, n_classes, head_dim)
        distances = torch.sum((h_expanded - proto_expanded) ** 2, dim=-1)

        # Convert to similarities with temperature
        similarities = torch.log(1 + 1 / (distances + 1e-6))

        # Weighted average across segments
        weights_sum = segment_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        normalized_weights = segment_weights / weights_sum
        weighted_sims = similarities * normalized_weights.unsqueeze(-1)
        trajectory_similarities = weighted_sims.sum(dim=1)

        # Apply temperature scaling and class weights
        temp = F.softplus(self.temperature) + 0.1  # Ensure positive temperature
        logits = trajectory_similarities * self.class_weights / temp

        return distances, similarities, trajectory_similarities, logits


class MHTPNEnhanced(nn.Module):
    """
    Enhanced Multi-Head Trajectory Prototype Network.

    Enhancements:
    1. Multi-scale convolution (captures patterns at different scales)
    2. Self-attention over temporal positions
    3. Delta features (first-order differences)
    4. Residual connections
    5. Temperature-scaled prototype heads
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
        dropout: float = 0.2,
        use_delta: bool = True,
        use_attention: bool = True,
        use_multiscale: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.use_delta = use_delta
        self.use_attention = use_attention
        self.use_multiscale = use_multiscale

        # Effective input dim with delta features
        effective_input = input_dim * 2 if use_delta else input_dim

        # Stage 1: Initial projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(effective_input, encoder_hidden, kernel_size=1),
            nn.BatchNorm1d(encoder_hidden),
            nn.ReLU(),
        )

        # Stage 2: Multi-scale or regular convolution
        if use_multiscale:
            self.conv_stage1 = MultiScaleConvBlock(
                encoder_hidden, encoder_hidden * 2, kernel_sizes=[3, 5, 7], dropout=dropout
            )
            stage1_out = (encoder_hidden * 2 // 3) * 3  # Adjusted for multi-scale
        else:
            self.conv_stage1 = nn.Sequential(
                nn.Conv1d(encoder_hidden, encoder_hidden * 2, kernel_size=7, padding=3),
                nn.BatchNorm1d(encoder_hidden * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            stage1_out = encoder_hidden * 2

        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Stage 3: Deeper processing
        self.conv_stage2 = nn.Sequential(
            nn.Conv1d(stage1_out, encoder_hidden * 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(encoder_hidden * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Stage 4: Residual refinement
        self.residual = ResidualBlock(encoder_hidden * 4, kernel_size=3, dropout=dropout)

        # Optional attention
        if use_attention:
            self.attention = TemporalAttention(encoder_hidden * 4, n_heads=4)

        self.encoder_output_dim = encoder_hidden * 4

        # Projection head
        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(encoder_hidden * 4, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )

        # Enhanced trajectory prototype heads
        self.heads = nn.ModuleList([
            EnhancedTrajectoryHead(latent_dim, head_dim, num_classes)
            for _ in range(n_heads)
        ])

        # Single segment
        self.register_buffer('t_default', torch.tensor([0.5]))

    def compute_delta(self, x):
        """Compute first-order differences."""
        delta = x[:, 1:, :] - x[:, :-1, :]
        pad = torch.zeros(x.size(0), 1, x.size(2), device=x.device, dtype=x.dtype)
        return torch.cat([pad, delta], dim=1)

    def encode(self, x):
        """Encode with all enhancements."""
        # Add delta features if enabled
        if self.use_delta:
            delta = self.compute_delta(x)
            x = torch.cat([x, delta], dim=2)

        # (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)

        # Encoding stages
        x = self.input_proj(x)
        x = self.conv_stage1(x)
        x = self.pool1(x)
        x = self.conv_stage2(x)
        x = self.pool2(x)
        x = self.residual(x)

        # Apply attention if enabled
        if self.use_attention:
            x = x + self.attention(x)  # Residual attention

        # Project to latent space
        z = self.projection_head(x)
        z = F.normalize(z, p=2, dim=1)

        return z.unsqueeze(1)  # (batch, 1, latent_dim)

    def forward(self, x, lengths=None):
        z_segments = self.encode(x)
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


def run_enhanced_experiments(
    dataset_name: str = 'LSST',
    output_dir: str = None,
    n_folds: int = 5,
):
    """Run enhanced architecture experiments."""
    set_all_seeds(RANDOM_SEED)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/ccece/sota_comparison/lsst_enhanced_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("MHTPN ENHANCED ARCHITECTURE EXPERIMENT")
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

    # Ablation experiments
    experiments = {
        'baseline': {
            'use_delta': False, 'use_attention': False, 'use_multiscale': False
        },
        'delta_only': {
            'use_delta': True, 'use_attention': False, 'use_multiscale': False
        },
        'attention_only': {
            'use_delta': False, 'use_attention': True, 'use_multiscale': False
        },
        'multiscale_only': {
            'use_delta': False, 'use_attention': False, 'use_multiscale': True
        },
        'delta_attention': {
            'use_delta': True, 'use_attention': True, 'use_multiscale': False
        },
        'delta_multiscale': {
            'use_delta': True, 'use_attention': False, 'use_multiscale': True
        },
        'full_enhanced': {
            'use_delta': True, 'use_attention': True, 'use_multiscale': True
        },
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
                model = MHTPNEnhanced(
                    input_dim=X_train.shape[2],
                    num_classes=dataset_config.n_classes,
                    seq_len=X_train.shape[1],
                    latent_dim=64,
                    n_heads=5,
                    head_dim=32,
                    encoder_hidden=64,
                    dropout=0.2,
                    **config,
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

    # Compare to target
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
        'gap_to_target': gap,
        'success': best_acc > target,
    }

    with open(os.path.join(output_dir, 'enhanced_results.json'), 'w') as f:
        json.dump(full_results, f, indent=2, default=float)

    print(f"\nResults saved to: {output_dir}")

    return full_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MHTPN Enhanced Architecture Experiment')
    parser.add_argument('--dataset', type=str, default='LSST', help='Dataset name')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')

    args = parser.parse_args()

    run_enhanced_experiments(
        dataset_name=args.dataset,
        output_dir=args.output,
        n_folds=args.folds,
    )
