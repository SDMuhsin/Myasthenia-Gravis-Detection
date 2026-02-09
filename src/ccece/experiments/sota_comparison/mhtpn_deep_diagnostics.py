"""
MHTPN Deep Diagnostic Script for LSST Dataset

Goes deeper into understanding why MHTPN fails on LSST:
1. Embedding visualization (t-SNE/UMAP) - are classes separable?
2. Encoder output analysis - what signals survive the encoder?
3. Ablation: raw encoder vs full MHTPN
4. Segment analysis: what happens with different n_segments?

Usage:
    python3 -m ccece.experiments.sota_comparison.mhtpn_deep_diagnostics \
        --output results/ccece/sota_comparison/lsst_deep_diagnostics
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt
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


class SimpleEncoderBaseline(nn.Module):
    """Simple CNN encoder without trajectory/prototype components."""

    def __init__(self, input_dim, num_classes, seq_len, hidden=64, layers=3, kernel_size=7, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seq_len = seq_len

        # Build encoder
        encoder_layers = []
        in_channels = input_dim
        for i in range(layers):
            out_channels = hidden * (2 ** min(i, 2))
            encoder_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.encoder = nn.Sequential(*encoder_layers)
        self.encoder_output_dim = in_channels

        # Compute output size after encoder
        with torch.no_grad():
            dummy = torch.zeros(1, input_dim, seq_len)
            enc_out = self.encoder(dummy)
            self.encoded_seq_len = enc_out.shape[2]

        # Global pooling + classifier
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.encoder_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def get_embeddings(self, x):
        """Get embeddings before classifier."""
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        features = self.encoder(x)
        pooled = self.pool(features).squeeze(-1)
        return pooled

    def forward(self, x):
        x = x.transpose(1, 2)
        features = self.encoder(x)
        pooled = self.pool(features)
        logits = self.classifier(pooled)
        return logits


def extract_mhtpn_embeddings(model, X, device, batch_size=64):
    """Extract segment embeddings and raw encoder features from MHTPN."""
    model.eval()
    all_segment_embeddings = []
    all_encoder_features = []

    dataset = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)

            # Get segment embeddings using MHTPN's method
            z_segments = model.encode_segments(X_batch)  # (batch, n_segments, latent_dim)

            # Also get raw encoder features for first segment
            segment = X_batch[:, :X_batch.size(1)//model.n_segments, :]
            segment = segment.transpose(1, 2)
            raw_features = model.encoder(segment)
            pooled = model.projection_head[:2](raw_features)  # Just pool and flatten

            all_segment_embeddings.append(z_segments.cpu().numpy())
            all_encoder_features.append(pooled.cpu().numpy())

    segment_embeddings = np.concatenate(all_segment_embeddings, axis=0)
    encoder_features = np.concatenate(all_encoder_features, axis=0)

    return segment_embeddings, encoder_features


def visualize_embeddings(embeddings, labels, class_names, output_path, title):
    """Visualize embeddings using t-SNE."""
    print(f"  Computing t-SNE for {title}...")

    # If embeddings are 3D (batch, segments, features), flatten
    if len(embeddings.shape) == 3:
        # Average across segments
        embeddings = embeddings.mean(axis=1)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=RANDOM_SEED)
    coords = tsne.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i]], label=class_names[label],
                   alpha=0.6, s=30)

    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.set_title(title)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return coords


def compute_embedding_separability(embeddings, labels):
    """Compute how separable the embeddings are using linear probe."""
    if len(embeddings.shape) == 3:
        embeddings = embeddings.mean(axis=1)

    # Use 80/20 split
    n = len(labels)
    indices = np.random.RandomState(RANDOM_SEED).permutation(n)
    train_idx = indices[:int(0.8 * n)]
    test_idx = indices[int(0.8 * n):]

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight='balanced')
    clf.fit(embeddings[train_idx], labels[train_idx])

    train_pred = clf.predict(embeddings[train_idx])
    test_pred = clf.predict(embeddings[test_idx])

    return {
        'train_accuracy': accuracy_score(labels[train_idx], train_pred),
        'train_balanced_accuracy': balanced_accuracy_score(labels[train_idx], train_pred),
        'test_accuracy': accuracy_score(labels[test_idx], test_pred),
        'test_balanced_accuracy': balanced_accuracy_score(labels[test_idx], test_pred),
    }


def train_simple_baseline(X_train, y_train, X_val, y_val, config, device, epochs=100, patience=20):
    """Train simple encoder baseline without MHTPN components."""
    model = SimpleEncoderBaseline(
        input_dim=X_train.shape[2],
        num_classes=len(np.unique(y_train)),
        seq_len=X_train.shape[1],
        hidden=config['encoder_hidden'],
        layers=config['encoder_layers'],
        kernel_size=config['kernel_size'],
        dropout=config['dropout'],
    ).to(device)

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

    # Class weights
    class_counts = np.bincount(y_train, minlength=len(np.unique(y_train)))
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts.astype(np.float32)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).float().to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
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
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    model.to(device)

    return model, best_val_acc


def run_deep_diagnostics(
    dataset_name: str = 'LSST',
    output_dir: str = None,
    n_folds: int = 3,  # Reduced for speed
):
    """Run deep diagnostic analysis."""
    set_all_seeds(RANDOM_SEED)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/ccece/sota_comparison/lsst_deep_diagnostics_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("MHTPN DEEP DIAGNOSTIC ANALYSIS")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print()

    # Load dataset
    print("Loading data...")
    X, y, groups, dataset_config = load_dataset(dataset_name, verbose=True)
    class_names = dataset_config.class_names

    print(f"\nDataset: {dataset_name}")
    print(f"  Shape: {X.shape}")
    print(f"  Classes: {len(class_names)}")

    # Cross-validation
    cv = get_cv_strategy(dataset_config, n_splits=n_folds, random_state=RANDOM_SEED)
    if dataset_config.has_groups:
        splits = list(cv.split(X, y, groups))
    else:
        splits = list(cv.split(X, y))

    results = {
        'dataset': dataset_name,
        'folds': [],
        'embedding_analysis': [],
        'ablation': [],
    }

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print('=' * 60)

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        X_train, X_val = standardize_data(X_train, X_val)

        # Get MHTPN config
        model_config = get_mhtpn_model_config(dataset_config.name, X_train.shape[1])
        train_config = get_mhtpn_training_config()

        print(f"\n  MHTPN Config:")
        print(f"    n_segments: {model_config.n_segments}")
        print(f"    latent_dim: {model_config.latent_dim}")
        print(f"    n_heads: {model_config.n_heads}")

        # ========================================
        # Part 1: Train MHTPN and analyze embeddings
        # ========================================
        print(f"\n  Training MHTPN...")
        from ccece.models.multi_head_trajectory_proto_net import MultiHeadTrajectoryProtoNet

        torch.manual_seed(RANDOM_SEED + fold_idx)
        mhtpn_model = MultiHeadTrajectoryProtoNet(
            input_dim=X_train.shape[2],
            num_classes=dataset_config.n_classes,
            seq_len=X_train.shape[1],
            latent_dim=model_config.latent_dim,
            n_heads=model_config.n_heads,
            head_dim=model_config.head_dim,
            n_segments=model_config.n_segments,
            encoder_hidden=model_config.encoder_hidden,
            encoder_layers=model_config.encoder_layers,
            kernel_size=model_config.kernel_size,
            dropout=model_config.dropout,
        ).to(device)

        # Train MHTPN
        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).long()
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).long()
        )
        train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=train_config.batch_size, shuffle=False)

        class_counts = np.bincount(y_train, minlength=dataset_config.n_classes)
        class_counts = np.maximum(class_counts, 1)
        class_weights = 1.0 / class_counts.astype(np.float32)
        class_weights = class_weights / class_weights.sum() * dataset_config.n_classes
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).float().to(device))

        optimizer = torch.optim.AdamW(mhtpn_model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config.epochs)

        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0

        for epoch in range(train_config.epochs):
            mhtpn_model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = mhtpn_model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mhtpn_model.parameters(), train_config.grad_clip_norm)
                optimizer.step()
            scheduler.step()

            mhtpn_model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    logits = mhtpn_model(X_batch)
                    preds = logits.argmax(dim=1)
                    correct += (preds == y_batch).sum().item()
                    total += y_batch.size(0)

            val_acc = correct / total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.cpu().clone() for k, v in mhtpn_model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= train_config.early_stopping_patience:
                    break

        if best_model_state:
            mhtpn_model.load_state_dict(best_model_state)
        mhtpn_model.to(device)

        print(f"    MHTPN Val Accuracy: {best_val_acc*100:.2f}%")

        # ========================================
        # Part 2: Extract and analyze embeddings
        # ========================================
        print(f"\n  Extracting embeddings...")
        segment_embeddings, encoder_features = extract_mhtpn_embeddings(mhtpn_model, X_val, device)

        print(f"    Segment embeddings shape: {segment_embeddings.shape}")
        print(f"    Encoder features shape: {encoder_features.shape}")

        # Compute embedding separability
        print(f"\n  Computing embedding separability (linear probe)...")
        seg_emb_separability = compute_embedding_separability(segment_embeddings, y_val)
        enc_separability = compute_embedding_separability(encoder_features, y_val)

        print(f"    Segment embeddings:")
        print(f"      Train acc: {seg_emb_separability['train_accuracy']*100:.1f}%")
        print(f"      Test acc: {seg_emb_separability['test_accuracy']*100:.1f}%")
        print(f"      Test balanced acc: {seg_emb_separability['test_balanced_accuracy']*100:.1f}%")

        print(f"    Raw encoder features:")
        print(f"      Train acc: {enc_separability['train_accuracy']*100:.1f}%")
        print(f"      Test acc: {enc_separability['test_accuracy']*100:.1f}%")
        print(f"      Test balanced acc: {enc_separability['test_balanced_accuracy']*100:.1f}%")

        # Visualize embeddings
        fold_output_dir = os.path.join(output_dir, f'fold{fold_idx+1}')
        os.makedirs(fold_output_dir, exist_ok=True)

        visualize_embeddings(
            segment_embeddings, y_val, class_names,
            os.path.join(fold_output_dir, 'segment_embeddings_tsne.png'),
            f'MHTPN Segment Embeddings (Fold {fold_idx+1})'
        )

        visualize_embeddings(
            encoder_features, y_val, class_names,
            os.path.join(fold_output_dir, 'encoder_features_tsne.png'),
            f'Raw Encoder Features (Fold {fold_idx+1})'
        )

        # ========================================
        # Part 3: Ablation - Simple encoder baseline
        # ========================================
        print(f"\n  Training simple encoder baseline (no trajectory/prototype)...")
        simple_config = {
            'encoder_hidden': model_config.encoder_hidden,
            'encoder_layers': model_config.encoder_layers,
            'kernel_size': model_config.kernel_size,
            'dropout': model_config.dropout,
        }

        simple_model, simple_acc = train_simple_baseline(
            X_train, y_train, X_val, y_val, simple_config, device
        )
        print(f"    Simple Encoder Val Accuracy: {simple_acc*100:.2f}%")

        # Compare
        improvement = simple_acc - best_val_acc
        print(f"\n  Comparison:")
        print(f"    MHTPN:          {best_val_acc*100:.2f}%")
        print(f"    Simple Encoder: {simple_acc*100:.2f}%")
        print(f"    Difference:     {improvement*100:+.2f}%")

        if improvement > 0:
            print(f"    >>> Simple encoder is BETTER - trajectory/prototype components may be HURTING")
        else:
            print(f"    >>> MHTPN is better - trajectory/prototype provides some benefit")

        # Extract simple encoder embeddings for comparison
        simple_embeddings = []
        simple_model.eval()
        with torch.no_grad():
            for (X_batch,) in DataLoader(TensorDataset(torch.from_numpy(X_val).float()), batch_size=64):
                X_batch = X_batch.to(device)
                emb = simple_model.get_embeddings(X_batch)
                simple_embeddings.append(emb.cpu().numpy())
        simple_embeddings = np.concatenate(simple_embeddings, axis=0)

        simple_separability = compute_embedding_separability(simple_embeddings, y_val)
        print(f"\n    Simple encoder embedding separability:")
        print(f"      Test acc: {simple_separability['test_accuracy']*100:.1f}%")
        print(f"      Test balanced acc: {simple_separability['test_balanced_accuracy']*100:.1f}%")

        visualize_embeddings(
            simple_embeddings, y_val, class_names,
            os.path.join(fold_output_dir, 'simple_encoder_embeddings_tsne.png'),
            f'Simple Encoder Embeddings (Fold {fold_idx+1})'
        )

        # Store results
        fold_results = {
            'mhtpn_accuracy': float(best_val_acc),
            'simple_encoder_accuracy': float(simple_acc),
            'segment_embedding_separability': seg_emb_separability,
            'encoder_feature_separability': enc_separability,
            'simple_embedding_separability': simple_separability,
        }
        results['folds'].append(fold_results)

        # Save fold results
        with open(os.path.join(fold_output_dir, 'results.json'), 'w') as f:
            json.dump(fold_results, f, indent=2)

        torch.cuda.empty_cache()

    # ========================================
    # Summary
    # ========================================
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print('=' * 70)

    mhtpn_accs = [f['mhtpn_accuracy'] for f in results['folds']]
    simple_accs = [f['simple_encoder_accuracy'] for f in results['folds']]

    print(f"\nMHTPN Accuracy: {np.mean(mhtpn_accs)*100:.2f}% ± {np.std(mhtpn_accs)*100:.2f}%")
    print(f"Simple Encoder: {np.mean(simple_accs)*100:.2f}% ± {np.std(simple_accs)*100:.2f}%")

    avg_improvement = np.mean(simple_accs) - np.mean(mhtpn_accs)
    print(f"\nSimple vs MHTPN: {avg_improvement*100:+.2f}%")

    if avg_improvement > 0.02:  # >2% improvement
        print("\n*** FINDING: Trajectory/prototype components are HURTING performance ***")
        print("    The simple encoder without MHTPN's special components performs better.")
        print("    MHTPN's architecture may be unsuitable for LSST's spectral data.")
    elif avg_improvement < -0.02:
        print("\n*** FINDING: MHTPN provides benefit over simple encoder ***")
    else:
        print("\n*** FINDING: MHTPN and simple encoder perform similarly ***")

    # Embedding separability analysis
    seg_test_accs = [f['segment_embedding_separability']['test_accuracy'] for f in results['folds']]
    simple_test_accs = [f['simple_embedding_separability']['test_accuracy'] for f in results['folds']]

    print(f"\nEmbedding Linear Probe Results:")
    print(f"  MHTPN segment embeddings: {np.mean(seg_test_accs)*100:.1f}%")
    print(f"  Simple encoder embeddings: {np.mean(simple_test_accs)*100:.1f}%")

    if np.mean(seg_test_accs) < np.mean(simple_test_accs):
        print("  >>> MHTPN embeddings are LESS separable than simple encoder")
        print("      This suggests MHTPN's projection/normalization loses discriminatory information")

    results['summary'] = {
        'mhtpn_mean_accuracy': float(np.mean(mhtpn_accs)),
        'mhtpn_std_accuracy': float(np.std(mhtpn_accs)),
        'simple_mean_accuracy': float(np.mean(simple_accs)),
        'simple_std_accuracy': float(np.std(simple_accs)),
        'simple_vs_mhtpn_improvement': float(avg_improvement),
    }

    with open(os.path.join(output_dir, 'full_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"DEEP DIAGNOSTIC COMPLETE")
    print(f"Results saved to: {output_dir}")
    print('=' * 70)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MHTPN Deep Diagnostic Analysis')
    parser.add_argument('--dataset', type=str, default='LSST', help='Dataset name')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--folds', type=int, default=3, help='Number of CV folds')

    args = parser.parse_args()

    run_deep_diagnostics(
        dataset_name=args.dataset,
        output_dir=args.output,
        n_folds=args.folds,
    )
