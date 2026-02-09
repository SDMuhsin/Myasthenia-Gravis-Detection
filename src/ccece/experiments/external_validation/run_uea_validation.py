"""
Run MHTPN on UEA Time Series Classification datasets.

This validates that MHTPN works on general time series classification tasks,
not just the MG dataset.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from dataclasses import dataclass, asdict
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from ccece.models.multi_head_trajectory_proto_net import MultiHeadTrajectoryProtoNet


RANDOM_SEED = 42


@dataclass
class UEAConfig:
    """Configuration for UEA dataset experiments."""
    # Model hyperparameters
    latent_dim: int = 64
    n_heads: int = 5
    head_dim: int = 32
    n_segments: int = None  # None = auto-select based on seq_len
    encoder_hidden: int = 64
    encoder_layers: int = 3
    kernel_size: int = 7
    dropout: float = 0.2

    # Training hyperparameters
    epochs: int = 150
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 30
    use_cosine_annealing: bool = True

    # Cross-validation
    n_folds: int = 5


def compute_optimal_n_segments(seq_len: int) -> int:
    """
    Compute optimal number of segments based on sequence length.

    Heuristic: Target segment size of ~100-200 timesteps.
    This preserves temporal dynamics while providing enough segments
    for trajectory learning.

    Returns:
        n_segments: Number of segments (2, 4, 8, or 16)
    """
    # Target segment size around 100-200 timesteps
    # For very short sequences, use fewer segments
    if seq_len <= 200:
        return 2
    elif seq_len <= 400:
        return 4
    elif seq_len <= 800:
        return 8
    else:
        # For long sequences like SelfRegulationSCP1 (896),
        # use fewer segments to preserve slow dynamics
        # Target ~400 timesteps per segment
        return max(2, min(8, seq_len // 400))


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_uea_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a UEA multivariate time series classification dataset.

    Returns:
        X_train, y_train, X_test, y_test
    """
    from aeon.datasets import load_classification

    X_train, y_train = load_classification(dataset_name, split="train")
    X_test, y_test = load_classification(dataset_name, split="test")

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Convert to float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # Transpose from (n_samples, n_channels, seq_len) to (n_samples, seq_len, n_channels)
    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    return X_train, y_train_enc, X_test, y_test_enc, le.classes_


def standardize_data(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize using training set statistics (per channel)."""
    n_samples, seq_len, n_channels = X_train.shape

    # Reshape to (n_samples * seq_len, n_channels)
    train_flat = X_train.reshape(-1, n_channels)

    # Compute mean and std
    mean = train_flat.mean(axis=0, keepdims=True)
    std = train_flat.std(axis=0, keepdims=True) + 1e-8

    # Standardize
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm.astype(np.float32), X_test_norm.astype(np.float32)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    return total_loss / len(data_loader), correct / total


def run_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Run simple baseline classifiers."""
    # Flatten time series for baselines
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    lr.fit(X_train_flat, y_train)
    results['LogisticRegression'] = lr.score(X_test_flat, y_test)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    rf.fit(X_train_flat, y_train)
    results['RandomForest'] = rf.score(X_test_flat, y_test)

    return results


def run_mhtpn_experiment(
    dataset_name: str,
    config: UEAConfig,
    output_dir: str,
    verbose: bool = True,
) -> Dict:
    """
    Run MHTPN on a UEA dataset.

    Args:
        dataset_name: Name of UEA dataset (e.g., "Heartbeat")
        config: Experiment configuration
        output_dir: Directory to save results
        verbose: Print progress

    Returns:
        Dict with results
    """
    set_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"Device: {device}")
        print(f"{'='*60}\n")

    # Load data
    if verbose:
        print("Loading dataset...")
    X_train, y_train, X_test, y_test, class_names = load_uea_dataset(dataset_name)

    n_samples, seq_len, n_features = X_train.shape
    n_classes = len(class_names)

    if verbose:
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"Classes: {n_classes} - {list(class_names)}")

    # Standardize
    X_train, X_test = standardize_data(X_train, X_test)

    # Run baselines
    if verbose:
        print("\nRunning baselines...")
    baseline_results = run_baselines(X_train, y_train, X_test, y_test)
    if verbose:
        for name, acc in baseline_results.items():
            print(f"  {name}: {acc*100:.1f}%")

    # Cross-validation on combined data
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)

    if verbose:
        print(f"\nRunning {config.n_folds}-fold cross-validation...")

    cv = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_all, y_all)):
        # Use different seed per fold for initialization diversity
        set_seed(RANDOM_SEED + fold_idx)

        if verbose:
            print(f"\n--- Fold {fold_idx + 1}/{config.n_folds} ---")

        X_train_fold = X_all[train_idx]
        y_train_fold = y_all[train_idx]
        X_val_fold = X_all[val_idx]
        y_val_fold = y_all[val_idx]

        # Standardize per fold
        X_train_fold, X_val_fold = standardize_data(X_train_fold, X_val_fold)

        # Create data loaders
        train_dataset = TensorDataset(
            torch.from_numpy(X_train_fold),
            torch.from_numpy(y_train_fold).long()
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val_fold),
            torch.from_numpy(y_val_fold).long()
        )

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        # Compute n_segments adaptively if not specified
        n_segments = config.n_segments
        if n_segments is None:
            n_segments = compute_optimal_n_segments(seq_len)
            if fold_idx == 0 and verbose:
                print(f"  Auto-selected n_segments={n_segments} for seq_len={seq_len}")

        # Create model
        model = MultiHeadTrajectoryProtoNet(
            input_dim=n_features,
            num_classes=n_classes,
            seq_len=seq_len,
            latent_dim=config.latent_dim,
            n_heads=config.n_heads,
            head_dim=config.head_dim,
            n_segments=n_segments,
            encoder_hidden=config.encoder_hidden,
            encoder_layers=config.encoder_layers,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        ).to(device)

        # Setup training
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = None
        if config.use_cosine_annealing:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.epochs
            )
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0

        pbar = tqdm(range(config.epochs), disable=not verbose, desc=f"Fold {fold_idx+1}")
        for epoch in pbar:
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            if scheduler is not None:
                scheduler.step()
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            pbar.set_postfix({'loss': f'{train_loss:.3f}', 'val_acc': f'{val_acc*100:.1f}%'})

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= config.early_stopping_patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        fold_results.append({
            'fold': fold_idx + 1,
            'best_val_accuracy': best_val_acc,
            'best_epoch': best_epoch,
        })

        if verbose:
            print(f"  Best accuracy: {best_val_acc*100:.1f}% (epoch {best_epoch+1})")

    # Aggregate results
    accuracies = [r['best_val_accuracy'] for r in fold_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    results = {
        'dataset': dataset_name,
        'n_samples': n_samples + len(y_test),
        'n_features': n_features,
        'seq_len': seq_len,
        'n_classes': n_classes,
        'class_names': list(class_names),
        'config': asdict(config),
        'baseline_results': baseline_results,
        'fold_results': fold_results,
        'mhtpn_mean_accuracy': mean_acc,
        'mhtpn_std_accuracy': std_acc,
        'timestamp': datetime.now().isoformat(),
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f'{dataset_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULTS: {dataset_name}")
        print(f"{'='*60}")
        print(f"Baselines:")
        for name, acc in baseline_results.items():
            print(f"  {name}: {acc*100:.1f}%")
        print(f"\nMHT Prototype Net:")
        print(f"  Mean accuracy: {mean_acc*100:.1f} ± {std_acc*100:.1f}%")
        print(f"\nResults saved to: {results_path}")

    return results


def main():
    """Run MHTPN on multiple UEA datasets."""
    import argparse

    parser = argparse.ArgumentParser(description='Run MHTPN on UEA datasets')
    parser.add_argument('--dataset', type=str, default='Heartbeat',
                       help='UEA dataset name')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--output-dir', type=str,
                       default='/workspace/Myasthenia-Gravis-Detection/results/ccece/external_validation/uea',
                       help='Output directory')

    args = parser.parse_args()

    config = UEAConfig(
        epochs=args.epochs,
        n_folds=args.n_folds,
    )

    results = run_mhtpn_experiment(
        dataset_name=args.dataset,
        config=config,
        output_dir=args.output_dir,
        verbose=True,
    )

    return results


if __name__ == '__main__':
    main()
