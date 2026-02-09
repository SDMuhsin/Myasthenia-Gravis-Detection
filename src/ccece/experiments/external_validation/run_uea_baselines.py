"""
Run all baseline models on UEA Time Series Classification datasets.

Compares MHTPN against the 8 SOTA baselines from the paper.
"""

import os
import sys
import json
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from ccece.experiments.sota_comparison.baselines import BASELINE_REGISTRY, get_baseline
from ccece.models.multi_head_trajectory_proto_net import MultiHeadTrajectoryProtoNet

RANDOM_SEED = 42


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_uea_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Load a UEA multivariate time series classification dataset."""
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

    return X_train, y_train_enc, X_test, y_test_enc, list(le.classes_)


def standardize_data(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize using training set statistics (per channel)."""
    n_samples, seq_len, n_channels = X_train.shape
    train_flat = X_train.reshape(-1, n_channels)
    mean = train_flat.mean(axis=0, keepdims=True)
    std = train_flat.std(axis=0, keepdims=True) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    return X_train_norm.astype(np.float32), X_test_norm.astype(np.float32)


def run_baseline_cv(
    model_name: str,
    X_all: np.ndarray,
    y_all: np.ndarray,
    n_folds: int = 5,
    n_classes: int = 2,
    verbose: bool = True,
) -> Dict:
    """Run cross-validation for a single baseline model."""
    set_seed(RANDOM_SEED)

    n_samples, seq_len, n_features = X_all.shape

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_all, y_all)):
        if verbose:
            print(f"    Fold {fold_idx + 1}/{n_folds}...", end=" ", flush=True)

        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        X_val = X_all[val_idx]
        y_val = y_all[val_idx]

        # Standardize per fold
        X_train, X_val = standardize_data(X_train, X_val)

        try:
            # Create and train model
            model = get_baseline(
                model_name,
                input_dim=n_features,
                seq_len=seq_len,
                num_classes=n_classes,
            )

            # Train
            train_info = model.fit(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                epochs=100,
                patience=20,
                batch_size=32,
                verbose=False,
            )

            # Evaluate
            preds = model.predict(X_val)
            accuracy = (preds == y_val).mean()

            fold_results.append({
                'fold': fold_idx + 1,
                'accuracy': float(accuracy),
            })

            if verbose:
                print(f"{accuracy*100:.1f}%")

        except Exception as e:
            if verbose:
                print(f"ERROR: {str(e)[:50]}")
            fold_results.append({
                'fold': fold_idx + 1,
                'accuracy': None,
                'error': str(e),
            })

    # Compute mean accuracy (excluding failed folds)
    valid_accs = [r['accuracy'] for r in fold_results if r['accuracy'] is not None]
    if valid_accs:
        mean_acc = np.mean(valid_accs)
        std_acc = np.std(valid_accs)
    else:
        mean_acc = None
        std_acc = None

    return {
        'model': model_name,
        'fold_results': fold_results,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
    }


def run_mhtpn_cv(
    X_all: np.ndarray,
    y_all: np.ndarray,
    n_folds: int = 5,
    n_classes: int = 2,
    verbose: bool = True,
) -> Dict:
    """Run cross-validation for MHTPN."""
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn
    from tqdm import tqdm

    set_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_samples, seq_len, n_features = X_all.shape

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_all, y_all)):
        if verbose:
            print(f"    Fold {fold_idx + 1}/{n_folds}...", end=" ", flush=True)

        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        X_val = X_all[val_idx]
        y_val = y_all[val_idx]

        # Standardize
        X_train, X_val = standardize_data(X_train, X_val)

        # Create data loaders
        train_dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train).long()
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val).long()
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Create model
        model = MultiHeadTrajectoryProtoNet(
            input_dim=n_features,
            num_classes=n_classes,
            seq_len=seq_len,
            latent_dim=64,
            n_heads=5,
            head_dim=32,
            n_segments=8,
            encoder_hidden=64,
            encoder_layers=3,
            kernel_size=7,
            dropout=0.2,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        patience_counter = 0

        for epoch in range(100):
            # Train
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Validate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    logits = model(X_batch)
                    preds = logits.argmax(dim=1)
                    correct += (preds == y_batch).sum().item()
                    total += y_batch.size(0)

            val_acc = correct / total
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 20:
                break

        fold_results.append({
            'fold': fold_idx + 1,
            'accuracy': float(best_acc),
        })

        if verbose:
            print(f"{best_acc*100:.1f}%")

    valid_accs = [r['accuracy'] for r in fold_results]
    return {
        'model': 'MHTPN',
        'fold_results': fold_results,
        'mean_accuracy': np.mean(valid_accs),
        'std_accuracy': np.std(valid_accs),
    }


def run_simple_baselines(X_all: np.ndarray, y_all: np.ndarray, n_folds: int = 5) -> Dict:
    """Run LogisticRegression and RandomForest baselines."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    results = {'LogisticRegression': [], 'RandomForest': []}

    for train_idx, val_idx in cv.split(X_all, y_all):
        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]

        X_train, X_val = standardize_data(X_train, X_val)

        # Flatten for sklearn
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)

        # LogReg
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        lr.fit(X_train_flat, y_train)
        results['LogisticRegression'].append(lr.score(X_val_flat, y_val))

        # RF
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
        rf.fit(X_train_flat, y_train)
        results['RandomForest'].append(rf.score(X_val_flat, y_val))

    return {
        name: {
            'model': name,
            'mean_accuracy': np.mean(accs),
            'std_accuracy': np.std(accs),
        }
        for name, accs in results.items()
    }


def run_all_models_on_dataset(
    dataset_name: str,
    output_dir: str,
    n_folds: int = 5,
    verbose: bool = True,
) -> Dict:
    """Run all models on a single dataset."""
    set_seed(RANDOM_SEED)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

    # Load data
    X_train, y_train, X_test, y_test, class_names = load_uea_dataset(dataset_name)
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)

    n_samples, seq_len, n_features = X_all.shape
    n_classes = len(class_names)

    if verbose:
        print(f"Samples: {n_samples}, Length: {seq_len}, Features: {n_features}, Classes: {n_classes}")

    results = {
        'dataset': dataset_name,
        'n_samples': n_samples,
        'seq_len': seq_len,
        'n_features': n_features,
        'n_classes': n_classes,
        'class_names': class_names,
        'models': {},
    }

    # Simple baselines
    if verbose:
        print("\nSimple Baselines:")
    simple_results = run_simple_baselines(X_all, y_all, n_folds)
    for name, res in simple_results.items():
        results['models'][name] = res
        if verbose:
            print(f"  {name}: {res['mean_accuracy']*100:.1f} ± {res['std_accuracy']*100:.1f}%")

    # MHTPN
    if verbose:
        print("\nMHT ProtoNet:")
    mhtpn_results = run_mhtpn_cv(X_all, y_all, n_folds, n_classes, verbose)
    results['models']['MHTPN'] = mhtpn_results
    if verbose:
        print(f"  Mean: {mhtpn_results['mean_accuracy']*100:.1f} ± {mhtpn_results['std_accuracy']*100:.1f}%")

    # Deep learning baselines
    baseline_names = ['1D-CNN', 'LSTM', 'InceptionTime', 'TST', 'ConvTran', 'PatchTST']
    # Skip ROCKET and TimesNet if they have issues

    for model_name in baseline_names:
        if verbose:
            print(f"\n{model_name}:")
        try:
            model_results = run_baseline_cv(model_name, X_all, y_all, n_folds, n_classes, verbose)
            results['models'][model_name] = model_results
            if model_results['mean_accuracy'] is not None:
                if verbose:
                    print(f"  Mean: {model_results['mean_accuracy']*100:.1f} ± {model_results['std_accuracy']*100:.1f}%")
        except Exception as e:
            if verbose:
                print(f"  ERROR: {str(e)[:80]}")
            results['models'][model_name] = {'error': str(e)}

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f'{dataset_name}_all_models.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\nResults saved to: {results_path}")

    return results


def print_comparison_table(results: Dict):
    """Print a comparison table of all models."""
    print(f"\n{'='*70}")
    print(f"COMPARISON: {results['dataset']}")
    print(f"{'='*70}")

    # Sort by accuracy
    model_accs = []
    for name, res in results['models'].items():
        if isinstance(res, dict) and res.get('mean_accuracy') is not None:
            model_accs.append((name, res['mean_accuracy'], res.get('std_accuracy', 0)))

    model_accs.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Model':<20} {'Accuracy':<15} {'Rank'}")
    print("-" * 45)
    for rank, (name, acc, std) in enumerate(model_accs, 1):
        print(f"{name:<20} {acc*100:.1f} ± {std*100:.1f}%     {rank}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run all baselines on UEA datasets')
    parser.add_argument('--dataset', type=str, default='Heartbeat',
                       help='UEA dataset name')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--output-dir', type=str,
                       default='/workspace/Myasthenia-Gravis-Detection/results/ccece/external_validation/uea',
                       help='Output directory')

    args = parser.parse_args()

    results = run_all_models_on_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        verbose=True,
    )

    print_comparison_table(results)


if __name__ == '__main__':
    main()
