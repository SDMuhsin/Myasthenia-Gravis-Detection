"""
MUSE Baseline Benchmark on LSST Dataset

This script reproduces the MUSE (WEASEL+MUSE) baseline from:
"The Great Multivariate Time Series Classification Bake Off" (Ruiz et al., 2020)

Published benchmark result: MUSE achieves 63.62% accuracy on LSST

Success Criteria (LOCKED):
- MUSE achieves accuracy within ±3% of published 63.62% (i.e., 60-67%)
- Results reproducible with random_state=42

Usage:
    source env/bin/activate
    export PYTHONPATH=/workspace/Myasthenia-Gravis-Detection/src
    python3 -m ccece.experiments.sota_comparison.muse_benchmark
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Constants
RANDOM_SEED = 42
N_FOLDS = 5
OUTPUT_DIR = 'results/ccece/sota_comparison/muse_baseline'


def load_lsst_dataset(verbose: bool = True):
    """
    Load LSST dataset from aeon.

    Returns:
        X: Data array, shape (n_samples, n_channels, seq_len) - sktime format
        y: Labels, shape (n_samples,), encoded as integers
        class_names: List of original class names
    """
    from aeon.datasets import load_classification

    if verbose:
        print("Loading LSST dataset...")

    # Load train and test splits
    X_train, y_train = load_classification('LSST', split='train')
    X_test, y_test = load_classification('LSST', split='test')

    # Combine train and test (we do our own CV)
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    # Encode labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = list(le.classes_)

    # X is already in (n_samples, n_channels, seq_len) format from aeon
    # which is what sktime expects
    X = X.astype(np.float64)  # MUSE works better with float64

    if verbose:
        print(f"  Shape: {X.shape} (samples, channels, seq_len)")
        print(f"  Classes: {len(class_names)} - {class_names}")
        print(f"  Total samples: {len(y)}")
        print(f"  Class distribution:")
        for cls, name in enumerate(class_names):
            count = np.sum(y_encoded == cls)
            print(f"    Class {name}: {count} ({100*count/len(y_encoded):.1f}%)")

    return X, y_encoded, class_names


def run_muse_benchmark(
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_SEED,
    output_dir: str = OUTPUT_DIR,
    verbose: bool = True,
):
    """
    Run MUSE benchmark on LSST with k-fold cross-validation.

    Args:
        n_folds: Number of CV folds
        random_state: Random seed for reproducibility
        output_dir: Directory to save results
        verbose: Print progress

    Returns:
        Dict with benchmark results
    """
    from sktime.classification.dictionary_based import MUSE

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("MUSE BASELINE BENCHMARK ON LSST")
    print("=" * 70)
    print(f"Target: 63.62% ± 3% (from published benchmark)")
    print(f"Folds: {n_folds}")
    print(f"Random seed: {random_state}")
    print(f"Output: {output_dir}")
    print()

    # Load data
    X, y, class_names = load_lsst_dataset(verbose=verbose)

    # Setup cross-validation
    cv = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )

    # Results storage
    fold_results = []
    all_predictions = []
    all_labels = []

    print()
    print("=" * 70)
    print("RUNNING CROSS-VALIDATION")
    print("=" * 70)

    total_start_time = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"  Train: {len(X_train)} samples")
        print(f"  Val: {len(X_val)} samples")

        # Create MUSE classifier with default parameters
        # These match the original paper settings
        muse = MUSE(
            anova=True,
            variance=False,
            bigrams=True,
            window_inc=2,
            alphabet_size=4,
            use_first_order_differences=True,
            feature_selection='chi2',
            p_threshold=0.05,
            n_jobs=-1,  # Use all cores
            random_state=random_state + fold_idx,  # Different seed per fold
        )

        # Train
        print(f"  Training MUSE...", end=' ', flush=True)
        fold_start = time.time()
        muse.fit(X_train, y_train)
        train_time = time.time() - fold_start
        print(f"done ({train_time:.1f}s)")

        # Predict
        print(f"  Predicting...", end=' ', flush=True)
        pred_start = time.time()
        y_pred = muse.predict(X_val)
        pred_time = time.time() - pred_start
        print(f"done ({pred_time:.1f}s)")

        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)

        fold_result = {
            'fold': fold_idx + 1,
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'train_time': float(train_time),
            'pred_time': float(pred_time),
            'n_train': len(X_train),
            'n_val': len(X_val),
        }
        fold_results.append(fold_result)

        all_predictions.extend(y_pred.tolist())
        all_labels.extend(y_val.tolist())

        print(f"  Accuracy: {accuracy * 100:.2f}%")
        print(f"  F1-Score: {f1 * 100:.2f}%")

        # Save intermediate results (crash protection)
        intermediate_path = os.path.join(output_dir, f'fold{fold_idx + 1}_results.json')
        with open(intermediate_path, 'w') as f:
            json.dump(fold_result, f, indent=2)
        print(f"  Saved: {intermediate_path}")

    total_time = time.time() - total_start_time

    # Aggregate results
    print()
    print("=" * 70)
    print("AGGREGATING RESULTS")
    print("=" * 70)

    accuracies = [r['accuracy'] for r in fold_results]
    f1_scores = [r['f1_score'] for r in fold_results]

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    # 95% confidence interval
    ci_lower = mean_accuracy - 1.96 * std_accuracy / np.sqrt(n_folds)
    ci_upper = mean_accuracy + 1.96 * std_accuracy / np.sqrt(n_folds)

    # Check if within target range (60-67%)
    target_min = 0.60
    target_max = 0.67
    target_published = 0.6362

    is_within_target = target_min <= mean_accuracy <= target_max
    deviation_from_published = (mean_accuracy - target_published) * 100

    results = {
        'experiment': {
            'name': 'MUSE Baseline on LSST',
            'model': 'MUSE (WEASEL+MUSE)',
            'dataset': 'LSST',
            'n_folds': n_folds,
            'random_state': random_state,
            'timestamp': datetime.now().isoformat(),
            'total_time_seconds': float(total_time),
        },
        'dataset': {
            'n_samples': len(y),
            'n_features': X.shape[1],
            'seq_len': X.shape[2],
            'n_classes': len(class_names),
            'class_names': class_names,
        },
        'per_fold': fold_results,
        'aggregated': {
            'accuracy': {
                'mean': float(mean_accuracy),
                'std': float(std_accuracy),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'values': accuracies,
            },
            'f1_score': {
                'mean': float(mean_f1),
                'std': float(std_f1),
                'values': f1_scores,
            },
        },
        'validation': {
            'target_published': target_published,
            'target_range': [target_min, target_max],
            'is_within_target': bool(is_within_target),
            'deviation_from_published_pct': float(deviation_from_published),
        },
    }

    # Save full results
    results_path = os.path.join(output_dir, 'full_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"MUSE Accuracy: {mean_accuracy * 100:.2f}% ± {std_accuracy * 100:.2f}%")
    print(f"95% CI: [{ci_lower * 100:.2f}%, {ci_upper * 100:.2f}%]")
    print(f"F1-Score: {mean_f1 * 100:.2f}% ± {std_f1 * 100:.2f}%")
    print()
    print(f"Per-fold accuracies: {[f'{a*100:.1f}%' for a in accuracies]}")
    print()
    print(f"Published benchmark: {target_published * 100:.2f}%")
    print(f"Our result: {mean_accuracy * 100:.2f}%")
    print(f"Deviation: {deviation_from_published:+.2f}%")
    print()

    if is_within_target:
        print("✓ SUCCESS: Result is within target range (60-67%)")
        print("  This confirms our benchmark setup is correct.")
        print("  Deep learning methods (MHTPN 41.2%, ConvTran 43.9%) genuinely")
        print("  underperform on LSST compared to feature-based methods like MUSE.")
    else:
        print("✗ FAILURE: Result is outside target range (60-67%)")
        if mean_accuracy < target_min:
            print(f"  Result ({mean_accuracy*100:.1f}%) is below expected ({target_min*100:.0f}%)")
        else:
            print(f"  Result ({mean_accuracy*100:.1f}%) is above expected ({target_max*100:.0f}%)")

    print()
    print(f"Total time: {total_time:.1f}s")
    print(f"Results saved to: {results_path}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MUSE Baseline Benchmark on LSST')
    parser.add_argument('--folds', type=int, default=N_FOLDS, help='Number of CV folds')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR, help='Output directory')

    args = parser.parse_args()

    run_muse_benchmark(
        n_folds=args.folds,
        random_state=args.seed,
        output_dir=args.output,
    )
