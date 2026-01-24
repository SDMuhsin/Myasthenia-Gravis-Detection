#!/usr/bin/env python3
"""
CCECE Paper: Experiment Runner

Main script for running time series classification experiments.
Supports multiple models and produces reproducible results.

Usage:
    python src/ccece/run_experiment.py --model bigru_attention
    python src/ccece/run_experiment.py --model bigru_attention --epochs 50 --hidden_dim 128
"""

import os
import sys
import json
import argparse
import random
import fcntl
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ccece.data_loader import load_binary_dataset, extract_arrays, get_sequence_stats
from ccece.models import get_model, list_models
from ccece.trainer import (
    Trainer, TrainingConfig, EvaluationMetrics,
    create_data_loaders, SequenceScaler
)
from ccece.dp_trainer import DPTrainer, DPConfig, OPACUS_AVAILABLE


# =============================================================================
# CONSTANTS
# =============================================================================

RESULTS_DIR = './results/ccece'
RANDOM_SEED = 42
SUBSAMPLE_FACTOR = 10  # Subsample 120Hz to 12Hz
DEFAULT_SEQ_LEN_PERCENTILE = 90
RESULTS_CSV_NAME = 'all_results.csv'


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_all_seeds(seed: int = RANDOM_SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# THREAD-SAFE CSV WRITING
# =============================================================================

def append_to_csv_thread_safe(csv_path: str, row_dict: Dict[str, Any], fieldnames: List[str]):
    """
    Append a row to a CSV file in a thread-safe manner using file locking.

    Args:
        csv_path: Path to the CSV file
        row_dict: Dictionary containing the row data
        fieldnames: List of column names (in order)
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists to determine if we need to write header
    file_exists = csv_path.exists()

    with open(csv_path, 'a', newline='') as f:
        # Acquire exclusive lock
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # Write header if file is new/empty
            if not file_exists or csv_path.stat().st_size == 0:
                writer.writeheader()

            writer.writerow(row_dict)
        finally:
            # Release lock
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# CSV column order for results
RESULTS_CSV_COLUMNS = [
    'timestamp',
    'experiment_name',
    'model_name',
    'fold',
    'accuracy',
    'sensitivity',
    'specificity',
    'precision',
    'f1',
    'auc_roc',
    'epochs_trained',
    'best_epoch',
    'hidden_dim',
    'num_layers',
    'dropout',
    'fc_dim',
    'learning_rate',
    'batch_size',
    'seq_len',
    'input_dim',
    'num_params',
    'train_samples',
    'val_samples',
    'dp_enabled',
    'dp_epsilon',
    'dp_delta',
]


# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def subsample_sequence(data: np.ndarray, factor: int = SUBSAMPLE_FACTOR) -> np.ndarray:
    """Subsample a sequence by taking every nth sample."""
    return data[::factor]


def add_engineered_features(data: np.ndarray) -> np.ndarray:
    """
    Add velocity and error features to raw 6-channel data.

    Input: (seq_len, 6) - [LH, RH, LV, RV, TargetH, TargetV]
    Output: (seq_len, 14) - original + 4 velocities + 4 errors
    """
    # Velocities (numerical derivative)
    lh_vel = np.gradient(data[:, 0])
    rh_vel = np.gradient(data[:, 1])
    lv_vel = np.gradient(data[:, 2])
    rv_vel = np.gradient(data[:, 3])

    # Tracking errors (eye position - target position)
    error_h_l = data[:, 0] - data[:, 4]
    error_h_r = data[:, 1] - data[:, 4]
    error_v_l = data[:, 2] - data[:, 5]
    error_v_r = data[:, 3] - data[:, 5]

    return np.column_stack([
        data,
        lh_vel, rh_vel, lv_vel, rv_vel,
        error_h_l, error_h_r, error_v_l, error_v_r
    ]).astype(np.float32)


def preprocess_items(
    items: List[Dict],
    subsample_factor: int = SUBSAMPLE_FACTOR,
    add_features: bool = True,
) -> List[Dict]:
    """
    Preprocess all data items.

    Args:
        items: Raw data items from data_loader
        subsample_factor: Factor to subsample sequences
        add_features: Whether to add engineered features

    Returns:
        Preprocessed items
    """
    processed = []
    for item in items:
        new_item = item.copy()
        data = item['data']

        # Subsample
        data = subsample_sequence(data, subsample_factor)

        # Add engineered features
        if add_features:
            data = add_engineered_features(data)

        new_item['data'] = data
        processed.append(new_item)

    return processed


def compute_target_seq_len(items: List[Dict], percentile: float = DEFAULT_SEQ_LEN_PERCENTILE) -> int:
    """Compute target sequence length based on percentile of actual lengths."""
    lengths = [item['data'].shape[0] for item in items]
    return int(np.percentile(lengths, percentile))


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class ExperimentRunner:
    """
    Main experiment runner for time series classification.

    Handles:
        - Cross-validation with patient grouping
        - Model training and evaluation
        - Results logging and saving (including thread-safe CSV)
    """

    def __init__(
        self,
        model_name: str,
        model_kwargs: Dict[str, Any],
        training_config: TrainingConfig,
        n_folds: int = 1,  # Default to 1 fold for faster iteration
        results_dir: str = RESULTS_DIR,
        experiment_name: Optional[str] = None,
        results_subdir: str = 'experiments',  # Subfolder within results_dir
        dp_config: Optional[DPConfig] = None,  # DP configuration (None = no DP)
    ):
        """
        Args:
            model_name: Name of the model (from MODEL_REGISTRY)
            model_kwargs: Additional kwargs for model initialization
            training_config: Training configuration
            n_folds: Number of cross-validation folds (default: 1)
            results_dir: Base directory for results
            experiment_name: Optional experiment name (auto-generated if None)
            results_subdir: Subfolder name within results_dir
            dp_config: Differential privacy configuration (None = no DP)
        """
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.training_config = training_config
        self.n_folds = n_folds
        self.results_dir = results_dir
        self.results_subdir = results_subdir
        self.dp_config = dp_config

        # Generate experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{model_name}_{timestamp}"
        self.experiment_name = experiment_name

        # Create results directory structure: results_dir/subdir/experiment_name/
        self.subdir_path = os.path.join(results_dir, results_subdir)
        self.experiment_dir = os.path.join(self.subdir_path, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        # CSV path for aggregated results (in subdir, shared across experiments)
        self.csv_path = os.path.join(self.subdir_path, RESULTS_CSV_NAME)

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Results storage
        self.fold_metrics: List[EvaluationMetrics] = []
        self.fold_histories = []

        # Track metadata for CSV
        self._input_dim = None
        self._seq_len = None
        self._num_params = None

    def run(self, items: List[Dict], verbose: bool = True) -> Dict[str, Any]:
        """
        Run the full experiment with cross-validation.

        Args:
            items: Preprocessed data items
            verbose: Whether to print progress

        Returns:
            Dictionary with aggregated results
        """
        print(f"\n{'='*60}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Folds: {self.n_folds}")
        if self.dp_config is not None:
            print(f"DP: ENABLED (ε={self.dp_config.target_epsilon}, δ={self.dp_config.target_delta})")
        else:
            print(f"DP: Disabled")
        print(f"{'='*60}\n")

        # Extract arrays for CV splitting
        X, y, patient_ids = extract_arrays(items)
        patient_ids_array = np.array(patient_ids)

        # Compute target sequence length
        self._seq_len = compute_target_seq_len(items)
        self._input_dim = items[0]['data'].shape[1]

        print(f"Data: {len(items)} sequences, {self._input_dim} features, target_len={self._seq_len}")
        print(f"Classes: HC={np.sum(y==0)}, MG={np.sum(y==1)}")
        print(f"Unique patients: {len(set(patient_ids))}")
        print()

        # Cross-validation setup
        # For n_folds=1, use GroupShuffleSplit for a single train/val split
        # For n_folds>1, use StratifiedGroupKFold for proper k-fold CV
        if self.n_folds == 1:
            cv = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
            split_iterator = cv.split(X, y, groups=patient_ids_array)
        else:
            cv = StratifiedGroupKFold(n_splits=self.n_folds, shuffle=True, random_state=RANDOM_SEED)
            split_iterator = cv.split(X, y, groups=patient_ids_array)

        for fold_idx, (train_idx, val_idx) in enumerate(split_iterator):
            print(f"\n--- Fold {fold_idx + 1}/{self.n_folds} ---")

            # Split data
            train_items = [items[i] for i in train_idx]
            val_items = [items[i] for i in val_idx]
            train_labels = y[train_idx]

            print(f"Train: {len(train_items)} (HC={np.sum(train_labels==0)}, MG={np.sum(train_labels==1)})")
            print(f"Val: {len(val_items)}")

            # Create data loaders
            train_loader, val_loader, scaler = create_data_loaders(
                train_items, val_items, self._seq_len,
                self.training_config.batch_size
            )

            # Create model
            model = get_model(
                self.model_name,
                input_dim=self._input_dim,
                num_classes=2,
                seq_len=self._seq_len,
                **self.model_kwargs
            )

            if fold_idx == 0:
                self._num_params = model.count_parameters()
                print(f"\n{model.get_model_summary()}\n")

            # Train (with or without DP)
            if self.dp_config is not None:
                trainer = DPTrainer(model, self.training_config, self.device, self.dp_config)
            else:
                trainer = Trainer(model, self.training_config, self.device)
            history = trainer.train(
                train_loader, val_loader, train_labels,
                verbose=verbose
            )
            self.fold_histories.append(history)

            # Track DP epsilon for this fold
            if self.dp_config is not None and hasattr(history, 'final_epsilon'):
                self._last_fold_epsilon = history.final_epsilon
                self._last_fold_delta = history.final_delta
            else:
                self._last_fold_epsilon = None
                self._last_fold_delta = None

            # Evaluate
            metrics = trainer.evaluate(val_loader)
            self.fold_metrics.append(metrics)

            print(f"Fold {fold_idx + 1} Results: {metrics}")

            # Save fold results to CSV (thread-safe)
            self._save_fold_to_csv(
                fold_idx=fold_idx,
                metrics=metrics,
                history=history,
                train_samples=len(train_items),
                val_samples=len(val_items),
                dp_epsilon=self._last_fold_epsilon,
                dp_delta=self._last_fold_delta,
            )

            # Clean up GPU memory
            del model, trainer
            torch.cuda.empty_cache()

        # Aggregate results
        results = self._aggregate_results()
        self._save_results(results)
        self._print_summary(results)

        return results

    def _save_fold_to_csv(
        self,
        fold_idx: int,
        metrics: EvaluationMetrics,
        history: Any,
        train_samples: int,
        val_samples: int,
        dp_epsilon: Optional[float] = None,
        dp_delta: Optional[float] = None,
    ):
        """Save a single fold's results to the shared CSV file (thread-safe)."""
        row = {
            'timestamp': datetime.now().isoformat(),
            'experiment_name': self.experiment_name,
            'model_name': self.model_name,
            'fold': fold_idx + 1,
            'accuracy': round(metrics.accuracy, 6),
            'sensitivity': round(metrics.sensitivity, 6),
            'specificity': round(metrics.specificity, 6),
            'precision': round(metrics.precision, 6),
            'f1': round(metrics.f1, 6),
            'auc_roc': round(metrics.auc_roc, 6),
            'epochs_trained': len(history.train_losses),
            'best_epoch': history.best_epoch + 1,
            'hidden_dim': self.model_kwargs.get('hidden_dim', ''),
            'num_layers': self.model_kwargs.get('num_layers', ''),
            'dropout': self.model_kwargs.get('dropout', ''),
            'fc_dim': self.model_kwargs.get('fc_dim', ''),
            'learning_rate': self.training_config.learning_rate,
            'batch_size': self.training_config.batch_size,
            'seq_len': self._seq_len,
            'input_dim': self._input_dim,
            'num_params': self._num_params,
            'train_samples': train_samples,
            'val_samples': val_samples,
            'dp_enabled': dp_epsilon is not None,
            'dp_epsilon': round(dp_epsilon, 4) if dp_epsilon is not None else '',
            'dp_delta': dp_delta if dp_delta is not None else '',
        }

        append_to_csv_thread_safe(self.csv_path, row, RESULTS_CSV_COLUMNS)
        print(f"Results saved to CSV: {self.csv_path}")

    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results across folds."""
        metric_names = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'auc_roc']

        results = {
            'experiment_name': self.experiment_name,
            'model_name': self.model_name,
            'n_folds': self.n_folds,
            'fold_results': [],
            'aggregated': {},
        }

        # Collect fold results
        for fold_idx, metrics in enumerate(self.fold_metrics):
            fold_dict = metrics.to_dict()
            fold_dict['fold'] = fold_idx + 1
            results['fold_results'].append(fold_dict)

        # Aggregate statistics
        for metric in metric_names:
            values = [m.to_dict()[metric] for m in self.fold_metrics]
            results['aggregated'][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }

        return results

    def _save_results(self, results: Dict[str, Any]):
        """Save results to files."""
        # Add configuration to results
        results['config'] = {
            'model_kwargs': self.model_kwargs,
            'training_config': {
                'epochs': self.training_config.epochs,
                'batch_size': self.training_config.batch_size,
                'learning_rate': self.training_config.learning_rate,
                'weight_decay': self.training_config.weight_decay,
                'early_stopping_patience': self.training_config.early_stopping_patience,
            },
            'input_dim': self._input_dim,
            'seq_len': self._seq_len,
            'num_params': self._num_params,
            'random_seed': RANDOM_SEED,
            'subsample_factor': SUBSAMPLE_FACTOR,
        }

        # Save JSON
        json_path = os.path.join(self.experiment_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Save human-readable summary
        summary_path = os.path.join(self.experiment_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"CSV: {self.csv_path}\n")
            f.write(f"\n{'='*50}\n")
            f.write("AGGREGATED RESULTS (mean ± std)\n")
            f.write(f"{'='*50}\n\n")

            for metric, stats in results['aggregated'].items():
                f.write(f"{metric:12s}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")

            f.write(f"\n{'='*50}\n")
            f.write("PER-FOLD RESULTS\n")
            f.write(f"{'='*50}\n\n")

            for fold_result in results['fold_results']:
                f.write(f"Fold {fold_result['fold']}: ")
                f.write(f"Acc={fold_result['accuracy']:.4f}, ")
                f.write(f"Sens={fold_result['sensitivity']:.4f}, ")
                f.write(f"Spec={fold_result['specificity']:.4f}, ")
                f.write(f"AUC={fold_result['auc_roc']:.4f}\n")

        print(f"\nResults saved to: {self.experiment_dir}")

    def _print_summary(self, results: Dict[str, Any]):
        """Print summary to console."""
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")

        agg = results['aggregated']
        print(f"\nAccuracy:    {agg['accuracy']['mean']:.4f} ± {agg['accuracy']['std']:.4f}")
        print(f"Sensitivity: {agg['sensitivity']['mean']:.4f} ± {agg['sensitivity']['std']:.4f}")
        print(f"Specificity: {agg['specificity']['mean']:.4f} ± {agg['specificity']['std']:.4f}")
        print(f"AUC-ROC:     {agg['auc_roc']['mean']:.4f} ± {agg['auc_roc']['std']:.4f}")

        print(f"\n{'='*60}\n")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CCECE Paper: Time Series Classification Experiment Runner"
    )

    # Model selection
    parser.add_argument(
        '--model', type=str, default='bigru_attention',
        choices=list_models(),
        help=f"Model to use. Available: {list_models()}"
    )

    # Model hyperparameters (generic)
    parser.add_argument('--hidden_dim', type=int, default=64, help="Hidden dimension")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of RNN layers")
    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout probability")
    parser.add_argument('--fc_dim', type=int, default=64, help="FC layer dimension")

    # Model-specific hyperparameters (InceptionTime)
    parser.add_argument('--num_filters', type=int, default=None,
                        help="InceptionTime: number of filters (defaults to hidden_dim if not set)")
    parser.add_argument('--depth', type=int, default=None,
                        help="InceptionTime: depth/number of inception modules (defaults to num_layers if not set)")
    parser.add_argument('--bottleneck_channels', type=int, default=32,
                        help="InceptionTime: bottleneck channels (default: 32)")

    # Model-specific hyperparameters (ResNet1D)
    parser.add_argument('--num_blocks', type=int, default=None,
                        help="ResNet1D: number of residual block groups (defaults to num_layers if not set)")

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help="Maximum epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--patience', type=int, default=15, help="Early stopping patience")

    # Experiment settings
    parser.add_argument('--n_folds', type=int, default=1, help="Number of CV folds (default: 1)")
    parser.add_argument('--name', type=str, default=None, help="Experiment name")
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument('--subdir', type=str, default='experiments', help="Results subdirectory")

    # Flags
    parser.add_argument('--no_features', action='store_true',
                        help="Don't add engineered features (use raw 6 channels only)")
    parser.add_argument('--quiet', action='store_true', help="Reduce output verbosity")

    # Differential Privacy arguments
    parser.add_argument('--dp', action='store_true',
                        help="Enable differential privacy training (DP-SGD)")
    parser.add_argument('--dp_epsilon', type=float, default=8.0,
                        help="Target privacy budget epsilon (default: 8.0)")
    parser.add_argument('--dp_delta', type=float, default=1e-5,
                        help="Target privacy failure probability delta (default: 1e-5)")
    parser.add_argument('--dp_max_grad_norm', type=float, default=1.0,
                        help="Max gradient norm for DP clipping (default: 1.0)")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Set seeds
    set_all_seeds(args.seed)

    print(f"\n{'='*60}")
    print("CCECE Paper: Time Series Classification")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    items = load_binary_dataset()

    # Preprocess
    print("Preprocessing...")
    items = preprocess_items(
        items,
        subsample_factor=SUBSAMPLE_FACTOR,
        add_features=not args.no_features
    )

    # Get sequence stats
    get_sequence_stats([item['data'] for item in items])

    # Model kwargs (generic)
    model_kwargs = {
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'fc_dim': args.fc_dim,
    }

    # Add model-specific kwargs (use generic values as defaults if not specified)
    # InceptionTime parameters
    model_kwargs['num_filters'] = args.num_filters if args.num_filters is not None else args.hidden_dim
    model_kwargs['depth'] = args.depth if args.depth is not None else args.num_layers
    model_kwargs['bottleneck_channels'] = args.bottleneck_channels

    # ResNet1D parameters
    model_kwargs['num_blocks'] = args.num_blocks if args.num_blocks is not None else args.num_layers

    # Training config
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        early_stopping_patience=args.patience,
    )

    # DP config (if enabled)
    dp_config = None
    if args.dp:
        if not OPACUS_AVAILABLE:
            print("ERROR: Differential privacy requested but opacus is not installed.")
            print("Install with: pip install opacus")
            sys.exit(1)
        dp_config = DPConfig(
            target_epsilon=args.dp_epsilon,
            target_delta=args.dp_delta,
            max_grad_norm=args.dp_max_grad_norm,
        )

    # Run experiment
    runner = ExperimentRunner(
        model_name=args.model,
        model_kwargs=model_kwargs,
        training_config=training_config,
        n_folds=args.n_folds,
        experiment_name=args.name,
        results_subdir=args.subdir,
        dp_config=dp_config,
    )

    results = runner.run(items, verbose=not args.quiet)

    return results


if __name__ == '__main__':
    main()
