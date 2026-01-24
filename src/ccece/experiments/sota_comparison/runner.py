"""
Main Runner for SOTA Comparison Experiment

Runs all 9 models with 5-fold cross-validation and generates:
- Classification performance tables
- Computational efficiency tables
- Statistical significance tests
- Figures (accuracy vs params, accuracy vs FLOPs)
"""

import os
import sys
import json
import time
import random
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

# Add parent to path for both module and script execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ccece.data_loader import load_binary_dataset, extract_arrays
from ccece.run_experiment import preprocess_items, set_all_seeds, compute_target_seq_len
from ccece.trainer import SequenceScaler

# Handle imports for both module and script execution
try:
    from .baselines import (
        SimpleCNN, SimpleLSTM, InceptionTimeWrapper, ROCKETWrapper,
        TimeSeriesTransformer, TimesNetWrapper, ConvTranWrapper, PatchTSTWrapper,
    )
    from .metrics import compute_classification_metrics, aggregate_metrics, ClassificationMetrics
    from .timing import compute_computational_metrics, ComputationalMetrics
    from .statistical_tests import run_all_comparisons, PairwiseComparison
    from .table_generator import generate_all_tables
except ImportError:
    from ccece.experiments.sota_comparison.baselines import (
        SimpleCNN, SimpleLSTM, InceptionTimeWrapper, ROCKETWrapper,
        TimeSeriesTransformer, TimesNetWrapper, ConvTranWrapper, PatchTSTWrapper,
    )
    from ccece.experiments.sota_comparison.metrics import compute_classification_metrics, aggregate_metrics, ClassificationMetrics
    from ccece.experiments.sota_comparison.timing import compute_computational_metrics, ComputationalMetrics
    from ccece.experiments.sota_comparison.statistical_tests import run_all_comparisons, PairwiseComparison
    from ccece.experiments.sota_comparison.table_generator import generate_all_tables


RANDOM_SEED = 42
N_FOLDS = 5

# Model configurations
MODEL_CONFIGS = {
    '1D-CNN': {
        'class': SimpleCNN,
        'is_pytorch': True,
        'params': {
            'hidden_channels': (64, 128, 256),
            'kernel_sizes': (7, 5, 3),
            'dropout': 0.3,
        },
        'train_params': {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'patience': 20,
        },
    },
    'LSTM': {
        'class': SimpleLSTM,
        'is_pytorch': True,
        'params': {
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'bidirectional': True,
        },
        'train_params': {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'patience': 20,
        },
    },
    'InceptionTime': {
        'class': InceptionTimeWrapper,
        'is_pytorch': True,
        'params': {
            'num_filters': 32,
            'depth': 6,
            'kernel_sizes': [10, 20, 40],
            'bottleneck_channels': 32,
            'dropout': 0.2,
        },
        'train_params': {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'patience': 20,
        },
    },
    'ROCKET': {
        'class': ROCKETWrapper,
        'is_pytorch': False,
        'params': {
            'num_kernels': 2000,  # Reduced from 10000 for computational efficiency
            'random_state': RANDOM_SEED,
        },
        'train_params': {},  # ROCKET doesn't need training params
    },
    'TST': {
        'class': TimeSeriesTransformer,
        'is_pytorch': True,
        'params': {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 3,
            'd_ff': 256,
            'dropout': 0.1,
        },
        'train_params': {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 1e-4,  # Lower LR for transformer
            'patience': 20,
        },
    },
    'TimesNet': {
        'class': TimesNetWrapper,
        'is_pytorch': True,
        'params': {
            'd_model': 64,
            'd_ff': 64,
            'n_layers': 2,
            'top_k': 5,
            'num_kernels': 6,
            'dropout': 0.1,
        },
        'train_params': {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'patience': 20,
        },
    },
    'ConvTran': {
        'class': ConvTranWrapper,
        'is_pytorch': True,
        'params': {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 3,
            'd_ff': 256,
            'conv_kernel_size': 3,
            'dropout': 0.1,
        },
        'train_params': {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'patience': 20,
        },
    },
    'PatchTST': {
        'class': PatchTSTWrapper,
        'is_pytorch': True,
        'params': {
            'patch_len': 16,
            'stride': 8,
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 3,
            'd_ff': 256,
            'dropout': 0.1,
        },
        'train_params': {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'patience': 20,
        },
    },
}


def prepare_data_for_fold(
    items: List[Dict],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare train and validation data for a fold.

    Returns:
        X_train, y_train, X_val, y_val as numpy arrays
    """
    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]

    # Fit scaler on training data
    scaler = SequenceScaler().fit(train_items)

    # Process data
    def process_items(items_list, scaler, seq_len):
        X_list = []
        y_list = []
        for item in items_list:
            data = scaler.transform(item['data'].copy())
            # Pad or truncate
            if len(data) >= seq_len:
                data = data[:seq_len]
            else:
                padding = np.zeros((seq_len - len(data), data.shape[1]), dtype=np.float32)
                data = np.vstack([data, padding])
            X_list.append(data)
            y_list.append(item['label'])
        return np.array(X_list), np.array(y_list)

    X_train, y_train = process_items(train_items, scaler, seq_len)
    X_val, y_val = process_items(val_items, scaler, seq_len)

    return X_train, y_train, X_val, y_val


def run_single_model_fold(
    model_name: str,
    config: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    verbose: bool = True,
) -> Tuple[ClassificationMetrics, ComputationalMetrics]:
    """
    Train and evaluate a single model on a single fold.
    """
    input_dim = X_train.shape[2]
    seq_len = X_train.shape[1]

    # Create model
    model_class = config['class']
    model_params = config['params'].copy()

    model = model_class(
        input_dim=input_dim,
        num_classes=2,
        seq_len=seq_len,
        device=device,
        **model_params
    )

    # Train
    train_params = config['train_params'].copy()
    train_params['verbose'] = verbose

    train_info = model.fit(
        X_train, y_train,
        X_val, y_val,
        **train_params
    )

    training_time = train_info.get('training_time', 0.0)

    # Evaluate
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)

    metrics = compute_classification_metrics(y_val, y_pred, y_proba)

    # Compute computational metrics
    is_pytorch = config['is_pytorch']
    if is_pytorch:
        model_for_timing = model
    else:
        model_for_timing = model

    comp_metrics = compute_computational_metrics(
        model_for_timing,
        input_shape=(seq_len, input_dim),
        device=device,
        training_time=training_time,
        is_pytorch=is_pytorch,
        X_sample=X_val[:min(100, len(X_val))],
    )

    return metrics, comp_metrics


def run_multihead_protonet_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    verbose: bool = True,
) -> Tuple[ClassificationMetrics, ComputationalMetrics]:
    """
    Train and evaluate MultiHeadProtoNet on a single fold.
    """
    from ccece.models.multi_head_proto_net import MultiHeadProtoNet
    from ccece.experiments.multi_head_proto_experiment import MultiHeadConfig, MultiHeadTrainer

    input_dim = X_train.shape[2]
    seq_len = X_train.shape[1]

    # Create model with successful configuration
    config = MultiHeadConfig(
        latent_dim=64,
        n_heads=5,
        head_dim=64,
        encoder_hidden=64,
        encoder_layers=3,
        kernel_size=7,
        dropout=0.2,
        epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        early_stopping_patience=20,
        cluster_loss_weight=0.3,
        separation_loss_weight=0.1,
        per_head_ce_weight=0.0,
    )

    model = MultiHeadProtoNet(
        input_dim=input_dim,
        num_classes=2,
        seq_len=seq_len,
        latent_dim=config.latent_dim,
        n_heads=config.n_heads,
        head_dim=config.head_dim,
        encoder_hidden=config.encoder_hidden,
        encoder_layers=config.encoder_layers,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
    )

    # Create data loaders
    from torch.utils.data import DataLoader, TensorDataset
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Train
    start_time = time.time()
    trainer = MultiHeadTrainer(model, config, device)
    trainer.train(train_loader, val_loader, y_train, verbose=verbose)
    training_time = time.time() - start_time

    # Evaluate
    model.eval()
    model.to(device)

    with torch.no_grad():
        X_val_t = X_val_t.to(device)
        outputs = model(X_val_t)
        y_proba = torch.softmax(outputs, dim=1).cpu().numpy()
        y_pred = outputs.argmax(dim=1).cpu().numpy()

    metrics = compute_classification_metrics(y_val, y_pred, y_proba)

    # Computational metrics
    try:
        from .timing import count_parameters, measure_flops, measure_inference_time, measure_gpu_memory
    except ImportError:
        from ccece.experiments.sota_comparison.timing import count_parameters, measure_flops, measure_inference_time, measure_gpu_memory

    comp_metrics = ComputationalMetrics(
        parameters=count_parameters(model),
        flops=measure_flops(model, (seq_len, input_dim), device),
        training_time=training_time,
        inference_time=measure_inference_time(model, (seq_len, input_dim), device),
        gpu_memory=measure_gpu_memory(model, (seq_len, input_dim), device),
    )

    return metrics, comp_metrics


def run_multihead_trajectory_protonet_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    verbose: bool = True,
) -> Tuple[ClassificationMetrics, ComputationalMetrics]:
    """
    Train and evaluate MultiHeadTrajectoryProtoNet on a single fold.

    This hybrid model combines:
    - MultiHeadProtoNet's structural constraint (2 prototypes per head)
    - PPT's temporal dynamics (trajectory prototypes with origin + velocity)
    """
    from ccece.models.multi_head_trajectory_proto_net import MultiHeadTrajectoryProtoNet
    from ccece.experiments.multi_head_trajectory_experiment import MHTConfig, MHTTrainer

    input_dim = X_train.shape[2]
    seq_len = X_train.shape[1]

    # Create model with validated configuration (72.4% accuracy, 5/5 temporal passes)
    config = MHTConfig(
        latent_dim=64,
        n_heads=5,
        head_dim=32,
        n_segments=8,
        encoder_hidden=64,
        encoder_layers=3,
        kernel_size=7,
        dropout=0.2,
        epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        early_stopping_patience=20,
        grad_clip_norm=1.0,
        cluster_loss_weight=0.3,
        separation_loss_weight=0.1,
        diversity_loss_weight=0.05,
    )

    model = MultiHeadTrajectoryProtoNet(
        input_dim=input_dim,
        num_classes=2,
        seq_len=seq_len,
        latent_dim=config.latent_dim,
        n_heads=config.n_heads,
        head_dim=config.head_dim,
        n_segments=config.n_segments,
        encoder_hidden=config.encoder_hidden,
        encoder_layers=config.encoder_layers,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
    )

    # Create data loaders
    from torch.utils.data import DataLoader, TensorDataset
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Train
    start_time = time.time()
    trainer = MHTTrainer(model, config, device)
    trainer.train(train_loader, val_loader, y_train, verbose=verbose)
    training_time = time.time() - start_time

    # Evaluate
    model.eval()
    model.to(device)

    with torch.no_grad():
        X_val_t = X_val_t.to(device)
        outputs = model(X_val_t)
        y_proba = torch.softmax(outputs, dim=1).cpu().numpy()
        y_pred = outputs.argmax(dim=1).cpu().numpy()

    metrics = compute_classification_metrics(y_val, y_pred, y_proba)

    # Computational metrics
    try:
        from .timing import count_parameters, measure_flops, measure_inference_time, measure_gpu_memory
    except ImportError:
        from ccece.experiments.sota_comparison.timing import count_parameters, measure_flops, measure_inference_time, measure_gpu_memory

    comp_metrics = ComputationalMetrics(
        parameters=count_parameters(model),
        flops=measure_flops(model, (seq_len, input_dim), device),
        training_time=training_time,
        inference_time=measure_inference_time(model, (seq_len, input_dim), device),
        gpu_memory=measure_gpu_memory(model, (seq_len, input_dim), device),
    )

    return metrics, comp_metrics


def save_intermediate_results(
    results: Dict,
    output_dir: str,
    fold: int,
):
    """Save intermediate results after each fold."""
    fold_dir = os.path.join(output_dir, 'per_fold')
    os.makedirs(fold_dir, exist_ok=True)

    fold_path = os.path.join(fold_dir, f'fold{fold}_results.json')

    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(fold_path, 'w') as f:
        json.dump(convert(results), f, indent=2)


def run_sota_comparison(
    output_dir: Optional[str] = None,
    models_to_run: Optional[List[str]] = None,
    n_folds: int = N_FOLDS,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full SOTA comparison experiment.

    Args:
        output_dir: Output directory (auto-generated if None)
        models_to_run: List of models to run (None = all models)
        n_folds: Number of cross-validation folds
        verbose: Print progress

    Returns:
        Dict with all results
    """
    set_all_seeds(RANDOM_SEED)

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/ccece/sota_comparison/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print("=" * 70)
        print("SOTA COMPARISON EXPERIMENT")
        print("=" * 70)
        print(f"Device: {device}")
        print(f"Output: {output_dir}")
        print(f"Folds: {n_folds}")
        print()

    # Load and preprocess data
    if verbose:
        print("Loading data...")
    items = load_binary_dataset(verbose=False)
    items = preprocess_items(items)

    X, y, patient_ids = extract_arrays(items)
    seq_len = compute_target_seq_len(items)
    input_dim = items[0]['data'].shape[1]

    if verbose:
        print(f"Data: {len(items)} samples, seq_len={seq_len}, input_dim={input_dim}")
        print(f"Class distribution: HC={sum(y==0)}, MG={sum(y==1)}")
        print()

    # Determine models to run
    if models_to_run is None:
        models_to_run = list(MODEL_CONFIGS.keys()) + ['MultiHeadProtoNet', 'MultiHeadTrajectoryProtoNet']

    if verbose:
        print(f"Models to evaluate: {models_to_run}")
        print()

    # Setup cross-validation
    if n_folds == 1:
        # Use GroupShuffleSplit for single fold
        from sklearn.model_selection import GroupShuffleSplit
        cv = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    else:
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    # Results storage
    all_results = {model: {'classification': [], 'computational': []} for model in models_to_run}

    # Run experiment
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, patient_ids)):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"FOLD {fold_idx + 1}/{n_folds}")
            print('=' * 60)

        # Prepare data
        X_train, y_train, X_val, y_val = prepare_data_for_fold(
            items, train_idx, val_idx, seq_len
        )

        if verbose:
            print(f"Train: {len(X_train)} samples (HC={sum(y_train==0)}, MG={sum(y_train==1)})")
            print(f"Val: {len(X_val)} samples")
            print()

        # Run each model
        for model_name in models_to_run:
            if verbose:
                print(f"\n--- {model_name} ---")

            try:
                if model_name == 'MultiHeadProtoNet':
                    cls_metrics, comp_metrics = run_multihead_protonet_fold(
                        X_train, y_train, X_val, y_val, device, verbose=False
                    )
                elif model_name == 'MultiHeadTrajectoryProtoNet':
                    cls_metrics, comp_metrics = run_multihead_trajectory_protonet_fold(
                        X_train, y_train, X_val, y_val, device, verbose=False
                    )
                else:
                    config = MODEL_CONFIGS[model_name]
                    cls_metrics, comp_metrics = run_single_model_fold(
                        model_name, config,
                        X_train, y_train, X_val, y_val,
                        device, verbose=False
                    )

                all_results[model_name]['classification'].append(cls_metrics)
                all_results[model_name]['computational'].append(comp_metrics)

                if verbose:
                    print(f"  {cls_metrics}")
                    print(f"  Params: {comp_metrics.parameters:,}, Time: {comp_metrics.training_time:.1f}s")

            except Exception as e:
                print(f"ERROR running {model_name}: {e}")
                traceback.print_exc()
                # Add placeholder metrics
                all_results[model_name]['classification'].append(None)
                all_results[model_name]['computational'].append(None)

            # Clear GPU memory
            torch.cuda.empty_cache()

        # Save intermediate results
        fold_results = {}
        for model_name in models_to_run:
            cls_list = all_results[model_name]['classification']
            if cls_list and cls_list[-1] is not None:
                fold_results[model_name] = cls_list[-1].to_dict()

        save_intermediate_results(fold_results, output_dir, fold_idx + 1)

    # Aggregate results
    if verbose:
        print("\n" + "=" * 70)
        print("AGGREGATING RESULTS")
        print("=" * 70)

    classification_aggregated = {}
    computational_aggregated = {}

    for model_name in models_to_run:
        # Filter out None results
        cls_metrics = [m for m in all_results[model_name]['classification'] if m is not None]
        comp_metrics = [m for m in all_results[model_name]['computational'] if m is not None]

        if cls_metrics:
            classification_aggregated[model_name] = aggregate_metrics(cls_metrics)

        if comp_metrics:
            # Average computational metrics
            computational_aggregated[model_name] = {
                'parameters': int(np.mean([m.parameters for m in comp_metrics])),
                'flops': int(np.mean([m.flops for m in comp_metrics if m.flops])) if any(m.flops for m in comp_metrics) else None,
                'training_time': np.mean([m.training_time for m in comp_metrics]),
                'inference_time': np.mean([m.inference_time for m in comp_metrics]),
                'gpu_memory': np.mean([m.gpu_memory for m in comp_metrics]),
            }

    # Statistical comparisons
    if verbose:
        print("\nComputing statistical tests...")

    # Prepare data for comparisons
    accuracy_per_model = {}
    for model_name in models_to_run:
        if model_name in classification_aggregated:
            accuracy_per_model[model_name] = {
                'accuracy': classification_aggregated[model_name]['accuracy']['values']
            }

    comparisons = []
    if 'MultiHeadProtoNet' in accuracy_per_model:
        comparisons = run_all_comparisons(
            accuracy_per_model,
            our_method_name='MultiHeadProtoNet',
            metric='accuracy'
        )

    # Generate tables
    if verbose:
        print("\nGenerating tables...")

    tables = generate_all_tables(
        classification_aggregated,
        computational_aggregated,
        comparisons,
        output_dir,
        our_method='MultiHeadProtoNet'
    )

    # Save full results
    full_results = {
        'config': {
            'random_seed': RANDOM_SEED,
            'n_folds': n_folds,
            'models': models_to_run,
            'seq_len': seq_len,
            'input_dim': input_dim,
            'n_samples': len(items),
        },
        'classification': {
            model: {
                metric: {
                    'mean': data['mean'],
                    'std': data['std'],
                    'ci_lower': data['ci_lower'],
                    'ci_upper': data['ci_upper'],
                    'values': data['values'],
                }
                for metric, data in metrics.items()
            }
            for model, metrics in classification_aggregated.items()
        },
        'computational': computational_aggregated,
        'comparisons': [
            {
                'baseline': c.baseline_name,
                'ttest_pvalue': c.ttest_pvalue,
                'cohens_d': c.cohens_d,
                'is_significant': c.is_significant,
                'is_better': c.is_better,
            }
            for c in comparisons
        ],
        'timestamp': datetime.now().isoformat(),
    }

    results_path = os.path.join(output_dir, 'full_results.json')
    with open(results_path, 'w') as f:
        json.dump(full_results, f, indent=2, default=float)

    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print("\nClassification Performance (Accuracy):")
        print("-" * 50)

        for model_name in models_to_run:
            if model_name in classification_aggregated:
                acc = classification_aggregated[model_name]['accuracy']
                marker = " (Ours)" if model_name == 'MultiHeadProtoNet' else ""
                print(f"  {model_name:20s}: {acc['mean']*100:.1f} ± {acc['std']*100:.1f}%{marker}")

        print("\n" + "-" * 50)
        print(f"\nResults saved to: {output_dir}")
        print(f"  - full_results.json")
        print(f"  - table1_classification.csv/tex")
        print(f"  - table2_computational.csv/tex")
        print(f"  - table3_significance.csv/tex")
        print(f"  - per_fold/")

    return full_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='SOTA Comparison Experiment')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Models to run (default: all)')
    parser.add_argument('--quiet', action='store_true', help='Reduce output')

    args = parser.parse_args()

    run_sota_comparison(
        output_dir=args.output,
        models_to_run=args.models,
        n_folds=args.folds,
        verbose=not args.quiet,
    )
