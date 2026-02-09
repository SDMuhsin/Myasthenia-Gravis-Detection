"""
Main Runner for Multi-Dataset SOTA Comparison Experiment

Runs all models with 5-fold cross-validation across multiple datasets:
- MG (Myasthenia Gravis) - Private eye-tracking dataset
- Heartbeat - UEA ECG dataset
- BasicMotions - UEA accelerometer dataset
- Epilepsy - UEA accelerometer dataset

Generates:
- Classification performance tables per dataset
- Computational efficiency tables
- Statistical significance tests
- Cross-dataset summary
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GroupShuffleSplit
from tqdm import tqdm

# Add parent to path for both module and script execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ccece.run_experiment import set_all_seeds

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
    from .datasets import (
        load_dataset, get_cv_strategy, standardize_data, DatasetConfig,
        SUPPORTED_DATASETS, compute_optimal_n_segments,
    )
    from .mhtpn_configs import (
        get_mhtpn_model_config, get_mhtpn_training_config,
        MHTPNModelConfig, MHTPNTrainingConfig, TRAINING_CONFIG,
    )
except ImportError:
    from ccece.experiments.sota_comparison.baselines import (
        SimpleCNN, SimpleLSTM, InceptionTimeWrapper, ROCKETWrapper,
        TimeSeriesTransformer, TimesNetWrapper, ConvTranWrapper, PatchTSTWrapper,
    )
    from ccece.experiments.sota_comparison.metrics import compute_classification_metrics, aggregate_metrics, ClassificationMetrics
    from ccece.experiments.sota_comparison.timing import compute_computational_metrics, ComputationalMetrics
    from ccece.experiments.sota_comparison.statistical_tests import run_all_comparisons, PairwiseComparison
    from ccece.experiments.sota_comparison.table_generator import generate_all_tables
    from ccece.experiments.sota_comparison.datasets import (
        load_dataset, get_cv_strategy, standardize_data, DatasetConfig,
        SUPPORTED_DATASETS, compute_optimal_n_segments,
    )
    from ccece.experiments.sota_comparison.mhtpn_configs import (
        get_mhtpn_model_config, get_mhtpn_training_config,
        MHTPNModelConfig, MHTPNTrainingConfig, TRAINING_CONFIG,
    )


RANDOM_SEED = 42
N_FOLDS = 5

# Standardized training config for ALL models (fair comparison)
# Based on EEG fix: epochs=150, patience=30, cosine_annealing=True
STANDARD_TRAIN_PARAMS = {
    'epochs': 150,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'patience': 30,
    'use_cosine_annealing': True,
}

# Model configurations - architecture only, training params are standardized
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
            'epochs': 150,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'patience': 30,
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
            'max_seq_len': 1024,  # Downsample sequences > 1024 to reduce memory
        },
        'train_params': {
            'epochs': 150,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'patience': 30,
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
            'epochs': 150,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'patience': 30,
        },
    },
    'ROCKET': {
        'class': ROCKETWrapper,
        'is_pytorch': False,
        'params': {
            # Updated to match Dempster et al. DMKD 2020 paper value
            'num_kernels': 10000,  # Paper recommends 10,000 kernels (was 2000)
            'random_state': RANDOM_SEED,
        },
        'train_params': {},  # ROCKET doesn't need training params
    },
    'TST': {
        'class': TimeSeriesTransformer,
        'is_pytorch': True,
        'params': {
            # Updated to match Zerveas et al. KDD 2021 paper values
            'd_model': 256,   # Paper: 128-512
            'n_heads': 8,     # Paper: 8
            'n_layers': 4,    # Paper: 3-6
            'd_ff': 512,      # Paper: 2-4x d_model
            'dropout': 0.1,
            'max_seq_len': 512,  # Downsample sequences > 512
        },
        'train_params': {
            'epochs': 150,
            'batch_size': 32,
            'learning_rate': 1e-4,  # Lower LR for transformer
            'patience': 30,
        },
    },
    'TimesNet': {
        'class': TimesNetWrapper,
        'is_pytorch': True,
        'params': {
            # Updated to match Wu et al. ICLR 2023 paper values
            'd_model': 128,  # Paper uses 128-512 for classification
            'd_ff': 256,     # Paper uses 256-512 for classification
            'n_layers': 2,   # Paper uses 2-3
            'top_k': 3,      # Paper uses 3 or 5 periods
            'num_kernels': 6,
            'dropout': 0.1,
            'max_seq_len': 512,  # Downsample sequences > 512 to prevent OOM
        },
        'train_params': {
            'epochs': 150,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'patience': 30,
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
            'epochs': 150,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'patience': 30,
        },
    },
    'PatchTST': {
        'class': PatchTSTWrapper,
        'is_pytorch': True,
        'params': {
            # patch_len and stride are now ADAPTIVE (computed automatically)
            # Based on Nie et al. ICLR 2023 - targets ~12-16 patches
            'patch_len': None,  # Will be computed from seq_len
            'stride': None,  # Will be computed from seq_len
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 3,
            'd_ff': 256,
            'dropout': 0.1,
        },
        'train_params': {
            'epochs': 150,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'patience': 30,
        },
    },
}


def run_single_model_fold(
    model_name: str,
    config: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    device: torch.device,
    verbose: bool = True,
) -> Tuple[ClassificationMetrics, ComputationalMetrics]:
    """
    Train and evaluate a single baseline model on a single fold.

    Args:
        model_name: Name of the model
        config: Model configuration dict
        X_train: Training data, shape (n_samples, seq_len, n_features)
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        num_classes: Number of classes in the dataset
        device: PyTorch device
        verbose: Print progress

    Returns:
        Tuple of (ClassificationMetrics, ComputationalMetrics)
    """
    input_dim = X_train.shape[2]
    seq_len = X_train.shape[1]

    # Create model
    model_class = config['class']
    model_params = config['params'].copy()

    model = model_class(
        input_dim=input_dim,
        num_classes=num_classes,
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
    comp_metrics = compute_computational_metrics(
        model,
        input_shape=(seq_len, input_dim),
        device=device,
        training_time=training_time,
        is_pytorch=is_pytorch,
        X_sample=X_val[:min(100, len(X_val))],
    )

    return metrics, comp_metrics


def run_mhtpn_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    dataset_config: DatasetConfig,
    fold_idx: int,
    device: torch.device,
    verbose: bool = True,
) -> Tuple[ClassificationMetrics, ComputationalMetrics]:
    """
    Train and evaluate MHTPN (MultiHeadTrajectoryProtoNet) on a single fold.

    Uses per-dataset model configuration and standardized training parameters.

    Args:
        X_train: Training data, shape (n_samples, seq_len, n_features)
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        dataset_config: Dataset configuration
        fold_idx: Current fold index (for seed variation)
        device: PyTorch device
        verbose: Print progress

    Returns:
        Tuple of (ClassificationMetrics, ComputationalMetrics)
    """
    from ccece.models.multi_head_trajectory_proto_net import MultiHeadTrajectoryProtoNet

    input_dim = X_train.shape[2]
    seq_len = X_train.shape[1]
    num_classes = dataset_config.n_classes

    # Get per-dataset model config
    model_config = get_mhtpn_model_config(dataset_config.name, seq_len)
    train_config = get_mhtpn_training_config()

    # Set per-fold seed for initialization diversity
    torch.manual_seed(RANDOM_SEED + fold_idx)
    np.random.seed(RANDOM_SEED + fold_idx)

    # Create model
    model = MultiHeadTrajectoryProtoNet(
        input_dim=input_dim,
        num_classes=num_classes,
        seq_len=seq_len,
        latent_dim=model_config.latent_dim,
        n_heads=model_config.n_heads,
        head_dim=model_config.head_dim,
        n_segments=model_config.n_segments,
        encoder_hidden=model_config.encoder_hidden,
        encoder_layers=model_config.encoder_layers,
        kernel_size=model_config.kernel_size,
        dropout=model_config.dropout,
    ).to(device)

    # Create data loaders
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
    )

    # Setup optimizer with class weights
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts.astype(np.float32)
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.from_numpy(class_weights).float().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    # Setup scheduler
    scheduler = None
    if train_config.use_cosine_annealing:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config.epochs,
        )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    start_time = time.time()

    pbar = tqdm(
        range(train_config.epochs),
        disable=not verbose,
        desc=f"MHTPN",
        leave=False,
    )

    for epoch in pbar:
        # Train
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip_norm)
            optimizer.step()

            total_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        # Validate
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_acc = correct / total

        pbar.set_postfix({
            'loss': f'{total_loss / len(train_loader):.3f}',
            'val_acc': f'{val_acc * 100:.1f}%',
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= train_config.early_stopping_patience:
            if verbose:
                tqdm.write(f"    Early stopping at epoch {epoch + 1}")
            break

    training_time = time.time() - start_time

    # Load best model and evaluate
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()

    with torch.no_grad():
        all_preds = []
        all_probs = []

        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_proba = np.concatenate(all_probs)

    metrics = compute_classification_metrics(y_val, y_pred, y_proba)

    # Compute computational metrics
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
    """Save intermediate results after each fold (crash protection)."""
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


def run_dataset_comparison(
    dataset_name: str,
    output_dir: str,
    models_to_run: Optional[List[str]] = None,
    n_folds: int = N_FOLDS,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run SOTA comparison on a single dataset.

    Args:
        dataset_name: Name of dataset ('MG', 'Heartbeat', 'BasicMotions', 'Epilepsy')
        output_dir: Output directory for this dataset's results
        models_to_run: List of models to run (None = all)
        n_folds: Number of cross-validation folds
        verbose: Print progress

    Returns:
        Dict with classification and computational results
    """
    set_all_seeds(RANDOM_SEED)

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"DATASET: {dataset_name}")
        print('=' * 70)
        print(f"Device: {device}")
        print(f"Output: {output_dir}")
        print()

    # Load dataset
    if verbose:
        print("Loading data...")

    X, y, groups, dataset_config = load_dataset(dataset_name, verbose=verbose)

    if verbose:
        print(f"Data: {len(y)} samples, seq_len={dataset_config.seq_len}, n_features={dataset_config.n_features}")
        print(f"Classes: {dataset_config.n_classes} - {dataset_config.class_names}")
        class_counts = np.bincount(y)
        print(f"Class distribution: {dict(zip(dataset_config.class_names, class_counts))}")
        print()

    # Determine models to run
    if models_to_run is None:
        models_to_run = list(MODEL_CONFIGS.keys()) + ['MHTPN']

    if verbose:
        print(f"Models to evaluate: {models_to_run}")
        print()

    # Setup cross-validation
    if n_folds == 1:
        # Use shuffle split for single fold
        if dataset_config.has_groups:
            cv = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
        else:
            from sklearn.model_selection import StratifiedShuffleSplit
            cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    else:
        cv = get_cv_strategy(dataset_config, n_splits=n_folds, random_state=RANDOM_SEED)

    # Results storage
    all_results = {model: {'classification': [], 'computational': []} for model in models_to_run}

    # Get CV splits
    if dataset_config.has_groups:
        splits = list(cv.split(X, y, groups))
    else:
        splits = list(cv.split(X, y))

    # Run experiment
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        if verbose:
            print(f"\n{'-' * 60}")
            print(f"FOLD {fold_idx + 1}/{n_folds}")
            print('-' * 60)

        # Split data
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        # Standardize per fold
        X_train, X_val = standardize_data(X_train, X_val)

        if verbose:
            train_counts = np.bincount(y_train)
            print(f"Train: {len(X_train)} samples, distribution: {train_counts}")
            print(f"Val: {len(X_val)} samples")
            print()

        # Run each model
        for model_name in models_to_run:
            if verbose:
                print(f"  Running {model_name}...", end=' ', flush=True)

            try:
                if model_name == 'MHTPN':
                    cls_metrics, comp_metrics = run_mhtpn_fold(
                        X_train, y_train, X_val, y_val,
                        dataset_config, fold_idx, device,
                        verbose=False,
                    )
                else:
                    config = MODEL_CONFIGS[model_name]
                    cls_metrics, comp_metrics = run_single_model_fold(
                        model_name, config,
                        X_train, y_train, X_val, y_val,
                        dataset_config.n_classes,
                        device, verbose=False,
                    )

                all_results[model_name]['classification'].append(cls_metrics)
                all_results[model_name]['computational'].append(comp_metrics)

                if verbose:
                    print(f"Acc: {cls_metrics.accuracy * 100:.1f}%, "
                          f"F1: {cls_metrics.f1_score * 100:.1f}%")

            except Exception as e:
                print(f"ERROR: {e}")
                traceback.print_exc()
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
        print(f"\n{'=' * 60}")
        print("AGGREGATING RESULTS")
        print('=' * 60)

    classification_aggregated = {}
    computational_aggregated = {}

    for model_name in models_to_run:
        cls_metrics = [m for m in all_results[model_name]['classification'] if m is not None]
        comp_metrics = [m for m in all_results[model_name]['computational'] if m is not None]

        if cls_metrics:
            classification_aggregated[model_name] = aggregate_metrics(cls_metrics)

        if comp_metrics:
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

    accuracy_per_model = {}
    for model_name in models_to_run:
        if model_name in classification_aggregated:
            accuracy_per_model[model_name] = {
                'accuracy': classification_aggregated[model_name]['accuracy']['values']
            }

    comparisons = []
    if 'MHTPN' in accuracy_per_model:
        comparisons = run_all_comparisons(
            accuracy_per_model,
            our_method_name='MHTPN',
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
        our_method='MHTPN'
    )

    # Save full results
    full_results = {
        'dataset': {
            'name': dataset_name,
            'n_samples': dataset_config.n_samples,
            'n_features': dataset_config.n_features,
            'seq_len': dataset_config.seq_len,
            'n_classes': dataset_config.n_classes,
            'class_names': dataset_config.class_names,
            'domain': dataset_config.domain,
        },
        'config': {
            'random_seed': RANDOM_SEED,
            'n_folds': n_folds,
            'models': models_to_run,
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
        print(f"\n{'=' * 60}")
        print(f"RESULTS SUMMARY: {dataset_name}")
        print('=' * 60)
        print("\nClassification Performance (Accuracy):")
        print("-" * 50)

        # Sort by accuracy
        sorted_models = sorted(
            classification_aggregated.keys(),
            key=lambda m: classification_aggregated[m]['accuracy']['mean'],
            reverse=True,
        )

        for rank, model_name in enumerate(sorted_models, 1):
            acc = classification_aggregated[model_name]['accuracy']
            marker = " (Ours)" if model_name == 'MHTPN' else ""
            print(f"  {rank}. {model_name:20s}: {acc['mean']*100:.1f} +/- {acc['std']*100:.1f}%{marker}")

        print(f"\nResults saved to: {output_dir}")

    return full_results


def run_sota_comparison(
    output_dir: Optional[str] = None,
    datasets: Optional[List[str]] = None,
    models_to_run: Optional[List[str]] = None,
    n_folds: int = N_FOLDS,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full multi-dataset SOTA comparison experiment.

    Args:
        output_dir: Output directory (auto-generated if None)
        datasets: List of datasets to run (None = all supported datasets)
        models_to_run: List of models to run (None = all models)
        n_folds: Number of cross-validation folds
        verbose: Print progress

    Returns:
        Dict with all results per dataset
    """
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/ccece/sota_comparison/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Determine datasets to run
    if datasets is None:
        datasets = SUPPORTED_DATASETS
    else:
        # Validate datasets
        for ds in datasets:
            if ds not in SUPPORTED_DATASETS:
                raise ValueError(f"Unknown dataset: {ds}. Supported: {SUPPORTED_DATASETS}")

    if verbose:
        print("=" * 70)
        print("MULTI-DATASET SOTA COMPARISON EXPERIMENT")
        print("=" * 70)
        print(f"Datasets: {datasets}")
        print(f"Output: {output_dir}")
        print(f"Folds: {n_folds}")
        print()

    # Save experiment config
    config = {
        'datasets': datasets,
        'models': models_to_run or (list(MODEL_CONFIGS.keys()) + ['MHTPN']),
        'n_folds': n_folds,
        'random_seed': RANDOM_SEED,
        'training_config': {
            'epochs': 150,
            'batch_size': 32,
            'patience': 30,
            'use_cosine_annealing': True,
        },
        'timestamp': datetime.now().isoformat(),
    }

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Run each dataset
    all_results = {}

    for dataset_name in datasets:
        dataset_output_dir = os.path.join(output_dir, dataset_name)

        results = run_dataset_comparison(
            dataset_name=dataset_name,
            output_dir=dataset_output_dir,
            models_to_run=models_to_run,
            n_folds=n_folds,
            verbose=verbose,
        )

        all_results[dataset_name] = results

    # Generate cross-dataset summary
    if len(datasets) > 1:
        generate_cross_dataset_summary(all_results, output_dir, verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETE")
        print("=" * 70)
        print(f"Results saved to: {output_dir}")

    return all_results


def generate_cross_dataset_summary(
    all_results: Dict[str, Dict],
    output_dir: str,
    verbose: bool = True,
) -> None:
    """
    Generate a cross-dataset summary table.

    Args:
        all_results: Dict mapping dataset name to results
        output_dir: Output directory
        verbose: Print summary
    """
    summary_dir = os.path.join(output_dir, 'combined')
    os.makedirs(summary_dir, exist_ok=True)

    # Build summary data
    summary_rows = []

    for dataset_name, results in all_results.items():
        dataset_info = results['dataset']
        classification = results['classification']

        # Get MHTPN performance
        if 'MHTPN' in classification:
            mhtpn_acc = classification['MHTPN']['accuracy']['mean']
            mhtpn_std = classification['MHTPN']['accuracy']['std']
        else:
            mhtpn_acc = None
            mhtpn_std = None

        # Find best baseline
        best_baseline = None
        best_baseline_acc = 0.0
        best_baseline_std = 0.0

        for model_name, metrics in classification.items():
            if model_name == 'MHTPN':
                continue
            acc = metrics['accuracy']['mean']
            if acc > best_baseline_acc:
                best_baseline_acc = acc
                best_baseline_std = metrics['accuracy']['std']
                best_baseline = model_name

        # Calculate improvement
        if mhtpn_acc is not None and best_baseline_acc > 0:
            improvement = (mhtpn_acc - best_baseline_acc) * 100
        else:
            improvement = 0.0

        # Determine rank
        if mhtpn_acc is not None:
            all_accs = [m['accuracy']['mean'] for m in classification.values()]
            rank = sorted(all_accs, reverse=True).index(mhtpn_acc) + 1
        else:
            rank = None

        summary_rows.append({
            'Dataset': dataset_name,
            'Domain': dataset_info['domain'],
            'Samples': dataset_info['n_samples'],
            'Classes': dataset_info['n_classes'],
            'MHTPN_Acc': mhtpn_acc,
            'MHTPN_Std': mhtpn_std,
            'Best_Baseline': best_baseline,
            'Baseline_Acc': best_baseline_acc,
            'Baseline_Std': best_baseline_std,
            'Improvement': improvement,
            'Rank': rank,
        })

    # Save as CSV
    import csv
    csv_path = os.path.join(summary_dir, 'cross_dataset_summary.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    # Save as JSON
    json_path = os.path.join(summary_dir, 'cross_dataset_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary_rows, f, indent=2)

    if verbose:
        print("\n" + "=" * 70)
        print("CROSS-DATASET SUMMARY")
        print("=" * 70)
        print(f"\n{'Dataset':<15} {'Domain':<15} {'MHTPN Acc':<12} {'Best Baseline':<15} {'Improvement':<12} {'Rank'}")
        print("-" * 80)

        for row in summary_rows:
            mhtpn_str = f"{row['MHTPN_Acc']*100:.1f}%" if row['MHTPN_Acc'] else "N/A"
            baseline_str = f"{row['Best_Baseline']} ({row['Baseline_Acc']*100:.1f}%)" if row['Best_Baseline'] else "N/A"
            imp_str = f"+{row['Improvement']:.1f}%" if row['Improvement'] >= 0 else f"{row['Improvement']:.1f}%"
            rank_str = f"#{row['Rank']}" if row['Rank'] else "N/A"

            print(f"{row['Dataset']:<15} {row['Domain']:<15} {mhtpn_str:<12} {baseline_str:<15} {imp_str:<12} {rank_str}")

        print(f"\nSummary saved to: {csv_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Multi-Dataset SOTA Comparison Experiment')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        choices=SUPPORTED_DATASETS + ['all'],
                        help='Datasets to run (default: all)')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=SUPPORTED_DATASETS,
                        help='Single dataset to run (shortcut)')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Models to run (default: all)')
    parser.add_argument('--quiet', action='store_true', help='Reduce output')

    args = parser.parse_args()

    # Handle dataset selection
    if args.dataset:
        datasets = [args.dataset]
    elif args.datasets:
        if 'all' in args.datasets:
            datasets = None  # Will use all supported datasets
        else:
            datasets = args.datasets
    else:
        datasets = None  # Will use all supported datasets

    run_sota_comparison(
        output_dir=args.output,
        datasets=datasets,
        models_to_run=args.models,
        n_folds=args.folds,
        verbose=not args.quiet,
    )
