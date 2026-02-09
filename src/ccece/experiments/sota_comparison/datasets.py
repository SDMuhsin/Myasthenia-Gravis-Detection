"""
Dataset Abstraction Layer for Multi-Dataset SOTA Comparison

Provides a unified interface for loading and processing multiple datasets:
- MG (Myasthenia Gravis) - Private eye-tracking dataset
- Heartbeat - UEA ECG dataset
- BasicMotions - UEA accelerometer dataset
- Epilepsy - UEA accelerometer dataset
"""

import os
import sys
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from ccece.data_loader import load_binary_dataset, extract_arrays
from ccece.run_experiment import preprocess_items, compute_target_seq_len


RANDOM_SEED = 42

# Supported datasets
SUPPORTED_DATASETS = ['MG', 'Heartbeat', 'BasicMotions', 'Epilepsy', 'SpokenArabicDigits', 'LSST', 'Handwriting']


@dataclass
class DatasetConfig:
    """Configuration and metadata for a dataset."""
    name: str
    n_samples: int
    n_features: int
    seq_len: int
    n_classes: int
    class_names: List[str]
    domain: str  # 'eye-tracking', 'ecg', 'accelerometer'
    has_groups: bool  # Whether dataset has patient/subject grouping


def compute_optimal_n_segments(seq_len: int) -> int:
    """
    Compute optimal number of segments based on sequence length.

    Heuristic: Target segment size of ~100-200 timesteps.
    This preserves temporal dynamics while providing enough segments
    for trajectory learning.

    Args:
        seq_len: Length of the input sequence

    Returns:
        n_segments: Number of segments (2, 4, or 8)
    """
    if seq_len <= 200:
        return 2
    elif seq_len <= 400:
        return 4
    elif seq_len <= 800:
        return 8
    else:
        # For long sequences, target ~400 timesteps per segment
        return max(2, min(8, seq_len // 400))


def standardize_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardize data using training set statistics (per channel).

    Args:
        X_train: Training data, shape (n_samples, seq_len, n_channels)
        X_val: Validation data, shape (n_samples, seq_len, n_channels)

    Returns:
        X_train_norm, X_val_norm: Normalized arrays as float32
    """
    n_samples, seq_len, n_channels = X_train.shape

    # Reshape to (n_samples * seq_len, n_channels)
    train_flat = X_train.reshape(-1, n_channels)

    # Compute mean and std
    mean = train_flat.mean(axis=0, keepdims=True)
    std = train_flat.std(axis=0, keepdims=True) + 1e-8

    # Standardize
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    return X_train_norm.astype(np.float32), X_val_norm.astype(np.float32)


def load_mg_dataset(
    base_dir: str = './data',
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DatasetConfig]:
    """
    Load the MG (Myasthenia Gravis) dataset.

    Args:
        base_dir: Base directory containing the data
        verbose: Print loading progress

    Returns:
        X: Data array, shape (n_samples, seq_len, n_features)
        y: Labels, shape (n_samples,)
        groups: Patient IDs for StratifiedGroupKFold
        config: Dataset configuration
    """
    # Load raw data
    items = load_binary_dataset(base_dir=base_dir, verbose=verbose)
    items = preprocess_items(items)

    # Extract arrays
    X_list, y, patient_ids = extract_arrays(items)
    seq_len = compute_target_seq_len(items)

    # Pad/truncate to consistent length
    X_processed = []
    for data in X_list:
        if len(data) >= seq_len:
            data = data[:seq_len]
        else:
            padding = np.zeros((seq_len - len(data), data.shape[1]), dtype=np.float32)
            data = np.vstack([data, padding])
        X_processed.append(data)

    X = np.array(X_processed, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Create groups array (patient IDs as integers)
    unique_patients = {pid: i for i, pid in enumerate(sorted(set(patient_ids)))}
    groups = np.array([unique_patients[pid] for pid in patient_ids], dtype=np.int64)

    config = DatasetConfig(
        name='MG',
        n_samples=len(y),
        n_features=X.shape[2],
        seq_len=seq_len,
        n_classes=2,
        class_names=['HC', 'MG'],
        domain='eye-tracking',
        has_groups=True,
    )

    return X, y, groups, config


def load_uea_dataset(
    dataset_name: str,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DatasetConfig]:
    """
    Load a UEA multivariate time series classification dataset.

    Args:
        dataset_name: Name of UEA dataset ('Heartbeat', 'BasicMotions', 'Epilepsy')
        verbose: Print loading progress

    Returns:
        X: Data array, shape (n_samples, seq_len, n_features)
        y: Labels, shape (n_samples,)
        groups: Sample indices (no actual grouping for UEA datasets)
        config: Dataset configuration
    """
    from aeon.datasets import load_classification

    if verbose:
        print(f"Loading UEA dataset: {dataset_name}...")

    # Load train and test splits
    X_train, y_train = load_classification(dataset_name, split="train")
    X_test, y_test = load_classification(dataset_name, split="test")

    # Combine train and test (we do our own CV)
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    class_names = list(le.classes_)

    # Convert to float32
    X = X.astype(np.float32)

    # Transpose from (n_samples, n_channels, seq_len) to (n_samples, seq_len, n_channels)
    X = np.transpose(X, (0, 2, 1))

    # UEA datasets don't have groups, use sample indices
    groups = np.arange(len(y), dtype=np.int64)

    # Determine domain based on dataset
    domain_map = {
        'Heartbeat': 'ecg',
        'BasicMotions': 'accelerometer',
        'Epilepsy': 'accelerometer',
        'SpokenArabicDigits': 'audio',
        'LSST': 'astronomy',
        'Handwriting': 'motion',  # smartwatch accelerometer/gyroscope
    }
    domain = domain_map.get(dataset_name, 'unknown')

    config = DatasetConfig(
        name=dataset_name,
        n_samples=len(y),
        n_features=X.shape[2],
        seq_len=X.shape[1],
        n_classes=len(class_names),
        class_names=class_names,
        domain=domain,
        has_groups=False,
    )

    if verbose:
        print(f"  Samples: {config.n_samples}")
        print(f"  Features: {config.n_features}")
        print(f"  Seq length: {config.seq_len}")
        print(f"  Classes: {config.n_classes} - {class_names}")

    return X, y, groups, config


def load_dataset(
    dataset_name: str,
    base_dir: str = './data',
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DatasetConfig]:
    """
    Load a dataset by name.

    Args:
        dataset_name: Name of dataset ('MG', 'Heartbeat', 'BasicMotions', 'Epilepsy')
        base_dir: Base directory for MG data
        verbose: Print loading progress

    Returns:
        X: Data array, shape (n_samples, seq_len, n_features)
        y: Labels, shape (n_samples,)
        groups: Group IDs for cross-validation
        config: Dataset configuration

    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported datasets: {SUPPORTED_DATASETS}"
        )

    if dataset_name == 'MG':
        return load_mg_dataset(base_dir=base_dir, verbose=verbose)
    else:
        return load_uea_dataset(dataset_name, verbose=verbose)


def get_cv_strategy(
    dataset_config: DatasetConfig,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> Union[StratifiedGroupKFold, StratifiedKFold]:
    """
    Get appropriate cross-validation strategy for a dataset.

    Args:
        dataset_config: Dataset configuration
        n_splits: Number of CV folds
        random_state: Random seed for reproducibility

    Returns:
        CV splitter object (StratifiedGroupKFold or StratifiedKFold)
    """
    if dataset_config.has_groups:
        return StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
    else:
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )


def get_dataset_info(dataset_name: str) -> Dict:
    """
    Get static information about a dataset without loading it.

    Args:
        dataset_name: Name of dataset

    Returns:
        Dict with dataset metadata
    """
    info = {
        'MG': {
            'full_name': 'Myasthenia Gravis',
            'n_samples': 1331,
            'n_features': 14,
            'seq_len': 2903,
            'n_classes': 2,
            'domain': 'eye-tracking',
            'class_names': ['HC', 'MG'],
            'n_segments': 8,
        },
        'Heartbeat': {
            'full_name': 'Heartbeat',
            'n_samples': 409,
            'n_features': 61,
            'seq_len': 405,
            'n_classes': 2,
            'domain': 'ecg',
            'class_names': ['normal', 'abnormal'],
            'n_segments': 4,
        },
        'BasicMotions': {
            'full_name': 'BasicMotions',
            'n_samples': 80,
            'n_features': 6,
            'seq_len': 100,
            'n_classes': 4,
            'domain': 'accelerometer',
            'class_names': ['badminton', 'running', 'standing', 'walking'],
            'n_segments': 2,
        },
        'Epilepsy': {
            'full_name': 'Epilepsy',
            'n_samples': 275,
            'n_features': 3,
            'seq_len': 206,
            'n_classes': 4,
            'domain': 'accelerometer',
            'class_names': ['EPILEPSY', 'WALKING', 'RUNNING', 'SAWING'],
            'n_segments': 2,
        },
        'SpokenArabicDigits': {
            'full_name': 'SpokenArabicDigits',
            'n_samples': 8798,
            'n_features': 13,
            'seq_len': 65,
            'n_classes': 10,
            'domain': 'audio',
            'class_names': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
            'n_segments': 2,
        },
        'LSST': {
            'full_name': 'LSST (Large Synoptic Survey Telescope)',
            'n_samples': 4925,
            'n_features': 6,
            'seq_len': 36,
            'n_classes': 14,
            'domain': 'astronomy',
            'class_names': ['6', '15', '16', '42', '52', '53', '62', '64', '65', '67', '88', '90', '92', '95'],
            'n_segments': 2,
        },
        'Handwriting': {
            'full_name': 'Handwriting (Smartwatch Character Recognition)',
            'n_samples': 1000,
            'n_features': 3,
            'seq_len': 152,
            'n_classes': 26,
            'domain': 'motion',
            'class_names': [str(i) for i in range(1, 27)],  # Letters A-Z encoded as 1-26
            'n_segments': 2,  # 152 timesteps -> 2 segments
        },
    }

    if dataset_name not in info:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return info[dataset_name]


def list_datasets() -> List[str]:
    """Return list of supported dataset names."""
    return SUPPORTED_DATASETS.copy()
