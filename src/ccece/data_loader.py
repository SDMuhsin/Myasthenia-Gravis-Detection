#!/usr/bin/env python3
"""
CCECE Paper: Data Loading Utilities

This module provides functions to load the MG vs HC saccade dataset.
Use this as the standard way to load data for all experiments.

Usage:
    from data_loader import load_binary_dataset, prepare_data_splits

    items = load_binary_dataset()
    splits = prepare_data_splits(items)
"""

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold

# =============================================================================
# CONSTANTS - DO NOT CHANGE
# =============================================================================

BASE_DIR = './data'
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']
CSV_ENCODING = 'utf-16-le'  # CRITICAL - the data uses this encoding
CSV_SEPARATOR = ','
MIN_SEQ_LEN_THRESHOLD = 50
SAMPLE_RATE = 120  # Hz


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data_from_folder(folder_path, class_name, label, verbose=True):
    """
    Load all CSV files from a folder containing patient subfolders.

    Args:
        folder_path: Path to the class folder (e.g., './data/Healthy control')
        class_name: Name of the class ('HC', 'MG', etc.)
        label: Integer label for this class
        verbose: Whether to show progress bar

    Returns:
        List of dicts, each containing:
            - 'data': numpy array of shape (seq_len, 6)
            - 'label': integer label
            - 'patient_id': string patient identifier
            - 'filename': original filename
            - 'class_name': class name string
    """
    items = []

    if not os.path.isdir(folder_path):
        print(f"Warning: Directory not found: {folder_path}")
        return items

    # Get patient directories
    patient_dirs = [d for d in os.listdir(folder_path)
                    if os.path.isdir(os.path.join(folder_path, d))]

    iterator = tqdm(patient_dirs, desc=f"Loading {class_name}") if verbose else patient_dirs

    for patient_folder in iterator:
        patient_path = os.path.join(folder_path, patient_folder)
        csv_files = glob.glob(os.path.join(patient_path, '*.csv'))

        for csv_file in csv_files:
            item = _load_single_csv(csv_file, class_name, label, patient_folder)
            if item is not None:
                items.append(item)

    return items


def _load_single_csv(csv_path, class_name, label, patient_id):
    """Load a single CSV file and return as dict, or None if invalid."""
    try:
        # Read CSV with correct encoding
        df = pd.read_csv(csv_path, encoding=CSV_ENCODING, sep=CSV_SEPARATOR)
        df.columns = [col.strip() for col in df.columns]

        # Check required columns
        if not all(col in df.columns for col in FEATURE_COLUMNS):
            return None

        # Check minimum length
        if len(df) < MIN_SEQ_LEN_THRESHOLD:
            return None

        # Extract feature columns
        df_features = df[FEATURE_COLUMNS].copy()

        # Convert to numeric, coercing errors
        for col in df_features.columns:
            df_features.loc[:, col] = pd.to_numeric(df_features[col], errors='coerce')

        # Check for excessive NaN
        if df_features.isnull().sum().sum() > 0.1 * df_features.size:
            return None

        # Fill remaining NaN with zeros
        df_features = df_features.fillna(0)

        return {
            'data': df_features.values.astype(np.float32),
            'label': label,
            'patient_id': patient_id,
            'filename': os.path.basename(csv_path),
            'class_name': class_name,
        }

    except Exception as e:
        return None


def load_binary_dataset(base_dir=BASE_DIR, verbose=True):
    """
    Load the complete HC vs MG dataset for binary classification.

    Args:
        base_dir: Base directory containing 'Healthy control', 'Definite MG', etc.
        verbose: Whether to show progress

    Returns:
        List of item dicts (see load_data_from_folder for structure)
    """
    all_items = []

    # Load Healthy Controls (label=0)
    hc_items = load_data_from_folder(
        os.path.join(base_dir, 'Healthy control'),
        class_name='HC',
        label=0,
        verbose=verbose
    )
    all_items.extend(hc_items)

    # Load MG - Definite (label=1)
    mg_definite = load_data_from_folder(
        os.path.join(base_dir, 'Definite MG'),
        class_name='MG',
        label=1,
        verbose=verbose
    )
    all_items.extend(mg_definite)

    # Load MG - Probable (label=1)
    mg_probable = load_data_from_folder(
        os.path.join(base_dir, 'Probable MG'),
        class_name='MG',
        label=1,
        verbose=verbose
    )
    all_items.extend(mg_probable)

    if verbose:
        n_hc = len(hc_items)
        n_mg = len(mg_definite) + len(mg_probable)
        print(f"\nDataset loaded:")
        print(f"  HC: {n_hc} sequences")
        print(f"  MG: {n_mg} sequences (Definite: {len(mg_definite)}, Probable: {len(mg_probable)})")
        print(f"  Total: {len(all_items)} sequences")

    return all_items


# =============================================================================
# DATA PREPARATION
# =============================================================================

def extract_arrays(items):
    """
    Extract numpy arrays from item list.

    Returns:
        X: List of numpy arrays, each of shape (seq_len, 6)
        y: numpy array of labels, shape (n_samples,)
        patient_ids: List of patient ID strings
    """
    X = [item['data'] for item in items]
    y = np.array([item['label'] for item in items])
    patient_ids = [item['patient_id'] for item in items]

    return X, y, patient_ids


def get_sequence_stats(X):
    """Print statistics about sequence lengths."""
    lengths = [x.shape[0] for x in X]
    print(f"Sequence length statistics:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {np.mean(lengths):.1f}")
    print(f"  Median: {np.median(lengths):.1f}")
    print(f"  90th percentile: {np.percentile(lengths, 90):.1f}")
    return lengths


def prepare_cv_splits(items, n_splits=5, random_state=42):
    """
    Prepare stratified cross-validation splits, grouped by patient.

    This ensures that sequences from the same patient are never split
    across train and validation sets.

    Args:
        items: List of item dicts from load_binary_dataset()
        n_splits: Number of CV folds
        random_state: Random seed for reproducibility

    Yields:
        (train_items, val_items) for each fold
    """
    X, y, patient_ids = extract_arrays(items)
    patient_ids_array = np.array(patient_ids)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=patient_ids_array)):
        train_items = [items[i] for i in train_idx]
        val_items = [items[i] for i in val_idx]
        yield train_items, val_items


# =============================================================================
# PREPROCESSING
# =============================================================================

def subsample_sequence(data, factor=10):
    """Subsample a sequence by taking every `factor`-th sample."""
    return data[::factor]


def pad_or_truncate(seq, target_len):
    """Pad with zeros or truncate to achieve target length."""
    if len(seq) >= target_len:
        return seq[:target_len]
    else:
        padding = np.zeros((target_len - len(seq), seq.shape[1]), dtype=seq.dtype)
        return np.vstack([seq, padding])


class StandardScaler:
    """Standard scaler for sequence data."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, sequences):
        """Fit scaler on list of sequences."""
        all_data = np.vstack(sequences)
        self.mean = all_data.mean(axis=0)
        self.std = all_data.std(axis=0) + 1e-8
        return self

    def transform(self, seq):
        """Transform a single sequence."""
        return (seq - self.mean) / self.std

    def fit_transform(self, sequences):
        """Fit and transform list of sequences."""
        self.fit(sequences)
        return [self.transform(seq) for seq in sequences]


def add_engineered_features(data):
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
    ])


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("Testing data loader...")

    # Load dataset
    items = load_binary_dataset()

    # Extract arrays
    X, y, patient_ids = extract_arrays(items)

    # Print stats
    print(f"\nClass distribution:")
    print(f"  HC (label=0): {np.sum(y == 0)}")
    print(f"  MG (label=1): {np.sum(y == 1)}")

    print(f"\nUnique patients: {len(set(patient_ids))}")

    get_sequence_stats(X)

    # Test CV splits
    print(f"\nTesting 5-fold CV splits...")
    for fold_idx, (train_items, val_items) in enumerate(prepare_cv_splits(items)):
        train_y = np.array([item['label'] for item in train_items])
        val_y = np.array([item['label'] for item in val_items])
        print(f"  Fold {fold_idx + 1}: Train={len(train_items)} "
              f"(HC:{np.sum(train_y==0)}, MG:{np.sum(train_y==1)}), "
              f"Val={len(val_items)} "
              f"(HC:{np.sum(val_y==0)}, MG:{np.sum(val_y==1)})")

    print("\nData loader test complete!")
