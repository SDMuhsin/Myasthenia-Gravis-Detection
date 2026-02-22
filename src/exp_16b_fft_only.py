#!/usr/bin/env python3
"""
Experiment 16b: FFT Frequency Feature Only
Tests the FFT frequency domain feature against the baseline model.

This is a standalone script to quickly test the new FFT feature
without re-running all other features from Experiment 16.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.data_loading import load_raw_sequences_and_labels
from utils.modeling import create_results_directory
from utils.deep_learning import (SaccadeRNN_Small, SaccadeStandardScaler, EarlyStopper,
                                subsample_data, train_epoch, evaluate_epoch)

# --- Configuration ---
BASE_DIR = './data'
BINARY_CLASS_DEFINITIONS = {
    'HC': {'path': 'Healthy control', 'label': 0},
    'MG': {'path': 'Definite MG', 'label': 1},
    'Probable_MG': {'path': 'Probable MG', 'label': 1},
}

FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']
CSV_ENCODING = 'utf-16-le'
CSV_SEPARATOR = ','
MIN_SEQ_LEN_THRESHOLD = 50
RESULTS_DIR = './results/exp_16_eyediff_features'
RANDOM_STATE = 42

# Deep Learning Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUBSAMPLE_FACTOR = 10
TARGET_SEQ_LEN_PERCENTILE = 90
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 8
NUM_FOLDS = 5
NUM_TRIALS = 3

# FFT parameters
SAMPLE_RATE = 120  # Hz


def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_binary_data(raw_items):
    """Prepare data for binary classification."""
    binary_items = []
    for item in raw_items:
        if item['class_name'] in ['MG', 'Probable_MG']:
            new_item = item.copy()
            new_item['class_name'] = 'MG'
            new_item['label'] = 1
            binary_items.append(new_item)
        elif item['class_name'] == 'HC':
            binary_items.append(item)
    return binary_items


def add_baseline_channels(binary_items):
    """Add 8 engineered channels to 6 raw = 14 total."""
    enhanced_items = []
    for item in binary_items:
        data = item['data']

        # Velocities
        lh_vel = np.gradient(data[:, 0])
        rh_vel = np.gradient(data[:, 1])
        lv_vel = np.gradient(data[:, 2])
        rv_vel = np.gradient(data[:, 3])

        # Errors
        errorh_l = data[:, 0] - data[:, 4]
        errorh_r = data[:, 1] - data[:, 4]
        errorv_l = data[:, 2] - data[:, 5]
        errorv_r = data[:, 3] - data[:, 5]

        enhanced_data = np.column_stack([
            data,
            lh_vel, rh_vel, lv_vel, rv_vel,
            errorh_l, errorh_r, errorv_l, errorv_r
        ])

        new_item = item.copy()
        new_item['data'] = enhanced_data
        enhanced_items.append(new_item)

    return enhanced_items


def compute_fft_features(data, window_size=60, hop_size=30):
    """
    FFT: Frequency domain features from eye position signals.

    Extracts spectral features that may capture:
    - Tremor frequencies (typically 3-8 Hz in MG)
    - Saccade dynamics in frequency domain
    - Eye movement instability patterns

    Returns 3 channels: spectral_power_L, spectral_power_R, spectral_asymmetry
    """
    lv = data[:, 2]  # Left vertical position
    rv = data[:, 3]  # Right vertical position
    n_samples = len(lv)

    # Initialize output channels
    spectral_l_channel = np.zeros(n_samples)
    spectral_r_channel = np.zeros(n_samples)
    asym_channel = np.zeros(n_samples)

    # Frequency bands (at 120 Hz sample rate, Nyquist = 60 Hz)
    freq_resolution = SAMPLE_RATE / window_size  # Hz per bin

    mid_band_start = int(2.0 / freq_resolution)
    mid_band_end = int(6.0 / freq_resolution)

    # Sliding window FFT
    for start in range(0, n_samples - window_size, hop_size):
        end = start + window_size

        # Extract windows
        window_l = lv[start:end]
        window_r = rv[start:end]

        # Apply Hanning window to reduce spectral leakage
        hanning = np.hanning(window_size)
        window_l = window_l * hanning
        window_r = window_r * hanning

        # Compute FFT
        fft_l = np.abs(np.fft.rfft(window_l))
        fft_r = np.abs(np.fft.rfft(window_r))

        # Compute power in mid band (tremor range - most relevant for MG)
        mid_power_l = np.sum(fft_l[mid_band_start:mid_band_end+1]**2)
        mid_power_r = np.sum(fft_r[mid_band_start:mid_band_end+1]**2)

        # Total power for normalization
        total_power_l = np.sum(fft_l**2) + 1e-10
        total_power_r = np.sum(fft_r**2) + 1e-10

        # Relative mid-band power (tremor proportion)
        rel_mid_l = mid_power_l / total_power_l
        rel_mid_r = mid_power_r / total_power_r

        # Asymmetry in spectral characteristics
        asym = np.abs(rel_mid_l - rel_mid_r)

        # Fill channels for this window
        spectral_l_channel[start:end] = rel_mid_l
        spectral_r_channel[start:end] = rel_mid_r
        asym_channel[start:end] = asym

    return spectral_l_channel, spectral_r_channel, asym_channel


def add_fft_channels(baseline_items):
    """Add FFT frequency domain channels to baseline."""
    enhanced_items = []
    for item in baseline_items:
        data = item['data']
        spectral_l, spectral_r, asym = compute_fft_features(data[:, :6])
        enhanced_data = np.column_stack([data, spectral_l, spectral_r, asym])
        new_item = item.copy()
        new_item['data'] = enhanced_data
        enhanced_items.append(new_item)
    return enhanced_items


def train_model_cv(items, num_channels, trial_seed):
    """Train model with cross-validation."""
    set_all_seeds(trial_seed)

    # Subsample
    subsampled_items = subsample_data(items, SUBSAMPLE_FACTOR)

    # Calculate target sequence length
    seq_lens = [item['data'].shape[0] for item in subsampled_items]
    target_seq_len = int(np.percentile(seq_lens, TARGET_SEQ_LEN_PERCENTILE))

    # Prepare labels and patient IDs
    labels = np.array([item['label'] for item in subsampled_items])
    patient_ids = np.array([item['patient_id'] for item in subsampled_items])

    # Cross-validation
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=trial_seed)
    fold_accs = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(patient_ids, labels)):
        train_items = [subsampled_items[i] for i in train_idx]
        val_items = [subsampled_items[i] for i in val_idx]

        # Prepare sequences
        X_train = [item['data'] for item in train_items]
        y_train = [item['label'] for item in train_items]
        X_val = [item['data'] for item in val_items]
        y_val = [item['label'] for item in val_items]

        # Standardize
        scaler = SaccadeStandardScaler()
        scaler.fit(train_items)
        X_train_scaled = [scaler.transform(seq) for seq in X_train]
        X_val_scaled = [scaler.transform(seq) for seq in X_val]

        # Pad/truncate
        X_train_padded = np.array([
            seq[:target_seq_len] if len(seq) >= target_seq_len
            else np.vstack([seq, np.zeros((target_seq_len - len(seq), num_channels))])
            for seq in X_train_scaled
        ])
        X_val_padded = np.array([
            seq[:target_seq_len] if len(seq) >= target_seq_len
            else np.vstack([seq, np.zeros((target_seq_len - len(seq), num_channels))])
            for seq in X_val_scaled
        ])

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_padded)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val_padded)
        y_val_tensor = torch.LongTensor(y_val)

        # Datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Model
        model = SaccadeRNN_Small(
            input_dim=num_channels,
            hidden_dim=128,
            output_dim=2,
            n_layers=2,
            seq_len=target_seq_len,
            dropout_prob=0.3
        ).to(DEVICE)

        # Class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        early_stopper = EarlyStopper(patience=EARLY_STOPPING_PATIENCE, min_delta=0.001)

        # Training loop
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc, _, _ = evaluate_epoch(model, val_loader, criterion, DEVICE)

            if early_stopper.early_stop(val_loss):
                break

        # Final validation accuracy
        _, final_val_acc, _, _ = evaluate_epoch(model, val_loader, criterion, DEVICE)
        fold_accs.append(final_val_acc)

    return np.mean(fold_accs), np.std(fold_accs)


def main():
    """Main experiment runner for FFT feature only."""
    print("="*80)
    print("Experiment 16b: FFT Frequency Feature Test")
    print("Testing Fourier transform based frequency features")
    print("="*80)

    # Create results directory
    create_results_directory(RESULTS_DIR)

    # Load data
    print("\n" + "="*50)
    print("Loading Data...")
    print("="*50)

    with open(os.devnull, 'w') as f_null:
        raw_items = load_raw_sequences_and_labels(
            BASE_DIR, BINARY_CLASS_DEFINITIONS, FEATURE_COLUMNS,
            CSV_ENCODING, CSV_SEPARATOR, MIN_SEQ_LEN_THRESHOLD, f_null
        )

    binary_items = prepare_binary_data(raw_items)
    baseline_items = add_baseline_channels(binary_items)

    print(f"Data loaded: {len(binary_items)} sequences")
    print(f"HC: {sum(1 for x in binary_items if x['label']==0)}, MG: {sum(1 for x in binary_items if x['label']==1)}")

    # Prepare FFT dataset
    print("\n" + "="*50)
    print("Preparing FFT Features...")
    print("="*50)
    fft_items = add_fft_channels(baseline_items)
    print(f"  FFT features added: 14 + 3 = 17 channels")

    # Train baseline
    print("\n" + "="*50)
    print(f"Training Baseline ({NUM_TRIALS} trials, {NUM_FOLDS}-fold CV)...")
    print("="*50)

    baseline_results = []
    for trial in range(NUM_TRIALS):
        trial_seed = RANDOM_STATE + trial
        print(f"  Baseline Trial {trial+1}/{NUM_TRIALS} (seed={trial_seed})...", end=" ", flush=True)
        mean_acc, std_acc = train_model_cv(baseline_items, num_channels=14, trial_seed=trial_seed)
        baseline_results.append(mean_acc)
        print(f"{mean_acc:.4f} +/- {std_acc:.4f}")

    baseline_mean = np.mean(baseline_results)
    baseline_std = np.std(baseline_results)
    print(f"\n  Baseline Overall: {baseline_mean:.4f} +/- {baseline_std:.4f}")

    # Train FFT model
    print("\n" + "="*50)
    print(f"Training FFT Feature Model ({NUM_TRIALS} trials, {NUM_FOLDS}-fold CV)...")
    print("="*50)

    fft_results = []
    for trial in range(NUM_TRIALS):
        trial_seed = RANDOM_STATE + trial
        print(f"  FFT Trial {trial+1}/{NUM_TRIALS} (seed={trial_seed})...", end=" ", flush=True)
        mean_acc, std_acc = train_model_cv(fft_items, num_channels=17, trial_seed=trial_seed)
        fft_results.append(mean_acc)
        print(f"{mean_acc:.4f} +/- {std_acc:.4f}")

    fft_mean = np.mean(fft_results)
    fft_std = np.std(fft_results)
    delta = fft_mean - baseline_mean
    delta_pct = delta * 100

    print(f"\n  FFT Overall: {fft_mean:.4f} +/- {fft_std:.4f}")
    print(f"  Delta: {delta:+.4f} ({delta_pct:+.2f}%)")

    # Write results
    results_path = os.path.join(RESULTS_DIR, 'exp16b_fft_results.txt')
    with open(results_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXPERIMENT 16b: FFT FREQUENCY FEATURE RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  - Epochs: {EPOCHS}\n")
        f.write(f"  - Folds: {NUM_FOLDS}\n")
        f.write(f"  - Trials: {NUM_TRIALS}\n")
        f.write(f"  - Random State: {RANDOM_STATE}\n")
        f.write(f"  - Early Stopping Patience: {EARLY_STOPPING_PATIENCE}\n")
        f.write(f"  - FFT Window Size: 60 samples (500ms)\n")
        f.write(f"  - FFT Hop Size: 30 samples (250ms)\n")
        f.write(f"  - Tremor Band: 2-6 Hz\n")
        f.write("\n")
        f.write(f"Results:\n")
        f.write(f"  Baseline (14 channels):  {baseline_mean:.4f} +/- {baseline_std:.4f}\n")
        f.write(f"  + FFT (17 channels):     {fft_mean:.4f} +/- {fft_std:.4f}\n")
        f.write(f"  Delta:                   {delta:+.4f} ({delta_pct:+.2f}%)\n")
        f.write("="*80 + "\n")

    print("\n" + "="*80)
    print("Experiment 16b completed!")
    print(f"Results saved to: {results_path}")
    print("\n" + "="*40)
    print("SUMMARY")
    print("="*40)
    print(f"  Baseline:     {baseline_mean*100:.2f}% +/- {baseline_std*100:.2f}%")
    print(f"  + FFT:        {fft_mean*100:.2f}% +/- {fft_std*100:.2f}%")
    print(f"  Improvement:  {delta_pct:+.2f}%")
    print("="*80)


if __name__ == "__main__":
    main()
