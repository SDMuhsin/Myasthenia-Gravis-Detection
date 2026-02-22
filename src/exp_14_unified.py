#!/usr/bin/env python3
"""
Experiment 14 Unified: Feature Discriminatory Power Study
Binary Classification (HC vs MG including Probable MG)

Improvements:
1. Train baseline ONCE, reuse for all feature comparisons
2. Set ALL random seeds (numpy, torch, torch.cuda)
3. Use 5-fold CV instead of 3-fold
4. Run 3 independent trials per experiment
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
import pandas as pd

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
RESULTS_DIR = './results/exp_14_unified'
RANDOM_STATE = 42

# Deep Learning Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUBSAMPLE_FACTOR = 10
TARGET_SEQ_LEN_PERCENTILE = 90
EPOCHS = 30  # Reduced from 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 8  # Reduced from 10
NUM_FOLDS = 5  # Increased from 3
NUM_TRIALS = 3  # Multiple runs

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

def add_peak_velocity_channels(baseline_items):
    """Add 4 peak velocity channels to 14 baseline = 18 total."""
    enhanced_items = []
    window_size = 10

    for item in baseline_items:
        data = item['data']

        lh_vel = np.abs(data[:, 6])
        rh_vel = np.abs(data[:, 7])
        lv_vel = np.abs(data[:, 8])
        rv_vel = np.abs(data[:, 9])

        lh_peak = pd.Series(lh_vel).rolling(window=window_size, min_periods=1).max().values
        rh_peak = pd.Series(rh_vel).rolling(window=window_size, min_periods=1).max().values
        lv_peak = pd.Series(lv_vel).rolling(window=window_size, min_periods=1).max().values
        rv_peak = pd.Series(rv_vel).rolling(window=window_size, min_periods=1).max().values

        enhanced_data = np.column_stack([data, lh_peak, rh_peak, lv_peak, rv_peak])

        new_item = item.copy()
        new_item['data'] = enhanced_data
        enhanced_items.append(new_item)

    return enhanced_items

def add_disconjugacy_channels(baseline_items):
    """Add 6 disconjugacy channels to 14 baseline = 20 total."""
    enhanced_items = []
    window_size = 20

    for item in baseline_items:
        data = item['data']

        lh = data[:, 0]
        rh = data[:, 1]
        lv = data[:, 2]
        rv = data[:, 3]
        lh_vel = data[:, 6]
        rh_vel = data[:, 7]
        lv_vel = data[:, 8]
        rv_vel = data[:, 9]

        h_pos_diff = np.abs(lh - rh)
        v_pos_diff = np.abs(lv - rv)
        h_vel_diff = np.abs(lh_vel - rh_vel)
        v_vel_diff = np.abs(lv_vel - rv_vel)

        h_corr = pd.Series(lh).rolling(window=window_size, min_periods=1).corr(pd.Series(rh)).fillna(0).values
        v_corr = pd.Series(lv).rolling(window=window_size, min_periods=1).corr(pd.Series(rv)).fillna(0).values

        h_corr = np.nan_to_num(h_corr, nan=0.0, posinf=1.0, neginf=-1.0)
        v_corr = np.nan_to_num(v_corr, nan=0.0, posinf=1.0, neginf=-1.0)

        enhanced_data = np.column_stack([
            data,
            h_pos_diff, v_pos_diff,
            h_vel_diff, v_vel_diff,
            h_corr, v_corr
        ])

        new_item = item.copy()
        new_item['data'] = enhanced_data
        enhanced_items.append(new_item)

    return enhanced_items

def add_fatigue_channels(baseline_items):
    """Add 4 fatigue channels to 14 baseline = 18 total."""
    enhanced_items = []

    for item in baseline_items:
        data = item['data']

        errorh_l = data[:, 10]
        errorh_r = data[:, 11]
        errorv_l = data[:, 12]
        errorv_r = data[:, 13]
        lh_vel = data[:, 6]
        rh_vel = data[:, 7]
        lv_vel = data[:, 8]
        rv_vel = data[:, 9]

        error_h = (np.abs(errorh_l) + np.abs(errorh_r)) / 2
        error_v = (np.abs(errorv_l) + np.abs(errorv_r)) / 2
        cumulative_error_h = np.cumsum(error_h)
        cumulative_error_v = np.cumsum(error_v)

        vel_h = (np.abs(lh_vel) + np.abs(rh_vel)) / 2
        vel_v = (np.abs(lv_vel) + np.abs(rv_vel)) / 2

        init_len = max(1, len(vel_h) // 10)
        init_vel_h = np.mean(vel_h[:init_len]) if init_len > 0 else 1.0
        init_vel_v = np.mean(vel_v[:init_len]) if init_len > 0 else 1.0

        init_vel_h = init_vel_h if init_vel_h > 0 else 1.0
        init_vel_v = init_vel_v if init_vel_v > 0 else 1.0

        velocity_decay_h = vel_h / init_vel_h
        velocity_decay_v = vel_v / init_vel_v

        enhanced_data = np.column_stack([
            data,
            cumulative_error_h,
            cumulative_error_v,
            velocity_decay_h,
            velocity_decay_v
        ])

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
    """Main experiment runner."""
    print("="*80)
    print("Starting Experiment 14 Unified: Improved Feature Study")
    print("Improvements: Shared baseline, all seeds set, 5-fold CV, 3 trials")
    print("="*80)

    # Create results directory
    create_results_directory(RESULTS_DIR)

    # Load data ONCE
    print("\n" + "="*50)
    print("Loading Data (ONCE for all experiments)...")
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

    # Prepare enhanced datasets
    print("\n" + "="*50)
    print("Preparing Enhanced Feature Sets...")
    print("="*50)

    peak_vel_items = add_peak_velocity_channels(baseline_items)
    disconjugacy_items = add_disconjugacy_channels(baseline_items)
    fatigue_items = add_fatigue_channels(baseline_items)

    print("✓ Peak Velocity (18 channels)")
    print("✓ Disconjugacy (20 channels)")
    print("✓ Fatigue (18 channels)")

    # Train baseline ONCE with multiple trials
    print("\n" + "="*50)
    print(f"Training SHARED Baseline ({NUM_TRIALS} trials, {NUM_FOLDS}-fold CV)...")
    print("="*50)

    baseline_results = []
    for trial in range(NUM_TRIALS):
        trial_seed = RANDOM_STATE + trial
        print(f"  Baseline Trial {trial+1}/{NUM_TRIALS} (seed={trial_seed})...", end=" ")
        mean_acc, std_acc = train_model_cv(baseline_items, num_channels=14, trial_seed=trial_seed)
        baseline_results.append(mean_acc)
        print(f"{mean_acc:.4f} ± {std_acc:.4f}")

    baseline_mean = np.mean(baseline_results)
    baseline_std = np.std(baseline_results)
    print(f"\n  Baseline Overall: {baseline_mean:.4f} ± {baseline_std:.4f}")

    # Train enhanced models
    results = {}

    for feature_name, feature_items, num_channels in [
        ("Peak Velocity", peak_vel_items, 18),
        ("Disconjugacy", disconjugacy_items, 20),
        ("Fatigue", fatigue_items, 18)
    ]:
        print("\n" + "="*50)
        print(f"Training {feature_name} ({NUM_TRIALS} trials, {NUM_FOLDS}-fold CV)...")
        print("="*50)

        enhanced_results = []
        for trial in range(NUM_TRIALS):
            trial_seed = RANDOM_STATE + trial
            print(f"  {feature_name} Trial {trial+1}/{NUM_TRIALS} (seed={trial_seed})...", end=" ")
            mean_acc, std_acc = train_model_cv(feature_items, num_channels=num_channels, trial_seed=trial_seed)
            enhanced_results.append(mean_acc)
            print(f"{mean_acc:.4f} ± {std_acc:.4f}")

        enhanced_mean = np.mean(enhanced_results)
        enhanced_std = np.std(enhanced_results)
        delta = enhanced_mean - baseline_mean
        delta_pct = delta * 100

        print(f"\n  {feature_name} Overall: {enhanced_mean:.4f} ± {enhanced_std:.4f}")
        print(f"  Delta: {delta:+.4f} ({delta_pct:+.2f}%)")

        results[feature_name] = {
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'enhanced_mean': enhanced_mean,
            'enhanced_std': enhanced_std,
            'delta': delta,
            'delta_pct': delta_pct
        }

    # Write summary
    summary_path = os.path.join(RESULTS_DIR, 'unified_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXPERIMENT 14 UNIFIED SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Configuration:\n")
        f.write(f"  - Epochs: {EPOCHS}\n")
        f.write(f"  - Folds: {NUM_FOLDS}\n")
        f.write(f"  - Trials: {NUM_TRIALS}\n")
        f.write(f"  - Random State: {RANDOM_STATE}\n")
        f.write(f"  - Early Stopping Patience: {EARLY_STOPPING_PATIENCE}\n")
        f.write("\n")
        f.write(f"Shared Baseline (14 channels):\n")
        f.write(f"  Accuracy: {baseline_mean:.4f} ± {baseline_std:.4f}\n")
        f.write("\n")

        for feature_name in ["Peak Velocity", "Disconjugacy", "Fatigue"]:
            res = results[feature_name]
            f.write(f"{feature_name} Features:\n")
            f.write(f"  Enhanced Accuracy: {res['enhanced_mean']:.4f} ± {res['enhanced_std']:.4f}\n")
            f.write(f"  Delta: {res['delta']:+.4f} ({res['delta_pct']:+.2f}%)\n")
            f.write("\n")

        # Ranking
        f.write("Ranking by Delta:\n")
        sorted_features = sorted(results.items(), key=lambda x: x[1]['delta'], reverse=True)
        for rank, (name, res) in enumerate(sorted_features, 1):
            f.write(f"  {rank}. {name}: {res['delta_pct']:+.2f}%\n")
        f.write("="*80 + "\n")

    print("\n" + "="*80)
    print("Experiment 14 Unified completed successfully!")
    print(f"Results saved to: {summary_path}")
    print("\nFinal Ranking:")
    for rank, (name, res) in enumerate(sorted_features, 1):
        print(f"  {rank}. {name}: {res['delta_pct']:+.2f}%")
    print("="*80)

if __name__ == "__main__":
    main()
