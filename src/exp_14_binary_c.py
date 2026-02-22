#!/usr/bin/env python3
"""
Experiment 14C: Patient Fatigue Feature Discriminatory Power Study
Binary Classification (HC vs MG including Probable MG)

Objective: Assess discriminatory power of patient fatigue features by comparing:
- Baseline: BiGRU+Attention on raw sequences with 14 channels (6 original + 4 velocities + 4 errors)
- Enhanced: BiGRU+Attention on raw sequences with 18 channels (14 baseline + 4 fatigue)

Fatigue hypothesis: MG patients may show more pronounced fatigue (performance degradation
over time within a recording session) compared to healthy controls.

Key fatigue features (temporal degradation indicators):
1. Cumulative error: Running sum of absolute error over time
2. Velocity decay: Ratio of current velocity to initial velocity
3. Error trend: Linear trend coefficient of error magnitude
4. Performance degradation: Rolling mean error (increasing = degradation)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from scipy.stats import linregress

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.data_loading import load_raw_sequences_and_labels
from utils.modeling import create_results_directory
from utils.deep_learning import (SaccadeRNN_Small, SaccadeStandardScaler, EarlyStopper,
                                subsample_data, train_epoch, evaluate_epoch)
from sklearn.utils.class_weight import compute_class_weight

# --- Configuration ---
BASE_DIR = './data'

# Binary classification: HC vs MG (including Probable MG)
BINARY_CLASS_DEFINITIONS = {
    'HC': {'path': 'Healthy control', 'label': 0},
    'MG': {'path': 'Definite MG', 'label': 1},
    'Probable_MG': {'path': 'Probable MG', 'label': 1},
}

CLASS_MAPPING = {name: details['label'] for name, details in BINARY_CLASS_DEFINITIONS.items()}
ORDERED_CLASS_NAMES = ['HC', 'MG']
MODEL_CLASS_LABELS = [0, 1]

# --- Core Parameters ---
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']
CSV_ENCODING = 'utf-16-le'
CSV_SEPARATOR = ','
MIN_SEQ_LEN_THRESHOLD = 50
RESULTS_DIR = './results/exp_14c'
EXP_PREFIX = 'EXP_14C_'
NUMERICAL_SUMMARY_FILENAME = f'{EXP_PREFIX}numerical_summary.txt'
RANDOM_STATE = 42

# Deep Learning Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUBSAMPLE_FACTOR = 10
TARGET_SEQ_LEN_PERCENTILE = 90
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10
NUM_FOLDS = 3

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

def prepare_binary_data(raw_items, f_out):
    """Prepare data for binary classification by combining MG classes."""
    print("\n" + "="*50 + "\nPreparing Binary Classification Data...\n" + "="*50)

    binary_items = []
    class_counts = {'HC': 0, 'MG': 0}

    for item in raw_items:
        if item['class_name'] in ['MG', 'Probable_MG']:
            new_item = item.copy()
            new_item['class_name'] = 'MG'
            new_item['label'] = 1
            binary_items.append(new_item)
            class_counts['MG'] += 1
        elif item['class_name'] == 'HC':
            binary_items.append(item)
            class_counts['HC'] += 1

    print(f"Binary data prepared: HC={class_counts['HC']}, MG={class_counts['MG']}")
    return binary_items

def add_baseline_channels(binary_items, f_out):
    """
    Add 8 engineered channels to 6 raw channels = 14 total.
    Channels: 4 velocities + 4 errors.
    """
    print("\n" + "="*50 + "\nAdding Baseline Channels...\n" + "="*50)

    enhanced_items = []
    for item in tqdm(binary_items, desc="  Adding baseline channels"):
        data = item['data']  # 6 channels: LH, RH, LV, RV, TargetH, TargetV

        # Velocities (numerical derivative)
        lh_vel = np.gradient(data[:, 0])
        rh_vel = np.gradient(data[:, 1])
        lv_vel = np.gradient(data[:, 2])
        rv_vel = np.gradient(data[:, 3])

        # Errors (position - target)
        errorh_l = data[:, 0] - data[:, 4]
        errorh_r = data[:, 1] - data[:, 4]
        errorv_l = data[:, 2] - data[:, 5]
        errorv_r = data[:, 3] - data[:, 5]

        # Stack to 14 channels
        enhanced_data = np.column_stack([
            data,                                      # 6 channels
            lh_vel, rh_vel, lv_vel, rv_vel,           # 4 velocity channels
            errorh_l, errorh_r, errorv_l, errorv_r    # 4 error channels
        ])

        new_item = item.copy()
        new_item['data'] = enhanced_data
        enhanced_items.append(new_item)

    return enhanced_items

def add_fatigue_channels(binary_items, f_out):
    """
    Add 4 fatigue channels to 14 baseline = 18 total.

    Fatigue channels (temporal degradation indicators):
    1. Cumulative_Error_H: Running sum of |horizontal error|
    2. Cumulative_Error_V: Running sum of |vertical error|
    3. Velocity_Decay_H: Ratio of current horizontal velocity to initial
    4. Velocity_Decay_V: Ratio of current vertical velocity to initial
    """
    print("\n" + "="*50 + "\nAdding Fatigue Channels...\n" + "="*50)

    enhanced_items = []

    for item in tqdm(binary_items, desc="  Adding fatigue"):
        data = item['data']  # 14 channels

        # Extract error channels (indices 10-13)
        errorh_l = data[:, 10]
        errorh_r = data[:, 11]
        errorv_l = data[:, 12]
        errorv_r = data[:, 13]

        # Extract velocity channels (indices 6-9)
        lh_vel = data[:, 6]
        rh_vel = data[:, 7]
        lv_vel = data[:, 8]
        rv_vel = data[:, 9]

        # 1. Cumulative error (fatigue accumulation)
        # Average horizontal error
        error_h = (np.abs(errorh_l) + np.abs(errorh_r)) / 2
        error_v = (np.abs(errorv_l) + np.abs(errorv_r)) / 2
        cumulative_error_h = np.cumsum(error_h)
        cumulative_error_v = np.cumsum(error_v)

        # 2. Velocity decay (normalized by initial velocity)
        # Average velocities
        vel_h = (np.abs(lh_vel) + np.abs(rh_vel)) / 2
        vel_v = (np.abs(lv_vel) + np.abs(rv_vel)) / 2

        # Initial velocity (mean of first 10% of sequence)
        init_len = max(1, len(vel_h) // 10)
        init_vel_h = np.mean(vel_h[:init_len]) if init_len > 0 else 1.0
        init_vel_v = np.mean(vel_v[:init_len]) if init_len > 0 else 1.0

        # Avoid division by zero
        init_vel_h = init_vel_h if init_vel_h > 0 else 1.0
        init_vel_v = init_vel_v if init_vel_v > 0 else 1.0

        velocity_decay_h = vel_h / init_vel_h
        velocity_decay_v = vel_v / init_vel_v

        # Stack to 18 channels
        enhanced_data = np.column_stack([
            data,                                      # 14 baseline channels
            cumulative_error_h,                        # Fatigue channel 1
            cumulative_error_v,                        # Fatigue channel 2
            velocity_decay_h,                          # Fatigue channel 3
            velocity_decay_v                           # Fatigue channel 4
        ])

        new_item = item.copy()
        new_item['data'] = enhanced_data
        enhanced_items.append(new_item)

    return enhanced_items

def run_bigru_attention(items, num_channels, model_suffix, results_dir, f_out):
    """
    Train BiGRU+Attention model on sequences with specified number of channels.
    """
    print("\n" + "="*50 + f"\nTraining BiGRU+Attention ({model_suffix})...\n" + "="*50)

    # Subsample
    subsampled_items = subsample_data(items, SUBSAMPLE_FACTOR)
    print(f"Using {len(subsampled_items)} subsampled sequences")

    # Calculate target sequence length
    seq_lens = [item['data'].shape[0] for item in subsampled_items]
    target_seq_len = int(np.percentile(seq_lens, TARGET_SEQ_LEN_PERCENTILE))

    # Prepare labels and patient IDs
    labels = np.array([item['label'] for item in subsampled_items])
    patient_ids = np.array([item['patient_id'] for item in subsampled_items])

    # Cross-validation
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_accs = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(patient_ids, labels)):
        print(f"  Fold {fold_idx + 1}/{NUM_FOLDS}")

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

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print(f"  BiGRU+Attention ({model_suffix}): {mean_acc:.4f} (+/- {std_acc:.4f})")

    return mean_acc, std_acc

def main():
    """Main experiment runner."""
    print("="*80)
    print("Starting Experiment 14C: Patient Fatigue Feature Study")
    print("Binary Classification (HC vs MG including Probable MG)")
    print("Using BiGRU+Attention on Raw Sequences")
    print("="*80)

    # Create results directory
    create_results_directory(RESULTS_DIR)
    summary_path = os.path.join(RESULTS_DIR, NUMERICAL_SUMMARY_FILENAME)

    with open(summary_path, 'w') as f_out:
        # Load data
        raw_items = load_raw_sequences_and_labels(
            BASE_DIR, BINARY_CLASS_DEFINITIONS, FEATURE_COLUMNS,
            CSV_ENCODING, CSV_SEPARATOR, MIN_SEQ_LEN_THRESHOLD, f_out
        )

        # Prepare binary data
        binary_items = prepare_binary_data(raw_items, f_out)

        # Add baseline channels (14 channels)
        baseline_items = add_baseline_channels(binary_items, f_out)

        # Train baseline model
        baseline_acc, baseline_std = run_bigru_attention(
            baseline_items, num_channels=14,
            model_suffix="Baseline_14ch",
            results_dir=RESULTS_DIR, f_out=f_out
        )

        # Add fatigue channels (18 channels)
        enhanced_items = add_fatigue_channels(baseline_items, f_out)

        # Train enhanced model
        enhanced_acc, enhanced_std = run_bigru_attention(
            enhanced_items, num_channels=18,
            model_suffix="Enhanced_Fatigue_18ch",
            results_dir=RESULTS_DIR, f_out=f_out
        )

        # Calculate delta
        delta = enhanced_acc - baseline_acc
        delta_pct = delta * 100

        # Write summary
        f_out.write("\n" + "="*80 + "\n")
        f_out.write("EXPERIMENT 14C SUMMARY\n")
        f_out.write("="*80 + "\n")
        f_out.write(f"Baseline Accuracy (14 channels): {baseline_acc:.4f} (+/- {baseline_std:.4f})\n")
        f_out.write(f"Enhanced Accuracy (18 channels): {enhanced_acc:.4f} (+/- {enhanced_std:.4f})\n")
        f_out.write(f"Delta: {delta:.4f} ({delta_pct:+.2f}%)\n")
        f_out.write(f"\nFatigue channels added (4):\n")
        f_out.write(f"  1. Cumulative_Error_H: Running sum of horizontal errors\n")
        f_out.write(f"  2. Cumulative_Error_V: Running sum of vertical errors\n")
        f_out.write(f"  3. Velocity_Decay_H: Horizontal velocity normalized by initial\n")
        f_out.write(f"  4. Velocity_Decay_V: Vertical velocity normalized by initial\n")
        f_out.write("="*80 + "\n")

        print("\n" + "="*80)
        print("Experiment 14C completed successfully!")
        print(f"Results saved to: {summary_path}")
        print(f"Baseline Accuracy: {baseline_acc:.4f} (+/- {baseline_std:.4f})")
        print(f"Enhanced Accuracy: {enhanced_acc:.4f} (+/- {enhanced_std:.4f})")
        print(f"Delta: {delta:.4f} ({delta_pct:+.2f}%)")
        print("="*80)

if __name__ == "__main__":
    main()
