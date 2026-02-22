#!/usr/bin/env python3
"""
Experiment 14A: Peak Velocity Feature Discriminatory Power Study
Binary Classification (HC vs MG including Probable MG)

Objective: Assess discriminatory power of peak velocity features by comparing:
- Baseline: BiGRU+Attention on original 14 channels
- Enhanced: BiGRU+Attention on 14 + 4 peak velocity channels (18 total)

Peak velocity hypothesis: MG patients may have reduced peak velocities due to
neuromuscular weakness affecting rapid eye movements.

This experiment adds peak velocity as ADDITIONAL CHANNELS to the raw time series.
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
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.data_loading import load_raw_sequences_and_labels
from utils.modeling import create_results_directory
from utils.deep_learning import (SaccadeRNN_Small, SaccadeStandardScaler, EarlyStopper,
                                calculate_class_weights, SaccadeDataset, subsample_data)

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
RESULTS_DIR = './results/exp_14a'
EXP_PREFIX = 'EXP_14A_'
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
HIDDEN_DIM = 128
N_LAYERS = 2
DROPOUT = 0.3

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.discriminant_analysis')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

def prepare_binary_data(raw_items, f_out):
    """Prepare data for binary classification by combining MG classes."""
    f_out.write("="*80 + "\n")
    f_out.write("Phase: Binary Classification Data Preparation\n")
    f_out.write("="*80 + "\n")
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

    f_out.write(f"Binary classification data prepared:\n")
    f_out.write(f"  HC (Healthy Control): {class_counts['HC']} samples\n")
    f_out.write(f"  MG (Definite + Probable): {class_counts['MG']} samples\n")
    f_out.write(f"  Total: {len(binary_items)} samples\n")
    f_out.write("-" * 80 + "\n\n")

    print(f"Binary data prepared: HC={class_counts['HC']}, MG={class_counts['MG']}")
    return binary_items

def add_baseline_channels(binary_items, f_out):
    """
    Add baseline engineered channels to raw data:
    - Original 6 channels: LH, RH, LV, RV, TargetH, TargetV
    - Add 4 velocity channels: LH_Vel, RH_Vel, LV_Vel, RV_Vel
    - Add 4 error channels: ErrorH_L, ErrorH_R, ErrorV_L, ErrorV_R
    Total: 14 channels (baseline)
    """
    f_out.write("="*80 + "\n")
    f_out.write("Phase: Adding Baseline Engineered Channels\n")
    f_out.write("="*80 + "\n")
    print("\n" + "="*50 + "\nAdding Baseline Channels...\n" + "="*50)

    enhanced_items = []

    for item in tqdm(binary_items, desc="  Adding baseline channels"):
        data = item['data']
        df = pd.DataFrame(data, columns=FEATURE_COLUMNS)

        # Add velocity channels
        lh_vel = df['LH'].diff().fillna(0).values
        rh_vel = df['RH'].diff().fillna(0).values
        lv_vel = df['LV'].diff().fillna(0).values
        rv_vel = df['RV'].diff().fillna(0).values

        # Add error channels
        errorh_l = (df['LH'] - df['TargetH']).values
        errorh_r = (df['RH'] - df['TargetH']).values
        errorv_l = (df['LV'] - df['TargetV']).values
        errorv_r = (df['RV'] - df['TargetV']).values

        # Stack all channels
        enhanced_data = np.column_stack([
            data,  # Original 6 channels
            lh_vel, rh_vel, lv_vel, rv_vel,  # 4 velocity channels
            errorh_l, errorh_r, errorv_l, errorv_r  # 4 error channels
        ])

        enhanced_item = item.copy()
        enhanced_item['data'] = enhanced_data.astype(np.float32)
        enhanced_items.append(enhanced_item)

    f_out.write(f"Added baseline channels: 6 original + 4 velocities + 4 errors = 14 total channels\n")
    f_out.write(f"Enhanced {len(enhanced_items)} sequences\n\n")

    return enhanced_items

def add_peak_velocity_channels(binary_items, f_out):
    """
    Add peak velocity channels to baseline data:
    - Baseline 14 channels (from add_baseline_channels)
    - Add 4 peak velocity channels: LH_PeakVel, RH_PeakVel, LV_PeakVel, RV_PeakVel
    Total: 18 channels

    Peak velocity at each timepoint = running maximum of absolute velocity in a sliding window
    """
    f_out.write("="*80 + "\n")
    f_out.write("Phase: Adding Peak Velocity Channels\n")
    f_out.write("="*80 + "\n")
    print("\n" + "="*50 + "\nAdding Peak Velocity Channels...\n" + "="*50)

    enhanced_items = []
    window_size = 10  # Rolling window for peak velocity

    for item in tqdm(binary_items, desc="  Adding peak velocity"):
        data = item['data']  # Already has 14 channels from baseline

        # Extract velocity channels (indices 6-9 in the 14-channel data)
        lh_vel = np.abs(data[:, 6])
        rh_vel = np.abs(data[:, 7])
        lv_vel = np.abs(data[:, 8])
        rv_vel = np.abs(data[:, 9])

        # Calculate running maximum (peak velocity) using pandas rolling
        lh_peak = pd.Series(lh_vel).rolling(window=window_size, min_periods=1).max().values
        rh_peak = pd.Series(rh_vel).rolling(window=window_size, min_periods=1).max().values
        lv_peak = pd.Series(lv_vel).rolling(window=window_size, min_periods=1).max().values
        rv_peak = pd.Series(rv_vel).rolling(window=window_size, min_periods=1).max().values

        # Stack with existing channels
        enhanced_data = np.column_stack([
            data,  # Existing 14 channels
            lh_peak, rh_peak, lv_peak, rv_peak  # 4 new peak velocity channels
        ])

        enhanced_item = item.copy()
        enhanced_item['data'] = enhanced_data.astype(np.float32)
        enhanced_items.append(enhanced_item)

    f_out.write(f"Added peak velocity channels: 14 baseline + 4 peak velocities = 18 total channels\n")
    f_out.write(f"Enhanced {len(enhanced_items)} sequences\n\n")

    return enhanced_items

def run_bigru_attention(items, num_channels, model_suffix, results_dir, f_out):
    """Train and evaluate BiGRU+Attention model."""
    f_out.write("\n" + "="*80 + "\n")
    f_out.write(f"Phase: BiGRU+Attention Training ({model_suffix})\n")
    f_out.write("="*80 + "\n")
    print(f"\n" + "="*50 + f"\nTraining BiGRU+Attention ({model_suffix})...\n" + "="*50)

    # Subsample for faster training
    subsampled_items = subsample_data(items, SUBSAMPLE_FACTOR)
    f_out.write(f"Subsampled data: {len(subsampled_items)} samples (factor: {SUBSAMPLE_FACTOR}x)\n")
    print(f"Using {len(subsampled_items)} subsampled sequences")

    # Determine target sequence length
    lengths = [item['data'].shape[0] for item in subsampled_items]
    target_seq_len = int(np.percentile(lengths, TARGET_SEQ_LEN_PERCENTILE))
    f_out.write(f"Target sequence length ({TARGET_SEQ_LEN_PERCENTILE}th percentile): {target_seq_len}\n")
    f_out.write(f"Number of channels: {num_channels}\n\n")

    # Prepare data arrays
    X = np.array(subsampled_items, dtype=object)
    y = np.array([item['label'] for item in subsampled_items])

    # Label mapping for binary classification
    label_map = {0: 0, 1: 1}  # HC: 0, MG: 1

    # Cross-validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    fold_accuracies = []
    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        f_out.write(f"\nFold {fold+1}/3:\n")
        print(f"  Fold {fold+1}/3")

        train_items = X[train_idx].tolist()
        val_items = X[val_idx].tolist()

        # Prepare data scaling
        scaler = SaccadeStandardScaler()
        scaler.fit(train_items)

        # Create datasets
        train_dataset = SaccadeDataset(train_items, target_seq_len, num_channels, label_map, scaler)
        val_dataset = SaccadeDataset(val_items, target_seq_len, num_channels, label_map, scaler)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize model
        model = SaccadeRNN_Small(
            input_dim=num_channels,
            hidden_dim=HIDDEN_DIM,
            output_dim=2,  # Binary classification
            n_layers=N_LAYERS,
            seq_len=target_seq_len,
            dropout_prob=DROPOUT
        ).to(DEVICE)

        # Calculate class weights
        class_weights = calculate_class_weights(train_items, label_map).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        early_stopper = EarlyStopper(patience=EARLY_STOPPING_PATIENCE)

        # Training loop
        best_val_acc = 0.0

        for epoch in range(EPOCHS):
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()

            train_acc = train_correct / train_total if train_total > 0 else 0

            # Validate
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_preds = []
            val_true = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += batch_y.size(0)
                    val_correct += predicted.eq(batch_y).sum().item()

                    val_preds.extend(predicted.cpu().numpy())
                    val_true.extend(batch_y.cpu().numpy())

            val_acc = val_correct / val_total if val_total > 0 else 0

            scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if early_stopper.early_stop(val_loss):
                f_out.write(f"  Early stopping at epoch {epoch+1}\n")
                break

        fold_accuracies.append(best_val_acc)
        all_y_true.extend(val_true)
        all_y_pred.extend(val_preds)
        f_out.write(f"  Fold {fold+1} best validation accuracy: {best_val_acc:.4f}\n")

        # Clean up GPU memory
        del model, optimizer, scheduler, train_dataset, val_dataset, train_loader, val_loader
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Calculate overall metrics
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)

    f_out.write(f"\nBiGRU+Attention ({model_suffix}) Average Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})\n")
    print(f"  BiGRU+Attention ({model_suffix}): {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")

    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
    f_out.write(f"\nConfusion Matrix ({model_suffix}):\n{cm}\n")

    report = classification_report(all_y_true, all_y_pred, target_names=ORDERED_CLASS_NAMES, labels=[0, 1])
    f_out.write(f"\nClassification Report ({model_suffix}):\n{report}\n")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ORDERED_CLASS_NAMES, yticklabels=ORDERED_CLASS_NAMES)
    plt.title(f'Confusion Matrix: BiGRU+Attention ({model_suffix})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'{EXP_PREFIX}BiGRU_Attention_{model_suffix}_confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()
    f_out.write(f"Confusion matrix plot saved to: {plot_path}\n")

    return mean_accuracy, std_accuracy

def main():
    """Main execution function."""
    print("="*80)
    print("Starting Experiment 14A: Peak Velocity Feature Study")
    print("Binary Classification (HC vs MG including Probable MG)")
    print("Using BiGRU+Attention on Raw Sequences")
    print("="*80)

    # Set random seeds
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_STATE)

    create_results_directory(RESULTS_DIR)
    summary_filepath = os.path.join(RESULTS_DIR, NUMERICAL_SUMMARY_FILENAME)

    with open(summary_filepath, 'w', encoding='utf-8') as f_report:
        f_report.write("="*80 + "\n")
        f_report.write("Experiment 14A: Peak Velocity Feature Discriminatory Power Study\n")
        f_report.write("="*80 + "\n")
        f_report.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_report.write(f"Base Directory: {BASE_DIR}\n")
        f_report.write(f"Classes: {ORDERED_CLASS_NAMES} (Binary Classification)\n")
        f_report.write(f"Device: {DEVICE}\n")
        f_report.write(f"Random State: {RANDOM_STATE}\n")
        f_report.write(f"Model: BiGRU+Attention (SaccadeRNN_Small from Experiment 13A)\n")
        f_report.write(f"Approach: Add peak velocity as additional channels to raw sequences\n")
        f_report.write("="*80 + "\n\n")

        # 1. Load and Process Data
        raw_items_list = load_raw_sequences_and_labels(
            BASE_DIR, BINARY_CLASS_DEFINITIONS, FEATURE_COLUMNS,
            CSV_ENCODING, CSV_SEPARATOR, MIN_SEQ_LEN_THRESHOLD, f_report
        )

        if not raw_items_list:
            f_report.write("\nCRITICAL: No data loaded. Experiments cannot proceed.\n")
            print("CRITICAL: No data loaded. Exiting.")
            return

        # 2. Prepare Binary Classification Data
        binary_items = prepare_binary_data(raw_items_list, f_report)

        # 3. Add Baseline Engineered Channels (14 channels total)
        baseline_items = add_baseline_channels(binary_items, f_report)

        # 4. Run Baseline Model (14 channels)
        baseline_acc, baseline_std = run_bigru_attention(
            baseline_items, 14, "Baseline_14ch", RESULTS_DIR, f_report
        )

        # 5. Add Peak Velocity Channels (18 channels total)
        enhanced_items = add_peak_velocity_channels(baseline_items, f_report)

        # 6. Run Enhanced Model (18 channels)
        enhanced_acc, enhanced_std = run_bigru_attention(
            enhanced_items, 18, "Enhanced_PeakVel_18ch", RESULTS_DIR, f_report
        )

        # 7. Calculate Delta
        delta_acc = enhanced_acc - baseline_acc

        f_report.write("\n" + "="*80 + "\n")
        f_report.write("EXPERIMENT 14A FINAL RESULTS SUMMARY\n")
        f_report.write("="*80 + "\n")
        f_report.write(f"Baseline Model (14 channels: 6 raw + 4 velocities + 4 errors):\n")
        f_report.write(f"  Accuracy: {baseline_acc:.4f} (+/- {baseline_std:.4f})\n\n")
        f_report.write(f"Enhanced Model (18 channels: 14 baseline + 4 peak velocities):\n")
        f_report.write(f"  Accuracy: {enhanced_acc:.4f} (+/- {enhanced_std:.4f})\n\n")
        f_report.write(f"Delta Improvement: {delta_acc:+.4f} ({delta_acc*100:+.2f}%)\n")

        if delta_acc > 0.01:  # >1% improvement
            f_report.write("\n✓ POSITIVE RESULT: Peak velocity features improve classification accuracy\n")
        elif delta_acc < -0.01:  # >1% decrease
            f_report.write("\n✗ NEGATIVE RESULT: Peak velocity features decrease classification accuracy\n")
        else:
            f_report.write("\n= NEUTRAL RESULT: Peak velocity features have minimal effect on accuracy\n")

        f_report.write("\n" + "="*80 + "\n")
        f_report.write("End of Experiment 14A Report\n")
        f_report.write("="*80 + "\n")

    print(f"\nExperiment 14A completed successfully!")
    print(f"Results saved to: {summary_filepath}")
    print(f"Baseline Accuracy: {baseline_acc:.4f} (+/- {baseline_std:.4f})")
    print(f"Enhanced Accuracy: {enhanced_acc:.4f} (+/- {enhanced_std:.4f})")
    print(f"Delta: {delta_acc:+.4f} ({delta_acc*100:+.2f}%)")
    print("="*80)

if __name__ == '__main__':
    main()
