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
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
from scipy.signal import find_peaks
from scipy import stats

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.data_loading import load_raw_sequences_and_labels, engineer_and_aggregate_features
from utils.modeling import (create_results_directory, train_and_evaluate_single_model, 
                           get_best_statistical_models)
from utils.deep_learning import (get_small_dl_models, SaccadeStandardScaler, EarlyStopper,
                                calculate_class_weights, SaccadeDataset, train_epoch, 
                                evaluate_epoch, plot_loss_curves, plot_dl_confusion_matrix,
                                subsample_data)

# --- Configuration ---
BASE_DIR = './data'

# Binary classification: HC vs MG (including Probable MG) - same as 13c
BINARY_CLASS_DEFINITIONS = {
    'HC': {'path': 'Healthy control', 'label': 0},
    'MG': {'path': 'Definite MG', 'label': 1},
    'Probable_MG': {'path': 'Probable MG', 'label': 1},  # Include Probable MG as MG class
}

CLASS_MAPPING = {name: details['label'] for name, details in BINARY_CLASS_DEFINITIONS.items()}
INV_CLASS_MAPPING = {details['label']: name for name, details in BINARY_CLASS_DEFINITIONS.items()}
ORDERED_CLASS_NAMES = ['HC', 'MG']  # Binary classification
MODEL_CLASS_LABELS = [0, 1]

# --- Core Parameters ---
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']
CSV_ENCODING = 'utf-16-le'
CSV_SEPARATOR = ','
MIN_SEQ_LEN_THRESHOLD = 50
RESULTS_DIR = './results/exp_13d'
EXP_PREFIX = 'EXP_13D_'
NUMERICAL_SUMMARY_FILENAME = f'{EXP_PREFIX}numerical_summary.txt'
RANDOM_STATE = 42

# Deep Learning Parameters - IMPROVED FOR RIGOR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUBSAMPLE_FACTOR = 3  # Reduced subsampling for more data
TARGET_SEQ_LEN_PERCENTILE = 85  # Slightly reduced for memory efficiency
EPOCHS = 100  # Increased for better convergence
BATCH_SIZE = 32  # Larger batch size for stability
LEARNING_RATE = 0.0005  # Lower learning rate for stability
EARLY_STOPPING_PATIENCE = 15  # More patience for convergence

# Saccade Analysis Parameters - IMPROVED FOR RIGOR
SACCADE_COUNTS_TO_TEST = [5, 10, 15, 20, 25, 30, 35, 40]  # All counts up to 40
MIN_SAMPLES_PER_BIN = 30  # Reduced minimum for higher saccade counts
SMOOTHING_WINDOW = 5  # Increased smoothing for better saccade detection
MAX_SEGMENTS_PER_SEQUENCE = 10  # More segments for data augmentation
MIN_SACCADE_SEPARATION_FACTOR = 0.02  # 2% of sequence length minimum separation

# Statistical Rigor Parameters - NEW
N_STATISTICAL_REPEATS = 10  # Multiple runs for statistical models
N_DL_REPEATS = 5  # Multiple runs for deep learning models
CONFIDENCE_LEVEL = 0.95  # For confidence intervals
MIN_SAMPLES_FOR_STATS = 100  # Minimum samples for statistical significance

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.discriminant_analysis')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

def detect_saccades_from_zero_crossings(horizontal_data, smoothing_window=5):
    """
    IMPROVED: Detect saccades from zero crossings of horizontal eye movement data.
    One full cycle (left→right + right→left) is considered 1 saccade.
    
    Args:
        horizontal_data: Array of horizontal eye position data (LH or RH)
        smoothing_window: Window size for smoothing the data
    
    Returns:
        saccade_count: Number of complete saccades detected
        zero_crossings: Indices of zero crossings
        smoothed_data: Smoothed horizontal data used for detection
    """
    # Improved smoothing with Gaussian-like kernel
    if len(horizontal_data) < smoothing_window:
        smoothed_data = horizontal_data.copy()
    else:
        # Use a more sophisticated smoothing kernel
        kernel = np.exp(-0.5 * np.linspace(-2, 2, smoothing_window)**2)
        kernel = kernel / np.sum(kernel)
        smoothed_data = np.convolve(horizontal_data, kernel, mode='same')
    
    # Remove DC offset by centering around mean
    smoothed_data = smoothed_data - np.mean(smoothed_data)
    
    # Find zero crossings in position
    zero_crossings = []
    for i in range(len(smoothed_data) - 1):
        if smoothed_data[i] * smoothed_data[i + 1] < 0:  # Sign change in position
            zero_crossings.append(i + 1)
    
    # Improved filtering of zero crossings
    min_separation = max(10, int(len(smoothed_data) * MIN_SACCADE_SEPARATION_FACTOR))
    filtered_crossings = []
    
    if zero_crossings:
        filtered_crossings.append(zero_crossings[0])
        for crossing in zero_crossings[1:]:
            if crossing - filtered_crossings[-1] >= min_separation:
                # Additional check: ensure significant amplitude change
                prev_crossing = filtered_crossings[-1]
                amplitude_change = abs(np.max(smoothed_data[prev_crossing:crossing]) - 
                                     np.min(smoothed_data[prev_crossing:crossing]))
                if amplitude_change > np.std(smoothed_data) * 0.5:  # Threshold based on signal variability
                    filtered_crossings.append(crossing)
    
    # Count complete saccades (pairs of zero crossings)
    saccade_count = len(filtered_crossings) // 2
    
    return saccade_count, filtered_crossings, smoothed_data

def analyze_saccade_counts(raw_items, f_out):
    """Analyze saccade counts for all sequences and add to items."""
    f_out.write("="*80 + "\n")
    f_out.write("Phase: IMPROVED Saccade Detection and Count Analysis\n")
    f_out.write("="*80 + "\n")
    print("\n" + "="*50 + "\nAnalyzing Saccade Counts (IMPROVED)...\n" + "="*50)
    
    saccade_counts = []
    enhanced_items = []
    
    for item in tqdm(raw_items, desc="  Detecting saccades"):
        # Use left horizontal (LH) data for saccade detection
        lh_data = item['data'][:, 0]  # LH is first column
        
        # Detect saccades with improved algorithm
        saccade_count, zero_crossings, smoothed_data = detect_saccades_from_zero_crossings(
            lh_data, SMOOTHING_WINDOW
        )
        
        # Add saccade information to item
        enhanced_item = item.copy()
        enhanced_item['saccade_count'] = saccade_count
        enhanced_item['zero_crossings'] = zero_crossings
        enhanced_item['smoothed_lh'] = smoothed_data
        
        enhanced_items.append(enhanced_item)
        saccade_counts.append(saccade_count)
    
    # Analyze saccade count distribution
    saccade_counts = np.array(saccade_counts)
    f_out.write(f"IMPROVED Saccade count statistics:\n")
    f_out.write(f"  Mean: {np.mean(saccade_counts):.2f}\n")
    f_out.write(f"  Std: {np.std(saccade_counts):.2f}\n")
    f_out.write(f"  Min: {np.min(saccade_counts)}\n")
    f_out.write(f"  Max: {np.max(saccade_counts)}\n")
    f_out.write(f"  Median: {np.median(saccade_counts):.2f}\n")
    f_out.write(f"  25th percentile: {np.percentile(saccade_counts, 25):.2f}\n")
    f_out.write(f"  75th percentile: {np.percentile(saccade_counts, 75):.2f}\n")
    
    # Count distribution by class
    hc_counts = [item['saccade_count'] for item in enhanced_items if item['label'] == 0]
    mg_counts = [item['saccade_count'] for item in enhanced_items if item['label'] == 1]
    
    f_out.write(f"\nSaccade counts by class:\n")
    f_out.write(f"  HC: Mean={np.mean(hc_counts):.2f}, Std={np.std(hc_counts):.2f}, n={len(hc_counts)}\n")
    f_out.write(f"  MG: Mean={np.mean(mg_counts):.2f}, Std={np.std(mg_counts):.2f}, n={len(mg_counts)}\n")
    
    # Statistical test for difference in saccade counts
    t_stat, p_value = stats.ttest_ind(hc_counts, mg_counts)
    f_out.write(f"\nStatistical test (HC vs MG saccade counts):\n")
    f_out.write(f"  t-statistic: {t_stat:.4f}\n")
    f_out.write(f"  p-value: {p_value:.6f}\n")
    f_out.write(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}\n")
    
    f_out.write("-" * 80 + "\n\n")
    
    return enhanced_items

def create_saccade_verification_plots(enhanced_items, results_dir, f_out, n_examples=9):
    """Create improved plots to verify saccade delimitation."""
    f_out.write("Creating IMPROVED saccade verification plots...\n")
    print("Creating IMPROVED saccade verification plots...")
    
    # Select examples with different saccade counts, ensuring good coverage
    examples = []
    saccade_counts_seen = set()
    
    # Sort items by saccade count to get good distribution
    sorted_items = sorted(enhanced_items, key=lambda x: x['saccade_count'])
    
    # Select examples across the range
    for item in sorted_items:
        count = item['saccade_count']
        if count not in saccade_counts_seen and len(examples) < n_examples:
            examples.append(item)
            saccade_counts_seen.add(count)
    
    # Create subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, item in enumerate(examples[:n_examples]):
        ax = axes[i]
        
        # Original and smoothed data
        lh_data = item['data'][:, 0]
        smoothed_data = item['smoothed_lh']
        zero_crossings = item['zero_crossings']
        
        # Plot original and smoothed data
        time_points = np.arange(len(lh_data))
        ax.plot(time_points, lh_data, 'b-', alpha=0.4, label='Original LH', linewidth=1)
        ax.plot(time_points, smoothed_data, 'r-', label='Smoothed LH', linewidth=2)
        
        # Mark zero crossings
        if zero_crossings:
            ax.scatter(zero_crossings, smoothed_data[zero_crossings], 
                      color='green', s=60, zorder=5, label='Zero Crossings', marker='o')
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Highlight saccade pairs
        if len(zero_crossings) >= 2:
            for j in range(0, len(zero_crossings)-1, 2):
                if j+1 < len(zero_crossings):
                    ax.axvspan(zero_crossings[j], zero_crossings[j+1], 
                             alpha=0.2, color='yellow', label='Saccade' if j == 0 else "")
        
        ax.set_title(f'{item["class_name"]} - {item["saccade_count"]} saccades\n'
                    f'File: {item["filename"][:25]}...', fontsize=10)
        ax.set_xlabel('Time Points')
        ax.set_ylabel('Horizontal Position')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'{EXP_PREFIX}saccade_verification_examples.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    f_out.write(f"IMPROVED saccade verification plot saved to: {plot_path}\n")
    
    # Create comprehensive saccade count distribution plot
    plt.figure(figsize=(16, 12))
    
    hc_counts = [item['saccade_count'] for item in enhanced_items if item['label'] == 0]
    mg_counts = [item['saccade_count'] for item in enhanced_items if item['label'] == 1]
    
    # Plot 1: HC distribution
    plt.subplot(2, 3, 1)
    plt.hist(hc_counts, bins=30, alpha=0.7, label='HC', color='blue', density=True)
    plt.xlabel('Number of Saccades')
    plt.ylabel('Density')
    plt.title('HC Saccade Count Distribution')
    plt.grid(True, alpha=0.3)
    plt.axvline(np.mean(hc_counts), color='blue', linestyle='--', label=f'Mean: {np.mean(hc_counts):.1f}')
    plt.legend()
    
    # Plot 2: MG distribution
    plt.subplot(2, 3, 2)
    plt.hist(mg_counts, bins=30, alpha=0.7, label='MG', color='red', density=True)
    plt.xlabel('Number of Saccades')
    plt.ylabel('Density')
    plt.title('MG Saccade Count Distribution')
    plt.grid(True, alpha=0.3)
    plt.axvline(np.mean(mg_counts), color='red', linestyle='--', label=f'Mean: {np.mean(mg_counts):.1f}')
    plt.legend()
    
    # Plot 3: Combined distribution
    plt.subplot(2, 3, 3)
    plt.hist([hc_counts, mg_counts], bins=30, alpha=0.7, label=['HC', 'MG'], 
             color=['blue', 'red'], density=True)
    plt.xlabel('Number of Saccades')
    plt.ylabel('Density')
    plt.title('Combined Saccade Count Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Box plot
    plt.subplot(2, 3, 4)
    plt.boxplot([hc_counts, mg_counts], labels=['HC', 'MG'])
    plt.ylabel('Number of Saccades')
    plt.title('Saccade Count Box Plot')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Cumulative distribution
    plt.subplot(2, 3, 5)
    hc_sorted = np.sort(hc_counts)
    mg_sorted = np.sort(mg_counts)
    plt.plot(hc_sorted, np.arange(1, len(hc_sorted)+1)/len(hc_sorted), 'b-', label='HC', linewidth=2)
    plt.plot(mg_sorted, np.arange(1, len(mg_sorted)+1)/len(mg_sorted), 'r-', label='MG', linewidth=2)
    plt.xlabel('Number of Saccades')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Sample availability by saccade count
    plt.subplot(2, 3, 6)
    all_counts = hc_counts + mg_counts
    unique_counts, count_freq = np.unique(all_counts, return_counts=True)
    plt.bar(unique_counts, count_freq, alpha=0.7, color='green')
    plt.xlabel('Number of Saccades')
    plt.ylabel('Number of Sequences')
    plt.title('Data Availability by Saccade Count')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    dist_plot_path = os.path.join(results_dir, f'{EXP_PREFIX}saccade_count_distributions.png')
    plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    f_out.write(f"IMPROVED saccade count distribution plot saved to: {dist_plot_path}\n\n")

def segment_sequences_by_saccade_count(enhanced_items, target_saccade_count, f_out):
    """IMPROVED: Segment longer sequences into chunks with specific saccade counts."""
    f_out.write(f"\nIMPROVED segmenting sequences for {target_saccade_count} saccades...\n")
    
    segmented_items = []
    
    for item in enhanced_items:
        if item['saccade_count'] < target_saccade_count:
            continue  # Skip sequences with insufficient saccades
        
        zero_crossings = item['zero_crossings']
        data = item['data']
        
        if len(zero_crossings) < target_saccade_count * 2:
            continue  # Need at least 2 crossings per saccade
        
        # Create segments with exactly target_saccade_count saccades
        segments_created = 0
        
        # Improved segmentation: use sliding window approach for more data
        step_size = max(1, target_saccade_count // 2)  # Overlap segments for more data
        
        for start_idx in range(0, len(zero_crossings) - target_saccade_count * 2 + 1, step_size):
            if segments_created >= MAX_SEGMENTS_PER_SEQUENCE:
                break
                
            # Get start and end indices for this segment
            segment_start = zero_crossings[start_idx]
            segment_end = zero_crossings[start_idx + target_saccade_count * 2 - 1]
            
            # Extract segment data
            segment_data = data[segment_start:segment_end + 1]
            
            if len(segment_data) < 50:  # Minimum sequence length
                continue
            
            # Create new item for this segment
            segment_item = {
                'data': segment_data,
                'label': item['label'],
                'class_name': item['class_name'],
                'filename': f"{item['filename']}_seg_{segments_created}",
                'patient_id': item.get('patient_id', item['filename']),
                'frequency': item.get('frequency', 'unknown'),
                'saccade_count': target_saccade_count,
                'original_filename': item['filename'],
                'segment_start': segment_start,
                'segment_end': segment_end
            }
            
            segmented_items.append(segment_item)
            segments_created += 1
    
    # Count by class
    hc_count = sum(1 for item in segmented_items if item['label'] == 0)
    mg_count = sum(1 for item in segmented_items if item['label'] == 1)
    
    f_out.write(f"  Created {len(segmented_items)} segments with {target_saccade_count} saccades\n")
    f_out.write(f"  HC: {hc_count}, MG: {mg_count}\n")
    f_out.write(f"  Class balance ratio (HC:MG): 1:{mg_count/max(hc_count, 1):.2f}\n")
    
    return segmented_items

def run_statistical_models_with_repeats(master_df, numerical_features, saccade_count, results_dir, f_out):
    """IMPROVED: Run statistical models with multiple repeats for statistical rigor."""
    f_out.write(f"\n--- RIGOROUS Statistical Models for {saccade_count} saccades ---\n")
    print(f"\nRunning RIGOROUS Statistical Models for {saccade_count} saccades...")
    
    X = master_df[numerical_features]
    y = master_df['label']
    
    models = get_best_statistical_models()
    results = {}
    
    for model_name, model in models.items():
        f_out.write(f"\n--- Model: {model_name} (Multiple Runs) ---\n")
        
        # Multiple runs with different random states
        accuracies = []
        
        for run in range(N_STATISTICAL_REPEATS):
            # Use RepeatedStratifiedKFold for more robust validation
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE + run)
            
            fold_accuracies = []
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict and evaluate
                y_pred = model.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                fold_accuracies.append(accuracy)
            
            # Average accuracy for this run
            run_accuracy = np.mean(fold_accuracies)
            accuracies.append(run_accuracy)
        
        # Calculate statistics
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        ci_lower = np.percentile(accuracies, (1 - CONFIDENCE_LEVEL) / 2 * 100)
        ci_upper = np.percentile(accuracies, (1 + CONFIDENCE_LEVEL) / 2 * 100)
        
        results[model_name] = {
            'mean': mean_accuracy,
            'std': std_accuracy,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'all_runs': accuracies
        }
        
        f_out.write(f"  Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}\n")
        f_out.write(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n")
        f_out.write(f"  Individual runs: {[f'{acc:.4f}' for acc in accuracies]}\n")
    
    f_out.write(f"\nRIGOROUS Statistical Models Summary for {saccade_count} saccades:\n")
    for model_name, result in results.items():
        f_out.write(f"  {model_name}: {result['mean']:.4f} ± {result['std']:.4f} "
                   f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]\n")
    
    return results

def run_deep_learning_models_with_repeats(segmented_items, num_features, saccade_count, results_dir, f_out):
    """IMPROVED: Run deep learning models with multiple repeats for statistical rigor."""
    f_out.write(f"\n--- RIGOROUS Deep Learning Models for {saccade_count} saccades ---\n")
    print(f"\nRunning RIGOROUS Deep Learning Models for {saccade_count} saccades...")
    
    # Improved subsampling - less aggressive
    subsampled_items = subsample_data(segmented_items, SUBSAMPLE_FACTOR)
    f_out.write(f"Subsampled data: {len(subsampled_items)} samples (factor: {SUBSAMPLE_FACTOR}x)\n")
    
    if len(subsampled_items) < MIN_SAMPLES_FOR_STATS:
        f_out.write(f"Insufficient subsampled data for rigorous deep learning ({len(subsampled_items)} samples). Skipping.\n")
        return {}
    
    # Determine target sequence length
    lengths = [item['data'].shape[0] for item in subsampled_items]
    target_seq_len = int(np.percentile(lengths, TARGET_SEQ_LEN_PERCENTILE))
    f_out.write(f"Target sequence length ({TARGET_SEQ_LEN_PERCENTILE}th percentile): {target_seq_len}\n")
    
    # Prepare data arrays
    X = np.array(subsampled_items, dtype=object)
    y = np.array([item['label'] for item in subsampled_items])
    
    # Check if we have both classes
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        f_out.write(f"Insufficient class diversity for deep learning (only {unique_labels}). Skipping.\n")
        return {}
    
    # Label mapping for binary classification
    label_map = {0: 0, 1: 1}  # HC: 0, MG: 1
    
    # Get models
    models = get_small_dl_models(num_features, 2, target_seq_len)
    
    dl_results = {}
    
    for model_name, model_class in models.items():
        f_out.write(f"\n--- Deep Learning Model: {model_name} (Multiple Runs) ---\n")
        
        all_run_accuracies = []
        
        # Multiple runs for statistical rigor
        for run in range(N_DL_REPEATS):
            f_out.write(f"\nRun {run+1}/{N_DL_REPEATS}:\n")
            
            # Cross-validation with different random state
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + run)
            fold_accuracies = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                train_items = X[train_idx].tolist()
                val_items = X[val_idx].tolist()
                
                # Prepare data scaling
                scaler = SaccadeStandardScaler()
                scaler.fit(train_items)
                
                # Create datasets
                train_dataset = SaccadeDataset(train_items, target_seq_len, num_features, label_map, scaler)
                val_dataset = SaccadeDataset(val_items, target_seq_len, num_features, label_map, scaler)
                
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
                
                # Initialize model
                model = model_class.to(DEVICE)
                
                # Calculate class weights
                class_weights = calculate_class_weights(train_items, label_map).to(DEVICE)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5)
                early_stopper = EarlyStopper(patience=EARLY_STOPPING_PATIENCE)
                
                # Training loop
                best_val_acc = 0.0
                
                for epoch in range(EPOCHS):
                    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
                    val_loss, val_acc, _, _ = evaluate_epoch(model, val_loader, criterion, DEVICE)
                    
                    scheduler.step(val_loss)
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                    
                    if early_stopper.early_stop(val_loss):
                        break
                
                fold_accuracies.append(best_val_acc)
                
                # Clean up GPU memory
                del model, optimizer, scheduler, train_dataset, val_dataset, train_loader, val_loader
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Average accuracy for this run
            run_accuracy = np.mean(fold_accuracies)
            all_run_accuracies.append(run_accuracy)
            f_out.write(f"  Run {run+1} average accuracy: {run_accuracy:.4f}\n")
        
        # Calculate statistics across runs
        mean_accuracy = np.mean(all_run_accuracies)
        std_accuracy = np.std(all_run_accuracies)
        ci_lower = np.percentile(all_run_accuracies, (1 - CONFIDENCE_LEVEL) / 2 * 100)
        ci_upper = np.percentile(all_run_accuracies, (1 + CONFIDENCE_LEVEL) / 2 * 100)
        
        dl_results[model_name] = {
            'mean': mean_accuracy,
            'std': std_accuracy,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'all_runs': all_run_accuracies
        }
        
        f_out.write(f"\n{model_name} Statistics:\n")
        f_out.write(f"  Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}\n")
        f_out.write(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n")
        f_out.write(f"  Individual runs: {[f'{acc:.4f}' for acc in all_run_accuracies]}\n")
    
    f_out.write(f"\nRIGOROUS Deep Learning Models Summary for {saccade_count} saccades:\n")
    for model_name, result in dl_results.items():
        f_out.write(f"  {model_name}: {result['mean']:.4f} ± {result['std']:.4f} "
                   f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]\n")
    
    return dl_results

def run_saccade_count_analysis_rigorous(enhanced_items, f_out):
    """IMPROVED: Run comprehensive saccade count analysis with statistical rigor."""
    f_out.write("\n" + "="*80 + "\n")
    f_out.write("Phase: RIGOROUS Saccade Count vs Model Performance Analysis\n")
    f_out.write("="*80 + "\n")
    print("\n" + "="*50 + "\nRunning RIGOROUS Saccade Count Analysis...\n" + "="*50)
    
    # Determine available saccade counts
    all_counts = [item['saccade_count'] for item in enhanced_items]
    min_count, max_count = min(all_counts), max(all_counts)
    
    f_out.write(f"Original saccade count range: {min_count} to {max_count}\n")
    f_out.write(f"Testing specific saccade counts: {SACCADE_COUNTS_TO_TEST}\n")
    f_out.write(f"Maximum segments per sequence: {MAX_SEGMENTS_PER_SEQUENCE}\n")
    f_out.write(f"Statistical repeats: {N_STATISTICAL_REPEATS}\n")
    f_out.write(f"Deep learning repeats: {N_DL_REPEATS}\n")
    f_out.write(f"Confidence level: {CONFIDENCE_LEVEL}\n\n")
    
    # Store results for comprehensive analysis
    all_stat_results = {}  # saccade_count -> model -> {mean, std, ci_lower, ci_upper}
    all_dl_results = {}    # saccade_count -> model -> {mean, std, ci_lower, ci_upper}
    valid_counts = []
    sample_counts = {}  # Track sample sizes for each saccade count
    
    # Run analysis for each specific saccade count
    for saccade_count in SACCADE_COUNTS_TO_TEST:
        if saccade_count > max_count:
            f_out.write(f"Skipping {saccade_count} saccades (exceeds maximum available: {max_count})\n")
            continue
            
        f_out.write(f"\n{'='*60}\n")
        f_out.write(f"RIGOROUSLY ANALYZING {saccade_count} SACCADES\n")
        f_out.write(f"{'='*60}\n")
        print(f"\nRigorously analyzing {saccade_count} saccades")
        
        # Segment sequences for this saccade count
        segmented_items = segment_sequences_by_saccade_count(enhanced_items, saccade_count, f_out)
        
        # Check if we have sufficient data for both classes
        hc_count = sum(1 for item in segmented_items if item['label'] == 0)
        mg_count = sum(1 for item in segmented_items if item['label'] == 1)
        total_count = len(segmented_items)
        
        sample_counts[saccade_count] = {'total': total_count, 'hc': hc_count, 'mg': mg_count}
        
        if hc_count < MIN_SAMPLES_PER_BIN or mg_count < MIN_SAMPLES_PER_BIN:
            f_out.write(f"Insufficient data for both classes (HC: {hc_count}, MG: {mg_count}). Skipping {saccade_count} saccades.\n")
            continue
        
        valid_counts.append(saccade_count)
        
        # Feature engineering for this saccade count
        master_df, numerical_features = engineer_and_aggregate_features(
            segmented_items, FEATURE_COLUMNS, f_out
        )
        
        if master_df.empty:
            f_out.write(f"Empty DataFrame for {saccade_count} saccades. Skipping.\n")
            continue
        
        # Run rigorous statistical models
        stat_results = run_statistical_models_with_repeats(
            master_df, numerical_features, saccade_count, RESULTS_DIR, f_out
        )
        all_stat_results[saccade_count] = stat_results
        
        # Run rigorous deep learning models
        if segmented_items:
            sample_item = segmented_items[0]
            num_features = sample_item['data'].shape[1]
            dl_results = run_deep_learning_models_with_repeats(
                segmented_items, num_features, saccade_count, RESULTS_DIR, f_out
            )
            all_dl_results[saccade_count] = dl_results
    
    return all_stat_results, all_dl_results, valid_counts, sample_counts

def plot_rigorous_saccade_performance_analysis(stat_results, dl_results, valid_counts, sample_counts, results_dir, f_out):
    """Create rigorous plots showing performance vs saccade count with error bars and confidence intervals."""
    f_out.write(f"\n{'='*80}\n")
    f_out.write("CREATING RIGOROUS SACCADE COUNT vs PERFORMANCE PLOTS\n")
    f_out.write(f"{'='*80}\n")
    
    # Prepare data for plotting
    saccade_counts = sorted(valid_counts)
    
    # Create comprehensive figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Plot 1: Statistical Models Performance with Error Bars
    ax1 = axes[0, 0]
    
    # Get all statistical model names
    all_stat_models = set()
    for count_results in stat_results.values():
        all_stat_models.update(count_results.keys())
    
    colors = ['blue', 'red', 'green']
    for i, model_name in enumerate(sorted(all_stat_models)):
        means = []
        stds = []
        ci_lowers = []
        ci_uppers = []
        
        for saccade_count in saccade_counts:
            if saccade_count in stat_results and model_name in stat_results[saccade_count]:
                result = stat_results[saccade_count][model_name]
                means.append(result['mean'])
                stds.append(result['std'])
                ci_lowers.append(result['ci_lower'])
                ci_uppers.append(result['ci_upper'])
            else:
                means.append(np.nan)
                stds.append(np.nan)
                ci_lowers.append(np.nan)
                ci_uppers.append(np.nan)
        
        # Plot with error bars
        ax1.errorbar(saccade_counts, means, yerr=stds, 
                    label=model_name, linewidth=2, markersize=8, 
                    marker='o', capsize=5, color=colors[i % len(colors)])
        
        # Add confidence interval as fill
        ax1.fill_between(saccade_counts, ci_lowers, ci_uppers, 
                        alpha=0.2, color=colors[i % len(colors)])
    
    ax1.set_xlabel('Number of Saccades')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Statistical Models: Performance vs Saccade Count\n(with 95% CI and error bars)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.5, 0.8)
    
    # Plot 2: Deep Learning Models Performance with Error Bars
    ax2 = axes[0, 1]
    
    # Get all deep learning model names
    all_dl_models = set()
    for count_results in dl_results.values():
        all_dl_models.update(count_results.keys())
    
    colors_dl = ['purple', 'orange', 'brown']
    for i, model_name in enumerate(sorted(all_dl_models)):
        means = []
        stds = []
        ci_lowers = []
        ci_uppers = []
        
        for saccade_count in saccade_counts:
            if saccade_count in dl_results and model_name in dl_results[saccade_count]:
                result = dl_results[saccade_count][model_name]
                means.append(result['mean'])
                stds.append(result['std'])
                ci_lowers.append(result['ci_lower'])
                ci_uppers.append(result['ci_upper'])
            else:
                means.append(np.nan)
                stds.append(np.nan)
                ci_lowers.append(np.nan)
                ci_uppers.append(np.nan)
        
        # Plot with error bars
        ax2.errorbar(saccade_counts, means, yerr=stds, 
                    label=model_name, linewidth=2, markersize=8, 
                    marker='s', capsize=5, color=colors_dl[i % len(colors_dl)])
        
        # Add confidence interval as fill
        ax2.fill_between(saccade_counts, ci_lowers, ci_uppers, 
                        alpha=0.2, color=colors_dl[i % len(colors_dl)])
    
    ax2.set_xlabel('Number of Saccades')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Deep Learning Models: Performance vs Saccade Count\n(with 95% CI and error bars)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.5, 0.8)
    
    # Plot 3: Best Model Performance Comparison
    ax3 = axes[0, 2]
    
    best_stat_means = []
    best_stat_stds = []
    best_dl_means = []
    best_dl_stds = []
    
    for saccade_count in saccade_counts:
        # Best statistical model for this count
        if saccade_count in stat_results and stat_results[saccade_count]:
            best_stat_result = max(stat_results[saccade_count].values(), key=lambda x: x['mean'])
            best_stat_means.append(best_stat_result['mean'])
            best_stat_stds.append(best_stat_result['std'])
        else:
            best_stat_means.append(np.nan)
            best_stat_stds.append(np.nan)
        
        # Best deep learning model for this count
        if saccade_count in dl_results and dl_results[saccade_count]:
            best_dl_result = max(dl_results[saccade_count].values(), key=lambda x: x['mean'])
            best_dl_means.append(best_dl_result['mean'])
            best_dl_stds.append(best_dl_result['std'])
        else:
            best_dl_means.append(np.nan)
            best_dl_stds.append(np.nan)
    
    ax3.errorbar(saccade_counts, best_stat_means, yerr=best_stat_stds, 
                label='Best Statistical', linewidth=3, markersize=10, 
                marker='o', capsize=5, color='blue')
    ax3.errorbar(saccade_counts, best_dl_means, yerr=best_dl_stds, 
                label='Best Deep Learning', linewidth=3, markersize=10, 
                marker='s', capsize=5, color='red')
    
    ax3.set_xlabel('Number of Saccades')
    ax3.set_ylabel('Best Accuracy')
    ax3.set_title('Best Model Performance vs Saccade Count\n(with error bars)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.5, 0.8)
    
    # Plot 4: Sample Count Distribution
    ax4 = axes[1, 0]
    
    total_samples = [sample_counts[sc]['total'] for sc in saccade_counts]
    hc_samples = [sample_counts[sc]['hc'] for sc in saccade_counts]
    mg_samples = [sample_counts[sc]['mg'] for sc in saccade_counts]
    
    width = 2
    ax4.bar([sc - width/2 for sc in saccade_counts], hc_samples, width/2, 
           label='HC', alpha=0.7, color='blue')
    ax4.bar([sc for sc in saccade_counts], mg_samples, width/2, 
           label='MG', alpha=0.7, color='red')
    
    ax4.set_xlabel('Number of Saccades')
    ax4.set_ylabel('Number of Segments')
    ax4.set_title('Sample Distribution by Saccade Count')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Statistical Significance Analysis
    ax5 = axes[1, 1]
    
    # Calculate effect sizes and significance
    effect_sizes = []
    p_values = []
    
    for i, saccade_count in enumerate(saccade_counts[:-1]):
        if (saccade_count in stat_results and saccade_counts[i+1] in stat_results):
            # Compare best models between consecutive saccade counts
            current_best = max(stat_results[saccade_count].values(), key=lambda x: x['mean'])
            next_best = max(stat_results[saccade_counts[i+1]].values(), key=lambda x: x['mean'])
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((current_best['std']**2 + next_best['std']**2) / 2)
            effect_size = abs(current_best['mean'] - next_best['mean']) / pooled_std if pooled_std > 0 else 0
            effect_sizes.append(effect_size)
            
            # Simple t-test approximation for p-value
            t_stat = effect_size * np.sqrt(N_STATISTICAL_REPEATS / 2)
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), N_STATISTICAL_REPEATS - 1))
            p_values.append(p_val)
        else:
            effect_sizes.append(0)
            p_values.append(1)
    
    x_pos = [(saccade_counts[i] + saccade_counts[i+1]) / 2 for i in range(len(saccade_counts)-1)]
    colors_sig = ['red' if p < 0.05 else 'gray' for p in p_values]
    
    bars = ax5.bar(x_pos, effect_sizes, width=3, color=colors_sig, alpha=0.7)
    ax5.axhline(y=0.2, color='orange', linestyle='--', label='Small Effect')
    ax5.axhline(y=0.5, color='red', linestyle='--', label='Medium Effect')
    ax5.axhline(y=0.8, color='darkred', linestyle='--', label='Large Effect')
    
    ax5.set_xlabel('Saccade Count Transition')
    ax5.set_ylabel('Effect Size (Cohen\'s d)')
    ax5.set_title('Effect Size Between Consecutive Saccade Counts\n(Red = p < 0.05)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Trend Analysis
    ax6 = axes[1, 2]
    
    # Fit polynomial trends to best performance
    x_data = np.array(saccade_counts)
    y_stat = np.array(best_stat_means)
    y_dl = np.array(best_dl_means)
    
    # Remove NaN values
    valid_stat = ~np.isnan(y_stat)
    valid_dl = ~np.isnan(y_dl)
    
    if np.sum(valid_stat) > 2:
        # Fit quadratic trend for statistical models
        stat_coeffs = np.polyfit(x_data[valid_stat], y_stat[valid_stat], 2)
        stat_trend = np.polyval(stat_coeffs, x_data)
        ax6.plot(x_data, stat_trend, '--', color='blue', linewidth=2, label='Statistical Trend')
    
    if np.sum(valid_dl) > 2:
        # Fit quadratic trend for deep learning models
        dl_coeffs = np.polyfit(x_data[valid_dl], y_dl[valid_dl], 2)
        dl_trend = np.polyval(dl_coeffs, x_data)
        ax6.plot(x_data, dl_trend, '--', color='red', linewidth=2, label='Deep Learning Trend')
    
    # Plot actual data points
    ax6.scatter(x_data[valid_stat], y_stat[valid_stat], color='blue', s=100, alpha=0.7, label='Statistical Data')
    ax6.scatter(x_data[valid_dl], y_dl[valid_dl], color='red', s=100, alpha=0.7, label='Deep Learning Data')
    
    ax6.set_xlabel('Number of Saccades')
    ax6.set_ylabel('Best Accuracy')
    ax6.set_title('Performance Trend Analysis\n(Quadratic Fit)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'{EXP_PREFIX}rigorous_saccade_performance_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    f_out.write(f"RIGOROUS saccade performance analysis plot saved to: {plot_path}\n\n")

def prepare_binary_data_with_saccades(raw_items, f_out):
    """Prepare data for binary classification by combining MG classes."""
    f_out.write("="*80 + "\n")
    f_out.write("Phase: Binary Classification Data Preparation\n")
    f_out.write("="*80 + "\n")
    print("\n" + "="*50 + "\nPreparing Binary Classification Data...\n" + "="*50)
    
    binary_items = []
    class_counts = {'HC': 0, 'MG': 0}
    
    for item in raw_items:
        if item['class_name'] in ['MG', 'Probable_MG']:
            # Combine Definite MG and Probable MG into MG class
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

def main():
    """Main execution function with improved statistical rigor."""
    print("="*80)
    print("Starting RIGOROUS Experiment 13D: Saccade Count vs Model Performance Analysis")
    print("Binary Classification (HC vs MG) with Statistical Rigor")
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
        f_report.write("RIGOROUS Experiment 13D: Saccade Count vs Model Performance Analysis\n")
        f_report.write("="*80 + "\n")
        f_report.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_report.write(f"Base Directory: {BASE_DIR}\n")
        f_report.write(f"Classes: {ORDERED_CLASS_NAMES} (Binary Classification)\n")
        f_report.write(f"Device: {DEVICE}\n")
        f_report.write(f"Random State: {RANDOM_STATE}\n")
        f_report.write(f"IMPROVED Saccade Detection Method: Zero crossings with Gaussian smoothing\n")
        f_report.write(f"Smoothing Window: {SMOOTHING_WINDOW}\n")
        f_report.write(f"Minimum Samples per Saccade Bin: {MIN_SAMPLES_PER_BIN}\n")
        f_report.write(f"Subsampling Factor for DL: {SUBSAMPLE_FACTOR}x (IMPROVED)\n")
        f_report.write(f"Statistical Repeats: {N_STATISTICAL_REPEATS}\n")
        f_report.write(f"Deep Learning Repeats: {N_DL_REPEATS}\n")
        f_report.write(f"Confidence Level: {CONFIDENCE_LEVEL}\n")
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
        binary_items = prepare_binary_data_with_saccades(raw_items_list, f_report)
        
        # 3. Detect Saccades and Analyze Counts (IMPROVED)
        enhanced_items = analyze_saccade_counts(binary_items, f_report)
        
        # 4. Create IMPROVED Saccade Verification Plots
        create_saccade_verification_plots(enhanced_items, RESULTS_DIR, f_report)
        
        # 5. Run RIGOROUS Saccade Count Analysis
        all_stat_results, all_dl_results, valid_counts, sample_counts = run_saccade_count_analysis_rigorous(
            enhanced_items, f_report
        )
        
        # 6. Create RIGOROUS Performance Analysis Plots
        plot_rigorous_saccade_performance_analysis(all_stat_results, all_dl_results, valid_counts, sample_counts, RESULTS_DIR, f_report)
        
        # 7. RIGOROUS Final Summary and Statistical Analysis
        f_report.write("\n" + "="*80 + "\n")
        f_report.write("RIGOROUS EXPERIMENT 13D FINAL RESULTS SUMMARY\n")
        f_report.write("="*80 + "\n")
        
        f_report.write("RIGOROUS SACCADE COUNT ANALYSIS RESULTS:\n")
        f_report.write("-" * 40 + "\n")
        
        # Overall saccade statistics
        all_saccade_counts = [item['saccade_count'] for item in enhanced_items]
        f_report.write(f"Overall saccade count statistics:\n")
        f_report.write(f"  Mean: {np.mean(all_saccade_counts):.2f}\n")
        f_report.write(f"  Std: {np.std(all_saccade_counts):.2f}\n")
        f_report.write(f"  Range: {min(all_saccade_counts)} - {max(all_saccade_counts)}\n")
        f_report.write(f"  Valid counts analyzed: {len(valid_counts)}\n\n")
        
        # Statistical models performance by saccade count WITH CONFIDENCE INTERVALS
        f_report.write("RIGOROUS STATISTICAL MODELS BY SACCADE COUNT:\n")
        f_report.write("-" * 40 + "\n")
        for saccade_count in valid_counts:
            f_report.write(f"\n{saccade_count} Saccades:\n")
            if saccade_count in all_stat_results:
                for model_name, result in all_stat_results[saccade_count].items():
                    f_report.write(f"  {model_name}: {result['mean']:.4f} ± {result['std']:.4f} "
                                  f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]\n")
        
        # Deep learning models performance by saccade count WITH CONFIDENCE INTERVALS
        f_report.write("\nRIGOROUS DEEP LEARNING MODELS BY SACCADE COUNT:\n")
        f_report.write("-" * 40 + "\n")
        for saccade_count in valid_counts:
            f_report.write(f"\n{saccade_count} Saccades:\n")
            if saccade_count in all_dl_results:
                for model_name, result in all_dl_results[saccade_count].items():
                    f_report.write(f"  {model_name}: {result['mean']:.4f} ± {result['std']:.4f} "
                                  f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]\n")
        
        # Find best performing saccade counts WITH STATISTICAL SIGNIFICANCE
        f_report.write("\nBEST PERFORMING SACCADE COUNTS (WITH CONFIDENCE INTERVALS):\n")
        f_report.write("-" * 40 + "\n")
        
        # Best statistical model across all saccade counts
        best_stat_performance = 0
        best_stat_count = None
        best_stat_model = None
        best_stat_result = None
        
        for saccade_count in all_stat_results:
            for model_name, result in all_stat_results[saccade_count].items():
                if result['mean'] > best_stat_performance:
                    best_stat_performance = result['mean']
                    best_stat_count = saccade_count
                    best_stat_model = model_name
                    best_stat_result = result
        
        if best_stat_model:
            f_report.write(f"Best Statistical: {best_stat_model} with {best_stat_count} saccades\n")
            f_report.write(f"  Performance: {best_stat_result['mean']:.4f} ± {best_stat_result['std']:.4f}\n")
            f_report.write(f"  95% CI: [{best_stat_result['ci_lower']:.4f}, {best_stat_result['ci_upper']:.4f}]\n")
        
        # Best deep learning model across all saccade counts
        best_dl_performance = 0
        best_dl_count = None
        best_dl_model = None
        best_dl_result = None
        
        for saccade_count in all_dl_results:
            for model_name, result in all_dl_results[saccade_count].items():
                if result['mean'] > best_dl_performance:
                    best_dl_performance = result['mean']
                    best_dl_count = saccade_count
                    best_dl_model = model_name
                    best_dl_result = result
        
        if best_dl_model:
            f_report.write(f"\nBest Deep Learning: {best_dl_model} with {best_dl_count} saccades\n")
            f_report.write(f"  Performance: {best_dl_result['mean']:.4f} ± {best_dl_result['std']:.4f}\n")
            f_report.write(f"  95% CI: [{best_dl_result['ci_lower']:.4f}, {best_dl_result['ci_upper']:.4f}]\n")
        
        # Statistical significance analysis
        f_report.write("\nSTATISTICAL SIGNIFICANCE ANALYSIS:\n")
        f_report.write("-" * 35 + "\n")
        
        # Test if there's a significant trend
        if len(valid_counts) > 2:
            # Collect best performances for trend analysis
            stat_performances = []
            dl_performances = []
            
            for saccade_count in valid_counts:
                if saccade_count in all_stat_results:
                    best_stat = max(all_stat_results[saccade_count].values(), key=lambda x: x['mean'])
                    stat_performances.append(best_stat['mean'])
                else:
                    stat_performances.append(np.nan)
                
                if saccade_count in all_dl_results:
                    best_dl = max(all_dl_results[saccade_count].values(), key=lambda x: x['mean'])
                    dl_performances.append(best_dl['mean'])
                else:
                    dl_performances.append(np.nan)
            
            # Correlation analysis
            valid_stat_idx = ~np.isnan(stat_performances)
            valid_dl_idx = ~np.isnan(dl_performances)
            
            if np.sum(valid_stat_idx) > 2:
                stat_corr, stat_p = stats.pearsonr(np.array(valid_counts)[valid_stat_idx], 
                                                  np.array(stat_performances)[valid_stat_idx])
                f_report.write(f"Statistical models trend correlation: r={stat_corr:.4f}, p={stat_p:.4f}\n")
            
            if np.sum(valid_dl_idx) > 2:
                dl_corr, dl_p = stats.pearsonr(np.array(valid_counts)[valid_dl_idx], 
                                              np.array(dl_performances)[valid_dl_idx])
                f_report.write(f"Deep learning models trend correlation: r={dl_corr:.4f}, p={dl_p:.4f}\n")
        
        # Key insights and recommendations
        f_report.write("\nKEY INSIGHTS AND RECOMMENDATIONS (RIGOROUS ANALYSIS):\n")
        f_report.write("-" * 40 + "\n")
        f_report.write("1. IMPROVED saccade detection with Gaussian smoothing and amplitude filtering\n")
        f_report.write("2. Multiple runs with confidence intervals provide statistical rigor\n")
        f_report.write("3. Performance trends analyzed with effect sizes and significance testing\n")
        f_report.write("4. Comprehensive visualization with error bars and trend analysis\n")
        
        # Data collection recommendations with statistical backing
        f_report.write("\nSTATISTICALLY-BACKED DATA COLLECTION RECOMMENDATIONS:\n")
        f_report.write("-" * 35 + "\n")
        if best_stat_count and best_dl_count:
            optimal_min = min(best_stat_count, best_dl_count)
            optimal_max = max(best_stat_count, best_dl_count)
            f_report.write(f"Recommended saccade count range: {optimal_min}-{optimal_max}\n")
            f_report.write(f"  Statistical models peak at: {best_stat_count} saccades\n")
            f_report.write(f"  Deep learning models peak at: {best_dl_count} saccades\n")
        elif best_stat_count:
            f_report.write(f"Recommended based on statistical models: {best_stat_count} saccades\n")
        elif best_dl_count:
            f_report.write(f"Recommended based on deep learning models: {best_dl_count} saccades\n")
        
        f_report.write("Future data collection should target sequences with optimal saccade counts\n")
        f_report.write("Protocol adjustments should ensure consistent saccade generation\n")
        f_report.write("Consider patient fatigue vs. performance trade-offs\n")
        
        f_report.write("\n" + "="*80 + "\n")
        f_report.write("End of RIGOROUS Experiment 13D Report\n")
        f_report.write("="*80 + "\n")
    
    print(f"\nRIGOROUS Experiment 13D completed successfully!")
    print(f"Results saved to: {summary_filepath}")
    print(f"All plots and logs saved in: {RESULTS_DIR}")
    print("="*80)

if __name__ == '__main__':
    main()
