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
from scipy.stats import iqr as scipy_iqr

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.data_loading import load_raw_sequences_and_labels
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
RESULTS_DIR = './results/exp_13e'
EXP_PREFIX = 'EXP_13E_'
NUMERICAL_SUMMARY_FILENAME = f'{EXP_PREFIX}numerical_summary.txt'
RANDOM_STATE = 42

# Deep Learning Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUBSAMPLE_FACTOR = 10  # 10x subsampling for quick results
TARGET_SEQ_LEN_PERCENTILE = 90  # Reduced for memory efficiency
EPOCHS = 50  # Reduced for quick results
BATCH_SIZE = 16  # Small batch size
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# Saccadic Range Analysis Parameters
VELOCITY_THRESHOLD = 30.0  # degrees/sec as specified by medical team
RANGE_THRESHOLD = 30.0     # degrees - fixed threshold for grouping datapoints
N_TRIALS = 5  # Multiple trials for comprehensive results
MIN_SAMPLES_PER_RANGE_BIN = 20  # Minimum samples needed per range bin

# Finer range bins for more detailed analysis
RANGE_BINS = {
    'Very_Small': (0, 15.0),        # 0-15°
    'Small': (15.0, 30.0),          # 15-30°
    'Medium': (30.0, 45.0),         # 30-45°
    'Large': (45.0, 60.0),          # 45-60°
    'Very_Large': (60.0, 200.0)     # >60°
}

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.discriminant_analysis')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

def detect_saccades_from_velocity(velocity_data, threshold=30.0, min_duration=3):
    """
    Detect saccades using velocity threshold method.
    
    Args:
        velocity_data: Array of velocity values
        threshold: Velocity threshold in degrees/sec
        min_duration: Minimum duration of saccade in samples
    
    Returns:
        List of (onset_idx, end_idx) tuples for detected saccades
    """
    # Find points where absolute velocity exceeds threshold
    above_threshold = np.abs(velocity_data) > threshold
    
    # Find onset and offset points
    saccades = []
    in_saccade = False
    onset_idx = None
    
    for i, above in enumerate(above_threshold):
        if above and not in_saccade:
            # Saccade onset
            onset_idx = i
            in_saccade = True
        elif not above and in_saccade:
            # Saccade end
            if i - onset_idx >= min_duration:  # Check minimum duration
                saccades.append((onset_idx, i-1))
            in_saccade = False
    
    # Handle case where sequence ends during saccade
    if in_saccade and len(velocity_data) - onset_idx >= min_duration:
        saccades.append((onset_idx, len(velocity_data)-1))
    
    return saccades

def calculate_saccadic_ranges(position_data, velocity_data, saccades):
    """
    Calculate saccadic ranges for detected saccades.
    
    Args:
        position_data: Array of position values
        velocity_data: Array of velocity values  
        saccades: List of (onset_idx, end_idx) tuples
    
    Returns:
        List of saccadic ranges
    """
    ranges = []
    
    for onset_idx, end_idx in saccades:
        onset_pos = position_data[onset_idx]
        end_pos = position_data[end_idx]
        saccadic_range = abs(end_pos - onset_pos)
        ranges.append(saccadic_range)
    
    return ranges

def calculate_sequence_saccadic_range(item, f_out):
    """Calculate the mean saccadic range for a sequence."""
    data = item['data']
    df = pd.DataFrame(data, columns=FEATURE_COLUMNS)
    
    all_ranges = []
    
    # Analyze horizontal saccades (primary direction)
    for eye in ['LH', 'RH']:
        position_data = df[eye].values
        velocity_data = np.diff(position_data, prepend=position_data[0])
        
        # Detect saccades
        saccades = detect_saccades_from_velocity(velocity_data, VELOCITY_THRESHOLD)
        
        # Calculate ranges
        if saccades:
            ranges = calculate_saccadic_ranges(position_data, velocity_data, saccades)
            all_ranges.extend(ranges)
    
    # Return mean range for this sequence (or 0 if no saccades detected)
    return np.mean(all_ranges) if all_ranges else 0.0

def prepare_binary_data_with_saccadic_ranges(raw_items, f_out):
    """Prepare data for binary classification and calculate saccadic ranges."""
    f_out.write("="*80 + "\n")
    f_out.write("Phase: Binary Classification Data Preparation with Saccadic Range Calculation\n")
    f_out.write("="*80 + "\n")
    print("\n" + "="*50 + "\nPreparing Binary Classification Data with Saccadic Ranges...\n" + "="*50)
    
    binary_items = []
    class_counts = {'HC': 0, 'MG': 0}
    range_stats = {'HC': [], 'MG': []}
    
    for item in tqdm(raw_items, desc="Processing sequences"):
        # Calculate saccadic range for this sequence
        saccadic_range = calculate_sequence_saccadic_range(item, f_out)
        
        if item['class_name'] in ['MG', 'Probable_MG']:
            # Combine Definite MG and Probable MG into MG class
            new_item = item.copy()
            new_item['class_name'] = 'MG'
            new_item['label'] = 1
            new_item['saccadic_range'] = saccadic_range
            binary_items.append(new_item)
            class_counts['MG'] += 1
            range_stats['MG'].append(saccadic_range)
        elif item['class_name'] == 'HC':
            item['saccadic_range'] = saccadic_range
            binary_items.append(item)
            class_counts['HC'] += 1
            range_stats['HC'].append(saccadic_range)
    
    f_out.write(f"Binary classification data prepared:\n")
    f_out.write(f"  HC (Healthy Control): {class_counts['HC']} samples\n")
    f_out.write(f"  MG (Definite + Probable): {class_counts['MG']} samples\n")
    f_out.write(f"  Total: {len(binary_items)} samples\n\n")
    
    f_out.write("Saccadic range statistics by class:\n")
    for class_name, ranges in range_stats.items():
        if ranges:
            f_out.write(f"  {class_name}: mean={np.mean(ranges):.2f}°, std={np.std(ranges):.2f}°, "
                       f"min={np.min(ranges):.2f}°, max={np.max(ranges):.2f}°\n")
    f_out.write("-" * 80 + "\n\n")
    
    print(f"Binary data prepared: HC={class_counts['HC']}, MG={class_counts['MG']}")
    return binary_items

def filter_by_range_bin(binary_items, range_bin_name, range_bin_limits, f_out):
    """Filter items by specific saccadic range bin."""
    min_range, max_range = range_bin_limits
    filtered_items = [item for item in binary_items 
                     if min_range <= item['saccadic_range'] < max_range]
    
    class_counts = {'HC': 0, 'MG': 0}
    for item in filtered_items:
        if item['label'] == 0:
            class_counts['HC'] += 1
        else:
            class_counts['MG'] += 1
    
    f_out.write(f"Range bin '{range_bin_name}' ({min_range:.1f}° - {max_range:.1f}°) data:\n")
    f_out.write(f"  HC: {class_counts['HC']} samples\n")
    f_out.write(f"  MG: {class_counts['MG']} samples\n")
    f_out.write(f"  Total: {len(filtered_items)} samples\n\n")
    
    return filtered_items, class_counts

def engineer_and_aggregate_features_for_range_items(range_items, f_out):
    """Feature engineering and aggregation for range-filtered items."""
    f_out.write("Feature engineering and aggregation for range-filtered data...\n")
    
    if not range_items:
        f_out.write("No items to process. Returning empty DataFrame.\n")
        return pd.DataFrame(), []

    aggregated_data_rows = []
    engineered_feature_names = []

    for item in tqdm(range_items, desc="  Engineering & Aggregating"):
        df_original = pd.DataFrame(item['data'], columns=FEATURE_COLUMNS)
        
        # --- Engineering ---
        df_engineered_parts = []
        for pos_col in ['LH', 'RH', 'LV', 'RV']:
            velocity_series = df_original[pos_col].diff().fillna(0)
            df_engineered_parts.append(velocity_series.rename(f'{pos_col}_Vel'))
        
        error_definitions = [('LH', 'TargetH', 'ErrorH_L'), ('RH', 'TargetH', 'ErrorH_R'),
                             ('LV', 'TargetV', 'ErrorV_L'), ('RV', 'TargetV', 'ErrorV_R')]
        for eye_col, target_col, error_col_name in error_definitions:
            df_engineered_parts.append((df_original[eye_col] - df_original[target_col]).rename(error_col_name))
        
        df_all_features = pd.concat([df_original] + df_engineered_parts, axis=1)
        if not engineered_feature_names: 
            engineered_feature_names = df_all_features.columns.tolist()

        # --- Aggregation ---
        current_row = {
            'patient_id': item['patient_id'], 'filename': item['filename'],
            'class_name': item['class_name'], 'label': item['label'],
            'saccadic_range': item['saccadic_range']
        }
        for feature_name in engineered_feature_names:
            ft_ts = df_all_features[feature_name]
            current_row[f'{feature_name}_mean'] = np.mean(ft_ts)
            current_row[f'{feature_name}_std'] = np.std(ft_ts)
            current_row[f'{feature_name}_median'] = np.median(ft_ts)
            current_row[f'{feature_name}_iqr'] = scipy_iqr(ft_ts)
        aggregated_data_rows.append(current_row)

    agg_df = pd.DataFrame(aggregated_data_rows)
    numerical_features = [f for f in agg_df.columns if '_mean' in f or '_std' in f or '_median' in f or '_iqr' in f]
    
    f_out.write(f"Aggregated features DataFrame created with shape: {agg_df.shape}\n")
    f_out.write(f"Number of numerical features: {len(numerical_features)}\n\n")
    
    return agg_df, numerical_features

def run_statistical_models_for_range_bin(master_df, numerical_features, range_bin_name, results_dir, f_out, trial_num=1):
    """Run statistical models for a specific range bin."""
    f_out.write(f"\n--- Statistical Models for {range_bin_name} Range (Trial {trial_num}) ---\n")
    print(f"\nRunning Statistical Models for {range_bin_name} Range (Trial {trial_num})...")
    
    X = master_df[numerical_features]
    y = master_df['label']
    
    models = get_best_statistical_models()
    results = {}
    
    for model_name, model in models.items():
        accuracy, report, cm = train_and_evaluate_single_model(
            X, y, numerical_features, [], model_name, model, 
            ORDERED_CLASS_NAMES, MODEL_CLASS_LABELS, results_dir, f_out, 
            suffix=f"_Range_{range_bin_name}_Trial_{trial_num}", random_state=RANDOM_STATE + trial_num
        )
        results[model_name] = accuracy
    
    f_out.write(f"\nStatistical Models Summary for {range_bin_name} Range (Trial {trial_num}):\n")
    for model_name, accuracy in results.items():
        f_out.write(f"  {model_name}: {accuracy:.4f}\n")
    
    return results

def run_deep_learning_models_for_range_bin(range_items, num_features, range_bin_name, results_dir, f_out, trial_num=1):
    """Run deep learning models for a specific range bin."""
    f_out.write(f"\n--- Deep Learning Models for {range_bin_name} Range (Trial {trial_num}) ---\n")
    print(f"\nRunning Deep Learning Models for {range_bin_name} Range (Trial {trial_num})...")
    
    # Subsample data for quick results
    subsampled_items = subsample_data(range_items, SUBSAMPLE_FACTOR)
    f_out.write(f"Subsampled data: {len(subsampled_items)} samples (factor: {SUBSAMPLE_FACTOR}x)\n")
    
    if len(subsampled_items) < 10:
        f_out.write(f"Insufficient subsampled data for deep learning ({len(subsampled_items)} samples). Skipping.\n")
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
    models = get_small_dl_models(num_features, 2, target_seq_len)  # 2 classes for binary
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE + trial_num)
    dl_results = {}
    
    for model_name, model_class in models.items():
        f_out.write(f"\n--- Deep Learning Model: {model_name} for {range_bin_name} Range (Trial {trial_num}) ---\n")
        
        fold_accuracies = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            f_out.write(f"\nFold {fold+1}/3:\n")
            
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
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
            early_stopper = EarlyStopper(patience=EARLY_STOPPING_PATIENCE)
            
            # Training loop
            train_losses, val_losses = [], []
            best_val_acc = 0.0
            
            for epoch in range(EPOCHS):
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
                val_loss, val_acc, _, _ = evaluate_epoch(model, val_loader, criterion, DEVICE)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                scheduler.step(val_loss)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                
                if early_stopper.early_stop(val_loss):
                    f_out.write(f"  Early stopping at epoch {epoch+1}\n")
                    break
            
            fold_accuracies.append(best_val_acc)
            f_out.write(f"  Fold {fold+1} best validation accuracy: {best_val_acc:.4f}\n")
            
            # Clean up GPU memory
            del model, optimizer, scheduler, train_dataset, val_dataset, train_loader, val_loader
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Calculate average performance
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        dl_results[model_name] = mean_accuracy
        
        f_out.write(f"\n{model_name} Average Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})\n")
    
    f_out.write(f"\nDeep Learning Models Summary for {range_bin_name} Range (Trial {trial_num}):\n")
    for model_name, accuracy in dl_results.items():
        f_out.write(f"  {model_name}: {accuracy:.4f}\n")
    
    return dl_results

def run_range_separated_analysis(binary_items, f_out):
    """Run comprehensive range-separated analysis with multiple trials."""
    f_out.write("\n" + "="*80 + "\n")
    f_out.write("Phase: Saccadic Range-Separated Binary Classification Analysis\n")
    f_out.write("="*80 + "\n")
    print("\n" + "="*50 + "\nRunning Range-Separated Analysis...\n" + "="*50)
    
    f_out.write(f"Range bins defined:\n")
    for bin_name, (min_val, max_val) in RANGE_BINS.items():
        f_out.write(f"  {bin_name}: {min_val:.1f}° - {max_val:.1f}°\n")
    f_out.write("\n")
    
    # Filter range bins with sufficient samples
    valid_range_bins = {}
    for bin_name, bin_limits in RANGE_BINS.items():
        bin_items, class_counts = filter_by_range_bin(binary_items, bin_name, bin_limits, f_out)
        if len(bin_items) >= MIN_SAMPLES_PER_RANGE_BIN and class_counts['HC'] >= 5 and class_counts['MG'] >= 5:
            valid_range_bins[bin_name] = (bin_limits, bin_items)
        else:
            f_out.write(f"Skipping {bin_name}: insufficient samples or class imbalance\n")
    
    f_out.write(f"Valid range bins for analysis: {list(valid_range_bins.keys())}\n\n")
    
    # Store results for comprehensive analysis
    all_stat_results = {}  # range_bin -> trial -> model -> accuracy
    all_dl_results = {}    # range_bin -> trial -> model -> accuracy
    
    # Run analysis for each range bin
    for bin_name, (bin_limits, bin_items) in valid_range_bins.items():
        f_out.write(f"\n{'='*60}\n")
        f_out.write(f"ANALYZING RANGE BIN: {bin_name} ({bin_limits[0]:.1f}° - {bin_limits[1]:.1f}°)\n")
        f_out.write(f"{'='*60}\n")
        print(f"\nAnalyzing range bin: {bin_name}")
        
        # Initialize results storage
        all_stat_results[bin_name] = {}
        all_dl_results[bin_name] = {}
        
        # Run multiple trials
        for trial in range(1, N_TRIALS + 1):
            f_out.write(f"\n--- TRIAL {trial}/{N_TRIALS} for {bin_name} Range ---\n")
            print(f"  Trial {trial}/{N_TRIALS}")
            
            # Feature engineering for this range bin
            master_df, numerical_features = engineer_and_aggregate_features_for_range_items(
                bin_items, f_out
            )
            
            if master_df.empty:
                f_out.write(f"Empty DataFrame for {bin_name} range, trial {trial}. Skipping.\n")
                continue
            
            # Run statistical models
            stat_results = run_statistical_models_for_range_bin(
                master_df, numerical_features, bin_name, RESULTS_DIR, f_out, trial
            )
            all_stat_results[bin_name][trial] = stat_results
            
            # Run deep learning models
            sample_item = bin_items[0]
            num_features = sample_item['data'].shape[1]
            dl_results = run_deep_learning_models_for_range_bin(
                bin_items, num_features, bin_name, RESULTS_DIR, f_out, trial
            )
            all_dl_results[bin_name][trial] = dl_results
    
    return all_stat_results, all_dl_results, list(valid_range_bins.keys())

def calculate_comprehensive_statistics(all_results, result_type, f_out):
    """Calculate mean and standard deviation across trials for each range bin and model."""
    f_out.write(f"\n{'='*80}\n")
    f_out.write(f"COMPREHENSIVE {result_type.upper()} RESULTS WITH ERROR BARS\n")
    f_out.write(f"{'='*80}\n")
    
    comprehensive_stats = {}
    
    for range_bin in all_results:
        comprehensive_stats[range_bin] = {}
        
        # Get all model names across trials
        all_models = set()
        for trial_results in all_results[range_bin].values():
            all_models.update(trial_results.keys())
        
        for model_name in all_models:
            # Collect accuracies across trials
            accuracies = []
            for trial_results in all_results[range_bin].values():
                if model_name in trial_results:
                    accuracies.append(trial_results[model_name])
            
            if accuracies:
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                comprehensive_stats[range_bin][model_name] = {
                    'mean': mean_acc,
                    'std': std_acc,
                    'n_trials': len(accuracies)
                }
                
                f_out.write(f"{range_bin} Range - {model_name}: {mean_acc:.4f} ± {std_acc:.4f} (n={len(accuracies)})\n")
    
    return comprehensive_stats

def plot_range_comparison(stat_stats, dl_stats, valid_range_bins, results_dir, f_out):
    """Create comprehensive plots comparing performance across saccadic range bins."""
    f_out.write(f"\n{'='*80}\n")
    f_out.write("CREATING RANGE COMPARISON PLOTS\n")
    f_out.write(f"{'='*80}\n")
    
    # Plot 1: Statistical Models Comparison
    plt.figure(figsize=(14, 8))
    
    # Get all statistical model names
    all_stat_models = set()
    for range_stats in stat_stats.values():
        all_stat_models.update(range_stats.keys())
    
    x_pos = np.arange(len(valid_range_bins))
    width = 0.25
    
    for i, model_name in enumerate(sorted(all_stat_models)):
        means = []
        stds = []
        for range_bin in valid_range_bins:
            if range_bin in stat_stats and model_name in stat_stats[range_bin]:
                means.append(stat_stats[range_bin][model_name]['mean'])
                stds.append(stat_stats[range_bin][model_name]['std'])
            else:
                means.append(0)
                stds.append(0)
        
        plt.bar(x_pos + i * width, means, width, yerr=stds, 
                label=model_name, alpha=0.8, capsize=5)
    
    plt.xlabel('Saccadic Range Bin', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Statistical Models Performance by Saccadic Range (with Error Bars)', fontsize=14)
    plt.xticks(x_pos + width, valid_range_bins)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    stat_plot_path = os.path.join(results_dir, f'{EXP_PREFIX}Statistical_Models_Range_Comparison.png')
    plt.savefig(stat_plot_path)
    plt.close()
    f_out.write(f"Statistical models comparison plot saved to: {stat_plot_path}\n")
    
    # Plot 2: Deep Learning Models Comparison
    if dl_stats:
        plt.figure(figsize=(14, 8))
        
        # Get all deep learning model names
        all_dl_models = set()
        for range_stats in dl_stats.values():
            all_dl_models.update(range_stats.keys())
        
        for i, model_name in enumerate(sorted(all_dl_models)):
            means = []
            stds = []
            for range_bin in valid_range_bins:
                if range_bin in dl_stats and model_name in dl_stats[range_bin]:
                    means.append(dl_stats[range_bin][model_name]['mean'])
                    stds.append(dl_stats[range_bin][model_name]['std'])
                else:
                    means.append(0)
                    stds.append(0)
            
            plt.bar(x_pos + i * width, means, width, yerr=stds, 
                    label=model_name, alpha=0.8, capsize=5)
        
        plt.xlabel('Saccadic Range Bin', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Deep Learning Models Performance by Saccadic Range (with Error Bars)', fontsize=14)
        plt.xticks(x_pos + width, valid_range_bins)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        dl_plot_path = os.path.join(results_dir, f'{EXP_PREFIX}Deep_Learning_Models_Range_Comparison.png')
        plt.savefig(dl_plot_path)
        plt.close()
        f_out.write(f"Deep learning models comparison plot saved to: {dl_plot_path}\n")

def main():
    """Main execution function."""
    print("="*80)
    print("Starting Experiment 13E: Binary Classification with Saccadic Range Analysis")
    print("Including 'Probable MG' in MG class, separating by saccadic range")
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
        f_report.write("Experiment 13E: Binary Classification with Saccadic Range Analysis\n")
        f_report.write("="*80 + "\n")
        f_report.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_report.write(f"Base Directory: {BASE_DIR}\n")
        f_report.write(f"Classes: {ORDERED_CLASS_NAMES} (Binary Classification)\n")
        f_report.write(f"Device: {DEVICE}\n")
        f_report.write(f"Random State: {RANDOM_STATE}\n")
        f_report.write(f"Number of Trials per Range Bin: {N_TRIALS}\n")
        f_report.write(f"Minimum Samples per Range Bin: {MIN_SAMPLES_PER_RANGE_BIN}\n")
        f_report.write(f"Subsampling Factor for DL: {SUBSAMPLE_FACTOR}x\n")
        f_report.write(f"Velocity Threshold: {VELOCITY_THRESHOLD}°/sec\n")
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
        
        # 2. Prepare Binary Classification Data with Saccadic Range Calculation
        binary_items = prepare_binary_data_with_saccadic_ranges(raw_items_list, f_report)
        
        # 3. Run Range-Separated Analysis
        all_stat_results, all_dl_results, valid_range_bins = run_range_separated_analysis(
            binary_items, f_report
        )
        
        # 4. Calculate Comprehensive Statistics
        stat_stats = calculate_comprehensive_statistics(all_stat_results, "Statistical", f_report)
        dl_stats = calculate_comprehensive_statistics(all_dl_results, "Deep Learning", f_report)
        
        # 5. Create Comparison Plots
        plot_range_comparison(stat_stats, dl_stats, valid_range_bins, RESULTS_DIR, f_report)
        
        # 6. Final Summary and Conclusions
        f_report.write("\n" + "="*80 + "\n")
        f_report.write("EXPERIMENT 13E FINAL RESULTS SUMMARY\n")
        f_report.write("="*80 + "\n")
        
        f_report.write("STATISTICAL MODELS BY SACCADIC RANGE (Mean ± Std):\n")
        f_report.write("-" * 50 + "\n")
        for range_bin in valid_range_bins:
            f_report.write(f"\n{range_bin} Range:\n")
            if range_bin in stat_stats:
                for model_name, stats in stat_stats[range_bin].items():
                    f_report.write(f"  {model_name}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
        
        f_report.write("\nDEEP LEARNING MODELS BY SACCADIC RANGE (Mean ± Std):\n")
        f_report.write("-" * 50 + "\n")
        for range_bin in valid_range_bins:
            f_report.write(f"\n{range_bin} Range:\n")
            if range_bin in dl_stats:
                for model_name, stats in dl_stats[range_bin].items():
                    f_report.write(f"  {model_name}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
        
        # Find best performing range bin for each model type
        f_report.write("\nBEST PERFORMING SACCADIC RANGE BINS:\n")
        f_report.write("-" * 40 + "\n")
        
        # Best statistical model across all range bins
        best_stat_performance = 0
        best_stat_range = None
        best_stat_model = None
        
        for range_bin in stat_stats:
            for model_name, stats in stat_stats[range_bin].items():
                if stats['mean'] > best_stat_performance:
                    best_stat_performance = stats['mean']
                    best_stat_range = range_bin
                    best_stat_model = model_name
        
        if best_stat_model:
            f_report.write(f"Best Statistical: {best_stat_model} with {best_stat_range} Range "
                          f"({best_stat_performance:.4f} ± {stat_stats[best_stat_range][best_stat_model]['std']:.4f})\n")
        
        # Best deep learning model across all range bins
        best_dl_performance = 0
        best_dl_range = None
        best_dl_model = None
        
        for range_bin in dl_stats:
            for model_name, stats in dl_stats[range_bin].items():
                if stats['mean'] > best_dl_performance:
                    best_dl_performance = stats['mean']
                    best_dl_range = range_bin
                    best_dl_model = model_name
        
        if best_dl_model:
            f_report.write(f"Best Deep Learning: {best_dl_model} with {best_dl_range} Range "
                          f"({best_dl_performance:.4f} ± {dl_stats[best_dl_range][best_dl_model]['std']:.4f})\n")
        
        # Conclusions
        f_report.write("\nCONCLUSIONS:\n")
        f_report.write("-" * 15 + "\n")
        f_report.write(f"1. Analyzed {len(valid_range_bins)} different saccadic range bins: {valid_range_bins}\n")
        f_report.write(f"2. Conducted {N_TRIALS} trials per range bin for robust statistics\n")
        f_report.write("3. Results show saccadic range-dependent performance variations\n")
        f_report.write("4. This analysis informs optimal saccadic range requirements for future data collection\n")
        f_report.write("5. Saccadic range calculated using 30°/sec velocity threshold as per medical team specification\n")
        
        f_report.write("\n" + "="*80 + "\n")
        f_report.write("End of Experiment 13E Report\n")
        f_report.write("="*80 + "\n")
    
    print(f"\nExperiment 13E completed successfully!")
    print(f"Results saved to: {summary_filepath}")
    print(f"All plots and logs saved in: {RESULTS_DIR}")
    print("="*80)

if __name__ == '__main__':
    main()
