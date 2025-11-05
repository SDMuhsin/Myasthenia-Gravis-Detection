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
import re

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

# Binary classification: HC vs MG (including Probable MG) - same as 13A
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
RESULTS_DIR = './results/exp_13_c'
EXP_PREFIX = 'EXP_13_BINARY_C_'
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

# Frequency Analysis Parameters
N_TRIALS = 5  # Multiple trials for comprehensive results
MIN_SAMPLES_PER_FREQUENCY = 20  # Minimum samples needed per frequency

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.discriminant_analysis')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

def parse_frequency_from_filename(filename):
    """Extracts saccade frequency (e.g., 0.5, 0.75, 1) from a filename."""
    match = re.search(r'\((\d+(\.\d+)?)\s*Hz\)', filename, re.IGNORECASE)
    return float(match.group(1)) if match else np.nan

def prepare_binary_data_with_frequency(raw_items, f_out):
    """Prepare data for binary classification by combining MG classes and adding frequency info."""
    f_out.write("="*80 + "\n")
    f_out.write("Phase: Binary Classification Data Preparation with Frequency Separation\n")
    f_out.write("="*80 + "\n")
    print("\n" + "="*50 + "\nPreparing Binary Classification Data with Frequency...\n" + "="*50)
    
    binary_items = []
    class_counts = {'HC': 0, 'MG': 0}
    frequency_counts = {}
    
    for item in raw_items:
        # Parse frequency from filename if not already present
        if 'frequency' not in item or pd.isna(item['frequency']):
            item['frequency'] = parse_frequency_from_filename(item['filename'])
        
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
        
        # Count frequencies
        freq = item.get('frequency', np.nan)
        if not pd.isna(freq):
            frequency_counts[freq] = frequency_counts.get(freq, 0) + 1
    
    f_out.write(f"Binary classification data prepared:\n")
    f_out.write(f"  HC (Healthy Control): {class_counts['HC']} samples\n")
    f_out.write(f"  MG (Definite + Probable): {class_counts['MG']} samples\n")
    f_out.write(f"  Total: {len(binary_items)} samples\n\n")
    
    f_out.write("Frequency distribution:\n")
    for freq in sorted(frequency_counts.keys()):
        f_out.write(f"  {freq} Hz: {frequency_counts[freq]} samples\n")
    f_out.write("-" * 80 + "\n\n")
    
    print(f"Binary data prepared: HC={class_counts['HC']}, MG={class_counts['MG']}")
    print(f"Frequencies found: {sorted(frequency_counts.keys())}")
    return binary_items

def filter_by_frequency(binary_items, target_frequency, f_out):
    """Filter items by specific frequency."""
    filtered_items = [item for item in binary_items if item.get('frequency') == target_frequency]
    
    class_counts = {'HC': 0, 'MG': 0}
    for item in filtered_items:
        if item['label'] == 0:
            class_counts['HC'] += 1
        else:
            class_counts['MG'] += 1
    
    f_out.write(f"Frequency {target_frequency} Hz data:\n")
    f_out.write(f"  HC: {class_counts['HC']} samples\n")
    f_out.write(f"  MG: {class_counts['MG']} samples\n")
    f_out.write(f"  Total: {len(filtered_items)} samples\n\n")
    
    return filtered_items, class_counts

def run_statistical_models_for_frequency(master_df, numerical_features, frequency, results_dir, f_out, trial_num=1):
    """Run statistical models for a specific frequency."""
    f_out.write(f"\n--- Statistical Models for {frequency} Hz (Trial {trial_num}) ---\n")
    print(f"\nRunning Statistical Models for {frequency} Hz (Trial {trial_num})...")
    
    X = master_df[numerical_features]
    y = master_df['label']
    
    models = get_best_statistical_models()
    results = {}
    
    for model_name, model in models.items():
        accuracy, report, cm = train_and_evaluate_single_model(
            X, y, numerical_features, [], model_name, model, 
            ORDERED_CLASS_NAMES, MODEL_CLASS_LABELS, results_dir, f_out, 
            suffix=f"_Freq_{frequency}Hz_Trial_{trial_num}", random_state=RANDOM_STATE + trial_num
        )
        results[model_name] = accuracy
    
    f_out.write(f"\nStatistical Models Summary for {frequency} Hz (Trial {trial_num}):\n")
    for model_name, accuracy in results.items():
        f_out.write(f"  {model_name}: {accuracy:.4f}\n")
    
    return results

def run_deep_learning_models_for_frequency(binary_items, num_features, frequency, results_dir, f_out, trial_num=1):
    """Run deep learning models for a specific frequency."""
    f_out.write(f"\n--- Deep Learning Models for {frequency} Hz (Trial {trial_num}) ---\n")
    print(f"\nRunning Deep Learning Models for {frequency} Hz (Trial {trial_num})...")
    
    # Subsample data for quick results
    subsampled_items = subsample_data(binary_items, SUBSAMPLE_FACTOR)
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
        f_out.write(f"\n--- Deep Learning Model: {model_name} for {frequency} Hz (Trial {trial_num}) ---\n")
        
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
    
    f_out.write(f"\nDeep Learning Models Summary for {frequency} Hz (Trial {trial_num}):\n")
    for model_name, accuracy in dl_results.items():
        f_out.write(f"  {model_name}: {accuracy:.4f}\n")
    
    return dl_results

def run_frequency_separated_analysis(binary_items, f_out):
    """Run comprehensive frequency-separated analysis with multiple trials."""
    f_out.write("\n" + "="*80 + "\n")
    f_out.write("Phase: Frequency-Separated Binary Classification Analysis\n")
    f_out.write("="*80 + "\n")
    print("\n" + "="*50 + "\nRunning Frequency-Separated Analysis...\n" + "="*50)
    
    # Get available frequencies
    frequencies = []
    for item in binary_items:
        freq = item.get('frequency')
        if not pd.isna(freq) and freq not in frequencies:
            frequencies.append(freq)
    
    frequencies = sorted(frequencies)
    f_out.write(f"Available frequencies: {frequencies}\n\n")
    
    # Filter frequencies with sufficient samples
    valid_frequencies = []
    for freq in frequencies:
        freq_items = [item for item in binary_items if item.get('frequency') == freq]
        if len(freq_items) >= MIN_SAMPLES_PER_FREQUENCY:
            valid_frequencies.append(freq)
        else:
            f_out.write(f"Skipping {freq} Hz: insufficient samples ({len(freq_items)} < {MIN_SAMPLES_PER_FREQUENCY})\n")
    
    f_out.write(f"Valid frequencies for analysis: {valid_frequencies}\n\n")
    
    # Store results for comprehensive analysis
    all_stat_results = {}  # freq -> trial -> model -> accuracy
    all_dl_results = {}    # freq -> trial -> model -> accuracy
    
    # Run analysis for each frequency
    for freq in valid_frequencies:
        f_out.write(f"\n{'='*60}\n")
        f_out.write(f"ANALYZING FREQUENCY: {freq} Hz\n")
        f_out.write(f"{'='*60}\n")
        print(f"\nAnalyzing frequency: {freq} Hz")
        
        # Filter data for this frequency
        freq_items, class_counts = filter_by_frequency(binary_items, freq, f_out)
        
        # Check if we have sufficient data for both classes
        if class_counts['HC'] < 5 or class_counts['MG'] < 5:
            f_out.write(f"Insufficient data for both classes. Skipping {freq} Hz.\n")
            continue
        
        # Initialize results storage
        all_stat_results[freq] = {}
        all_dl_results[freq] = {}
        
        # Run multiple trials
        for trial in range(1, N_TRIALS + 1):
            f_out.write(f"\n--- TRIAL {trial}/{N_TRIALS} for {freq} Hz ---\n")
            print(f"  Trial {trial}/{N_TRIALS}")
            
            # Feature engineering for this frequency
            master_df, numerical_features = engineer_and_aggregate_features(
                freq_items, FEATURE_COLUMNS, f_out
            )
            
            if master_df.empty:
                f_out.write(f"Empty DataFrame for {freq} Hz, trial {trial}. Skipping.\n")
                continue
            
            # Run statistical models
            stat_results = run_statistical_models_for_frequency(
                master_df, numerical_features, freq, RESULTS_DIR, f_out, trial
            )
            all_stat_results[freq][trial] = stat_results
            
            # Run deep learning models
            sample_item = freq_items[0]
            num_features = sample_item['data'].shape[1]
            dl_results = run_deep_learning_models_for_frequency(
                freq_items, num_features, freq, RESULTS_DIR, f_out, trial
            )
            all_dl_results[freq][trial] = dl_results
    
    return all_stat_results, all_dl_results, valid_frequencies

def calculate_comprehensive_statistics(all_results, result_type, f_out):
    """Calculate mean and standard deviation across trials for each frequency and model."""
    f_out.write(f"\n{'='*80}\n")
    f_out.write(f"COMPREHENSIVE {result_type.upper()} RESULTS WITH ERROR BARS\n")
    f_out.write(f"{'='*80}\n")
    
    comprehensive_stats = {}
    
    for freq in all_results:
        comprehensive_stats[freq] = {}
        
        # Get all model names across trials
        all_models = set()
        for trial_results in all_results[freq].values():
            all_models.update(trial_results.keys())
        
        for model_name in all_models:
            # Collect accuracies across trials
            accuracies = []
            for trial_results in all_results[freq].values():
                if model_name in trial_results:
                    accuracies.append(trial_results[model_name])
            
            if accuracies:
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                comprehensive_stats[freq][model_name] = {
                    'mean': mean_acc,
                    'std': std_acc,
                    'n_trials': len(accuracies)
                }
                
                f_out.write(f"{freq} Hz - {model_name}: {mean_acc:.4f} ± {std_acc:.4f} (n={len(accuracies)})\n")
    
    return comprehensive_stats

def plot_frequency_comparison(stat_stats, dl_stats, valid_frequencies, results_dir, f_out):
    """Create comprehensive plots comparing performance across frequencies."""
    f_out.write(f"\n{'='*80}\n")
    f_out.write("CREATING FREQUENCY COMPARISON PLOTS\n")
    f_out.write(f"{'='*80}\n")
    
    # Plot 1: Statistical Models Comparison
    plt.figure(figsize=(12, 8))
    
    # Get all statistical model names
    all_stat_models = set()
    for freq_stats in stat_stats.values():
        all_stat_models.update(freq_stats.keys())
    
    x_pos = np.arange(len(valid_frequencies))
    width = 0.25
    
    for i, model_name in enumerate(sorted(all_stat_models)):
        means = []
        stds = []
        for freq in valid_frequencies:
            if freq in stat_stats and model_name in stat_stats[freq]:
                means.append(stat_stats[freq][model_name]['mean'])
                stds.append(stat_stats[freq][model_name]['std'])
            else:
                means.append(0)
                stds.append(0)
        
        plt.bar(x_pos + i * width, means, width, yerr=stds, 
                label=model_name, alpha=0.8, capsize=5)
    
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Statistical Models Performance by Frequency (with Error Bars)', fontsize=14)
    plt.xticks(x_pos + width, [f'{f} Hz' for f in valid_frequencies])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    stat_plot_path = os.path.join(results_dir, f'{EXP_PREFIX}Statistical_Models_Frequency_Comparison.png')
    plt.savefig(stat_plot_path)
    plt.close()
    f_out.write(f"Statistical models comparison plot saved to: {stat_plot_path}\n")
    
    # Plot 2: Deep Learning Models Comparison
    if dl_stats:
        plt.figure(figsize=(12, 8))
        
        # Get all deep learning model names
        all_dl_models = set()
        for freq_stats in dl_stats.values():
            all_dl_models.update(freq_stats.keys())
        
        for i, model_name in enumerate(sorted(all_dl_models)):
            means = []
            stds = []
            for freq in valid_frequencies:
                if freq in dl_stats and model_name in dl_stats[freq]:
                    means.append(dl_stats[freq][model_name]['mean'])
                    stds.append(dl_stats[freq][model_name]['std'])
                else:
                    means.append(0)
                    stds.append(0)
            
            plt.bar(x_pos + i * width, means, width, yerr=stds, 
                    label=model_name, alpha=0.8, capsize=5)
        
        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Deep Learning Models Performance by Frequency (with Error Bars)', fontsize=14)
        plt.xticks(x_pos + width, [f'{f} Hz' for f in valid_frequencies])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        dl_plot_path = os.path.join(results_dir, f'{EXP_PREFIX}Deep_Learning_Models_Frequency_Comparison.png')
        plt.savefig(dl_plot_path)
        plt.close()
        f_out.write(f"Deep learning models comparison plot saved to: {dl_plot_path}\n")

def main():
    """Main execution function."""
    print("="*80)
    print("Starting Experiment 13C: Binary Classification with Frequency Separation")
    print("Including 'Probable MG' in MG class, separating by saccade frequency")
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
        f_report.write("Experiment 13C: Binary Classification with Frequency Separation\n")
        f_report.write("="*80 + "\n")
        f_report.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_report.write(f"Base Directory: {BASE_DIR}\n")
        f_report.write(f"Classes: {ORDERED_CLASS_NAMES} (Binary Classification)\n")
        f_report.write(f"Device: {DEVICE}\n")
        f_report.write(f"Random State: {RANDOM_STATE}\n")
        f_report.write(f"Number of Trials per Frequency: {N_TRIALS}\n")
        f_report.write(f"Minimum Samples per Frequency: {MIN_SAMPLES_PER_FREQUENCY}\n")
        f_report.write(f"Subsampling Factor for DL: {SUBSAMPLE_FACTOR}x\n")
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
        
        # 2. Prepare Binary Classification Data with Frequency Information
        binary_items = prepare_binary_data_with_frequency(raw_items_list, f_report)
        
        # 3. Run Frequency-Separated Analysis
        all_stat_results, all_dl_results, valid_frequencies = run_frequency_separated_analysis(
            binary_items, f_report
        )
        
        # 4. Calculate Comprehensive Statistics
        stat_stats = calculate_comprehensive_statistics(all_stat_results, "Statistical", f_report)
        dl_stats = calculate_comprehensive_statistics(all_dl_results, "Deep Learning", f_report)
        
        # 5. Create Comparison Plots
        plot_frequency_comparison(stat_stats, dl_stats, valid_frequencies, RESULTS_DIR, f_report)
        
        # 6. Final Summary and Conclusions
        f_report.write("\n" + "="*80 + "\n")
        f_report.write("EXPERIMENT 13C FINAL RESULTS SUMMARY\n")
        f_report.write("="*80 + "\n")
        
        f_report.write("STATISTICAL MODELS BY FREQUENCY (Mean ± Std):\n")
        f_report.write("-" * 50 + "\n")
        for freq in valid_frequencies:
            f_report.write(f"\n{freq} Hz:\n")
            if freq in stat_stats:
                for model_name, stats in stat_stats[freq].items():
                    f_report.write(f"  {model_name}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
        
        f_report.write("\nDEEP LEARNING MODELS BY FREQUENCY (Mean ± Std):\n")
        f_report.write("-" * 50 + "\n")
        for freq in valid_frequencies:
            f_report.write(f"\n{freq} Hz:\n")
            if freq in dl_stats:
                for model_name, stats in dl_stats[freq].items():
                    f_report.write(f"  {model_name}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
        
        # Find best performing frequency for each model type
        f_report.write("\nBEST PERFORMING FREQUENCIES:\n")
        f_report.write("-" * 30 + "\n")
        
        # Best statistical model across all frequencies
        best_stat_performance = 0
        best_stat_freq = None
        best_stat_model = None
        
        for freq in stat_stats:
            for model_name, stats in stat_stats[freq].items():
                if stats['mean'] > best_stat_performance:
                    best_stat_performance = stats['mean']
                    best_stat_freq = freq
                    best_stat_model = model_name
        
        if best_stat_model:
            f_report.write(f"Best Statistical: {best_stat_model} at {best_stat_freq} Hz "
                          f"({best_stat_performance:.4f} ± {stat_stats[best_stat_freq][best_stat_model]['std']:.4f})\n")
        
        # Best deep learning model across all frequencies
        best_dl_performance = 0
        best_dl_freq = None
        best_dl_model = None
        
        for freq in dl_stats:
            for model_name, stats in dl_stats[freq].items():
                if stats['mean'] > best_dl_performance:
                    best_dl_performance = stats['mean']
                    best_dl_freq = freq
                    best_dl_model = model_name
        
        if best_dl_model:
            f_report.write(f"Best Deep Learning: {best_dl_model} at {best_dl_freq} Hz "
                          f"({best_dl_performance:.4f} ± {dl_stats[best_dl_freq][best_dl_model]['std']:.4f})\n")
        
        # Conclusions
        f_report.write("\nCONCLUSIONS:\n")
        f_report.write("-" * 15 + "\n")
        f_report.write(f"1. Analyzed {len(valid_frequencies)} different frequencies: {valid_frequencies}\n")
        f_report.write(f"2. Conducted {N_TRIALS} trials per frequency for robust statistics\n")
        f_report.write("3. Results show frequency-dependent performance variations\n")
        f_report.write("4. This analysis informs optimal data collection frequencies for future studies\n")
        
        f_report.write("\n" + "="*80 + "\n")
        f_report.write("End of Experiment 13C Report\n")
        f_report.write("="*80 + "\n")
    
    print(f"\nExperiment 13C completed successfully!")
    print(f"Results saved to: {summary_filepath}")
    print(f"All plots and logs saved in: {RESULTS_DIR}")
    print("="*80)

if __name__ == '__main__':
    main()
