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

from utils.data_loading import load_raw_sequences_and_labels, engineer_and_aggregate_features
from utils.modeling import (create_results_directory, train_and_evaluate_single_model, 
                           get_best_statistical_models)
from utils.deep_learning import (get_small_dl_models, SaccadeStandardScaler, EarlyStopper,
                                calculate_class_weights, SaccadeDataset, train_epoch, 
                                evaluate_epoch, plot_loss_curves, plot_dl_confusion_matrix,
                                subsample_data)

# --- Configuration ---
BASE_DIR = './data'

# Binary classification: HC vs MG (including Probable MG)
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
RESULTS_DIR = './results/exp_13_a'
EXP_PREFIX = 'EXP_13_BINARY_A_'
NUMERICAL_SUMMARY_FILENAME = f'{EXP_PREFIX}numerical_summary.txt'
RANDOM_STATE = 42

# Deep Learning Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU 1 as specified
SUBSAMPLE_FACTOR = 10  # 10x subsampling for quick results
TARGET_SEQ_LEN_PERCENTILE = 90  # Reduced for memory efficiency
EPOCHS = 50  # Reduced for quick results
BATCH_SIZE = 16  # Small batch size
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# Suppress warnings for cleaner output
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

def run_statistical_models(master_df, numerical_features, results_dir, f_out):
    """Run statistical models for binary classification."""
    f_out.write("\n" + "="*80 + "\n")
    f_out.write("Phase: Statistical Models for Binary Classification\n")
    f_out.write("="*80 + "\n")
    print("\n" + "="*50 + "\nRunning Statistical Models...\n" + "="*50)
    
    X = master_df[numerical_features]
    y = master_df['label']
    
    models = get_best_statistical_models()
    results = {}
    
    for model_name, model in models.items():
        accuracy, report, cm = train_and_evaluate_single_model(
            X, y, numerical_features, [], model_name, model, 
            ORDERED_CLASS_NAMES, MODEL_CLASS_LABELS, results_dir, f_out, 
            suffix="_Binary", random_state=RANDOM_STATE
        )
        results[model_name] = accuracy
    
    f_out.write("\nStatistical Models Summary:\n")
    for model_name, accuracy in results.items():
        f_out.write(f"  {model_name}: {accuracy:.4f}\n")
    
    return results

def run_deep_learning_models(binary_items, num_features, results_dir, f_out):
    """Run deep learning models with subsampling for quick results."""
    f_out.write("\n" + "="*80 + "\n")
    f_out.write("Phase: Deep Learning Models (with 10x Subsampling)\n")
    f_out.write("="*80 + "\n")
    print("\n" + "="*50 + "\nRunning Deep Learning Models...\n" + "="*50)
    
    # Subsample data for quick results
    subsampled_items = subsample_data(binary_items, SUBSAMPLE_FACTOR)
    f_out.write(f"Subsampled data: {len(subsampled_items)} samples (factor: {SUBSAMPLE_FACTOR}x)\n")
    print(f"Using {len(subsampled_items)} subsampled sequences for deep learning")
    
    # Determine target sequence length
    lengths = [item['data'].shape[0] for item in subsampled_items]
    target_seq_len = int(np.percentile(lengths, TARGET_SEQ_LEN_PERCENTILE))
    f_out.write(f"Target sequence length ({TARGET_SEQ_LEN_PERCENTILE}th percentile): {target_seq_len}\n")
    
    # Prepare data arrays
    X = np.array(subsampled_items, dtype=object)
    y = np.array([item['label'] for item in subsampled_items])
    groups = np.array([item['patient_id'] for item in subsampled_items])
    
    # Label mapping for binary classification
    label_map = {0: 0, 1: 1}  # HC: 0, MG: 1
    
    # Get models
    models = get_small_dl_models(num_features, 2, target_seq_len)  # 2 classes for binary
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)  # Reduced folds for speed
    dl_results = {}
    
    for model_name, model_class in models.items():
        f_out.write(f"\n--- Deep Learning Model: {model_name} ---\n")
        print(f"\nTraining {model_name}...")
        
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
        print(f"  {model_name}: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
    
    f_out.write("\nDeep Learning Models Summary:\n")
    for model_name, accuracy in dl_results.items():
        f_out.write(f"  {model_name}: {accuracy:.4f}\n")
    
    return dl_results

def main():
    """Main execution function."""
    print("="*80)
    print("Starting Experiment 13A: Binary Classification (HC vs MG)")
    print("Including 'Probable MG' in MG class")
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
        f_report.write("Experiment 13A: Binary Classification (HC vs MG) - Numerical Summary\n")
        f_report.write("="*80 + "\n")
        f_report.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_report.write(f"Base Directory: {BASE_DIR}\n")
        f_report.write(f"Classes: {ORDERED_CLASS_NAMES} (Binary Classification)\n")
        f_report.write(f"Device: {DEVICE}\n")
        f_report.write(f"Random State: {RANDOM_STATE}\n")
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
        
        # 2. Prepare Binary Classification Data
        binary_items = prepare_binary_data(raw_items_list, f_report)
        
        # 3. Feature Engineering and Aggregation
        master_df, numerical_features = engineer_and_aggregate_features(
            binary_items, FEATURE_COLUMNS, f_report
        )
        
        if master_df.empty:
            f_report.write("\nCRITICAL: Aggregated features DataFrame is empty. Experiments cannot proceed.\n")
            print("CRITICAL: Empty DataFrame. Exiting.")
            return
        
        f_report.write(f"Final dataset shape: {master_df.shape}\n")
        f_report.write(f"Number of numerical features: {len(numerical_features)}\n\n")
        
        # 4. Run Statistical Models
        stat_results = run_statistical_models(master_df, numerical_features, RESULTS_DIR, f_report)
        
        # 5. Run Deep Learning Models (with subsampling)
        # Calculate actual number of features from the engineered data
        sample_item = binary_items[0]
        num_features = sample_item['data'].shape[1]
        f_report.write(f"Number of features for deep learning: {num_features}\n")
        dl_results = run_deep_learning_models(binary_items, num_features, RESULTS_DIR, f_report)
        
        # 6. Final Summary
        f_report.write("\n" + "="*80 + "\n")
        f_report.write("EXPERIMENT 13A FINAL RESULTS SUMMARY\n")
        f_report.write("="*80 + "\n")
        f_report.write("Statistical Models:\n")
        for model_name, accuracy in stat_results.items():
            f_report.write(f"  {model_name}: {accuracy:.4f}\n")
        f_report.write("\nDeep Learning Models (10x subsampled):\n")
        for model_name, accuracy in dl_results.items():
            f_report.write(f"  {model_name}: {accuracy:.4f}\n")
        
        # Find best models
        best_stat_model = max(stat_results.items(), key=lambda x: x[1])
        best_dl_model = max(dl_results.items(), key=lambda x: x[1])
        
        f_report.write(f"\nBest Statistical Model: {best_stat_model[0]} ({best_stat_model[1]:.4f})\n")
        f_report.write(f"Best Deep Learning Model: {best_dl_model[0]} ({best_dl_model[1]:.4f})\n")
        
        f_report.write("\n" + "="*80 + "\n")
        f_report.write("End of Experiment 13A Report\n")
        f_report.write("="*80 + "\n")
    
    print(f"\nExperiment 13A completed successfully!")
    print(f"Results saved to: {summary_filepath}")
    print(f"All plots and logs saved in: {RESULTS_DIR}")
    print("="*80)

if __name__ == '__main__':
    main()
