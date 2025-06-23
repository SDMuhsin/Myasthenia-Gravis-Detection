import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import torch # Still used for tensor ops if needed, though model is not PyTorch
import warnings
import gc

# SOTA STRATEGY: Import new libraries for signal processing and gradient boosting
from scipy.fft import rfft
from scipy.signal import welch
import lightgbm as lgb

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')


# --- Configuration ---
# --- Basic Setup ---
BASE_DIR = './data'
EXP_NAME = 'EXP_4_Spectral_LGBM' # New experiment name for the new strategy
RESULTS_DIR = os.path.join('./results', EXP_NAME)
LOG_FILENAME = f'{EXP_NAME}_results_log.txt'
RANDOM_STATE = 42

# --- Data Loading & Preprocessing (Unchanged) ---
CLASS_DEFINITIONS = {
    'HC': {'path': 'Healthy control', 'label': 0},
    'MG': {'path': 'Definite MG', 'label': 1},
    'CNP3': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '3rd'), 'label': 2},
    'CNP4': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '4th'), 'label': 3},
    'CNP6': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '6th'), 'label': 4},
    'TAO': {'path': os.path.join('Non-MG diplopia (CNP, etc)', 'TAO'), 'label': 5},
}
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']
CSV_ENCODING = 'utf-16-le'
CSV_SEPARATOR = ','
MIN_SEQ_LEN_THRESHOLD = 50
CLASS_TO_EXCLUDE = 'TAO'

# --- Model & Training Hyperparameters ---
N_FOLDS = 5
# Note: EPOCHS, LR, etc. are not needed for LightGBM in the same way.
# LGBM has its own hyperparameters.
LGBM_PARAMS = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': RANDOM_STATE,
    'boosting_type': 'gbdt',
}


# --- Utility Functions (Unchanged) ---
def create_results_directory(dir_path=RESULTS_DIR):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"INFO: Created directory for results: {dir_path}")

def get_file_handler(filepath):
    return open(filepath, 'w', encoding='utf-8')

def log_message(f_handler, message, print_to_console=True):
    f_handler.write(message + '\n')
    f_handler.flush()
    if print_to_console:
        print(message)

# --- Data Loading (Unchanged) ---
def load_raw_sequences_and_labels(base_dir, class_definitions_dict, feature_columns_expected, encoding, separator, min_seq_len_threshold, f_out):
    log_message(f_out, "="*70 + "\nPhase: Data Loading\n" + "="*70, print_to_console=False)
    print("="*50 + "\nStarting Data Loading...\n" + "="*50)
    raw_items = []
    for class_name_key, class_details in class_definitions_dict.items():
        label, class_dir_abs = class_details['label'], os.path.join(base_dir, class_details['path'])
        log_message(f_out, f"\nProcessing Class: '{class_name_key}' (Label: {label}) from path: {class_dir_abs}", True)
        if not os.path.isdir(class_dir_abs):
            log_message(f_out, f"WARNING: Class directory not found: {class_dir_abs}", True); continue
        patient_dirs = [d for d in os.listdir(class_dir_abs) if os.path.isdir(os.path.join(class_dir_abs, d))]
        if not patient_dirs:
            log_message(f_out, f"INFO: No patient directories found in {class_dir_abs}", True); continue
        for patient_folder_name in tqdm(patient_dirs, desc=f"  Patients in {class_name_key}"):
            patient_id, patient_dir_path = f"{class_name_key}_{patient_folder_name}", os.path.join(class_dir_abs, patient_folder_name)
            for csv_file_path in glob.glob(os.path.join(patient_dir_path, '*.csv')):
                try:
                    df_full = pd.read_csv(csv_file_path, encoding=encoding, sep=separator)
                    df_full.columns = [col.strip() for col in df_full.columns]
                    if not all(col in df_full.columns for col in feature_columns_expected): continue
                    df_features = df_full[feature_columns_expected].copy()
                    if df_features.empty or len(df_features) < min_seq_len_threshold: continue
                    for col in df_features.columns: df_features.loc[:, col] = pd.to_numeric(df_features[col], errors='coerce')
                    if df_features.isnull().sum().sum() > 0.1 * df_features.size: continue
                    sequence_data = df_features.fillna(0).values.astype(np.float32)
                    raw_items.append({'data': sequence_data, 'label': label, 'patient_id': patient_id, 'class_name': class_name_key})
                except Exception as e:
                    print(f"ERROR processing {os.path.basename(csv_file_path)}: {e}. Skipping.")
    log_message(f_out, f"\nData loading complete. Loaded {len(raw_items)} raw sequences.\n" + "-"*70 + "\n", True)
    return raw_items

# --- Feature Engineering (Unchanged) ---
def engineer_features_from_raw_data(raw_items_dicts, original_feature_names, f_out):
    log_message(f_out, "="*70 + "\nPhase: Feature Engineering\n" + "="*70, print_to_console=False)
    print("\n" + "="*50 + "\nStarting Feature Engineering...\n" + "="*50)
    engineered_items, final_feature_names = [], []
    for item in tqdm(raw_items_dicts, desc="  Engineering Features"):
        df_original = pd.DataFrame(item['data'], columns=original_feature_names)
        df_engineered_parts = []
        for pos_col in ['LH', 'RH', 'LV', 'RV']:
            df_engineered_parts.append(df_original[pos_col].diff().fillna(0).rename(f'{pos_col}_Vel'))
        for eye_col, target_col, error_col_name in [('LH', 'TargetH', 'ErrorH_L'), ('RH', 'TargetH', 'ErrorH_R'), ('LV', 'TargetV', 'ErrorV_L'), ('RV', 'TargetV', 'ErrorV_R')]:
            df_engineered_parts.append((df_original[eye_col] - df_original[target_col]).rename(error_col_name))
        df_all_features = pd.concat([df_original, pd.concat(df_engineered_parts, axis=1)], axis=1)
        if not final_feature_names: final_feature_names = df_all_features.columns.tolist()
        engineered_items.append({**item, 'data': df_all_features.values.astype(np.float32), 'original_length': len(df_all_features)})
    final_num_channels = engineered_items[0]['data'].shape[1] if engineered_items else 0
    log_message(f_out, f"Feature engineering complete. Number of features: {final_num_channels}", True)
    log_message(f_out, f"Feature names: {final_feature_names}\n" + "-"*70 + "\n", True)
    return engineered_items, final_feature_names

# ========================================================================================
# --- PIVOTAL STRATEGY: Spectral Feature Extraction ---
# ========================================================================================
def extract_spectral_features(items, feature_names, f_out):
    """
    Transforms time-series data into a rich feature set using spectral analysis.
    This is a "many-to-one" transformation, converting a long sequence into a
    fixed-size vector of highly informative features.
    """
    log_message(f_out, "="*70 + "\nPhase: Spectral Feature Extraction\n" + "="*70, print_to_console=False)
    print("\n" + "="*50 + "\nStarting Spectral Feature Extraction...\n" + "="*50)
    
    feature_vectors = []
    spectral_feature_names = []

    # Assuming a sample rate of around 100 Hz for physiological data
    fs = 100.0

    for item in tqdm(items, desc="  Extracting Spectral Features"):
        sequence = item['data']
        all_channel_features = []
        
        for i in range(sequence.shape[1]): # Iterate over each of the 14 channels
            channel_data = sequence[:, i]
            
            # Calculate Power Spectral Density using Welch's method for a stable estimate
            freqs, psd = welch(channel_data, fs=fs, nperseg=256)
            psd = psd / np.sum(psd) if np.sum(psd) > 0 else psd # Normalize PSD

            if np.sum(psd) > 0:
                # Spectral Centroid: The "center of mass" of the spectrum
                spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
                # Spectral Bandwidth: The spread of the spectrum
                spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * psd))
            else:
                spectral_centroid = 0
                spectral_bandwidth = 0
            
            # Additional robust features
            mean_psd = np.mean(psd)
            std_psd = np.std(psd)
            
            all_channel_features.extend([spectral_centroid, spectral_bandwidth, mean_psd, std_psd])

        feature_vectors.append(all_channel_features)

    # Create descriptive names for the new features
    base_spectral_names = ['centroid', 'bandwidth', 'mean_psd', 'std_psd']
    for ch_name in feature_names:
        for stat_name in base_spectral_names:
            spectral_feature_names.append(f"{ch_name}_{stat_name}")
            
    feature_df = pd.DataFrame(feature_vectors, columns=spectral_feature_names)
    log_message(f_out, f"Spectral feature extraction complete. New feature shape: {feature_df.shape}", True)
    log_message(f_out, f"Total number of spectral features: {len(spectral_feature_names)}\n" + "-"*70 + "\n", True)
    
    return feature_df.values, spectral_feature_names


# --- Plotting Functions (Updated for LGBM) ---
def plot_loss_curves(history, fold, filepath):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train'], label='Training LogLoss')
    plt.plot(history['valid'], label='Validation LogLoss')
    plt.title(f'Fold {fold+1} - LightGBM Training History')
    plt.xlabel('Boosting Round')
    plt.ylabel('LogLoss')
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()

def plot_confusion_matrix(cm, class_names, title, filepath):
    plt.figure(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title); plt.ylabel('Actual'); plt.xlabel('Predicted'); plt.tight_layout(); plt.savefig(filepath); plt.close()


# --- Main Execution Block ---
if __name__ == '__main__':
    # 1. Setup
    np.random.seed(RANDOM_STATE)
    create_results_directory()
    f_log = get_file_handler(os.path.join(RESULTS_DIR, LOG_FILENAME))

    log_message(f_log, f"Experiment: {EXP_NAME} - Spectral Features with LightGBM Classifier")
    log_message(f_log, f"\n--- Configuration ---\nModel: LightGBM\n{LGBM_PARAMS}\n" + "-"*70)

    # 2. Data Loading and Initial Feature Engineering
    raw_items = load_raw_sequences_and_labels(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, CSV_ENCODING, CSV_SEPARATOR, MIN_SEQ_LEN_THRESHOLD, f_log)
    engineered_items, time_domain_feature_names = engineer_features_from_raw_data(raw_items, FEATURE_COLUMNS, f_log)
    
    # 3. Filter classes and create labels
    filtered_items = [item for item in engineered_items if item['class_name'] != CLASS_TO_EXCLUDE]
    remaining_class_names = sorted([c for c in CLASS_DEFINITIONS if c != CLASS_TO_EXCLUDE], key=lambda c: CLASS_DEFINITIONS[c]['label'])
    final_class_map_original_label_to_new = {CLASS_DEFINITIONS[name]['label']: i for i, name in enumerate(remaining_class_names)}
    num_classes = len(remaining_class_names)
    
    log_message(f_log, f"Filtered out '{CLASS_TO_EXCLUDE}'. {len(filtered_items)} sequences remaining.")
    log_message(f_log, f"Final class mapping: { {name: i for i, name in enumerate(remaining_class_names)} }\n" + "-"*70)

    # 4. SPECTRAL FEATURE EXTRACTION (The Core New Step)
    X_spectral, spectral_feature_names = extract_spectral_features(filtered_items, time_domain_feature_names, f_log)

    # Create final labels and groups for cross-validation
    y = np.array([final_class_map_original_label_to_new[item['label']] for item in filtered_items])
    groups = np.array([item['patient_id'] for item in filtered_items])
    
    # 5. Cross-Validation Loop with LightGBM
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    overall_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []}
    aggregated_cm = np.zeros((num_classes, num_classes))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_spectral, y, groups)):
        log_message(f_log, "\n" + "="*70 + f"\nStarting Fold {fold+1}/{N_FOLDS}\n" + "="*70)
        X_train, X_val = X_spectral[train_idx], X_spectral[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Data scaling is good practice, even for LGBM
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        log_message(f_log, f"Train size: {len(X_train)}, Validation size: {len(X_val)}.")

        # Calculate class weights for the model
        class_weights_values = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = {i: w for i, w in enumerate(class_weights_values)}

        # Initialize and train the LightGBM model
        model = lgb.LGBMClassifier(**LGBM_PARAMS, class_weight=class_weights_dict)
        
        # Use callbacks for early stopping and recording training history
        eval_set = [(X_train, y_train), (X_val, y_val)]
        callbacks = [
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(period=100)
        ]

        model.fit(X_train, y_train,
                  eval_set=eval_set,
                  eval_metric='multi_logloss',
                  callbacks=callbacks)

        # Evaluation
        y_pred = model.predict(X_val)
        
        log_message(f_log, f"\n--- Fold {fold+1} Results ---")
        log_message(f_log, f"Best iteration: {model.best_iteration_}")
        report_text = classification_report(y_val, y_pred, target_names=remaining_class_names, zero_division=0)
        log_message(f_log, report_text)
        
        report_dict = classification_report(y_val, y_pred, target_names=remaining_class_names, output_dict=True, zero_division=0)
        overall_metrics['accuracy'].append(report_dict['accuracy'])
        overall_metrics['precision'].append(report_dict['macro avg']['precision'])
        overall_metrics['recall'].append(report_dict['macro avg']['recall'])
        overall_metrics['f1-score'].append(report_dict['macro avg']['f1-score'])

        # Store and plot results
        cm = confusion_matrix(y_val, y_pred, labels=range(num_classes)); aggregated_cm += cm
        cm_path = os.path.join(RESULTS_DIR, f'{EXP_NAME}_fold_{fold+1}_confusion_matrix.png')
        plot_confusion_matrix(cm, remaining_class_names, f'Fold {fold+1} - Confusion Matrix', cm_path)
        
        loss_history = {
            'train': model.evals_result_['training']['multi_logloss'],
            'valid': model.evals_result_['valid_1']['multi_logloss']
        }
        loss_curve_path = os.path.join(RESULTS_DIR, f'{EXP_NAME}_fold_{fold+1}_loss_curve.png')
        plot_loss_curves(loss_history, fold, loss_curve_path)
        log_message(f_log, f"Plots for fold {fold+1} saved.")

        del model, scaler
        gc.collect()

    # 6. Final Aggregated Results
    log_message(f_log, "\n" + "="*70 + "\nCross-Validation Summary\n" + "="*70)
    log_message(f_log, f"Average Accuracy: {np.mean(overall_metrics['accuracy']):.4f} ± {np.std(overall_metrics['accuracy']):.4f}")
    log_message(f_log, f"Average Macro F1-Score: {np.mean(overall_metrics['f1-score']):.4f} ± {np.std(overall_metrics['f1-score']):.4f}")
    agg_cm_path = os.path.join(RESULTS_DIR, f'{EXP_NAME}_aggregated_confusion_matrix.png')
    plot_confusion_matrix(aggregated_cm, remaining_class_names, 'Aggregated Confusion Matrix (All Folds)', agg_cm_path)
    log_message(f_log, f"\nAggregated Confusion Matrix (sum over folds):\n{np.array2string(aggregated_cm, separator=', ')}")
    log_message(f_log, f"Aggregated confusion matrix plot saved to: {agg_cm_path}")
    log_message(f_log, "\n" + "="*70 + "\nExperiment Finished.\n" + "="*70)
    f_log.close()
    print(f"\nExperiment complete. All results saved in: {RESULTS_DIR}")

