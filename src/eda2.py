import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings

# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=FutureWarning)

# ========================================================================================
# --- Configuration ---
# ========================================================================================
# --- Basic Setup ---
BASE_DIR = './data'
EXP_NAME = 'EDA_02_SegmentFeatureAnalysis'
RESULTS_DIR = os.path.join('./results', EXP_NAME)
LOG_FILENAME = f'{EXP_NAME}_results_log.txt'
RANDOM_STATE = 42
SAMPLING_RATE_HZ = 1000  # Assuming 1000Hz

# --- Data Loading & Preprocessing ---
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

# --- Segmentation Hyperparameters ---
SEGMENT_WINDOW_BEFORE = 100  # ms
SEGMENT_WINDOW_AFTER = 400   # ms
FIXED_SEGMENT_LENGTH = SEGMENT_WINDOW_BEFORE + SEGMENT_WINDOW_AFTER

# --- Model & CV Hyperparameters ---
N_FOLDS = 5
LGB_PARAMS = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 5, # Will be updated dynamically
    'n_estimators': 1000,
    'learning_rate': 0.05,
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

# ========================================================================================
# --- Utility and Setup Functions ---
# ========================================================================================
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

# ========================================================================================
# --- Data Loading and Segmentation (as before) ---
# ========================================================================================
def load_and_engineer_raw_data(base_dir, class_defs, feature_cols, encoding, sep, min_len, f_out):
    """Loads and performs initial feature engineering, returning a list of DataFrames."""
    log_message(f_out, "="*70 + "\nPhase 1: Loading and Engineering Raw Trial Data\n" + "="*70)
    all_trials = []
    for class_name, details in class_defs.items():
        class_path = os.path.join(base_dir, details['path'])
        if not os.path.isdir(class_path): continue
        
        patient_dirs = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]
        for p_name in tqdm(patient_dirs, desc=f"Processing {class_name}"):
            p_id = f"{class_name}_{p_name}"
            for csv_path in glob.glob(os.path.join(class_path, p_name, '*.csv')):
                try:
                    df = pd.read_csv(csv_path, encoding=encoding, sep=sep)
                    df.columns = [c.strip() for c in df.columns]
                    if not all(c in df.columns for c in feature_cols) or len(df) < min_len:
                        continue

                    df_feat = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(method='ffill').fillna(0)
                    
                    # Engineer velocity and error features
                    df_feat['LH_Vel'] = df_feat['LH'].diff().fillna(0) * SAMPLING_RATE_HZ
                    df_feat['RH_Vel'] = df_feat['RH'].diff().fillna(0) * SAMPLING_RATE_HZ
                    df_feat['LV_Vel'] = df_feat['LV'].diff().fillna(0) * SAMPLING_RATE_HZ
                    df_feat['RV_Vel'] = df_feat['RV'].diff().fillna(0) * SAMPLING_RATE_HZ
                    df_feat['ErrorH_L'] = df_feat['LH'] - df_feat['TargetH']
                    df_feat['ErrorH_R'] = df_feat['RH'] - df_feat['TargetH']
                    df_feat['ErrorV_L'] = df_feat['LV'] - df_feat['TargetV']
                    df_feat['ErrorV_R'] = df_feat['RV'] - df_feat['TargetV']
                    
                    all_trials.append({
                        'df': df_feat, 'label': details['label'], 'patient_id': p_id,
                        'class_name': class_name
                    })
                except Exception:
                    continue
    log_message(f_out, f"Loaded and engineered {len(all_trials)} trials.")
    return all_trials

# ========================================================================================
# --- DIAGNOSTIC: Feature Extraction from Segments ---
# ========================================================================================
def extract_features_from_segment(segment_df):
    """
    This is the core of the diagnostic script. For a given segment DataFrame,
    it calculates a variety of statistical and kinematic features.
    """
    if segment_df.empty:
        return None
    
    features = {}
    
    # Analyze Horizontal (H) and Vertical (V) components separately
    for eye, direction in [('L', 'H'), ('R', 'H'), ('L', 'V'), ('R', 'V')]:
        pos_col = f'{eye}{direction}'
        vel_col = f'{pos_col}_Vel'
        error_col = f'Error{direction}_{eye}'
        
        # Saccade Amplitude: Total change in position
        amplitude = abs(segment_df[pos_col].iloc[-1] - segment_df[pos_col].iloc[0])
        features[f'amp_{pos_col}'] = amplitude
        
        # Peak Velocity & Time to Peak
        peak_velocity = segment_df[vel_col].abs().max()
        time_to_peak = segment_df[vel_col].abs().idxmax()
        features[f'peak_vel_{pos_col}'] = peak_velocity
        features[f'time_to_peak_{pos_col}'] = time_to_peak / SAMPLING_RATE_HZ # in seconds
        
        # Main Sequence Ratio (Peak Velocity / Amplitude)
        features[f'main_seq_ratio_{pos_col}'] = peak_velocity / (amplitude + 1e-6)
        
        # Error metrics
        final_error = segment_df[error_col].iloc[-1]
        mean_error_last_100ms = segment_df[error_col].iloc[-100:].mean()
        std_error_last_100ms = segment_df[error_col].iloc[-100:].std()
        features[f'final_error_{pos_col}'] = final_error
        features[f'mean_err_end_{pos_col}'] = mean_error_last_100ms
        features[f'std_err_end_{pos_col}'] = std_error_last_100ms # Post-saccadic stability
        
        # General statistics of the signal
        features[f'mean_{pos_col}'] = segment_df[pos_col].mean()
        features[f'std_{pos_col}'] = segment_df[pos_col].std()
        features[f'skew_{pos_col}'] = segment_df[pos_col].skew()
        features[f'kurt_{pos_col}'] = segment_df[pos_col].kurt()

    return features

def process_all_trials_to_feature_df(all_trials, f_out):
    """
    Iterates through all trials, segments them, extracts features from each segment,
    and compiles everything into a single DataFrame for modeling.
    """
    log_message(f_out, "\n" + "="*70 + "\nPhase 2: Segmenting Trials and Extracting Statistical Features\n" + "="*70)
    all_segment_features = []
    
    for trial in tqdm(all_trials, desc="Extracting segment features"):
        df = trial['df']
        target_changes = df[['TargetH', 'TargetV']].diff().abs().sum(axis=1) > 0.1
        saccade_start_indices = target_changes[target_changes].index
        
        for i, start_idx in enumerate(saccade_start_indices):
            window_start = max(0, start_idx - SEGMENT_WINDOW_BEFORE)
            window_end = min(len(df), start_idx + SEGMENT_WINDOW_AFTER)
            segment_df = df.iloc[window_start:window_end]

            if len(segment_df) < FIXED_SEGMENT_LENGTH: # Skip incomplete segments at the end
                continue

            segment_features = extract_features_from_segment(segment_df)
            if segment_features:
                segment_features['segment_id'] = f"{trial['patient_id']}_{i}"
                segment_features['patient_id'] = trial['patient_id']
                segment_features['label'] = trial['label']
                segment_features['class_name'] = trial['class_name']
                all_segment_features.append(segment_features)

    feature_df = pd.DataFrame(all_segment_features).fillna(0)
    log_message(f_out, f"Created a feature matrix with {len(feature_df)} segments and {len(feature_df.columns)} columns.")
    return feature_df

# ========================================================================================
# --- Main Execution Block ---
# ========================================================================================
if __name__ == '__main__':
    # 1. Setup
    create_results_directory()
    f_log = get_file_handler(os.path.join(RESULTS_DIR, LOG_FILENAME))
    log_message(f_log, f"Starting Diagnostic Analysis: {EXP_NAME}")

    # 2. Data Loading, Engineering, and Feature Extraction
    all_trials = load_and_engineer_raw_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, CSV_ENCODING, CSV_SEPARATOR, MIN_SEQ_LEN_THRESHOLD, f_log)
    feature_df = process_all_trials_to_feature_df(all_trials, f_log)
    
    # 3. Prepare data for modeling
    df_model = feature_df[feature_df['class_name'] != CLASS_TO_EXCLUDE].copy()
    
    # Remap labels to be contiguous (0, 1, 2, ...)
    unique_labels = sorted(df_model['label'].unique())
    label_map = {original: new for new, original in enumerate(unique_labels)}
    df_model['label'] = df_model['label'].map(label_map)
    
    class_names = [name for name, details in CLASS_DEFINITIONS.items() if details['label'] in unique_labels]
    class_names = [name for _, name in sorted(zip(unique_labels, class_names))] # Sort by original label order
    
    LGB_PARAMS['num_class'] = len(class_names)
    log_message(f_log, f"\nClasses for modeling: {class_names}")

    y = df_model['label']
    X = df_model.drop(columns=['label', 'class_name', 'segment_id', 'patient_id'])
    groups = df_model['patient_id']

    # 4. Cross-Validation with LightGBM
    log_message(f_log, "\n" + "="*70 + "\nPhase 3: Cross-Validating with LightGBM on Segment Features\n" + "="*70)
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros(len(df_model))
    feature_importances = pd.DataFrame(index=X.columns)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y, groups)):
        log_message(f_log, f"--- Starting Fold {fold+1}/{N_FOLDS} ---")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(X_train_scaled, y_train,
                  eval_set=[(X_val_scaled, y_val)],
                  eval_metric='multi_logloss',
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        
        preds = model.predict(X_val_scaled)
        oof_preds[val_idx] = preds
        
        feature_importances[f'fold_{fold+1}'] = model.feature_importances_
        acc = accuracy_score(y_val, preds)
        log_message(f_log, f"Fold {fold+1} Accuracy: {acc:.4f}")

    # 5. Final Report
    log_message(f_log, "\n" + "="*70 + "\nFinal Cross-Validation Results\n" + "="*70)
    overall_accuracy = accuracy_score(y, oof_preds)
    log_message(f_log, f"Overall CV Accuracy on Segments: {overall_accuracy:.4f}")
    
    report_text = classification_report(y, oof_preds, target_names=class_names, zero_division=0)
    log_message(f_log, "\nClassification Report (OOF Predictions):\n" + report_text)
    
    # Feature Importance Analysis
    feature_importances['mean'] = feature_importances.mean(axis=1)
    feature_importances.sort_values('mean', ascending=False, inplace=True)
    
    log_message(f_log, "\n" + "="*70 + "\nTop 20 Most Important Features (Mean over folds)\n" + "="*70)
    log_message(f_log, feature_importances['mean'].head(20).to_string())
    
    plt.figure(figsize=(12, 15))
    sns.barplot(x=feature_importances['mean'].head(30), y=feature_importances.index[:30])
    plt.title('Top 30 Feature Importances (LGBM)')
    plt.xlabel('Mean Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importances.png'))
    plt.close()
    log_message(f_log, f"\nFeature importance plot saved to {RESULTS_DIR}")
    
    log_message(f_log, "\nDiagnostic script finished.")
    f_log.close()
    print(f"\nDiagnostic analysis complete. Results saved in: {RESULTS_DIR}")

