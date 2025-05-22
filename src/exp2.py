import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import iqr # For Interquartile Range, though pd.quantile works too
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

# --- Configuration ---
BASE_DIR = './data'
CLASS_MAPPING = {'Healthy control': 0, 'Definite MG': 1}
ORIGINAL_FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV'] # Keep this for loading
EYE_POSITION_COLUMNS = ['LH', 'RH', 'LV', 'RV'] # For velocity and error calculations

CSV_ENCODING = 'utf-16-le'
CSV_SEPARATOR = ','
MIN_SEQ_LEN_THRESHOLD = 50 # Minimum rows in a CSV to be considered

# ML Model & CV Configuration
N_SPLITS_CV = 5
RANDOM_STATE = 42 # For reproducibility

print(f"Starting classical ML approach with feature extraction.")

# --- Data Loading (largely same, but we'll process sequences differently) ---
def load_sequences_for_feature_extraction(base_dir, class_mapping, original_feature_columns, encoding, separator, min_seq_len_threshold):
    print("Loading sequences for feature extraction...")
    items_for_extraction = [] # Store (df_features, label, patient_id, filename)

    for class_name, label in class_mapping.items():
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Class directory not found: {class_dir}")
            continue
        print(f"Processing class: {class_name}")
        patient_dirs = [d for d in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, d))]

        for patient_folder_name in tqdm(patient_dirs, desc=f"Patients in {class_name}"):
            patient_id = patient_folder_name
            patient_dir_path = os.path.join(class_dir, patient_folder_name)
            csv_files = glob.glob(os.path.join(patient_dir_path, '*.csv'))
            if not csv_files:
                # print(f"Warning: No CSV files found for patient {patient_id} in {patient_dir_path}") # Can be verbose
                continue
            for csv_file_path in csv_files:
                try:
                    df_full = pd.read_csv(csv_file_path, encoding=encoding, sep=separator)
                    df_full.columns = [col.strip() for col in df_full.columns]
                    missing_cols = [col for col in original_feature_columns if col not in df_full.columns]
                    if missing_cols:
                        # print(f"Warning: CSV {os.path.basename(csv_file_path)} missing original columns: {missing_cols}. Skipping.")
                        continue
                    df_features = df_full[original_feature_columns]
                    if df_features.empty or len(df_features) < min_seq_len_threshold:
                        # print(f"Warning: CSV {os.path.basename(csv_file_path)} empty/short. Skipping.")
                        continue
                    for col in df_features.columns: # Ensure numeric
                        df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
                    if df_features.isnull().sum().sum() > 0.1 * df_features.size: # Check NaNs post-coercion
                         # print(f"Warning: CSV {os.path.basename(csv_file_path)} has >10% NaNs after to_numeric. Skipping.")
                         continue
                    df_features = df_features.fillna(0) # Simple fill for any remaining NaNs
                    
                    items_for_extraction.append((df_features, label, patient_id, os.path.basename(csv_file_path)))
                except Exception as e:
                    print(f"Warning: Could not load/process {os.path.basename(csv_file_path)} for FE: {type(e).__name__} - {e}. Skipping.")
    
    print(f"Loaded {len(items_for_extraction)} CSVs for feature extraction.")
    return items_for_extraction

# --- Feature Extraction Function ---
def extract_features_from_trial(df_trial, original_feature_names, eye_position_cols):
    """ Extracts a flat feature vector from a single trial's DataFrame. """
    features = []
    feature_names = []

    # 1. Basic statistics for all original channels
    for col in original_feature_names:
        series = df_trial[col]
        features.extend([
            series.mean(), series.std(), series.median(),
            series.min(), series.max(), series.max() - series.min(), # Range
            series.quantile(0.25), series.quantile(0.75)
        ])
        feature_names.extend([
            f'{col}_mean', f'{col}_std', f'{col}_median',
            f'{col}_min', f'{col}_max', f'{col}_range',
            f'{col}_q25', f'{col}_q75'
        ])

    # 2. Velocity statistics for eye position channels
    for eye_col in eye_position_cols:
        if eye_col in df_trial.columns:
            velocity = df_trial[eye_col].diff().fillna(0)
            abs_velocity = velocity.abs()
            features.extend([
                velocity.mean(), velocity.std(),
                abs_velocity.max(), abs_velocity.mean() 
            ])
            feature_names.extend([
                f'{eye_col}_vel_mean', f'{eye_col}_vel_std',
                f'{eye_col}_abs_vel_max', f'{eye_col}_abs_vel_mean'
            ])
        else: # Should not happen if eye_position_cols are subset of original
            features.extend([0,0,0,0]) 
            feature_names.extend([f'{eye_col}_vel_mean', f'{eye_col}_vel_std', f'{eye_col}_abs_vel_max', f'{eye_col}_abs_vel_mean'])


    # 3. Position Error statistics for eye position channels
    for eye_col in eye_position_cols:
        target_col = ''
        if 'H' in eye_col and 'TargetH' in df_trial.columns: # LH, RH use TargetH
            target_col = 'TargetH'
        elif 'V' in eye_col and 'TargetV' in df_trial.columns: # LV, RV use TargetV
            target_col = 'TargetV'
        
        if target_col and eye_col in df_trial.columns:
            error = df_trial[eye_col] - df_trial[target_col]
            abs_error = error.abs()
            squared_error = error**2
            features.extend([
                error.mean(), error.std(),
                np.sqrt(squared_error.mean()), # RMSE
                abs_error.mean()
            ])
            feature_names.extend([
                f'{eye_col}_err_mean', f'{eye_col}_err_std',
                f'{eye_col}_err_rmse', f'{eye_col}_abs_err_mean'
            ])
        else: # If target or eye_col missing, append zeros
            features.extend([0,0,0,0])
            feature_names.extend([f'{eye_col}_err_mean', f'{eye_col}_err_std', f'{eye_col}_err_rmse', f'{eye_col}_abs_err_mean'])

    return features, feature_names # Return names once to build columns for DataFrame

# --- Main Script Execution ---
if __name__ == '__main__':
    # 1. Load sequence data (as DataFrames)
    items_for_fe = load_sequences_for_feature_extraction(
        BASE_DIR, CLASS_MAPPING, ORIGINAL_FEATURE_COLUMNS, CSV_ENCODING, CSV_SEPARATOR, MIN_SEQ_LEN_THRESHOLD
    )

    if not items_for_fe:
        print("No data loaded for feature extraction. Exiting."); exit()

    # 2. Extract features for each trial
    all_feature_vectors = []
    all_labels = []
    all_groups = [] # For patient-aware CV
    processed_feature_names = None # To store feature names from the first processed trial

    print("Extracting features from all trials...")
    for df_trial, label, patient_id, filename in tqdm(items_for_fe, desc="Extracting Features"):
        features, current_feature_names = extract_features_from_trial(df_trial, ORIGINAL_FEATURE_COLUMNS, EYE_POSITION_COLUMNS)
        if processed_feature_names is None:
            processed_feature_names = current_feature_names # Get feature names once
        
        all_feature_vectors.append(features)
        all_labels.append(label)
        all_groups.append(patient_id)

    if not all_feature_vectors:
        print("No feature vectors extracted. Exiting."); exit()

    X = np.array(all_feature_vectors)
    y = np.array(all_labels)
    groups = np.array(all_groups)
    
    print(f"Feature extraction complete. Shape of X: {X.shape}, Shape of y: {y.shape}")
    if X.shape[1] == 0:
        print(f"Error: Extracted feature vectors have 0 features. Check extract_features_from_trial. Feature names: {processed_feature_names}")
        exit()


    # 3. Cross-validation with Random Forest
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    
    fold_results_list = []

    print(f"\nStarting {N_SPLITS_CV}-Fold Cross-Validation with Random Forest...")

    for fold_idx, (train_indices, val_indices) in enumerate(sgkf.split(X, y, groups=groups)):
        print(f"\n--- Fold {fold_idx + 1}/{N_SPLITS_CV} ---")

        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        # Scale features (important for many ML models, good practice for RF too)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        print(f"Fold {fold_idx + 1}: Training on {X_train_scaled.shape[0]} samples, Validating on {X_val_scaled.shape[0]} samples.")

        # Initialize and train Random Forest
        # Consider adding class_weight='balanced' if classes are imbalanced
        model_rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
        model_rf.fit(X_train_scaled, y_train)

        # Predictions
        y_pred_val = model_rf.predict(X_val_scaled)
        y_pred_proba_val = model_rf.predict_proba(X_val_scaled)[:, 1] # Probabilities for the positive class

        # Evaluate
        accuracy = accuracy_score(y_val, y_pred_val)
        precision = precision_score(y_val, y_pred_val, zero_division=0)
        recall = recall_score(y_val, y_pred_val, zero_division=0)
        f1 = f1_score(y_val, y_pred_val, zero_division=0)
        roc_auc = 0.0
        if len(np.unique(y_val)) > 1:
             try: roc_auc = roc_auc_score(y_val, y_pred_proba_val)
             except ValueError as e: print(f"Warning: ROC AUC not computed for fold {fold_idx+1}. {e}")
        else: print(f"Warning: ROC AUC not computed for fold {fold_idx+1} (1 class in y_val).")


        print(f"Fold {fold_idx + 1} Results => "
              f"Val Acc: {accuracy:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")
        
        fold_results_list.append({
            'fold': fold_idx + 1,
            'val_acc': accuracy, 'val_precision': precision, 'val_recall': recall,
            'val_f1': f1, 'val_roc_auc': roc_auc
        })
        
        # Optional: Print feature importances for the last fold or averaged
        if fold_idx == N_SPLITS_CV - 1 and hasattr(model_rf, 'feature_importances_'):
            print("\nFeature Importances (from last fold):")
            importances = model_rf.feature_importances_
            sorted_indices = np.argsort(importances)[::-1]
            for i in range(min(20, X.shape[1])): # Print top 20 features
                print(f"{processed_feature_names[sorted_indices[i]]}: {importances[sorted_indices[i]]:.4f}")


    # 4. Aggregate and print cross-validation results
    print("\n--- Cross-Validation Summary (Classical ML - Random Forest) ---")
    if fold_results_list:
        avg_val_acc = np.mean([res['val_acc'] for res in fold_results_list])
        avg_val_f1 = np.mean([res['val_f1'] for res in fold_results_list])
        avg_val_precision = np.mean([res['val_precision'] for res in fold_results_list])
        avg_val_recall = np.mean([res['val_recall'] for res in fold_results_list])
        avg_val_roc_auc = np.mean([res['val_roc_auc'] for res in fold_results_list])

        for res_item in fold_results_list:
            print(f"Fold {res_item['fold']}: Acc={res_item['val_acc']:.4f}, P={res_item['val_precision']:.4f}, R={res_item['val_recall']:.4f}, F1={res_item['val_f1']:.4f}, AUC={res_item['val_roc_auc']:.4f}")

        print("\nAverage Cross-Validation Metrics:")
        print(f"  Average Validation Accuracy: {avg_val_acc:.4f}")
        print(f"  Average Validation F1-Score: {avg_val_f1:.4f}")
        print(f"  Average Validation Precision: {avg_val_precision:.4f}")
        print(f"  Average Validation Recall: {avg_val_recall:.4f}")
        print(f"  Average Validation ROC AUC: {avg_val_roc_auc:.4f}")
    else:
        print("No fold results to summarize.")

    print("\nClassical ML with feature extraction experiment finished.")
