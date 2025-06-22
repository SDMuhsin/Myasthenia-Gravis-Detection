import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import iqr as scipy_iqr
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
import warnings

# --- Configuration (Adapted from EDA) ---
BASE_DIR = './data'

# Original CLASS_DEFINITIONS from EDA
ORIGINAL_CLASS_DEFINITIONS = {
    'HC': {'path': 'Healthy control', 'label': 0},
    'MG': {'path': 'Definite MG', 'label': 1},
    'CNP3': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '3rd'), 'label': 2},
    'CNP4': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '4th'), 'label': 3},
    'CNP6': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '6th'), 'label': 4},
    'TAO': {'path': os.path.join('Non-MG diplopia (CNP, etc)', 'TAO'), 'label': 5},
}

# Filter out TAO class due to underrepresentation for statistical modeling
CLASSES_TO_EXCLUDE_FOR_MODELING = ['TAO']
CLASS_DEFINITIONS = {
    name: details for name, details in ORIGINAL_CLASS_DEFINITIONS.items()
    if name not in CLASSES_TO_EXCLUDE_FOR_MODELING
}
# Re-generate mappings and ordered names for the classes being modeled
CLASS_MAPPING = {name: details['label'] for name, details in CLASS_DEFINITIONS.items()}
INV_CLASS_MAPPING = {details['label']: name for name, details in CLASS_DEFINITIONS.items()}
ORDERED_CLASS_NAMES = [INV_CLASS_MAPPING[i] for i in sorted(INV_CLASS_MAPPING.keys())]
MODEL_CLASS_LABELS = sorted(CLASS_MAPPING.values()) # Numeric labels for models

FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']
CSV_ENCODING = 'utf-16-le'
CSV_SEPARATOR = ','
MIN_SEQ_LEN_THRESHOLD = 50
RESULTS_DIR = './results'
EXP_PREFIX = 'EXP_1_STATMODELS_'
NUMERICAL_SUMMARY_FILENAME = f'{EXP_PREFIX}numerical_summary.txt'
RANDOM_STATE = 42 # For reproducibility

# --- Helper for descriptive stats (from EDA) ---
def get_descriptive_stats_df(data_series, name="Value"):
    if data_series.empty:
        return pd.Series({
            'Count': 0, 'Mean': np.nan, 'Std': np.nan, 'Min': np.nan,
            'Q1 (25%)': np.nan, 'Median (50%)': np.nan, 'Q3 (75%)': np.nan, 'Max': np.nan, 'IQR': np.nan
        }, name=name)

    stats = {
        'Count': data_series.count(),
        'Mean': data_series.mean(),
        'Std': data_series.std(),
        'Min': data_series.min(),
        'Q1 (25%)': data_series.quantile(0.25),
        'Median (50%)': data_series.median(),
        'Q3 (75%)': data_series.quantile(0.75),
        'Max': data_series.max(),
    }
    stats['IQR'] = stats['Q3 (75%)'] - stats['Q1 (25%)']
    return pd.Series(stats, name=name)

# --- Data Loading and Feature Engineering (Adapted from EDA code) ---
def load_raw_sequences_and_labels(base_dir, class_definitions_dict, feature_columns_expected, encoding, separator, min_seq_len_threshold, f_out):
    f_out.write("="*70 + "\n")
    f_out.write("Phase: Data Loading\n")
    f_out.write("="*70 + "\n")
    print("="*50)
    print("Starting Data Loading...")
    print("="*50)
    raw_items = []

    for class_name_key, class_details in class_definitions_dict.items():
        label = class_details['label']
        current_class_path_segment = class_details['path']
        class_dir_abs = os.path.join(base_dir, current_class_path_segment)

        msg = f"\nProcessing Class: '{class_name_key}' (Label: {label}) from path: {class_dir_abs}\n"
        print(msg.strip())
        f_out.write(msg)

        if not os.path.isdir(class_dir_abs):
            warning_msg = f"WARNING: Class directory not found: {class_dir_abs}\n"
            print(warning_msg.strip())
            f_out.write(warning_msg)
            continue

        patient_dirs = [d for d in os.listdir(class_dir_abs) if os.path.isdir(os.path.join(class_dir_abs, d))]
        if not patient_dirs:
            info_msg = f"INFO: No patient directories found in {class_dir_abs} for class '{class_name_key}'\n"
            print(info_msg.strip())
            f_out.write(info_msg)
            continue

        for patient_folder_name in tqdm(patient_dirs, desc=f"  Patients in {class_name_key}"):
            patient_id = patient_folder_name
            patient_dir_path = os.path.join(class_dir_abs, patient_folder_name)
            csv_files = glob.glob(os.path.join(patient_dir_path, '*.csv'))
            if not csv_files: continue

            for csv_file_path in csv_files:
                try:
                    df_full = pd.read_csv(csv_file_path, encoding=encoding, sep=separator)
                    original_columns = df_full.columns.tolist()
                    df_full.columns = [col.strip() for col in original_columns]

                    missing_cols = [col for col in feature_columns_expected if col not in df_full.columns]
                    if missing_cols: continue

                    df_features = df_full[feature_columns_expected]
                    if df_features.empty or len(df_features) < min_seq_len_threshold: continue

                    for col in df_features.columns:
                        df_features.loc[:, col] = pd.to_numeric(df_features[col], errors='coerce')

                    if df_features.isnull().sum().sum() > 0.1 * df_features.size: continue

                    df_features = df_features.fillna(0)
                    sequence_data = df_features.values.astype(np.float32)
                    raw_items.append({
                        'data': sequence_data, 'label': label, 'patient_id': patient_id,
                        'filename': os.path.basename(csv_file_path),
                        'class_name': class_name_key,
                        'original_length': len(sequence_data)
                    })
                except Exception as e:
                    error_msg = f"ERROR processing {os.path.basename(csv_file_path)} (Patient: {patient_id}, Class: {class_name_key}): {type(e).__name__} - {e}. Skipping.\n"
                    print(error_msg.strip())
                    # f_out.write(error_msg)

    summary_msg = f"\nData loading complete. Loaded {len(raw_items)} raw sequences meeting criteria for defined classes.\n"
    print(summary_msg.strip())
    f_out.write(summary_msg)
    if not raw_items:
        crit_warning_msg = "CRITICAL WARNING: No raw sequences were loaded. Please check `BASE_DIR`, `CLASS_DEFINITIONS`, CSV files, and `MIN_SEQ_LEN_THRESHOLD`.\n"
        print(crit_warning_msg.strip())
        f_out.write(crit_warning_msg)
    f_out.write("-" * 70 + "\n\n")
    return raw_items

def engineer_features_from_raw_data(raw_items_dicts, original_feature_names, f_out):
    f_out.write("="*70 + "\n")
    f_out.write("Phase: Feature Engineering\n")
    f_out.write("="*70 + "\n")
    print("\n" + "="*50)
    print("Starting Feature Engineering (velocities, errors)...")
    print("="*50)

    if not raw_items_dicts:
        no_items_msg = "INFO: No raw items to engineer features from.\n"
        print(no_items_msg.strip())
        f_out.write(no_items_msg)
        f_out.write("-" * 70 + "\n\n")
        return [], 0, []

    engineered_items = []
    final_num_channels = 0
    final_feature_names = []

    for item in tqdm(raw_items_dicts, desc="  Engineering Features"):
        sequence_data_original = item['data']
        df_original = pd.DataFrame(sequence_data_original, columns=original_feature_names)
        df_engineered_parts = []
        for pos_col in ['LH', 'RH', 'LV', 'RV']:
            if pos_col in df_original.columns:
                velocity_series = df_original[pos_col].diff().fillna(0)
                df_engineered_parts.append(velocity_series.rename(f'{pos_col}_Vel'))
            else:
                df_engineered_parts.append(pd.Series(np.zeros(len(df_original)), name=f'{pos_col}_Vel'))
        target_h_exists = 'TargetH' in df_original.columns
        target_v_exists = 'TargetV' in df_original.columns
        error_definitions = [
            ('LH', 'TargetH', 'ErrorH_L', target_h_exists), ('RH', 'TargetH', 'ErrorH_R', target_h_exists),
            ('LV', 'TargetV', 'ErrorV_L', target_v_exists), ('RV', 'TargetV', 'ErrorV_R', target_v_exists)
        ]
        for eye_col, target_col, error_col_name, target_exists in error_definitions:
            if target_exists and eye_col in df_original.columns:
                df_engineered_parts.append((df_original[eye_col] - df_original[target_col]).rename(error_col_name))
            else:
                df_engineered_parts.append(pd.Series(np.zeros(len(df_original)), name=error_col_name))
        if df_engineered_parts:
            df_engineered_combined = pd.concat(df_engineered_parts, axis=1)
            df_all_features = pd.concat([df_original, df_engineered_combined], axis=1)
        else: df_all_features = df_original
        if not final_feature_names: final_feature_names = df_all_features.columns.tolist()
        engineered_sequence_data = df_all_features.values.astype(np.float32)
        engineered_items.append({
            'data': engineered_sequence_data, 'label': item['label'],
            'patient_id': item['patient_id'], 'filename': item['filename'],
            'class_name': item['class_name'], 'original_length': item['original_length']
        })
        if final_num_channels == 0:
            final_num_channels = engineered_sequence_data.shape[1]
            if final_num_channels != len(final_feature_names):
                crit_warn_msg = f"CRITICAL WARNING: Mismatch in derived channel count ({final_num_channels}) and feature name list length ({len(final_feature_names)}).\n"
                print(crit_warn_msg.strip())
                f_out.write(crit_warn_msg)

    summary_msg1 = f"\nFeature engineering complete.\n"
    summary_msg2 = f"  Number of features (channels) per sequence: {final_num_channels}\n"
    summary_msg3 = f"  Engineered feature names: {final_feature_names}\n"
    print(summary_msg1.strip())
    print(summary_msg2.strip())
    print(summary_msg3.strip())
    f_out.write(summary_msg1 + summary_msg2 + summary_msg3)

    if not engineered_items:
        no_items_warn = "WARNING: No items after feature engineering.\n"
        print(no_items_warn.strip())
        f_out.write(no_items_warn)
    f_out.write("-" * 70 + "\n\n")
    return engineered_items, final_num_channels, final_feature_names

def build_aggregated_features_df(engineered_items_list_of_dicts, list_of_feature_names, f_out):
    f_out.write("="*70 + "\n")
    f_out.write("Phase: Building Aggregated Features DataFrame (Input for Statistical Models)\n")
    f_out.write("="*70 + "\n")
    print("\n" + "="*50)
    print("Building DataFrame of Aggregated Features (mean, median, std, etc. per sequence)...")
    print("="*50)

    if not engineered_items_list_of_dicts:
        no_items_msg = "  No engineered items to process for aggregation. Returning empty DataFrame.\n"
        print(no_items_msg.strip())
        f_out.write(no_items_msg)
        f_out.write("-" * 70 + "\n\n")
        return pd.DataFrame()

    aggregated_data_rows = []
    for item in tqdm(engineered_items_list_of_dicts, desc="  Aggregating features for each sequence"):
        seq_data_np = item['data']
        current_row = {'patient_id': item['patient_id'], 'filename': item['filename'],
                       'class_name': item['class_name'], 'label': item['label'],
                       'original_length': item['original_length']}
        for i, feature_name_str in enumerate(list_of_feature_names):
            ft_ts = seq_data_np[:, i]
            current_row[f'{feature_name_str}_mean'] = np.mean(ft_ts)
            current_row[f'{feature_name_str}_median'] = np.median(ft_ts)
            current_row[f'{feature_name_str}_std'] = np.std(ft_ts)
            current_row[f'{feature_name_str}_min'] = np.min(ft_ts)
            current_row[f'{feature_name_str}_max'] = np.max(ft_ts)
            current_row[f'{feature_name_str}_range'] = np.max(ft_ts) - np.min(ft_ts)
            current_row[f'{feature_name_str}_iqr'] = scipy_iqr(ft_ts)
        aggregated_data_rows.append(current_row)
    agg_df = pd.DataFrame(aggregated_data_rows)

    summary_msg = f"  Aggregated features DataFrame created with shape: {agg_df.shape}\n"
    print(summary_msg.strip())
    f_out.write(summary_msg)

    if agg_df.empty:
        warn_msg = "WARNING: Aggregated features DataFrame is empty.\n"
        print(warn_msg.strip())
        f_out.write(warn_msg)
    f_out.write("-" * 70 + "\n\n")
    return agg_df

def create_results_directory(dir_path=RESULTS_DIR):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"INFO: Created directory for results: {dir_path}")

# --- Statistical Modeling Functions ---
def plot_confusion_matrix(cm, classes, model_name, results_dir, f_out):
    plt.figure(figsize=(max(8, len(classes)*1.2), max(6, len(classes)*1)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'{EXP_PREFIX}{model_name.replace(" ", "_")}_confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()
    f_out.write(f"Confusion matrix plot saved to: {plot_path}\n")
    print(f"  Saved confusion matrix for {model_name} to {plot_path}")

def train_and_evaluate_models(X_df, y_series, ordered_target_names, numeric_labels, results_dir, f_out):
    f_out.write("="*70 + "\n")
    f_out.write("Phase: Statistical Model Training and Evaluation\n")
    f_out.write("="*70 + "\n")
    print("\n" + "="*50)
    print("Starting Statistical Model Training and Evaluation...")
    print("="*50)

    # Prepare data
    X = X_df.values
    y = y_series.values

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, max_iter=1000, multi_class='ovr', class_weight='balanced'),
        "LDA": LinearDiscriminantAnalysis(solver='svd'), # SVD does not require feature scaling as much, good for colinearity
        "QDA": QuadraticDiscriminantAnalysis(),
        "Linear SVM": SVC(kernel='linear', random_state=RANDOM_STATE, probability=False, class_weight='balanced') # probability=False for speed, not doing ROC here
    }
    
    # Suppress QDA warnings about constant covariance within groups if it happens
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.discriminant_analysis')
    warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn.linear_model')


    for model_name, model_unscaled in models.items():
        f_out.write(f"\n--- Model: {model_name} ---\n")
        print(f"\nTraining and evaluating: {model_name}")

        # Create a pipeline with scaling for models that benefit from it
        if model_name in ["Logistic Regression", "Linear SVM", "QDA"]: # QDA can also benefit
             pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model_unscaled)
            ])
        else: # LDA can handle unscaled data, especially with 'svd' solver
            pipeline = Pipeline([('model', model_unscaled)])


        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        
        # Need to collect all predictions and true labels for a single confusion matrix and classification report
        all_y_true = []
        all_y_pred = []

        accuracies = []
        
        fold_num = 1
        for train_idx, test_idx in tqdm(cv.split(X, y), total=cv.get_n_splits(), desc=f"  CV for {model_name}"):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            try:
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                
                all_y_true.extend(y_test)
                all_y_pred.extend(y_pred)
                accuracies.append(accuracy_score(y_test, y_pred))
            except Exception as e:
                error_msg = f"ERROR during training/prediction for {model_name} fold {fold_num}: {e}\n"
                print(error_msg.strip())
                f_out.write(error_msg)
                # Skip this fold for this model if it fails critically (e.g. QDA with singular covariance)
                continue
            fold_num += 1

        if not all_y_pred: # If all folds failed
            f_out.write("  Model training failed for all folds. Skipping evaluation.\n")
            print(f"  Model training failed for all folds for {model_name}. Skipping.")
            continue

        # Overall metrics from all folds
        mean_accuracy = np.mean(accuracies) if accuracies else 0.0
        std_accuracy = np.std(accuracies) if accuracies else 0.0
        f_out.write(f"Cross-validated Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})\n\n")
        print(f"  Cross-validated Accuracy for {model_name}: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")

        report = classification_report(all_y_true, all_y_pred, target_names=ordered_target_names, labels=numeric_labels, zero_division=0)
        f_out.write("Classification Report (from aggregated CV predictions):\n")
        f_out.write(report + "\n\n")
        print(f"  Classification Report for {model_name}:\n{report}")

        cm = confusion_matrix(all_y_true, all_y_pred, labels=numeric_labels)
        f_out.write("Confusion Matrix (from aggregated CV predictions):\n")
        f_out.write(np.array2string(cm) + "\n")
        plot_confusion_matrix(cm, ordered_target_names, model_name, results_dir, f_out)

        # Feature importances for Logistic Regression
        if model_name == "Logistic Regression" and hasattr(pipeline.named_steps['model'], 'coef_'):
            # Refit on all data to get final coefficients (standard practice for reporting)
            # For OvR, coefficients are per class
            try:
                final_pipeline_lr = Pipeline([('scaler', StandardScaler()), ('model', model_unscaled)])
                final_pipeline_lr.fit(X, y) # Fit on all X, y
                coefficients = final_pipeline_lr.named_steps['model'].coef_

                f_out.write("\nFeature Coefficients (Importance) for Logistic Regression (OvR):\n")
                f_out.write(f"  Feature names: {X_df.columns.tolist()}\n")
                for i, class_name in enumerate(ordered_target_names):
                    if i < coefficients.shape[0]: # Check if coefs exist for this class (can be less in binary cases implicitly)
                        f_out.write(f"  Coefficients for class '{class_name}' vs Rest:\n")
                        coef_series = pd.Series(coefficients[i], index=X_df.columns).sort_values(ascending=False)
                        f_out.write(coef_series.to_string() + "\n\n")
                    else: # Should not happen with OvR if all classes are present
                         f_out.write(f"  Could not retrieve coefficients for class '{class_name}'.\n")


            except Exception as e:
                f_out.write(f"  Could not extract feature coefficients for {model_name}: {e}\n")
                print(f"  Could not extract feature coefficients for {model_name}: {e}\n")
        
        # Feature importances for LDA (coefficients of linear discriminants)
        if model_name == "LDA" and hasattr(pipeline.named_steps['model'], 'coef_'):
            try:
                # Refit on all data to get final coefficients
                final_pipeline_lda = Pipeline([('scaler', StandardScaler()), ('model', model_unscaled)]) # LDA also benefits from scaling for coef interpret.
                final_pipeline_lda.fit(X, y)
                coefficients_lda = final_pipeline_lda.named_steps['model'].coef_ # (n_classes-1, n_features)

                f_out.write("\nCoefficients of Linear Discriminants for LDA:\n")
                f_out.write(f"  Feature names: {X_df.columns.tolist()}\n")
                # For LDA, coef_ gives weights for linear discriminants, not direct per-class odds like LogReg.
                # For (n_classes) classes, there are (n_classes - 1) discriminants.
                for i in range(coefficients_lda.shape[0]):
                    f_out.write(f"  Coefficients for Linear Discriminant {i+1}:\n")
                    coef_series = pd.Series(coefficients_lda[i], index=X_df.columns).sort_values(ascending=False)
                    f_out.write(coef_series.to_string() + "\n\n")
            except Exception as e:
                f_out.write(f"  Could not extract LDA coefficients: {e}\n")
                print(f"  Could not extract LDA coefficients: {e}\n")


        f_out.write("-" * 50 + "\n")
    
    warnings.resetwarnings() # Reset warnings to default
    f_out.write("-" * 70 + "\n\n")
    print("="*50)
    print("Statistical Model Training and Evaluation Complete.")
    print("="*50)

# --- Main Script Execution ---
if __name__ == '__main__':
    print("="*70)
    print("Starting Statistical Models Experiment (EXP_1_STATMODELS)...")
    print("This script will use aggregated features for classification.")
    print(f"Target classes for modeling: {ORDERED_CLASS_NAMES}")
    print(f"Classes excluded: {CLASSES_TO_EXCLUDE_FOR_MODELING} (due to underrepresentation)")
    print("="*70)

    create_results_directory(RESULTS_DIR)
    numerical_summary_filepath = os.path.join(RESULTS_DIR, NUMERICAL_SUMMARY_FILENAME)

    with open(numerical_summary_filepath, 'w', encoding='utf-8') as f_report:
        f_report.write("======================================================================\n")
        f_report.write("Statistical Models (EXP_1_STATMODELS) - Numerical Summary\n")
        f_report.write("======================================================================\n")
        f_report.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_report.write(f"Base Directory: {BASE_DIR}\n")
        f_report.write(f"Class Definitions Used (for modeling): {CLASS_DEFINITIONS}\n")
        f_report.write(f"Classes excluded from modeling due to underrepresentation: {CLASSES_TO_EXCLUDE_FOR_MODELING}\n")
        f_report.write(f"Original Class Definitions (from EDA): {ORIGINAL_CLASS_DEFINITIONS}\n")
        f_report.write(f"Minimum Sequence Length Threshold: {MIN_SEQ_LEN_THRESHOLD}\n")
        f_report.write(f"Features used for aggregation: {FEATURE_COLUMNS}\n")
        f_report.write(f"Random State for reproducibility: {RANDOM_STATE}\n")
        f_report.write("======================================================================\n\n")

        # 1. Load Raw Data (using CLASS_DEFINITIONS which excludes TAO by default now)
        raw_items_list = load_raw_sequences_and_labels(
            BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, CSV_ENCODING, CSV_SEPARATOR, MIN_SEQ_LEN_THRESHOLD, f_report
        )
        if not raw_items_list:
            error_msg = "\nCRITICAL: No data loaded. Statistical modeling cannot proceed.\n"
            print(error_msg.strip())
            f_report.write(error_msg)
            exit()

        # 2. Engineer Features (Velocities, Errors)
        engineered_items_list, num_engineered_channels, list_engineered_feature_names = engineer_features_from_raw_data(
            raw_items_list, FEATURE_COLUMNS, f_report
        )
        if not engineered_items_list or not list_engineered_feature_names:
            error_msg = "\nCRITICAL: No data after feature engineering. Statistical modeling cannot proceed.\n"
            print(error_msg.strip())
            f_report.write(error_msg)
            exit()
        f_report.write(f"List of engineered features (before aggregation): {list_engineered_feature_names}\n\n")

        # 3. Build Aggregated Features DataFrame (Input for models)
        aggregated_features_master_df = build_aggregated_features_df(engineered_items_list, list_engineered_feature_names, f_report)
        if aggregated_features_master_df.empty:
            error_msg = "\nCRITICAL: Aggregated features DataFrame is empty. Statistical modeling cannot proceed.\n"
            print(error_msg.strip())
            f_report.write(error_msg)
            exit()
        
        f_report.write(f"Shape of aggregated features DataFrame: {aggregated_features_master_df.shape}\n")
        f_report.write(f"Columns in aggregated_features_master_df: {aggregated_features_master_df.columns.tolist()}\n\n")
        f_report.write("Brief look at aggregated data (first 5 rows):\n")
        f_report.write(aggregated_features_master_df.head().to_string() + "\n\n")

        # Prepare data for modeling
        # Ensure 'label' and 'class_name' are correct for the filtered classes
        y_labels_series = aggregated_features_master_df['label']
        # The feature matrix X should not contain label, class_name, patient_id, etc.
        feature_cols_for_model = [col for col in aggregated_features_master_df.columns if col not in ['patient_id', 'filename', 'class_name', 'label', 'original_length']]
        X_features_df = aggregated_features_master_df[feature_cols_for_model]
        
        f_report.write(f"Number of features for modeling: {len(feature_cols_for_model)}\n")
        f_report.write(f"List of features used for modeling: {feature_cols_for_model}\n\n")


        # Check for missing values in X_features_df before modeling
        if X_features_df.isnull().sum().sum() > 0:
            nan_warning = "WARNING: NaN values found in feature matrix X before modeling. This might cause issues.\n"
            print(nan_warning.strip())
            f_report.write(nan_warning)
            f_report.write("Sum of NaNs per column:\n" + X_features_df.isnull().sum().to_string() + "\n\n")
            # Optional: Add imputation here if necessary, though EDA fillna(0) should handle it upstream.
            # X_features_df = X_features_df.fillna(X_features_df.mean()) # Example: mean imputation
            # f_report.write("Applied mean imputation to handle NaNs.\n")


        # 4. Train and Evaluate Statistical Models
        train_and_evaluate_models(X_features_df, y_labels_series, ORDERED_CLASS_NAMES, MODEL_CLASS_LABELS, RESULTS_DIR, f_report)

        f_report.write("\n======================================================================\n")
        f_report.write("End of Statistical Models Numerical Summary Report (EXP_1_STATMODELS)\n")
        f_report.write("======================================================================\n")

    print(f"\nNumerical summary for statistical models saved to: {numerical_summary_filepath}")
    print("\n" + "="*70)
    print("Statistical Models Experiment (EXP_1_STATMODELS) Finished.")
    print(f"All plots and the numerical summary have been saved in: {RESULTS_DIR}")
    print("="*70)
