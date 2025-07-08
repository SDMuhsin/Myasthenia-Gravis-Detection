import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import iqr as scipy_iqr
from datetime import datetime
import re
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning

# --- Configuration ---
BASE_DIR = './data'

# Original class definitions to ensure we can map everything correctly
ORIGINAL_CLASS_DEFINITIONS = {
    'HC': {'path': 'Healthy control', 'label': 0},
    'MG': {'path': 'Definite MG', 'label': 1},
    'CNP3': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '3rd'), 'label': 2},
    'CNP4': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '4th'), 'label': 3},
    'CNP6': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '6th'), 'label': 4},
    'TAO': {'path': os.path.join('Non-MG diplopia (CNP, etc)', 'TAO'), 'label': 5},
}

# For the main multiclass analysis, TAO is excluded due to low sample size
CLASSES_TO_EXCLUDE_FOR_MODELING = ['TAO']
CLASS_DEFINITIONS = {
    name: details for name, details in ORIGINAL_CLASS_DEFINITIONS.items()
    if name not in CLASSES_TO_EXCLUDE_FOR_MODELING
}
CLASS_MAPPING = {name: details['label'] for name, details in CLASS_DEFINITIONS.items()}
INV_CLASS_MAPPING = {details['label']: name for name, details in CLASS_DEFINITIONS.items()}

# --- Core Parameters ---
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']
CSV_ENCODING = 'utf-16-le'
CSV_SEPARATOR = ','
MIN_SEQ_LEN_THRESHOLD = 50
RESULTS_DIR = './results'
EXP_PREFIX = 'EXP_7_FEATURE_ANALYSIS_'
NUMERICAL_SUMMARY_FILENAME = f'{EXP_PREFIX}numerical_summary.txt'
RANDOM_STATE = 42
N_TOP_FEATURES = 20 # Number of top features to select for optimized models

# Suppress common warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.discriminant_analysis')
warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn.linear_model')


# --- Utility and Plotting Functions ---
def create_results_directory(dir_path=RESULTS_DIR):
    """Creates the results directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"INFO: Created directory for results: {dir_path}")

def plot_confusion_matrix(cm, classes, model_name, results_dir, f_out, suffix=""):
    """Saves a plot of the confusion matrix."""
    plt.figure(figsize=(max(8, len(classes) * 1.5), max(6, len(classes) * 1.2)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={"size": 14})
    plt.title(f'Confusion Matrix: {model_name}{suffix}', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'{EXP_PREFIX}{model_name.replace(" ", "_")}{suffix}_confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()
    f_out.write(f"Confusion matrix plot saved to: {plot_path}\n")
    print(f"  Saved confusion matrix for {model_name}{suffix} to {plot_path}")

def plot_feature_importance(importances, feature_names, title, results_dir, f_out):
    """Saves a bar plot of feature importances."""
    indices = np.argsort(importances)[-N_TOP_FEATURES:]
    plt.figure(figsize=(12, 8))
    plt.title(f'Feature Importance: {title}')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance (LDA Coefficient Magnitude)')
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'{EXP_PREFIX}{title.replace(" ", "_")}_feature_importance.png')
    plt.savefig(plot_path)
    plt.close()
    f_out.write(f"Feature importance plot saved to: {plot_path}\n")
    print(f"  Saved feature importance plot for {title}")


# --- Data Loading and Feature Engineering (Identical to Exp 6) ---
def parse_frequency_from_filename(filename):
    """Extracts saccade frequency (e.g., 0.5, 0.75, 1) from a filename."""
    match = re.search(r'\((\d+(\.\d+)?)\s*Hz\)', filename, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return np.nan

def load_raw_sequences_and_labels(base_dir, class_definitions_dict, feature_columns_expected, encoding, separator, min_seq_len_threshold, f_out):
    """Loads data from the directory structure."""
    f_out.write("=" * 80 + "\nPhase: Data Loading\n" + "=" * 80 + "\n")
    print("\n" + "="*50 + "\nStarting Data Loading...\n" + "="*50)
    raw_items = []
    for class_name_key, class_details in class_definitions_dict.items():
        label = class_details['label']
        class_dir_abs = os.path.join(base_dir, class_details['path'])
        if not os.path.isdir(class_dir_abs):
            f_out.write(f"WARNING: Class directory not found: {class_dir_abs}\n")
            continue
        patient_dirs = [d for d in os.listdir(class_dir_abs) if os.path.isdir(os.path.join(class_dir_abs, d))]
        for patient_folder_name in tqdm(patient_dirs, desc=f"  Patients in {class_name_key}"):
            patient_dir_path = os.path.join(class_dir_abs, patient_folder_name)
            csv_files = glob.glob(os.path.join(patient_dir_path, '*.csv'))
            if not csv_files: continue
            for csv_file_path in csv_files:
                try:
                    df_full = pd.read_csv(csv_file_path, encoding=encoding, sep=separator)
                    df_full.columns = [col.strip() for col in df_full.columns]
                    if any(col not in df_full.columns for col in feature_columns_expected) or len(df_full) < min_seq_len_threshold:
                        continue
                    df_features = df_full[feature_columns_expected].copy()
                    for col in df_features.columns:
                        df_features.loc[:, col] = pd.to_numeric(df_features[col], errors='coerce')
                    if df_features.isnull().sum().sum() > 0.1 * df_features.size: continue
                    df_features = df_features.fillna(0)
                    raw_items.append({
                        'data': df_features.values.astype(np.float32), 'label': label,
                        'patient_id': patient_folder_name, 'filename': os.path.basename(csv_file_path),
                        'class_name': class_name_key
                    })
                except Exception:
                    pass
    f_out.write(f"\nData loading complete. Loaded {len(raw_items)} raw sequences.\n\n")
    print(f"\nData loading complete. Loaded {len(raw_items)} raw sequences.")
    return raw_items

def engineer_and_aggregate_features(raw_items_dicts, original_feature_names, f_out):
    """Creates aggregated statistical features from time-series data."""
    f_out.write("="*80 + "\nPhase: Feature Engineering and Aggregation\n" + "="*80 + "\n")
    print("\n" + "="*50 + "\nStarting Feature Engineering and Aggregation...\n" + "="*50)
    if not raw_items_dicts:
        f_out.write("No raw items to process. Returning empty DataFrame.\n")
        return pd.DataFrame(), []
    aggregated_data_rows = []
    engineered_feature_names = []
    for item in tqdm(raw_items_dicts, desc="  Engineering & Aggregating"):
        df_original = pd.DataFrame(item['data'], columns=original_feature_names)
        df_engineered_parts = []
        for pos_col in ['LH', 'RH', 'LV', 'RV']:
            velocity_series = df_original[pos_col].diff().fillna(0)
            df_engineered_parts.append(velocity_series.rename(f'{pos_col}_Vel'))
        error_definitions = [('LH', 'TargetH', 'ErrorH_L'), ('RH', 'TargetH', 'ErrorH_R'),
                               ('LV', 'TargetV', 'ErrorV_L'), ('RV', 'TargetV', 'ErrorV_R')]
        for eye_col, target_col, error_col_name in error_definitions:
            df_engineered_parts.append((df_original[eye_col] - df_original[target_col]).rename(error_col_name))
        df_all_features = pd.concat([df_original] + df_engineered_parts, axis=1)
        if not engineered_feature_names: engineered_feature_names = df_all_features.columns.tolist()
        current_row = {
            'patient_id': item['patient_id'], 'filename': item['filename'],
            'class_name': item['class_name'], 'label': item['label'],
        }
        for feature_name in engineered_feature_names:
            ft_ts = df_all_features[feature_name]
            current_row[f'{feature_name}_mean'] = np.mean(ft_ts)
            current_row[f'{feature_name}_std'] = np.std(ft_ts)
            current_row[f'{feature_name}_median'] = np.median(ft_ts)
            current_row[f'{feature_name}_iqr'] = scipy_iqr(ft_ts)
        aggregated_data_rows.append(current_row)
    agg_df = pd.DataFrame(aggregated_data_rows)
    numerical_feature_cols = [f for f in agg_df.columns if '_mean' in f or '_std' in f or '_median' in f or '_iqr' in f]
    f_out.write(f"Aggregated features DataFrame created with shape: {agg_df.shape}\n")
    f_out.write(f"Total numerical features created: {len(numerical_feature_cols)}\n\n")
    return agg_df, numerical_feature_cols


# --- Modeling Core ---
def get_model_pipeline(numerical_features, model):
    """Creates a full preprocessing and modeling pipeline for numerical features."""
    preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    return Pipeline(steps=[
        ('preprocessor', ColumnTransformer([('num', preprocessor, numerical_features)])),
        ('model', model)
    ])

def train_and_evaluate_single_model(X_df, y_series, numerical_features, model_name, model, ordered_target_names, numeric_labels, results_dir, f_out, suffix=""):
    """Trains and evaluates a single model pipeline, returning key metrics."""
    f_out.write(f"\n--- Model: {model_name}{suffix} ---\n")
    print(f"\nTraining and evaluating: {model_name}{suffix}")
    pipeline = get_model_pipeline(numerical_features, model)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    all_y_true, all_y_pred, accuracies = [], [], []

    for train_idx, test_idx in tqdm(cv.split(X_df, y_series), total=cv.get_n_splits(), desc=f"  CV for {model_name}{suffix}"):
        X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
        y_train, y_test = y_series.iloc[train_idx], y_series.iloc[test_idx]
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        accuracies.append(accuracy_score(y_test, y_pred))

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    f_out.write(f"Cross-validated Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})\n\n")
    print(f"  Cross-validated Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
    report = classification_report(all_y_true, all_y_pred, target_names=ordered_target_names, labels=numeric_labels, zero_division=0)
    f_out.write("Classification Report (from aggregated CV predictions):\n" + report + "\n\n")
    cm = confusion_matrix(all_y_true, all_y_pred, labels=numeric_labels)
    f_out.write("Confusion Matrix (from aggregated CV predictions):\n" + np.array2string(cm) + "\n")
    plot_confusion_matrix(cm, ordered_target_names, model_name, results_dir, f_out, suffix)
    return mean_accuracy, report, cm


# --- Experiment 7 Specific Functions ---

def analyze_and_select_features(master_df, all_numerical_features, scenarios, results_dir, f_out):
    """
    Part 1: Analyzes feature importance for different classification scenarios.
    Returns a dictionary of top features for each scenario.
    """
    f_out.write("\n" + "="*80 + "\n")
    f_out.write("Part 1: Feature Importance Analysis\n")
    f_out.write("="*80 + "\n")
    print("\n" + "="*50 + "\nRunning Part 1: Feature Importance Analysis...\n" + "="*50)

    top_features_per_scenario = {}

    for name, details in scenarios.items():
        f_out.write(f"\n--- Analyzing Scenario: {name} ---\n")
        print(f"\nAnalyzing Scenario: {name}")

        df_subset = master_df[master_df['class_name'].isin(details['classes'])].copy()
        if 'relabel_map' in details:
            y = df_subset['class_name'].map(details['relabel_map'])
        else:
            y = df_subset['label']
        X = df_subset[all_numerical_features]

        # Impute and Scale data before getting coefficients
        preprocessor = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        X_processed = preprocessor.fit_transform(X, y)

        lda = LinearDiscriminantAnalysis(solver='svd')
        lda.fit(X_processed, y)

        # For multiclass, average the absolute coefficients across components
        if lda.coef_.shape[0] > 1:
            importances = np.mean(np.abs(lda.coef_), axis=0)
        else:
            importances = np.abs(lda.coef_[0])

        feature_importance_df = pd.DataFrame({
            'feature': all_numerical_features,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
        
        top_features = feature_importance_df.head(N_TOP_FEATURES)['feature'].tolist()
        top_features_per_scenario[name] = top_features
        
        f_out.write(f"Top {N_TOP_FEATURES} features for {name}:\n")
        for i, feature in enumerate(top_features):
            f_out.write(f"  {i+1}. {feature}\n")
        f_out.write("\n")
        
        plot_feature_importance(importances, all_numerical_features, name, results_dir, f_out)

    return top_features_per_scenario

def run_optimized_classification(master_df, scenarios, top_features_dict, results_dir, f_out):
    """
    Part 2: Runs classification using only the top features selected for each scenario.
    """
    f_out.write("\n" + "="*80 + "\n")
    f_out.write("Part 2: Optimized Classification with Selected Features\n")
    f_out.write("="*80 + "\n")
    print("\n" + "="*50 + "\nRunning Part 2: Optimized Classification...\n" + "="*50)

    for name, details in scenarios.items():
        df_subset = master_df[master_df['class_name'].isin(details['classes'])].copy()
        
        if 'relabel_map' in details:
            y = df_subset['class_name'].map(details['relabel_map'])
        else:
            y = df_subset['label']
            
        selected_features = top_features_dict[name]
        X = df_subset[selected_features]
        
        target_names = details['target_names']
        target_labels = details['target_labels']

        train_and_evaluate_single_model(
            X, y, selected_features, "LDA",
            LinearDiscriminantAnalysis(solver='svd'),
            target_names, target_labels,
            results_dir, f_out, suffix=f"_{name}_Optimized"
        )

def evaluate_multistage_classifier(master_df, all_numerical_features, top_features_dict, results_dir, f_out):
    """
    Part 3: Evaluates the performance of a multi-stage classification system.
    """
    f_out.write("\n" + "="*80 + "\n")
    f_out.write("Part 3: Multi-Stage Classifier Evaluation\n")
    f_out.write("="*80 + "\n")
    print("\n" + "="*50 + "\nRunning Part 3: Multi-Stage Classifier Evaluation...\n" + "="*50)

    # Use the full dataset for cross-validation
    y_true_full = master_df['label']
    y_pred_full = np.zeros_like(y_true_full)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for train_idx, test_idx in tqdm(cv.split(master_df, y_true_full), total=cv.get_n_splits(), desc="  CV for Multi-Stage System"):
        df_train, df_test = master_df.iloc[train_idx], master_df.iloc[test_idx]

        # --- Stage 1: HC vs. All Else ---
        s1_features = top_features_dict['HC vs All Else']
        s1_model = get_model_pipeline(s1_features, LinearDiscriminantAnalysis(solver='svd'))
        
        df_train_s1 = df_train.copy()
        df_train_s1['s1_label'] = df_train_s1['class_name'].apply(lambda x: 0 if x == 'HC' else 1)
        s1_model.fit(df_train_s1[s1_features], df_train_s1['s1_label'])
        s1_preds = s1_model.predict(df_test[s1_features])

        # Initialize test predictions with HC label
        current_preds = np.full(df_test.shape[0], CLASS_MAPPING['HC'])
        disease_indices = (s1_preds == 1) # Indices of samples predicted as 'disease'

        # --- Stage 2: MG vs. CNP ---
        # Only proceed if there are samples predicted as 'disease'
        if np.any(disease_indices):
            s2_features = top_features_dict['MG vs CNP']
            s2_model = get_model_pipeline(s2_features, LinearDiscriminantAnalysis(solver='svd'))
            
            df_train_s2 = df_train[df_train['class_name'] != 'HC'].copy()
            df_train_s2['s2_label'] = df_train_s2['class_name'].apply(lambda x: 0 if x == 'MG' else 1) # 0=MG, 1=CNP
            
            # Check for sufficient data to train stage 2
            if df_train_s2['s2_label'].nunique() > 1:
                s2_model.fit(df_train_s2[s2_features], df_train_s2['s2_label'])
                
                df_test_s2 = df_test[disease_indices]
                s2_preds = s2_model.predict(df_test_s2[s2_features])

                # Update predictions: predicted as MG or CNP
                mg_indices = (s2_preds == 0)
                cnp_indices = (s2_preds == 1)
                
                # Get original indices from df_test
                test_disease_original_indices = df_test.index[disease_indices]
                
                current_preds[disease_indices] = np.where(s2_preds == 0, CLASS_MAPPING['MG'], -1) # Temp label for CNP pool

                # --- Stage 3: CNP Sub-classification ---
                # Only proceed if there are samples predicted as 'CNP'
                if np.any(cnp_indices):
                    s3_features = top_features_dict['CNP Sub-types']
                    s3_model = get_model_pipeline(s3_features, LinearDiscriminantAnalysis(solver='svd'))
                    
                    df_train_s3 = df_train[df_train['class_name'].str.startswith('CNP')].copy()
                    
                    # Check for sufficient data to train stage 3
                    if df_train_s3['label'].nunique() > 1:
                        s3_model.fit(df_train_s3[s3_features], df_train_s3['label'])
                        
                        df_test_s3 = df_test_s2[cnp_indices]
                        s3_preds = s3_model.predict(df_test_s3[s3_features])
                        
                        # Find where in current_preds the CNP cases are
                        cnp_locs_in_current_preds = (current_preds == -1)
                        current_preds[cnp_locs_in_current_preds] = s3_preds
        
        # Store predictions for this fold
        y_pred_full[test_idx] = current_preds

    # Final Evaluation
    f_out.write("\n--- Final Evaluation of Multi-Stage System ---\n")
    final_accuracy = accuracy_score(y_true_full, y_pred_full)
    f_out.write(f"Overall Cross-Validated Accuracy of the Multi-Stage System: {final_accuracy:.4f}\n\n")
    print(f"  Overall CV Accuracy of Multi-Stage System: {final_accuracy:.4f}")

    ordered_names = [INV_CLASS_MAPPING[i] for i in sorted(INV_CLASS_MAPPING.keys())]
    ordered_labels = sorted(INV_CLASS_MAPPING.keys())

    report = classification_report(y_true_full, y_pred_full, target_names=ordered_names, labels=ordered_labels, zero_division=0)
    f_out.write("Final Classification Report (from aggregated CV predictions):\n" + report + "\n\n")

    cm = confusion_matrix(y_true_full, y_pred_full, labels=ordered_labels)
    f_out.write("Final Confusion Matrix (from aggregated CV predictions):\n" + np.array2string(cm) + "\n")
    plot_confusion_matrix(cm, ordered_names, "Multi_Stage_System", results_dir, f_out)


# --- Main Script Execution ---
if __name__ == '__main__':
    print("="*80)
    print("Starting Feature Analysis and Staged Classification (EXP_7)...")
    print("="*80)

    create_results_directory(RESULTS_DIR)
    summary_filepath = os.path.join(RESULTS_DIR, NUMERICAL_SUMMARY_FILENAME)

    with open(summary_filepath, 'w', encoding='utf-8') as f_report:
        f_report.write("="*80 + "\n")
        f_report.write("Feature Analysis & Staged Classification (EXP_7) - Numerical Summary\n")
        f_report.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_report.write(f"Classes modeled: {list(CLASS_DEFINITIONS.keys())}\n")
        f_report.write(f"Random State for reproducibility: {RANDOM_STATE}\n")
        f_report.write("="*80 + "\n\n")

        # 1. Load and Process Data
        raw_items_list = load_raw_sequences_and_labels(
            BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, CSV_ENCODING, CSV_SEPARATOR, MIN_SEQ_LEN_THRESHOLD, f_report
        )
        if not raw_items_list:
            f_report.write("\nCRITICAL: No data loaded. Experiments cannot proceed.\n")
            exit()

        master_df, all_numerical_features = engineer_and_aggregate_features(
            raw_items_list, FEATURE_COLUMNS, f_report
        )
        if master_df.empty:
            f_report.write("\nCRITICAL: Aggregated features DataFrame is empty. Experiments cannot proceed.\n")
            exit()
        
        # Define the scenarios for analysis
        cnp_classes = ['CNP3', 'CNP4', 'CNP6']
        scenarios = {
            'HC vs MG': {
                'classes': ['HC', 'MG'],
                'target_names': ['HC', 'MG'],
                'target_labels': [CLASS_MAPPING['HC'], CLASS_MAPPING['MG']]
            },
            'HC vs CNP': {
                'classes': ['HC'] + cnp_classes,
                'relabel_map': {'HC': 0, 'CNP3': 1, 'CNP4': 1, 'CNP6': 1},
                'target_names': ['HC', 'CNP_Pooled'],
                'target_labels': [0, 1]
            },
            'MG vs CNP': {
                'classes': ['MG'] + cnp_classes,
                'relabel_map': {'MG': 0, 'CNP3': 1, 'CNP4': 1, 'CNP6': 1},
                'target_names': ['MG', 'CNP_Pooled'],
                'target_labels': [0, 1]
            },
            'HC vs All Else': {
                'classes': list(CLASS_DEFINITIONS.keys()),
                'relabel_map': {cls: (0 if cls == 'HC' else 1) for cls in CLASS_DEFINITIONS},
                'target_names': ['HC', 'Diseased'],
                'target_labels': [0, 1]
            },
            'CNP Sub-types': {
                'classes': cnp_classes,
                'target_names': cnp_classes,
                'target_labels': [CLASS_MAPPING[c] for c in cnp_classes]
            }
        }

        # Part 1: Analyze and select top features for each scenario
        top_features_per_scenario = analyze_and_select_features(master_df, all_numerical_features, scenarios, RESULTS_DIR, f_report)

        # Part 2: Run optimized classification for each scenario
        run_optimized_classification(master_df, scenarios, top_features_per_scenario, RESULTS_DIR, f_report)

        # Part 3: Evaluate the multi-stage classifier
        evaluate_multistage_classifier(master_df, all_numerical_features, top_features_per_scenario, RESULTS_DIR, f_report)

        f_report.write("\n" + "="*80 + "\n")
        f_report.write("End of Experiment 7 Report\n")
        f_report.write("="*80 + "\n")

    print(f"\nNumerical summary for all experiments saved to: {summary_filepath}")
    print("\n" + "="*80)
    print("Experiment 7 Finished.")
    print(f"All plots and the summary report have been saved in: {RESULTS_DIR}")
    print("="*80)
