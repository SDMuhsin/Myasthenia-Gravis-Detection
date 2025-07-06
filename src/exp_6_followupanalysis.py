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

from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


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
ORDERED_CLASS_NAMES = [INV_CLASS_MAPPING[i] for i in sorted(INV_CLASS_MAPPING.keys())]
MODEL_CLASS_LABELS = sorted(CLASS_MAPPING.values())

# --- Core Parameters ---
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']
CSV_ENCODING = 'utf-16-le'
CSV_SEPARATOR = ','
MIN_SEQ_LEN_THRESHOLD = 50
RESULTS_DIR = './results'
EXP_PREFIX = 'EXP_6_ADDITIONAL_INVESTIGATIONS_'
NUMERICAL_SUMMARY_FILENAME = f'{EXP_PREFIX}numerical_summary.txt'
RANDOM_STATE = 42

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

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.
    (Adapted from sklearn documentation)
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True, random_state=RANDOM_STATE
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    axes[0].plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Fit times (s)")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_corrected = fit_times_mean + fit_times_std  # Add std to avoid division by zero
    axes[2].grid()
    axes[2].plot(fit_time_corrected, test_scores_mean, "o-")
    axes[2].fill_between(fit_time_corrected, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("Fit times (s)")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


# --- Data Loading and Feature Engineering ---
def parse_frequency_from_filename(filename):
    """Extracts saccade frequency (e.g., 0.5, 0.75, 1) from a filename."""
    match = re.search(r'\((\d+(\.\d+)?)\s*Hz\)', filename, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 'N/A' # Return a string for easy handling later

def load_raw_sequences_and_labels(base_dir, class_definitions_dict, feature_columns_expected, encoding, separator, min_seq_len_threshold, f_out):
    """Loads data and now also extracts saccade frequency from filenames."""
    f_out.write("=" * 80 + "\n")
    f_out.write("Phase: Data Loading and Frequency Extraction\n")
    f_out.write("=" * 80 + "\n")
    print("\n" + "="*50 + "\nStarting Data Loading and Frequency Extraction...\n" + "="*50)
    raw_items = []

    for class_name_key, class_details in class_definitions_dict.items():
        label = class_details['label']
        class_dir_abs = os.path.join(base_dir, class_details['path'])
        msg = f"\nProcessing Class: '{class_name_key}' (Label: {label}) from path: {class_dir_abs}\n"
        print(msg.strip())
        f_out.write(msg)
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

                    # NEW: Extract frequency
                    frequency = parse_frequency_from_filename(os.path.basename(csv_file_path))

                    raw_items.append({
                        'data': df_features.values.astype(np.float32), 'label': label,
                        'patient_id': patient_folder_name, 'filename': os.path.basename(csv_file_path),
                        'class_name': class_name_key, 'frequency': frequency
                    })
                except Exception as e:
                    pass # Silently skip problematic files for this experiment

    summary_msg = f"\nData loading complete. Loaded {len(raw_items)} raw sequences.\n"
    print(summary_msg.strip())
    f_out.write(summary_msg + "-" * 80 + "\n\n")
    return raw_items

def engineer_and_aggregate_features(raw_items_dicts, original_feature_names, f_out):
    """Combines feature engineering and aggregation into one step for efficiency."""
    f_out.write("="*80 + "\n")
    f_out.write("Phase: Feature Engineering and Aggregation\n")
    f_out.write("="*80 + "\n")
    print("\n" + "="*50 + "\nStarting Feature Engineering and Aggregation...\n" + "="*50)
    
    if not raw_items_dicts:
        f_out.write("No raw items to process. Returning empty DataFrame.\n")
        return pd.DataFrame(), []

    aggregated_data_rows = []
    engineered_feature_names = []

    for item in tqdm(raw_items_dicts, desc="  Engineering & Aggregating"):
        df_original = pd.DataFrame(item['data'], columns=original_feature_names)
        
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
        if not engineered_feature_names: engineered_feature_names = df_all_features.columns.tolist()

        # --- Aggregation ---
        current_row = {
            'patient_id': item['patient_id'], 'filename': item['filename'],
            'class_name': item['class_name'], 'label': item['label'],
            'frequency': item['frequency']
        }
        for feature_name in engineered_feature_names:
            ft_ts = df_all_features[feature_name]
            current_row[f'{feature_name}_mean'] = np.mean(ft_ts)
            current_row[f'{feature_name}_std'] = np.std(ft_ts)
            current_row[f'{feature_name}_median'] = np.median(ft_ts)
            current_row[f'{feature_name}_iqr'] = scipy_iqr(ft_ts)
        aggregated_data_rows.append(current_row)

    agg_df = pd.DataFrame(aggregated_data_rows)
    f_out.write(f"Aggregated features DataFrame created with shape: {agg_df.shape}\n")
    f_out.write(f"Aggregated numerical features list: {[f for f in agg_df.columns if f not in ['patient_id', 'filename', 'class_name', 'label', 'frequency']]}\n\n")
    
    return agg_df, [f for f in agg_df.columns if '_mean' in f or '_std' in f or '_median' in f or '_iqr' in f]


# --- Modeling Core ---
def get_model_pipeline(numerical_features, categorical_features, model):
    """Creates a full preprocessing and modeling pipeline."""
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough'
    )
    return Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

def train_and_evaluate_single_model(X_df, y_series, numerical_features, categorical_features, model_name, model, ordered_target_names, numeric_labels, results_dir, f_out, suffix=""):
    """Trains and evaluates a single model pipeline, returning key metrics."""
    f_out.write(f"\n--- Model: {model_name}{suffix} ---\n")
    print(f"\nTraining and evaluating: {model_name}{suffix}")
    
    pipeline = get_model_pipeline(numerical_features, categorical_features, model)
    
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

    if not all_y_pred:
        f_out.write("Model training failed for all folds. Skipping evaluation.\n")
        return 0.0, None, None

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


# --- Experiment-Specific Functions ---

def run_learning_curve_analysis(X_df, y_series, numerical_features, categorical_features, results_dir, f_out):
    """Q1: Generates and analyzes learning curves to predict gains from more data."""
    f_out.write("\n" + "="*80 + "\n")
    f_out.write("Experiment for Q1: Sample Size (Learning Curve Analysis)\n")
    f_out.write("="*80 + "\n")
    print("\n" + "="*50 + "\nRunning Experiment for Q1: Sample Size...\n" + "="*50)

    # Use LDA as it was a strong performer and is computationally efficient
    model = LinearDiscriminantAnalysis(solver='svd')
    pipeline = get_model_pipeline(numerical_features, categorical_features, model)
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    title = f"Learning Curves (LDA)"
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    plot_learning_curve(pipeline, title, X_df, y_series, axes=axes, ylim=(0.1, 1.01), cv=cv, n_jobs=-1)
    
    plot_path = os.path.join(results_dir, f'{EXP_PREFIX}Learning_Curve_LDA.png')
    plt.savefig(plot_path)
    plt.close()
    
    f_out.write("Learning curve analysis complete.\n")
    f_out.write(f"Plot saved to: {plot_path}\n\n")
    f_out.write("Interpretation:\n")
    f_out.write("1.  **Examine the gap between the red (Training) and green (Cross-validation) curves.** A wide gap suggests high variance (overfitting), meaning the model could benefit from more data.\n")
    f_out.write("2.  **Examine the slope of the green (Cross-validation) curve.** If the curve is still trending upwards as more samples are added, it indicates that model performance is not yet saturated and will likely improve with a larger cohort.\n")
    f_out.write("3.  **Examine the final score.** If the cross-validation score has plateaued at a low value, it may indicate high bias (underfitting), and simply adding more data of the same kind might not help. In this case, feature engineering or a more complex model would be needed.\n\n")
    print(f"  Learning curve plot saved to {plot_path}")

def run_frequency_analysis(master_df, numerical_features, results_dir, f_out):
    """Q2: Analyzes the impact of saccade frequency on model performance."""
    f_out.write("\n" + "="*80 + "\n")
    f_out.write("Experiment for Q2: Recording Parameters (Frequency Analysis)\n")
    f_out.write("="*80 + "\n")
    print("\n" + "="*50 + "\nRunning Experiment for Q2: Recording Parameters...\n" + "="*50)
    
    # --- Part 1: Frequency as a feature ---
    f_out.write("\nPart 1: Using Saccade Frequency as a Model Feature\n" + "-"*50 + "\n")
    df_freq = master_df[master_df['frequency'] != 'N/A'].copy()
    df_freq['frequency'] = df_freq['frequency'].astype(str) # Treat as categorical
    
    X = df_freq[numerical_features + ['frequency']]
    y = df_freq['label']
    
    train_and_evaluate_single_model(X, y, numerical_features, ['frequency'], "LDA", 
                                    LinearDiscriminantAnalysis(solver='svd'), 
                                    ORDERED_CLASS_NAMES, MODEL_CLASS_LABELS, results_dir, f_out, suffix="_with_Frequency_Feature")

    # --- Part 2: Stratified analysis by frequency ---
    f_out.write("\nPart 2: Performance Stratified by Saccade Frequency\n" + "-"*50 + "\n")
    available_frequencies = sorted([f for f in master_df['frequency'].unique() if f != 'N/A'])
    f_out.write(f"Found frequencies to analyze: {available_frequencies}\n")
    
    for freq in available_frequencies:
        f_out.write(f"\n--- Analysis for Frequency: {freq} Hz ---\n")
        df_subset = master_df[master_df['frequency'] == freq].copy()
        
        if df_subset['class_name'].nunique() < len(CLASS_DEFINITIONS):
            f_out.write(f"Skipping {freq}Hz analysis: Not all classes are present in this subset.\n")
            continue
        
        X_sub = df_subset[numerical_features]
        y_sub = df_subset['label']
        
        # Check if any class has fewer samples than n_splits for CV
        if any(y_sub.value_counts() < 5):
             f_out.write(f"Skipping {freq}Hz analysis: One or more classes has fewer than 5 samples, which is too few for StratifiedKFold.\n")
             continue

        train_and_evaluate_single_model(X_sub, y_sub, numerical_features, [], "LDA",
                                        LinearDiscriminantAnalysis(solver='svd'), 
                                        ORDERED_CLASS_NAMES, MODEL_CLASS_LABELS, results_dir, f_out, suffix=f"_{freq}Hz_Only")
    f_out.write("\nFrequency analysis complete.\n")

def run_group_contrast_analysis(master_df, numerical_features, results_dir, f_out):
    """Q3: Analyzes performance on specific group comparisons."""
    f_out.write("\n" + "="*80 + "\n")
    f_out.write("Experiment for Q3: Group Contrasts\n")
    f_out.write("="*80 + "\n")
    print("\n" + "="*50 + "\nRunning Experiment for Q3: Group Contrasts...\n" + "="*50)

    # --- Contrast 1: MG vs HC ---
    f_out.write("\n--- Contrast 1: MG vs. HC ---\n")
    df_mg_hc = master_df[master_df['class_name'].isin(['MG', 'HC'])].copy()
    X_mg_hc = df_mg_hc[numerical_features]
    y_mg_hc = df_mg_hc['label']
    mg_hc_names = ['HC', 'MG']
    mg_hc_labels = [CLASS_MAPPING[name] for name in mg_hc_names]
    train_and_evaluate_single_model(X_mg_hc, y_mg_hc, numerical_features, [], "LDA", 
                                    LinearDiscriminantAnalysis(solver='svd'), 
                                    mg_hc_names, mg_hc_labels, results_dir, f_out, suffix="_MG_vs_HC")

    # --- Contrast 2: All CNP (pooled) vs HC ---
    f_out.write("\n--- Contrast 2: CNP (Pooled) vs. HC ---\n")
    cnp_classes = ['CNP3', 'CNP4', 'CNP6']
    df_cnp_hc = master_df[master_df['class_name'].isin(cnp_classes + ['HC'])].copy()
    # Create a new pooled label: 0 for HC, 1 for CNP
    df_cnp_hc['pooled_label'] = df_cnp_hc['class_name'].apply(lambda x: 0 if x == 'HC' else 1)
    X_cnp_hc = df_cnp_hc[numerical_features]
    y_cnp_hc = df_cnp_hc['pooled_label']
    cnp_hc_names = ['HC', 'CNP_Pooled']
    cnp_hc_labels = [0, 1]
    train_and_evaluate_single_model(X_cnp_hc, y_cnp_hc, numerical_features, [], "LDA", 
                                    LinearDiscriminantAnalysis(solver='svd'), 
                                    cnp_hc_names, cnp_hc_labels, results_dir, f_out, suffix="_CNP_vs_HC")

    # --- Contrast 3: All CNP subgroups vs HC ---
    f_out.write("\n--- Contrast 3: CNP Subgroups vs. HC ---\n")
    df_cnps_hc = master_df[master_df['class_name'].isin(cnp_classes + ['HC'])].copy()
    X_cnps_hc = df_cnps_hc[numerical_features]
    y_cnps_hc = df_cnps_hc['label']
    cnps_hc_names = ['HC'] + cnp_classes
    cnps_hc_labels = [CLASS_MAPPING[name] for name in cnps_hc_names]
    train_and_evaluate_single_model(X_cnps_hc, y_cnps_hc, numerical_features, [], "LDA", 
                                    LinearDiscriminantAnalysis(solver='svd'), 
                                    cnps_hc_names, cnps_hc_labels, results_dir, f_out, suffix="_CNPsubgroups_vs_HC")
    
    f_out.write("\nGroup contrast analysis complete.\n")


# --- Main Script Execution ---
if __name__ == '__main__':
    print("="*80)
    print("Starting Additional Investigations (EXP_6)...")
    print("This script will perform analyses for sample size, recording parameters, and group contrasts.")
    print("="*80)

    create_results_directory(RESULTS_DIR)
    summary_filepath = os.path.join(RESULTS_DIR, NUMERICAL_SUMMARY_FILENAME)

    with open(summary_filepath, 'w', encoding='utf-8') as f_report:
        f_report.write("="*80 + "\n")
        f_report.write("Additional Investigations (EXP_6) - Numerical Summary\n")
        f_report.write("="*80 + "\n")
        f_report.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_report.write(f"Base Directory: {BASE_DIR}\n")
        f_report.write(f"Classes modeled in multi-class analysis: {list(CLASS_DEFINITIONS.keys())}\n")
        f_report.write(f"Random State for reproducibility: {RANDOM_STATE}\n")
        f_report.write("="*80 + "\n\n")

        # 1. Load and Process Data
        raw_items_list = load_raw_sequences_and_labels(
            BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, CSV_ENCODING, CSV_SEPARATOR, MIN_SEQ_LEN_THRESHOLD, f_report
        )
        if not raw_items_list:
            f_report.write("\nCRITICAL: No data loaded. Experiments cannot proceed.\n")
            exit()

        master_df, numerical_features = engineer_and_aggregate_features(
            raw_items_list, FEATURE_COLUMNS, f_report
        )
        if master_df.empty:
            f_report.write("\nCRITICAL: Aggregated features DataFrame is empty. Experiments cannot proceed.\n")
            exit()
        
        # --- Run Experiments ---
        # Q1: Sample Size
        run_learning_curve_analysis(master_df[numerical_features], master_df['label'], numerical_features, [], RESULTS_DIR, f_report)

        # Q2: Recording Parameters
        run_frequency_analysis(master_df, numerical_features, RESULTS_DIR, f_report)
        
        # Q3: Group Contrasts
        run_group_contrast_analysis(master_df, numerical_features, RESULTS_DIR, f_report)

        f_report.write("\n" + "="*80 + "\n")
        f_report.write("End of Additional Investigations Report (EXP_6)\n")
        f_report.write("="*80 + "\n")

    print(f"\nNumerical summary for all experiments saved to: {summary_filepath}")
    print("\n" + "="*80)
    print("Additional Investigations (EXP_6) Finished.")
    print(f"All plots and the summary report have been saved in: {RESULTS_DIR}")
    print("="*80)
