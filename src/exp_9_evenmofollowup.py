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
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# --- Configuration ---
BASE_DIR = './data'
RESULTS_DIR = './results/EXP_9'
os.makedirs(RESULTS_DIR, exist_ok=True) # Ensure directory is created

# Original class definitions
ORIGINAL_CLASS_DEFINITIONS = {
    'HC': {'path': 'Healthy control', 'label': 0},
    'MG': {'path': 'Definite MG', 'label': 1},
    'CNP3': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '3rd'), 'label': 2},
    'CNP4': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '4th'), 'label': 3},
    'CNP6': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '6th'), 'label': 4},
    'TAO': {'path': os.path.join('Non-MG diplopia (CNP, etc)', 'TAO'), 'label': 5},
}

# Classes for modeling (excluding TAO)
CLASSES_TO_EXCLUDE_FOR_MODELING = ['TAO']
CLASS_DEFINITIONS = {
    name: details for name, details in ORIGINAL_CLASS_DEFINITIONS.items()
    if name not in CLASSES_TO_EXCLUDE_FOR_MODELING
}
CLASS_MAPPING = {name: details['label'] for name, details in CLASS_DEFINITIONS.items()}
INV_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}
CLASS_LABELS = sorted(INV_CLASS_MAPPING.keys())
CLASS_NAMES = [INV_CLASS_MAPPING[l] for l in CLASS_LABELS]

# --- Core Parameters ---
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']
CSV_ENCODING = 'utf-16-le'
CSV_SEPARATOR = ','
MIN_SEQ_LEN_THRESHOLD = 50
EXP_PREFIX = 'EXP_9_'
NUMERICAL_SUMMARY_FILENAME = f'{EXP_PREFIX}numerical_summary.txt'
RANDOM_STATE = 42
N_TOP_FEATURES = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.discriminant_analysis')
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# --- Utility and Plotting Functions ---
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
    print(f"  Saved confusion matrix for {model_name}{suffix}")

def plot_accuracy_vs_param(param_values, accuracies, param_name, title, results_dir, f_out):
    """Saves a plot of accuracy vs. a given parameter."""
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, accuracies, marker='o', linestyle='-')
    plt.title(title, fontsize=16)
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel('Mean CV Accuracy', fontsize=12)
    plt.grid(True)
    if 'Sampling' in param_name:
        plt.xscale('log')
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'{EXP_PREFIX}{title.replace(" ", "_")}.png')
    plt.savefig(plot_path)
    plt.close()
    f_out.write(f"Accuracy plot saved to: {plot_path}\n")
    print(f"  Saved plot: {title}")

# --- Data Loading and Feature Engineering Functions ---
def parse_frequency_from_filename(filename):
    """Extracts saccade frequency (e.g., 0.5, 0.75, 1) from a filename."""
    match = re.search(r'\((\d+(\.\d+)?)\s*Hz\)', filename, re.IGNORECASE)
    return float(match.group(1)) if match else np.nan

def load_raw_sequences_and_labels(base_dir, class_definitions_dict, feature_columns_expected, encoding, separator, min_seq_len_threshold, f_out):
    """Loads raw time-series data, including metadata needed for all tasks."""
    f_out.write("\n" + "="*80 + "\nPhase: Data Loading\n" + "="*80 + "\n")
    print("\n" + "="*50 + "\nStarting Data Loading...\n" + "="*50)
    raw_items = []
    for class_name_key, class_details in class_definitions_dict.items():
        label = class_details['label']
        class_dir_abs = os.path.join(base_dir, class_details['path'])
        if not os.path.isdir(class_dir_abs):
            continue
        patient_dirs = [d for d in os.listdir(class_dir_abs) if os.path.isdir(os.path.join(class_dir_abs, d))]
        for patient_folder_name in tqdm(patient_dirs, desc=f"  Patients in {class_name_key}"):
            patient_dir_path = os.path.join(class_dir_abs, patient_folder_name)
            csv_files = glob.glob(os.path.join(patient_dir_path, '*.csv'))
            for csv_file_path in csv_files:
                try:
                    df_full = pd.read_csv(csv_file_path, encoding=encoding, sep=separator)
                    df_full.columns = [col.strip() for col in df_full.columns]
                    if any(col not in df_full.columns for col in feature_columns_expected) or len(df_full) < min_seq_len_threshold:
                        continue
                    
                    df_features = df_full[feature_columns_expected].copy()
                    for col in df_features.columns:
                        df_features.loc[:, col] = pd.to_numeric(df_features[col], errors='coerce')
                    if df_features.isnull().sum().sum() > 0.1 * df_features.size:
                        continue
                    df_features = df_features.fillna(0)
                    
                    # **FIX**: Added back all required keys
                    raw_items.append({
                        'data': df_features.values.astype(np.float32), 
                        'label': label,
                        'patient_id': patient_folder_name,
                        'filename': os.path.basename(csv_file_path),
                        'class_name': class_name_key,
                        'frequency': parse_frequency_from_filename(os.path.basename(csv_file_path))
                    })
                except Exception:
                    pass
    f_out.write(f"Data loading complete. Loaded {len(raw_items)} raw sequences.\n\n")
    print(f"Data loading complete. Loaded {len(raw_items)} raw sequences.")
    return raw_items

def engineer_and_aggregate_features(raw_items_dicts, original_feature_names, f_out):
    """Creates aggregated statistical features from time-series data."""
    f_out.write("="*80 + "\nPhase: Feature Engineering and Aggregation\n" + "="*80 + "\n")
    print("\n" + "="*50 + "\nStarting Feature Engineering and Aggregation...\n" + "="*50)
    if not raw_items_dicts:
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
        if not engineered_feature_names: 
            engineered_feature_names = df_all_features.columns.tolist()
            
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

# --- Modeling Core for LDA (copied from Exp 7) ---
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
    all_y_true, all_y_pred = [], []

    for train_idx, test_idx in tqdm(cv.split(X_df, y_series), total=cv.get_n_splits(), desc=f"  CV for {model_name}{suffix}"):
        X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
        y_train, y_test = y_series.iloc[train_idx], y_series.iloc[test_idx]
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    mean_accuracy = accuracy_score(all_y_true, all_y_pred)
    f_out.write(f"Aggregated Cross-validated Accuracy: {mean_accuracy:.4f}\n\n")
    print(f"  Aggregated Cross-validated Accuracy: {mean_accuracy:.4f}")
    
    report = classification_report(all_y_true, all_y_pred, target_names=ordered_target_names, labels=numeric_labels, zero_division=0)
    f_out.write("Classification Report:\n" + report + "\n\n")
    
    cm = confusion_matrix(all_y_true, all_y_pred, labels=numeric_labels)
    f_out.write("Confusion Matrix:\n" + np.array2string(cm) + "\n")
    plot_confusion_matrix(cm, ordered_target_names, model_name, results_dir, f_out, suffix)
    return mean_accuracy

# --- PyTorch NN Components (For Tasks 1 & 2) ---
class SaccadeSequenceDataset(Dataset):
    """Custom PyTorch Dataset for saccade sequences."""
    def __init__(self, sequences, labels):
        self.sequences = [torch.tensor(s, dtype=torch.float32) for s in sequences]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch):
    """Pads sequences within a batch to equal length."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences])
    sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return sequences_padded, labels, lengths

class SimpleLSTMClassifier(nn.Module):
    """A simple LSTM-based classifier."""
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed_input)
        out = self.fc(hidden[-1])
        return out

def train_and_evaluate_nn(sequences, labels, n_features, n_classes, f_out, model_name_suffix):
    """Full training and CV evaluation pipeline for the NN model."""
    f_out.write(f"\n--- Evaluating NN Model for: {model_name_suffix} ---\n")
    print(f"\nEvaluating NN Model for: {model_name_suffix}")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    accuracies = []

    for fold, (train_idx, test_idx) in enumerate(tqdm(skf.split(sequences, labels), total=5, desc=f"  CV for {model_name_suffix}")):
        X_train_seq, X_test_seq = [sequences[i] for i in train_idx], [sequences[i] for i in test_idx]
        y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
        
        train_dataset = SaccadeSequenceDataset(X_train_seq, y_train)
        test_dataset = SaccadeSequenceDataset(X_test_seq, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        model = SimpleLSTMClassifier(input_dim=n_features, hidden_dim=64, output_dim=n_classes, n_layers=2).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(10):
            for seq_padded, labels_batch, lengths in train_loader:
                seq_padded, labels_batch = seq_padded.to(DEVICE), labels_batch.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(seq_padded, lengths)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for seq_padded, labels_batch, lengths in test_loader:
                seq_padded, labels_batch = seq_padded.to(DEVICE), labels_batch.to(DEVICE)
                outputs = model(seq_padded, lengths)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_true.extend(labels_batch.cpu().numpy())
        
        accuracies.append(accuracy_score(all_true, all_preds))
        
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    f_out.write(f"Mean CV Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})\n")
    print(f"  Mean CV Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
    return mean_accuracy

# --- Task 1: Optimal Target Velocity Analysis ---
def run_frequency_analysis(raw_items_list, n_features, n_classes, f_out):
    f_out.write("\n" + "="*80 + "\nPart 1: Optimal Target Velocity (Frequency) Analysis\n" + "="*80 + "\n")
    print("\n" + "="*50 + "\nRunning Part 1: Frequency Analysis...\n" + "="*50)
    
    frequencies = sorted([f for f in pd.Series([item['frequency'] for item in raw_items_list]).dropna().unique()])
    freq_accuracies = []

    for freq in frequencies:
        freq_items = [item for item in raw_items_list if item['frequency'] == freq]
        sequences = [item['data'] for item in freq_items]
        labels = [item['label'] for item in freq_items]
        
        if len(sequences) < 20:
            f_out.write(f"Skipping frequency {freq}Hz due to insufficient samples ({len(sequences)}).\n")
            continue
            
        accuracy = train_and_evaluate_nn(sequences, labels, n_features, n_classes, f_out, f"{freq}Hz")
        freq_accuracies.append(accuracy)

    f_out.write("\n--- Frequency Analysis Summary ---\n")
    for freq, acc in zip(frequencies, freq_accuracies):
        f_out.write(f"  - {freq} Hz: {acc:.4f} accuracy\n")

    if freq_accuracies:
        best_freq_idx = np.argmax(freq_accuracies)
        f_out.write(f"\nConclusion: The optimal frequency appears to be {frequencies[best_freq_idx]} Hz with an accuracy of {freq_accuracies[best_freq_idx]:.4f}.\n")

# --- Task 2: Time Series Downsampling Analysis ---
def run_downsampling_analysis(raw_items_list, n_features, n_classes, f_out):
    f_out.write("\n" + "="*80 + "\nPart 2: Time Series Downsampling Analysis\n" + "="*80 + "\n")
    print("\n" + "="*50 + "\nRunning Part 2: Downsampling Analysis...\n" + "="*50)

    sampling_rates = [x for x in range(50,2000,50)]
    sampling_accuracies = []
    
    sequences = [item['data'] for item in raw_items_list]
    labels = [item['label'] for item in raw_items_list]

    for rate in sampling_rates:
        downsampled_sequences = [seq[::rate] for seq in sequences if len(seq[::rate]) > 10]
        downsampled_labels = [labels[i] for i, seq in enumerate(sequences) if len(seq[::rate]) > 10]

        accuracy = train_and_evaluate_nn(downsampled_sequences, downsampled_labels, n_features, n_classes, f_out, f"Downsample 1:{rate}")
        sampling_accuracies.append(accuracy)

    plot_accuracy_vs_param(
        sampling_rates, sampling_accuracies,
        "Sampling Rate (Keep 1 every N points)",
        "Model Accuracy vs. Downsampling Rate",
        RESULTS_DIR, f_out
    )
    
    best_rate_idx = np.argmax(sampling_accuracies)
    f_out.write("\n--- Downsampling Analysis Summary ---\n")
    f_out.write(f"Conclusion: The optimal sampling rate is 1:{sampling_rates[best_rate_idx]} with an accuracy of {sampling_accuracies[best_rate_idx]:.4f}.\n")

# --- Task 3: Saccadic Fatigue and Advanced Feature Analysis ---
def engineer_fatigue_features(raw_items_dicts, original_feature_names, f_out):
    f_out.write("\n" + "="*80 + "\nPhase: Engineering Fatigue and Advanced Features\n" + "="*80 + "\n")
    print("\n" + "="*50 + "\nStarting Advanced Feature Engineering...\n" + "="*50)
    
    advanced_rows = []
    for item in tqdm(raw_items_dicts, desc="  Engineering Advanced Features"):
        df = pd.DataFrame(item['data'], columns=original_feature_names)
        
        for pos_col in ['LH', 'RH', 'LV', 'RV']:
            df[f'{pos_col}_Vel'] = df[pos_col].diff().fillna(0)
        df['ErrorH_L'] = df['LH'] - df['TargetH']
        df['ErrorH_R'] = df['RH'] - df['TargetH']
        
        row = {'label': item['label'], 'class_name': INV_CLASS_MAPPING[item['label']]}
        
        row['LH_range'] = df['LH'].max() - df['LH'].min()
        row['RH_range'] = df['RH'].max() - df['RH'].min()
        
        row['LH_Vel_peak'] = df['LH_Vel'].abs().max()
        row['RH_Vel_peak'] = df['RH_Vel'].abs().max()

        row['LH_main_seq_ratio'] = row['LH_Vel_peak'] / (row['LH_range'] + 1e-6)
        row['RH_main_seq_ratio'] = row['RH_Vel_peak'] / (row['RH_range'] + 1e-6)

        midpoint = len(df) // 2
        err_h_l_1st_half = df['ErrorH_L'].iloc[:midpoint].abs().mean()
        err_h_l_2nd_half = df['ErrorH_L'].iloc[midpoint:].abs().mean()
        row['ErrorH_L_decrement'] = err_h_l_2nd_half - err_h_l_1st_half
        
        err_h_r_1st_half = df['ErrorH_R'].iloc[:midpoint].abs().mean()
        err_h_r_2nd_half = df['ErrorH_R'].iloc[midpoint:].abs().mean()
        row['ErrorH_R_decrement'] = err_h_r_2nd_half - err_h_r_1st_half

        advanced_rows.append(row)
        
    adv_df = pd.DataFrame(advanced_rows)
    numerical_cols = [c for c in adv_df.columns if c not in ['label', 'class_name']]
    f_out.write(f"Advanced features DataFrame created with shape: {adv_df.shape}\n")
    return adv_df, numerical_cols

def run_fatigue_feature_analysis(raw_items_list, f_out):
    f_out.write("\n" + "="*80 + "\nPart 3: Saccadic Fatigue Feature Analysis\n" + "="*80 + "\n")
    print("\n" + "="*50 + "\nRunning Part 3: Fatigue Feature Analysis...\n" + "="*50)
    
    adv_df, numerical_features = engineer_fatigue_features(raw_items_list, FEATURE_COLUMNS, f_out)
    X = adv_df[numerical_features]
    y = adv_df['label']

    train_and_evaluate_single_model(
        X, y, numerical_features, "LDA",
        LinearDiscriminantAnalysis(solver='svd'),
        CLASS_NAMES, CLASS_LABELS,
        RESULTS_DIR, f_out, suffix="_Fatigue_Features"
    )

# --- Task 4: More Affected vs. Less Affected Eye Analysis ---
def create_affected_eye_features(master_df, f_out):
    f_out.write("\n" + "="*80 + "\nPhase: Engineering Affected-Eye Features\n" + "="*80 + "\n")
    print("\n" + "="*50 + "\nStarting Affected-Eye Feature Engineering...\n" + "="*50)
    
    df = master_df.copy()
    affected_eye_rows = []
    
    # Define a list of base feature names without eye-specific identifiers
    base_measures = ['H_Vel', 'V_Vel', 'ErrorH', 'ErrorV', 'LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']
    stat_suffixes = ['_mean', '_std', '_median', '_iqr']

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Processing Affected Eye"):
        left_error = abs(row.get('ErrorH_L_mean', 0)) + abs(row.get('ErrorV_L_mean', 0))
        right_error = abs(row.get('ErrorH_R_mean', 0)) + abs(row.get('ErrorV_R_mean', 0))
        
        is_left_more_affected = left_error >= right_error
        new_row = {'label': row['label'], 'class_name': row['class_name']}
        
        for meas in base_measures:
            for stat in stat_suffixes:
                if 'H' in meas or 'V' in meas or 'Vel' in meas:
                    # Generic features like ErrorH, LV_Vel, etc.
                    base_name = f"{meas}{stat}"
                    left_feature = base_name.replace('H', 'H_L').replace('V', 'V_L')
                    right_feature = base_name.replace('H', 'H_R').replace('V', 'V_R')
                else:
                    # Positional features like LH, RV
                    left_feature = f"{meas}{stat}"
                    right_feature = f"{meas}{stat}"

                if 'L' in left_feature and left_feature in row and right_feature in row:
                    mae_col_name = f"MAE_{left_feature.replace('_L', '')}"
                    lae_col_name = f"LAE_{left_feature.replace('_L', '')}"
                    if is_left_more_affected:
                        new_row[mae_col_name] = row[left_feature]
                        new_row[lae_col_name] = row[right_feature]
                    else:
                        new_row[mae_col_name] = row[right_feature]
                        new_row[lae_col_name] = row[left_feature]

        affected_eye_rows.append(new_row)
        
    affected_df = pd.DataFrame(affected_eye_rows).dropna(axis=1, how='all')
    f_out.write(f"Affected-eye DataFrame created with shape: {affected_df.shape}\n")
    return affected_df

def run_affected_eye_analysis(master_df, f_out):
    f_out.write("\n" + "="*80 + "\nPart 4: Affected vs. Less-Affected Eye Analysis\n" + "="*80 + "\n")
    print("\n" + "="*50 + "\nRunning Part 4: Affected Eye Analysis...\n" + "="*50)

    affected_df = create_affected_eye_features(master_df, f_out)
    
    # Scenario 1: More Affected Eye Only
    mae_features = [c for c in affected_df.columns if c.startswith('MAE_')]
    if mae_features:
        X_mae = affected_df[mae_features]
        y_mae = affected_df['label']
        f_out.write("\n--- Evaluating on More Affected Eye (MAE) Features Only ---\n")
        train_and_evaluate_single_model(
            X_mae, y_mae, mae_features, "LDA", LinearDiscriminantAnalysis(solver='svd'),
            CLASS_NAMES, CLASS_LABELS, RESULTS_DIR, f_out, suffix="_MAE_Only"
        )

    # Scenario 2: Both Eyes (MAE + LAE features)
    all_eye_features = [c for c in affected_df.columns if c.startswith('MAE_') or c.startswith('LAE_')]
    if all_eye_features:
        X_both = affected_df[all_eye_features]
        y_both = affected_df['label']
        f_out.write("\n--- Evaluating on Both MAE and LAE Features ---\n")
        train_and_evaluate_single_model(
            X_both, y_both, all_eye_features, "LDA", LinearDiscriminantAnalysis(solver='svd'),
            CLASS_NAMES, CLASS_LABELS, RESULTS_DIR, f_out, suffix="_MAE_and_LAE"
        )

# --- Main Script Execution ---
if __name__ == '__main__':
    print("="*80)
    print("Starting Advanced Saccade Analysis (EXP_9)...")
    print("="*80)
    
    summary_filepath = os.path.join(RESULTS_DIR, NUMERICAL_SUMMARY_FILENAME)
    
    with open(summary_filepath, 'w', encoding='utf-8') as f_report:
        f_report.write("="*80 + "\n")
        f_report.write("Advanced Saccade Analysis (EXP_9) - Numerical Summary\n")
        f_report.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_report.write(f"Device used for NN models: {DEVICE}\n")
        f_report.write("="*80 + "\n\n")

        # Load raw data - needed for all experiments
        raw_items_list = load_raw_sequences_and_labels(
            BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, CSV_ENCODING, CSV_SEPARATOR, MIN_SEQ_LEN_THRESHOLD, f_report
        )
        if not raw_items_list:
            f_report.write("CRITICAL: No data loaded. Exiting.\n")
            exit()
            
        n_features = raw_items_list[0]['data'].shape[1]
        n_classes = len(CLASS_DEFINITIONS)
        
        # --- Run Task 1 & 2 (NN-based) ---
        #run_frequency_analysis(raw_items_list, n_features, n_classes, f_report)
        run_downsampling_analysis(raw_items_list, n_features, n_classes, f_report)
        
        # --- Run Task 3 ---
        #run_fatigue_feature_analysis(raw_items_list, f_report)
        
        # --- Run Task 4 ---
        master_df, _ = engineer_and_aggregate_features(raw_items_list, FEATURE_COLUMNS, f_report)
        if not master_df.empty:
            run_affected_eye_analysis(master_df, f_report)
        else:
            f_report.write("\nCRITICAL: Master DataFrame for aggregated features is empty. Skipping Task 4.\n")
            print("\nERROR: Could not create master_df. Skipping Task 4.")

        f_report.write("\n" + "="*80 + "\nEnd of Experiment 9 Report\n" + "="*80 + "\n")

    print(f"\nNumerical summary for all experiments saved to: {summary_filepath}")
    print("\n" + "="*80)
    print("Experiment 9 Finished.")
    print(f"All plots and the summary report have been saved in: {RESULTS_DIR}")
    print("="*80)
