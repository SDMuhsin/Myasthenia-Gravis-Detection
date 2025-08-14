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
RESULTS_DIR = './results/EXP_10'
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
EXP_PREFIX = 'EXP_10_'
NUMERICAL_SUMMARY_FILENAME = f'{EXP_PREFIX}frequency_analysis_summary.txt'
RANDOM_STATE = 42
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

def plot_frequency_performance(results_df, results_dir, f_out):
    """Plots the performance of different models across various frequencies."""
    plt.figure(figsize=(12, 8))
    for model in results_df.columns:
        plt.plot(results_df.index, results_df[model], marker='o', linestyle='-', label=model)
    
    plt.title('Model Accuracy vs. Saccade Frequency 📊', fontsize=16)
    plt.xlabel('Saccade Frequency (Hz)', fontsize=12)
    plt.ylabel('Mean Cross-Validated Accuracy', fontsize=12)
    plt.legend(title='Model')
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'{EXP_PREFIX}frequency_vs_accuracy_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    f_out.write(f"\nComparison plot saved to: {plot_path}\n")
    print(f"\n📈 Saved frequency performance comparison plot.")

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
                    df_features = df_features.apply(pd.to_numeric, errors='coerce').fillna(0)
                    
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
    f_out.write(f"Data loading complete. Loaded {len(raw_items)} raw sequences.\n")
    print(f"Data loading complete. Loaded {len(raw_items)} raw sequences.")
    return raw_items

def engineer_and_aggregate_features(raw_items_dicts, original_feature_names, f_out):
    """Creates aggregated statistical features from time-series data for a subset of items."""
    f_out.write("  - Engineering aggregated features for LDA...\n")
    aggregated_data_rows = []
    
    for item in raw_items_dicts:
        df_original = pd.DataFrame(item['data'], columns=original_feature_names)
        
        # Engineer velocity and error features
        engineered_parts = []
        for pos_col in ['LH', 'RH', 'LV', 'RV']:
            engineered_parts.append(df_original[pos_col].diff().fillna(0).rename(f'{pos_col}_Vel'))
        error_defs = [('LH', 'TargetH', 'ErrorH_L'), ('RH', 'TargetH', 'ErrorH_R'), ('LV', 'TargetV', 'ErrorV_L'), ('RV', 'TargetV', 'ErrorV_R')]
        for eye_col, target_col, error_col in error_defs:
            engineered_parts.append((df_original[eye_col] - df_original[target_col]).rename(error_col))
        
        df_all = pd.concat([df_original] + engineered_parts, axis=1)
        
        current_row = {'label': item['label']}
        for feature_name in df_all.columns:
            ts = df_all[feature_name]
            current_row[f'{feature_name}_mean'] = np.mean(ts)
            current_row[f'{feature_name}_std'] = np.std(ts)
            current_row[f'{feature_name}_median'] = np.median(ts)
            current_row[f'{feature_name}_iqr'] = scipy_iqr(ts)
        aggregated_data_rows.append(current_row)
        
    agg_df = pd.DataFrame(aggregated_data_rows)
    numerical_cols = [f for f in agg_df.columns if f != 'label']
    f_out.write(f"  - Aggregated features created with shape: {agg_df.shape}\n")
    return agg_df, numerical_cols

# --- Modeling Core for Statistical Models ---
def get_model_pipeline(numerical_features, model):
    """Creates a full preprocessing and modeling pipeline for numerical features."""
    preprocessor = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    return Pipeline(steps=[('preprocessor', ColumnTransformer([('num', preprocessor, numerical_features)])), ('model', model)])

def train_and_evaluate_statistical_model(X_df, y_series, numerical_features, model_name, model, results_dir, f_out, suffix=""):
    """Trains and evaluates a single statistical model pipeline, returning accuracy."""
    f_out.write(f"\n--- Model: {model_name}{suffix} ---\n")
    print(f"\nTraining and evaluating: {model_name}{suffix}")
    pipeline = get_model_pipeline(numerical_features, model)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    all_y_true, all_y_pred = [], []

    for train_idx, test_idx in tqdm(cv.split(X_df, y_series), total=cv.get_n_splits(), desc=f"  CV for {model_name}{suffix}"):
        pipeline.fit(X_df.iloc[train_idx], y_series.iloc[train_idx])
        y_pred = pipeline.predict(X_df.iloc[test_idx])
        all_y_true.extend(y_series.iloc[test_idx])
        all_y_pred.extend(y_pred)

    mean_accuracy = accuracy_score(all_y_true, all_y_pred)
    f_out.write(f"Aggregated Cross-validated Accuracy: {mean_accuracy:.4f}\n")
    print(f"  Aggregated CV Accuracy: {mean_accuracy:.4f}")
    
    report = classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, labels=CLASS_LABELS, zero_division=0)
    f_out.write("Classification Report:\n" + report + "\n")
    
    cm = confusion_matrix(all_y_true, all_y_pred, labels=CLASS_LABELS)
    plot_confusion_matrix(cm, CLASS_NAMES, model_name, results_dir, f_out, suffix)
    return mean_accuracy

# --- PyTorch NN Components ---
class SaccadeSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = [torch.tensor(s, dtype=torch.float32) for s in sequences]
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.labels[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences])
    sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    return sequences_padded, torch.stack(labels), lengths

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x, lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed_input)
        return self.fc(hidden[-1])

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x, lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed_input)
        return self.fc(hidden[-1])

class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, n_filters=64, filter_size=5, dropout=0.5):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=n_filters, kernel_size=filter_size),
            nn.ReLU(),
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=filter_size),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1) # Global Max Pooling
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_filters, output_dim)
        )
    def forward(self, x, lengths): # lengths is unused but kept for API consistency
        x = x.permute(0, 2, 1) # (B, Seq, Feat) -> (B, Feat, Seq)
        features = self.conv_stack(x).squeeze(2)
        return self.classifier(features)

def train_and_evaluate_nn(sequences, labels, n_features, n_classes, f_out, model_class, model_name_suffix, model_hyperparams={}):
    """Full training and CV evaluation pipeline for a given NN model class."""
    f_out.write(f"\n--- Evaluating NN Model: {model_name_suffix} ---\n")
    print(f"\nEvaluating NN Model: {model_name_suffix}")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    accuracies = []

    for fold, (train_idx, test_idx) in enumerate(tqdm(skf.split(sequences, labels), total=5, desc=f"  CV for {model_name_suffix}")):
        train_dataset = SaccadeSequenceDataset([sequences[i] for i in train_idx], np.array(labels)[train_idx])
        test_dataset = SaccadeSequenceDataset([sequences[i] for i in test_idx], np.array(labels)[test_idx])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        model = model_class(input_dim=n_features, output_dim=n_classes, **model_hyperparams).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        for epoch in range(15): # Increased epochs slightly for better convergence
            for seq, lab, le in train_loader:
                seq, lab = seq.to(DEVICE), lab.to(DEVICE)
                optimizer.zero_grad()
                out = model(seq, le)
                loss = criterion(out, lab)
                loss.backward()
                optimizer.step()

        # Evaluation loop
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for seq, lab, le in test_loader:
                seq, lab = seq.to(DEVICE), lab.to(DEVICE)
                _, predicted = torch.max(model(seq, le).data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_true.extend(lab.cpu().numpy())
        accuracies.append(accuracy_score(all_true, all_preds))
        
    mean_accuracy = np.mean(accuracies)
    f_out.write(f"Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})\n")
    print(f"  Mean CV Accuracy: {mean_accuracy:.4f}")
    return mean_accuracy

# --- Main Experiment Function ---
def run_exhaustive_frequency_analysis(raw_items_list, n_features, n_classes, f_out):
    """Runs a comprehensive frequency analysis across multiple model types."""
    f_out.write("\n" + "="*80 + "\nExperiment 10: Exhaustive Frequency Analysis\n" + "="*80 + "\n")
    print("\n" + "="*50 + "\nRunning Exhaustive Frequency Analysis...\n" + "="*50)
    
    all_frequencies = sorted([f for f in pd.Series([item['frequency'] for item in raw_items_list]).dropna().unique()])
    
    nn_models_to_test = {
        'LSTM': {'class': LSTMClassifier, 'hyperparams': {}},
        'GRU': {'class': GRUClassifier, 'hyperparams': {}},
        '1D-CNN': {'class': CNN1DClassifier, 'hyperparams': {}}
    }
    
    results = {}

    for freq in all_frequencies:
        f_out.write(f"\n{'='*20} Analyzing Frequency: {freq} Hz {'='*20}\n")
        print(f"\n--- Analyzing Frequency: {freq} Hz ---")
        
        freq_items = [item for item in raw_items_list if item['frequency'] == freq]
        
        if len(freq_items) < 30:
            f_out.write(f"Skipping frequency {freq}Hz due to insufficient samples ({len(freq_items)}). Minimum required is 30.\n")
            print(f"Skipping frequency {freq}Hz: only {len(freq_items)} samples found.")
            continue
            
        sequences = [item['data'] for item in freq_items]
        labels = [item['label'] for item in freq_items]
        results[freq] = {}
        
        # 1. Statistical Model: LDA
        agg_df, num_feats = engineer_and_aggregate_features(freq_items, FEATURE_COLUMNS, f_out)
        if not agg_df.empty:
            results[freq]['LDA'] = train_and_evaluate_statistical_model(
                agg_df, agg_df['label'], num_feats, "LDA", LinearDiscriminantAnalysis(solver='svd'),
                RESULTS_DIR, f_out, suffix=f"_{freq}Hz"
            )
        
        # 2. Neural Network Models
        for model_name, model_info in nn_models_to_test.items():
            results[freq][model_name] = train_and_evaluate_nn(
                sequences, labels, n_features, n_classes, f_out,
                model_class=model_info['class'],
                model_name_suffix=f"{model_name}_{freq}Hz",
                model_hyperparams=model_info['hyperparams']
            )

    # --- Summarize and Plot Final Results ---
    f_out.write("\n\n" + "="*80 + "\n🏆 Final Results Summary 🏆\n" + "="*80 + "\n")
    print("\n" + "="*50 + "\nFinal Results Summary\n" + "="*50)
    
    if not results:
        f_out.write("No frequencies were analyzed. Ending report.\n")
        return
        
    results_df = pd.DataFrame.from_dict(results, orient='index').dropna(how='all')
    results_df.index.name = 'Frequency (Hz)'
    
    f_out.write("\n--- Accuracy Table ---\n")
    f_out.write(results_df.to_string(float_format="%.4f"))
    f_out.write("\n\n")
    print("\nAccuracy Table:")
    print(results_df.to_string(float_format="%.4f"))
    
    # Find the best overall combination
    if not results_df.empty:
        best_acc_per_freq = results_df.max(axis=1)
        overall_best_freq = best_acc_per_freq.idxmax()
        overall_best_model = results_df.idxmax(axis=1)[overall_best_freq]
        overall_best_acc = best_acc_per_freq.max()

        f_out.write("--- Conclusion ---\n")
        f_out.write(f"The best overall performance was achieved with the '{overall_best_model}' model "
                    f"at a frequency of {overall_best_freq} Hz, reaching an accuracy of {overall_best_acc:.4f}.\n")
        print(f"\n✅ CONCLUSION: The best performance was '{overall_best_model}' at {overall_best_freq} Hz (Accuracy: {overall_best_acc:.4f}).")

        plot_frequency_performance(results_df, RESULTS_DIR, f_out)
    else:
        f_out.write("\nNo conclusive results were generated.")
        print("\nNo conclusive results were generated.")


# --- Main Script Execution ---
if __name__ == '__main__':
    print("="*80)
    print("Starting Exhaustive Saccade Frequency Analysis (EXP_10)...")
    print(f"Using device: {DEVICE}")
    print("="*80)
    
    summary_filepath = os.path.join(RESULTS_DIR, NUMERICAL_SUMMARY_FILENAME)
    
    with open(summary_filepath, 'w', encoding='utf-8') as f_report:
        f_report.write("="*80 + "\n")
        f_report.write("Exhaustive Saccade Frequency Analysis (EXP_10) - Numerical Summary\n")
        f_report.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_report.write(f"Device used for NN models: {DEVICE}\n")
        f_report.write("="*80 + "\n")

        # Load raw data once
        raw_items_list = load_raw_sequences_and_labels(
            BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, CSV_ENCODING, CSV_SEPARATOR, MIN_SEQ_LEN_THRESHOLD, f_report
        )
        if not raw_items_list:
            f_report.write("\nCRITICAL: No data loaded. Exiting.\n")
            print("\nCRITICAL: No data loaded. Exiting.")
            exit()
            
        n_features = raw_items_list[0]['data'].shape[1]
        n_classes = len(CLASS_DEFINITIONS)
        
        # --- Run Experiment 10 ---
        run_exhaustive_frequency_analysis(raw_items_list, n_features, n_classes, f_report)

        f_report.write("\n" + "="*80 + "\nEnd of Experiment 10 Report\n" + "="*80 + "\n")

    print(f"\nNumerical summary for the experiment saved to: {summary_filepath}")
    print("\n" + "="*80)
    print("Experiment 10 Finished.")
    print(f"All plots and the summary report have been saved in: {RESULTS_DIR}")
    print("="*80)