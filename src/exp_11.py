import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import iqr as scipy_iqr
from datetime import datetime
import warnings

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.exceptions import ConvergenceWarning

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# --- Configuration ---
BASE_DIR = './data'
RESULTS_DIR = './results/EXP_11'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Class definitions (excluding TAO due to underrepresentation)
ORIGINAL_CLASS_DEFINITIONS = {
    'HC': {'path': 'Healthy control', 'label': 0},
    'MG': {'path': 'Definite MG', 'label': 1},
    'CNP3': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '3rd'), 'label': 2},
    'CNP4': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '4th'), 'label': 3},
    'CNP6': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '6th'), 'label': 4},
    'TAO': {'path': os.path.join('Non-MG diplopia (CNP, etc)', 'TAO'), 'label': 5},
}

CLASSES_TO_EXCLUDE_FOR_MODELING = ['TAO']
CLASS_DEFINITIONS = {
    name: details for name, details in ORIGINAL_CLASS_DEFINITIONS.items()
    if name not in CLASSES_TO_EXCLUDE_FOR_MODELING
}
CLASS_MAPPING = {name: details['label'] for name, details in CLASS_DEFINITIONS.items()}
INV_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}
CLASS_LABELS = sorted(INV_CLASS_MAPPING.keys())
CLASS_NAMES = [INV_CLASS_MAPPING[l] for l in CLASS_LABELS]

# Core parameters
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']
CSV_ENCODING = 'utf-16-le'
CSV_SEPARATOR = ','
MIN_SEQ_LEN_THRESHOLD = 50
EXP_PREFIX = 'EXP_11_'
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.discriminant_analysis')
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# --- Utility Functions ---
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

# --- Data Loading Functions ---
def load_raw_sequences_and_labels(base_dir, class_definitions_dict, feature_columns_expected, encoding, separator, min_seq_len_threshold, f_out):
    """Loads raw time-series data with robust Korean filename handling."""
    f_out.write("\n" + "="*80 + "\nPhase: Data Loading\n" + "="*80 + "\n")
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
            
            for csv_file_path in csv_files:
                try:
                    df_full = pd.read_csv(csv_file_path, encoding=encoding, sep=separator)
                    df_full.columns = [col.strip() for col in df_full.columns]
                    
                    # Check for required columns
                    if any(col not in df_full.columns for col in feature_columns_expected):
                        continue
                    
                    # Check minimum sequence length
                    if len(df_full) < min_seq_len_threshold:
                        continue
                    
                    df_features = df_full[feature_columns_expected].copy()
                    
                    # Convert to numeric and handle NaNs
                    for col in df_features.columns:
                        df_features.loc[:, col] = pd.to_numeric(df_features[col], errors='coerce')
                    
                    # Skip if too many NaNs
                    if df_features.isnull().sum().sum() > 0.1 * df_features.size:
                        continue
                    
                    df_features = df_features.fillna(0)
                    
                    raw_items.append({
                        'data': df_features.values.astype(np.float32),
                        'label': label,
                        'patient_id': patient_folder_name,
                        'filename': os.path.basename(csv_file_path),
                        'class_name': class_name_key,
                        'original_length': len(df_features)
                    })
                    
                except Exception as e:
                    # Silently skip problematic files
                    continue
    
    f_out.write(f"Data loading complete. Loaded {len(raw_items)} raw sequences.\n")
    print(f"Data loading complete. Loaded {len(raw_items)} raw sequences.")
    
    # Print class distribution
    class_counts = {}
    for item in raw_items:
        class_counts[item['class_name']] = class_counts.get(item['class_name'], 0) + 1
    
    f_out.write("\nClass distribution:\n")
    for class_name, count in class_counts.items():
        f_out.write(f"  {class_name}: {count} sequences\n")
        print(f"  {class_name}: {count} sequences")
    
    return raw_items

# --- Enhanced Feature Engineering ---
def engineer_comprehensive_features(raw_items_dicts, original_feature_names, f_out):
    """Creates comprehensive aggregated features with advanced statistical measures."""
    f_out.write("\n" + "="*80 + "\nPhase: Comprehensive Feature Engineering\n" + "="*80 + "\n")
    print("\n" + "="*50 + "\nStarting Comprehensive Feature Engineering...\n" + "="*50)
    
    if not raw_items_dicts:
        return pd.DataFrame(), []
    
    aggregated_data_rows = []
    
    for item in tqdm(raw_items_dicts, desc="  Engineering comprehensive features"):
        df_original = pd.DataFrame(item['data'], columns=original_feature_names)
        
        # Engineer velocity features
        velocity_features = []
        for pos_col in ['LH', 'RH', 'LV', 'RV']:
            velocity_series = df_original[pos_col].diff().fillna(0)
            velocity_features.append(velocity_series.rename(f'{pos_col}_Vel'))
        
        # Engineer acceleration features
        acceleration_features = []
        for pos_col in ['LH', 'RH', 'LV', 'RV']:
            velocity_series = df_original[pos_col].diff().fillna(0)
            acceleration_series = velocity_series.diff().fillna(0)
            acceleration_features.append(acceleration_series.rename(f'{pos_col}_Acc'))
        
        # Engineer error features
        error_features = []
        error_definitions = [
            ('LH', 'TargetH', 'ErrorH_L'), ('RH', 'TargetH', 'ErrorH_R'),
            ('LV', 'TargetV', 'ErrorV_L'), ('RV', 'TargetV', 'ErrorV_R')
        ]
        for eye_col, target_col, error_col_name in error_definitions:
            error_series = df_original[eye_col] - df_original[target_col]
            error_features.append(error_series.rename(error_col_name))
        
        # Combine all features
        all_features = [df_original] + velocity_features + acceleration_features + error_features
        df_all_features = pd.concat(all_features, axis=1)
        
        # Create comprehensive statistical aggregations
        current_row = {
            'patient_id': item['patient_id'],
            'filename': item['filename'],
            'class_name': item['class_name'],
            'label': item['label'],
            'original_length': item['original_length']
        }
        
        for feature_name in df_all_features.columns:
            ft_ts = df_all_features[feature_name]
            
            # Basic statistics
            current_row[f'{feature_name}_mean'] = np.mean(ft_ts)
            current_row[f'{feature_name}_std'] = np.std(ft_ts)
            current_row[f'{feature_name}_median'] = np.median(ft_ts)
            current_row[f'{feature_name}_iqr'] = scipy_iqr(ft_ts)
            current_row[f'{feature_name}_min'] = np.min(ft_ts)
            current_row[f'{feature_name}_max'] = np.max(ft_ts)
            current_row[f'{feature_name}_range'] = np.max(ft_ts) - np.min(ft_ts)
            
            # Advanced statistics
            current_row[f'{feature_name}_skew'] = pd.Series(ft_ts).skew()
            current_row[f'{feature_name}_kurtosis'] = pd.Series(ft_ts).kurtosis()
            current_row[f'{feature_name}_q25'] = np.percentile(ft_ts, 25)
            current_row[f'{feature_name}_q75'] = np.percentile(ft_ts, 75)
            current_row[f'{feature_name}_q90'] = np.percentile(ft_ts, 90)
            current_row[f'{feature_name}_q10'] = np.percentile(ft_ts, 10)
            
            # Variability measures
            current_row[f'{feature_name}_cv'] = np.std(ft_ts) / (np.abs(np.mean(ft_ts)) + 1e-8)  # Coefficient of variation
            current_row[f'{feature_name}_mad'] = np.median(np.abs(ft_ts - np.median(ft_ts)))  # Median absolute deviation
            
            # Temporal features (first half vs second half)
            midpoint = len(ft_ts) // 2
            first_half = ft_ts[:midpoint]
            second_half = ft_ts[midpoint:]
            
            current_row[f'{feature_name}_first_half_mean'] = np.mean(first_half)
            current_row[f'{feature_name}_second_half_mean'] = np.mean(second_half)
            current_row[f'{feature_name}_temporal_change'] = np.mean(second_half) - np.mean(first_half)
            
        # Cross-feature relationships
        # Eye coordination features
        current_row['LH_RH_correlation'] = np.corrcoef(df_all_features['LH'], df_all_features['RH'])[0, 1]
        current_row['LV_RV_correlation'] = np.corrcoef(df_all_features['LV'], df_all_features['RV'])[0, 1]
        
        # Velocity coordination
        current_row['LH_RH_vel_correlation'] = np.corrcoef(df_all_features['LH_Vel'], df_all_features['RH_Vel'])[0, 1]
        current_row['LV_RV_vel_correlation'] = np.corrcoef(df_all_features['LV_Vel'], df_all_features['RV_Vel'])[0, 1]
        
        # Error asymmetry
        current_row['horizontal_error_asymmetry'] = np.abs(np.mean(df_all_features['ErrorH_L']) - np.mean(df_all_features['ErrorH_R']))
        current_row['vertical_error_asymmetry'] = np.abs(np.mean(df_all_features['ErrorV_L']) - np.mean(df_all_features['ErrorV_R']))
        
        aggregated_data_rows.append(current_row)
    
    agg_df = pd.DataFrame(aggregated_data_rows)
    
    # Get numerical feature columns (excluding metadata)
    numerical_feature_cols = [col for col in agg_df.columns 
                             if col not in ['patient_id', 'filename', 'class_name', 'label', 'original_length']]
    
    f_out.write(f"Comprehensive features DataFrame created with shape: {agg_df.shape}\n")
    f_out.write(f"Total numerical features created: {len(numerical_feature_cols)}\n")
    print(f"Created {len(numerical_feature_cols)} comprehensive features")
    
    return agg_df, numerical_feature_cols

# --- Advanced Modeling Pipeline ---
def create_advanced_pipeline(numerical_features, model, feature_selection_k=None, scaler_type='standard'):
    """Creates an advanced preprocessing and modeling pipeline."""
    steps = []
    
    # Imputation
    steps.append(('imputer', SimpleImputer(strategy='median')))
    
    # Scaling
    if scaler_type == 'standard':
        steps.append(('scaler', StandardScaler()))
    elif scaler_type == 'robust':
        steps.append(('scaler', RobustScaler()))
    
    # Feature selection
    if feature_selection_k and feature_selection_k < len(numerical_features):
        steps.append(('feature_selection', SelectKBest(score_func=f_classif, k=feature_selection_k)))
    
    preprocessor = Pipeline(steps=steps)
    
    return Pipeline(steps=[
        ('preprocessor', ColumnTransformer([('num', preprocessor, numerical_features)])),
        ('model', model)
    ])

def train_and_evaluate_model(X_df, y_series, numerical_features, model_name, model, results_dir, f_out, 
                           feature_selection_k=None, scaler_type='standard', suffix=""):
    """Trains and evaluates a single model with advanced preprocessing."""
    f_out.write(f"\n--- Model: {model_name}{suffix} ---\n")
    print(f"\nTraining and evaluating: {model_name}{suffix}")
    
    pipeline = create_advanced_pipeline(numerical_features, model, feature_selection_k, scaler_type)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    all_y_true, all_y_pred = [], []
    fold_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(tqdm(cv.split(X_df, y_series), total=cv.get_n_splits(), desc=f"  CV for {model_name}{suffix}")):
        X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
        y_train, y_test = y_series.iloc[train_idx], y_series.iloc[test_idx]
        
        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            fold_acc = accuracy_score(y_test, y_pred)
            fold_accuracies.append(fold_acc)
            
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            
        except Exception as e:
            f_out.write(f"  Error in fold {fold}: {e}\n")
            continue

    if not all_y_pred:
        f_out.write("  Model training failed for all folds. Skipping evaluation.\n")
        return 0.0

    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    f_out.write(f"Cross-validated Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})\n")
    print(f"  Cross-validated Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
    
    # Classification report
    report = classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, labels=CLASS_LABELS, zero_division=0)
    f_out.write("Classification Report:\n" + report + "\n")
    
    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred, labels=CLASS_LABELS)
    f_out.write("Confusion Matrix:\n" + np.array2string(cm) + "\n")
    plot_confusion_matrix(cm, CLASS_NAMES, model_name, results_dir, f_out, suffix)
    
    return mean_accuracy

# --- Enhanced Neural Network ---
class AdvancedSaccadeClassifier(nn.Module):
    """Memory-efficient neural network for 24GB VRAM limit."""
    
    def __init__(self, input_dim, output_dim, hidden_dim=32, n_layers=1, dropout=0.3):
        super().__init__()
        
        # Single LSTM layer (very small)
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, 
                           batch_first=True, dropout=0.0, bidirectional=False)
        
        # Simple classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x, lengths):
        # Simple LSTM forward pass - use only last output
        lstm_output, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state for each sequence
        batch_indices = torch.arange(x.size(0), device=x.device)
        last_indices = (lengths - 1).clamp(min=0).to(x.device)
        last_outputs = lstm_output[batch_indices, last_indices]
        
        # Classification
        output = self.classifier(last_outputs)
        return output

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

def train_and_evaluate_advanced_nn(sequences, labels, n_features, n_classes, f_out, model_name_suffix="Advanced_NN"):
    """Train and evaluate the advanced neural network."""
    f_out.write(f"\n--- Evaluating Advanced NN Model: {model_name_suffix} ---\n")
    print(f"\nEvaluating Advanced NN Model: {model_name_suffix}")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    accuracies = []

    for fold, (train_idx, test_idx) in enumerate(tqdm(skf.split(sequences, labels), total=5, desc=f"  CV for {model_name_suffix}")):
        X_train_seq = [sequences[i] for i in train_idx]
        X_test_seq = [sequences[i] for i in test_idx]
        y_train = np.array(labels)[train_idx]
        y_test = np.array(labels)[test_idx]
        
        train_dataset = SaccadeSequenceDataset(X_train_seq, y_train)
        test_dataset = SaccadeSequenceDataset(X_test_seq, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        model = AdvancedSaccadeClassifier(input_dim=n_features, output_dim=n_classes).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        for epoch in range(20):
            epoch_loss = 0
            for seq_padded, labels_batch, lengths in train_loader:
                seq_padded, labels_batch = seq_padded.to(DEVICE), labels_batch.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(seq_padded, lengths)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step(epoch_loss)

        # Evaluation
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for seq_padded, labels_batch, lengths in test_loader:
                seq_padded, labels_batch = seq_padded.to(DEVICE), labels_batch.to(DEVICE)
                outputs = model(seq_padded, lengths)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_true.extend(labels_batch.cpu().numpy())
        
        fold_accuracy = accuracy_score(all_true, all_preds)
        accuracies.append(fold_accuracy)
        
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    f_out.write(f"Mean CV Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})\n")
    print(f"  Mean CV Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
    return mean_accuracy

# --- Main Experiment Function ---
def run_experiment_11(raw_items_list, f_out):
    """Run comprehensive experiment 11 with multiple approaches."""
    f_out.write("\n" + "="*80 + "\nExperiment 11: Advanced Multi-Modal Approach\n" + "="*80 + "\n")
    print("\n" + "="*50 + "\nRunning Experiment 11: Advanced Multi-Modal Approach\n" + "="*50)
    
    # 1. Create comprehensive features
    agg_df, numerical_features = engineer_comprehensive_features(raw_items_list, FEATURE_COLUMNS, f_out)
    
    if agg_df.empty:
        f_out.write("ERROR: No aggregated features created. Exiting.\n")
        return
    
    X = agg_df[numerical_features]
    y = agg_df['label']
    
    f_out.write(f"\nDataset shape: {X.shape}\n")
    f_out.write(f"Number of features: {len(numerical_features)}\n")
    
    results = {}
    
    # 2. Traditional statistical models with feature selection
    f_out.write("\n" + "="*60 + "\nPhase 1: Statistical Models with Feature Selection\n" + "="*60 + "\n")
    
    # LDA with different feature selection strategies
    for k in [50, 100, 150, 200]:
        if k < len(numerical_features):
            model_name = f"LDA_SelectK{k}"
            acc = train_and_evaluate_model(X, y, numerical_features, model_name, 
                                         LinearDiscriminantAnalysis(solver='svd'), 
                                         RESULTS_DIR, f_out, feature_selection_k=k, suffix=f"_k{k}")
            results[model_name] = acc
    
    # 3. Ensemble methods
    f_out.write("\n" + "="*60 + "\nPhase 2: Ensemble Methods\n" + "="*60 + "\n")
    
    # Random Forest with different configurations
    rf_configs = [
        {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5},
        {'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 3},
        {'n_estimators': 500, 'max_depth': None, 'min_samples_split': 2}
    ]
    
    for i, config in enumerate(rf_configs):
        model_name = f"RandomForest_Config{i+1}"
        rf_model = RandomForestClassifier(random_state=RANDOM_STATE, **config)
        acc = train_and_evaluate_model(X, y, numerical_features, model_name, rf_model, 
                                     RESULTS_DIR, f_out, feature_selection_k=100, suffix=f"_config{i+1}")
        results[model_name] = acc
    
    # Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=RANDOM_STATE)
    acc = train_and_evaluate_model(X, y, numerical_features, "GradientBoosting", gb_model, 
                                 RESULTS_DIR, f_out, feature_selection_k=100)
    results["GradientBoosting"] = acc
    
    # 4. SVM with different kernels
    f_out.write("\n" + "="*60 + "\nPhase 3: Support Vector Machines\n" + "="*60 + "\n")
    
    svm_configs = [
        {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale'},
        {'kernel': 'poly', 'degree': 3, 'C': 1.0}
    ]
    
    for i, config in enumerate(svm_configs):
        model_name = f"SVM_{config['kernel']}_C{config['C']}"
        svm_model = SVC(random_state=RANDOM_STATE, **config)
        acc = train_and_evaluate_model(X, y, numerical_features, model_name, svm_model, 
                                     RESULTS_DIR, f_out, feature_selection_k=50, scaler_type='robust', 
                                     suffix=f"_config{i+1}")
        results[model_name] = acc
    
    # 5. Advanced Neural Network
    f_out.write("\n" + "="*60 + "\nPhase 4: Advanced Neural Network\n" + "="*60 + "\n")
    
    sequences = [item['data'] for item in raw_items_list]
    labels = [item['label'] for item in raw_items_list]
    n_features = sequences[0].shape[1]
    n_classes = len(CLASS_DEFINITIONS)
    
    nn_acc = train_and_evaluate_advanced_nn(sequences, labels, n_features, n_classes, f_out)
    results["Advanced_NN"] = nn_acc
    
    # 6. Results Summary
    f_out.write("\n" + "="*80 + "\nFinal Results Summary\n" + "="*80 + "\n")
    print("\n" + "="*50 + "\nFinal Results Summary\n" + "="*50)
    
    # Sort results by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    f_out.write("Model Performance Ranking:\n")
    for i, (model_name, accuracy) in enumerate(sorted_results, 1):
        f_out.write(f"{i:2d}. {model_name:<25}: {accuracy:.4f}\n")
        print(f"{i:2d}. {model_name:<25}: {accuracy:.4f}")
    
    if sorted_results:
        best_model, best_accuracy = sorted_results[0]
        f_out.write(f"\nBest performing model: {best_model} with accuracy: {best_accuracy:.4f}\n")
        print(f"\nBest performing model: {best_model} with accuracy: {best_accuracy:.4f}")
        
        if best_accuracy > 0.45:
            f_out.write("🎉 SUCCESS: Achieved accuracy above 45% baseline!\n")
            print("🎉 SUCCESS: Achieved accuracy above 45% baseline!")
        else:
            f_out.write("⚠️  Did not exceed 45% baseline. Consider further iterations.\n")
            print("⚠️  Did not exceed 45% baseline. Consider further iterations.")
    
    return results

# --- Main Script Execution ---
if __name__ == '__main__':
    print("="*80)
    print("Starting Experiment 11: Advanced Multi-Modal Myasthenia Gravis Detection")
    print(f"Using device: {DEVICE}")
    print("="*80)
    
    summary_filepath = os.path.join(RESULTS_DIR, f'{EXP_PREFIX}summary.txt')
    
    with open(summary_filepath, 'w', encoding='utf-8') as f_report:
        f_report.write("="*80 + "\n")
        f_report.write("Experiment 11: Advanced Multi-Modal Approach - Summary Report\n")
        f_report.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_report.write(f"Device used for NN models: {DEVICE}\n")
        f_report.write(f"Target classes: {CLASS_NAMES}\n")
        f_report.write(f"Baseline to beat: 45% accuracy\n")
        f_report.write("="*80 + "\n")

        # Load raw data
        raw_items_list = load_raw_sequences_and_labels(
            BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, CSV_ENCODING, CSV_SEPARATOR, MIN_SEQ_LEN_THRESHOLD, f_report
        )
        
        if not raw_items_list:
            f_report.write("\nCRITICAL: No data loaded. Exiting.\n")
            print("\nCRITICAL: No data loaded. Exiting.")
            exit()
        
        # Run the comprehensive experiment
        results = run_experiment_11(raw_items_list, f_report)
        
        f_report.write("\n" + "="*80 + "\nEnd of Experiment 11 Report\n" + "="*80 + "\n")

    print(f"\nExperiment 11 complete! Summary saved to: {summary_filepath}")
    print(f"All results and plots saved in: {RESULTS_DIR}")
    print("="*80)
