import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import warnings
import gc

# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# ========================================================================================
# --- Configuration for Experiment 05 ---
# ========================================================================================
# --- Basic Setup ---
BASE_DIR = './data'
EXP_NAME = 'EXP_05_Saccade_Segmentation' # New experiment name for the new strategy
RESULTS_DIR = os.path.join('./results', EXP_NAME)
LOG_FILENAME = f'{EXP_NAME}_results_log.txt'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42

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

# --- NEW: Saccade Event Segmentation Parameters ---
PRE_ONSET_WINDOW = 50   # Number of time steps to include before the target change
POST_ONSET_WINDOW = 450 # Number of time steps to include after the target change
SEGMENT_LENGTH = PRE_ONSET_WINDOW + POST_ONSET_WINDOW # Total length of each saccade segment

# --- Model & Training Hyperparameters ---
N_FOLDS = 5
EPOCHS = 100
BATCH_SIZE = 64 # Can be increased due to smaller sequence lengths
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 128
N_LAYERS = 2  # For the GRU part
DROPOUT_PROB = 0.5
EARLY_STOPPING_PATIENCE = 15
LABEL_SMOOTHING = 0.1 # Retained as a good regularization practice

# ========================================================================================
# --- Utility and Setup Functions (Largely Unchanged) ---
# ========================================================================================
def create_results_directory(dir_path=RESULTS_DIR):
    """Creates the directory to store experiment results."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"INFO: Created directory for results: {dir_path}")

def get_file_handler(filepath):
    """Opens a file for logging."""
    return open(filepath, 'w', encoding='utf-8')

def log_message(f_handler, message, print_to_console=True):
    """Logs a message to the file and optionally prints to console."""
    f_handler.write(message + '\n')
    f_handler.flush()
    if print_to_console:
        print(message)

# ========================================================================================
# --- Data Loading and Processing Pipeline ---
# ========================================================================================
def load_raw_sequences_and_labels(base_dir, class_definitions_dict, feature_columns_expected, encoding, separator, min_seq_len_threshold, f_out):
    """Loads raw time-series data from CSV files for each patient and class."""
    log_message(f_out, "="*70 + "\nPhase: 1. Raw Data Loading\n" + "="*70, print_to_console=False)
    print("\n" + "="*50 + "\n1. Starting Raw Data Loading...\n" + "="*50)
    raw_items = []
    for class_name_key, class_details in class_definitions_dict.items():
        label, class_dir_abs = class_details['label'], os.path.join(base_dir, class_details['path'])
        log_message(f_out, f"\nProcessing Class: '{class_name_key}' (Label: {label}) from path: {class_dir_abs}", True)
        if not os.path.isdir(class_dir_abs):
            log_message(f_out, f"WARNING: Class directory not found: {class_dir_abs}", True); continue
        
        patient_dirs = [d for d in os.listdir(class_dir_abs) if os.path.isdir(os.path.join(class_dir_abs, d))]
        if not patient_dirs:
            log_message(f_out, f"INFO: No patient directories found in {class_dir_abs}", True); continue
            
        for patient_folder_name in tqdm(patient_dirs, desc=f"  Patients in {class_name_key}", ncols=100):
            patient_id = f"{class_name_key}_{patient_folder_name}"
            patient_dir_path = os.path.join(class_dir_abs, patient_folder_name)
            for csv_file_path in glob.glob(os.path.join(patient_dir_path, '*.csv')):
                try:
                    df_full = pd.read_csv(csv_file_path, encoding=encoding, sep=separator, low_memory=False)
                    df_full.columns = [col.strip() for col in df_full.columns]
                    if not all(col in df_full.columns for col in feature_columns_expected): continue
                    
                    df_features = df_full[feature_columns_expected].copy()
                    if df_features.empty or len(df_features) < min_seq_len_threshold: continue
                    
                    for col in df_features.columns:
                        df_features.loc[:, col] = pd.to_numeric(df_features[col], errors='coerce')
                    
                    # Impute NaNs before processing
                    if df_features.isnull().values.any():
                        df_features.fillna(method='ffill', inplace=True)
                        df_features.fillna(method='bfill', inplace=True)
                        df_features.fillna(0, inplace=True) # Final fallback

                    sequence_data = df_features.values.astype(np.float32)
                    raw_items.append({'data': sequence_data, 'label': label, 'patient_id': patient_id, 'class_name': class_name_key})
                except Exception as e:
                    print(f"ERROR processing {os.path.basename(csv_file_path)}: {e}. Skipping.")
    log_message(f_out, f"\nRaw data loading complete. Loaded {len(raw_items)} raw sequences.\n" + "-"*70 + "\n", True)
    return raw_items

def engineer_features_from_raw_data(raw_items_dicts, original_feature_names, f_out):
    """Engineers velocity and error features from the raw position data."""
    log_message(f_out, "="*70 + "\nPhase: 2. Feature Engineering\n" + "="*70, print_to_console=False)
    print("\n" + "="*50 + "\n2. Starting Feature Engineering...\n" + "="*50)
    engineered_items, final_feature_names = [], []
    for item in tqdm(raw_items_dicts, desc="  Engineering Features", ncols=100):
        df_original = pd.DataFrame(item['data'], columns=original_feature_names)
        df_engineered_parts = []
        
        # Velocity features
        for pos_col in ['LH', 'RH', 'LV', 'RV']:
            df_engineered_parts.append(df_original[pos_col].diff().fillna(0).rename(f'{pos_col}_Vel'))
        
        # Error features
        for eye_col, target_col, error_col_name in [('LH', 'TargetH', 'ErrorH_L'), ('RH', 'TargetH', 'ErrorH_R'), ('LV', 'TargetV', 'ErrorV_L'), ('RV', 'TargetV', 'ErrorV_R')]:
            df_engineered_parts.append((df_original[eye_col] - df_original[target_col]).rename(error_col_name))
            
        df_all_features = pd.concat([df_original, pd.concat(df_engineered_parts, axis=1)], axis=1)
        if not final_feature_names: final_feature_names = df_all_features.columns.tolist()
        
        engineered_items.append({**item, 'data': df_all_features.values.astype(np.float32)})
    
    final_num_channels = engineered_items[0]['data'].shape[1] if engineered_items else 0
    log_message(f_out, f"Feature engineering complete. Total features per time step: {final_num_channels}", True)
    log_message(f_out, f"Feature names: {final_feature_names}\n" + "-"*70 + "\n", True)
    return engineered_items, final_num_channels, final_feature_names

# ========================================================================================
# --- NEW: Saccade Event Segmentation Logic ---
# ========================================================================================
def segment_saccade_events(engineered_items, target_feature_names, pre_window, post_window, f_out):
    """
    Segments long sequences into shorter, meaningful events based on target changes.
    This is the core of the new data representation strategy.
    """
    log_message(f_out, "="*70 + "\nPhase: 3. Saccade Event Segmentation\n" + "="*70, print_to_console=False)
    print("\n" + "="*50 + "\n3. Segmenting into Saccade Events...\n" + "="*50)
    
    segmented_items = []
    total_segments = 0
    
    # Get indices of TargetH and TargetV to monitor for changes
    target_h_idx = target_feature_names.index('TargetH')
    target_v_idx = target_feature_names.index('TargetV')
    
    for item in tqdm(engineered_items, desc="  Segmenting Trials", ncols=100):
        sequence = item['data']
        # Detect changes in either horizontal or vertical target position
        target_changes_h = np.diff(sequence[:, target_h_idx], prepend=sequence[0, target_h_idx]) != 0
        target_changes_v = np.diff(sequence[:, target_v_idx], prepend=sequence[0, target_v_idx]) != 0
        event_indices = np.where(target_changes_h | target_changes_v)[0]
        
        segments_from_trial = 0
        for idx in event_indices:
            # Ensure the window is within the sequence bounds
            start, end = idx - pre_window, idx + post_window
            if start >= 0 and end <= len(sequence):
                segment_data = sequence[start:end, :]
                segmented_items.append({
                    'data': segment_data,
                    'label': item['label'],
                    'patient_id': item['patient_id'],
                    'class_name': item['class_name']
                })
                segments_from_trial += 1
        
        if segments_from_trial > 0:
            total_segments += segments_from_trial
    
    log_message(f_out, f"Segmentation complete.", True)
    log_message(f_out, f"  - Original number of sequences: {len(engineered_items)}", True)
    log_message(f_out, f"  - New number of sequences (segments): {total_segments}", True)
    log_message(f_out, f"  - Segment Length: {SEGMENT_LENGTH} ({PRE_ONSET_WINDOW} pre + {POST_ONSET_WINDOW} post)\n" + "-"*70 + "\n", True)
    return segmented_items

# ========================================================================================
# --- Core ML Components (Dataset, Model, Training) ---
# ========================================================================================
class SaccadeStandardScaler:
    """A scaler that fits on a list of items and transforms sequences individually."""
    def __init__(self): self.scaler = StandardScaler()
    def fit(self, data_items): self.scaler.fit(np.vstack([item['data'] for item in data_items]))
    def transform(self, sequence): return self.scaler.transform(sequence)

class SaccadeDataset(Dataset):
    """PyTorch Dataset for saccade segments."""
    def __init__(self, items, label_map, scaler=None):
        self.items = items
        self.label_map = label_map
        self.scaler = scaler
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        item = self.items[idx]
        sequence = item['data']
        label = item['label']
        if self.scaler:
            sequence = self.scaler.transform(sequence)
        # No padding needed as all segments have the same length
        return torch.from_numpy(sequence).float(), torch.tensor(self.label_map[label], dtype=torch.long)

class Attention(nn.Module):
    """Attention mechanism to weigh GRU outputs."""
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        eij = torch.mm(x.contiguous().view(-1, self.feature_dim), self.weight).view(-1, self.step_dim)
        if self.bias: eij = eij + self.b
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        if mask is not None: a = a * mask
        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)
        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class SaccadeConvGRU(nn.Module):
    """A Conv-GRU hybrid model, adapted for fixed-length saccade segments."""
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, segment_len, dropout_prob):
        super(SaccadeConvGRU, self).__init__()
        # 1D CNN layers to extract local temporal patterns
        self.conv_extractor = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=128, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout_prob)
        )
        # Calculate the sequence length after convolutions and pooling
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_dim, segment_len)
            conv_output_seq_len = self.conv_extractor(dummy_input).shape[2]
        
        # Bidirectional GRU to capture sequential context
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=n_layers,
                          batch_first=True, dropout=dropout_prob if n_layers > 1 else 0, bidirectional=True)
                          
        # Attention layer to focus on the most important time steps
        self.attention = Attention(feature_dim=hidden_dim * 2, step_dim=conv_output_seq_len)
        
        # Fully connected classifier head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1) # -> (batch_size, input_dim, seq_len) for Conv1D
        conv_out = self.conv_extractor(x)
        conv_out = conv_out.permute(0, 2, 1) # -> (batch_size, new_seq_len, hidden_dim) for GRU
        gru_out, _ = self.gru(conv_out)
        context_vector = self.attention(gru_out)
        return self.fc(context_vector)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience, self.min_delta, self.counter = patience, min_delta, 0
        self.min_validation_loss = np.inf
    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    running_loss, correct_predictions, total_samples = 0.0, 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += labels.size(0)
    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_predictions.double() / total_samples).item()
    return epoch_loss, epoch_acc

def evaluate_epoch(model, dataloader, criterion, device):
    """Evaluates the model for one epoch."""
    model.eval()
    running_loss, correct_predictions, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / total_samples, (correct_predictions.double() / total_samples).item(), all_preds, all_labels

# ========================================================================================
# --- Plotting and Main Execution ---
# ========================================================================================
def plot_loss_curves(train_losses, val_losses, fold, filepath):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Fold {fold+1} - Training & Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True); plt.savefig(filepath); plt.close()

def plot_confusion_matrix(cm, class_names, title, filepath):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title); plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout(); plt.savefig(filepath); plt.close()
    
def calculate_class_weights(items, label_map):
    labels = [label_map[item['label']] for item in items]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float)
    
if __name__ == '__main__':
    # 1. Setup
    np.random.seed(RANDOM_STATE); torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available(): torch.cuda.manual_seed(RANDOM_STATE)
    create_results_directory()
    f_log = get_file_handler(os.path.join(RESULTS_DIR, LOG_FILENAME))

    log_message(f_log, f"Experiment: {EXP_NAME}\nStrategy: Segmenting long sequences into short, 500-step saccade events based on target changes.")
    log_message(f_log, f"\n--- Configuration ---\nDevice: {DEVICE}\nEpochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}\nModel: Conv-Bi-GRU+Attention on {SEGMENT_LENGTH}-step segments\n" + "-"*70)

    # 2. Data Loading -> Feature Engineering -> Segmentation
    raw_items = load_raw_sequences_and_labels(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, CSV_ENCODING, CSV_SEPARATOR, MIN_SEQ_LEN_THRESHOLD, f_log)
    engineered_items, num_features, feature_names = engineer_features_from_raw_data(raw_items, FEATURE_COLUMNS, f_log)
    segmented_items = segment_saccade_events(engineered_items, feature_names, PRE_ONSET_WINDOW, POST_ONSET_WINDOW, f_log)
    
    # Filter out excluded class and create final label mapping
    filtered_items = [item for item in segmented_items if item['class_name'] != CLASS_TO_EXCLUDE]
    remaining_class_names = sorted([c for c in CLASS_DEFINITIONS if c != CLASS_TO_EXCLUDE], key=lambda c: CLASS_DEFINITIONS[c]['label'])
    final_class_map = {CLASS_DEFINITIONS[name]['label']: i for i, name in enumerate(remaining_class_names)}
    num_classes = len(remaining_class_names)
    log_message(f_log, f"Filtered out '{CLASS_TO_EXCLUDE}'. {len(filtered_items)} segments remaining for training/validation.")
    log_message(f_log, f"Final class mapping: { {name: i for i, name in enumerate(remaining_class_names)} }\n" + "-"*70)

    X, y, groups = np.array(filtered_items, dtype=object), np.array([item['label'] for item in filtered_items]), np.array([item['patient_id'] for item in filtered_items])
    
    # 3. Cross-Validation Loop
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    overall_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []}
    aggregated_cm = np.zeros((num_classes, num_classes))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y, groups)):
        log_message(f_log, "\n" + "="*70 + f"\nStarting Fold {fold+1}/{N_FOLDS}\n" + "="*70)
        train_items, val_items = X[train_idx].tolist(), X[val_idx].tolist()

        scaler = SaccadeStandardScaler(); scaler.fit(train_items)
        
        train_dataset = SaccadeDataset(train_items, final_class_map, scaler)
        val_dataset = SaccadeDataset(val_items, final_class_map, scaler)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        log_message(f_log, f"Train segments: {len(train_dataset)}, Validation segments: {len(val_dataset)}.")

        class_weights = calculate_class_weights(train_items, final_class_map).to(DEVICE)
        log_message(f_log, f"Fold {fold+1} Class Weights: {class_weights.cpu().numpy()}")
        
        model = SaccadeConvGRU(input_dim=num_features, hidden_dim=HIDDEN_DIM, output_dim=num_classes,
                               n_layers=N_LAYERS, segment_len=SEGMENT_LENGTH, dropout_prob=DROPOUT_PROB).to(DEVICE)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5 )
        early_stopper = EarlyStopper(patience=EARLY_STOPPING_PATIENCE, min_delta=0.001)
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc, _, _ = evaluate_epoch(model, val_loader, criterion, DEVICE)
            history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
            log_message(f_log, f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}", True)
            scheduler.step(val_loss)
            if early_stopper.early_stop(val_loss):
                log_message(f_log, f"Early stopping triggered at epoch {epoch+1}"); break
        
        # Final evaluation for the fold
        _, _, y_pred, y_true = evaluate_epoch(model, val_loader, criterion, DEVICE)
        log_message(f_log, f"\n--- Fold {fold+1} Results ---")
        report_text = classification_report(y_true, y_pred, target_names=remaining_class_names, zero_division=0)
        log_message(f_log, report_text)
        report_dict = classification_report(y_true, y_pred, target_names=remaining_class_names, output_dict=True, zero_division=0)
        overall_metrics['accuracy'].append(report_dict['accuracy'])
        overall_metrics['precision'].append(report_dict['macro avg']['precision'])
        overall_metrics['recall'].append(report_dict['macro avg']['recall'])
        overall_metrics['f1-score'].append(report_dict['macro avg']['f1-score'])

        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes)); aggregated_cm += cm
        plot_confusion_matrix(cm, remaining_class_names, f'Fold {fold+1} - Confusion Matrix', os.path.join(RESULTS_DIR, f'{EXP_NAME}_fold_{fold+1}_confusion_matrix.png'))
        plot_loss_curves(history['train_loss'], history['val_loss'], fold, os.path.join(RESULTS_DIR, f'{EXP_NAME}_fold_{fold+1}_loss_curve.png'))
        log_message(f_log, f"Plots for fold {fold+1} saved.")

        del model, optimizer, scheduler, train_dataset, val_dataset, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # 4. Final Aggregated Results
    log_message(f_log, "\n" + "="*70 + "\nCross-Validation Summary\n" + "="*70)
    log_message(f_log, f"Average Accuracy: {np.mean(overall_metrics['accuracy']):.4f} ± {np.std(overall_metrics['accuracy']):.4f}")
    log_message(f_log, f"Average Macro F1-Score: {np.mean(overall_metrics['f1-score']):.4f} ± {np.std(overall_metrics['f1-score']):.4f}")
    log_message(f_log, f"Average Macro Precision: {np.mean(overall_metrics['precision']):.4f} ± {np.std(overall_metrics['precision']):.4f}")
    log_message(f_log, f"Average Macro Recall: {np.mean(overall_metrics['recall']):.4f} ± {np.std(overall_metrics['recall']):.4f}")
    
    plot_confusion_matrix(aggregated_cm, remaining_class_names, 'Aggregated Confusion Matrix (All Folds)', os.path.join(RESULTS_DIR, f'{EXP_NAME}_aggregated_confusion_matrix.png'))
    log_message(f_log, f"\nAggregated Confusion Matrix (sum over folds):\n{np.array2string(aggregated_cm.astype(int), separator=', ')}")
    log_message(f_log, f"Aggregated confusion matrix plot saved.")
    log_message(f_log, "\n" + "="*70 + "\nExperiment Finished.\n" + "="*70)
    f_log.close()
    
    print(f"\nExperiment complete. All results saved in: {RESULTS_DIR}")


