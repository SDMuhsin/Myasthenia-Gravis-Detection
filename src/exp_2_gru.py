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
import warnings

# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# --- Configuration ---
# --- Basic Setup ---
BASE_DIR = './data'
EXP_NAME = 'EXP_2_IMPROVED' # New experiment name
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

# --- Model & Training Hyperparameters (Revised for better performance) ---
N_FOLDS = 5
EPOCHS = 100 # Increased epochs, but with early stopping
BATCH_SIZE = 64 # Larger batch size for GPU
LEARNING_RATE = 0.0005 # Lower initial learning rate
WEIGHT_DECAY = 1e-4 # Added L2 regularization
HIDDEN_DIM = 256 # Increased model capacity
N_LAYERS = 3 # Deeper model
DROPOUT_PROB = 0.5 # Stronger regularization
EARLY_STOPPING_PATIENCE = 10 # Stop if no improvement for 10 epochs

# --- Utility Functions ---
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

# --- Data Loading (Adapted from EDA, no changes here) ---
def load_raw_sequences_and_labels(base_dir, class_definitions_dict, feature_columns_expected, encoding, separator, min_seq_len_threshold, f_out):
    log_message(f_out, "="*70 + "\nPhase: Data Loading\n" + "="*70, print_to_console=False)
    print("="*50)
    print("Starting Data Loading...")
    print("="*50)
    raw_items = []

    for class_name_key, class_details in class_definitions_dict.items():
        label = class_details['label']
        class_dir_abs = os.path.join(base_dir, class_details['path'])
        log_message(f_out, f"\nProcessing Class: '{class_name_key}' (Label: {label}) from path: {class_dir_abs}", True)

        if not os.path.isdir(class_dir_abs):
            log_message(f_out, f"WARNING: Class directory not found: {class_dir_abs}", True)
            continue

        patient_dirs = [d for d in os.listdir(class_dir_abs) if os.path.isdir(os.path.join(class_dir_abs, d))]
        if not patient_dirs:
            log_message(f_out, f"INFO: No patient directories found in {class_dir_abs}", True)
            continue

        for patient_folder_name in tqdm(patient_dirs, desc=f"  Patients in {class_name_key}"):
            patient_id = f"{class_name_key}_{patient_folder_name}"
            patient_dir_path = os.path.join(class_dir_abs, patient_folder_name)
            csv_files = glob.glob(os.path.join(patient_dir_path, '*.csv'))

            for csv_file_path in csv_files:
                try:
                    df_full = pd.read_csv(csv_file_path, encoding=encoding, sep=separator)
                    df_full.columns = [col.strip() for col in df_full.columns]
                    if not all(col in df_full.columns for col in feature_columns_expected): continue
                    df_features = df_full[feature_columns_expected].copy()
                    if df_features.empty or len(df_features) < min_seq_len_threshold: continue
                    for col in df_features.columns:
                        df_features.loc[:, col] = pd.to_numeric(df_features[col], errors='coerce')
                    if df_features.isnull().sum().sum() > 0.1 * df_features.size: continue
                    df_features = df_features.fillna(0)
                    sequence_data = df_features.values.astype(np.float32)
                    raw_items.append({'data': sequence_data, 'label': label, 'patient_id': patient_id, 'class_name': class_name_key})
                except Exception as e:
                    print(f"ERROR processing {os.path.basename(csv_file_path)}: {e}. Skipping.")

    log_message(f_out, f"\nData loading complete. Loaded {len(raw_items)} raw sequences.\n" + "-"*70 + "\n", True)
    return raw_items

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
    return engineered_items, final_num_channels, final_feature_names

def pad_sequence(sequence, target_len, num_features):
    current_len = sequence.shape[0]
    if current_len >= target_len: return sequence[:target_len, :]
    padding = np.zeros((target_len - current_len, num_features), dtype=np.float32)
    return np.vstack((sequence, padding))

# --- NEW: Data Normalization Class ---
class SaccadeStandardScaler:
    """A scaler to handle normalization for 3D sequence data correctly within CV folds."""
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, data_items):
        # Concatenate all sequences in the training data to fit the scaler
        all_sequences = np.vstack([item['data'] for item in data_items])
        self.scaler.fit(all_sequences)

    def transform(self, sequence):
        return self.scaler.transform(sequence)

# --- NEW: Early Stopping Class ---
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
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

# --- PyTorch Dataset and Model (Revised) ---
class SaccadeDataset(Dataset):
    """Custom PyTorch Dataset for saccade sequences with normalization."""
    def __init__(self, items, target_seq_len, num_features, label_map, scaler=None):
        self.items = items
        self.target_seq_len = target_seq_len
        self.num_features = num_features
        self.label_map = label_map
        self.scaler = scaler

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        sequence = item['data']
        label = item['label']
        
        # 1. Normalize the data
        if self.scaler:
            sequence = self.scaler.transform(sequence)
        
        # 2. Pad/Truncate the sequence
        processed_sequence = pad_sequence(sequence, self.target_seq_len, self.num_features)

        # 3. Apply the new label mapping
        final_label = self.label_map[label]

        return torch.from_numpy(processed_sequence).float(), torch.tensor(final_label, dtype=torch.long)

# --- NEW: Attention Module ---
class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True
        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim 
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

# --- NEW: V2 Model Architecture ---
class SaccadeRNN_V2(nn.Module):
    """Bidirectional GRU with Self-Attention."""
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, seq_len, dropout_prob):
        super(SaccadeRNN_V2, self).__init__()
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True,
                          dropout=dropout_prob, bidirectional=True)
        
        self.attention = Attention(hidden_dim * 2, seq_len) # *2 for bidirectional
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        # GRU output: (batch_size, seq_length, hidden_dim * 2)
        gru_out, _ = self.gru(x)
        
        # Attention layer
        # context_vector shape: (batch_size, hidden_dim * 2)
        context_vector = self.attention(gru_out)
        
        # Final classification
        out = self.fc(context_vector)
        return out

# --- Training and Evaluation Functions (Unchanged) ---
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct_predictions, total_samples = 0.0, 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += labels.size(0)
    return running_loss / total_samples, (correct_predictions.double() / total_samples).item()

def evaluate_epoch(model, dataloader, criterion, device):
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

# --- Plotting Functions (Unchanged) ---
def plot_loss_curves(train_losses, val_losses, fold, filepath):
    plt.figure(figsize=(10, 6)); plt.plot(train_losses, label='Training Loss'); plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Fold {fold+1} - Training & Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True); plt.savefig(filepath); plt.close()

def plot_confusion_matrix(cm, class_names, title, filepath):
    plt.figure(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title); plt.ylabel('Actual'); plt.xlabel('Predicted'); plt.savefig(filepath); plt.close()

# --- Main Execution Block ---
if __name__ == '__main__':
    # 1. Setup
    np.random.seed(RANDOM_STATE); torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available(): torch.cuda.manual_seed(RANDOM_STATE)
    create_results_directory()
    f_log = get_file_handler(os.path.join(RESULTS_DIR, LOG_FILENAME))

    log_message(f_log, f"Experiment: {EXP_NAME} - Advanced RNN Classification")
    log_message(f_log, f"\n--- Configuration ---\nDevice: {DEVICE}\nEpochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}, Weight Decay: {WEIGHT_DECAY}\nModel: Bi-GRU+Attention, Hidden Dim: {HIDDEN_DIM}, Layers: {N_LAYERS}, Dropout: {DROPOUT_PROB}\nEarly Stopping Patience: {EARLY_STOPPING_PATIENCE}\n" + "-"*70)

    # 2. Data Loading and Preprocessing
    raw_items = load_raw_sequences_and_labels(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, CSV_ENCODING, CSV_SEPARATOR, MIN_SEQ_LEN_THRESHOLD, f_log)
    engineered_items, num_features, feature_names = engineer_features_from_raw_data(raw_items, FEATURE_COLUMNS, f_log)
    
    # Filter out excluded class and create new label mappings
    filtered_items = [item for item in engineered_items if item['class_name'] != CLASS_TO_EXCLUDE]
    remaining_class_names = sorted([c for c in CLASS_DEFINITIONS if c != CLASS_TO_EXCLUDE], key=lambda c: CLASS_DEFINITIONS[c]['label'])
    final_class_map_original_label_to_new = {CLASS_DEFINITIONS[name]['label']: i for i, name in enumerate(remaining_class_names)}
    num_classes = len(remaining_class_names)
    log_message(f_log, f"Filtered out '{CLASS_TO_EXCLUDE}'. {len(filtered_items)} sequences remaining.")
    log_message(f_log, f"Final class mapping: { {name: i for i, name in enumerate(remaining_class_names)} }\n" + "-"*70)

    # Determine sequence length - use MAX length
    lengths = [item['original_length'] for item in filtered_items]
    target_seq_len = np.max(lengths)
    log_message(f_log, f"Sequence length for padding (max): {target_seq_len}")
    
    X, y, groups = np.array(filtered_items), np.array([item['label'] for item in filtered_items]), np.array([item['patient_id'] for item in filtered_items])
    
    # 3. Cross-Validation Loop
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    overall_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []}
    aggregated_cm = np.zeros((num_classes, num_classes))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y, groups)):
        log_message(f_log, "\n" + "="*70 + f"\nStarting Fold {fold+1}/{N_FOLDS}\n" + "="*70)
        train_items, val_items = X[train_idx].tolist(), X[val_idx].tolist()

        # NEW: Fit scaler ONLY on training data for this fold
        scaler = SaccadeStandardScaler()
        scaler.fit(train_items)
        
        # Create datasets with the fold-specific scaler
        train_dataset = SaccadeDataset(train_items, target_seq_len, num_features, final_class_map_original_label_to_new, scaler)
        val_dataset = SaccadeDataset(val_items, target_seq_len, num_features, final_class_map_original_label_to_new, scaler)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        log_message(f_log, f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}.")

        # Initialize model, loss, optimizer, and NEW scheduler/stopper
        model = SaccadeRNN_V2(num_features, HIDDEN_DIM, num_classes, N_LAYERS, target_seq_len, DROPOUT_PROB).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5)
        early_stopper = EarlyStopper(patience=EARLY_STOPPING_PATIENCE)

        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc, _, _ = evaluate_epoch(model, val_loader, criterion, DEVICE)
            
            history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
            log_message(f_log, f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}", True)
            
            scheduler.step(val_loss)
            if early_stopper.early_stop(val_loss):
                log_message(f_log, f"Early stopping triggered at epoch {epoch+1}")
                break

        _, _, y_pred, y_true = evaluate_epoch(model, val_loader, criterion, DEVICE)
        log_message(f_log, f"\n--- Fold {fold+1} Results ---")
        report_text = classification_report(y_true, y_pred, target_names=remaining_class_names)
        log_message(f_log, report_text)
        report_dict = classification_report(y_true, y_pred, target_names=remaining_class_names, output_dict=True)
        overall_metrics['accuracy'].append(report_dict['accuracy'])
        overall_metrics['precision'].append(report_dict['macro avg']['precision'])
        overall_metrics['recall'].append(report_dict['macro avg']['recall'])
        overall_metrics['f1-score'].append(report_dict['macro avg']['f1-score'])

        cm = confusion_matrix(y_true, y_pred)
        aggregated_cm += cm
        cm_path = os.path.join(RESULTS_DIR, f'{EXP_NAME}_fold_{fold+1}_confusion_matrix.png')
        plot_confusion_matrix(cm, remaining_class_names, f'Fold {fold+1} - Confusion Matrix', cm_path)
        
        loss_curve_path = os.path.join(RESULTS_DIR, f'{EXP_NAME}_fold_{fold+1}_loss_curve.png')
        plot_loss_curves(history['train_loss'], history['val_loss'], fold, loss_curve_path)
        log_message(f_log, f"Plots for fold {fold+1} saved.")

    # 4. Final Aggregated Results
    log_message(f_log, "\n" + "="*70 + "\nCross-Validation Summary\n" + "="*70)
    log_message(f_log, f"Average Accuracy: {np.mean(overall_metrics['accuracy']):.4f} ± {np.std(overall_metrics['accuracy']):.4f}")
    log_message(f_log, f"Average Macro F1-Score: {np.mean(overall_metrics['f1-score']):.4f} ± {np.std(overall_metrics['f1-score']):.4f}")

    agg_cm_path = os.path.join(RESULTS_DIR, f'{EXP_NAME}_aggregated_confusion_matrix.png')
    plot_confusion_matrix(aggregated_cm, remaining_class_names, 'Aggregated Confusion Matrix (All Folds)', agg_cm_path)
    log_message(f_log, f"\nAggregated Confusion Matrix (sum over folds):\n{aggregated_cm}")
    log_message(f_log, f"Aggregated confusion matrix plot saved to: {agg_cm_path}")

    log_message(f_log, "\n" + "="*70 + "\nExperiment Finished.\n" + "="*70)
    f_log.close()
    print(f"\nExperiment complete. All results saved in: {RESULTS_DIR}")

