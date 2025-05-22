import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

# --- Configuration ---
BASE_DIR = './data'
CLASS_MAPPING = {'Healthy control': 0, 'Definite MG': 1}
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV'] # Original features
# NUM_CHANNELS will be determined after feature engineering
CSV_ENCODING = 'utf-16-le'
CSV_SEPARATOR = ','

TARGET_SEQ_LEN_PERCENTILE = 95
MIN_SEQ_LEN_THRESHOLD = 50

# Model & Training Hyperparameters
NUM_EPOCHS = 25 # Increased slightly to see effect of early stopping
BATCH_SIZE = 16
LEARNING_RATE = 1e-4 # Initial learning rate
N_SPLITS_CV = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Regularization & Early Stopping Parameters
WEIGHT_DECAY = 1e-5 # L2 regularization
CONV_DROPOUT_RATE = 0.25 # Dropout after conv blocks
FC_DROPOUT_RATE = 0.5  # Dropout in FC layers
EARLY_STOPPING_PATIENCE = 7 # Stop if no improvement for this many epochs
EARLY_STOPPING_METRIC = 'val_f1' # or 'val_loss'
EARLY_STOPPING_MODE = 'max' if EARLY_STOPPING_METRIC in ['val_f1', 'val_auc', 'val_acc'] else 'min'


print(f"Using device: {DEVICE}")
print(f"Early stopping metric: {EARLY_STOPPING_METRIC}, mode: {EARLY_STOPPING_MODE}, patience: {EARLY_STOPPING_PATIENCE}")

# --- Data Loading and Feature Engineering (from previous successful iteration) ---
def load_raw_sequences_and_labels(base_dir, class_mapping, feature_columns_expected, encoding, separator, min_seq_len_threshold):
    print("Starting data loading (with robust column handling)...")
    raw_items = []
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
                print(f"Warning: No CSV files found for patient {patient_id} in {patient_dir_path}")
                continue
            for csv_file_path in csv_files:
                try:
                    df_full = pd.read_csv(csv_file_path, encoding=encoding, sep=separator)
                    original_columns = df_full.columns.tolist()
                    df_full.columns = [col.strip() for col in original_columns]
                    cleaned_columns = df_full.columns.tolist()
                    missing_cols = [col for col in feature_columns_expected if col not in df_full.columns]
                    if missing_cols:
                        print(f"Warning: CSV {os.path.basename(csv_file_path)} missing expected columns: {missing_cols}. Skipping.")
                        continue
                    df_features = df_full[feature_columns_expected]
                    if df_features.empty or len(df_features) < min_seq_len_threshold:
                        print(f"Warning: CSV {os.path.basename(csv_file_path)} empty/short after selecting features. Skipping.")
                        continue
                    for col in df_features.columns:
                        df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
                    if df_features.isnull().sum().sum() > 0.1 * df_features.size :
                         print(f"Warning: CSV {os.path.basename(csv_file_path)} has >10% NaNs after to_numeric. Skipping.")
                         continue
                    df_features = df_features.fillna(0)
                    sequence_data = df_features.values.astype(np.float32)
                    raw_items.append((sequence_data, label, patient_id, os.path.basename(csv_file_path)))
                except Exception as e:
                    print(f"Warning: Could not process {os.path.basename(csv_file_path)}: {type(e).__name__} - {e}. Skipping.")
    print(f"Loaded {len(raw_items)} raw sequences.")
    return raw_items

def engineer_features_from_raw_data(raw_items, original_feature_names):
    print("Engineering new features (velocities, errors)...")
    engineered_items = []
    final_num_channels = 0
    for sequence_data, label, patient_id, filename in tqdm(raw_items, desc="Engineering Features"):
        df_original = pd.DataFrame(sequence_data, columns=original_feature_names)
        df_engineered_parts = []
        for pos_col in ['LH', 'RH', 'LV', 'RV']:
            if pos_col in df_original.columns:
                velocity_series = df_original[pos_col].diff().fillna(0)
                df_engineered_parts.append(velocity_series.rename(f'{pos_col}_Vel'))
            else:
                df_engineered_parts.append(pd.Series(np.zeros(len(df_original)), name=f'{pos_col}_Vel'))
        target_h_exists = 'TargetH' in df_original.columns
        target_v_exists = 'TargetV' in df_original.columns
        if target_h_exists:
            df_engineered_parts.append((df_original['LH'] - df_original['TargetH']).rename('ErrorH_L'))
            df_engineered_parts.append((df_original['RH'] - df_original['TargetH']).rename('ErrorH_R'))
        else:
            df_engineered_parts.append(pd.Series(np.zeros(len(df_original)), name='ErrorH_L'))
            df_engineered_parts.append(pd.Series(np.zeros(len(df_original)), name='ErrorH_R'))
        if target_v_exists:
            df_engineered_parts.append((df_original['LV'] - df_original['TargetV']).rename('ErrorV_L'))
            df_engineered_parts.append((df_original['RV'] - df_original['TargetV']).rename('ErrorV_R'))
        else:
            df_engineered_parts.append(pd.Series(np.zeros(len(df_original)), name='ErrorV_L'))
            df_engineered_parts.append(pd.Series(np.zeros(len(df_original)), name='ErrorV_R'))
        if df_engineered_parts:
             df_engineered_combined = pd.concat(df_engineered_parts, axis=1)
             df_all_features = pd.concat([df_original, df_engineered_combined], axis=1)
        else:
            df_all_features = df_original
        engineered_sequence_data = df_all_features.values.astype(np.float32)
        engineered_items.append((engineered_sequence_data, label, patient_id, filename))
        if final_num_channels == 0:
            final_num_channels = engineered_sequence_data.shape[1]
    print(f"Feature engineering complete. New number of channels: {final_num_channels}")
    return engineered_items, final_num_channels

def pad_or_truncate(sequence, target_len, num_channels):
    current_len = sequence.shape[0]
    if current_len == target_len: return sequence
    if current_len > target_len: return sequence[:target_len, :]
    padding = np.zeros((target_len - current_len, num_channels), dtype=np.float32)
    return np.vstack((sequence, padding))

def preprocess_data(engineered_items, target_seq_len_percentile, num_actual_channels):
    if not engineered_items: return [], [], [], 0
    print("Preprocessing engineered data...")
    lengths = [item[0].shape[0] for item in engineered_items]
    if not lengths: return [], [], [], 0
    target_len = int(np.percentile(lengths, target_seq_len_percentile))
    print(f"Sequence lengths stats: Min={np.min(lengths)}, Max={np.max(lengths)}, Mean={np.mean(lengths):.2f}, Median={np.median(lengths)}")
    print(f"Target sequence length ({target_seq_len_percentile}th percentile): {target_len}")
    if target_len == 0 and engineered_items: target_len = max(1, np.max(lengths))
    all_data_processed, all_labels_processed, all_groups_processed = [], [], []
    for sequence_data, label, patient_id, filename in tqdm(engineered_items, desc="Padding/Truncating"):
        if sequence_data.shape[1] != num_actual_channels:
             print(f"Error: Mismatch channels for {filename}. Exp {num_actual_channels}, Got {sequence_data.shape[1]}. Skipping.")
             continue
        processed_seq = pad_or_truncate(sequence_data, target_len, num_actual_channels)
        all_data_processed.append(np.transpose(processed_seq, (1, 0)))
        all_labels_processed.append(label)
        all_groups_processed.append(patient_id)
    print(f"Processed {len(all_data_processed)} samples with {num_actual_channels} channels each.")
    return all_data_processed, all_labels_processed, all_groups_processed, target_len

# --- PyTorch Dataset ---
class SaccadeDataset(Dataset):
    def __init__(self, data_list, labels_list):
        self.data = [torch.tensor(d, dtype=torch.float32) for d in data_list]
        self.labels = [torch.tensor([l], dtype=torch.float32) for l in labels_list]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

# --- 1D CNN Model with More Regularization ---
class Regularized1DCNN(nn.Module):
    def __init__(self, num_input_channels, num_classes, sequence_length, conv_dropout_rate, fc_dropout_rate):
        super(Regularized1DCNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=num_input_channels, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32), # Added BatchNorm
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(conv_dropout_rate) # Added Dropout
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64), # Added BatchNorm
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(conv_dropout_rate) # Added Dropout
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128), # Added BatchNorm
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2) 
            # No dropout after last conv maxpool, usually before FC
        )
        
        if sequence_length <= 0:
            print(f"Warning: sequence_length is {sequence_length}. Using fallback for flattened_size.")
            self.flattened_size = 1024 # Adjust if this seems too off, or ensure seq_len is always positive
        else:
            with torch.no_grad():
                dummy_input = torch.zeros(1, num_input_channels, sequence_length)
                x = self.conv_block1(dummy_input)
                x = self.conv_block2(x)
                x = self.conv_block3(x)
                self.flattened_size = x.shape[1] * x.shape[2]
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 96), # Slightly reduced from 128
            nn.ReLU(),
            nn.Dropout(fc_dropout_rate), # Using the passed fc_dropout_rate
            nn.Linear(96, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.fc_layers(x)
        return x

# --- Training and Evaluation Functions (Adapted for Early Stopping) ---
def train_epoch(model, dataloader, optimizer, criterion, device):
    # (Same as before, returns: avg_loss, accuracy, precision, recall, f1)
    model.train()
    total_loss = 0
    all_preds_probs_epoch, all_targets_epoch = [], []
    for data, targets in tqdm(dataloader, desc="Training Batch", leave=False):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_preds_probs_epoch.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        all_targets_epoch.extend(targets.detach().cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    all_targets_np = np.array(all_targets_epoch)
    all_preds_binary_np = (np.array(all_preds_probs_epoch) > 0.5).astype(int)
    accuracy = accuracy_score(all_targets_np, all_preds_binary_np)
    precision = precision_score(all_targets_np, all_preds_binary_np, zero_division=0)
    recall = recall_score(all_targets_np, all_preds_binary_np, zero_division=0)
    f1 = f1_score(all_targets_np, all_preds_binary_np, zero_division=0)
    return avg_loss, accuracy, precision, recall, f1


def evaluate_model(model, dataloader, criterion, device):
    # (Same as before, returns: avg_loss, accuracy, precision, recall, f1, roc_auc)
    model.eval()
    total_loss = 0
    all_preds_probs, all_targets = [], []
    with torch.no_grad():
        for data, targets in tqdm(dataloader, desc="Evaluating Batch", leave=False):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds_probs.extend(probs)
            all_targets.extend(targets.detach().cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    all_targets_np = np.array(all_targets)
    all_preds_probs_np = np.array(all_preds_probs)
    all_preds_binary_np = (all_preds_probs_np > 0.5).astype(int)
    accuracy = accuracy_score(all_targets_np, all_preds_binary_np)
    precision = precision_score(all_targets_np, all_preds_binary_np, zero_division=0)
    recall = recall_score(all_targets_np, all_preds_binary_np, zero_division=0)
    f1 = f1_score(all_targets_np, all_preds_binary_np, zero_division=0)
    roc_auc = 0.0
    if len(np.unique(all_targets_np)) > 1:
        try: roc_auc = roc_auc_score(all_targets_np, all_preds_probs_np)
        except ValueError as e: print(f"Warning: ROC AUC not computed. {e}")
    else: print("Warning: ROC AUC not computed (1 class in targets).")
    return avg_loss, accuracy, precision, recall, f1, roc_auc


# --- Main Script Execution ---
if __name__ == '__main__':
    raw_items_original_features = load_raw_sequences_and_labels(
        BASE_DIR, CLASS_MAPPING, FEATURE_COLUMNS, CSV_ENCODING, CSV_SEPARATOR, MIN_SEQ_LEN_THRESHOLD
    )
    if not raw_items_original_features:
        print("No data loaded. Exiting."); exit()

    engineered_items_with_new_features, final_num_channels = engineer_features_from_raw_data(
        raw_items_original_features, FEATURE_COLUMNS
    )
    if not engineered_items_with_new_features:
        print("Feature engineering failed. Exiting."); exit()
    
    all_data_np, all_labels_np, all_groups_np, final_sequence_length = preprocess_data(
        engineered_items_with_new_features, TARGET_SEQ_LEN_PERCENTILE, final_num_channels
    )
    if not all_data_np:
        print("Data preprocessing failed. Exiting."); exit()
    if final_sequence_length <= 0:
        print(f"Error: final_sequence_length {final_sequence_length} invalid. Exiting."); exit()

    all_labels_np_1d = np.array(all_labels_np).ravel()
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=42)
    fold_results = []
    print(f"\nStarting {N_SPLITS_CV}-Fold Cross-Validation with {final_num_channels} input channels...")

    for fold_idx, (train_indices, val_indices) in enumerate(sgkf.split(all_data_np, all_labels_np_1d, groups=all_groups_np)):
        print(f"\n--- Fold {fold_idx + 1}/{N_SPLITS_CV} ---")
        train_data = [all_data_np[i] for i in train_indices]
        train_labels = [all_labels_np[i] for i in train_indices]
        val_data = [all_data_np[i] for i in val_indices]
        val_labels = [all_labels_np[i] for i in val_indices]

        train_dataset = SaccadeDataset(train_data, train_labels)
        val_dataset = SaccadeDataset(val_data, val_labels)
        if not train_dataset or not val_dataset:
            print(f"Warning: Fold {fold_idx + 1} has empty train/val set. Skipping."); fold_results.append({}); continue

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=DEVICE.type=='cuda')
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=DEVICE.type=='cuda')

        model = Regularized1DCNN(
            num_input_channels=final_num_channels, num_classes=1, sequence_length=final_sequence_length,
            conv_dropout_rate=CONV_DROPOUT_RATE, fc_dropout_rate=FC_DROPOUT_RATE
        ).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # Added weight_decay
        criterion = nn.BCEWithLogitsLoss()

        print(f"Fold {fold_idx + 1}: Training on {len(train_dataset)}, Validating on {len(val_dataset)} samples.")
        
        best_metric_value = -np.inf if EARLY_STOPPING_MODE == 'max' else np.inf
        epochs_no_improve = 0
        current_fold_best_epoch_info = {}

        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_acc, val_prec, val_rec, val_f1, val_roc_auc = evaluate_model(model, val_loader, criterion, DEVICE)
            
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} => "
                  f"Tr Ls: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
                  f"Val Ls: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_roc_auc:.4f}")

            # Early Stopping Check
            current_metric_value = 0
            if EARLY_STOPPING_METRIC == 'val_f1': current_metric_value = val_f1
            elif EARLY_STOPPING_METRIC == 'val_auc': current_metric_value = val_roc_auc
            elif EARLY_STOPPING_METRIC == 'val_acc': current_metric_value = val_acc
            else: current_metric_value = val_loss # Default to val_loss

            improved = False
            if EARLY_STOPPING_MODE == 'max':
                if current_metric_value > best_metric_value:
                    best_metric_value = current_metric_value
                    improved = True
            else: # min mode
                if current_metric_value < best_metric_value:
                    best_metric_value = current_metric_value
                    improved = True
            
            if improved:
                epochs_no_improve = 0
                current_fold_best_epoch_info = {
                    'fold': fold_idx + 1, 'epoch': epoch + 1, 
                    'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1,
                    'val_precision': val_prec, 'val_recall': val_rec, 'val_roc_auc': val_roc_auc,
                    'train_loss': train_loss, 'train_f1': train_f1 # Also log train metrics for best epoch
                }
                # torch.save(model.state_dict(), f"model_fold_{fold_idx+1}_best_reg.pth") # Optional: save best model
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch + 1} for fold {fold_idx + 1}. Best {EARLY_STOPPING_METRIC}: {best_metric_value:.4f}")
                break
        
        if current_fold_best_epoch_info:
            fold_results.append(current_fold_best_epoch_info)
            print(f"Best {EARLY_STOPPING_METRIC} for Fold {fold_idx + 1} was {best_metric_value:.4f} at Epoch {current_fold_best_epoch_info['epoch']}")
        else:
            print(f"No valid epoch results (or no improvement) for Fold {fold_idx + 1}.")
            # If no improvement from epoch 1, current_fold_best_epoch_info might be empty
            # Add a placeholder or the last epoch's info if needed for consistent reporting
            if not current_fold_best_epoch_info and epoch == NUM_EPOCHS -1 : # if loop finished without improvement
                 current_fold_best_epoch_info = { # Log last epoch if nothing better
                    'fold': fold_idx + 1, 'epoch': epoch + 1, 'val_f1': val_f1, 'val_acc': val_acc,
                     'val_precision': val_prec, 'val_recall': val_rec, 'val_roc_auc': val_roc_auc, 'val_loss': val_loss
                 }
            fold_results.append(current_fold_best_epoch_info if current_fold_best_epoch_info else {})


    print("\n--- Cross-Validation Summary (Best Epoch per Fold based on Val F1) ---")
    valid_fold_results = [res for res in fold_results if res and 'val_f1' in res] # Ensure key metrics exist
    if valid_fold_results:
        avg_val_acc = np.mean([res['val_acc'] for res in valid_fold_results])
        avg_val_f1 = np.mean([res['val_f1'] for res in valid_fold_results])
        # ... (rest of the averaging and printing logic)
        avg_val_precision = np.mean([res['val_precision'] for res in valid_fold_results])
        avg_val_recall = np.mean([res['val_recall'] for res in valid_fold_results])
        avg_val_roc_auc = np.mean([res['val_roc_auc'] for res in valid_fold_results])

        for res_item in valid_fold_results: 
            print(f"Fold {res_item['fold']}: Acc={res_item['val_acc']:.4f}, F1={res_item['val_f1']:.4f}, Precision={res_item['val_precision']:.4f}, Recall={res_item['val_recall']:.4f}, AUC={res_item['val_roc_auc']:.4f} (Epoch {res_item['epoch']})")

        print("\nAverage Cross-Validation Metrics (from best epochs of valid folds):")
        print(f"  Average Validation Accuracy: {avg_val_acc:.4f}")
        print(f"  Average Validation F1-Score: {avg_val_f1:.4f}")
        print(f"  Average Validation Precision: {avg_val_precision:.4f}")
        print(f"  Average Validation Recall: {avg_val_recall:.4f}")
        print(f"  Average Validation ROC AUC: {avg_val_roc_auc:.4f}")

    else:
        print("No valid fold results to summarize.")
    print("\nExperiment script with regularization finished.")
