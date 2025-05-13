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
from tqdm import tqdm # For progress bars

# --- Configuration ---
BASE_DIR = './data'
# Corrected based on your initial problem description, if it's 'Healthy control' ensure it matches
CLASS_MAPPING = {'Healthy control': 0, 'Definite MG': 1} 
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']
NUM_CHANNELS = len(FEATURE_COLUMNS)
CSV_ENCODING = 'utf-16-le'
CSV_SEPARATOR = ','

TARGET_SEQ_LEN_PERCENTILE = 95 
MIN_SEQ_LEN_THRESHOLD = 50 

NUM_EPOCHS = 15 
BATCH_SIZE = 16 
LEARNING_RATE = 1e-4
N_SPLITS_CV = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# --- Data Loading and Preprocessing (Corrected) ---

def load_raw_sequences_and_labels(base_dir, class_mapping, feature_columns_expected, encoding, separator, min_seq_len_threshold):
    """
    Loads all raw sequences from CSV files for all patients,
    using the robust parsing method.
    """
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
                    # Step 1: Read all columns using the proven encoding, without usecols initially
                    df_full = pd.read_csv(csv_file_path, encoding=encoding, sep=separator)
                    
                    # Step 2: Clean the DataFrame's actual column names
                    original_columns = df_full.columns.tolist() # For logging
                    df_full.columns = [col.strip() for col in df_full.columns]
                    cleaned_columns = df_full.columns.tolist() # For logging

                    # Step 3: Check if all expected feature columns are present in the cleaned column names
                    missing_cols = [col for col in feature_columns_expected if col not in df_full.columns]
                    if missing_cols:
                        print(f"Warning: CSV {os.path.basename(csv_file_path)} is missing expected columns after cleaning: {missing_cols}. \n"
                              f"         Original cols: {original_columns}\n"
                              f"         Cleaned cols: {cleaned_columns}. Skipping file.")
                        continue
                        
                    # Step 4: Select the desired feature columns
                    df_features = df_full[feature_columns_expected]

                    if df_features.empty or len(df_features) < min_seq_len_threshold:
                        print(f"Warning: CSV {os.path.basename(csv_file_path)} results in empty or too short DataFrame ({len(df_features)} rows) after selecting features. Skipping.")
                        continue
                    
                    # Ensure data is numeric, coercing errors to NaN.
                    # This might be redundant if CSVs are clean, but good for robustness.
                    for col in df_features.columns:
                        df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
                    
                    # Check for NaNs introduced by coercion, if any significant, skip or warn
                    if df_features.isnull().sum().sum() > 0.1 * df_features.size : # Example: if >10% values are NaN
                         print(f"Warning: CSV {os.path.basename(csv_file_path)} has significant NaNs after to_numeric. Skipping.")
                         continue
                    df_features = df_features.fillna(0) # Or another imputation strategy like ffill/bfill


                    sequence_data = df_features.values.astype(np.float32) 
                    raw_items.append((sequence_data, label, patient_id, os.path.basename(csv_file_path)))

                except FileNotFoundError:
                    print(f"Warning: File not found {csv_file_path}. Skipping.")
                except pd.errors.EmptyDataError:
                    print(f"Warning: Empty CSV {os.path.basename(csv_file_path)}. Skipping.")
                except Exception as e:
                    print(f"Warning: Could not read/process {os.path.basename(csv_file_path)}: {type(e).__name__} - {e}. Skipping.")
    
    print(f"Loaded {len(raw_items)} raw sequences.")
    return raw_items

def pad_or_truncate(sequence, target_len, num_channels):
    """Pads or truncates a sequence to the target length."""
    current_len = sequence.shape[0]
    if current_len == target_len:
        return sequence
    elif current_len > target_len:
        return sequence[:target_len, :]
    else: # current_len < target_len
        padding_needed = target_len - current_len
        padding = np.zeros((padding_needed, num_channels), dtype=np.float32)
        return np.vstack((sequence, padding))

def preprocess_data(raw_items, target_seq_len_percentile, num_channels):
    if not raw_items:
        print("Error: No raw items to preprocess.")
        return [], [], [], 0

    print("Preprocessing data...")
    lengths = [item[0].shape[0] for item in raw_items]
    if not lengths: # Should not happen if raw_items is not empty
        print("Error: No valid sequence lengths found among loaded items.")
        return [], [], [], 0

    target_len = int(np.percentile(lengths, target_seq_len_percentile))
    print(f"Sequence lengths stats: Min={np.min(lengths)}, Max={np.max(lengths)}, Mean={np.mean(lengths):.2f}, Median={np.median(lengths)}")
    print(f"Target sequence length ({target_seq_len_percentile}th percentile): {target_len}")
    
    if target_len == 0 and len(raw_items) > 0 : # Handle cases where percentile might give 0 if all sequences are short but non-empty
        target_len = max(1, np.max(lengths)) # Use max length or at least 1
        print(f"Warning: Percentile target length was 0, adjusted to max length: {target_len}")


    all_data_processed = []
    all_labels_processed = []
    all_groups_processed = []

    for sequence_data, label, patient_id, filename in tqdm(raw_items, desc="Padding/Truncating"):
        # Ensure sequence_data has 2 dimensions even if it was a single column after bad selection
        if sequence_data.ndim == 1:
            print(f"Warning: sequence_data for {filename} is 1D. Reshaping. Original shape: {sequence_data.shape}")
            # This case should ideally not be hit if column selection and checks are robust
            sequence_data = sequence_data.reshape(-1, 1) 
            # If it's 1D, num_channels might be mismatched. This indicates a prior problem.
            # For now, we'll proceed, but this needs monitoring.
            # If we expect multiple channels, this sequence is problematic.
            if sequence_data.shape[1] != num_channels:
                print(f"Error: sequence_data for {filename} has {sequence_data.shape[1]} channels, expected {num_channels}. Skipping.")
                continue


        processed_seq = pad_or_truncate(sequence_data, target_len, sequence_data.shape[1]) # Use actual num_channels from data
        processed_seq_transposed = np.transpose(processed_seq, (1, 0)) 
        
        all_data_processed.append(processed_seq_transposed)
        all_labels_processed.append(label)
        all_groups_processed.append(patient_id)
        
    print(f"Processed {len(all_data_processed)} samples.")
    return all_data_processed, all_labels_processed, all_groups_processed, target_len

# --- PyTorch Dataset ---
class SaccadeDataset(Dataset):
    def __init__(self, data_list, labels_list):
        self.data = [torch.tensor(d, dtype=torch.float32) for d in data_list]
        self.labels = [torch.tensor([l], dtype=torch.float32) for l in labels_list] 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# --- 1D CNN Model ---
class Simple1DCNN(nn.Module):
    def __init__(self, num_input_channels, num_classes, sequence_length): 
        super(Simple1DCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=num_input_channels, out_channels=32, kernel_size=7, stride=1, padding=3), 
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Dynamic calculation of flattened_size
        if sequence_length <= 0: # Add a check for valid sequence_length
            print(f"Warning: sequence_length is {sequence_length}. Cannot dynamically calculate flattened size. Defaulting to a preset value (e.g. 1024) or expecting error.")
            self.flattened_size = 1024 # Fallback, might need adjustment or error if sequence_length isn't positive
        else:
             with torch.no_grad():
                dummy_input = torch.zeros(1, num_input_channels, sequence_length)
                dummy_out = self.conv_layers(dummy_input)
                self.flattened_size = dummy_out.shape[1] * dummy_out.shape[2]

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes) 
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# --- Training and Evaluation Functions (largely unchanged) ---
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds_probs_epoch = [] # Store probabilities for more flexible metric calculation if needed later
    all_targets_epoch = []

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
    model.eval()
    total_loss = 0
    all_preds_probs = []
    all_targets = []

    with torch.no_grad():
        for data, targets in tqdm(dataloader, desc="Evaluating Batch", leave=False):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets) # Ensure criterion is appropriate if targets are not one-hot
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds_probs.extend(probs)
            all_targets.extend(targets.detach().cpu().numpy()) # Assuming targets are (batch, 1) or (batch,)
            
    avg_loss = total_loss / len(dataloader)
    
    all_targets_np = np.array(all_targets)
    all_preds_probs_np = np.array(all_preds_probs)
    all_preds_binary_np = (all_preds_probs_np > 0.5).astype(int)
    
    accuracy = accuracy_score(all_targets_np, all_preds_binary_np)
    precision = precision_score(all_targets_np, all_preds_binary_np, zero_division=0)
    recall = recall_score(all_targets_np, all_preds_binary_np, zero_division=0)
    f1 = f1_score(all_targets_np, all_preds_binary_np, zero_division=0)
    roc_auc = 0.0
    if len(np.unique(all_targets_np)) > 1: # ROC AUC is only valid if there's more than one class in targets
        try:
            roc_auc = roc_auc_score(all_targets_np, all_preds_probs_np)
        except ValueError as e:
            print(f"Warning: ROC AUC could not be computed. {e}")
    else:
        print("Warning: ROC AUC not computed because only one class present in evaluation targets.")

    return avg_loss, accuracy, precision, recall, f1, roc_auc


# --- Main Script Execution ---
if __name__ == '__main__':
    raw_items = load_raw_sequences_and_labels(
        BASE_DIR, CLASS_MAPPING, FEATURE_COLUMNS, CSV_ENCODING, CSV_SEPARATOR, MIN_SEQ_LEN_THRESHOLD
    )

    if not raw_items:
        print("No data loaded after attempting robust parsing. Exiting.")
        exit()

    all_data_np, all_labels_np, all_groups_np, final_sequence_length = preprocess_data(
        raw_items, TARGET_SEQ_LEN_PERCENTILE, NUM_CHANNELS
    )
    
    if not all_data_np:
        print("Data preprocessing failed or resulted in no data. Exiting.")
        exit()
    
    if final_sequence_length <= 0 : # Critical check for model initialization
        print(f"Error: final_sequence_length is {final_sequence_length}, which is invalid for model. Exiting.")
        exit()

    all_labels_np_1d = np.array(all_labels_np).ravel() # Ensure labels are 1D for StratifiedGroupKFold

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=42)
    fold_results = []
    print(f"\nStarting {N_SPLITS_CV}-Fold Cross-Validation...")

    for fold_idx, (train_indices, val_indices) in enumerate(sgkf.split(all_data_np, all_labels_np_1d, groups=all_groups_np)):
        print(f"\n--- Fold {fold_idx + 1}/{N_SPLITS_CV} ---")

        train_data = [all_data_np[i] for i in train_indices]
        train_labels = [all_labels_np[i] for i in train_indices] # Keep as list for dataset
        val_data = [all_data_np[i] for i in val_indices]
        val_labels = [all_labels_np[i] for i in val_indices] # Keep as list for dataset

        train_dataset = SaccadeDataset(train_data, train_labels)
        val_dataset = SaccadeDataset(val_data, val_labels)

        # Handle cases where a split might result in an empty dataset (though unlikely with StratifiedGroupKFold if data is sufficient)
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print(f"Warning: Fold {fold_idx + 1} has an empty train or validation set. Skipping this fold.")
            fold_results.append({}) # Append empty dict or handle as appropriate
            continue

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False)

        model = Simple1DCNN(num_input_channels=NUM_CHANNELS, num_classes=1, sequence_length=final_sequence_length).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()

        print(f"Fold {fold_idx + 1}: Training on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples.")
        best_val_f1 = -1
        current_fold_best_epoch_info = {}

        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_acc, val_prec, val_rec, val_f1, val_roc_auc = evaluate_model(model, val_loader, criterion, DEVICE)
            
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} => "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_roc_auc:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                current_fold_best_epoch_info = {
                    'fold': fold_idx + 1, 'epoch': epoch + 1, 
                    'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1,
                    'val_precision': val_prec, 'val_recall': val_rec, 'val_roc_auc': val_roc_auc
                }
        
        if current_fold_best_epoch_info: # Check if any training happened / results were logged
            fold_results.append(current_fold_best_epoch_info)
            print(f"Best F1 for Fold {fold_idx + 1} was {current_fold_best_epoch_info['val_f1']:.4f} at Epoch {current_fold_best_epoch_info['epoch']}")
        else:
            print(f"No valid epoch results recorded for Fold {fold_idx + 1}.")
            fold_results.append({}) # Append empty dict to maintain result list length

    print("\n--- Cross-Validation Summary (Best Epoch per Fold based on Val F1) ---")
    # Filter out empty results before calculating mean
    valid_fold_results = [res for res in fold_results if res] 

    if valid_fold_results:
        avg_val_acc = np.mean([res['val_acc'] for res in valid_fold_results])
        avg_val_f1 = np.mean([res['val_f1'] for res in valid_fold_results])
        avg_val_precision = np.mean([res['val_precision'] for res in valid_fold_results])
        avg_val_recall = np.mean([res['val_recall'] for res in valid_fold_results])
        avg_val_roc_auc = np.mean([res['val_roc_auc'] for res in valid_fold_results])

        for i, res in enumerate(valid_fold_results): # Iterate through valid results
            print(f"Fold {res['fold']}: Acc={res['val_acc']:.4f}, F1={res['val_f1']:.4f}, Precision={res['val_precision']:.4f}, Recall={res['val_recall']:.4f}, AUC={res['val_roc_auc']:.4f} (Epoch {res['epoch']})")

        print("\nAverage Cross-Validation Metrics (from best epochs of valid folds):")
        print(f"  Average Validation Accuracy: {avg_val_acc:.4f}")
        print(f"  Average Validation F1-Score: {avg_val_f1:.4f}")
        print(f"  Average Validation Precision: {avg_val_precision:.4f}")
        print(f"  Average Validation Recall: {avg_val_recall:.4f}")
        print(f"  Average Validation ROC AUC: {avg_val_roc_auc:.4f}")
    else:
        print("No valid fold results to summarize.")

    print("\nInitial experiment script finished.")
