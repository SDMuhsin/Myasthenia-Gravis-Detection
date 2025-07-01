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
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
import gc

# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# ========================================================================================
# --- Configuration for Experiment 07 ---
# ========================================================================================
# --- Basic Setup ---
BASE_DIR = './data'
EXP_NAME = 'EXP_07_Patient_Level_Diagnosis'
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

# --- Segmentation Parameters ---
PRE_ONSET_WINDOW = 50
POST_ONSET_WINDOW = 446
SEGMENT_LENGTH = PRE_ONSET_WINDOW + POST_ONSET_WINDOW # 496, divisible by 8

# --- Hyperparameters for the Two-Stage Model ---
N_FOLDS = 5
# Stage 1: Autoencoder Hyperparameters
CAE_EPOCHS = 50
CAE_BATCH_SIZE = 128
CAE_LR = 0.001
EMBEDDING_DIM = 256

# Stage 2: XGBoost Hyperparameters
# --- FIX: Moved early_stopping_rounds into the constructor parameters ---
XGB_PARAMS = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'n_estimators': 250,
    'learning_rate': 0.05,
    'max_depth': 64,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'early_stopping_rounds': 100, # Moved here
    'use_label_encoder': False,
    'seed': RANDOM_STATE
}

# ========================================================================================
# --- Utility and Setup Functions ---
# ========================================================================================
def create_results_directory(dir_path=RESULTS_DIR):
    if not os.path.exists(dir_path): os.makedirs(dir_path); print(f"INFO: Created directory: {dir_path}")
def get_file_handler(filepath): return open(filepath, 'w', encoding='utf-8')
def log_message(f_handler, message, print_to_console=True):
    f_handler.write(message + '\n'); f_handler.flush()
    if print_to_console: print(message)

# ========================================================================================
# --- Data Loading and Processing Pipeline ---
# ========================================================================================
def load_raw_sequences_and_labels(base_dir, class_definitions_dict, f_out):
    print("\n" + "="*50 + "\n1. Starting Raw Data Loading...\n" + "="*50)
    raw_items = []
    for class_name_key, class_details in class_definitions_dict.items():
        label, class_dir_abs = class_details['label'], os.path.join(base_dir, class_details['path'])
        if not os.path.isdir(class_dir_abs): continue
        patient_dirs = [d for d in os.listdir(class_dir_abs) if os.path.isdir(os.path.join(class_dir_abs, d))]
        for patient_folder_name in tqdm(patient_dirs, desc=f"  Patients in {class_name_key}", ncols=100):
            patient_id = f"{class_name_key}_{patient_folder_name}"
            patient_dir_path = os.path.join(class_dir_abs, patient_folder_name)
            for csv_file_path in glob.glob(os.path.join(patient_dir_path, '*.csv')):
                try:
                    df_full = pd.read_csv(csv_file_path, encoding=CSV_ENCODING, sep=CSV_SEPARATOR, low_memory=False)
                    df_full.columns = [col.strip() for col in df_full.columns]
                    if not all(col in df_full.columns for col in FEATURE_COLUMNS): continue
                    df_features = df_full[FEATURE_COLUMNS].copy()
                    if df_features.empty or len(df_features) < MIN_SEQ_LEN_THRESHOLD: continue
                    for col in df_features.columns: df_features.loc[:, col] = pd.to_numeric(df_features[col], errors='coerce')
                    df_features.ffill(inplace=True); df_features.bfill(inplace=True); df_features.fillna(0, inplace=True)
                    raw_items.append({'data': df_features.values.astype(np.float32), 'label': label, 'patient_id': patient_id, 'class_name': class_name_key})
                except Exception as e: print(f"ERROR processing {os.path.basename(csv_file_path)}: {e}. Skipping.")
    log_message(f_out, f"Raw data loading complete. Loaded {len(raw_items)} raw sequences.", True)
    return raw_items

def engineer_and_segment_data(raw_items, f_out):
    print("\n" + "="*50 + "\n2. Engineering Features & Segmenting Saccades...\n" + "="*50)
    segmented_items = []
    for item in tqdm(raw_items, desc="  Processing Trials", ncols=100):
        df = pd.DataFrame(item['data'], columns=FEATURE_COLUMNS)
        df['LH_Vel'] = df['LH'].diff().fillna(0); df['RH_Vel'] = df['RH'].diff().fillna(0)
        df['LV_Vel'] = df['LV'].diff().fillna(0); df['RV_Vel'] = df['RV'].diff().fillna(0)
        df['ErrorH_L'] = df['LH'] - df['TargetH']; df['ErrorH_R'] = df['RH'] - df['TargetH']
        df['ErrorV_L'] = df['LV'] - df['TargetV']; df['ErrorV_R'] = df['RV'] - df['TargetV']
        sequence = df.values.astype(np.float32)
        target_h_idx, target_v_idx = list(df.columns).index('TargetH'), list(df.columns).index('TargetV')
        target_changes_h = np.diff(sequence[:, target_h_idx], prepend=sequence[0, target_h_idx]) != 0
        target_changes_v = np.diff(sequence[:, target_v_idx], prepend=sequence[0, target_v_idx]) != 0
        event_indices = np.where(target_changes_h | target_changes_v)[0]
        for idx in event_indices:
            start, end = idx - PRE_ONSET_WINDOW, idx + POST_ONSET_WINDOW
            if start >= 0 and end <= len(sequence):
                segmented_items.append({
                    'data': sequence[start:end, :], 'label': item['label'],
                    'patient_id': item['patient_id'], 'class_name': item['class_name']
                })
    log_message(f_out, f"Segmentation complete. Found {len(segmented_items)} segments.", True)
    return segmented_items, segmented_items[0]['data'].shape[1]

# ========================================================================================
# --- Stage 1 Components (Convolutional Autoencoder) ---
# ========================================================================================
class SaccadeCAEDataset(Dataset):
    def __init__(self, items, scaler):
        self.items = items
        self.scaler = scaler
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        sequence = self.items[idx]['data'].astype(np.float32)
        sequence = self.scaler.transform(sequence)
        return torch.from_numpy(sequence).float()

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(ConvolutionalAutoencoder, self).__init__()
        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=7, padding=3), nn.ReLU(True), nn.MaxPool1d(2, stride=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(True), nn.MaxPool1d(2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(True), nn.MaxPool1d(2, stride=2),
            nn.Flatten(),
            nn.Linear(128 * (SEGMENT_LENGTH // 8), embedding_dim)
        )
        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128 * (SEGMENT_LENGTH // 8)), nn.ReLU(True),
            nn.Unflatten(1, (128, (SEGMENT_LENGTH // 8))),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose1d(32, input_dim, kernel_size=7, stride=2, padding=3, output_padding=1),
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        reconstruction = reconstruction.permute(0, 2, 1)
        return reconstruction

def train_autoencoder(cae, train_loader, epochs, lr, f_out):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(cae.parameters(), lr=lr)
    log_message(f_out, "\n--- Training Convolutional Autoencoder ---", True)
    cae.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data in tqdm(train_loader, desc=f"CAE Epoch {epoch+1}/{epochs}", ncols=100, leave=False):
            inputs = data.to(DEVICE)
            optimizer.zero_grad()
            outputs = cae(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        log_message(f_out, f"CAE Epoch {epoch+1}/{epochs}, Reconstruction Loss: {epoch_loss:.6f}", True)
    return cae

# ========================================================================================
# --- Stage 2 Components (Patient-Level Feature Generation & XGBoost) ---
# ========================================================================================
def create_patient_level_features(items, encoder, scaler, f_out):
    log_message(f_out, "--- Generating Patient-Level Features ---", True)
    encoder.eval()
    all_embeddings, patient_ids, labels = [], [], []
    dataset = SaccadeCAEDataset(items, scaler)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    
    with torch.no_grad():
        for batch_data in tqdm(loader, desc="  Extracting segment embeddings", ncols=100, leave=False):
            batch_data = batch_data.to(DEVICE)
            embeddings = encoder(batch_data.permute(0, 2, 1))
            all_embeddings.append(embeddings.cpu().numpy())
            
    for item in items:
        patient_ids.append(item['patient_id'])
        labels.append(item['label'])
        
    df_embeddings = pd.DataFrame(np.vstack(all_embeddings))
    df_embeddings.columns = [f'emb_{i}' for i in range(EMBEDDING_DIM)]
    df_embeddings['patient_id'] = patient_ids
    df_embeddings['label'] = labels

    agg_funcs = ['mean', 'std', 'min', 'max', 'median']
    df_patient = df_embeddings.groupby('patient_id').agg({col: agg_funcs for col in df_embeddings.columns if 'emb' in col})
    df_patient.columns = ['_'.join(col).strip() for col in df_patient.columns.values]
    
    patient_labels = df_embeddings.groupby('patient_id')['label'].first()
    df_patient = df_patient.join(patient_labels)
    
    log_message(f_out, f"Generated feature matrix for {len(df_patient)} patients.", True)
    return df_patient

# --- FIX: Re-added the missing plot_confusion_matrix function ---
def plot_confusion_matrix(cm, class_names, title, filepath):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title); plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout(); plt.savefig(filepath); plt.close()

def plot_feature_importance(booster, filepath):
    fig, ax = plt.subplots(figsize=(12, 16))
    xgb.plot_importance(booster, ax=ax, max_num_features=50, height=0.8)
    plt.title('XGBoost Feature Importance (Top 50)')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_loss_curves(results, fold, filepath):
    plt.figure(figsize=(10, 6))
    evals = list(results.keys())
    plt.plot(results[evals[0]]['mlogloss'], label='Validation Loss')
    plt.plot(results[evals[1]]['mlogloss'], label='Training Loss')
    plt.title(f'Fold {fold+1} - XGBoost Training Loss')
    plt.xlabel('Epoch'); plt.ylabel('LogLoss')
    plt.legend(); plt.grid(True); plt.savefig(filepath); plt.close()

# ========================================================================================
# --- Main Execution Logic ---
# ========================================================================================
if __name__ == '__main__':
    np.random.seed(RANDOM_STATE); torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available(): torch.cuda.manual_seed(RANDOM_STATE)
    create_results_directory()
    f_log = get_file_handler(os.path.join(RESULTS_DIR, LOG_FILENAME))

    log_message(f_log, f"Experiment: {EXP_NAME}\nStrategy: Two-stage, patient-level diagnosis via Autoencoder features and XGBoost.")
    log_message(f_log, f"\n--- STAGE 1: Unsupervised CAE | Embedding Dim: {EMBEDDING_DIM}, Epochs: {CAE_EPOCHS}, Segment Length: {SEGMENT_LENGTH} ---")
    log_message(f_log, f"--- STAGE 2: Patient-Level XGBoost Classifier ---")
    log_message(f_log, "-"*70)

    raw_items = load_raw_sequences_and_labels(BASE_DIR, CLASS_DEFINITIONS, f_log)
    segmented_items, num_features = engineer_and_segment_data(raw_items, f_log)
    
    filtered_items = [item for item in segmented_items if item['class_name'] != CLASS_TO_EXCLUDE]
    remaining_class_names = sorted([c for c in CLASS_DEFINITIONS if c != CLASS_TO_EXCLUDE], key=lambda c: CLASS_DEFINITIONS[c]['label'])
    final_class_map = {CLASS_DEFINITIONS[name]['label']: i for i, name in enumerate(remaining_class_names)}
    num_classes = len(remaining_class_names)
    X_items, y_labels, groups = np.array(filtered_items, dtype=object), np.array([item['label'] for item in filtered_items]), np.array([item['patient_id'] for item in filtered_items])
    
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    overall_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []}
    aggregated_cm = np.zeros((num_classes, num_classes))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_items, y_labels, groups)):
        log_message(f_log, "\n" + "="*70 + f"\nStarting Fold {fold+1}/{N_FOLDS}\n" + "="*70)
        train_items, val_items = X_items[train_idx].tolist(), X_items[val_idx].tolist()
        
        scaler = StandardScaler().fit(np.vstack([item['data'] for item in train_items]))

        # --- STAGE 1: Train Autoencoder ---
        cae_dataset = SaccadeCAEDataset(train_items, scaler)
        cae_loader = DataLoader(cae_dataset, batch_size=CAE_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        cae_model = ConvolutionalAutoencoder(num_features, EMBEDDING_DIM).to(DEVICE)
        cae_model = train_autoencoder(cae_model, cae_loader, CAE_EPOCHS, CAE_LR, f_log)

        # --- STAGE 2: Create Patient-Level Datasets ---
        df_train_patient = create_patient_level_features(train_items, cae_model.encoder, scaler, f_log)
        df_val_patient = create_patient_level_features(val_items, cae_model.encoder, scaler, f_log)
        
        y_train = df_train_patient['label'].map(final_class_map)
        X_train = df_train_patient.drop(columns=['label'])
        y_val = df_val_patient['label'].map(final_class_map)
        X_val = df_val_patient.drop(columns=['label'])

        # --- STAGE 2: Train and Evaluate XGBoost Classifier ---
        log_message(f_log, "\n--- Training Patient-Level XGBoost Classifier ---", True)
        xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
        
        # --- FIX: Removed early_stopping_rounds from .fit() call ---
        xgb_model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val), (X_train, y_train)],
                      verbose=False)
        
        y_pred = xgb_model.predict(X_val)
        log_message(f_log, f"\n--- Fold {fold+1} Patient-Level Results ---")
        report_text = classification_report(y_val, y_pred, target_names=remaining_class_names, zero_division=0)
        log_message(f_log, report_text)
        
        report_dict = classification_report(y_val, y_pred, target_names=remaining_class_names, output_dict=True, zero_division=0)
        overall_metrics['accuracy'].append(report_dict['accuracy'])
        overall_metrics['precision'].append(report_dict['macro avg']['precision'])
        overall_metrics['recall'].append(report_dict['macro avg']['recall'])
        overall_metrics['f1-score'].append(report_dict['macro avg']['f1-score'])
        
        cm = confusion_matrix(y_val, y_pred, labels=range(num_classes)); aggregated_cm += cm
        plot_confusion_matrix(cm, remaining_class_names, f'Fold {fold+1} - Patient-Level Confusion Matrix', os.path.join(RESULTS_DIR, f'{EXP_NAME}_fold_{fold+1}_confusion_matrix.png'))
        plot_feature_importance(xgb_model, os.path.join(RESULTS_DIR, f'{EXP_NAME}_fold_{fold+1}_feature_importance.png'))
        plot_loss_curves(xgb_model.evals_result(), fold, os.path.join(RESULTS_DIR, f'{EXP_NAME}_fold_{fold+1}_loss_curve.png'))
        log_message(f_log, f"Plots for fold {fold+1} saved.")
        
        del cae_model, xgb_model, df_train_patient, df_val_patient
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Final Aggregated Results
    log_message(f_log, "\n" + "="*70 + "\nCross-Validation Summary (Patient-Level)\n" + "="*70)
    log_message(f_log, f"Average Accuracy: {np.mean(overall_metrics['accuracy']):.4f} ± {np.std(overall_metrics['accuracy']):.4f}")
    log_message(f_log, f"Average Macro F1-Score: {np.mean(overall_metrics['f1-score']):.4f} ± {np.std(overall_metrics['f1-score']):.4f}")
    log_message(f_log, f"Average Macro Precision: {np.mean(overall_metrics['precision']):.4f} ± {np.std(overall_metrics['precision']):.4f}")
    log_message(f_log, f"Average Macro Recall: {np.mean(overall_metrics['recall']):.4f} ± {np.std(overall_metrics['recall']):.4f}")
    
    plot_confusion_matrix(aggregated_cm, remaining_class_names, 'Aggregated Patient-Level CM (All Folds)', os.path.join(RESULTS_DIR, f'{EXP_NAME}_aggregated_confusion_matrix.png'))
    log_message(f_log, f"\nAggregated Patient-Level Confusion Matrix (sum over folds):\n{np.array2string(aggregated_cm.astype(int), separator=', ')}")
    
    log_message(f_log, "\n" + "="*70 + "\nExperiment Finished.\n" + "="*70)
    f_log.close()
    
    print(f"\nExperiment complete. All results saved in: {RESULTS_DIR}")
