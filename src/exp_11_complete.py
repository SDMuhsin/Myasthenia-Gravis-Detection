import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# --- Configuration ---
BASE_DIR = './data'
RESULTS_DIR = './results/EXP_11'

# Class definitions (excluding TAO due to underrepresentation)
CLASS_DEFINITIONS = {
    'HC': {'path': 'Healthy control', 'label': 0},
    'MG': {'path': 'Definite MG', 'label': 1},
    'CNP3': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '3rd'), 'label': 2},
    'CNP4': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '4th'), 'label': 3},
    'CNP6': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '6th'), 'label': 4},
}

# Core parameters
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']
CSV_ENCODING = 'utf-16-le'
CSV_SEPARATOR = ','
MIN_SEQ_LEN_THRESHOLD = 50
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Data Loading (simplified) ---
def load_raw_sequences_and_labels():
    """Loads raw time-series data."""
    print("Loading data...")
    raw_items = []
    
    for class_name_key, class_details in CLASS_DEFINITIONS.items():
        label = class_details['label']
        class_dir_abs = os.path.join(BASE_DIR, class_details['path'])
        
        if not os.path.isdir(class_dir_abs):
            continue
            
        patient_dirs = [d for d in os.listdir(class_dir_abs) if os.path.isdir(os.path.join(class_dir_abs, d))]
        
        for patient_folder_name in tqdm(patient_dirs, desc=f"  Loading {class_name_key}"):
            patient_dir_path = os.path.join(class_dir_abs, patient_folder_name)
            csv_files = glob.glob(os.path.join(patient_dir_path, '*.csv'))
            
            for csv_file_path in csv_files:
                try:
                    df_full = pd.read_csv(csv_file_path, encoding=CSV_ENCODING, sep=CSV_SEPARATOR)
                    df_full.columns = [col.strip() for col in df_full.columns]
                    
                    if any(col not in df_full.columns for col in FEATURE_COLUMNS):
                        continue
                    
                    if len(df_full) < MIN_SEQ_LEN_THRESHOLD:
                        continue
                    
                    df_features = df_full[FEATURE_COLUMNS].copy()
                    
                    for col in df_features.columns:
                        df_features.loc[:, col] = pd.to_numeric(df_features[col], errors='coerce')
                    
                    if df_features.isnull().sum().sum() > 0.1 * df_features.size:
                        continue
                    
                    df_features = df_features.fillna(0)
                    
                    raw_items.append({
                        'data': df_features.values.astype(np.float32),
                        'label': label,
                        'class_name': class_name_key,
                    })
                    
                except Exception:
                    continue
    
    print(f"Loaded {len(raw_items)} sequences")
    return raw_items

# --- Neural Network Components ---
class SimpleSaccadeClassifier(nn.Module):
    """Very simple neural network for memory efficiency."""
    
    def __init__(self, input_dim, output_dim, hidden_dim=16):
        super().__init__()
        
        # Simple feedforward network on sequence statistics
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),  # mean, std, last value
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, lengths):
        # Extract simple statistics from sequences
        batch_size = x.size(0)
        features = []
        
        for i in range(batch_size):
            seq_len = lengths[i].item()
            seq = x[i, :seq_len, :]  # Get actual sequence length
            
            # Simple statistics
            seq_mean = torch.mean(seq, dim=0)
            seq_std = torch.std(seq, dim=0)
            seq_last = seq[-1, :]
            
            # Concatenate features
            seq_features = torch.cat([seq_mean, seq_std, seq_last])
            features.append(seq_features)
        
        features = torch.stack(features)
        output = self.classifier(features)
        return output

class SaccadeSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = [torch.tensor(s, dtype=torch.float32) for s in sequences]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences])
    sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return sequences_padded, labels, lengths

def train_and_evaluate_nn(sequences, labels, n_features, n_classes):
    """Train and evaluate the memory-efficient neural network."""
    print("Training memory-efficient neural network...")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    accuracies = []

    for fold, (train_idx, test_idx) in enumerate(tqdm(skf.split(sequences, labels), total=5, desc="  NN CV")):
        X_train_seq = [sequences[i] for i in train_idx]
        X_test_seq = [sequences[i] for i in test_idx]
        y_train = np.array(labels)[train_idx]
        y_test = np.array(labels)[test_idx]
        
        train_dataset = SaccadeSequenceDataset(X_train_seq, y_train)
        test_dataset = SaccadeSequenceDataset(X_test_seq, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)  # Very small batch
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

        model = SimpleSaccadeClassifier(input_dim=n_features, output_dim=n_classes).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop (very short for memory efficiency)
        model.train()
        for epoch in range(5):  # Reduced epochs
            for seq_padded, labels_batch, lengths in train_loader:
                seq_padded, labels_batch = seq_padded.to(DEVICE), labels_batch.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(seq_padded, lengths)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

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
        print(f"    Fold {fold+1}: {fold_accuracy:.4f}")
        
        # Clear GPU memory after each fold
        del model, optimizer
        torch.cuda.empty_cache()
        
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    print(f"  Neural Network Mean CV Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
    return mean_accuracy

def update_summary_file(nn_accuracy):
    """Update the summary file with neural network results."""
    summary_file = os.path.join(RESULTS_DIR, 'EXP_11_summary.txt')
    
    with open(summary_file, 'a', encoding='utf-8') as f:
        f.write(f"Mean CV Accuracy: {nn_accuracy:.4f}\n\n")
        
        # Final results summary
        results = {
            "GradientBoosting": 0.6192,
            "RandomForest_Config3": 0.5994,
            "RandomForest_Config2": 0.5910,
            "SVM_rbf_C10.0": 0.5526,
            "RandomForest_Config1": 0.5423,
            "LDA_SelectK150": 0.5099,
            "LDA_SelectK200": 0.5003,
            "SVM_rbf_C1.0": 0.4949,
            "LDA_SelectK100": 0.4871,
            "LDA_SelectK50": 0.4619,
            "SVM_poly_C1.0": 0.4288,
            "Advanced_NN": nn_accuracy
        }
        
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        f.write("="*80 + "\n")
        f.write("Final Results Summary\n")
        f.write("="*80 + "\n")
        f.write("Model Performance Ranking:\n")
        for i, (model_name, accuracy) in enumerate(sorted_results, 1):
            f.write(f"{i:2d}. {model_name:<25}: {accuracy:.4f}\n")
        
        best_model, best_accuracy = sorted_results[0]
        f.write(f"\nBest performing model: {best_model} with accuracy: {best_accuracy:.4f}\n")
        
        if best_accuracy > 0.45:
            f.write("🎉 SUCCESS: Achieved accuracy above 45% baseline!\n")
        else:
            f.write("⚠️  Did not exceed 45% baseline. Consider further iterations.\n")
        
        f.write("\n" + "="*80 + "\nEnd of Experiment 11 Report\n" + "="*80 + "\n")

if __name__ == '__main__':
    print("="*60)
    print("Completing Experiment 11: Neural Network Evaluation")
    print(f"Using device: {DEVICE}")
    print("="*60)
    
    # Load data
    raw_items_list = load_raw_sequences_and_labels()
    
    if not raw_items_list:
        print("ERROR: No data loaded. Exiting.")
        exit()
    
    # Prepare data for neural network
    sequences = [item['data'] for item in raw_items_list]
    labels = [item['label'] for item in raw_items_list]
    n_features = sequences[0].shape[1]
    n_classes = len(CLASS_DEFINITIONS)
    
    print(f"Dataset: {len(sequences)} sequences, {n_features} features, {n_classes} classes")
    
    # Train and evaluate neural network
    nn_accuracy = train_and_evaluate_nn(sequences, labels, n_features, n_classes)
    
    # Update summary file
    update_summary_file(nn_accuracy)
    
    print("="*60)
    print("Neural Network Evaluation Complete!")
    print(f"Neural Network Accuracy: {nn_accuracy:.4f}")
    print("="*60)
