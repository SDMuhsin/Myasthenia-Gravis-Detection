import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import warnings
import random

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.exceptions import ConvergenceWarning

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# --- Configuration ---
BASE_DIR = './data'
RESULTS_DIR = './results/EXP_12'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Class definitions (excluding TAO due to underrepresentation)
CLASS_DEFINITIONS = {
    'HC': {'path': 'Healthy control', 'label': 0},
    'MG': {'path': 'Definite MG', 'label': 1},
    'CNP3': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '3rd'), 'label': 2},
    'CNP4': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '4th'), 'label': 3},
    'CNP6': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '6th'), 'label': 4},
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
EXP_PREFIX = 'EXP_12_'
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds for reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

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
        
        for patient_folder_name in tqdm(patient_dirs, desc=f"  Loading {class_name_key}"):
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

# --- Advanced Data Augmentation ---
def augment_sequence(sequence, augmentation_factor=0.1):
    """Apply data augmentation to a sequence."""
    augmented = sequence.copy()
    
    # Add small amount of noise
    noise = np.random.normal(0, augmentation_factor * np.std(sequence), sequence.shape)
    augmented += noise
    
    # Random scaling
    scale_factor = np.random.uniform(0.95, 1.05)
    augmented *= scale_factor
    
    return augmented.astype(np.float32)

# --- Memory-Efficient Neural Network Architectures ---

class MemoryEfficientAttention(nn.Module):
    """Memory-efficient attention mechanism."""
    
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
            scores.masked_fill_(~mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        
        return self.out_proj(attn_output)

class AdvancedSaccadeClassifier(nn.Module):
    """Memory-efficient neural network for saccade classification - no attention."""
    
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Bidirectional LSTM layers (smaller)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0, 
                           bidirectional=True)
        
        # Simple 1D CNN for local patterns
        self.conv1d = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
        # Classification head (simpler)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for avg + max pooling
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, lengths):
        batch_size = x.size(0)
        
        # Input projection
        x = self.input_proj(x)
        
        # Pack sequences for LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM forward pass
        packed_output, (hidden, cell) = self.lstm(packed_input)
        
        # Unpack sequences
        lstm_output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Apply 1D convolution for local patterns
        # Transpose for conv1d: (batch, channels, seq_len)
        conv_input = lstm_output.transpose(1, 2)
        conv_output = F.relu(self.batch_norm(self.conv1d(conv_input)))
        conv_output = conv_output.transpose(1, 2)  # Back to (batch, seq_len, channels)
        
        # Create mask for pooling
        max_len = conv_output.size(1)
        mask = torch.arange(max_len, device=conv_output.device).expand(batch_size, max_len) < output_lengths.unsqueeze(1).to(conv_output.device)
        
        # Global pooling (average and max)
        # Average pooling (excluding padding)
        avg_pooled = torch.sum(conv_output * mask.unsqueeze(-1).float(), dim=1) / output_lengths.unsqueeze(-1).float().to(conv_output.device)
        
        # Max pooling (excluding padding)
        masked_output = conv_output.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        max_pooled, _ = torch.max(masked_output, dim=1)
        max_pooled = torch.where(torch.isinf(max_pooled), torch.zeros_like(max_pooled), max_pooled)
        
        # Concatenate pooled features
        pooled_features = torch.cat([avg_pooled, max_pooled], dim=1)
        
        # Classification
        output = self.classifier(pooled_features)
        return output

class SaccadeSequenceDataset(Dataset):
    """Custom PyTorch Dataset for saccade sequences with augmentation."""
    
    def __init__(self, sequences, labels, augment=False, augmentation_prob=0.5):
        self.sequences = sequences
        self.labels = labels
        self.augment = augment
        self.augmentation_prob = augmentation_prob

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx].copy()
        label = self.labels[idx]
        
        # Apply augmentation during training
        if self.augment and random.random() < self.augmentation_prob:
            sequence = augment_sequence(sequence)
        
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    """Pads sequences within a batch to equal length."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences])
    sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return sequences_padded, labels, lengths

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

def train_and_evaluate_advanced_nn(sequences, labels, n_features, n_classes, f_out, 
                                 model_name="Advanced_NN", epochs=30, batch_size=16, 
                                 learning_rate=0.001, hidden_dim=128):
    """Train and evaluate the advanced neural network."""
    f_out.write(f"\n--- Evaluating Advanced NN Model: {model_name} ---\n")
    f_out.write(f"Architecture: hidden_dim={hidden_dim}, epochs={epochs}, batch_size={batch_size}, lr={learning_rate}\n")
    print(f"\nEvaluating Advanced NN Model: {model_name}")
    print(f"Architecture: hidden_dim={hidden_dim}, epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    accuracies = []
    all_y_true, all_y_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(tqdm(skf.split(sequences, labels), total=5, desc=f"  CV for {model_name}")):
        X_train_seq = [sequences[i] for i in train_idx]
        X_test_seq = [sequences[i] for i in test_idx]
        y_train = np.array(labels)[train_idx]
        y_test = np.array(labels)[test_idx]
        
        # Create datasets with augmentation for training
        train_dataset = SaccadeSequenceDataset(X_train_seq, y_train, augment=True)
        test_dataset = SaccadeSequenceDataset(X_test_seq, y_test, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # Initialize model
        model = AdvancedSaccadeClassifier(input_dim=n_features, output_dim=n_classes, hidden_dim=hidden_dim).to(DEVICE)
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            for seq_padded, labels_batch, lengths in train_loader:
                seq_padded, labels_batch = seq_padded.to(DEVICE), labels_batch.to(DEVICE)
                lengths = lengths.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(seq_padded, lengths)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches
            scheduler.step(avg_epoch_loss)
            
            # Early stopping
            if avg_epoch_loss < best_val_loss:
                best_val_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:  # Early stopping
                    break

        # Evaluation
        model.eval()
        fold_preds, fold_true = [], []
        with torch.no_grad():
            for seq_padded, labels_batch, lengths in test_loader:
                seq_padded, labels_batch = seq_padded.to(DEVICE), labels_batch.to(DEVICE)
                lengths = lengths.to(DEVICE)
                outputs = model(seq_padded, lengths)
                _, predicted = torch.max(outputs.data, 1)
                fold_preds.extend(predicted.cpu().numpy())
                fold_true.extend(labels_batch.cpu().numpy())
        
        fold_accuracy = accuracy_score(fold_true, fold_preds)
        accuracies.append(fold_accuracy)
        all_y_true.extend(fold_true)
        all_y_pred.extend(fold_preds)
        
        f_out.write(f"  Fold {fold+1}: {fold_accuracy:.4f}\n")
        print(f"    Fold {fold+1}: {fold_accuracy:.4f}")
        
        # Clear GPU memory
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    f_out.write(f"Mean CV Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})\n")
    print(f"  Mean CV Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
    
    # Classification report
    report = classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, labels=CLASS_LABELS, zero_division=0)
    f_out.write("Classification Report:\n" + report + "\n")
    
    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred, labels=CLASS_LABELS)
    f_out.write("Confusion Matrix:\n" + np.array2string(cm) + "\n")
    plot_confusion_matrix(cm, CLASS_NAMES, model_name, RESULTS_DIR, f_out)
    
    return mean_accuracy

# --- Main Experiment Function ---
def run_experiment_12_iteration(raw_items_list, f_out, iteration=1):
    """Run a single iteration of experiment 12."""
    f_out.write(f"\n" + "="*80 + f"\nExperiment 12 - Iteration {iteration}: Advanced Neural Network\n" + "="*80 + "\n")
    print(f"\n" + "="*50 + f"\nRunning Experiment 12 - Iteration {iteration}\n" + "="*50)
    
    # Prepare data for neural network
    sequences = [item['data'] for item in raw_items_list]
    labels = [item['label'] for item in raw_items_list]
    n_features = sequences[0].shape[1]
    n_classes = len(CLASS_DEFINITIONS)
    
    f_out.write(f"Dataset: {len(sequences)} sequences, {n_features} features, {n_classes} classes\n")
    print(f"Dataset: {len(sequences)} sequences, {n_features} features, {n_classes} classes")
    
    results = {}
    
    # Iteration 1: Baseline advanced architecture
    if iteration == 1:
        f_out.write("\n--- Iteration 1: Baseline Advanced Architecture ---\n")
        print("--- Iteration 1: Baseline Advanced Architecture ---")
        
        acc = train_and_evaluate_advanced_nn(
            sequences, labels, n_features, n_classes, f_out,
            model_name="Advanced_NN_v1", epochs=25, batch_size=16, 
            learning_rate=0.001, hidden_dim=64
        )
        results["Advanced_NN_v1"] = acc
    
    return results

# --- Main Script Execution ---
if __name__ == '__main__':
    print("="*80)
    print("Starting Experiment 12: Advanced Neural Network for Myasthenia Gravis Detection")
    print(f"Using device: {DEVICE}")
    print(f"Target: Beat 60% accuracy with <10GB VRAM")
    print("="*80)
    
    summary_filepath = os.path.join(RESULTS_DIR, f'{EXP_PREFIX}summary.txt')
    
    with open(summary_filepath, 'w', encoding='utf-8') as f_report:
        f_report.write("="*80 + "\n")
        f_report.write("Experiment 12: Advanced Neural Network - Summary Report\n")
        f_report.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_report.write(f"Device used: {DEVICE}\n")
        f_report.write(f"Target classes: {CLASS_NAMES}\n")
        f_report.write(f"Target: Beat 60% accuracy with <10GB VRAM\n")
        f_report.write("="*80 + "\n")

        # Load raw data
        raw_items_list = load_raw_sequences_and_labels(
            BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, CSV_ENCODING, CSV_SEPARATOR, MIN_SEQ_LEN_THRESHOLD, f_report
        )
        
        if not raw_items_list:
            f_report.write("\nCRITICAL: No data loaded. Exiting.\n")
            print("\nCRITICAL: No data loaded. Exiting.")
            exit()
        
        # Run iteration 1
        results = run_experiment_12_iteration(raw_items_list, f_report, iteration=1)
        
        # Results summary
        f_report.write("\n" + "="*80 + "\nIteration 1 Results Summary\n" + "="*80 + "\n")
        print("\n" + "="*50 + "\nIteration 1 Results Summary\n" + "="*50)
        
        for model_name, accuracy in results.items():
            f_report.write(f"{model_name}: {accuracy:.4f}\n")
            print(f"{model_name}: {accuracy:.4f}")
            
            if accuracy > 0.60:
                f_report.write("🎉 SUCCESS: Achieved accuracy above 60% target!\n")
                print("🎉 SUCCESS: Achieved accuracy above 60% target!")
            else:
                f_report.write("⚠️  Did not exceed 60% target. Need further iterations.\n")
                print("⚠️  Did not exceed 60% target. Need further iterations.")
        
        f_report.write("\n" + "="*80 + "\nEnd of Iteration 1 Report\n" + "="*80 + "\n")

    print(f"\nIteration 1 complete! Summary saved to: {summary_filepath}")
    print(f"All results and plots saved in: {RESULTS_DIR}")
    print("="*80)
