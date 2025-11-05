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
import argparse

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
def load_raw_sequences_and_labels(base_dir, class_definitions_dict, feature_columns_expected, encoding, separator, min_seq_len_threshold, f_out, time_subsample_factor=3):
    """Loads raw time-series data with temporal subsampling to reduce memory usage."""
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
        
        # Keep all patient directories - no patient subsampling
        for patient_folder_name in tqdm(patient_dirs, desc=f"  Loading {class_name_key}"):
            patient_dir_path = os.path.join(class_dir_abs, patient_folder_name)
            csv_files = glob.glob(os.path.join(patient_dir_path, '*.csv'))
            
            # Keep all CSV files per patient
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
                    
                    # Apply temporal subsampling - keep every Nth time point
                    sequence_data = df_features.values.astype(np.float32)
                    
                    # Subsample along time axis (every time_subsample_factor-th point)
                    subsampled_sequence = sequence_data[::time_subsample_factor]
                    
                    # Ensure minimum length after subsampling
                    if len(subsampled_sequence) < min_seq_len_threshold // time_subsample_factor:
                        continue
                    
                    raw_items.append({
                        'data': subsampled_sequence,
                        'label': label,
                        'patient_id': patient_folder_name,
                        'filename': os.path.basename(csv_file_path),
                        'class_name': class_name_key,
                        'original_length': len(sequence_data),
                        'subsampled_length': len(subsampled_sequence)
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

class CompactSaccadeClassifier(nn.Module):
    """Ultra-compact neural network for saccade classification with <6GB VRAM."""
    
    def __init__(self, input_dim, output_dim, hidden_dim=16, dropout=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Simple input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Single LSTM layer (bidirectional)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, 1, 
                           batch_first=True, bidirectional=True)
        
        # Simple attention for pooling
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Compact classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, lengths):
        batch_size = x.size(0)
        
        # Input projection
        x_proj = F.relu(self.input_proj(x))
        
        # Pack sequences for LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(x_proj, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM forward pass
        packed_output, _ = self.lstm(packed_input)
        
        # Unpack sequences
        lstm_output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Simple attention pooling
        attention_weights = self.attention(lstm_output)  # (batch, seq_len, 1)
        
        # Create mask for attention
        max_len = lstm_output.size(1)
        mask = torch.arange(max_len, device=lstm_output.device).expand(batch_size, max_len) < output_lengths.unsqueeze(1).to(lstm_output.device)
        
        # Apply mask to attention weights
        attention_weights = attention_weights.squeeze(-1)  # (batch, seq_len)
        attention_weights = attention_weights.masked_fill(~mask, float('-inf'))
        attention_weights = F.softmax(attention_weights, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        
        # Weighted sum of LSTM features
        pooled_features = torch.sum(lstm_output * attention_weights, dim=1)  # (batch, hidden_dim)
        
        # Classification
        output = self.classifier(pooled_features)
        return output

class SaccadeSequenceDataset(Dataset):
    """Custom PyTorch Dataset for saccade sequences with advanced augmentation."""
    
    def __init__(self, sequences, labels, augment=False, augmentation_prob=0.5, 
                 augmentation_func=None):
        self.sequences = sequences
        self.labels = labels
        self.augment = augment
        self.augmentation_prob = augmentation_prob
        self.augmentation_func = augmentation_func or augment_sequence

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx].copy()
        label = self.labels[idx]
        
        # Apply augmentation during training
        if self.augment and random.random() < self.augmentation_prob:
            sequence = self.augmentation_func(sequence)
        
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
                                 model_name="Advanced_NN", epochs=50, batch_size=16, 
                                 learning_rate=0.0005, hidden_dim=96, dropout=0.4,
                                 quick_check=False):
    """Train and evaluate the advanced neural network with enhanced techniques."""
    f_out.write(f"\n--- Evaluating Advanced NN Model: {model_name} ---\n")
    f_out.write(f"Architecture: hidden_dim={hidden_dim}, epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, dropout={dropout}\n")
    print(f"\nEvaluating Advanced NN Model: {model_name}")
    print(f"Architecture: hidden_dim={hidden_dim}, epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, dropout={dropout}")
    
    # More sophisticated class weight calculation
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    f_out.write(f"Class weights: {dict(zip(CLASS_NAMES, class_weights.cpu().numpy()))}\n")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    accuracies = []
    all_y_true, all_y_pred = [], []

    # If quick_check is True, only run the first fold
    fold_range = [next(skf.split(sequences, labels))] if quick_check else list(skf.split(sequences, labels))

    for fold, (train_idx, test_idx) in enumerate(tqdm(fold_range, total=1 if quick_check else 5, desc=f"  CV for {model_name}")):
        X_train_seq = [sequences[i] for i in train_idx]
        X_test_seq = [sequences[i] for i in test_idx]
        y_train = np.array(labels)[train_idx]
        y_test = np.array(labels)[test_idx]
        
        # Advanced data augmentation with more variations
        def advanced_augment_sequence(sequence, augmentation_factor=0.2):
            """Enhanced data augmentation with robust transformations."""
            augmented = sequence.copy()
            
            # Adaptive noise injection with feature-wise variation
            noise_intensity = np.random.uniform(0, augmentation_factor)
            feature_noise_scales = np.random.uniform(0.8, 1.2, size=sequence.shape[1])
            noise = np.random.normal(0, noise_intensity * np.std(sequence, axis=0) * feature_noise_scales, sequence.shape)
            augmented += noise
            
            # Advanced scaling with feature-specific scaling
            scale_factors = np.random.uniform(0.85, 1.15, size=sequence.shape[1])
            augmented *= scale_factors
            
            # More robust time warping
            if np.random.random() < 0.4:
                # Non-linear time warping with safety checks
                warp_type = np.random.choice(['compress', 'stretch'])
                warp_factor = np.random.uniform(0.7, 1.3)
                
                if warp_type == 'compress':
                    # Ensure minimum sequence length
                    min_length = max(10, int(len(augmented) * 0.5))
                    compressed = np.compress(np.random.uniform(0, 1, len(augmented)) < warp_factor, augmented, axis=0)
                    augmented = compressed if len(compressed) >= min_length else augmented
                else:
                    # Stretch by interpolation with safety
                    target_length = int(len(augmented) * warp_factor)
                    max_length = min(target_length, len(augmented) * 2)  # Prevent extreme stretching
                    
                    if 10 <= target_length <= max_length:
                        # Ensure 2D array for interpolation
                        if augmented.ndim == 1:
                            augmented = augmented.reshape(-1, 1)
                        
                        # Perform interpolation for each feature
                        stretched_features = []
                        for feature_col in range(augmented.shape[1]):
                            x_orig = np.linspace(0, 1, len(augmented))
                            x_warped = np.linspace(0, 1, target_length)
                            stretched_feature = np.interp(x_warped, x_orig, augmented[:, feature_col])
                            stretched_features.append(stretched_feature)
                        
                        augmented = np.column_stack(stretched_features)
            
            # Occasional sign flipping for robustness
            if np.random.random() < 0.2:
                flip_mask = np.random.random(sequence.shape[1]) < 0.5
                augmented[:, flip_mask] *= -1
            
            return augmented.astype(np.float32)
        
        # Create datasets with advanced augmentation
        train_dataset = SaccadeSequenceDataset(
            X_train_seq, y_train, 
            augment=True, 
            augmentation_prob=0.7,
            augmentation_func=advanced_augment_sequence
        )
        test_dataset = SaccadeSequenceDataset(X_test_seq, y_test, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # Initialize compact model
        model = CompactSaccadeClassifier(
            input_dim=n_features, 
            output_dim=n_classes, 
            hidden_dim=hidden_dim, 
            dropout=dropout
        ).to(DEVICE)
        
        # Optimizer with adaptive learning rate and weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.02,  # Increased regularization
            amsgrad=True  # Improved adaptive learning
        )
        
        # Advanced learning rate scheduler with warmup and cosine annealing
        warmup_epochs = max(1, epochs // 10)
        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                # Linear warmup
                return float(current_epoch) / float(max(1, warmup_epochs))
            # Cosine decay
            progress = float(current_epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        # Enhanced loss function with focal loss characteristics
        class_weights = class_weights.to(DEVICE)
        def focal_cross_entropy_loss(inputs, targets, alpha=class_weights, gamma=2.0):
            """Focal loss to handle class imbalance and hard example mining."""
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = (alpha[targets] * (1-pt)**gamma * ce_loss).mean()
            return focal_loss
        
        criterion = focal_cross_entropy_loss
        
        # Training loop with more sophisticated tracking
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
                
                # L2 regularization
                l2_lambda = 0.001
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss += l2_lambda * l2_norm
                
                loss.backward()
                
                # Gradient clipping with adaptive norm
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches
            scheduler.step()
            
            # Early stopping with more patience
            if avg_epoch_loss < best_val_loss:
                best_val_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:  # Increased patience
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
def extract_statistical_features(sequence):
    """Extract comprehensive and advanced statistical features from a sequence."""
    import scipy.stats
    import scipy.special
    
    features = []
    
    # Robust handling of potential input variations
    if sequence.ndim == 1:
        sequence = sequence.reshape(-1, 1)
    
    for channel in range(sequence.shape[1]):
        channel_data = sequence[:, channel]
        
        # Basic statistical moments with robust estimators
        features.extend([
            np.mean(channel_data),           # Central tendency
            np.median(channel_data),          # Robust central tendency
            np.std(channel_data),             # Dispersion
            np.var(channel_data),             # Variance
            scipy.stats.mstats.trimmed_mean(channel_data, proportiontocut=0.1),  # Trimmed mean
            scipy.stats.skew(channel_data),   # Asymmetry
            scipy.stats.kurtosis(channel_data)  # Tail heaviness
        ])
        
        # Robust distribution characteristics
        features.extend([
            np.percentile(channel_data, 10),   # Lower decile
            np.percentile(channel_data, 25),   # First quartile
            np.percentile(channel_data, 50),   # Median
            np.percentile(channel_data, 75),   # Third quartile
            np.percentile(channel_data, 90),   # Upper decile
            scipy.stats.iqr(channel_data),     # Interquartile range
            scipy.stats.mstats.winsorize(channel_data, limits=[0.1, 0.1]).mean()  # Winsorized mean
        ])
        
        # Advanced temporal features with robust error handling
        if len(channel_data) > 3:
            # Derivative-based features
            diff = np.diff(channel_data)
            diff2 = np.diff(diff)
            
            features.extend([
                np.mean(diff),                 # Average rate of change
                np.std(diff),                  # Variability of change
                np.mean(np.abs(diff)),         # Mean absolute change
                np.max(np.abs(diff)),          # Maximum absolute change
                
                # Higher-order derivative statistics
                np.mean(diff2),                # Average acceleration
                np.std(diff2),                 # Variability of acceleration
                
                # Trend and change point indicators
                np.sum(diff > 0) / len(diff),  # Proportion of increases
                np.sum(np.sign(diff2) != np.sign(diff[:-1])) / len(diff2),  # Change in trend direction
                
                # Advanced trend features
                scipy.stats.linregress(np.arange(len(diff)), diff)[0],  # Linear trend slope
                scipy.stats.spearmanr(np.arange(len(diff)), diff)[0]    # Rank correlation trend
            ])
        else:
            features.extend([0] * 10)  # Pad with zeros if insufficient data
        
        # Non-linear and information-theoretic transformations
        if len(channel_data) > 0:
            # Robust non-linear transformations
            abs_data = np.abs(channel_data) + 1e-8
            features.extend([
                np.log(abs_data.mean()),       # Log-transformed mean
                np.sqrt(abs_data.mean()),      # Root-transformed mean
                scipy.special.boxcox(abs_data, lmbda=0.5)[0],  # Box-Cox transformation
                
                # Information-theoretic features
                scipy.stats.entropy(abs_data / abs_data.sum()),  # Shannon entropy
                scipy.stats.variation(channel_data)  # Coefficient of variation
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Add frequency domain features
        if len(channel_data) > 10:
            fft_result = np.fft.fft(channel_data)
            fft_magnitude = np.abs(fft_result)
            
            features.extend([
                np.mean(fft_magnitude),    # Average frequency magnitude
                np.max(fft_magnitude),     # Dominant frequency magnitude
                np.argmax(fft_magnitude)   # Dominant frequency index
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
    
    return np.array(features, dtype=np.float32)

def extract_enhanced_statistical_features(sequence):
    """Enhanced statistical feature extraction with advanced techniques."""
    import scipy.stats
    import scipy.special
    
    # Robust error handling and preprocessing
    try:
        # Attempt advanced feature extraction
        features = extract_statistical_features(sequence)
        
        # Additional frequency domain features (if possible)
        if sequence.shape[0] > 10:  # Ensure sufficient data for FFT
            for channel in range(sequence.shape[1]):
                channel_data = sequence[:, channel]
                
                # Fast Fourier Transform features
                fft_result = np.fft.fft(channel_data)
                fft_magnitude = np.abs(fft_result)
                
                features.extend([
                    np.mean(fft_magnitude),    # Average frequency magnitude
                    np.max(fft_magnitude),     # Dominant frequency magnitude
                    np.argmax(fft_magnitude)   # Dominant frequency index
                ])
        else:
            features.extend([0, 0, 0])
        
        return features
    
    except Exception as e:
        # Fallback to basic feature extraction if advanced method fails
        print(f"Feature extraction error: {e}. Falling back to basic method.")
        return extract_statistical_features(sequence)

def run_experiment_12_iteration(raw_items_list, f_out, iteration=1, quick_check=False):
    """Run a single iteration of experiment 12."""
    f_out.write(f"\n" + "="*80 + f"\nExperiment 12 - Iteration {iteration}: Advanced Neural Network Approaches\n" + "="*80 + "\n")
    print(f"\n" + "="*50 + f"\nRunning Experiment 12 - Iteration {iteration}\n" + "="*50)
    
    # Prepare data using statistical features (proven to work better)
    statistical_features = []
    sequences = []
    labels = []
    
    f_out.write("Extracting enhanced statistical features from sequences...\n")
    print("Extracting enhanced statistical features from sequences...")
    
    for item in tqdm(raw_items_list, desc="Processing sequences"):
        raw_seq = item['data']
        
        # Extract enhanced statistical features
        stat_features = extract_enhanced_statistical_features(raw_seq)
        statistical_features.append(stat_features)
        
        # Use 10x subsampling for sequences as requested
        subsampled_seq = raw_seq[::10]  # 10x subsampling as requested
        if len(subsampled_seq) < 5:
            subsampled_seq = raw_seq[:5] if len(raw_seq) >= 5 else raw_seq
        
        # Normalize the subsampled sequence
        seq_mean = np.mean(subsampled_seq, axis=0, keepdims=True)
        seq_std = np.std(subsampled_seq, axis=0, keepdims=True) + 1e-8
        normalized_seq = (subsampled_seq - seq_mean) / seq_std
        
        sequences.append(normalized_seq)
        labels.append(item['label'])
    
    statistical_features = np.array(statistical_features)
    n_stat_features = statistical_features.shape[1]
    n_seq_features = sequences[0].shape[1]
    n_classes = len(CLASS_DEFINITIONS)
    
    f_out.write(f"Dataset: {len(sequences)} sequences\n")
    f_out.write(f"Enhanced statistical features: {n_stat_features} features\n")
    f_out.write(f"Sequence features: {n_seq_features} features, avg length: {np.mean([len(s) for s in sequences]):.1f}\n")
    f_out.write(f"Classes: {n_classes}\n")
    
    print(f"Dataset: {len(sequences)} sequences")
    print(f"Enhanced statistical features: {n_stat_features} features")
    print(f"Sequence features: {n_seq_features} features, avg length: {np.mean([len(s) for s in sequences]):.1f}")
    
    results = {}
    
    if iteration == 1:
        # Focus on neural network approaches to achieve 60%+ accuracy
        f_out.write("\n--- Iteration 1a: Enhanced Neural Network on Statistical Features ---\n")
        print("--- Iteration 1a: Enhanced Neural Network on Statistical Features ---")
        
        acc1 = train_enhanced_statistical_nn(statistical_features, labels, f_out, quick_check=quick_check)
        results["Enhanced_Statistical_NN"] = acc1
        
        # Second try: Deep neural network with better architecture
        f_out.write("\n--- Iteration 1b: Deep Neural Network with Advanced Features ---\n")
        print("--- Iteration 1b: Deep Neural Network with Advanced Features ---")
        
        acc2 = train_deep_statistical_nn(statistical_features, labels, f_out, quick_check=quick_check)
        results["Deep_Statistical_NN"] = acc2
        
        # Third try: Neural network ensemble if needed
        if max(acc1, acc2) < 0.60:
            f_out.write("\n--- Iteration 1c: Neural Network Ensemble ---\n")
            print("--- Iteration 1c: Neural Network Ensemble ---")
            
            acc3 = train_neural_ensemble(statistical_features, labels, f_out, quick_check=quick_check)
            results["Neural_Ensemble"] = acc3
    
    elif iteration == 2:
        # Advanced neural network approaches for iteration 2
        f_out.write("\n--- Iteration 2a: Ultra-Advanced Statistical NN ---\n")
        print("--- Iteration 2a: Ultra-Advanced Statistical NN ---")
        
        # Try ultra-advanced statistical NN with all techniques
        acc1 = train_ultra_advanced_statistical_nn(statistical_features, labels, f_out, quick_check=quick_check)
        results["Ultra_Advanced_Statistical_NN"] = acc1
        
        # Try advanced sequence-based model
        f_out.write("\n--- Iteration 2b: Advanced Sequence Model ---\n")
        print("--- Iteration 2b: Advanced Sequence Model ---")
        
        acc2 = train_advanced_sequence_model(sequences, labels, n_seq_features, n_classes, f_out, quick_check=quick_check)
        results["Advanced_Sequence_Model"] = acc2
        
        # Try hybrid statistical + sequence model
        f_out.write("\n--- Iteration 2c: Hybrid Statistical-Sequence Model ---\n")
        print("--- Iteration 2c: Hybrid Statistical-Sequence Model ---")
        
        acc3 = train_hybrid_statistical_sequence_model(statistical_features, sequences, labels, f_out, quick_check=quick_check)
        results["Hybrid_Statistical_Sequence"] = acc3
    
    elif iteration == 3:
        # Ultra-advanced approaches for iteration 3
        f_out.write("\n--- Iteration 3: Ultra-Advanced Neural Network Approaches ---\n")
        print("--- Iteration 3: Ultra-Advanced Neural Network Approaches ---")
        
        # Try gradient boosting + neural network hybrid
        acc1 = train_gradient_boosting_nn_hybrid(statistical_features, labels, f_out, quick_check=quick_check)
        results["GradientBoosting_NN_Hybrid"] = acc1
        
        # Try meta-learning ensemble
        acc2 = train_meta_learning_ensemble(statistical_features, sequences, labels, f_out, quick_check=quick_check)
        results["Meta_Learning_Ensemble"] = acc2
        
        # Try ultra-advanced feature engineering + NN
        acc3 = train_ultra_advanced_nn(statistical_features, sequences, labels, f_out, quick_check=quick_check)
        results["Ultra_Advanced_NN"] = acc3
    
    elif iteration == 4:
        # Revolutionary approaches for iteration 4 - focus on breakthrough techniques
        f_out.write("\n--- Iteration 4: Revolutionary Neural Network Approaches ---\n")
        print("--- Iteration 4: Revolutionary Neural Network Approaches ---")
        
        # Try advanced data preprocessing + neural network
        acc1 = train_revolutionary_preprocessing_nn(statistical_features, sequences, labels, f_out, quick_check=quick_check)
        results["Revolutionary_Preprocessing_NN"] = acc1
        
        # Try advanced ensemble with sophisticated voting
        acc2 = train_sophisticated_ensemble(statistical_features, sequences, labels, f_out, quick_check=quick_check)
        results["Sophisticated_Ensemble"] = acc2
        
        # Try breakthrough neural architecture
        acc3 = train_breakthrough_neural_architecture(statistical_features, sequences, labels, f_out, quick_check=quick_check)
        results["Breakthrough_Neural_Architecture"] = acc3
    
    return results

def train_ensemble_model(sequences, labels, n_features, n_classes, f_out, quick_check=False):
    """Train an ensemble of models for better performance."""
    f_out.write(f"\n--- Training Ensemble Model ---\n")
    print(f"\nTraining Ensemble Model")
    
    # Train multiple models with different configurations
    model_configs = [
        {"hidden_dim": 32, "epochs": 40, "lr": 0.001, "batch_size": 8},
        {"hidden_dim": 48, "epochs": 50, "lr": 0.0008, "batch_size": 6},
        {"hidden_dim": 64, "epochs": 35, "lr": 0.0012, "batch_size": 4}
    ]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_range = [next(skf.split(sequences, labels))] if quick_check else list(skf.split(sequences, labels))
    
    ensemble_accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(tqdm(fold_range, total=1 if quick_check else 5, desc="  Ensemble CV")):
        X_train_seq = [sequences[i] for i in train_idx]
        X_test_seq = [sequences[i] for i in test_idx]
        y_train = np.array(labels)[train_idx]
        y_test = np.array(labels)[test_idx]
        
        # Train multiple models
        models = []
        for config in model_configs:
            # Create datasets
            train_dataset = SaccadeSequenceDataset(X_train_seq, y_train, augment=True, augmentation_prob=0.5)
            train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
            
            # Initialize model
            model = CompactSaccadeClassifier(
                input_dim=n_features, 
                output_dim=n_classes, 
                hidden_dim=config["hidden_dim"], 
                dropout=0.3
            ).to(DEVICE)
            
            # Train model
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(config["epochs"]):
                for seq_padded, labels_batch, lengths in train_loader:
                    seq_padded, labels_batch = seq_padded.to(DEVICE), labels_batch.to(DEVICE)
                    lengths = lengths.to(DEVICE)
                    
                    optimizer.zero_grad()
                    outputs = model(seq_padded, lengths)
                    loss = criterion(outputs, labels_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            
            models.append(model)
        
        # Ensemble prediction
        test_dataset = SaccadeSequenceDataset(X_test_seq, y_test, augment=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
        
        ensemble_preds = []
        fold_true = []
        
        with torch.no_grad():
            for seq_padded, labels_batch, lengths in test_loader:
                seq_padded, labels_batch = seq_padded.to(DEVICE), labels_batch.to(DEVICE)
                lengths = lengths.to(DEVICE)
                
                # Get predictions from all models
                model_outputs = []
                for model in models:
                    model.eval()
                    outputs = model(seq_padded, lengths)
                    model_outputs.append(F.softmax(outputs, dim=1))
                
                # Average predictions
                ensemble_output = torch.mean(torch.stack(model_outputs), dim=0)
                _, predicted = torch.max(ensemble_output, 1)
                
                ensemble_preds.extend(predicted.cpu().numpy())
                fold_true.extend(labels_batch.cpu().numpy())
        
        fold_accuracy = accuracy_score(fold_true, ensemble_preds)
        ensemble_accuracies.append(fold_accuracy)
        
        f_out.write(f"  Ensemble Fold {fold+1}: {fold_accuracy:.4f}\n")
        print(f"    Ensemble Fold {fold+1}: {fold_accuracy:.4f}")
        
        # Clear memory
        for model in models:
            del model
        torch.cuda.empty_cache()
    
    mean_accuracy = np.mean(ensemble_accuracies)
    f_out.write(f"Ensemble Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(ensemble_accuracies):.4f})\n")
    print(f"  Ensemble Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(ensemble_accuracies):.4f})")
    
    return mean_accuracy

def train_statistical_model(features, labels, f_out, quick_check=False):
    """Train LDA on statistical features (replicating successful Experiment 1)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    
    f_out.write("Training Linear Discriminant Analysis on statistical features...\n")
    print("Training LDA on statistical features...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_range = [next(skf.split(features, labels))] if quick_check else list(skf.split(features, labels))
    
    accuracies = []
    all_y_true, all_y_pred = [], []
    
    for fold, (train_idx, test_idx) in enumerate(tqdm(fold_range, total=1 if quick_check else 5, desc="  LDA CV")):
        X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
        y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
        
        # Train LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        
        # Predict
        y_pred = lda.predict(X_test)
        
        fold_accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(fold_accuracy)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        f_out.write(f"  LDA Fold {fold+1}: {fold_accuracy:.4f}\n")
        print(f"    LDA Fold {fold+1}: {fold_accuracy:.4f}")
    
    mean_accuracy = np.mean(accuracies)
    f_out.write(f"LDA Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})\n")
    print(f"  LDA Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})")
    
    # Classification report and confusion matrix
    report = classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, labels=CLASS_LABELS, zero_division=0)
    f_out.write("LDA Classification Report:\n" + report + "\n")
    
    cm = confusion_matrix(all_y_true, all_y_pred, labels=CLASS_LABELS)
    plot_confusion_matrix(cm, CLASS_NAMES, "Statistical_LDA", RESULTS_DIR, f_out)
    
    return mean_accuracy

def train_statistical_nn(features, labels, f_out, quick_check=False):
    """Train compact neural network on statistical features."""
    f_out.write("Training Neural Network on statistical features...\n")
    print("Training NN on statistical features...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_range = [next(skf.split(features, labels))] if quick_check else list(skf.split(features, labels))
    
    accuracies = []
    all_y_true, all_y_pred = [], []
    
    for fold, (train_idx, test_idx) in enumerate(tqdm(fold_range, total=1 if quick_check else 5, desc="  Statistical NN CV")):
        X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
        y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
        y_test_tensor = torch.LongTensor(y_test).to(DEVICE)
        
        # Simple feedforward network
        model = nn.Sequential(
            nn.Linear(features_scaled.shape[1], 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, len(CLASS_DEFINITIONS))
        ).to(DEVICE)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Training
        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            y_pred = predicted.cpu().numpy()
        
        fold_accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(fold_accuracy)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        f_out.write(f"  Statistical NN Fold {fold+1}: {fold_accuracy:.4f}\n")
        print(f"    Statistical NN Fold {fold+1}: {fold_accuracy:.4f}")
        
        del model, optimizer
        torch.cuda.empty_cache()
    
    mean_accuracy = np.mean(accuracies)
    f_out.write(f"Statistical NN Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})\n")
    print(f"  Statistical NN Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})")
    
    # Classification report and confusion matrix
    report = classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, labels=CLASS_LABELS, zero_division=0)
    f_out.write("Statistical NN Classification Report:\n" + report + "\n")
    
    cm = confusion_matrix(all_y_true, all_y_pred, labels=CLASS_LABELS)
    plot_confusion_matrix(cm, CLASS_NAMES, "Statistical_NN", RESULTS_DIR, f_out)
    
    return mean_accuracy

def train_hybrid_model(stat_features, sequences, labels, f_out, quick_check=False):
    """Train hybrid model combining statistical and sequence features."""
    f_out.write("Training Hybrid Statistical-Sequence Model...\n")
    print("Training Hybrid Model...")
    
    # Standardize statistical features
    scaler = StandardScaler()
    stat_features_scaled = scaler.fit_transform(stat_features)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_range = [next(skf.split(sequences, labels))] if quick_check else list(skf.split(sequences, labels))
    
    accuracies = []
    all_y_true, all_y_pred = [], []
    
    for fold, (train_idx, test_idx) in enumerate(tqdm(fold_range, total=1 if quick_check else 5, desc="  Hybrid CV")):
        # Statistical features
        X_stat_train, X_stat_test = stat_features_scaled[train_idx], stat_features_scaled[test_idx]
        
        # Sequence features
        X_seq_train = [sequences[i] for i in train_idx]
        X_seq_test = [sequences[i] for i in test_idx]
        
        y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
        
        # Create hybrid model
        class HybridModel(nn.Module):
            def __init__(self, stat_dim, seq_dim, hidden_dim=64):
                super().__init__()
                # Statistical branch
                self.stat_branch = nn.Sequential(
                    nn.Linear(stat_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                )
                
                # Sequence branch (simple)
                self.seq_branch = nn.Sequential(
                    nn.Linear(seq_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                )
                
                # Fusion
                self.fusion = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, len(CLASS_DEFINITIONS))
                )
            
            def forward(self, stat_features, seq_features):
                stat_out = self.stat_branch(stat_features)
                
                # Simple sequence aggregation (mean pooling)
                seq_mean = torch.mean(seq_features, dim=1)
                seq_out = self.seq_branch(seq_mean)
                
                # Combine
                combined = torch.cat([stat_out, seq_out], dim=1)
                return self.fusion(combined)
        
        model = HybridModel(stat_dim=X_stat_train.shape[1], seq_dim=sequences[0].shape[1]).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Convert to tensors
        X_stat_train_tensor = torch.FloatTensor(X_stat_train).to(DEVICE)
        X_stat_test_tensor = torch.FloatTensor(X_stat_test).to(DEVICE)
        y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
        
        # Pad sequences for batch processing
        max_len = max(len(seq) for seq in X_seq_train)
        X_seq_train_padded = np.zeros((len(X_seq_train), max_len, sequences[0].shape[1]))
        for i, seq in enumerate(X_seq_train):
            X_seq_train_padded[i, :len(seq)] = seq
        X_seq_train_tensor = torch.FloatTensor(X_seq_train_padded).to(DEVICE)
        
        max_len_test = max(len(seq) for seq in X_seq_test)
        X_seq_test_padded = np.zeros((len(X_seq_test), max_len_test, sequences[0].shape[1]))
        for i, seq in enumerate(X_seq_test):
            X_seq_test_padded[i, :len(seq)] = seq
        X_seq_test_tensor = torch.FloatTensor(X_seq_test_padded).to(DEVICE)
        
        # Training
        model.train()
        for epoch in range(30):
            optimizer.zero_grad()
            outputs = model(X_stat_train_tensor, X_seq_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_stat_test_tensor, X_seq_test_tensor)
            _, predicted = torch.max(outputs, 1)
            y_pred = predicted.cpu().numpy()
        
        fold_accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(fold_accuracy)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        f_out.write(f"  Hybrid Fold {fold+1}: {fold_accuracy:.4f}\n")
        print(f"    Hybrid Fold {fold+1}: {fold_accuracy:.4f}")
        
        del model, optimizer
        torch.cuda.empty_cache()
    
    mean_accuracy = np.mean(accuracies)
    f_out.write(f"Hybrid Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})\n")
    print(f"  Hybrid Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})")
    
    # Classification report and confusion matrix
    report = classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, labels=CLASS_LABELS, zero_division=0)
    f_out.write("Hybrid Classification Report:\n" + report + "\n")
    
    cm = confusion_matrix(all_y_true, all_y_pred, labels=CLASS_LABELS)
    plot_confusion_matrix(cm, CLASS_NAMES, "Hybrid_Model", RESULTS_DIR, f_out)
    
    return mean_accuracy

def train_advanced_ensemble(stat_features, sequences, labels, f_out, quick_check=False):
    """Train advanced ensemble combining multiple approaches."""
    f_out.write("Training Advanced Ensemble Model...\n")
    print("Training Advanced Ensemble...")
    
    # This will combine LDA, Statistical NN, and simple voting
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier
    
    scaler = StandardScaler()
    stat_features_scaled = scaler.fit_transform(stat_features)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_range = [next(skf.split(sequences, labels))] if quick_check else list(skf.split(sequences, labels))
    
    accuracies = []
    all_y_true, all_y_pred = [], []
    
    for fold, (train_idx, test_idx) in enumerate(tqdm(fold_range, total=1 if quick_check else 5, desc="  Advanced Ensemble CV")):
        X_stat_train, X_stat_test = stat_features_scaled[train_idx], stat_features_scaled[test_idx]
        y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
        
        # Model 1: LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_stat_train, y_train)
        lda_pred = lda.predict(X_stat_test)
        
        # Model 2: Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        rf.fit(X_stat_train, y_train)
        rf_pred = rf.predict(X_stat_test)
        
        # Model 3: Simple NN
        X_train_tensor = torch.FloatTensor(X_stat_train).to(DEVICE)
        X_test_tensor = torch.FloatTensor(X_stat_test).to(DEVICE)
        y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
        
        nn_model = nn.Sequential(
            nn.Linear(X_stat_train.shape[1], 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, len(CLASS_DEFINITIONS))
        ).to(DEVICE)
        
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Train NN
        nn_model.train()
        for epoch in range(30):
            optimizer.zero_grad()
            outputs = nn_model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
        
        # NN prediction
        nn_model.eval()
        with torch.no_grad():
            outputs = nn_model(X_test_tensor)
            _, nn_pred = torch.max(outputs, 1)
            nn_pred = nn_pred.cpu().numpy()
        
        # Ensemble voting
        ensemble_pred = []
        for i in range(len(y_test)):
            votes = [lda_pred[i], rf_pred[i], nn_pred[i]]
            # Simple majority voting
            ensemble_pred.append(max(set(votes), key=votes.count))
        
        fold_accuracy = accuracy_score(y_test, ensemble_pred)
        accuracies.append(fold_accuracy)
        all_y_true.extend(y_test)
        all_y_pred.extend(ensemble_pred)
        
        f_out.write(f"  Advanced Ensemble Fold {fold+1}: {fold_accuracy:.4f}\n")
        print(f"    Advanced Ensemble Fold {fold+1}: {fold_accuracy:.4f}")
        
        del nn_model, optimizer
        torch.cuda.empty_cache()
    
    mean_accuracy = np.mean(accuracies)
    f_out.write(f"Advanced Ensemble Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})\n")
    print(f"  Advanced Ensemble Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})")
    
    # Classification report and confusion matrix
    report = classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, labels=CLASS_LABELS, zero_division=0)
    f_out.write("Advanced Ensemble Classification Report:\n" + report + "\n")
    
    cm = confusion_matrix(all_y_true, all_y_pred, labels=CLASS_LABELS)
    plot_confusion_matrix(cm, CLASS_NAMES, "Advanced_Ensemble", RESULTS_DIR, f_out)
    
    return mean_accuracy

def train_enhanced_statistical_nn(features, labels, f_out, quick_check=False):
    """Train improved neural network with better architecture and training."""
    f_out.write("Training Improved Neural Network on statistical features...\n")
    print("Training Improved NN on statistical features...")
    
    # Better preprocessing
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_range = [next(skf.split(features, labels))] if quick_check else list(skf.split(features, labels))
    
    accuracies = []
    all_y_true, all_y_pred = [], []
    
    for fold, (train_idx, test_idx) in enumerate(tqdm(fold_range, total=1 if quick_check else 5, desc="  Improved NN CV")):
        X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
        y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
        
        # Improved neural network with batch normalization and better architecture
        class ImprovedNN(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    
                    nn.Linear(256, 192),
                    nn.BatchNorm1d(192),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    
                    nn.Linear(192, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    
                    nn.Linear(64, output_dim)
                )
                
                # Initialize weights properly
                self.apply(self._init_weights)
            
            def _init_weights(self, module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
            
            def forward(self, x):
                return self.network(x)
        
        model = ImprovedNN(features_scaled.shape[1], len(CLASS_DEFINITIONS)).to(DEVICE)
        
        # Better optimizer with learning rate scheduling
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # Focal loss for better class imbalance handling
        class FocalLoss(nn.Module):
            def __init__(self, alpha=None, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                
            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
                pt = torch.exp(-ce_loss)
                focal_loss = (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
        
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        
        # Training with validation split and early stopping
        val_size = int(0.2 * len(X_train))
        train_size = len(X_train) - val_size
        
        train_data, val_data = torch.utils.data.random_split(
            torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
            [train_size, val_size],
            generator=torch.Generator().manual_seed(RANDOM_STATE)
        )
        
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(200):
            # Training
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    break
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            y_pred = predicted.cpu().numpy()
        
        fold_accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(fold_accuracy)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        f_out.write(f"  Improved NN Fold {fold+1}: {fold_accuracy:.4f}\n")
        print(f"    Improved NN Fold {fold+1}: {fold_accuracy:.4f}")
        
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
    
    mean_accuracy = np.mean(accuracies)
    f_out.write(f"Improved NN Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})\n")
    print(f"  Improved NN Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})")
    
    # Classification report and confusion matrix
    report = classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, labels=CLASS_LABELS, zero_division=0)
    f_out.write("Improved NN Classification Report:\n" + report + "\n")
    
    cm = confusion_matrix(all_y_true, all_y_pred, labels=CLASS_LABELS)
    plot_confusion_matrix(cm, CLASS_NAMES, "Enhanced_Statistical_NN", RESULTS_DIR, f_out)
    
    return mean_accuracy

def train_deep_statistical_nn(features, labels, f_out, quick_check=False):
    """Train deep neural network with advanced architecture."""
    f_out.write("Training Deep Neural Network on statistical features...\n")
    print("Training Deep NN on statistical features...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_range = [next(skf.split(features, labels))] if quick_check else list(skf.split(features, labels))
    
    accuracies = []
    all_y_true, all_y_pred = [], []
    
    for fold, (train_idx, test_idx) in enumerate(tqdm(fold_range, total=1 if quick_check else 5, desc="  Deep NN CV")):
        X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
        y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
        
        # Deep neural network with residual connections
        class DeepNN(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.input_layer = nn.Linear(input_dim, 512)
                self.input_bn = nn.BatchNorm1d(512)
                
                # Residual blocks
                self.block1 = self._make_block(512, 256)
                self.block2 = self._make_block(256, 128)
                self.block3 = self._make_block(128, 64)
                
                # Output layer
                self.output_layer = nn.Linear(64, output_dim)
                self.dropout = nn.Dropout(0.5)
                
                # Initialize weights
                self.apply(self._init_weights)
            
            def _make_block(self, in_dim, out_dim):
                return nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(out_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
            
            def _init_weights(self, module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
            
            def forward(self, x):
                x = F.relu(self.input_bn(self.input_layer(x)))
                
                # Residual blocks with skip connections
                x1 = self.block1(x)
                x1 = F.adaptive_avg_pool1d(x1.unsqueeze(1), 256).squeeze(1)  # Downsample for skip
                
                x2 = self.block2(x1)
                x2 = F.adaptive_avg_pool1d(x2.unsqueeze(1), 128).squeeze(1)  # Downsample for skip
                
                x3 = self.block3(x2)
                
                x = self.dropout(x3)
                return self.output_layer(x)
        
        model = DeepNN(features_scaled.shape[1], len(CLASS_DEFINITIONS)).to(DEVICE)
        
        # Advanced optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.02, amsgrad=True)
        
        # Focal loss for class imbalance
        class FocalLoss(nn.Module):
            def __init__(self, alpha=None, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                
            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
                pt = torch.exp(-ce_loss)
                focal_loss = (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
        
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        
        # Training with validation split
        val_size = int(0.2 * len(X_train))
        train_size = len(X_train) - val_size
        
        train_data, val_data = torch.utils.data.random_split(
            torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
            [train_size, val_size]
        )
        
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(80):
            # Training
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                # L2 regularization
                l2_lambda = 0.001
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss += l2_lambda * l2_norm
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    _, predicted = torch.max(outputs, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_acc = val_correct / val_total
            scheduler.step()
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    break
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            y_pred = predicted.cpu().numpy()
        
        fold_accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(fold_accuracy)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        f_out.write(f"  Deep NN Fold {fold+1}: {fold_accuracy:.4f}\n")
        print(f"    Deep NN Fold {fold+1}: {fold_accuracy:.4f}")
        
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
    
    mean_accuracy = np.mean(accuracies)
    f_out.write(f"Deep NN Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})\n")
    print(f"  Deep NN Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})")
    
    # Classification report and confusion matrix
    report = classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, labels=CLASS_LABELS, zero_division=0)
    f_out.write("Deep NN Classification Report:\n" + report + "\n")
    
    cm = confusion_matrix(all_y_true, all_y_pred, labels=CLASS_LABELS)
    plot_confusion_matrix(cm, CLASS_NAMES, "Deep_Statistical_NN", RESULTS_DIR, f_out)
    
    return mean_accuracy

def train_neural_ensemble(features, labels, f_out, quick_check=False):
    """Train ensemble of neural networks."""
    f_out.write("Training Neural Network Ensemble...\n")
    print("Training Neural Network Ensemble...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_range = [next(skf.split(features, labels))] if quick_check else list(skf.split(features, labels))
    
    accuracies = []
    all_y_true, all_y_pred = [], []
    
    for fold, (train_idx, test_idx) in enumerate(tqdm(fold_range, total=1 if quick_check else 5, desc="  Neural Ensemble CV")):
        X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
        y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
        
        # Define different neural network architectures
        models = []
        
        # Model 1: Wide network
        model1 = nn.Sequential(
            nn.Linear(features_scaled.shape[1], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(CLASS_DEFINITIONS))
        ).to(DEVICE)
        
        # Model 2: Deep network
        model2 = nn.Sequential(
            nn.Linear(features_scaled.shape[1], 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, len(CLASS_DEFINITIONS))
        ).to(DEVICE)
        
        # Model 3: Compact network
        model3 = nn.Sequential(
            nn.Linear(features_scaled.shape[1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, len(CLASS_DEFINITIONS))
        ).to(DEVICE)
        
        models = [model1, model2, model3]
        optimizers = [torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01) for model in models]
        
        # Class weights
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Train each model
        for model, optimizer in zip(models, optimizers):
            model.train()
            for epoch in range(60):
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        # Ensemble prediction
        ensemble_outputs = []
        for model in models:
            model.eval()
            with torch.no_grad():
                outputs = model(X_test_tensor)
                ensemble_outputs.append(F.softmax(outputs, dim=1))
        
        # Average ensemble
        avg_output = torch.mean(torch.stack(ensemble_outputs), dim=0)
        _, predicted = torch.max(avg_output, 1)
        y_pred = predicted.cpu().numpy()
        
        fold_accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(fold_accuracy)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        f_out.write(f"  Neural Ensemble Fold {fold+1}: {fold_accuracy:.4f}\n")
        print(f"    Neural Ensemble Fold {fold+1}: {fold_accuracy:.4f}")
        
        # Clear memory
        for model, optimizer in zip(models, optimizers):
            del model, optimizer
        torch.cuda.empty_cache()
    
    mean_accuracy = np.mean(accuracies)
    f_out.write(f"Neural Ensemble Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})\n")
    print(f"  Neural Ensemble Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})")
    
    # Classification report and confusion matrix
    report = classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, labels=CLASS_LABELS, zero_division=0)
    f_out.write("Neural Ensemble Classification Report:\n" + report + "\n")
    
    cm = confusion_matrix(all_y_true, all_y_pred, labels=CLASS_LABELS)
    plot_confusion_matrix(cm, CLASS_NAMES, "Neural_Ensemble", RESULTS_DIR, f_out)
    
    return mean_accuracy

def train_transformer_nn(stat_features, sequences, labels, f_out, quick_check=False):
    """Train transformer-based neural network."""
    f_out.write("Training Transformer Neural Network...\n")
    print("Training Transformer NN...")
    
    # For now, use statistical features with transformer-like attention
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(stat_features)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_range = [next(skf.split(features_scaled, labels))] if quick_check else list(skf.split(features_scaled, labels))
    
    accuracies = []
    all_y_true, all_y_pred = [], []
    
    for fold, (train_idx, test_idx) in enumerate(tqdm(fold_range, total=1 if quick_check else 5, desc="  Transformer NN CV")):
        X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
        y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
        
        # Transformer-inspired architecture
        class TransformerNN(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.input_proj = nn.Linear(input_dim, 256)
                
                # Multi-head attention (simplified)
                self.attention = nn.MultiheadAttention(256, num_heads=8, batch_first=True)
                self.norm1 = nn.LayerNorm(256)
                
                # Feed forward
                self.ff = nn.Sequential(
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256)
                )
                self.norm2 = nn.LayerNorm(256)
                
                # Output
                self.output = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(128, output_dim)
                )
            
            def forward(self, x):
                # Project input
                x = self.input_proj(x).unsqueeze(1)  # Add sequence dimension
                
                # Self-attention
                attn_out, _ = self.attention(x, x, x)
                x = self.norm1(x + attn_out)
                
                # Feed forward
                ff_out = self.ff(x)
                x = self.norm2(x + ff_out)
                
                # Output
                x = x.squeeze(1)  # Remove sequence dimension
                return self.output(x)
        
        model = TransformerNN(features_scaled.shape[1], len(CLASS_DEFINITIONS)).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
        
        # Class weights
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training
        model.train()
        for epoch in range(70):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            y_pred = predicted.cpu().numpy()
        
        fold_accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(fold_accuracy)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        f_out.write(f"  Transformer NN Fold {fold+1}: {fold_accuracy:.4f}\n")
        print(f"    Transformer NN Fold {fold+1}: {fold_accuracy:.4f}")
        
        del model, optimizer
        torch.cuda.empty_cache()
    
    mean_accuracy = np.mean(accuracies)
    f_out.write(f"Transformer NN Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})\n")
    print(f"  Transformer NN Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})")
    
    # Classification report and confusion matrix
    report = classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, labels=CLASS_LABELS, zero_division=0)
    f_out.write("Transformer NN Classification Report:\n" + report + "\n")
    
    cm = confusion_matrix(all_y_true, all_y_pred, labels=CLASS_LABELS)
    plot_confusion_matrix(cm, CLASS_NAMES, "Transformer_NN", RESULTS_DIR, f_out)
    
    return mean_accuracy

def train_attention_cnn(stat_features, sequences, labels, f_out, quick_check=False):
    """Train attention-based CNN."""
    f_out.write("Training Attention-based CNN...\n")
    print("Training Attention CNN...")
    
    # Use statistical features for now
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(stat_features)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_range = [next(skf.split(features_scaled, labels))] if quick_check else list(skf.split(features_scaled, labels))
    
    accuracies = []
    all_y_true, all_y_pred = [], []
    
    for fold, (train_idx, test_idx) in enumerate(tqdm(fold_range, total=1 if quick_check else 5, desc="  Attention CNN CV")):
        X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
        y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
        
        # Attention CNN architecture
        class AttentionCNN(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                # Reshape input for 1D CNN
                self.input_dim = input_dim
                
                # 1D CNN layers
                self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
                self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
                
                # Attention mechanism
                self.attention = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.Tanh(),
                    nn.Linear(128, 1)
                )
                
                # Classification head
                self.classifier = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, output_dim)
                )
            
            def forward(self, x):
                # Reshape for 1D CNN
                x = x.unsqueeze(1)  # (batch, 1, features)
                
                # CNN layers
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                
                # Transpose for attention
                x = x.transpose(1, 2)  # (batch, features, channels)
                
                # Attention weights
                attn_weights = self.attention(x)  # (batch, features, 1)
                attn_weights = F.softmax(attn_weights, dim=1)
                
                # Apply attention
                x = torch.sum(x * attn_weights, dim=1)  # (batch, channels)
                
                # Classification
                return self.classifier(x)
        
        model = AttentionCNN(features_scaled.shape[1], len(CLASS_DEFINITIONS)).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Class weights
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training
        model.train()
        for epoch in range(60):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            y_pred = predicted.cpu().numpy()
        
        fold_accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(fold_accuracy)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        f_out.write(f"  Attention CNN Fold {fold+1}: {fold_accuracy:.4f}\n")
        print(f"    Attention CNN Fold {fold+1}: {fold_accuracy:.4f}")
        
        del model, optimizer
        torch.cuda.empty_cache()
    
    mean_accuracy = np.mean(accuracies)
    f_out.write(f"Attention CNN Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})\n")
    print(f"  Attention CNN Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})")
    
    # Classification report and confusion matrix
    report = classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, labels=CLASS_LABELS, zero_division=0)
    f_out.write("Attention CNN Classification Report:\n" + report + "\n")
    
    cm = confusion_matrix(all_y_true, all_y_pred, labels=CLASS_LABELS)
    plot_confusion_matrix(cm, CLASS_NAMES, "Attention_CNN", RESULTS_DIR, f_out)
    
    return mean_accuracy

def train_gradient_boosting_nn_hybrid(features, labels, f_out, quick_check=False):
    """Train gradient boosting + neural network hybrid."""
    f_out.write("Training Gradient Boosting + Neural Network Hybrid...\n")
    print("Training Gradient Boosting + NN Hybrid...")
    
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_range = [next(skf.split(features, labels))] if quick_check else list(skf.split(features, labels))
    
    accuracies = []
    all_y_true, all_y_pred = [], []
    
    for fold, (train_idx, test_idx) in enumerate(tqdm(fold_range, total=1 if quick_check else 5, desc="  GB+NN Hybrid CV")):
        X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
        y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
        
        # Stage 1: Gradient Boosting for feature learning
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=RANDOM_STATE)
        gb.fit(X_train, y_train)
        
        # Extract learned features from GB (leaf indices)
        gb_features_train = gb.apply(X_train).reshape(X_train.shape[0], -1)
        gb_features_test = gb.apply(X_test).reshape(X_test.shape[0], -1)
        
        # Combine original features with GB features
        combined_train = np.concatenate([X_train, gb_features_train], axis=1)
        combined_test = np.concatenate([X_test, gb_features_test], axis=1)
        
        # Stage 2: Neural Network on combined features
        X_train_tensor = torch.FloatTensor(combined_train).to(DEVICE)
        X_test_tensor = torch.FloatTensor(combined_test).to(DEVICE)
        y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
        
        # Hybrid neural network
        model = nn.Sequential(
            nn.Linear(combined_train.shape[1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, len(CLASS_DEFINITIONS))
        ).to(DEVICE)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Class weights
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training
        model.train()
        for epoch in range(80):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            y_pred = predicted.cpu().numpy()
        
        fold_accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(fold_accuracy)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        f_out.write(f"  GB+NN Hybrid Fold {fold+1}: {fold_accuracy:.4f}\n")
        print(f"    GB+NN Hybrid Fold {fold+1}: {fold_accuracy:.4f}")
        
        del model, optimizer
        torch.cuda.empty_cache()
    
    mean_accuracy = np.mean(accuracies)
    f_out.write(f"GB+NN Hybrid Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})\n")
    print(f"  GB+NN Hybrid Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})")
    
    # Classification report and confusion matrix
    report = classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, labels=CLASS_LABELS, zero_division=0)
    f_out.write("GB+NN Hybrid Classification Report:\n" + report + "\n")
    
    cm = confusion_matrix(all_y_true, all_y_pred, labels=CLASS_LABELS)
    plot_confusion_matrix(cm, CLASS_NAMES, "GradientBoosting_NN_Hybrid", RESULTS_DIR, f_out)
    
    return mean_accuracy

def train_meta_learning_ensemble(stat_features, sequences, labels, f_out, quick_check=False):
    """Train meta-learning ensemble with stacking."""
    f_out.write("Training Meta-Learning Ensemble...\n")
    print("Training Meta-Learning Ensemble...")
    
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(stat_features)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_range = [next(skf.split(features_scaled, labels))] if quick_check else list(skf.split(features_scaled, labels))
    
    accuracies = []
    all_y_true, all_y_pred = [], []
    
    for fold, (train_idx, test_idx) in enumerate(tqdm(fold_range, total=1 if quick_check else 5, desc="  Meta-Learning CV")):
        X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
        y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
        
        # Split training data for meta-learning
        meta_train_idx, meta_val_idx = train_test_split(range(len(X_train)), test_size=0.3, random_state=RANDOM_STATE, stratify=y_train)
        
        X_meta_train, X_meta_val = X_train[meta_train_idx], X_train[meta_val_idx]
        y_meta_train, y_meta_val = y_train[meta_train_idx], y_train[meta_val_idx]
        
        # Base models
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
            ('et', ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE)),
            ('lda', LinearDiscriminantAnalysis()),
        ]
        
        # Train base models and get meta-features
        meta_features_val = []
        meta_features_test = []
        
        for name, model in base_models:
            # Train on meta-train set
            model.fit(X_meta_train, y_meta_train)
            
            # Predict on meta-validation set
            val_pred_proba = model.predict_proba(X_meta_val)
            meta_features_val.append(val_pred_proba)
            
            # Predict on test set
            test_pred_proba = model.predict_proba(X_test)
            meta_features_test.append(test_pred_proba)
        
        # Combine meta-features
        meta_X_val = np.concatenate(meta_features_val, axis=1)
        meta_X_test = np.concatenate(meta_features_test, axis=1)
        
        # Train neural network meta-learner
        meta_X_val_tensor = torch.FloatTensor(meta_X_val).to(DEVICE)
        meta_X_test_tensor = torch.FloatTensor(meta_X_test).to(DEVICE)
        y_meta_val_tensor = torch.LongTensor(y_meta_val).to(DEVICE)
        
        meta_model = nn.Sequential(
            nn.Linear(meta_X_val.shape[1], 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, len(CLASS_DEFINITIONS))
        ).to(DEVICE)
        
        optimizer = torch.optim.AdamW(meta_model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Train meta-learner
        meta_model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = meta_model(meta_X_val_tensor)
            loss = criterion(outputs, y_meta_val_tensor)
            loss.backward()
            optimizer.step()
        
        # Final prediction
        meta_model.eval()
        with torch.no_grad():
            outputs = meta_model(meta_X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            y_pred = predicted.cpu().numpy()
        
        fold_accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(fold_accuracy)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        f_out.write(f"  Meta-Learning Fold {fold+1}: {fold_accuracy:.4f}\n")
        print(f"    Meta-Learning Fold {fold+1}: {fold_accuracy:.4f}")
        
        del meta_model, optimizer
        torch.cuda.empty_cache()
    
    mean_accuracy = np.mean(accuracies)
    f_out.write(f"Meta-Learning Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})\n")
    print(f"  Meta-Learning Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})")
    
    # Classification report and confusion matrix
    report = classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, labels=CLASS_LABELS, zero_division=0)
    f_out.write("Meta-Learning Classification Report:\n" + report + "\n")
    
    cm = confusion_matrix(all_y_true, all_y_pred, labels=CLASS_LABELS)
    plot_confusion_matrix(cm, CLASS_NAMES, "Meta_Learning_Ensemble", RESULTS_DIR, f_out)
    
    return mean_accuracy

def train_ultra_advanced_statistical_nn(features, labels, f_out, quick_check=False):
    """Train ultra-advanced statistical neural network with cutting-edge techniques."""
    f_out.write("Training Ultra-Advanced Statistical Neural Network...\n")
    print("Training Ultra-Advanced Statistical NN...")
    
    from sklearn.preprocessing import RobustScaler, QuantileTransformer, PowerTransformer
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    from sklearn.decomposition import PCA
    
    # Multi-stage preprocessing pipeline
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Power transformation for normality
    power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    features_power = power_transformer.fit_transform(features_scaled)
    
    # Feature selection with multiple criteria
    selector1 = SelectKBest(f_classif, k=min(50, features_power.shape[1]))
    features_selected1 = selector1.fit_transform(features_power, labels)
    
    selector2 = SelectKBest(mutual_info_classif, k=min(40, features_selected1.shape[1]))
    features_selected2 = selector2.fit_transform(features_selected1, labels)
    
    # PCA for dimensionality reduction while preserving variance
    pca = PCA(n_components=min(30, features_selected2.shape[1]), random_state=RANDOM_STATE)
    features_pca = pca.fit_transform(features_selected2)
    
    # Final quantile transformation
    quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
    features_final = quantile_transformer.fit_transform(features_pca)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_range = [next(skf.split(features_final, labels))] if quick_check else list(skf.split(features_final, labels))
    
    accuracies = []
    all_y_true, all_y_pred = [], []
    
    for fold, (train_idx, test_idx) in enumerate(tqdm(fold_range, total=1 if quick_check else 5, desc="  Ultra-Advanced Statistical NN CV")):
        X_train, X_test = features_final[train_idx], features_final[test_idx]
        y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
        
        # Ultra-advanced architecture with multiple innovations
        class UltraAdvancedStatisticalNN(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                
                # Multi-path feature extraction
                self.path1 = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.BatchNorm1d(128),
                    nn.GELU(),  # Better activation than ReLU
                    nn.Dropout(0.3)
                )
                
                self.path2 = nn.Sequential(
                    nn.Linear(input_dim, 96),
                    nn.BatchNorm1d(96),
                    nn.GELU(),
                    nn.Dropout(0.3)
                )
                
                self.path3 = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.BatchNorm1d(64),
                    nn.GELU(),
                    nn.Dropout(0.3)
                )
                
                # Self-attention mechanism
                self.self_attention = nn.MultiheadAttention(
                    embed_dim=128+96+64, num_heads=8, batch_first=True, dropout=0.1
                )
                self.attention_norm = nn.LayerNorm(128+96+64)
                
                # Residual dense blocks
                self.dense_block1 = self._make_dense_block(128+96+64, 256)
                self.dense_block2 = self._make_dense_block(256, 192)
                self.dense_block3 = self._make_dense_block(192, 128)
                
                # Squeeze-and-excitation module
                self.se_module = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(128, 32),
                    nn.ReLU(),
                    nn.Linear(32, 128),
                    nn.Sigmoid()
                )
                
                # Multiple expert heads with different specializations
                self.expert1 = nn.Sequential(
                    nn.Linear(128, 96),
                    nn.GELU(),
                    nn.Dropout(0.4),
                    nn.Linear(96, 64),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, output_dim)
                )
                
                self.expert2 = nn.Sequential(
                    nn.Linear(128, 80),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(80, output_dim)
                )
                
                self.expert3 = nn.Sequential(
                    nn.Linear(128, 48),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(48, output_dim)
                )
                
                # Gating network for expert combination
                self.gate = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.GELU(),
                    nn.Linear(64, 3),
                    nn.Softmax(dim=1)
                )
                
                # Initialize weights with advanced techniques
                self.apply(self._init_weights)
            
            def _make_dense_block(self, in_dim, out_dim):
                return nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(out_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.GELU(),
                    nn.Dropout(0.2)
                )
            
            def _init_weights(self, module):
                if isinstance(module, nn.Linear):
                    # Xavier initialization for GELU
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, nn.BatchNorm1d):
                    torch.nn.init.ones_(module.weight)
                    torch.nn.init.zeros_(module.bias)
            
            def forward(self, x):
                # Multi-path feature extraction
                p1 = self.path1(x)
                p2 = self.path2(x)
                p3 = self.path3(x)
                
                # Concatenate paths
                multi_path = torch.cat([p1, p2, p3], dim=1)
                
                # Self-attention (add sequence dimension)
                multi_path_seq = multi_path.unsqueeze(1)
                attn_out, _ = self.self_attention(multi_path_seq, multi_path_seq, multi_path_seq)
                attn_out = self.attention_norm(multi_path_seq + attn_out)
                attn_out = attn_out.squeeze(1)
                
                # Dense blocks with residual connections
                x = self.dense_block1(attn_out)
                x = self.dense_block2(x)
                x = self.dense_block3(x)
                
                # Squeeze-and-excitation
                se_weights = self.se_module(x.unsqueeze(-1))
                x = x * se_weights
                
                # Expert predictions
                expert1_out = self.expert1(x)
                expert2_out = self.expert2(x)
                expert3_out = self.expert3(x)
                
                # Gating mechanism
                gate_weights = self.gate(x)
                
                # Weighted combination of experts
                final_output = (gate_weights[:, 0:1] * expert1_out + 
                               gate_weights[:, 1:2] * expert2_out + 
                               gate_weights[:, 2:3] * expert3_out)
                
                return final_output
        
        model = UltraAdvancedStatisticalNN(features_final.shape[1], len(CLASS_DEFINITIONS)).to(DEVICE)
        
        # Advanced optimizer with different parameter groups
        param_groups = [
            {'params': [p for n, p in model.named_parameters() if 'expert' in n], 'lr': 0.003, 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if 'path' in n], 'lr': 0.002, 'weight_decay': 0.015},
            {'params': [p for n, p in model.named_parameters() if 'dense' in n], 'lr': 0.0025, 'weight_decay': 0.012},
            {'params': [p for n, p in model.named_parameters() if 'attention' in n or 'gate' in n], 'lr': 0.001, 'weight_decay': 0.02}
        ]
        optimizer = torch.optim.AdamW(param_groups, amsgrad=True)
        
        # Advanced loss with multiple components
        class UltraAdvancedLoss(nn.Module):
            def __init__(self, class_weights, smoothing=0.15, temperature=3.0):
                super().__init__()
                self.class_weights = class_weights
                self.smoothing = smoothing
                self.temperature = temperature
                
            def forward(self, pred, target):
                # Temperature scaling
                pred_scaled = pred / self.temperature
                
                # Label smoothing with class weights
                confidence = 1.0 - self.smoothing
                log_probs = F.log_softmax(pred_scaled, dim=1)
                nll_loss = F.nll_loss(log_probs, target, weight=self.class_weights, reduction='none')
                smooth_loss = -log_probs.mean(dim=1)
                loss = confidence * nll_loss + self.smoothing * smooth_loss
                
                # Add entropy regularization
                entropy_reg = -0.01 * torch.sum(F.softmax(pred_scaled, dim=1) * log_probs, dim=1)
                
                return (loss + entropy_reg).mean()
        
        # Class weights with advanced balancing
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * class_counts)
        # Apply square root to reduce extreme weights
        class_weights = np.sqrt(class_weights)
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)
        criterion = UltraAdvancedLoss(class_weights, smoothing=0.15, temperature=3.0)
        
        # Advanced scheduler with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        
        # Training with advanced techniques
        model.train()
        best_loss = float('inf')
        patience = 0
        
        for epoch in range(120):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            
            # Advanced regularization
            l1_lambda = 0.0001
            l2_lambda = 0.001
            spectral_lambda = 0.0005
            
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            
            # Spectral normalization penalty
            spectral_norm = 0
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    U, S, V = torch.svd(module.weight)
                    spectral_norm += torch.max(S)
            
            total_loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm + spectral_lambda * spectral_norm
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            
            # Early stopping with patience
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience = 0
            else:
                patience += 1
                if patience >= 25:
                    break
        
        # Evaluation with test-time augmentation
        model.eval()
        test_predictions = []
        
        # Multiple forward passes with different dropout patterns
        for _ in range(5):
            model.train()  # Enable dropout
            with torch.no_grad():
                outputs = model(X_test_tensor)
                test_predictions.append(F.softmax(outputs, dim=1))
        
        # Average predictions
        avg_predictions = torch.mean(torch.stack(test_predictions), dim=0)
        _, predicted = torch.max(avg_predictions, 1)
        y_pred = predicted.cpu().numpy()
        
        fold_accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(fold_accuracy)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        f_out.write(f"  Ultra-Advanced Statistical NN Fold {fold+1}: {fold_accuracy:.4f}\n")
        print(f"    Ultra-Advanced Statistical NN Fold {fold+1}: {fold_accuracy:.4f}")
        
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
    
    mean_accuracy = np.mean(accuracies)
    f_out.write(f"Ultra-Advanced Statistical NN Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})\n")
    print(f"  Ultra-Advanced Statistical NN Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})")
    
    # Classification report and confusion matrix
    report = classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, labels=CLASS_LABELS, zero_division=0)
    f_out.write("Ultra-Advanced Statistical NN Classification Report:\n" + report + "\n")
    
    cm = confusion_matrix(all_y_true, all_y_pred, labels=CLASS_LABELS)
    plot_confusion_matrix(cm, CLASS_NAMES, "Ultra_Advanced_Statistical_NN", RESULTS_DIR, f_out)
    
    return mean_accuracy

def train_advanced_sequence_model(sequences, labels, n_features, n_classes, f_out, quick_check=False):
    """Train advanced sequence-based model with state-of-the-art techniques."""
    f_out.write("Training Advanced Sequence Model...\n")
    print("Training Advanced Sequence Model...")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_range = [next(skf.split(sequences, labels))] if quick_check else list(skf.split(sequences, labels))
    
    accuracies = []
    all_y_true, all_y_pred = [], []
    
    for fold, (train_idx, test_idx) in enumerate(tqdm(fold_range, total=1 if quick_check else 5, desc="  Advanced Sequence CV")):
        X_train_seq = [sequences[i] for i in train_idx]
        X_test_seq = [sequences[i] for i in test_idx]
        y_train = np.array(labels)[train_idx]
        y_test = np.array(labels)[test_idx]
        
        # Advanced sequence model architecture
        class AdvancedSequenceModel(nn.Module):
            def __init__(self, input_dim, output_dim, hidden_dim=64):
                super().__init__()
                
                # Multi-scale temporal convolutions
                self.conv1d_1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
                self.conv1d_2 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
                self.conv1d_3 = nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3)
                
                # Bidirectional LSTM with multiple layers
                self.lstm = nn.LSTM(
                    hidden_dim * 3, hidden_dim, num_layers=2, 
                    batch_first=True, bidirectional=True, dropout=0.3
                )
                
                # Multi-head attention
                self.attention = nn.MultiheadAttention(
                    hidden_dim * 2, num_heads=8, batch_first=True, dropout=0.1
                )
                self.attention_norm = nn.LayerNorm(hidden_dim * 2)
                
                # Temporal pooling strategies
                self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
                self.max_pool = nn.AdaptiveMaxPool1d(1)
                
                # Classification head with multiple paths
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim * 4, hidden_dim * 2),  # *4 because of avg+max pooling
                    nn.BatchNorm1d(hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(0.4),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, output_dim)
                )
                
                self.apply(self._init_weights)
            
            def _init_weights(self, module):
                if isinstance(module, (nn.Linear, nn.Conv1d)):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LSTM):
                    for name, param in module.named_parameters():
                        if 'weight' in name:
                            torch.nn.init.xavier_uniform_(param)
                        elif 'bias' in name:
                            torch.nn.init.zeros_(param)
            
            def forward(self, x, lengths):
                batch_size, seq_len, features = x.size()
                
                # Multi-scale convolutions
                x_conv = x.transpose(1, 2)  # (batch, features, seq_len)
                conv1 = F.relu(self.conv1d_1(x_conv))
                conv2 = F.relu(self.conv1d_2(x_conv))
                conv3 = F.relu(self.conv1d_3(x_conv))
                
                # Concatenate multi-scale features
                multi_scale = torch.cat([conv1, conv2, conv3], dim=1)
                multi_scale = multi_scale.transpose(1, 2)  # Back to (batch, seq_len, features)
                
                # Pack sequences for LSTM
                packed_input = nn.utils.rnn.pack_padded_sequence(
                    multi_scale, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                
                # LSTM processing
                packed_output, _ = self.lstm(packed_input)
                lstm_output, output_lengths = nn.utils.rnn.pad_packed_sequence(
                    packed_output, batch_first=True
                )
                
                # Self-attention
                attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
                attn_output = self.attention_norm(lstm_output + attn_output)
                
                # Temporal pooling
                attn_transposed = attn_output.transpose(1, 2)  # (batch, features, seq_len)
                avg_pooled = self.adaptive_pool(attn_transposed).squeeze(-1)  # (batch, features)
                max_pooled = self.max_pool(attn_transposed).squeeze(-1)  # (batch, features)
                
                # Combine pooled features
                combined_features = torch.cat([avg_pooled, max_pooled], dim=1)
                
                # Classification
                output = self.classifier(combined_features)
                return output
        
        # Create datasets
        train_dataset = SaccadeSequenceDataset(X_train_seq, y_train, augment=True, augmentation_prob=0.6)
        test_dataset = SaccadeSequenceDataset(X_test_seq, y_test, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
        
        # Initialize model
        model = AdvancedSequenceModel(
            input_dim=n_features, 
            output_dim=n_classes, 
            hidden_dim=48  # Reduced for memory efficiency
        ).to(DEVICE)
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)
        
        # Class weights
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training
        model.train()
        for epoch in range(40):
            epoch_loss = 0
            for seq_padded, labels_batch, lengths in train_loader:
                seq_padded, labels_batch = seq_padded.to(DEVICE), labels_batch.to(DEVICE)
                lengths = lengths.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(seq_padded, lengths)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step()
        
        # Evaluation
        model.eval()
        fold_preds, fold_true = [], []
        with torch.no_grad():
            for seq_padded, labels_batch, lengths in test_loader:
                seq_padded, labels_batch = seq_padded.to(DEVICE), labels_batch.to(DEVICE)
                lengths = lengths.to(DEVICE)
                outputs = model(seq_padded, lengths)
                _, predicted = torch.max(outputs, 1)
                fold_preds.extend(predicted.cpu().numpy())
                fold_true.extend(labels_batch.cpu().numpy())
        
        fold_accuracy = accuracy_score(fold_true, fold_preds)
        accuracies.append(fold_accuracy)
        all_y_true.extend(fold_true)
        all_y_pred.extend(fold_preds)
        
        f_out.write(f"  Advanced Sequence Fold {fold+1}: {fold_accuracy:.4f}\n")
        print(f"    Advanced Sequence Fold {fold+1}: {fold_accuracy:.4f}")
        
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
    
    mean_accuracy = np.mean(accuracies)
    f_out.write(f"Advanced Sequence Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})\n")
    print(f"  Advanced Sequence Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})")
    
    # Classification report and confusion matrix
    report = classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, labels=CLASS_LABELS, zero_division=0)
    f_out.write("Advanced Sequence Classification Report:\n" + report + "\n")
    
    cm = confusion_matrix(all_y_true, all_y_pred, labels=CLASS_LABELS)
    plot_confusion_matrix(cm, CLASS_NAMES, "Advanced_Sequence_Model", RESULTS_DIR, f_out)
    
    return mean_accuracy

def train_hybrid_statistical_sequence_model(stat_features, sequences, labels, f_out, quick_check=False):
    """Train hybrid model combining advanced statistical and sequence features."""
    f_out.write("Training Hybrid Statistical-Sequence Model...\n")
    print("Training Hybrid Statistical-Sequence Model...")
    
    # Advanced preprocessing for statistical features
    from sklearn.preprocessing import RobustScaler, QuantileTransformer
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # Multi-stage preprocessing
    scaler = RobustScaler()
    stat_features_scaled = scaler.fit_transform(stat_features)
    
    # Feature selection
    selector = SelectKBest(f_classif, k=min(40, stat_features_scaled.shape[1]))
    stat_features_selected = selector.fit_transform(stat_features_scaled, labels)
    
    # Quantile transformation
    quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
    stat_features_final = quantile_transformer.fit_transform(stat_features_selected)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_range = [next(skf.split(sequences, labels))] if quick_check else list(skf.split(sequences, labels))
    
    accuracies = []
    all_y_true, all_y_pred = [], []
    
    for fold, (train_idx, test_idx) in enumerate(tqdm(fold_range, total=1 if quick_check else 5, desc="  Hybrid Statistical-Sequence CV")):
        # Statistical features
        X_stat_train, X_stat_test = stat_features_final[train_idx], stat_features_final[test_idx]
        
        # Sequence features
        X_seq_train = [sequences[i] for i in train_idx]
        X_seq_test = [sequences[i] for i in test_idx]
        
        y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
        
        # Advanced hybrid model
        class AdvancedHybridModel(nn.Module):
            def __init__(self, stat_dim, seq_dim, hidden_dim=64):
                super().__init__()
                
                # Statistical branch with attention
                self.stat_branch = nn.Sequential(
                    nn.Linear(stat_dim, hidden_dim * 2),
                    nn.BatchNorm1d(hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.2)
                )
                
                # Sequence branch with LSTM and attention
                self.seq_lstm = nn.LSTM(seq_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
                self.seq_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
                self.seq_norm = nn.LayerNorm(hidden_dim)
                
                # Cross-modal attention
                self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
                self.cross_norm = nn.LayerNorm(hidden_dim)
                
                # Fusion network
                self.fusion = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
                    nn.BatchNorm1d(hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(0.4),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, len(CLASS_DEFINITIONS))
                )
                
                self.apply(self._init_weights)
            
            def _init_weights(self, module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
            
            def forward(self, stat_features, seq_features, seq_lengths):
                batch_size = stat_features.size(0)
                
                # Statistical branch
                stat_out = self.stat_branch(stat_features)  # (batch, hidden_dim)
                
                # Sequence branch
                packed_seq = nn.utils.rnn.pack_padded_sequence(
                    seq_features, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                lstm_out, _ = self.seq_lstm(packed_seq)
                seq_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
                
                # Self-attention on sequences
                seq_attn, _ = self.seq_attention(seq_out, seq_out, seq_out)
                seq_out = self.seq_norm(seq_out + seq_attn)
                
                # Global average pooling for sequences
                seq_pooled = torch.mean(seq_out, dim=1)  # (batch, hidden_dim)
                
                # Cross-modal attention
                stat_expanded = stat_out.unsqueeze(1)  # (batch, 1, hidden_dim)
                seq_expanded = seq_pooled.unsqueeze(1)  # (batch, 1, hidden_dim)
                
                cross_attn, _ = self.cross_attention(stat_expanded, seq_expanded, seq_expanded)
                stat_enhanced = self.cross_norm(stat_expanded + cross_attn).squeeze(1)
                
                # Fusion
                combined = torch.cat([stat_enhanced, seq_pooled], dim=1)
                output = self.fusion(combined)
                
                return output
        
        # Create datasets
        train_dataset = SaccadeSequenceDataset(X_seq_train, y_train, augment=True, augmentation_prob=0.5)
        test_dataset = SaccadeSequenceDataset(X_seq_test, y_test, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
        
        # Initialize model
        model = AdvancedHybridModel(
            stat_dim=X_stat_train.shape[1], 
            seq_dim=sequences[0].shape[1], 
            hidden_dim=48
        ).to(DEVICE)
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        
        # Class weights
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Convert statistical features to tensors
        X_stat_train_tensor = torch.FloatTensor(X_stat_train).to(DEVICE)
        X_stat_test_tensor = torch.FloatTensor(X_stat_test).to(DEVICE)
        
        # Training
        model.train()
        for epoch in range(50):
            epoch_loss = 0
            stat_idx = 0
            
            for seq_padded, labels_batch, lengths in train_loader:
                seq_padded, labels_batch = seq_padded.to(DEVICE), labels_batch.to(DEVICE)
                lengths = lengths.to(DEVICE)
                
                # Get corresponding statistical features
                batch_size = seq_padded.size(0)
                stat_batch = X_stat_train_tensor[stat_idx:stat_idx + batch_size]
                stat_idx += batch_size
                
                optimizer.zero_grad()
                outputs = model(stat_batch, seq_padded, lengths)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step()
        
        # Evaluation
        model.eval()
        fold_preds, fold_true = [], []
        stat_idx = 0
        
        with torch.no_grad():
            for seq_padded, labels_batch, lengths in test_loader:
                seq_padded, labels_batch = seq_padded.to(DEVICE), labels_batch.to(DEVICE)
                lengths = lengths.to(DEVICE)
                
                # Get corresponding statistical features
                batch_size = seq_padded.size(0)
                stat_batch = X_stat_test_tensor[stat_idx:stat_idx + batch_size]
                stat_idx += batch_size
                
                outputs = model(stat_batch, seq_padded, lengths)
                _, predicted = torch.max(outputs, 1)
                fold_preds.extend(predicted.cpu().numpy())
                fold_true.extend(labels_batch.cpu().numpy())
        
        fold_accuracy = accuracy_score(fold_true, fold_preds)
        accuracies.append(fold_accuracy)
        all_y_true.extend(fold_true)
        all_y_pred.extend(fold_preds)
        
        f_out.write(f"  Hybrid Statistical-Sequence Fold {fold+1}: {fold_accuracy:.4f}\n")
        print(f"    Hybrid Statistical-Sequence Fold {fold+1}: {fold_accuracy:.4f}")
        
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
    
    mean_accuracy = np.mean(accuracies)
    f_out.write(f"Hybrid Statistical-Sequence Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})\n")
    print(f"  Hybrid Statistical-Sequence Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})")
    
    # Classification report and confusion matrix
    report = classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, labels=CLASS_LABELS, zero_division=0)
    f_out.write("Hybrid Statistical-Sequence Classification Report:\n" + report + "\n")
    
    cm = confusion_matrix(all_y_true, all_y_pred, labels=CLASS_LABELS)
    plot_confusion_matrix(cm, CLASS_NAMES, "Hybrid_Statistical_Sequence", RESULTS_DIR, f_out)
    
    return mean_accuracy

def train_ultra_advanced_nn(stat_features, sequences, labels, f_out, quick_check=False):
    """Train ultra-advanced neural network with all techniques combined."""
    f_out.write("Training Ultra-Advanced Neural Network...\n")
    print("Training Ultra-Advanced NN...")
    
    from sklearn.preprocessing import RobustScaler, QuantileTransformer
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # Advanced preprocessing pipeline
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(stat_features)
    
    # Feature selection
    selector = SelectKBest(f_classif, k=min(100, features_scaled.shape[1]))
    features_selected = selector.fit_transform(features_scaled, labels)
    
    # Quantile transformation
    quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
    features_transformed = quantile_transformer.fit_transform(features_selected)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_range = [next(skf.split(features_transformed, labels))] if quick_check else list(skf.split(features_transformed, labels))
    
    accuracies = []
    all_y_true, all_y_pred = [], []
    
    for fold, (train_idx, test_idx) in enumerate(tqdm(fold_range, total=1 if quick_check else 5, desc="  Ultra-Advanced NN CV")):
        X_train, X_test = features_transformed[train_idx], features_transformed[test_idx]
        y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
        
        # Ultra-advanced architecture
        class UltraAdvancedNN(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                
                # Multi-scale feature extraction
                self.scale1 = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                )
                
                self.scale2 = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                )
                
                self.scale3 = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                )
                
                # Attention fusion
                self.attention_fusion = nn.Sequential(
                    nn.Linear(256 + 128 + 64, 128),
                    nn.Tanh(),
                    nn.Linear(128, 3),
                    nn.Softmax(dim=1)
                )
                
                # Residual blocks
                self.res_block1 = self._make_residual_block(256 + 128 + 64, 256)
                self.res_block2 = self._make_residual_block(256, 128)
                
                # Multiple classification heads
                self.head1 = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(64, output_dim)
                )
                
                self.head2 = nn.Sequential(
                    nn.Linear(128, 96),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(96, 48),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(48, output_dim)
                )
                
                self.head3 = nn.Sequential(
                    nn.Linear(128, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, output_dim)
                )
                
                # Initialize weights
                self.apply(self._init_weights)
            
            def _make_residual_block(self, in_dim, out_dim):
                return nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(out_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
            
            def _init_weights(self, module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
            
            def forward(self, x):
                # Multi-scale feature extraction
                s1 = self.scale1(x)
                s2 = self.scale2(x)
                s3 = self.scale3(x)
                
                # Concatenate scales
                multi_scale = torch.cat([s1, s2, s3], dim=1)
                
                # Attention weights for scales
                attn_weights = self.attention_fusion(multi_scale)
                
                # Apply attention - fix dimension mismatch
                # attn_weights is (batch, 3), we need to expand to match feature dimensions
                attn_1 = attn_weights[:, 0:1].expand(-1, s1.size(1))  # (batch, 256)
                attn_2 = attn_weights[:, 1:2].expand(-1, s2.size(1))  # (batch, 128)  
                attn_3 = attn_weights[:, 2:3].expand(-1, s3.size(1))  # (batch, 64)
                
                # Weighted combination - concatenate instead of element-wise multiply
                attended = torch.cat([s1 * attn_1, s2 * attn_2, s3 * attn_3], dim=1)
                
                # Residual processing
                x = self.res_block1(multi_scale)
                x = self.res_block2(x)
                
                # Multiple heads with different architectures
                out1 = self.head1(x)
                out2 = self.head2(x)
                out3 = self.head3(x)
                
                # Ensemble the heads
                return 0.5 * out1 + 0.3 * out2 + 0.2 * out3
        
        model = UltraAdvancedNN(features_transformed.shape[1], len(CLASS_DEFINITIONS)).to(DEVICE)
        
        # Advanced optimizer with different learning rates
        param_groups = [
            {'params': [p for n, p in model.named_parameters() if 'head' in n], 'lr': 0.003},
            {'params': [p for n, p in model.named_parameters() if 'scale' in n], 'lr': 0.001},
            {'params': [p for n, p in model.named_parameters() if 'res_block' in n], 'lr': 0.0015},
            {'params': [p for n, p in model.named_parameters() if 'attention' in n], 'lr': 0.002}
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.02, amsgrad=True)
        
        # Advanced loss with multiple components
        class AdvancedLoss(nn.Module):
            def __init__(self, class_weights, smoothing=0.1):
                super().__init__()
                self.class_weights = class_weights
                self.smoothing = smoothing
                
            def forward(self, pred, target):
                # Label smoothing cross entropy
                confidence = 1.0 - self.smoothing
                log_probs = F.log_softmax(pred, dim=1)
                nll_loss = F.nll_loss(log_probs, target, weight=self.class_weights, reduction='none')
                smooth_loss = -log_probs.mean(dim=1)
                loss = confidence * nll_loss + self.smoothing * smooth_loss
                return loss.mean()
        
        # Class weights
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)
        criterion = AdvancedLoss(class_weights, smoothing=0.1)
        
        # Advanced scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=[0.003, 0.001, 0.0015, 0.002], 
            epochs=100, steps_per_epoch=1
        )
        
        # Training with advanced techniques
        model.train()
        best_loss = float('inf')
        patience = 0
        
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            
            # Additional regularization
            l1_lambda = 0.0001
            l2_lambda = 0.001
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss += l1_lambda * l1_norm + l2_lambda * l2_norm
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()
            scheduler.step()
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience = 0
            else:
                patience += 1
                if patience >= 20:
                    break
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            y_pred = predicted.cpu().numpy()
        
        fold_accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(fold_accuracy)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        f_out.write(f"  Ultra-Advanced NN Fold {fold+1}: {fold_accuracy:.4f}\n")
        print(f"    Ultra-Advanced NN Fold {fold+1}: {fold_accuracy:.4f}")
        
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
    
    mean_accuracy = np.mean(accuracies)
    f_out.write(f"Ultra-Advanced NN Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})\n")
    print(f"  Ultra-Advanced NN Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})")
    
    # Classification report and confusion matrix
    report = classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, labels=CLASS_LABELS, zero_division=0)
    f_out.write("Ultra-Advanced NN Classification Report:\n" + report + "\n")
    
    cm = confusion_matrix(all_y_true, all_y_pred, labels=CLASS_LABELS)
    plot_confusion_matrix(cm, CLASS_NAMES, "Ultra_Advanced_NN", RESULTS_DIR, f_out)
    
    return mean_accuracy

def train_revolutionary_preprocessing_nn(stat_features, sequences, labels, f_out, quick_check=False):
    """Train neural network with revolutionary data preprocessing techniques."""
    f_out.write("Training Revolutionary Preprocessing Neural Network...\n")
    print("Training Revolutionary Preprocessing NN...")
    
    from sklearn.preprocessing import RobustScaler, PowerTransformer, QuantileTransformer
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
    from sklearn.decomposition import PCA, FastICA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.manifold import LocallyLinearEmbedding
    
    # Revolutionary multi-stage preprocessing pipeline
    # Stage 1: Robust scaling
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(stat_features)
    
    # Stage 2: Power transformation for normality
    power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    features_power = power_transformer.fit_transform(features_scaled)
    
    # Stage 3: Multiple feature selection approaches
    # F-test based selection
    selector_f = SelectKBest(f_classif, k=min(60, features_power.shape[1]))
    features_f = selector_f.fit_transform(features_power, labels)
    
    # Mutual information based selection
    selector_mi = SelectKBest(mutual_info_classif, k=min(50, features_f.shape[1]))
    features_mi = selector_mi.fit_transform(features_f, labels)
    
    # Recursive feature elimination with Random Forest
    rf_selector = RFE(RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE), 
                      n_features_to_select=min(40, features_mi.shape[1]))
    features_rfe = rf_selector.fit_transform(features_mi, labels)
    
    # Stage 4: Dimensionality reduction ensemble
    # PCA for linear relationships
    pca = PCA(n_components=min(25, features_rfe.shape[1]), random_state=RANDOM_STATE)
    features_pca = pca.fit_transform(features_rfe)
    
    # ICA for independent components
    ica = FastICA(n_components=min(20, features_pca.shape[1]), random_state=RANDOM_STATE, max_iter=1000)
    features_ica = ica.fit_transform(features_pca)
    
    # Combine PCA and ICA features
    features_combined = np.concatenate([features_pca, features_ica], axis=1)
    
    # Stage 5: Final quantile transformation
    quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
    features_final = quantile_transformer.fit_transform(features_combined)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_range = [next(skf.split(features_final, labels))] if quick_check else list(skf.split(features_final, labels))
    
    accuracies = []
    all_y_true, all_y_pred = [], []
    
    for fold, (train_idx, test_idx) in enumerate(tqdm(fold_range, total=1 if quick_check else 5, desc="  Revolutionary Preprocessing NN CV")):
        X_train, X_test = features_final[train_idx], features_final[test_idx]
        y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
        
        # Revolutionary neural network architecture
        class RevolutionaryNN(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                
                # Adaptive input layer that learns optimal feature combinations
                self.adaptive_input = nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.LayerNorm(input_dim),
                    nn.GELU(),
                    nn.Dropout(0.2)
                )
                
                # Multi-branch architecture with different activation functions
                self.branch_relu = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                )
                
                self.branch_gelu = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.BatchNorm1d(128),
                    nn.GELU(),
                    nn.Dropout(0.3)
                )
                
                self.branch_swish = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.BatchNorm1d(128),
                    nn.SiLU(),  # Swish activation
                    nn.Dropout(0.3)
                )
                
                # Attention-based branch fusion
                self.branch_attention = nn.Sequential(
                    nn.Linear(128 * 3, 64),
                    nn.Tanh(),
                    nn.Linear(64, 3),
                    nn.Softmax(dim=1)
                )
                
                # Advanced residual processing
                self.residual_layers = nn.ModuleList([
                    self._make_residual_layer(128, 128) for _ in range(3)
                ])
                
                # Mixture of experts
                self.expert_1 = nn.Sequential(
                    nn.Linear(128, 96),
                    nn.GELU(),
                    nn.Dropout(0.4),
                    nn.Linear(96, output_dim)
                )
                
                self.expert_2 = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, output_dim)
                )
                
                self.expert_3 = nn.Sequential(
                    nn.Linear(128, 80),
                    nn.SiLU(),
                    nn.Dropout(0.35),
                    nn.Linear(80, output_dim)
                )
                
                # Gating network for expert selection
                self.gate = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.GELU(),
                    nn.Linear(64, 3),
                    nn.Softmax(dim=1)
                )
                
                self.apply(self._init_weights)
            
            def _make_residual_layer(self, in_dim, out_dim):
                return nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(out_dim, out_dim),
                    nn.BatchNorm1d(out_dim)
                )
            
            def _init_weights(self, module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
            
            def forward(self, x):
                # Adaptive input processing
                x_adapted = self.adaptive_input(x)
                
                # Multi-branch processing
                branch1 = self.branch_relu(x_adapted)
                branch2 = self.branch_gelu(x_adapted)
                branch3 = self.branch_swish(x_adapted)
                
                # Attention-based fusion
                branches_concat = torch.cat([branch1, branch2, branch3], dim=1)
                attention_weights = self.branch_attention(branches_concat)
                
                # Weighted combination of branches
                fused = (attention_weights[:, 0:1] * branch1 + 
                        attention_weights[:, 1:2] * branch2 + 
                        attention_weights[:, 2:3] * branch3)
                
                # Residual processing
                x = fused
                for residual_layer in self.residual_layers:
                    residual = residual_layer(x)
                    x = F.gelu(x + residual)  # Residual connection
                
                # Mixture of experts
                expert1_out = self.expert_1(x)
                expert2_out = self.expert_2(x)
                expert3_out = self.expert_3(x)
                
                # Gating
                gate_weights = self.gate(x)
                
                # Final output
                output = (gate_weights[:, 0:1] * expert1_out + 
                         gate_weights[:, 1:2] * expert2_out + 
                         gate_weights[:, 2:3] * expert3_out)
                
                return output
        
        model = RevolutionaryNN(features_final.shape[1], len(CLASS_DEFINITIONS)).to(DEVICE)
        
        # Revolutionary optimizer with adaptive learning rates
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01, amsgrad=True)
        
        # Revolutionary loss function with multiple objectives
        class RevolutionaryLoss(nn.Module):
            def __init__(self, class_weights, alpha=0.25, gamma=2.0, smoothing=0.1):
                super().__init__()
                self.class_weights = class_weights
                self.alpha = alpha
                self.gamma = gamma
                self.smoothing = smoothing
                
            def forward(self, pred, target):
                # Focal loss component
                ce_loss = F.cross_entropy(pred, target, reduction='none', weight=self.class_weights)
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
                
                # Label smoothing component
                log_probs = F.log_softmax(pred, dim=1)
                smooth_loss = -log_probs.mean(dim=1)
                
                # Combine losses
                total_loss = (1 - self.smoothing) * focal_loss + self.smoothing * smooth_loss
                
                return total_loss.mean()
        
        # Advanced class weighting
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * class_counts)
        # Apply cube root to further reduce extreme weights
        class_weights = np.power(class_weights, 1/3)
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)
        criterion = RevolutionaryLoss(class_weights, alpha=0.25, gamma=2.0, smoothing=0.1)
        
        # Revolutionary scheduler with multiple phases
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.005, epochs=150, steps_per_epoch=1,
            pct_start=0.3, anneal_strategy='cos'
        )
        
        # Training with revolutionary techniques
        model.train()
        best_loss = float('inf')
        patience = 0
        
        for epoch in range(150):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            
            # Advanced regularization cocktail
            l1_lambda = 0.0001
            l2_lambda = 0.001
            elastic_lambda = 0.0005
            
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            elastic_norm = l1_norm + l2_norm
            
            total_loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm + elastic_lambda * elastic_norm
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Advanced early stopping
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience = 0
            else:
                patience += 1
                if patience >= 30:
                    break
        
        # Revolutionary evaluation with test-time augmentation
        model.eval()
        test_predictions = []
        
        # Multiple forward passes with different dropout patterns
        for _ in range(10):
            model.train()  # Enable dropout
            with torch.no_grad():
                outputs = model(X_test_tensor)
                test_predictions.append(F.softmax(outputs, dim=1))
        
        # Ensemble predictions with confidence weighting
        predictions_stack = torch.stack(test_predictions)
        mean_pred = torch.mean(predictions_stack, dim=0)
        confidence = torch.max(mean_pred, dim=1)[0]
        
        # Weight predictions by confidence
        weighted_pred = mean_pred * confidence.unsqueeze(1)
        _, predicted = torch.max(weighted_pred, 1)
        y_pred = predicted.cpu().numpy()
        
        fold_accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(fold_accuracy)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        f_out.write(f"  Revolutionary Preprocessing NN Fold {fold+1}: {fold_accuracy:.4f}\n")
        print(f"    Revolutionary Preprocessing NN Fold {fold+1}: {fold_accuracy:.4f}")
        
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
    
    mean_accuracy = np.mean(accuracies)
    f_out.write(f"Revolutionary Preprocessing NN Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})\n")
    print(f"  Revolutionary Preprocessing NN Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})")
    
    # Classification report and confusion matrix
    report = classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, labels=CLASS_LABELS, zero_division=0)
    f_out.write("Revolutionary Preprocessing NN Classification Report:\n" + report + "\n")
    
    cm = confusion_matrix(all_y_true, all_y_pred, labels=CLASS_LABELS)
    plot_confusion_matrix(cm, CLASS_NAMES, "Revolutionary_Preprocessing_NN", RESULTS_DIR, f_out)
    
    return mean_accuracy

def train_sophisticated_ensemble(stat_features, sequences, labels, f_out, quick_check=False):
    """Train sophisticated ensemble with advanced voting mechanisms."""
    f_out.write("Training Sophisticated Ensemble...\n")
    print("Training Sophisticated Ensemble...")
    
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    
    # Advanced preprocessing
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(stat_features)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_range = [next(skf.split(features_scaled, labels))] if quick_check else list(skf.split(features_scaled, labels))
    
    accuracies = []
    all_y_true, all_y_pred = [], []
    
    for fold, (train_idx, test_idx) in enumerate(tqdm(fold_range, total=1 if quick_check else 5, desc="  Sophisticated Ensemble CV")):
        X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
        y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
        
        # Diverse base models with different strengths
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE)),
            ('et', ExtraTreesClassifier(n_estimators=200, max_depth=12, random_state=RANDOM_STATE)),
            ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=RANDOM_STATE)),
            ('lda', LinearDiscriminantAnalysis()),
            ('qda', QuadraticDiscriminantAnalysis()),
            ('svm', SVC(probability=True, kernel='rbf', C=1.0, random_state=RANDOM_STATE)),
            ('nb', GaussianNB()),
            ('knn', KNeighborsClassifier(n_neighbors=7, weights='distance')),
            ('lr', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
        ]
        
        # Train base models and collect predictions
        base_predictions = []
        base_probabilities = []
        model_weights = []
        
        for name, model in base_models:
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Get predictions and probabilities
                pred = model.predict(X_test)
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_test)
                else:
                    # For models without predict_proba, create one-hot encoding
                    proba = np.zeros((len(pred), len(CLASS_DEFINITIONS)))
                    for i, p in enumerate(pred):
                        proba[i, p] = 1.0
                
                base_predictions.append(pred)
                base_probabilities.append(proba)
                
                # Calculate model weight based on training accuracy
                train_pred = model.predict(X_train)
                train_acc = accuracy_score(y_train, train_pred)
                model_weights.append(train_acc)
                
            except Exception as e:
                # Skip problematic models
                continue
        
        if not base_predictions:
            continue
        
        # Convert to arrays
        base_predictions = np.array(base_predictions)
        base_probabilities = np.array(base_probabilities)
        model_weights = np.array(model_weights)
        
        # Normalize weights
        model_weights = model_weights / np.sum(model_weights)
        
        # Sophisticated voting mechanisms
        
        # 1. Weighted majority voting
        weighted_votes = np.zeros(len(y_test))
        for i in range(len(y_test)):
            vote_counts = np.zeros(len(CLASS_DEFINITIONS))
            for j, pred in enumerate(base_predictions[:, i]):
                vote_counts[pred] += model_weights[j]
            weighted_votes[i] = np.argmax(vote_counts)
        
        # 2. Weighted probability averaging
        weighted_proba = np.average(base_probabilities, axis=0, weights=model_weights)
        prob_predictions = np.argmax(weighted_proba, axis=1)
        
        # 3. Confidence-based voting
        confidence_votes = np.zeros(len(y_test))
        for i in range(len(y_test)):
            confidences = np.max(base_probabilities[:, i, :], axis=1)
            best_model_idx = np.argmax(confidences)
            confidence_votes[i] = base_predictions[best_model_idx, i]
        
        # 4. Neural network meta-learner
        meta_features = base_probabilities.transpose(1, 0, 2).reshape(len(y_test), -1)
        
        # Train neural network meta-learner
        meta_X_train = []
        meta_y_train = []
        
        # Create meta-training data using cross-validation on training set
        meta_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        for meta_train_idx, meta_val_idx in meta_skf.split(X_train, y_train):
            X_meta_train, X_meta_val = X_train[meta_train_idx], X_train[meta_val_idx]
            y_meta_train, y_meta_val = y_train[meta_train_idx], y_train[meta_val_idx]
            
            meta_base_proba = []
            for name, model in base_models:
                try:
                    model.fit(X_meta_train, y_meta_train)
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_meta_val)
                    else:
                        pred = model.predict(X_meta_val)
                        proba = np.zeros((len(pred), len(CLASS_DEFINITIONS)))
                        for i, p in enumerate(pred):
                            proba[i, p] = 1.0
                    meta_base_proba.append(proba)
                except:
                    continue
            
            if meta_base_proba:
                meta_base_proba = np.array(meta_base_proba)
                meta_features_fold = meta_base_proba.transpose(1, 0, 2).reshape(len(y_meta_val), -1)
                meta_X_train.extend(meta_features_fold)
                meta_y_train.extend(y_meta_val)
        
        if meta_X_train:
            meta_X_train = np.array(meta_X_train)
            meta_y_train = np.array(meta_y_train)
            
            # Convert to tensors
            meta_X_train_tensor = torch.FloatTensor(meta_X_train).to(DEVICE)
            meta_X_test_tensor = torch.FloatTensor(meta_features).to(DEVICE)
            meta_y_train_tensor = torch.LongTensor(meta_y_train).to(DEVICE)
            
            # Meta-learner neural network
            meta_model = nn.Sequential(
                nn.Linear(meta_X_train.shape[1], 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, len(CLASS_DEFINITIONS))
            ).to(DEVICE)
            
            optimizer = torch.optim.AdamW(meta_model.parameters(), lr=0.001, weight_decay=0.01)
            criterion = nn.CrossEntropyLoss()
            
            # Train meta-learner
            meta_model.train()
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = meta_model(meta_X_train_tensor)
                loss = criterion(outputs, meta_y_train_tensor)
                loss.backward()
                optimizer.step()
            
            # Meta-learner predictions
            meta_model.eval()
            with torch.no_grad():
                meta_outputs = meta_model(meta_X_test_tensor)
                _, meta_predictions = torch.max(meta_outputs, 1)
                meta_predictions = meta_predictions.cpu().numpy()
            
            del meta_model, optimizer
            torch.cuda.empty_cache()
        else:
            meta_predictions = weighted_votes
        
        # Final ensemble decision using multiple voting mechanisms
        final_predictions = []
        for i in range(len(y_test)):
            votes = [
                weighted_votes[i],
                prob_predictions[i],
                confidence_votes[i],
                meta_predictions[i]
            ]
            
            # Majority vote among the different mechanisms
            final_pred = max(set(votes), key=votes.count)
            final_predictions.append(final_pred)
        
        final_predictions = np.array(final_predictions)
        
        fold_accuracy = accuracy_score(y_test, final_predictions)
        accuracies.append(fold_accuracy)
        all_y_true.extend(y_test)
        all_y_pred.extend(final_predictions)
        
        f_out.write(f"  Sophisticated Ensemble Fold {fold+1}: {fold_accuracy:.4f}\n")
        print(f"    Sophisticated Ensemble Fold {fold+1}: {fold_accuracy:.4f}")
    
    mean_accuracy = np.mean(accuracies)
    f_out.write(f"Sophisticated Ensemble Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})\n")
    print(f"  Sophisticated Ensemble Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})")
    
    # Classification report and confusion matrix
    report = classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, labels=CLASS_LABELS, zero_division=0)
    f_out.write("Sophisticated Ensemble Classification Report:\n" + report + "\n")
    
    cm = confusion_matrix(all_y_true, all_y_pred, labels=CLASS_LABELS)
    plot_confusion_matrix(cm, CLASS_NAMES, "Sophisticated_Ensemble", RESULTS_DIR, f_out)
    
    return mean_accuracy

def train_breakthrough_neural_architecture(stat_features, sequences, labels, f_out, quick_check=False):
    """Train breakthrough neural architecture with cutting-edge techniques."""
    f_out.write("Training Breakthrough Neural Architecture...\n")
    print("Training Breakthrough Neural Architecture...")
    
    from sklearn.preprocessing import RobustScaler, QuantileTransformer
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # Advanced preprocessing
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(stat_features)
    
    # Feature selection
    selector = SelectKBest(f_classif, k=min(50, features_scaled.shape[1]))
    features_selected = selector.fit_transform(features_scaled, labels)
    
    # Quantile transformation
    quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
    features_final = quantile_transformer.fit_transform(features_selected)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_range = [next(skf.split(features_final, labels))] if quick_check else list(skf.split(features_final, labels))
    
    accuracies = []
    all_y_true, all_y_pred = [], []
    
    for fold, (train_idx, test_idx) in enumerate(tqdm(fold_range, total=1 if quick_check else 5, desc="  Breakthrough Neural Architecture CV")):
        X_train, X_test = features_final[train_idx], features_final[test_idx]
        y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
        
        # Breakthrough neural architecture
        class BreakthroughNN(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                
                # Adaptive feature embedding with multi-scale processing
                self.feature_embedding = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(input_dim, 128 * (i+1)),
                        nn.BatchNorm1d(128 * (i+1)),
                        nn.GELU(),
                        nn.Dropout(0.2)
                    ) for i in range(3)
                ])
                
                # Multi-head attention for feature interactions
                self.multi_head_attention = nn.MultiheadAttention(
                    embed_dim=384, num_heads=8, batch_first=True, dropout=0.1
                )
                self.attention_norm = nn.LayerNorm(384)
                
                # Capsule-inspired routing mechanism
                self.capsule_dim = 16
                self.num_capsules = 8
                self.capsule_layers = nn.ModuleList([
                    nn.Linear(384, self.num_capsules * self.capsule_dim)
                    for _ in range(3)
                ])
                
                # Dynamic routing weights
                self.routing_weights = nn.Parameter(torch.randn(3, self.num_capsules, self.num_capsules))
                
                # Mixture of experts with gating
                self.num_experts = 4
                self.experts = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.num_capsules * self.capsule_dim, 128),
                        nn.GELU(),
                        nn.Dropout(0.3),
                        nn.Linear(128, 64),
                        nn.GELU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, output_dim)
                    ) for _ in range(self.num_experts)
                ])
                
                # Gating network
                self.gate = nn.Sequential(
                    nn.Linear(self.num_capsules * self.capsule_dim, 64),
                    nn.GELU(),
                    nn.Linear(64, self.num_experts),
                    nn.Softmax(dim=1)
                )
                
                # Uncertainty estimation head
                self.uncertainty_head = nn.Sequential(
                    nn.Linear(self.num_capsules * self.capsule_dim, 32),
                    nn.GELU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
                
                self.apply(self._init_weights)
            
            def _init_weights(self, module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
            
            def dynamic_routing(self, capsule_outputs):
                """Implement dynamic routing between capsules."""
                batch_size = capsule_outputs[0].size(0)
                
                # Initialize routing coefficients
                b = torch.zeros(batch_size, len(capsule_outputs), self.num_capsules, self.num_capsules).to(DEVICE)
                
                # Routing iterations
                for iteration in range(3):
                    # Softmax over capsules
                    c = F.softmax(b, dim=-1)
                    
                    # Weighted sum of capsule outputs
                    s = torch.zeros(batch_size, self.num_capsules, self.capsule_dim).to(DEVICE)
                    for layer_idx, layer_output in enumerate(capsule_outputs):
                        # Reshape layer output
                        layer_output = layer_output.view(batch_size, self.num_capsules, self.capsule_dim)
                        
                        # Apply routing weights
                        routing_weight = self.routing_weights[layer_idx]
                        weighted_output = torch.matmul(c[:, layer_idx], routing_weight)
                        
                        # Weighted sum
                        s += torch.matmul(weighted_output.transpose(-2, -1), layer_output)
                    
                    # Squash the output
                    s_norm = torch.norm(s, dim=-1, keepdim=True)
                    s_squashed = (s_norm / (1 + s_norm**2)) * (s / s_norm)
                    
                    # Update routing coefficients
                    b += torch.matmul(s_squashed.unsqueeze(-2), layer_output.unsqueeze(-1)).squeeze(-1)
                
                return s_squashed
            
            def forward(self, x):
                # Multi-scale feature embedding
                multi_scale_features = [layer(x) for layer in self.feature_embedding]
                multi_path = torch.cat(multi_scale_features, dim=1)
                
                # Multi-head attention
                multi_path_seq = multi_path.unsqueeze(1)
                attn_out, _ = self.multi_head_attention(multi_path_seq, multi_path_seq, multi_path_seq)
                attn_out = self.attention_norm(multi_path_seq + attn_out).squeeze(1)
                
                # Capsule layers
                capsule_outputs = [
                    layer(attn_out).view(-1, self.num_capsules, self.capsule_dim)
                    for layer in self.capsule_layers
                ]
                
                # Dynamic routing
                routed_capsules = self.dynamic_routing(capsule_outputs)
                
                # Flatten routed capsules
                x = routed_capsules.view(x.size(0), -1)
                
                # Uncertainty estimation
                uncertainty = self.uncertainty_head(x)
                
                # Mixture of experts
                expert_outputs = [expert(x) for expert in self.experts]
                
                # Gating
                gate_weights = self.gate(x)
                
                # Weighted expert outputs
                output = torch.stack(expert_outputs, dim=1)
                output = torch.sum(output * gate_weights.unsqueeze(-1), dim=1)
                
                return output
        
        model = BreakthroughNN(features_final.shape[1], len(CLASS_DEFINITIONS)).to(DEVICE)
        
        # Advanced optimizer with different learning rates
        param_groups = [
            {'params': [p for n, p in model.named_parameters() if 'expert' in n], 'lr': 0.003},
            {'params': [p for n, p in model.named_parameters() if 'feature_embedding' in n], 'lr': 0.001},
            {'params': [p for n, p in model.named_parameters() if 'capsule' in n], 'lr': 0.0015},
            {'params': [p for n, p in model.named_parameters() if 'attention' in n or 'gate' in n], 'lr': 0.002}
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.02, amsgrad=True)
        
        # Advanced loss with multiple components
        class AdvancedLoss(nn.Module):
            def __init__(self, class_weights, smoothing=0.1):
                super().__init__()
                self.class_weights = class_weights
                self.smoothing = smoothing
                
            def forward(self, pred, target):
                # Label smoothing cross entropy
                confidence = 1.0 - self.smoothing
                log_probs = F.log_softmax(pred, dim=1)
                nll_loss = F.nll_loss(log_probs, target, weight=self.class_weights, reduction='none')
                smooth_loss = -log_probs.mean(dim=1)
                loss = confidence * nll_loss + self.smoothing * smooth_loss
                return loss.mean()
        
        # Class weights
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)
        criterion = AdvancedLoss(class_weights, smoothing=0.1)
        
        # Advanced scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=[0.003, 0.001, 0.0015, 0.002], 
            epochs=100, steps_per_epoch=1
        )
        
        # Training with advanced techniques
        model.train()
        best_loss = float('inf')
        patience = 0
        
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            
            # Additional regularization
            l1_lambda = 0.0001
            l2_lambda = 0.001
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss += l1_lambda * l1_norm + l2_lambda * l2_norm
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()
            scheduler.step()
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience = 0
            else:
                patience += 1
                if patience >= 20:
                    break
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            y_pred = predicted.cpu().numpy()
        
        fold_accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(fold_accuracy)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        f_out.write(f"  Breakthrough Neural Architecture Fold {fold+1}: {fold_accuracy:.4f}\n")
        print(f"    Breakthrough Neural Architecture Fold {fold+1}: {fold_accuracy:.4f}")
        
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
    
    mean_accuracy = np.mean(accuracies)
    f_out.write(f"Breakthrough Neural Architecture Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})\n")
    print(f"  Breakthrough Neural Architecture Mean CV Accuracy: {mean_accuracy:.4f} (+/- {np.std(accuracies):.4f})")
    
    # Classification report and confusion matrix
    report = classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, labels=CLASS_LABELS, zero_division=0)
    f_out.write("Breakthrough Neural Architecture Classification Report:\n" + report + "\n")
    
    cm = confusion_matrix(all_y_true, all_y_pred, labels=CLASS_LABELS)
    plot_confusion_matrix(cm, CLASS_NAMES, "Breakthrough_Neural_Architecture", RESULTS_DIR, f_out)
    
    return mean_accuracy
