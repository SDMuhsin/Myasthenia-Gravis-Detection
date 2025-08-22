import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

# Import from main experiment
from exp_12 import (
    CLASS_DEFINITIONS, CLASS_NAMES, CLASS_LABELS, DEVICE, RANDOM_STATE,
    SaccadeSequenceDataset, collate_fn, plot_confusion_matrix, RESULTS_DIR, EXP_PREFIX
)

# --- Alternative Neural Network Architectures ---

class CNNLSTMClassifier(nn.Module):
    """CNN-LSTM hybrid architecture for saccade classification."""
    
    def __init__(self, input_dim, output_dim, cnn_channels=64, lstm_hidden=128, dropout=0.2):
        super().__init__()
        
        # 1D CNN layers for local pattern extraction
        self.conv1 = nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2)
        
        self.batch_norm1 = nn.BatchNorm1d(cnn_channels)
        self.batch_norm2 = nn.BatchNorm1d(cnn_channels)
        self.batch_norm3 = nn.BatchNorm1d(cnn_channels)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden, batch_first=True, bidirectional=True)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.BatchNorm1d(lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.BatchNorm1d(lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, output_dim)
        )
        
    def forward(self, x, lengths):
        batch_size, seq_len, input_dim = x.size()
        
        # Transpose for CNN (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # CNN layers
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = self.dropout(x)
        
        # Transpose back for LSTM (batch, seq_len, channels)
        x = x.transpose(1, 2)
        
        # Pack sequences for LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM forward pass
        packed_output, (hidden, cell) = self.lstm(packed_input)
        
        # Use final hidden state
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # Concatenate forward and backward
        
        # Classification
        output = self.classifier(final_hidden)
        return output

class TransformerClassifier(nn.Module):
    """Lightweight Transformer architecture for saccade classification."""
    
    def __init__(self, input_dim, output_dim, d_model=128, nhead=8, num_layers=3, dropout=0.2):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))  # Max sequence length 1000
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, 
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
    def forward(self, x, lengths):
        batch_size, seq_len, _ = x.size()
        
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_enc
        
        # Create padding mask
        mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len) >= lengths.unsqueeze(1)
        
        # Transformer forward pass
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Global average pooling (excluding padding)
        mask_expanded = (~mask).unsqueeze(-1).float()
        x_masked = x * mask_expanded
        x_pooled = torch.sum(x_masked, dim=1) / torch.sum(mask_expanded, dim=1)
        
        # Classification
        output = self.classifier(x_pooled)
        return output

class EnsembleClassifier(nn.Module):
    """Ensemble of multiple small networks."""
    
    def __init__(self, input_dim, output_dim, num_models=3, hidden_dim=64):
        super().__init__()
        
        self.num_models = num_models
        self.models = nn.ModuleList()
        
        for i in range(num_models):
            model = nn.Sequential(
                nn.Linear(input_dim * 3, hidden_dim),  # mean, std, last
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, output_dim)
            )
            self.models.append(model)
    
    def forward(self, x, lengths):
        batch_size = x.size(0)
        
        # Extract features for each sequence
        features_list = []
        for i in range(batch_size):
            seq_len = lengths[i].item()
            seq = x[i, :seq_len, :]
            
            # Simple statistics
            seq_mean = torch.mean(seq, dim=0)
            seq_std = torch.std(seq, dim=0)
            seq_last = seq[-1, :]
            
            seq_features = torch.cat([seq_mean, seq_std, seq_last])
            features_list.append(seq_features)
        
        features = torch.stack(features_list)
        
        # Get predictions from all models
        outputs = []
        for model in self.models:
            output = model(features)
            outputs.append(output)
        
        # Average ensemble
        ensemble_output = torch.stack(outputs).mean(dim=0)
        return ensemble_output

# --- Training and Evaluation Functions ---

def train_and_evaluate_architecture(sequences, labels, n_features, n_classes, f_out, 
                                  architecture_class, model_name, **model_kwargs):
    """Train and evaluate a specific architecture."""
    f_out.write(f"\n--- Evaluating {model_name} ---\n")
    f_out.write(f"Architecture: {architecture_class.__name__}\n")
    f_out.write(f"Parameters: {model_kwargs}\n")
    print(f"\nEvaluating {model_name}")
    print(f"Architecture: {architecture_class.__name__}")
    
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    accuracies = []
    all_y_true, all_y_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(tqdm(skf.split(sequences, labels), total=5, desc=f"  CV for {model_name}")):
        X_train_seq = [sequences[i] for i in train_idx]
        X_test_seq = [sequences[i] for i in test_idx]
        y_train = np.array(labels)[train_idx]
        y_test = np.array(labels)[test_idx]
        
        # Create datasets
        train_dataset = SaccadeSequenceDataset(X_train_seq, y_train, augment=True)
        test_dataset = SaccadeSequenceDataset(X_test_seq, y_test, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

        # Initialize model
        model = architecture_class(input_dim=n_features, output_dim=n_classes, **model_kwargs).to(DEVICE)
        
        # Optimizer and criterion
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        for epoch in range(20):  # Shorter training for quick iterations
            for seq_padded, labels_batch, lengths in train_loader:
                seq_padded, labels_batch = seq_padded.to(DEVICE), labels_batch.to(DEVICE)
                lengths = lengths.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(seq_padded, lengths)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

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
        del model, optimizer
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

# --- Analysis Functions ---

def analyze_sequence_statistics(raw_items_list, f_out):
    """Analyze sequence statistics to inform architecture design."""
    f_out.write("\n" + "="*60 + "\nSequence Statistics Analysis\n" + "="*60 + "\n")
    print("\n" + "="*40 + "\nSequence Statistics Analysis\n" + "="*40)
    
    lengths = [item['original_length'] for item in raw_items_list]
    
    stats = {
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths),
        'median_length': np.median(lengths),
        'q25_length': np.percentile(lengths, 25),
        'q75_length': np.percentile(lengths, 75)
    }
    
    f_out.write("Sequence Length Statistics:\n")
    for key, value in stats.items():
        f_out.write(f"  {key}: {value:.2f}\n")
        print(f"  {key}: {value:.2f}")
    
    # Class-wise statistics
    class_lengths = {}
    for item in raw_items_list:
        class_name = item['class_name']
        if class_name not in class_lengths:
            class_lengths[class_name] = []
        class_lengths[class_name].append(item['original_length'])
    
    f_out.write("\nClass-wise Length Statistics:\n")
    print("\nClass-wise Length Statistics:")
    for class_name, lengths_list in class_lengths.items():
        mean_len = np.mean(lengths_list)
        std_len = np.std(lengths_list)
        f_out.write(f"  {class_name}: {mean_len:.2f} ± {std_len:.2f}\n")
        print(f"  {class_name}: {mean_len:.2f} ± {std_len:.2f}")
    
    return stats

def plot_training_curves(train_losses, val_accuracies, model_name, results_dir):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Training loss
    ax1.plot(train_losses)
    ax1.set_title(f'{model_name} - Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Validation accuracy
    ax2.plot(val_accuracies)
    ax2.set_title(f'{model_name} - Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'{EXP_PREFIX}{model_name}_training_curves.png')
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

def memory_usage_analysis():
    """Analyze GPU memory usage."""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        max_memory = torch.cuda.max_memory_allocated() / 1024**3   # GB
        
        print(f"\nGPU Memory Usage:")
        print(f"  Currently allocated: {memory_allocated:.2f} GB")
        print(f"  Currently reserved: {memory_reserved:.2f} GB")
        print(f"  Max allocated: {max_memory:.2f} GB")
        
        return {
            'allocated': memory_allocated,
            'reserved': memory_reserved,
            'max_allocated': max_memory
        }
    else:
        print("CUDA not available")
        return None

def generate_architecture_recommendations(sequence_stats, f_out):
    """Generate architecture recommendations based on data analysis."""
    f_out.write("\n" + "="*60 + "\nArchitecture Recommendations\n" + "="*60 + "\n")
    print("\n" + "="*40 + "\nArchitecture Recommendations\n" + "="*40)
    
    mean_length = sequence_stats['mean_length']
    max_length = sequence_stats['max_length']
    
    recommendations = []
    
    if mean_length < 200:
        recommendations.append("Short sequences: Consider CNN-based architectures for local pattern extraction")
    
    if max_length > 1000:
        recommendations.append("Long sequences present: Use attention mechanisms or hierarchical approaches")
    
    if sequence_stats['std_length'] > sequence_stats['mean_length'] * 0.5:
        recommendations.append("High length variability: Implement robust padding and masking strategies")
    
    recommendations.extend([
        "Memory efficiency: Use gradient checkpointing for deeper models",
        "Data augmentation: Apply noise injection and temporal perturbations",
        "Ensemble methods: Combine multiple smaller models for better performance",
        "Regularization: Use dropout, batch normalization, and weight decay"
    ])
    
    f_out.write("Recommendations:\n")
    for i, rec in enumerate(recommendations, 1):
        f_out.write(f"  {i}. {rec}\n")
        print(f"  {i}. {rec}")
    
    return recommendations

if __name__ == '__main__':
    print("Support module for Experiment 12")
    print("This module provides additional architectures and analysis functions")
