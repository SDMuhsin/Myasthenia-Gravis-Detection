import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import warnings
import gc

class SaccadeStandardScaler:
    """Standard scaler for saccade data."""
    def __init__(self): 
        self.scaler = StandardScaler()
    
    def fit(self, data_items): 
        self.scaler.fit(np.vstack([item['data'] for item in data_items]))
    
    def transform(self, sequence): 
        return self.scaler.transform(sequence)

class EarlyStopper:
    """Early stopping utility."""
    def __init__(self, patience=1, min_delta=0):
        self.patience, self.min_delta, self.counter, self.min_validation_loss = patience, min_delta, 0, np.inf
    
    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience: 
                return True
        return False

def calculate_class_weights(items, label_map):
    """Calculates class weights inversely proportional to class frequencies."""
    labels = [label_map[item['label']] for item in items]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float)

def pad_sequence(sequence, target_len, num_features):
    """Pads or truncates sequence to target length."""
    current_len = sequence.shape[0]
    if current_len >= target_len: 
        return sequence[:target_len, :]
    padding = np.zeros((target_len - current_len, num_features), dtype=np.float32)
    return np.vstack((sequence, padding))

def subsample_data(items, subsample_factor=10):
    """Subsamples data by taking every nth item for quick training."""
    return items[::subsample_factor]

class SaccadeDataset(Dataset):
    """PyTorch dataset for saccade data."""
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
        sequence, label = item['data'], item['label']
        if self.scaler: 
            sequence = self.scaler.transform(sequence)
        processed_sequence = pad_sequence(sequence, self.target_seq_len, self.num_features)
        final_label = self.label_map[label]
        return torch.from_numpy(processed_sequence).float(), torch.tensor(final_label, dtype=torch.long)

class Attention(nn.Module):
    """Attention mechanism for RNN."""
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.bias, self.feature_dim, self.step_dim = bias, feature_dim, step_dim
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        if bias: 
            self.b = nn.Parameter(torch.zeros(step_dim))
    
    def forward(self, x, mask=None):
        eij = torch.mm(x.contiguous().view(-1, self.feature_dim), self.weight).view(-1, self.step_dim)
        if self.bias: 
            eij = eij + self.b
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        if mask is not None: 
            a = a * mask
        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)
        return torch.sum(x * torch.unsqueeze(a, -1), 1)

class SimpleLSTM(nn.Module):
    """Simple LSTM model for binary classification."""
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1, dropout_prob=0.3):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout_prob if n_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        lstm_out, (hidden, _) = self.lstm(x)
        # Use the last hidden state
        return self.fc(hidden[-1])

class SimpleGRU(nn.Module):
    """Simple GRU model for binary classification."""
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1, dropout_prob=0.3):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout_prob if n_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        gru_out, hidden = self.gru(x)
        # Use the last hidden state
        return self.fc(hidden[-1])

class SaccadeRNN_Small(nn.Module):
    """Small bidirectional GRU with attention for binary classification."""
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, seq_len, dropout_prob):
        super(SaccadeRNN_Small, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout_prob, bidirectional=True)
        self.attention = Attention(hidden_dim * 2, seq_len)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), 
            nn.ReLU(), 
            nn.Dropout(dropout_prob),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        context_vector = self.attention(gru_out)
        return self.fc(context_vector)

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch."""
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
    return running_loss / total_samples, (correct_predictions.double() / total_samples).item()

def evaluate_epoch(model, dataloader, criterion, device):
    """Evaluate model for one epoch."""
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

def plot_loss_curves(train_losses, val_losses, model_name, filepath):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'{model_name} - Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()

def plot_dl_confusion_matrix(cm, class_names, title, filepath):
    """Plot confusion matrix for deep learning models."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def get_small_dl_models(input_dim, output_dim, seq_len):
    """Returns small deep learning models for quick training."""
    return {
        "Simple LSTM": SimpleLSTM(input_dim, 32, output_dim, n_layers=1, dropout_prob=0.3),
        "Simple GRU": SimpleGRU(input_dim, 32, output_dim, n_layers=1, dropout_prob=0.3),
        "Small BiGRU+Attention": SaccadeRNN_Small(input_dim, 32, output_dim, 1, seq_len, 0.3)
    }
