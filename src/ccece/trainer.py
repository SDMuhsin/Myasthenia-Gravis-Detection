"""
CCECE Paper: Training and Evaluation Module

Handles training loops, evaluation, and metrics computation.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from tqdm import tqdm


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 15
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    grad_clip_norm: float = 1.0
    use_class_weights: bool = True


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    accuracy: float
    sensitivity: float  # Recall for positive class (MG)
    specificity: float  # Recall for negative class (HC)
    precision: float
    f1: float
    auc_roc: float
    confusion_matrix: np.ndarray
    predictions: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding arrays)."""
        return {
            'accuracy': self.accuracy,
            'sensitivity': self.sensitivity,
            'specificity': self.specificity,
            'precision': self.precision,
            'f1': self.f1,
            'auc_roc': self.auc_roc,
        }

    def __str__(self) -> str:
        return (
            f"Accuracy: {self.accuracy:.4f} | "
            f"Sens: {self.sensitivity:.4f} | "
            f"Spec: {self.specificity:.4f} | "
            f"AUC: {self.auc_roc:.4f}"
        )


@dataclass
class TrainingHistory:
    """Container for training history."""
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)
    best_val_accuracy: float = 0.0
    best_epoch: int = 0


# =============================================================================
# DATASET
# =============================================================================

class SaccadeDataset(Dataset):
    """PyTorch Dataset for saccade sequences."""

    def __init__(
        self,
        items: List[Dict],
        seq_len: int,
        scaler: Optional[Any] = None,
    ):
        """
        Args:
            items: List of dicts with 'data' and 'label' keys
            seq_len: Target sequence length (pad/truncate to this)
            scaler: Optional scaler with transform method
        """
        self.items = items
        self.seq_len = seq_len
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        data = item['data'].copy()
        label = item['label']

        # Apply scaling if provided
        if self.scaler is not None:
            data = self.scaler.transform(data)

        # Pad or truncate to target length
        data = self._pad_or_truncate(data)

        return (
            torch.from_numpy(data).float(),
            torch.tensor(label, dtype=torch.long)
        )

    def _pad_or_truncate(self, data: np.ndarray) -> np.ndarray:
        """Pad with zeros or truncate to target sequence length."""
        current_len = data.shape[0]
        num_features = data.shape[1]

        if current_len >= self.seq_len:
            return data[:self.seq_len]
        else:
            padding = np.zeros((self.seq_len - current_len, num_features), dtype=np.float32)
            return np.vstack([data, padding])


# =============================================================================
# SCALER
# =============================================================================

class SequenceScaler:
    """Standard scaler for sequence data."""

    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False

    def fit(self, items: List[Dict]) -> 'SequenceScaler':
        """Fit scaler on training data."""
        all_data = np.vstack([item['data'] for item in items])
        self.mean = all_data.mean(axis=0)
        self.std = all_data.std(axis=0) + 1e-8
        self.fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform a single sequence."""
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before transform")
        return (data - self.mean) / self.std


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """
    Trainer class for time series classification models.

    Handles training loop, validation, early stopping, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
    ):
        """
        Args:
            model: PyTorch model to train
            config: Training configuration
            device: Device to train on (cuda/cpu)
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.history = TrainingHistory()
        self.best_model_state = None

    def _setup_training(self, train_labels: np.ndarray):
        """Setup optimizer, scheduler, and loss function."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.lr_scheduler_factor,
            patience=self.config.lr_scheduler_patience,
        )

        # Class weights for imbalanced data
        if self.config.use_class_weights:
            class_counts = np.bincount(train_labels)
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum() * len(class_weights)
            weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_labels: np.ndarray,
        verbose: bool = True,
    ) -> TrainingHistory:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            train_labels: Training labels (for class weights)
            verbose: Whether to show progress

        Returns:
            TrainingHistory with losses and accuracies
        """
        self._setup_training(train_labels)
        early_stopping = EarlyStopping(patience=self.config.early_stopping_patience)

        epoch_iter = range(self.config.epochs)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc="Training", unit="epoch")

        for epoch in epoch_iter:
            # Training
            train_loss, train_acc = self._train_epoch(train_loader)
            self.history.train_losses.append(train_loss)
            self.history.train_accuracies.append(train_acc)

            # Validation
            val_loss, val_acc = self._validate_epoch(val_loader)
            self.history.val_losses.append(val_loss)
            self.history.val_accuracies.append(val_acc)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Track best model
            if val_acc > self.history.best_val_accuracy:
                self.history.best_val_accuracy = val_acc
                self.history.best_epoch = epoch
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

            # Update progress bar
            if verbose:
                epoch_iter.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'val_acc': f'{val_acc:.4f}',
                    'best': f'{self.history.best_val_accuracy:.4f}',
                })

            # Early stopping
            if early_stopping(val_loss):
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self.history

    def _is_concept_bottleneck_model(self) -> bool:
        """Check if the model is a ConceptBottleneckTCN."""
        return hasattr(self.model, 'forward_with_concepts')

    def _is_enhanced_tcdn_model(self) -> bool:
        """Check if the model is an EnhancedTCDN with fatigue-aware loss."""
        return (hasattr(self.model, 'compute_loss') and
                hasattr(self.model, 'fatigue_loss') and
                not hasattr(self.model, 'forward_with_concepts'))

    def _train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        is_concept_model = self._is_concept_bottleneck_model()
        is_enhanced_tcdn = self._is_enhanced_tcdn_model()

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if is_concept_model:
                # Concept Bottleneck Model training
                outputs, concepts_pred = self.model.forward_with_concepts(inputs)
                concepts_target = self.model.get_concept_targets(inputs)

                # Get class weights from criterion if available
                class_weights = None
                if hasattr(self.criterion, 'weight') and self.criterion.weight is not None:
                    class_weights = self.criterion.weight

                loss, _ = self.model.compute_loss(
                    outputs, labels, concepts_pred, concepts_target, class_weights
                )
            elif is_enhanced_tcdn:
                # Enhanced TCDN with fatigue-aware loss
                outputs = self.model(inputs)

                # Get class weights from criterion if available
                class_weights = None
                if hasattr(self.criterion, 'weight') and self.criterion.weight is not None:
                    class_weights = self.criterion.weight

                loss, _ = self.model.compute_loss(outputs, labels, class_weights)
            else:
                # Standard model training
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip_norm
            )

            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    def _validate_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        return total_loss / total, correct / total

    def evaluate(self, dataloader: DataLoader) -> EvaluationMetrics:
        """
        Evaluate the model and compute metrics.

        Args:
            dataloader: Data loader for evaluation

        Returns:
            EvaluationMetrics with all computed metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)

                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of positive class

        predictions = np.array(all_preds)
        labels = np.array(all_labels)
        probabilities = np.array(all_probs)

        # Compute metrics
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel()

        return EvaluationMetrics(
            accuracy=accuracy_score(labels, predictions),
            sensitivity=tp / (tp + fn) if (tp + fn) > 0 else 0.0,  # Recall for MG
            specificity=tn / (tn + fp) if (tn + fp) > 0 else 0.0,  # Recall for HC
            precision=precision_score(labels, predictions, zero_division=0),
            f1=f1_score(labels, predictions, zero_division=0),
            auc_roc=roc_auc_score(labels, probabilities),
            confusion_matrix=cm,
            predictions=predictions,
            labels=labels,
            probabilities=probabilities,
        )


def create_data_loaders(
    train_items: List[Dict],
    val_items: List[Dict],
    seq_len: int,
    batch_size: int,
    scaler: Optional[SequenceScaler] = None,
) -> Tuple[DataLoader, DataLoader, SequenceScaler]:
    """
    Create training and validation data loaders.

    Args:
        train_items: Training data items
        val_items: Validation data items
        seq_len: Target sequence length
        batch_size: Batch size
        scaler: Optional pre-fitted scaler (if None, fits on train)

    Returns:
        train_loader, val_loader, scaler
    """
    # Fit scaler on training data if not provided
    if scaler is None:
        scaler = SequenceScaler().fit(train_items)

    train_dataset = SaccadeDataset(train_items, seq_len, scaler)
    val_dataset = SaccadeDataset(val_items, seq_len, scaler)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, scaler
