"""
Base class for all baseline models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class BaselineModel(ABC):
    """
    Abstract base class for all baseline models in the SOTA comparison.

    This provides a unified interface for:
    - PyTorch-based models (inherit from nn.Module)
    - Sklearn-based models (like ROCKET)

    All models must implement:
    - fit(X_train, y_train): Train the model
    - predict(X): Get class predictions
    - predict_proba(X): Get class probabilities
    - count_parameters(): Return number of trainable parameters
    """

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training data (n_samples, seq_len, n_features)
            y_train: Training labels (n_samples,)
            X_val: Optional validation data
            y_val: Optional validation labels
            **kwargs: Additional training arguments

        Returns:
            Dict with training info (e.g., training_time, best_epoch)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get class predictions.

        Args:
            X: Input data (n_samples, seq_len, n_features)

        Returns:
            Predicted labels (n_samples,)
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get class probabilities.

        Args:
            X: Input data (n_samples, seq_len, n_features)

        Returns:
            Class probabilities (n_samples, n_classes)
        """
        pass

    @abstractmethod
    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        pass

    def get_model_name(self) -> str:
        """Return model name for display."""
        return self.__class__.__name__

    def to(self, device: torch.device) -> 'BaselineModel':
        """Move model to device (no-op for non-PyTorch models)."""
        return self


class PyTorchBaselineModel(BaselineModel, nn.Module):
    """
    Base class for PyTorch-based baseline models.

    Provides common functionality for neural network models:
    - Training loop with early stopping
    - GPU support
    - Standard metrics computation
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        device: Optional[torch.device] = None,
    ):
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seq_len = seq_len
        self._device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device) -> 'PyTorchBaselineModel':
        """Move model to device."""
        self._device = device
        return nn.Module.to(self, device)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 20,
        verbose: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Standard training loop for PyTorch models.
        """
        import time
        from torch.utils.data import DataLoader, TensorDataset

        self.to(self._device)

        # Convert to tensors
        X_train_t = torch.from_numpy(X_train).float()
        y_train_t = torch.from_numpy(y_train).long()

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_t = torch.from_numpy(X_val).float()
            y_val_t = torch.from_numpy(y_val).long()
            val_dataset = TensorDataset(X_val_t, y_val_t)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Class weights for imbalanced data
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        weights = torch.tensor(class_weights, dtype=torch.float32).to(self._device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        patience_counter = 0

        start_time = time.time()

        for epoch in range(epochs):
            # Training
            self.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.to(self._device)
                labels = labels.to(self._device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(train_dataset)

            # Validation
            val_loss = train_loss
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.to(self._device)
                        labels = labels.to(self._device)
                        outputs = self(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * inputs.size(0)
                val_loss /= len(val_dataset)

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        training_time = time.time() - start_time

        # Restore best model
        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        return {
            'training_time': training_time,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions."""
        self.eval()
        X_t = torch.from_numpy(X).float().to(self._device)

        with torch.no_grad():
            outputs = self(X_t)
            _, predicted = outputs.max(1)

        return predicted.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get class probabilities."""
        self.eval()
        X_t = torch.from_numpy(X).float().to(self._device)

        with torch.no_grad():
            outputs = self(X_t)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()
