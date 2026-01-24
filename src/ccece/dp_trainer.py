"""
CCECE Paper: Differentially Private Training Module

Implements DP-SGD training using Opacus for privacy-preserving model training.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from tqdm import tqdm

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False

from .trainer import (
    Trainer, TrainingConfig, EvaluationMetrics, TrainingHistory,
    EarlyStopping
)


# =============================================================================
# DP CONFIGURATION
# =============================================================================

@dataclass
class DPConfig:
    """Differential Privacy configuration."""
    target_epsilon: float = 8.0  # Privacy budget (lower = more private)
    target_delta: float = 1e-5  # Probability of privacy failure
    max_grad_norm: float = 1.0  # Per-sample gradient clipping norm
    noise_multiplier: Optional[float] = None  # If set, overrides auto-calibration


@dataclass
class DPTrainingHistory(TrainingHistory):
    """Training history with DP-specific metrics."""
    epsilons: List[float] = field(default_factory=list)
    final_epsilon: float = 0.0
    final_delta: float = 0.0


# =============================================================================
# DP TRAINER
# =============================================================================

class DPTrainer(Trainer):
    """
    Differentially Private Trainer using Opacus.

    Extends the base Trainer with DP-SGD capabilities:
    - Per-sample gradient clipping
    - Gradient noise injection
    - Privacy accounting (tracks epsilon)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
        dp_config: DPConfig,
    ):
        """
        Args:
            model: PyTorch model to train
            config: Training configuration
            device: Device to train on
            dp_config: Differential privacy configuration
        """
        if not OPACUS_AVAILABLE:
            raise ImportError(
                "Opacus is required for differential privacy training. "
                "Install with: pip install opacus"
            )

        super().__init__(model, config, device)
        self.dp_config = dp_config
        self.privacy_engine = None
        self.history = DPTrainingHistory()

    def _validate_and_fix_model(self):
        """Validate model compatibility with Opacus and fix if needed."""
        errors = ModuleValidator.validate(self.model, strict=False)

        if errors:
            print(f"Model has {len(errors)} Opacus compatibility issues. Attempting to fix...")
            self.model = ModuleValidator.fix(self.model)
            print("Model fixed for DP compatibility.")

            # Revalidate
            errors = ModuleValidator.validate(self.model, strict=False)
            if errors:
                raise ValueError(
                    f"Could not fix model for DP training. Remaining errors: {errors}"
                )

    def _setup_training(self, train_labels: np.ndarray):
        """Setup optimizer with DP privacy engine."""
        # First validate/fix the model
        self._validate_and_fix_model()

        # Setup optimizer (without weight decay - not well supported in DP-SGD)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        # We don't use LR scheduler with DP-SGD (complicates privacy analysis)
        self.scheduler = None

        # Class weights for imbalanced data
        if self.config.use_class_weights:
            class_counts = np.bincount(train_labels)
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum() * len(class_weights)
            weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def _setup_privacy_engine(self, train_loader: DataLoader):
        """Setup Opacus privacy engine."""
        self.privacy_engine = PrivacyEngine()

        # Make the model, optimizer, and data loader private
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=train_loader,
            epochs=self.config.epochs,
            target_epsilon=self.dp_config.target_epsilon,
            target_delta=self.dp_config.target_delta,
            max_grad_norm=self.dp_config.max_grad_norm,
        )

        print(f"DP Training enabled:")
        print(f"  Target ε: {self.dp_config.target_epsilon}")
        print(f"  Target δ: {self.dp_config.target_delta}")
        print(f"  Max grad norm: {self.dp_config.max_grad_norm}")
        print(f"  Noise multiplier: {self.optimizer.noise_multiplier:.4f}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_labels: np.ndarray,
        verbose: bool = True,
    ) -> DPTrainingHistory:
        """
        Train the model with differential privacy.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            train_labels: Training labels (for class weights)
            verbose: Whether to show progress

        Returns:
            DPTrainingHistory with losses, accuracies, and privacy metrics
        """
        self._setup_training(train_labels)
        self._setup_privacy_engine(train_loader)

        early_stopping = EarlyStopping(patience=self.config.early_stopping_patience)

        epoch_iter = range(self.config.epochs)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc="DP Training", unit="epoch")

        for epoch in epoch_iter:
            # Training with DP
            train_loss, train_acc = self._train_epoch_dp()
            self.history.train_losses.append(train_loss)
            self.history.train_accuracies.append(train_acc)

            # Validation (no DP needed for validation)
            val_loss, val_acc = self._validate_epoch(val_loader)
            self.history.val_losses.append(val_loss)
            self.history.val_accuracies.append(val_acc)

            # Track privacy spent
            epsilon = self.privacy_engine.get_epsilon(self.dp_config.target_delta)
            self.history.epsilons.append(epsilon)

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
                    'val_acc': f'{val_acc:.4f}',
                    'ε': f'{epsilon:.2f}',
                    'best': f'{self.history.best_val_accuracy:.4f}',
                })

            # Early stopping based on validation loss
            if early_stopping(val_loss):
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break

            # Stop if privacy budget exceeded
            if epsilon > self.dp_config.target_epsilon * 1.5:
                if verbose:
                    print(f"\nStopping: Privacy budget significantly exceeded (ε={epsilon:.2f})")
                break

        # Record final privacy
        self.history.final_epsilon = self.privacy_engine.get_epsilon(self.dp_config.target_delta)
        self.history.final_delta = self.dp_config.target_delta

        # Restore best model
        if self.best_model_state is not None:
            # Note: With DP, the state dict has different structure
            # We need to handle the GradSampleModule wrapper
            try:
                self.model.load_state_dict(self.best_model_state)
            except RuntimeError:
                # Try loading into the wrapped module
                if hasattr(self.model, '_module'):
                    self.model._module.load_state_dict(self.best_model_state)

        print(f"\nFinal privacy: (ε={self.history.final_epsilon:.2f}, δ={self.history.final_delta})")

        return self.history

    def _train_epoch_dp(self) -> Tuple[float, float]:
        """Train for one epoch with DP."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Opacus handles gradient clipping and noise internally
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get the current privacy spent (epsilon, delta)."""
        if self.privacy_engine is None:
            return 0.0, 0.0
        return (
            self.privacy_engine.get_epsilon(self.dp_config.target_delta),
            self.dp_config.target_delta
        )


def check_dp_compatibility(model: nn.Module) -> Tuple[bool, List[str]]:
    """
    Check if a model is compatible with Opacus DP training.

    Args:
        model: PyTorch model to check

    Returns:
        (is_compatible, list_of_errors)
    """
    if not OPACUS_AVAILABLE:
        return False, ["Opacus is not installed"]

    errors = ModuleValidator.validate(model, strict=False)
    return len(errors) == 0, errors


def get_dp_model(model: nn.Module) -> nn.Module:
    """
    Convert a model to be DP-compatible.

    Args:
        model: Original model

    Returns:
        DP-compatible model
    """
    if not OPACUS_AVAILABLE:
        raise ImportError("Opacus is required")

    return ModuleValidator.fix(model)
