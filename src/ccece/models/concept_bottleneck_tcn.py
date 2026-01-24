"""
CCECE Paper: Concept Bottleneck TCN for Explainable MG Detection

This model provides intrinsic explainability by forcing predictions through
a layer of clinically meaningful concepts. The concepts are derived from
known discriminative features identified in the root cause analysis.

Key Features:
1. TCN backbone extracts temporal features
2. Concept layer predicts clinical concepts (tracking error severity, etc.)
3. Classification is performed ONLY through the concept bottleneck
4. Provides human-interpretable explanations

Reference:
- Koh et al., "Concept Bottleneck Models", ICML 2020
- ROOT_CAUSE_ANALYSIS.md - Tracking errors are the key discriminative signal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass

from .base import BaseTimeSeriesModel


# =============================================================================
# CLINICAL CONCEPT DEFINITIONS
# =============================================================================

@dataclass
class ClinicalConcepts:
    """
    Clinical concepts derived from eye-tracking data.

    These concepts map to known physiological markers that distinguish
    MG patients from healthy controls based on the root cause analysis.
    """
    # Concept names for display
    NAMES = [
        'Horizontal Tracking Error',    # 0: Mean |Error H Left + Right|
        'Vertical Tracking Error',      # 1: Mean |Error V Left + Right|
        'Tracking Variability',         # 2: Std of tracking errors
        'Saccade Smoothness',           # 3: Inverse of velocity variance
        'Fatigue Effect',               # 4: Q4 error - Q1 error
    ]

    # Clinical interpretation
    INTERPRETATIONS = {
        0: 'Higher values indicate worse horizontal eye tracking (MG indicator)',
        1: 'Higher values indicate worse vertical eye tracking (MG indicator)',
        2: 'Higher values indicate more unstable tracking (MG indicator)',
        3: 'Lower values indicate jerky, non-smooth saccades (MG indicator)',
        4: 'Higher values indicate fatigue during recording (MG indicator)',
    }

    NUM_CONCEPTS = 5

    # Feature indices in the 14-channel input
    # [LH, RH, LV, RV, TargetH, TargetV, LH_vel, RH_vel, LV_vel, RV_vel,
    #  Error_H_L, Error_H_R, Error_V_L, Error_V_R]
    ERROR_H_LEFT_IDX = 10
    ERROR_H_RIGHT_IDX = 11
    ERROR_V_LEFT_IDX = 12
    ERROR_V_RIGHT_IDX = 13
    VELOCITY_INDICES = [6, 7, 8, 9]


def compute_concept_targets(x: torch.Tensor) -> torch.Tensor:
    """
    Compute ground-truth concept values from raw input data.

    These serve as soft supervision targets during training.
    The concepts are normalized to roughly [0, 1] range.

    Args:
        x: Input tensor of shape (batch_size, seq_len, 14)

    Returns:
        Concept targets of shape (batch_size, 5)
    """
    batch_size = x.shape[0]
    concepts = torch.zeros(batch_size, ClinicalConcepts.NUM_CONCEPTS, device=x.device)

    # 1. Horizontal Tracking Error (mean absolute)
    error_h = (torch.abs(x[:, :, ClinicalConcepts.ERROR_H_LEFT_IDX]) +
               torch.abs(x[:, :, ClinicalConcepts.ERROR_H_RIGHT_IDX])) / 2
    concepts[:, 0] = error_h.mean(dim=1)

    # 2. Vertical Tracking Error (mean absolute)
    error_v = (torch.abs(x[:, :, ClinicalConcepts.ERROR_V_LEFT_IDX]) +
               torch.abs(x[:, :, ClinicalConcepts.ERROR_V_RIGHT_IDX])) / 2
    concepts[:, 1] = error_v.mean(dim=1)

    # 3. Tracking Variability (std of all tracking errors)
    all_errors = torch.cat([
        x[:, :, ClinicalConcepts.ERROR_H_LEFT_IDX:ClinicalConcepts.ERROR_V_RIGHT_IDX + 1]
    ], dim=2)
    concepts[:, 2] = all_errors.std(dim=(1, 2))

    # 4. Saccade Smoothness (inverse of velocity variance, normalized)
    velocities = x[:, :, ClinicalConcepts.VELOCITY_INDICES]
    velocity_var = velocities.var(dim=(1, 2))
    # Add small epsilon and invert (higher smoothness = lower variance)
    concepts[:, 3] = 1.0 / (1.0 + velocity_var)

    # 5. Fatigue Effect (Q4 error - Q1 error)
    seq_len = x.shape[1]
    q1_end = seq_len // 4
    q4_start = 3 * seq_len // 4

    q1_error = error_h[:, :q1_end].mean(dim=1)
    q4_error = error_h[:, q4_start:].mean(dim=1)
    concepts[:, 4] = q4_error - q1_error

    return concepts


# =============================================================================
# CAUSAL CONVOLUTION (same as regular TCN)
# =============================================================================

class CausalConv1d(nn.Module):
    """Causal 1D convolution with padding to preserve sequence length."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """Temporal block with two causal convolutions and residual connection."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.relu(self.bn1(self.conv1(x))))
        out = self.dropout(self.relu(self.bn2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# =============================================================================
# CONCEPT BOTTLENECK TCN
# =============================================================================

class ConceptBottleneckTCN(BaseTimeSeriesModel):
    """
    Temporal Convolutional Network with Clinical Concept Bottleneck.

    Architecture:
        Input (batch, seq_len, 14)
        -> TCN Backbone -> (batch, hidden_dim, seq_len)
        -> Global Pooling -> (batch, hidden_dim)
        -> Concept Layer -> (batch, num_concepts)  [BOTTLENECK]
        -> Classifier -> (batch, num_classes)

    The concept bottleneck ensures that:
    1. The model's decision is based ONLY on interpretable concepts
    2. Clinicians can understand WHY the model made a prediction
    3. Concepts can be intervened on (corrected) at test time

    Clinical Concepts:
    - Horizontal/Vertical Tracking Error (MG patients have higher)
    - Tracking Variability (MG patients are more variable)
    - Saccade Smoothness (MG patients have jerkier movements)
    - Fatigue Effect (MG patients show more fatigue over time)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.2,
        concept_loss_weight: float = 0.5,
        **kwargs,
    ):
        """
        Args:
            input_dim: Number of input features (14)
            num_classes: Number of output classes (2)
            seq_len: Sequence length (2903)
            hidden_dim: Hidden dimension in TCN blocks
            num_layers: Number of temporal blocks
            kernel_size: Convolution kernel size
            dropout: Dropout probability
            concept_loss_weight: Weight for concept prediction loss
            **kwargs: Extra arguments (ignored)
        """
        super().__init__(input_dim, num_classes, seq_len)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.concept_loss_weight = concept_loss_weight
        self.num_concepts = ClinicalConcepts.NUM_CONCEPTS

        # Build TCN backbone
        layers = []
        in_channels = input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(in_channels, hidden_dim, kernel_size, dilation, dropout)
            )
            in_channels = hidden_dim
        self.temporal_blocks = nn.Sequential(*layers)

        # Concept prediction layer (the bottleneck)
        self.concept_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_concepts),
        )

        # Classifier from concepts only
        # This is a simple linear layer to maintain interpretability
        self.concept_classifier = nn.Linear(self.num_concepts, num_classes)

        # Store last computed concepts for explainability
        self._last_concepts = None
        self._last_concept_contributions = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with concept bottleneck.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            Logits (batch_size, num_classes)
        """
        # TCN backbone
        x_t = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        features = self.temporal_blocks(x_t)  # (batch, hidden_dim, seq_len)

        # Global average pooling
        pooled = features.mean(dim=2)  # (batch, hidden_dim)

        # Predict concepts (this is the bottleneck)
        concepts = self.concept_layer(pooled)  # (batch, num_concepts)

        # Store for explainability
        self._last_concepts = concepts.detach()

        # Classify from concepts only
        logits = self.concept_classifier(concepts)  # (batch, num_classes)

        # Compute contribution of each concept to the prediction
        # This is simply weight * concept_value for the predicted class
        with torch.no_grad():
            weights = self.concept_classifier.weight  # (num_classes, num_concepts)
            # For MG class (index 1), contribution = weight[1] * concepts
            self._last_concept_contributions = weights[1] * concepts

        return logits

    def forward_with_concepts(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both logits and predicted concepts.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            logits: (batch_size, num_classes)
            concepts: (batch_size, num_concepts)
        """
        logits = self.forward(x)
        return logits, self._last_concepts

    def get_concept_targets(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute ground-truth concept values from input data.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            Concept targets (batch_size, num_concepts)
        """
        return compute_concept_targets(x)

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        concepts_pred: torch.Tensor,
        concepts_target: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss: classification + concept prediction.

        Args:
            logits: Predicted logits (batch, num_classes)
            labels: True labels (batch,)
            concepts_pred: Predicted concepts (batch, num_concepts)
            concepts_target: Target concepts (batch, num_concepts)
            class_weights: Optional class weights for classification loss

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Classification loss (cross-entropy)
        if class_weights is not None:
            cls_loss = F.cross_entropy(logits, labels, weight=class_weights)
        else:
            cls_loss = F.cross_entropy(logits, labels)

        # Concept prediction loss (MSE)
        # Normalize targets to have similar scale
        with torch.no_grad():
            target_mean = concepts_target.mean(dim=0, keepdim=True)
            target_std = concepts_target.std(dim=0, keepdim=True) + 1e-8
            normalized_target = (concepts_target - target_mean) / target_std

        normalized_pred = (concepts_pred - target_mean) / target_std
        concept_loss = F.mse_loss(normalized_pred, normalized_target)

        # Combined loss
        total_loss = cls_loss + self.concept_loss_weight * concept_loss

        return total_loss, {
            'total_loss': total_loss.item(),
            'cls_loss': cls_loss.item(),
            'concept_loss': concept_loss.item(),
        }

    def explain_prediction(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Generate a human-interpretable explanation for a prediction.

        Args:
            x: Input tensor (1, seq_len, input_dim) - single sample

        Returns:
            Dictionary with explanation details
        """
        self.eval()
        with torch.no_grad():
            logits, concepts = self.forward_with_concepts(x)
            probs = F.softmax(logits, dim=1)
            predicted_class = logits.argmax(dim=1).item()
            confidence = probs[0, predicted_class].item()

            # Get concept contributions
            contributions = self._last_concept_contributions[0]

            # Get target concepts (ground truth from data)
            target_concepts = self.get_concept_targets(x)[0]

            # Build explanation
            explanation = {
                'prediction': 'MG' if predicted_class == 1 else 'HC',
                'confidence': confidence,
                'concepts': {},
                'concept_contributions': {},
                'decision_summary': '',
            }

            # Add each concept
            for i, name in enumerate(ClinicalConcepts.NAMES):
                pred_val = concepts[0, i].item()
                target_val = target_concepts[i].item()
                contribution = contributions[i].item()

                explanation['concepts'][name] = {
                    'predicted_value': pred_val,
                    'actual_value': target_val,
                    'contribution_to_mg': contribution,
                    'interpretation': ClinicalConcepts.INTERPRETATIONS[i],
                }
                explanation['concept_contributions'][name] = contribution

            # Generate decision summary
            sorted_concepts = sorted(
                explanation['concept_contributions'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            top_reasons = []
            for name, contrib in sorted_concepts[:3]:
                direction = 'high' if contrib > 0 else 'low'
                top_reasons.append(f"{name} is {direction} ({contrib:.3f})")

            explanation['decision_summary'] = (
                f"Predicted {explanation['prediction']} with {confidence:.1%} confidence. "
                f"Top contributing factors: {'; '.join(top_reasons)}"
            )

            return explanation

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': 'ConceptBottleneckTCN',
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'seq_len': self.seq_len,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout,
            'num_concepts': self.num_concepts,
            'concept_loss_weight': self.concept_loss_weight,
        }

    @staticmethod
    def get_concept_names() -> List[str]:
        """Get list of concept names."""
        return ClinicalConcepts.NAMES


# =============================================================================
# CONCEPT BOTTLENECK TRAINER MIXIN
# =============================================================================

class ConceptBottleneckTrainerMixin:
    """
    Mixin class providing training utilities for Concept Bottleneck models.

    This can be used alongside the regular Trainer class.
    """

    @staticmethod
    def concept_bottleneck_training_step(
        model: ConceptBottleneckTCN,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Single training step for concept bottleneck model.

        Args:
            model: ConceptBottleneckTCN model
            batch_x: Input batch (batch, seq_len, input_dim)
            batch_y: Labels (batch,)
            optimizer: Optimizer
            class_weights: Optional class weights

        Returns:
            Dictionary with loss values
        """
        model.train()
        optimizer.zero_grad()

        # Forward pass
        logits, concepts_pred = model.forward_with_concepts(batch_x)

        # Compute concept targets
        concepts_target = model.get_concept_targets(batch_x)

        # Compute loss
        loss, loss_dict = model.compute_loss(
            logits, batch_y, concepts_pred, concepts_target, class_weights
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        return loss_dict


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    'hidden_dim': 64,
    'num_layers': 4,
    'kernel_size': 7,
    'dropout': 0.2,
    'concept_loss_weight': 0.5,
}
