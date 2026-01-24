"""
CCECE Paper: Explainability Module

Provides interpretability tools for time series classification models:
- Attention weight visualization
- Feature importance analysis
- Gradient-based saliency maps (Integrated Gradients, GradCAM)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExplanationResult:
    """Container for model explanation results."""
    input_sequence: np.ndarray  # (seq_len, features)
    label: int
    predicted_label: int
    predicted_probability: float

    # Feature importance scores (if computed)
    feature_importance: Optional[np.ndarray] = None  # (features,)

    # Temporal importance (saliency over time)
    temporal_saliency: Optional[np.ndarray] = None  # (seq_len,)

    # Full saliency map
    saliency_map: Optional[np.ndarray] = None  # (seq_len, features)

    # Attention weights (if model has attention)
    attention_weights: Optional[np.ndarray] = None  # (seq_len,)

    def get_top_features(self, k: int = 5) -> List[Tuple[int, float]]:
        """Get top-k most important features."""
        if self.feature_importance is None:
            return []
        indices = np.argsort(self.feature_importance)[::-1][:k]
        return [(int(i), float(self.feature_importance[i])) for i in indices]

    def get_important_timepoints(self, k: int = 10) -> List[Tuple[int, float]]:
        """Get top-k most important time points."""
        if self.temporal_saliency is None:
            return []
        indices = np.argsort(self.temporal_saliency)[::-1][:k]
        return [(int(i), float(self.temporal_saliency[i])) for i in indices]


# =============================================================================
# GRADIENT-BASED SALIENCY
# =============================================================================

class GradientSaliency:
    """
    Compute gradient-based saliency maps for time series models.

    Methods:
        - Vanilla gradients
        - Integrated gradients
        - SmoothGrad
    """

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Args:
            model: Trained PyTorch model
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.model.eval()

    def vanilla_gradients(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute vanilla gradient saliency.

        Args:
            input_tensor: Input tensor (1, seq_len, features)
            target_class: Target class for gradient computation (None = predicted)

        Returns:
            Saliency map (seq_len, features)
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True

        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()

        # Get gradients
        gradients = input_tensor.grad.cpu().numpy()[0]  # (seq_len, features)

        return np.abs(gradients)

    def integrated_gradients(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50,
    ) -> np.ndarray:
        """
        Compute Integrated Gradients saliency.

        This method attributes importance to input features by integrating
        gradients along a path from a baseline to the input.

        Args:
            input_tensor: Input tensor (1, seq_len, features)
            target_class: Target class (None = predicted)
            baseline: Baseline input (None = zeros)
            steps: Number of integration steps

        Returns:
            Attribution map (seq_len, features)
        """
        input_tensor = input_tensor.detach().to(self.device)

        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()

        # Create baseline (zeros by default)
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        baseline = baseline.to(self.device)

        # Compute integrated gradients step by step
        all_gradients = []
        for alpha in np.linspace(0, 1, steps):
            scaled = baseline + alpha * (input_tensor - baseline)
            scaled = scaled.clone().detach().requires_grad_(True)

            output = self.model(scaled)
            self.model.zero_grad()
            output[0, target_class].backward()

            gradients = scaled.grad.detach().cpu().numpy()[0]
            all_gradients.append(gradients)

        # Average gradients
        avg_gradients = np.mean(all_gradients, axis=0)

        # Multiply by input difference
        input_diff = (input_tensor - baseline).detach().cpu().numpy()[0]
        attributions = avg_gradients * input_diff

        return np.abs(attributions)

    def smooth_grad(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        num_samples: int = 50,
        noise_level: float = 0.1,
    ) -> np.ndarray:
        """
        Compute SmoothGrad saliency.

        Averages gradients over noisy versions of the input to reduce noise.

        Args:
            input_tensor: Input tensor (1, seq_len, features)
            target_class: Target class (None = predicted)
            num_samples: Number of noisy samples
            noise_level: Standard deviation of noise (as fraction of input range)

        Returns:
            Smoothed saliency map (seq_len, features)
        """
        input_tensor = input_tensor.to(self.device)

        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()

        # Compute noise scale
        input_range = input_tensor.max() - input_tensor.min()
        noise_std = noise_level * input_range.item()

        # Collect gradients for noisy inputs
        all_gradients = []

        for _ in range(num_samples):
            noisy_input = input_tensor + torch.randn_like(input_tensor) * noise_std
            noisy_input.requires_grad = True

            output = self.model(noisy_input)
            self.model.zero_grad()
            output[0, target_class].backward()

            gradients = noisy_input.grad.cpu().numpy()[0]
            all_gradients.append(gradients)

        # Average gradients
        avg_gradients = np.mean(all_gradients, axis=0)

        return np.abs(avg_gradients)


# =============================================================================
# ATTENTION EXTRACTION
# =============================================================================

def extract_attention_weights(model: nn.Module, input_tensor: torch.Tensor) -> Optional[np.ndarray]:
    """
    Extract attention weights from models with attention mechanisms.

    Works with BiGRUAttention and BiLSTMAttention models.

    Args:
        model: Model with attention mechanism
        input_tensor: Input tensor (1, seq_len, features)

    Returns:
        Attention weights (seq_len,) or None if not available
    """
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Check if model has attention
    if not hasattr(model, 'attention'):
        return None

    # Forward pass with attention capture
    model.eval()
    attention_weights = None

    # Register hook to capture attention
    def attention_hook(module, input, output):
        nonlocal attention_weights
        # Attention output is typically the weighted context
        # We need to get the raw attention scores
        pass

    # Try to get attention directly
    with torch.no_grad():
        # Get RNN output
        if hasattr(model, 'gru'):
            rnn_out, _ = model.gru(input_tensor)
        elif hasattr(model, 'lstm'):
            rnn_out, _ = model.lstm(input_tensor)
        else:
            return None

        # Compute attention scores
        attention = model.attention
        if isinstance(attention, nn.Sequential):
            # Typical attention: Linear -> Tanh -> Linear
            scores = attention(rnn_out)
            attention_weights = torch.softmax(scores.squeeze(-1), dim=1)
        elif hasattr(attention, 'weight'):
            # Single linear layer attention
            scores = torch.matmul(rnn_out, attention.weight.T)
            attention_weights = torch.softmax(scores.squeeze(-1), dim=1)

    if attention_weights is not None:
        return attention_weights.cpu().numpy()[0]

    return None


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def compute_feature_importance(
    model: nn.Module,
    input_tensor: torch.Tensor,
    method: str = 'gradient',
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Compute feature importance by aggregating saliency over time.

    Args:
        model: Trained model
        input_tensor: Input tensor (1, seq_len, features)
        method: 'gradient', 'integrated', or 'permutation'
        device: Device for computation

    Returns:
        Feature importance scores (features,)
    """
    if device is None:
        device = next(model.parameters()).device

    if method == 'gradient':
        saliency = GradientSaliency(model, device)
        saliency_map = saliency.vanilla_gradients(input_tensor)
    elif method == 'integrated':
        saliency = GradientSaliency(model, device)
        saliency_map = saliency.integrated_gradients(input_tensor)
    elif method == 'permutation':
        return _permutation_importance(model, input_tensor, device)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Aggregate over time dimension
    feature_importance = np.mean(saliency_map, axis=0)

    # Normalize
    if feature_importance.sum() > 0:
        feature_importance = feature_importance / feature_importance.sum()

    return feature_importance


def _permutation_importance(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    num_permutations: int = 10,
) -> np.ndarray:
    """Compute feature importance via permutation."""
    model.eval()
    input_tensor = input_tensor.to(device)

    # Get baseline prediction
    with torch.no_grad():
        baseline_output = model(input_tensor)
        baseline_prob = torch.softmax(baseline_output, dim=1)[0, 1].item()

    num_features = input_tensor.shape[2]
    importance_scores = np.zeros(num_features)

    for feat_idx in range(num_features):
        score_drops = []

        for _ in range(num_permutations):
            # Create permuted input
            permuted_input = input_tensor.clone()
            perm_idx = torch.randperm(permuted_input.shape[1])
            permuted_input[0, :, feat_idx] = permuted_input[0, perm_idx, feat_idx]

            # Get prediction with permuted feature
            with torch.no_grad():
                perm_output = model(permuted_input)
                perm_prob = torch.softmax(perm_output, dim=1)[0, 1].item()

            score_drops.append(abs(baseline_prob - perm_prob))

        importance_scores[feat_idx] = np.mean(score_drops)

    # Normalize
    if importance_scores.sum() > 0:
        importance_scores = importance_scores / importance_scores.sum()

    return importance_scores


# =============================================================================
# EXPLAINER CLASS
# =============================================================================

class ModelExplainer:
    """
    High-level class for generating explanations for model predictions.

    Provides a unified interface for various explainability methods.
    """

    # Feature names for the 14-channel input
    FEATURE_NAMES = [
        'Left Horizontal', 'Right Horizontal', 'Left Vertical', 'Right Vertical',
        'Target Horizontal', 'Target Vertical',
        'LH Velocity', 'RH Velocity', 'LV Velocity', 'RV Velocity',
        'Error H Left', 'Error H Right', 'Error V Left', 'Error V Right'
    ]

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Args:
            model: Trained model
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.model.eval()
        self.saliency = GradientSaliency(model, device)

    def explain(
        self,
        input_tensor: torch.Tensor,
        label: int,
        method: str = 'integrated',
    ) -> ExplanationResult:
        """
        Generate a comprehensive explanation for a prediction.

        Args:
            input_tensor: Input tensor (1, seq_len, features)
            label: True label
            method: Saliency method ('gradient', 'integrated', 'smoothgrad')

        Returns:
            ExplanationResult with all computed explanations
        """
        input_tensor = input_tensor.to(self.device)

        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            predicted_label = output.argmax(dim=1).item()
            predicted_prob = probs[0, predicted_label].item()

        # Compute saliency map
        if method == 'gradient':
            saliency_map = self.saliency.vanilla_gradients(input_tensor)
        elif method == 'integrated':
            saliency_map = self.saliency.integrated_gradients(input_tensor)
        elif method == 'smoothgrad':
            saliency_map = self.saliency.smooth_grad(input_tensor)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Compute feature importance (aggregate over time)
        feature_importance = np.mean(saliency_map, axis=0)
        if feature_importance.sum() > 0:
            feature_importance = feature_importance / feature_importance.sum()

        # Compute temporal saliency (aggregate over features)
        temporal_saliency = np.mean(saliency_map, axis=1)
        if temporal_saliency.sum() > 0:
            temporal_saliency = temporal_saliency / temporal_saliency.sum()

        # Extract attention weights if available
        attention_weights = extract_attention_weights(self.model, input_tensor)

        return ExplanationResult(
            input_sequence=input_tensor.detach().cpu().numpy()[0],
            label=label,
            predicted_label=predicted_label,
            predicted_probability=predicted_prob,
            feature_importance=feature_importance,
            temporal_saliency=temporal_saliency,
            saliency_map=saliency_map,
            attention_weights=attention_weights,
        )

    def get_feature_names(self) -> List[str]:
        """Get human-readable feature names."""
        return self.FEATURE_NAMES

    def print_explanation(self, result: ExplanationResult):
        """Print a human-readable explanation."""
        print("\n" + "="*60)
        print("MODEL EXPLANATION")
        print("="*60)

        print(f"\nPrediction: {'MG' if result.predicted_label == 1 else 'HC'}")
        print(f"Confidence: {result.predicted_probability:.2%}")
        print(f"True Label: {'MG' if result.label == 1 else 'HC'}")
        print(f"Correct: {'Yes' if result.predicted_label == result.label else 'No'}")

        print("\n--- Top 5 Important Features ---")
        top_features = result.get_top_features(5)
        for idx, score in top_features:
            name = self.FEATURE_NAMES[idx] if idx < len(self.FEATURE_NAMES) else f"Feature {idx}"
            print(f"  {name}: {score:.4f}")

        print("\n--- Top 10 Important Time Points ---")
        top_times = result.get_important_timepoints(10)
        for t, score in top_times:
            print(f"  t={t}: {score:.4f}")

        if result.attention_weights is not None:
            print("\n--- Attention Summary ---")
            peak_attention = np.argmax(result.attention_weights)
            print(f"  Peak attention at t={peak_attention}")
            print(f"  Peak attention value: {result.attention_weights[peak_attention]:.4f}")

        print("\n" + "="*60)
