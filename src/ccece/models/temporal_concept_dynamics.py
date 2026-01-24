"""
CCECE Paper: Temporal Concept Dynamics Network (TCDN)

A novel architecture for explainable MG detection that models the temporal
evolution of clinical concepts, not just their static values.

Key Innovation:
- Divides the sequence into temporal segments
- Extracts clinical concepts per segment
- Models concept trajectories (how concepts change over time)
- Classifies based on trajectory features (slope, acceleration, variance)

This is particularly suited for MG because:
- MG causes FATIGUE: performance degrades over time
- The trajectory of degradation is clinically meaningful
- Doctors can verify: "Error increased 40% from start to end"

Architecture:
    Input → Segment Encoder → Per-Segment Concepts → Trajectory Encoder → Classification
                                      ↓
                              [Q1, Q2, Q3, Q4] concept values
                                      ↓
                              Trajectory features:
                              - Initial value (baseline)
                              - Final value (end state)
                              - Slope (rate of change)
                              - Curvature (acceleration)
                              - Variance (instability)

Reference:
- Concept Bottleneck Models (Koh et al., ICML 2020) - base concept idea
- This work: Novel temporal extension for fatigue-based diseases
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
class TemporalConcepts:
    """
    Clinical concepts computed per temporal segment.

    For MG detection, we track how these concepts EVOLVE over time,
    not just their average values.
    """
    # Concept names
    NAMES = [
        'Tracking Accuracy',      # 0: How well eyes follow target (inverse of error)
        'Horizontal Stability',   # 1: Stability in horizontal tracking
        'Vertical Stability',     # 2: Stability in vertical tracking
        'Saccade Smoothness',     # 3: Smoothness of eye movements
        'Binocular Coordination', # 4: How well left/right eyes coordinate
    ]

    # Trajectory feature names
    TRAJECTORY_FEATURES = [
        'initial',     # Value at start
        'final',       # Value at end
        'slope',       # Rate of change (positive = increasing)
        'curvature',   # Acceleration (positive = accelerating increase)
        'variance',    # Instability over time
    ]

    NUM_CONCEPTS = 5
    NUM_TRAJECTORY_FEATURES = 5

    # Clinical interpretation for trajectories
    INTERPRETATIONS = {
        'Tracking Accuracy': {
            'slope_negative': 'Accuracy decreases over time (fatigue indicator)',
            'slope_positive': 'Accuracy improves or stays stable',
            'high_variance': 'Inconsistent tracking (instability indicator)',
        },
        'Saccade Smoothness': {
            'slope_negative': 'Movements become jerkier over time (fatigue)',
            'slope_positive': 'Movements stay smooth',
        },
        'Binocular Coordination': {
            'slope_negative': 'Eyes become less coordinated (fatigue)',
            'high_variance': 'Intermittent coordination problems',
        },
    }

    # Feature indices
    ERROR_H_LEFT_IDX = 10
    ERROR_H_RIGHT_IDX = 11
    ERROR_V_LEFT_IDX = 12
    ERROR_V_RIGHT_IDX = 13
    VELOCITY_INDICES = [6, 7, 8, 9]
    LEFT_H_IDX = 0
    RIGHT_H_IDX = 1
    LEFT_V_IDX = 2
    RIGHT_V_IDX = 3


# =============================================================================
# SEGMENT-WISE CONCEPT COMPUTATION
# =============================================================================

def compute_segment_concepts(x: torch.Tensor, num_segments: int = 4) -> torch.Tensor:
    """
    Compute clinical concepts for each temporal segment.

    Args:
        x: Input tensor (batch, seq_len, 14)
        num_segments: Number of temporal segments (default: 4 quarters)

    Returns:
        Segment concepts (batch, num_segments, num_concepts)
    """
    batch_size, seq_len, _ = x.shape
    segment_len = seq_len // num_segments

    concepts = torch.zeros(
        batch_size, num_segments, TemporalConcepts.NUM_CONCEPTS,
        device=x.device
    )

    for seg_idx in range(num_segments):
        start = seg_idx * segment_len
        end = start + segment_len if seg_idx < num_segments - 1 else seq_len
        segment = x[:, start:end, :]

        # 1. Tracking Accuracy (inverse of mean tracking error)
        error_h = (torch.abs(segment[:, :, TemporalConcepts.ERROR_H_LEFT_IDX]) +
                   torch.abs(segment[:, :, TemporalConcepts.ERROR_H_RIGHT_IDX])) / 2
        error_v = (torch.abs(segment[:, :, TemporalConcepts.ERROR_V_LEFT_IDX]) +
                   torch.abs(segment[:, :, TemporalConcepts.ERROR_V_RIGHT_IDX])) / 2
        total_error = (error_h + error_v).mean(dim=1)
        concepts[:, seg_idx, 0] = 1.0 / (1.0 + total_error)  # Higher = better accuracy

        # 2. Horizontal Stability (inverse of horizontal error variance)
        h_error_var = error_h.var(dim=1)
        concepts[:, seg_idx, 1] = 1.0 / (1.0 + h_error_var)

        # 3. Vertical Stability (inverse of vertical error variance)
        v_error_var = error_v.var(dim=1)
        concepts[:, seg_idx, 2] = 1.0 / (1.0 + v_error_var)

        # 4. Saccade Smoothness (inverse of velocity variance)
        velocities = segment[:, :, TemporalConcepts.VELOCITY_INDICES]
        vel_var = velocities.var(dim=(1, 2))
        concepts[:, seg_idx, 3] = 1.0 / (1.0 + vel_var)

        # 5. Binocular Coordination (inverse of L-R difference)
        lr_diff_h = torch.abs(segment[:, :, TemporalConcepts.LEFT_H_IDX] -
                              segment[:, :, TemporalConcepts.RIGHT_H_IDX]).mean(dim=1)
        lr_diff_v = torch.abs(segment[:, :, TemporalConcepts.LEFT_V_IDX] -
                              segment[:, :, TemporalConcepts.RIGHT_V_IDX]).mean(dim=1)
        concepts[:, seg_idx, 4] = 1.0 / (1.0 + lr_diff_h + lr_diff_v)

    return concepts


def compute_trajectory_features(segment_concepts: torch.Tensor) -> torch.Tensor:
    """
    Compute trajectory features from segment-wise concepts.

    For each concept, we compute:
    - Initial value (Q1)
    - Final value (Q4)
    - Slope (linear trend)
    - Curvature (acceleration)
    - Variance (instability)

    Args:
        segment_concepts: (batch, num_segments, num_concepts)

    Returns:
        Trajectory features: (batch, num_concepts * num_trajectory_features)
    """
    batch_size, num_segments, num_concepts = segment_concepts.shape
    device = segment_concepts.device

    # Time points for regression (normalized to [0, 1])
    t = torch.linspace(0, 1, num_segments, device=device)
    t_squared = t ** 2

    trajectory_features = []

    for c in range(num_concepts):
        concept_values = segment_concepts[:, :, c]  # (batch, num_segments)

        # 1. Initial value
        initial = concept_values[:, 0]

        # 2. Final value
        final = concept_values[:, -1]

        # 3. Slope (linear regression coefficient)
        # Using least squares: slope = cov(t, y) / var(t)
        t_mean = t.mean()
        y_mean = concept_values.mean(dim=1, keepdim=True)
        cov = ((t - t_mean) * (concept_values - y_mean)).mean(dim=1)
        var_t = ((t - t_mean) ** 2).mean()
        slope = cov / (var_t + 1e-8)

        # 4. Curvature (second derivative approximation)
        # Using finite differences: curvature ≈ (v[n] - 2*v[n-1] + v[n-2])
        if num_segments >= 3:
            second_diff = (concept_values[:, 2:] - 2 * concept_values[:, 1:-1] +
                          concept_values[:, :-2])
            curvature = second_diff.mean(dim=1)
        else:
            curvature = torch.zeros(batch_size, device=device)

        # 5. Variance (instability)
        variance = concept_values.var(dim=1)

        trajectory_features.extend([initial, final, slope, curvature, variance])

    return torch.stack(trajectory_features, dim=1)


# =============================================================================
# CAUSAL CONVOLUTION LAYERS
# =============================================================================

class CausalConv1d(nn.Module):
    """Causal 1D convolution with padding."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """Temporal block with residual connection."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = (nn.Conv1d(in_channels, out_channels, 1)
                          if in_channels != out_channels else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.relu(self.bn1(self.conv1(x))))
        out = self.dropout(self.relu(self.bn2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# =============================================================================
# TEMPORAL CONCEPT DYNAMICS NETWORK
# =============================================================================

class TemporalConceptDynamicsNetwork(BaseTimeSeriesModel):
    """
    Temporal Concept Dynamics Network (TCDN) for Explainable MG Detection.

    Novel Architecture Features:
    1. Segment-wise feature extraction (not global pooling)
    2. Clinical concepts computed per temporal segment
    3. Trajectory modeling of concept evolution
    4. Classification based on trajectory features

    This captures the FATIGUE signature of MG: concepts degrade over time.

    Explainability:
    - "Tracking accuracy: 0.8 → 0.5 (slope=-0.3, indicating fatigue)"
    - "Binocular coordination variance high (0.15), indicating instability"
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
        num_segments: int = 4,
        use_learned_concepts: bool = True,
        trajectory_hidden_dim: int = 32,
        **kwargs,
    ):
        """
        Args:
            input_dim: Number of input features (14)
            num_classes: Number of output classes (2)
            seq_len: Sequence length (2903)
            hidden_dim: Hidden dimension in TCN
            num_layers: Number of temporal blocks
            kernel_size: Convolution kernel size
            dropout: Dropout probability
            num_segments: Number of temporal segments for trajectory
            use_learned_concepts: If True, learn concept extraction; if False, use handcrafted
            trajectory_hidden_dim: Hidden dimension for trajectory encoder
        """
        super().__init__(input_dim, num_classes, seq_len)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout_rate = dropout
        self.num_segments = num_segments
        self.use_learned_concepts = use_learned_concepts
        self.num_concepts = TemporalConcepts.NUM_CONCEPTS
        self.num_trajectory_features = TemporalConcepts.NUM_TRAJECTORY_FEATURES

        # TCN backbone for feature extraction
        layers = []
        in_channels = input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_channels, hidden_dim, kernel_size,
                                        dilation, dropout))
            in_channels = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Learned concept extraction (optional enhancement over handcrafted)
        if use_learned_concepts:
            self.concept_extractor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_concepts),
            )

        # Trajectory encoder: models how concepts evolve over segments
        # Using a small transformer for segment-to-segment attention
        self.trajectory_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.num_concepts,
                nhead=1,  # Single head for interpretability
                dim_feedforward=trajectory_hidden_dim,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=1,
        )

        # Trajectory feature computation is differentiable
        # Final classifier takes trajectory features
        total_trajectory_features = self.num_concepts * self.num_trajectory_features

        self.classifier = nn.Sequential(
            nn.Linear(total_trajectory_features + self.num_concepts, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Store for explainability
        self._segment_concepts = None
        self._trajectory_features = None
        self._encoded_trajectory = None

    def extract_segment_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features per temporal segment.

        Args:
            x: Input (batch, seq_len, input_dim)

        Returns:
            Segment features (batch, num_segments, hidden_dim)
        """
        batch_size = x.shape[0]
        segment_len = self.seq_len // self.num_segments

        # Apply backbone
        x_t = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        features = self.backbone(x_t)  # (batch, hidden_dim, seq_len)
        features = features.transpose(1, 2)  # (batch, seq_len, hidden_dim)

        # Pool within each segment
        segment_features = []
        for seg_idx in range(self.num_segments):
            start = seg_idx * segment_len
            end = start + segment_len if seg_idx < self.num_segments - 1 else self.seq_len
            seg_feat = features[:, start:end, :].mean(dim=1)  # (batch, hidden_dim)
            segment_features.append(seg_feat)

        return torch.stack(segment_features, dim=1)  # (batch, num_segments, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with temporal concept dynamics.

        Args:
            x: Input (batch, seq_len, input_dim)

        Returns:
            Logits (batch, num_classes)
        """
        batch_size = x.shape[0]

        # Compute handcrafted segment concepts from raw input
        raw_concepts = compute_segment_concepts(x, self.num_segments)

        if self.use_learned_concepts:
            # Also extract learned features
            segment_features = self.extract_segment_features(x)
            learned_concepts = self.concept_extractor(segment_features)

            # Combine handcrafted and learned (residual connection)
            segment_concepts = raw_concepts + 0.5 * learned_concepts
        else:
            segment_concepts = raw_concepts

        self._segment_concepts = segment_concepts.detach()

        # Encode trajectory with transformer (captures segment interactions)
        encoded_trajectory = self.trajectory_encoder(segment_concepts)
        self._encoded_trajectory = encoded_trajectory.detach()

        # Compute trajectory features
        trajectory_features = compute_trajectory_features(encoded_trajectory)
        self._trajectory_features = trajectory_features.detach()

        # Also include mean concept values for baseline information
        mean_concepts = segment_concepts.mean(dim=1)

        # Classify from trajectory features + mean concepts
        combined = torch.cat([trajectory_features, mean_concepts], dim=1)
        logits = self.classifier(combined)

        return logits

    def forward_with_trajectory(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass returning trajectory information for explainability.

        Returns:
            logits: (batch, num_classes)
            trajectory_info: Dict with segment concepts and trajectory features
        """
        logits = self.forward(x)

        trajectory_info = {
            'segment_concepts': self._segment_concepts,
            'trajectory_features': self._trajectory_features,
            'encoded_trajectory': self._encoded_trajectory,
        }

        return logits, trajectory_info

    def explain_prediction(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Generate detailed trajectory-based explanation.

        Args:
            x: Single sample (1, seq_len, input_dim)

        Returns:
            Explanation dictionary with trajectory analysis
        """
        self.eval()
        with torch.no_grad():
            logits, traj_info = self.forward_with_trajectory(x)
            probs = F.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()

            segment_concepts = traj_info['segment_concepts'][0]  # (num_segments, num_concepts)

            explanation = {
                'prediction': 'MG' if pred_class == 1 else 'HC',
                'confidence': confidence,
                'concept_trajectories': {},
                'fatigue_indicators': [],
                'stability_indicators': [],
                'summary': '',
            }

            # Analyze each concept's trajectory
            for c_idx, c_name in enumerate(TemporalConcepts.NAMES):
                values = segment_concepts[:, c_idx].cpu().numpy()

                # Compute trajectory statistics
                initial = values[0]
                final = values[-1]
                change = final - initial
                slope = np.polyfit(range(len(values)), values, 1)[0]
                variance = np.var(values)

                trajectory = {
                    'values': values.tolist(),
                    'initial': float(initial),
                    'final': float(final),
                    'change': float(change),
                    'slope': float(slope),
                    'variance': float(variance),
                    'trend': 'increasing' if slope > 0.01 else ('decreasing' if slope < -0.01 else 'stable'),
                }
                explanation['concept_trajectories'][c_name] = trajectory

                # Identify fatigue indicators (decreasing performance over time)
                if c_name in ['Tracking Accuracy', 'Saccade Smoothness', 'Binocular Coordination']:
                    if slope < -0.02:  # Significant decrease
                        explanation['fatigue_indicators'].append(
                            f"{c_name} decreases from {initial:.2f} to {final:.2f} "
                            f"(slope={slope:.3f}, indicates fatigue)"
                        )

                # Identify instability indicators
                if variance > 0.01:
                    explanation['stability_indicators'].append(
                        f"{c_name} shows high variability (var={variance:.3f})"
                    )

            # Generate summary
            if explanation['fatigue_indicators']:
                fatigue_summary = "; ".join(explanation['fatigue_indicators'][:2])
                explanation['summary'] = (
                    f"Predicted {explanation['prediction']} with {confidence:.0%} confidence. "
                    f"Fatigue pattern detected: {fatigue_summary}"
                )
            else:
                explanation['summary'] = (
                    f"Predicted {explanation['prediction']} with {confidence:.0%} confidence. "
                    f"No significant fatigue pattern detected."
                )

            return explanation

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': 'TemporalConceptDynamicsNetwork',
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'seq_len': self.seq_len,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout_rate,
            'num_segments': self.num_segments,
            'num_concepts': self.num_concepts,
            'use_learned_concepts': self.use_learned_concepts,
        }

    @staticmethod
    def get_concept_names() -> List[str]:
        return TemporalConcepts.NAMES

    @staticmethod
    def get_trajectory_feature_names() -> List[str]:
        return TemporalConcepts.TRAJECTORY_FEATURES


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    'hidden_dim': 64,
    'num_layers': 4,
    'kernel_size': 7,
    'dropout': 0.2,
    'num_segments': 4,
    'use_learned_concepts': True,
    'trajectory_hidden_dim': 32,
}
