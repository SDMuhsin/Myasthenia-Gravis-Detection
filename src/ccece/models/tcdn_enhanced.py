"""
CCECE Paper: Enhanced Temporal Concept Dynamics Network (TCDN-E)

An enhanced version of TCDN with three performance-boosting strategies:

1. FATIGUE-AWARE LOSS: Encourages the model to learn that MG patients show
   concept degradation over time (negative slope = fatigue pattern)

2. MULTI-SCALE TRAJECTORIES: Captures patterns at multiple temporal scales
   (4 segments for coarse trends, 8 segments for fine-grained patterns)

3. ADAPTIVE SEGMENT WEIGHTING: Learns which temporal segments are most
   informative for classification (attention over segments)

These enhancements are specifically designed for fatigue-based diseases like MG,
where the temporal evolution of symptoms is a key diagnostic signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

from .base import BaseTimeSeriesModel
from .temporal_concept_dynamics import (
    TemporalConcepts,
    compute_segment_concepts,
    compute_trajectory_features,
    CausalConv1d,
    TemporalBlock,
)


# =============================================================================
# MULTI-SCALE TRAJECTORY MODULE
# =============================================================================

class MultiScaleTrajectoryEncoder(nn.Module):
    """
    Encodes concept trajectories at multiple temporal scales.

    Combines coarse-grained (4 segments) and fine-grained (8 segments)
    trajectories to capture both overall trends and local patterns.
    """

    def __init__(
        self,
        num_concepts: int,
        hidden_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_concepts = num_concepts

        # Coarse scale encoder (4 segments)
        self.coarse_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=num_concepts,
                nhead=1,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=1,
        )

        # Fine scale encoder (8 segments)
        self.fine_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=num_concepts,
                nhead=1,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=1,
        )

        # Scale fusion layer
        self.scale_fusion = nn.Sequential(
            nn.Linear(num_concepts * 2, num_concepts),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        coarse_concepts: torch.Tensor,  # (batch, 4, num_concepts)
        fine_concepts: torch.Tensor,     # (batch, 8, num_concepts)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused_trajectory: (batch, num_segments, num_concepts)
            scale_features: (batch, num_concepts) - fused scale representation
        """
        # Encode each scale
        coarse_encoded = self.coarse_encoder(coarse_concepts)  # (batch, 4, num_concepts)
        fine_encoded = self.fine_encoder(fine_concepts)        # (batch, 8, num_concepts)

        # Global representation from each scale
        coarse_global = coarse_encoded.mean(dim=1)  # (batch, num_concepts)
        fine_global = fine_encoded.mean(dim=1)      # (batch, num_concepts)

        # Fuse scales
        scale_features = self.scale_fusion(
            torch.cat([coarse_global, fine_global], dim=1)
        )

        return coarse_encoded, scale_features


# =============================================================================
# ADAPTIVE SEGMENT ATTENTION
# =============================================================================

class AdaptiveSegmentAttention(nn.Module):
    """
    Learns to weight temporal segments based on their informativeness.

    Some segments may be more diagnostic than others (e.g., late segments
    showing fatigue may be more informative for MG detection).
    """

    def __init__(self, num_concepts: int, num_segments: int):
        super().__init__()

        # Learnable segment importance weights
        self.segment_attention = nn.Sequential(
            nn.Linear(num_concepts, num_concepts // 2),
            nn.Tanh(),
            nn.Linear(num_concepts // 2, 1),
        )

        # Position embedding to encode temporal position
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_segments, num_concepts) * 0.02
        )

    def forward(self, segment_concepts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            segment_concepts: (batch, num_segments, num_concepts)

        Returns:
            weighted_concepts: (batch, num_concepts) - attention-weighted summary
            attention_weights: (batch, num_segments) - learned segment importance
        """
        batch_size, num_segments, num_concepts = segment_concepts.shape

        # Add position information
        positioned = segment_concepts + self.position_embedding[:, :num_segments, :]

        # Compute attention scores
        scores = self.segment_attention(positioned).squeeze(-1)  # (batch, num_segments)
        attention_weights = F.softmax(scores, dim=1)

        # Weighted sum
        weighted_concepts = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, num_segments)
            segment_concepts                  # (batch, num_segments, num_concepts)
        ).squeeze(1)  # (batch, num_concepts)

        return weighted_concepts, attention_weights


# =============================================================================
# FATIGUE-AWARE LOSS
# =============================================================================

class FatigueAwareLoss(nn.Module):
    """
    Loss function that encourages the model to learn fatigue patterns.

    For MG patients (label=1), we expect:
    - Negative slopes (concepts decrease over time)
    - Higher variance (less stable performance)

    For HC patients (label=0), we expect:
    - Stable or positive slopes
    - Lower variance

    This loss term provides weak supervision based on clinical knowledge.
    """

    def __init__(self, fatigue_weight: float = 0.3):
        super().__init__()
        self.fatigue_weight = fatigue_weight

    def forward(
        self,
        trajectory_features: torch.Tensor,  # (batch, num_concepts * 5)
        labels: torch.Tensor,               # (batch,)
        num_concepts: int = 5,
    ) -> torch.Tensor:
        """
        Compute fatigue-aware regularization loss.

        Trajectory features are organized as:
        [initial_0, final_0, slope_0, curvature_0, variance_0,
         initial_1, final_1, slope_1, curvature_1, variance_1, ...]
        """
        batch_size = labels.shape[0]

        # Extract slopes (index 2 for each concept)
        slopes = []
        for c in range(num_concepts):
            slope_idx = c * 5 + 2  # slope is at index 2 within each concept's features
            slopes.append(trajectory_features[:, slope_idx])
        slopes = torch.stack(slopes, dim=1)  # (batch, num_concepts)

        # Extract variances (index 4 for each concept)
        variances = []
        for c in range(num_concepts):
            var_idx = c * 5 + 4  # variance is at index 4
            variances.append(trajectory_features[:, var_idx])
        variances = torch.stack(variances, dim=1)  # (batch, num_concepts)

        # For MG patients (label=1): encourage negative slopes (fatigue)
        # For HC patients (label=0): encourage stable/positive slopes
        mg_mask = labels.float().unsqueeze(1)  # (batch, 1)
        hc_mask = 1.0 - mg_mask

        # MG loss: penalize positive slopes (want negative = fatigue)
        # We use a soft margin: slope should be < 0 for MG
        mg_slope_loss = F.relu(slopes + 0.05) * mg_mask  # Penalize slopes > -0.05 for MG

        # HC loss: penalize very negative slopes (want stable)
        hc_slope_loss = F.relu(-slopes - 0.05) * hc_mask  # Penalize slopes < -0.05 for HC

        # Variance loss: MG should have higher variance
        # We don't enforce this strongly, just add a small regularization
        variance_diff = (variances * hc_mask - variances * mg_mask).mean()

        total_loss = (mg_slope_loss.mean() + hc_slope_loss.mean() +
                      0.1 * F.relu(-variance_diff))

        return self.fatigue_weight * total_loss


# =============================================================================
# ENHANCED TCDN MODEL
# =============================================================================

class EnhancedTCDN(BaseTimeSeriesModel):
    """
    Enhanced Temporal Concept Dynamics Network (TCDN-E).

    Improvements over base TCDN:
    1. Multi-scale trajectory encoding (4 + 8 segments)
    2. Adaptive segment attention (learns segment importance)
    3. Fatigue-aware loss (clinical knowledge regularization)

    Architecture:
        Input → TCN Backbone → Segment Pooling (4 & 8 segments)
                                    ↓
                            Multi-Scale Concepts
                                    ↓
                        Trajectory Encoder + Attention
                                    ↓
                        Fatigue-Aware Classification
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
        fatigue_loss_weight: float = 0.3,
        use_multi_scale: bool = True,
        use_segment_attention: bool = True,
        **kwargs,
    ):
        super().__init__(input_dim, num_classes, seq_len)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout_rate = dropout
        self.fatigue_loss_weight = fatigue_loss_weight
        self.use_multi_scale = use_multi_scale
        self.use_segment_attention = use_segment_attention
        self.num_concepts = TemporalConcepts.NUM_CONCEPTS
        self.num_trajectory_features = TemporalConcepts.NUM_TRAJECTORY_FEATURES

        # Segment configurations
        self.coarse_segments = 4
        self.fine_segments = 8

        # TCN backbone
        layers = []
        in_channels = input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_channels, hidden_dim, kernel_size,
                                        dilation, dropout))
            in_channels = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Learned concept refinement
        self.concept_refiner = nn.Sequential(
            nn.Linear(self.num_concepts, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.num_concepts),
        )

        # Multi-scale trajectory encoder
        if use_multi_scale:
            self.multi_scale_encoder = MultiScaleTrajectoryEncoder(
                self.num_concepts, hidden_dim // 2, dropout
            )

        # Segment attention
        if use_segment_attention:
            self.segment_attention = AdaptiveSegmentAttention(
                self.num_concepts, self.coarse_segments
            )

        # Fatigue-aware loss
        self.fatigue_loss = FatigueAwareLoss(fatigue_loss_weight)

        # Calculate classifier input dimension
        classifier_input_dim = self.num_concepts * self.num_trajectory_features  # trajectory features
        classifier_input_dim += self.num_concepts  # mean concepts
        if use_multi_scale:
            classifier_input_dim += self.num_concepts  # scale fusion features
        if use_segment_attention:
            classifier_input_dim += self.num_concepts  # attention-weighted concepts

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Storage for explainability
        self._coarse_concepts = None
        self._fine_concepts = None
        self._trajectory_features = None
        self._attention_weights = None

    def _compute_segment_concepts(self, x: torch.Tensor, num_segments: int) -> torch.Tensor:
        """Compute concepts for given number of segments."""
        raw_concepts = compute_segment_concepts(x, num_segments)

        # Apply learned refinement
        batch_size, n_seg, n_concepts = raw_concepts.shape
        refined = self.concept_refiner(raw_concepts.view(-1, n_concepts))
        refined = refined.view(batch_size, n_seg, n_concepts)

        # Residual connection
        return raw_concepts + 0.3 * refined

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.shape[0]

        # Compute concepts at multiple scales
        coarse_concepts = self._compute_segment_concepts(x, self.coarse_segments)
        self._coarse_concepts = coarse_concepts.detach()

        features_list = []

        if self.use_multi_scale:
            fine_concepts = self._compute_segment_concepts(x, self.fine_segments)
            self._fine_concepts = fine_concepts.detach()

            # Multi-scale encoding
            encoded_trajectory, scale_features = self.multi_scale_encoder(
                coarse_concepts, fine_concepts
            )
            features_list.append(scale_features)
        else:
            encoded_trajectory = coarse_concepts

        # Compute trajectory features from coarse scale
        trajectory_features = compute_trajectory_features(encoded_trajectory)
        self._trajectory_features = trajectory_features.detach()
        features_list.append(trajectory_features)

        # Mean concepts
        mean_concepts = coarse_concepts.mean(dim=1)
        features_list.append(mean_concepts)

        # Segment attention
        if self.use_segment_attention:
            weighted_concepts, attention_weights = self.segment_attention(coarse_concepts)
            self._attention_weights = attention_weights.detach()
            features_list.append(weighted_concepts)

        # Concatenate all features
        combined_features = torch.cat(features_list, dim=1)

        # Classify
        logits = self.classifier(combined_features)

        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss including fatigue-aware regularization.
        """
        # Classification loss
        if class_weights is not None:
            cls_loss = F.cross_entropy(logits, labels, weight=class_weights)
        else:
            cls_loss = F.cross_entropy(logits, labels)

        # Fatigue-aware loss
        fatigue_loss = self.fatigue_loss(
            self._trajectory_features,
            labels,
            self.num_concepts,
        )

        total_loss = cls_loss + fatigue_loss

        return total_loss, {
            'total_loss': total_loss.item(),
            'cls_loss': cls_loss.item(),
            'fatigue_loss': fatigue_loss.item(),
        }

    def explain_prediction(self, x: torch.Tensor) -> Dict[str, Any]:
        """Generate detailed explanation with all enhancement features."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()

            explanation = {
                'prediction': 'MG' if pred_class == 1 else 'HC',
                'confidence': confidence,
                'concept_trajectories': {},
                'fatigue_indicators': [],
                'segment_importance': {},
                'multi_scale_analysis': {},
                'summary': '',
            }

            # Analyze coarse concept trajectories
            coarse_concepts = self._coarse_concepts[0].cpu().numpy()

            for c_idx, c_name in enumerate(TemporalConcepts.NAMES):
                values = coarse_concepts[:, c_idx]
                slope = np.polyfit(range(len(values)), values, 1)[0]

                explanation['concept_trajectories'][c_name] = {
                    'Q1': float(values[0]),
                    'Q2': float(values[1]),
                    'Q3': float(values[2]),
                    'Q4': float(values[3]),
                    'slope': float(slope),
                    'change': float(values[-1] - values[0]),
                }

                # Identify fatigue patterns
                if slope < -0.02:
                    pct_change = (values[-1] - values[0]) / (values[0] + 1e-8) * 100
                    explanation['fatigue_indicators'].append(
                        f"{c_name}: {values[0]:.2f} → {values[-1]:.2f} "
                        f"({pct_change:.0f}% change, slope={slope:.3f})"
                    )

            # Segment importance from attention
            if self._attention_weights is not None:
                attention = self._attention_weights[0].cpu().numpy()
                for i, weight in enumerate(attention):
                    explanation['segment_importance'][f'Q{i+1}'] = float(weight)

                # Find most important segment
                most_important = np.argmax(attention)
                explanation['multi_scale_analysis']['most_important_segment'] = f'Q{most_important+1}'
                explanation['multi_scale_analysis']['attention_weights'] = attention.tolist()

            # Multi-scale comparison
            if self._fine_concepts is not None:
                fine_concepts = self._fine_concepts[0].cpu().numpy()
                explanation['multi_scale_analysis']['fine_scale_available'] = True
                explanation['multi_scale_analysis']['fine_segments'] = 8

            # Generate summary
            if explanation['fatigue_indicators']:
                fatigue_summary = "; ".join(explanation['fatigue_indicators'][:2])
                explanation['summary'] = (
                    f"Predicted {explanation['prediction']} ({confidence:.0%}). "
                    f"FATIGUE DETECTED: {fatigue_summary}"
                )
            else:
                explanation['summary'] = (
                    f"Predicted {explanation['prediction']} ({confidence:.0%}). "
                    f"No significant fatigue pattern detected (stable performance)."
                )

            return explanation

    def get_config(self) -> Dict[str, Any]:
        return {
            'model_type': 'EnhancedTCDN',
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'seq_len': self.seq_len,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout_rate,
            'num_concepts': self.num_concepts,
            'use_multi_scale': self.use_multi_scale,
            'use_segment_attention': self.use_segment_attention,
            'fatigue_loss_weight': self.fatigue_loss_weight,
            'coarse_segments': self.coarse_segments,
            'fine_segments': self.fine_segments,
        }

    @staticmethod
    def get_concept_names() -> List[str]:
        return TemporalConcepts.NAMES


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    'hidden_dim': 64,
    'num_layers': 4,
    'kernel_size': 7,
    'dropout': 0.2,
    'fatigue_loss_weight': 0.3,
    'use_multi_scale': True,
    'use_segment_attention': True,
}
