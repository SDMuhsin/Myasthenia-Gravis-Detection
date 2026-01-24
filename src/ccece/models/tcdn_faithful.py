"""
CCECE Paper: TCDN-Faithful - Per-Timestep Interpretability Architecture

A variant of TCDN designed to achieve high faithfulness metrics by:
1. Learning per-timestep attention within segments (not segment-level pooling)
2. Generating sparse importance scores (~50% sparsity target)
3. Combining attention and gradient signals for faithful explanations

Key Innovation:
- Original TCDN broadcasts identical importance to all ~726 timesteps per segment
- TCDN-Faithful learns which specific timesteps matter within each segment
- Sparse importance generation ensures focused explanations

Architecture:
    Input -> TCN Backbone -> Per-Segment Attention -> Concepts -> Trajectory -> Classifier
                                    |
                           Timestep attention weights
                                    |
                           Sparse importance generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass

from .base import BaseTimeSeriesModel
from .temporal_concept_dynamics import (
    TemporalConcepts,
    compute_segment_concepts,
    compute_trajectory_features,
    CausalConv1d,
    TemporalBlock,
)


# =============================================================================
# SEGMENT-TIMESTEP ATTENTION
# =============================================================================

class SegmentTimestepAttention(nn.Module):
    """
    Learns per-timestep attention weights within each temporal segment.

    Unlike mean-pooling which treats all timesteps equally, this module
    learns to focus on specific timesteps within each segment.

    Features:
    - Query-based attention over timesteps in segment
    - Temperature parameter for controlling sparsity
    - Returns attended features + attention weights
    - Sparsity regularization loss
    """

    def __init__(
        self,
        hidden_dim: int,
        num_segments: int = 4,
        temperature: float = 1.0,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: Dimension of input features
            num_segments: Number of temporal segments
            temperature: Softmax temperature (lower = sparser attention)
            dropout: Dropout rate for attention
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_segments = num_segments
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Query projection for attention
        self.query = nn.Parameter(torch.randn(hidden_dim) * 0.01)

        # Key and value projections
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # Learnable position encoding within segment
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim) * 0.01)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Store attention weights for explainability
        self._attention_weights = None

    def forward(
        self,
        features: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply per-timestep attention within each segment.

        Args:
            features: TCN output features (batch, seq_len, hidden_dim)
            seq_len: Original sequence length

        Returns:
            segment_features: Attended features (batch, num_segments, hidden_dim)
            attention_weights: Per-timestep weights (batch, seq_len)
        """
        batch_size = features.shape[0]
        segment_len = seq_len // self.num_segments

        # Add positional encoding
        if features.shape[1] <= self.pos_encoding.shape[1]:
            features = features + self.pos_encoding[:, :features.shape[1], :]

        # Initialize outputs
        segment_features = []
        all_attention_weights = []

        for seg_idx in range(self.num_segments):
            start = seg_idx * segment_len
            end = start + segment_len if seg_idx < self.num_segments - 1 else seq_len

            # Get segment features
            seg_features = features[:, start:end, :]  # (batch, seg_len, hidden_dim)
            seg_len_actual = seg_features.shape[1]

            # Compute keys and values
            keys = self.key_proj(seg_features)  # (batch, seg_len, hidden_dim)
            values = self.value_proj(seg_features)  # (batch, seg_len, hidden_dim)

            # Compute attention scores: query @ keys.T
            # query: (hidden_dim,), keys: (batch, seg_len, hidden_dim)
            scores = torch.einsum('d,bsd->bs', self.query, keys)  # (batch, seg_len)

            # Apply temperature and softmax
            # Lower temperature = sparser (more peaked) attention
            attention = F.softmax(scores / self.temperature.abs().clamp(min=0.1), dim=-1)
            attention = self.dropout(attention)

            # Weighted sum of values
            attended = torch.einsum('bs,bsd->bd', attention, values)  # (batch, hidden_dim)
            attended = self.layer_norm(attended)

            segment_features.append(attended)
            all_attention_weights.append(attention)

        # Stack segment features
        segment_features = torch.stack(segment_features, dim=1)  # (batch, num_segments, hidden_dim)

        # Concatenate attention weights for full sequence
        # Need to pad last segment if necessary
        attention_weights = torch.zeros(batch_size, seq_len, device=features.device)
        for seg_idx in range(self.num_segments):
            start = seg_idx * segment_len
            end = start + segment_len if seg_idx < self.num_segments - 1 else seq_len
            seg_len_actual = end - start
            attention_weights[:, start:end] = all_attention_weights[seg_idx][:, :seg_len_actual]

        self._attention_weights = attention_weights.detach()

        return segment_features, attention_weights

    def get_sparsity_loss(self) -> torch.Tensor:
        """
        Compute sparsity regularization loss.

        Encourages sparse attention by penalizing uniform distributions.
        Uses entropy-based regularization: we want LOW entropy (peaky attention).
        """
        if self._attention_weights is None:
            return torch.tensor(0.0)

        # Compute entropy of attention distribution per segment
        # Lower entropy = sparser attention
        attention = self._attention_weights

        # Add small epsilon for numerical stability
        eps = 1e-8
        entropy = -(attention * (attention + eps).log()).sum(dim=-1).mean()

        # We want low entropy, so we return positive entropy as loss
        # The model will minimize this, leading to sparser attention
        return entropy


# =============================================================================
# SPARSE IMPORTANCE GENERATOR
# =============================================================================

class SparseImportanceGenerator(nn.Module):
    """
    Combines attention weights and gradient signals into sparse importance scores.

    Features:
    - Fusion network combining attention and gradient signals
    - Aggressive thresholding to achieve ~50% sparsity
    - Percentile-based adaptive thresholding
    """

    def __init__(
        self,
        target_sparsity: float = 0.5,
        threshold_init: float = 0.3,
    ):
        """
        Args:
            target_sparsity: Target fraction of zero importance (0.5 = 50%)
            threshold_init: Initial threshold value (higher = more sparse)
        """
        super().__init__()
        self.target_sparsity = target_sparsity

        # Use higher threshold for more aggressive sparsity
        self.threshold = nn.Parameter(torch.tensor(threshold_init))

        # Fusion weights for combining attention and gradients
        self.attention_weight = nn.Parameter(torch.tensor(0.6))
        self.gradient_weight = nn.Parameter(torch.tensor(0.4))

    def forward(
        self,
        attention_weights: torch.Tensor,
        gradient_importance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate sparse importance scores.

        Args:
            attention_weights: Attention from SegmentTimestepAttention (batch, seq_len)
            gradient_importance: Optional gradient-based importance (batch, seq_len)

        Returns:
            sparse_importance: Thresholded importance scores (batch, seq_len)
        """
        # Normalize attention weights to [0, 1]
        attention = attention_weights / (attention_weights.max(dim=-1, keepdim=True)[0] + 1e-8)

        if gradient_importance is not None:
            # Normalize gradient importance
            grad_imp = gradient_importance / (gradient_importance.max(dim=-1, keepdim=True)[0] + 1e-8)

            # Fuse attention and gradient signals - weight gradients more heavily
            # as they directly relate to model decision-making
            weights = F.softmax(torch.stack([self.attention_weight, self.gradient_weight]), dim=0)
            combined = weights[0] * attention + weights[1] * grad_imp
        else:
            combined = attention

        # Use percentile-based thresholding to achieve target sparsity
        # This ensures we get ~50% zeros regardless of value distribution
        batch_size, seq_len = combined.shape

        # Compute threshold as the percentile that gives target sparsity
        # Sort values and find the cutoff
        sorted_vals, _ = combined.sort(dim=-1)
        percentile_idx = int(self.target_sparsity * seq_len)
        percentile_idx = min(percentile_idx, seq_len - 1)
        adaptive_threshold = sorted_vals[:, percentile_idx:percentile_idx+1]

        # Apply thresholding: values below threshold become 0
        sparse_importance = torch.where(
            combined > adaptive_threshold,
            combined,
            torch.zeros_like(combined)
        )

        # Re-normalize to [0, 1]
        max_val = sparse_importance.max(dim=-1, keepdim=True)[0]
        sparse_importance = sparse_importance / (max_val + 1e-8)

        return sparse_importance

    def get_sparsity_loss(self, importance: torch.Tensor) -> torch.Tensor:
        """
        Compute loss to encourage target sparsity.

        Args:
            importance: Importance scores (batch, seq_len)

        Returns:
            sparsity_loss: Loss encouraging target sparsity level
        """
        # Compute current sparsity (fraction of near-zero values)
        near_zero = (importance < 0.1).float().mean()

        # L2 loss to match target sparsity
        sparsity_loss = (near_zero - self.target_sparsity) ** 2

        return sparsity_loss


# =============================================================================
# TCDN-FAITHFUL MODEL
# =============================================================================

class TCDNFaithful(BaseTimeSeriesModel):
    """
    TCDN with Faithful Per-Timestep Explanations.

    Key differences from base TCDN:
    1. Uses SegmentTimestepAttention instead of mean-pooling
    2. Includes forward_with_explanation() for gradient-based attribution
    3. Generates sparse importance scores (~50% sparsity)
    4. Preserves clinical interpretability through concepts

    Architecture:
        Input -> TCN Backbone -> Per-Segment Attention -> Concepts -> Trajectory -> Classifier
                                        |
                                Attention weights
                                        +
                                Gradient importance
                                        |
                                Sparse importance (output)
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
        attention_temperature: float = 0.5,
        target_sparsity: float = 0.5,
        sparsity_weight: float = 0.1,
        **kwargs,
    ):
        """
        Args:
            input_dim: Number of input features (14)
            num_classes: Number of output classes (2)
            seq_len: Sequence length
            hidden_dim: Hidden dimension in TCN
            num_layers: Number of temporal blocks
            kernel_size: Convolution kernel size
            dropout: Dropout probability
            num_segments: Number of temporal segments for trajectory
            use_learned_concepts: If True, learn concept extraction
            trajectory_hidden_dim: Hidden dimension for trajectory encoder
            attention_temperature: Temperature for attention (lower = sparser)
            target_sparsity: Target sparsity level (0.5 = 50%)
            sparsity_weight: Weight for sparsity regularization in loss
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
        self.attention_temperature = attention_temperature
        self.target_sparsity = target_sparsity
        self.sparsity_weight = sparsity_weight

        # TCN backbone for feature extraction
        layers = []
        in_channels = input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_channels, hidden_dim, kernel_size,
                                        dilation, dropout))
            in_channels = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Per-segment timestep attention (replaces mean pooling)
        self.segment_attention = SegmentTimestepAttention(
            hidden_dim=hidden_dim,
            num_segments=num_segments,
            temperature=attention_temperature,
            dropout=dropout,
        )

        # Sparse importance generator
        self.importance_generator = SparseImportanceGenerator(
            target_sparsity=target_sparsity,
        )

        # Learned concept extraction (optional enhancement over handcrafted)
        if use_learned_concepts:
            self.concept_extractor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_concepts),
            )

        # Trajectory encoder
        self.trajectory_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.num_concepts,
                nhead=1,
                dim_feedforward=trajectory_hidden_dim,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=1,
        )

        # Final classifier
        total_trajectory_features = self.num_concepts * self.num_trajectory_features
        self.classifier = nn.Sequential(
            nn.Linear(total_trajectory_features + self.num_concepts, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Storage for explainability
        self._segment_concepts = None
        self._trajectory_features = None
        self._encoded_trajectory = None
        self._attention_weights = None
        self._backbone_features = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with per-timestep attention.

        Args:
            x: Input (batch, seq_len, input_dim)

        Returns:
            Logits (batch, num_classes)
        """
        batch_size, seq_len, _ = x.shape

        # Compute handcrafted segment concepts from raw input
        raw_concepts = compute_segment_concepts(x, self.num_segments)

        # Apply TCN backbone
        x_t = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        features = self.backbone(x_t)  # (batch, hidden_dim, seq_len)
        features = features.transpose(1, 2)  # (batch, seq_len, hidden_dim)
        self._backbone_features = features

        if self.use_learned_concepts:
            # Apply per-segment attention (instead of mean pooling)
            segment_features, attention_weights = self.segment_attention(
                features, seq_len
            )
            self._attention_weights = attention_weights

            # Extract learned concepts from attended features
            learned_concepts = self.concept_extractor(segment_features)

            # Combine handcrafted and learned
            segment_concepts = raw_concepts + 0.5 * learned_concepts
        else:
            # Use raw concepts with attention for feature selection
            segment_features, attention_weights = self.segment_attention(
                features, seq_len
            )
            self._attention_weights = attention_weights
            segment_concepts = raw_concepts

        self._segment_concepts = segment_concepts.detach()

        # Encode trajectory
        encoded_trajectory = self.trajectory_encoder(segment_concepts)
        self._encoded_trajectory = encoded_trajectory.detach()

        # Compute trajectory features
        trajectory_features = compute_trajectory_features(encoded_trajectory)
        self._trajectory_features = trajectory_features.detach()

        # Mean concepts for baseline info
        mean_concepts = segment_concepts.mean(dim=1)

        # Classify
        combined = torch.cat([trajectory_features, mean_concepts], dim=1)
        logits = self.classifier(combined)

        return logits

    def forward_with_explanation(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with gradient-based explanation.

        This computes both attention-based and gradient-based importance,
        then combines them into sparse importance scores.

        Args:
            x: Input (batch, seq_len, input_dim)
            target_class: Class to explain (None = predicted class)

        Returns:
            logits: (batch, num_classes)
            explanation: Dict containing:
                - attention_weights: Per-timestep attention (batch, seq_len)
                - gradient_importance: Gradient-based importance (batch, seq_len)
                - sparse_importance: Combined sparse importance (batch, seq_len)
                - segment_concepts: Concept values per segment
                - trajectory_features: Trajectory analysis features
        """
        # Enable gradients for input
        x_grad = x.clone().requires_grad_(True)

        # Forward pass
        logits = self.forward(x_grad)

        # Determine target class
        if target_class is None:
            target_class = logits.argmax(dim=1)
        elif isinstance(target_class, int):
            target_class = torch.full((x.shape[0],), target_class, device=x.device)

        # Compute gradients w.r.t. input
        # This tells us which input timesteps affect the prediction
        batch_size = x.shape[0]
        target_logits = logits[torch.arange(batch_size), target_class]

        # Sum target logits and backprop
        target_logits.sum().backward(retain_graph=True)

        # Gradient importance: absolute gradient magnitude per timestep
        gradient_importance = x_grad.grad.abs().sum(dim=-1)  # (batch, seq_len)
        gradient_importance = gradient_importance / (gradient_importance.max(dim=-1, keepdim=True)[0] + 1e-8)

        # Get attention weights
        attention_weights = self._attention_weights

        # Generate sparse importance
        sparse_importance = self.importance_generator(
            attention_weights.detach(),
            gradient_importance.detach(),
        )

        explanation = {
            'attention_weights': attention_weights.detach(),
            'gradient_importance': gradient_importance.detach(),
            'sparse_importance': sparse_importance.detach(),
            'segment_concepts': self._segment_concepts,
            'trajectory_features': self._trajectory_features,
        }

        return logits.detach(), explanation

    def get_sparsity_loss(self) -> torch.Tensor:
        """
        Get total sparsity regularization loss.

        Combines attention sparsity and importance sparsity losses.
        """
        attention_loss = self.segment_attention.get_sparsity_loss()

        # If we have attention weights, also compute importance sparsity
        if self._attention_weights is not None:
            sparse_imp = self.importance_generator(self._attention_weights, None)
            importance_loss = self.importance_generator.get_sparsity_loss(sparse_imp)
        else:
            importance_loss = torch.tensor(0.0, device=attention_loss.device)

        return self.sparsity_weight * (attention_loss + importance_loss)

    def forward_with_trajectory(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass returning trajectory information (for compatibility with base TCDN).

        Returns:
            logits: (batch, num_classes)
            trajectory_info: Dict with segment concepts and trajectory features
        """
        logits = self.forward(x)

        trajectory_info = {
            'segment_concepts': self._segment_concepts,
            'trajectory_features': self._trajectory_features,
            'encoded_trajectory': self._encoded_trajectory,
            'attention_weights': self._attention_weights,
        }

        return logits, trajectory_info

    def explain_prediction(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Generate detailed trajectory-based explanation with faithful importance.

        Args:
            x: Single sample (1, seq_len, input_dim)

        Returns:
            Explanation dictionary with trajectory analysis and importance
        """
        self.eval()

        # Get explanation with gradients
        logits, expl = self.forward_with_explanation(x)
        probs = F.softmax(logits, dim=1)
        pred_class = logits.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

        segment_concepts = self._segment_concepts[0]  # (num_segments, num_concepts)
        sparse_importance = expl['sparse_importance'][0].cpu().numpy()

        # Compute sparsity
        sparsity = (sparse_importance < 0.1).mean()

        explanation = {
            'prediction': 'MG' if pred_class == 1 else 'HC',
            'confidence': confidence,
            'sparsity': float(sparsity),
            'concept_trajectories': {},
            'fatigue_indicators': [],
            'stability_indicators': [],
            'important_regions': [],
            'summary': '',
        }

        # Find important temporal regions
        seq_len = x.shape[1]
        segment_len = seq_len // self.num_segments

        for seg_idx in range(self.num_segments):
            start = seg_idx * segment_len
            end = start + segment_len if seg_idx < self.num_segments - 1 else seq_len
            seg_importance = sparse_importance[start:end]

            # Find peaks within segment
            if seg_importance.max() > 0.5:
                peak_idx = start + seg_importance.argmax()
                explanation['important_regions'].append({
                    'segment': seg_idx + 1,
                    'timestep': int(peak_idx),
                    'importance': float(seg_importance.max()),
                })

        # Analyze concept trajectories
        for c_idx, c_name in enumerate(TemporalConcepts.NAMES):
            values = segment_concepts[:, c_idx].cpu().numpy()

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

            # Identify fatigue indicators
            if c_name in ['Tracking Accuracy', 'Saccade Smoothness', 'Binocular Coordination']:
                if slope < -0.02:
                    explanation['fatigue_indicators'].append(
                        f"{c_name} decreases from {initial:.2f} to {final:.2f} "
                        f"(slope={slope:.3f}, indicates fatigue)"
                    )

            if variance > 0.01:
                explanation['stability_indicators'].append(
                    f"{c_name} shows high variability (var={variance:.3f})"
                )

        # Generate summary
        important_count = len(explanation['important_regions'])
        if explanation['fatigue_indicators']:
            fatigue_summary = "; ".join(explanation['fatigue_indicators'][:2])
            explanation['summary'] = (
                f"Predicted {explanation['prediction']} ({confidence:.0%} confidence). "
                f"Sparsity: {sparsity:.0%}. "
                f"Found {important_count} key regions. "
                f"Fatigue: {fatigue_summary}"
            )
        else:
            explanation['summary'] = (
                f"Predicted {explanation['prediction']} ({confidence:.0%} confidence). "
                f"Sparsity: {sparsity:.0%}. "
                f"Found {important_count} key regions. No significant fatigue."
            )

        return explanation

    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Get attention weights for explainability."""
        self.forward(x)
        return self._attention_weights

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': 'TCDNFaithful',
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
            'attention_temperature': self.attention_temperature,
            'target_sparsity': self.target_sparsity,
            'sparsity_weight': self.sparsity_weight,
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
    'attention_temperature': 0.5,
    'target_sparsity': 0.5,
    'sparsity_weight': 0.1,
}
