"""
CCECE Paper: TCDN with Gradient-Integrated Segment Attribution (GISA)

Three approaches to achieve faithful explanations while preserving TCDN's clinical concepts:

Approach 1: TCDN-GISA
- Uses IntegratedGradients on base TCDN for per-timestep attribution
- Gradients flow through concept computation → trajectory encoding → classifier
- Preserves TCDN architecture completely, just uses IG for explanation

Approach 2: TCDN-FaithfulBottleneck
- Multiplicative gating: importance scores literally mask input before processing
- Faithful by construction: if timestep importance is 0, it cannot affect prediction
- Based on Information Bottleneck principle

Approach 3: TCDN-MultiScale (conceptual - implemented as explanation method)
- Accepts segment-level concepts as one layer of explanation
- Combines with IG for timestep-level attribution
- Multi-scale framework: IG says WHICH, concepts say WHY

Reference:
- Integrated Gradients: Sundararajan et al., ICML 2017
- Information Bottleneck: Tishby et al., 2000
- Concept Bottleneck Models: Koh et al., ICML 2020
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
# APPROACH 2: FAITHFUL BOTTLENECK ARCHITECTURE
# =============================================================================

class FaithfulImportanceModule(nn.Module):
    """
    Computes per-timestep importance scores that DIRECTLY gate the input.

    This is "faithful by construction":
    - importance[t] = 0 → timestep t cannot affect prediction
    - importance[t] = 1 → timestep t fully contributes

    The importance module is trained jointly with the classifier, so it learns
    to identify genuinely important timesteps for classification.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.temperature = temperature

        # Small network to compute per-timestep importance
        # Uses 1D convolutions to capture local context
        self.importance_net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),  # Output: 1 importance score per timestep
        )

        # Store for explainability
        self._importance_scores = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute importance scores and apply multiplicative gating.

        Args:
            x: Input (batch, seq_len, input_dim)

        Returns:
            gated_x: Input multiplied by importance (batch, seq_len, input_dim)
            importance: Per-timestep importance scores (batch, seq_len)
        """
        batch_size, seq_len, input_dim = x.shape

        # Compute importance scores
        x_t = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        importance_logits = self.importance_net(x_t)  # (batch, 1, seq_len)
        importance_logits = importance_logits.squeeze(1)  # (batch, seq_len)

        # Apply sigmoid with temperature for soft gating
        # Lower temperature → more binary (0 or 1)
        importance = torch.sigmoid(importance_logits / self.temperature)

        self._importance_scores = importance.detach()

        # Multiplicative gating: x * importance
        # This is the key to faithfulness - if importance[t] = 0, timestep t has no effect
        gated_x = x * importance.unsqueeze(-1)

        return gated_x, importance

    def get_sparsity_loss(self, target_sparsity: float = 0.5) -> torch.Tensor:
        """
        Encourage target sparsity in importance scores.

        Args:
            target_sparsity: Target fraction of low-importance timesteps

        Returns:
            Sparsity regularization loss
        """
        if self._importance_scores is None:
            return torch.tensor(0.0)

        # L1 penalty on importance scores to encourage sparsity
        l1_loss = self._importance_scores.mean()

        # Also penalize deviation from target sparsity
        current_sparsity = (self._importance_scores < 0.5).float().mean()
        sparsity_loss = (current_sparsity - target_sparsity) ** 2

        return 0.1 * l1_loss + 0.1 * sparsity_loss


class TCDNFaithfulBottleneck(BaseTimeSeriesModel):
    """
    TCDN with Faithful Bottleneck: Per-timestep importance that directly gates input.

    Architecture:
        Input → Importance Module → Gated Input → TCN Backbone →
        Segment Concepts → Trajectory Encoder → Classifier

    Faithfulness Guarantee:
    - The importance scores multiplicatively gate the input BEFORE any processing
    - This means the importance directly affects what the model sees
    - If importance[t] = 0, timestep t mathematically cannot influence the output

    Key Innovation:
    - Combines TCDN's clinical concept interpretation with guaranteed faithful attribution
    - Importance module learns which timesteps matter for classification
    - Concept trajectories explain WHY those timesteps matter clinically
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
        importance_temperature: float = 1.0,
        sparsity_weight: float = 0.05,
        **kwargs,
    ):
        super().__init__(input_dim, num_classes, seq_len)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout_rate = dropout
        self.num_segments = num_segments
        self.use_learned_concepts = use_learned_concepts
        self.num_concepts = TemporalConcepts.NUM_CONCEPTS
        self.num_trajectory_features = TemporalConcepts.NUM_TRAJECTORY_FEATURES
        self.sparsity_weight = sparsity_weight

        # Faithful importance module - computes and applies per-timestep gating
        self.importance_module = FaithfulImportanceModule(
            input_dim=input_dim,
            hidden_dim=32,
            dropout=dropout,
            temperature=importance_temperature,
        )

        # TCN backbone for feature extraction
        layers = []
        in_channels = input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_channels, hidden_dim, kernel_size,
                                        dilation, dropout))
            in_channels = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Learned concept extraction
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
        self._importance_scores = None

    def extract_segment_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features per temporal segment."""
        batch_size = x.shape[0]
        segment_len = self.seq_len // self.num_segments

        # Apply backbone
        x_t = x.transpose(1, 2)
        features = self.backbone(x_t)
        features = features.transpose(1, 2)

        # Pool within each segment
        segment_features = []
        for seg_idx in range(self.num_segments):
            start = seg_idx * segment_len
            end = start + segment_len if seg_idx < self.num_segments - 1 else self.seq_len
            seg_feat = features[:, start:end, :].mean(dim=1)
            segment_features.append(seg_feat)

        return torch.stack(segment_features, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with faithful importance gating.

        The input is first multiplied by importance scores, then processed.
        This ensures importance scores are faithful by construction.
        """
        batch_size = x.shape[0]

        # Step 1: Compute importance and gate input
        gated_x, importance = self.importance_module(x)
        self._importance_scores = importance

        # Step 2: Compute handcrafted segment concepts from GATED input
        # This is crucial - concepts are computed from the gated input
        raw_concepts = compute_segment_concepts(gated_x, self.num_segments)

        if self.use_learned_concepts:
            # Extract learned features from gated input
            segment_features = self.extract_segment_features(gated_x)
            learned_concepts = self.concept_extractor(segment_features)
            segment_concepts = raw_concepts + 0.5 * learned_concepts
        else:
            segment_concepts = raw_concepts

        self._segment_concepts = segment_concepts.detach()

        # Step 3: Encode trajectory
        encoded_trajectory = self.trajectory_encoder(segment_concepts)

        # Step 4: Compute trajectory features
        trajectory_features = compute_trajectory_features(encoded_trajectory)
        self._trajectory_features = trajectory_features.detach()

        # Step 5: Mean concepts for baseline info
        mean_concepts = segment_concepts.mean(dim=1)

        # Step 6: Classify
        combined = torch.cat([trajectory_features, mean_concepts], dim=1)
        logits = self.classifier(combined)

        return logits

    def forward_with_trajectory(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass returning trajectory information for explainability."""
        logits = self.forward(x)

        trajectory_info = {
            'segment_concepts': self._segment_concepts,
            'trajectory_features': self._trajectory_features,
            'importance_scores': self._importance_scores,
        }

        return logits, trajectory_info

    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Return faithful importance scores as 'attention' for compatibility."""
        self.forward(x)
        return self._importance_scores

    def get_sparsity_loss(self) -> torch.Tensor:
        """Get sparsity regularization loss for importance scores."""
        return self.sparsity_weight * self.importance_module.get_sparsity_loss()

    def explain_prediction(self, x: torch.Tensor) -> Dict[str, Any]:
        """Generate detailed explanation with faithful importance."""
        self.eval()
        with torch.no_grad():
            logits, traj_info = self.forward_with_trajectory(x)
            probs = F.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()

            importance = self._importance_scores[0].cpu().numpy()
            segment_concepts = self._segment_concepts[0].cpu().numpy()

            explanation = {
                'prediction': 'MG' if pred_class == 1 else 'HC',
                'confidence': confidence,
                'importance_scores': importance,
                'sparsity': float((importance < 0.5).mean()),
                'concept_trajectories': {},
                'important_timesteps': [],
            }

            # Find most important timesteps
            top_k = 20
            top_indices = np.argsort(importance)[-top_k:]
            for idx in top_indices:
                explanation['important_timesteps'].append({
                    'timestep': int(idx),
                    'importance': float(importance[idx]),
                    'segment': int(idx // (len(importance) // self.num_segments)),
                })

            # Analyze concept trajectories
            for c_idx, c_name in enumerate(TemporalConcepts.NAMES):
                values = segment_concepts[:, c_idx]
                slope = np.polyfit(range(len(values)), values, 1)[0]
                explanation['concept_trajectories'][c_name] = {
                    'values': values.tolist(),
                    'slope': float(slope),
                    'trend': 'increasing' if slope > 0.01 else ('decreasing' if slope < -0.01 else 'stable'),
                }

            return explanation

    def get_config(self) -> Dict[str, Any]:
        return {
            'model_type': 'TCDNFaithfulBottleneck',
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
            'sparsity_weight': self.sparsity_weight,
        }

    @staticmethod
    def get_concept_names() -> List[str]:
        return TemporalConcepts.NAMES


# =============================================================================
# DEFAULT CONFIGURATIONS
# =============================================================================

TCDN_FAITHFUL_BOTTLENECK_CONFIG = {
    'hidden_dim': 64,
    'num_layers': 4,
    'kernel_size': 7,
    'dropout': 0.2,
    'num_segments': 4,
    'use_learned_concepts': True,
    'trajectory_hidden_dim': 32,
    'importance_temperature': 1.0,
    'sparsity_weight': 0.05,
}
