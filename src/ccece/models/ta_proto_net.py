"""
CCECE Paper: Temporally-Attentive Prototype Network (TAProtoNet)

A minimal enhancement to TempProtoNet that adds temporal attention to learn
which temporal regions are most discriminative before prototype comparison.

Novelty: Combines learned temporal attention with prototype-based classification,
enabling the model to focus on the most discriminative time segments (e.g., late
segments showing MG fatigue) while maintaining intrinsic explainability.

Key difference from TempProtoNet:
- TempProtoNet: Global pooling → all timesteps contribute equally
- TAProtoNet: Temporal attention → learns which timesteps matter most

This is a minimal, targeted change to a working baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from .base import BaseTimeSeriesModel


class TemporalAttention(nn.Module):
    """
    Learnable temporal attention mechanism.

    Learns to weight different temporal segments before aggregation.
    Provides explainability: "Which time segments contributed most?"
    """

    def __init__(self, hidden_dim: int, n_segments: int):
        """
        Args:
            hidden_dim: Dimension of hidden features
            n_segments: Number of temporal segments
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_segments = n_segments

        # Attention network: maps segment features to attention scores
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        segment_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal attention.

        Args:
            segment_features: (batch_size, n_segments, hidden_dim)

        Returns:
            attended: Attended representation (batch_size, hidden_dim)
            attention_weights: (batch_size, n_segments)
        """
        # Compute attention scores for each segment
        # (batch_size, n_segments, 1)
        scores = self.attention_net(segment_features)

        # Softmax over segments
        attention_weights = F.softmax(scores.squeeze(-1), dim=1)

        # Weighted sum: (batch_size, hidden_dim)
        attended = torch.sum(
            segment_features * attention_weights.unsqueeze(-1),
            dim=1
        )

        return attended, attention_weights


class TAPrototypeLayer(nn.Module):
    """
    Prototype layer (same as TempProtoNet, reused for clarity).
    """

    def __init__(
        self,
        latent_dim: int,
        n_prototypes_per_class: int,
        n_classes: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_prototypes_per_class = n_prototypes_per_class
        self.n_classes = n_classes
        self.n_prototypes = n_prototypes_per_class * n_classes

        # Learnable prototypes
        self.prototypes = nn.Parameter(
            torch.randn(self.n_prototypes, latent_dim) * 0.1
        )

        # Prototype-to-class mapping
        prototype_class = torch.zeros(self.n_prototypes, dtype=torch.long)
        for c in range(n_classes):
            start_idx = c * n_prototypes_per_class
            end_idx = start_idx + n_prototypes_per_class
            prototype_class[start_idx:end_idx] = c
        self.register_buffer('prototype_class', prototype_class)

        # Class identity
        class_identity = torch.zeros(self.n_prototypes, n_classes)
        for i in range(self.n_prototypes):
            class_identity[i, prototype_class[i]] = 1.0
        self.register_buffer('class_identity', class_identity)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute prototype similarities and class logits."""
        z_expanded = z.unsqueeze(1)
        proto_expanded = self.prototypes.unsqueeze(0)
        distances = torch.sum((z_expanded - proto_expanded) ** 2, dim=2)
        similarities = torch.log(1 + 1 / (distances + 1e-6))
        logits = torch.matmul(similarities, self.class_identity)
        return distances, similarities, logits


class TAProtoNet(BaseTimeSeriesModel):
    """
    Temporally-Attentive Prototype Network for explainable MG detection.

    Architecture:
        Input (batch, seq_len, input_dim)
            |
            v
        CNN Encoder -> Feature map (batch, hidden, T_reduced)
            |
            v
        Segment-wise features (batch, n_segments, hidden)
            |
            v
        Temporal Attention -> Attended representation (batch, latent_dim)
            |
            v
        Prototype Layer -> Similarities (batch, n_prototypes)
            |
            v
        Classification -> Logits (batch, n_classes)

    Explainability:
        1. Temporal attention weights show WHEN discriminative patterns appear
        2. Prototype similarities show WHAT patterns are detected
        Combined: "Classified as MG because late-sequence (attention=0.7)
                  matches MG prototype 3 (similarity=0.85)"
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        latent_dim: int = 64,
        n_prototypes_per_class: int = 5,
        encoder_hidden: int = 64,
        encoder_layers: int = 3,
        kernel_size: int = 7,
        dropout: float = 0.2,
        n_segments: int = 8,
        **kwargs,
    ):
        """
        Args:
            input_dim: Number of input features per timestep
            num_classes: Number of output classes
            seq_len: Expected sequence length
            latent_dim: Dimension of latent space
            n_prototypes_per_class: Number of prototypes per class
            encoder_hidden: Hidden dimension in encoder
            encoder_layers: Number of encoder layers
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
            n_segments: Number of temporal segments for attention
        """
        super().__init__(input_dim, num_classes, seq_len)

        self.latent_dim = latent_dim
        self.n_prototypes_per_class = n_prototypes_per_class
        self.encoder_hidden = encoder_hidden
        self.encoder_layers = encoder_layers
        self.kernel_size = kernel_size
        self.dropout_rate = dropout
        self.n_segments = n_segments

        # Build CNN encoder (same as TempProtoNet)
        self.encoder = self._build_encoder()

        # Segment pooling
        self.segment_pool = nn.AdaptiveAvgPool1d(n_segments)

        # Temporal attention
        self.temporal_attention = TemporalAttention(
            hidden_dim=self.encoder_output_dim,
            n_segments=n_segments,
        )

        # Projection to latent space
        self.projection = nn.Sequential(
            nn.Linear(self.encoder_output_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )

        # Prototype layer
        self.prototype_layer = TAPrototypeLayer(
            latent_dim=latent_dim,
            n_prototypes_per_class=n_prototypes_per_class,
            n_classes=num_classes,
        )

        # For storing training embeddings
        self._training_embeddings: Optional[torch.Tensor] = None
        self._training_labels: Optional[torch.Tensor] = None

    def _build_encoder(self) -> nn.Module:
        """Build 1D CNN encoder (same as TempProtoNet)."""
        layers = []
        in_channels = self.input_dim

        for i in range(self.encoder_layers):
            out_channels = self.encoder_hidden * (2 ** min(i, 2))

            layers.extend([
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(self.dropout_rate),
            ])
            in_channels = out_channels

        self.encoder_output_dim = in_channels
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input with temporal attention.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            z: Latent representation (batch_size, latent_dim)
            attention_weights: (batch_size, n_segments)
        """
        # Transpose for conv1d: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)

        # Encode: (batch_size, encoder_output_dim, reduced_seq_len)
        features = self.encoder(x)

        # Pool to segments: (batch_size, encoder_output_dim, n_segments)
        segment_features = self.segment_pool(features)

        # Transpose for attention: (batch_size, n_segments, encoder_output_dim)
        segment_features = segment_features.transpose(1, 2)

        # Apply temporal attention: (batch_size, encoder_output_dim)
        attended, attention_weights = self.temporal_attention(segment_features)

        # Project to latent: (batch_size, latent_dim)
        z = self.projection(attended)

        # L2 normalize
        z = F.normalize(z, p=2, dim=1)

        return z, attention_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        z, _ = self.encode(x)
        _, _, logits = self.prototype_layer(z)
        return logits

    def forward_with_explanations(
        self,
        x: torch.Tensor,
    ) -> Dict[str, Any]:
        """Forward pass with full explanation outputs."""
        z, attention_weights = self.encode(x)
        distances, similarities, logits = self.prototype_layer(z)

        return {
            'logits': logits,
            'z': z,
            'attention_weights': attention_weights,
            'distances': distances,
            'similarities': similarities,
        }

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get temporal attention weights."""
        _, attention_weights = self.encode(x)
        return attention_weights

    def compute_prototype_loss(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        distances: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute prototype-specific losses (same as TempProtoNet)."""
        batch_size = z.size(0)
        prototype_classes = self.prototype_layer.prototype_class
        n_prototypes = distances.size(1)

        # Cluster loss
        cluster_losses = []
        for i in range(batch_size):
            label = labels[i].item()
            same_class_mask = (prototype_classes == label)
            same_class_distances = distances[i][same_class_mask]
            min_dist = same_class_distances.min()
            cluster_losses.append(min_dist)

        cluster_loss = torch.stack(cluster_losses).mean()

        # Separation loss
        prototypes = self.prototype_layer.prototypes
        sep_losses = []
        for c in range(self.num_classes):
            class_c_mask = (prototype_classes == c)
            other_mask = (prototype_classes != c)
            class_c_protos = prototypes[class_c_mask]
            other_protos = prototypes[other_mask]

            if len(class_c_protos) > 0 and len(other_protos) > 0:
                dists = torch.cdist(class_c_protos, other_protos)
                min_dist = dists.min()
                sep_losses.append(-min_dist)

        if sep_losses:
            separation_loss = torch.stack(sep_losses).mean()
        else:
            separation_loss = torch.tensor(0.0, device=z.device)

        return cluster_loss, separation_loss

    def compute_temporal_loss(
        self,
        attention_weights: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optional: Encourage late-segment attention for MG samples.

        Clinical motivation: MG causes fatigue that appears in late segments.
        This is a soft constraint, not a hard-coded rule.
        """
        batch_size = labels.size(0)
        n_segments = attention_weights.size(1)

        # Create position weights: later segments get higher weight
        position_weights = torch.linspace(0.0, 1.0, n_segments, device=attention_weights.device)

        # For MG samples (label=1), encourage attention on later segments
        # For HC samples (label=0), no temporal constraint
        mg_mask = (labels == 1).float()

        # Weighted sum of attention (higher = more late-focused)
        late_attention = (attention_weights * position_weights.unsqueeze(0)).sum(dim=1)

        # Loss: MG samples should have high late_attention
        # This is a soft regularization, not a hard constraint
        temporal_loss = -mg_mask * late_attention  # Negative because we maximize

        return temporal_loss.mean()

    def store_training_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Store training embeddings for prototype projection."""
        self._training_embeddings = embeddings.detach()
        self._training_labels = labels.detach()

    def project_prototypes(self, k: int = 5) -> List[Dict[str, Any]]:
        """Project prototypes to nearest training samples."""
        if self._training_embeddings is None:
            raise ValueError("Training embeddings not stored.")

        prototypes = self.prototype_layer.prototypes.detach()
        prototype_classes = self.prototype_layer.prototype_class

        projections = []
        for proto_idx in range(prototypes.size(0)):
            proto = prototypes[proto_idx:proto_idx+1]
            dists = torch.cdist(proto, self._training_embeddings).squeeze(0)
            topk_dists, topk_indices = torch.topk(dists, k, largest=False)

            projection = {
                'prototype_idx': proto_idx,
                'prototype_class': prototype_classes[proto_idx].item(),
                'nearest_indices': topk_indices.cpu().numpy(),
                'nearest_distances': topk_dists.cpu().numpy(),
                'nearest_labels': self._training_labels[topk_indices].cpu().numpy(),
            }
            projections.append(projection)

        return projections

    def explain_prediction(
        self,
        x: torch.Tensor,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Generate explanations for predictions."""
        with torch.no_grad():
            outputs = self.forward_with_explanations(x)
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=1)
            probabilities = F.softmax(logits, dim=1)
            attention = outputs['attention_weights']
            similarities = outputs['similarities']
            prototype_classes = self.prototype_layer.prototype_class

            explanations = []

            for i in range(x.size(0)):
                pred = predictions[i].item()
                prob = probabilities[i, pred].item()
                attn = attention[i]
                sims = similarities[i]

                # Top-k most similar prototypes
                topk_sims, topk_indices = torch.topk(sims, top_k)
                similar_prototypes = []
                for j in range(top_k):
                    idx = topk_indices[j].item()
                    similar_prototypes.append({
                        'prototype_idx': idx,
                        'prototype_class': prototype_classes[idx].item(),
                        'prototype_class_name': 'MG' if prototype_classes[idx].item() == 1 else 'HC',
                        'similarity': topk_sims[j].item(),
                    })

                # Temporal focus (which segment got most attention)
                top_segment = torch.argmax(attn).item()
                segment_position = 'early' if top_segment < self.n_segments // 2 else 'late'

                explanation = {
                    'prediction': pred,
                    'prediction_name': 'MG' if pred == 1 else 'HC',
                    'confidence': prob,
                    'top_similar_prototypes': similar_prototypes,
                    'attention_weights': attn.cpu().numpy(),
                    'temporal_focus': {
                        'top_segment': top_segment,
                        'top_segment_attention': attn[top_segment].item(),
                        'position': segment_position,
                    },
                }
                explanations.append(explanation)

            return explanations

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': 'TAProtoNet',
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'seq_len': self.seq_len,
            'latent_dim': self.latent_dim,
            'n_prototypes_per_class': self.n_prototypes_per_class,
            'encoder_hidden': self.encoder_hidden,
            'encoder_layers': self.encoder_layers,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout_rate,
            'n_segments': self.n_segments,
            'n_total_prototypes': self.prototype_layer.n_prototypes,
        }

    def compute_prototype_diversity(self) -> Dict[str, float]:
        """Compute prototype diversity metrics (same as TempProtoNet)."""
        prototypes = self.prototype_layer.prototypes.detach()
        prototype_classes = self.prototype_layer.prototype_class

        pairwise_dists = torch.cdist(prototypes, prototypes)
        n = prototypes.size(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=prototypes.device)
        off_diagonal = pairwise_dists[mask]

        mean_dist = off_diagonal.mean().item()
        min_dist = off_diagonal.min().item()

        inter_dists = []
        intra_dists = []
        for i in range(n):
            for j in range(i+1, n):
                dist = pairwise_dists[i, j].item()
                if prototype_classes[i] == prototype_classes[j]:
                    intra_dists.append(dist)
                else:
                    inter_dists.append(dist)

        return {
            'mean_pairwise_distance': mean_dist,
            'min_pairwise_distance': min_dist,
            'inter_class_distance': np.mean(inter_dists) if inter_dists else 0.0,
            'intra_class_distance': np.mean(intra_dists) if intra_dists else 0.0,
        }


# Default configuration
DEFAULT_CONFIG = {
    'latent_dim': 64,
    'n_prototypes_per_class': 5,
    'encoder_hidden': 64,
    'encoder_layers': 3,
    'kernel_size': 7,
    'dropout': 0.2,
    'n_segments': 8,
}
