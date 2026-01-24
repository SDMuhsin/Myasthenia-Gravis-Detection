"""
CCECE Paper: Progressive Prototype Trajectories (PPT)

A novel extension to prototype-based learning that models prototypes as temporal
trajectories rather than static points. Designed to capture progressive fatigue
patterns characteristic of Myasthenia Gravis.

Key Novelty:
- Standard prototype networks: static prototypes p_k in R^d
- PPT: trajectory prototypes P_k(t) = p_k^0 + t * v_k (or polynomial extension)

This enables the model to capture HOW patterns evolve over time, not just
WHAT patterns look like at a snapshot.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from .base import BaseTimeSeriesModel


class TrajectoryPrototypeLayer(nn.Module):
    """
    Prototype layer with trajectory functions instead of static prototypes.

    Each prototype is defined as:
        P_k(t) = p_k^0 + t * v_k   (linear trajectory)
    or:
        P_k(t) = p_k^0 + t * v_k^1 + t^2 * v_k^2   (polynomial trajectory)

    where t is normalized time in [0, 1].
    """

    def __init__(
        self,
        latent_dim: int,
        n_prototypes_per_class: int,
        n_classes: int = 2,
        trajectory_type: str = 'linear',  # 'linear' or 'polynomial'
    ):
        """
        Args:
            latent_dim: Dimension of the latent space
            n_prototypes_per_class: Number of prototypes per class
            n_classes: Number of classes (default 2 for binary)
            trajectory_type: 'linear' (P = p + t*v) or 'polynomial' (P = p + t*v1 + t^2*v2)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.n_prototypes_per_class = n_prototypes_per_class
        self.n_classes = n_classes
        self.n_prototypes = n_prototypes_per_class * n_classes
        self.trajectory_type = trajectory_type

        # Initial prototype positions: p_k^0
        self.prototype_origins = nn.Parameter(
            torch.randn(self.n_prototypes, latent_dim) * 0.1
        )

        # Trajectory velocities: v_k (linear term)
        # Initialize with small non-zero values to avoid collapse
        self.prototype_velocities = nn.Parameter(
            torch.randn(self.n_prototypes, latent_dim) * 0.05
        )

        # Quadratic term for polynomial trajectories
        if trajectory_type == 'polynomial':
            self.prototype_accelerations = nn.Parameter(
                torch.randn(self.n_prototypes, latent_dim) * 0.02
            )

        # Prototype-to-class mapping (fixed)
        prototype_class = torch.zeros(self.n_prototypes, dtype=torch.long)
        for c in range(n_classes):
            start_idx = c * n_prototypes_per_class
            end_idx = start_idx + n_prototypes_per_class
            prototype_class[start_idx:end_idx] = c
        self.register_buffer('prototype_class', prototype_class)

        # One-hot encoding for class prediction
        class_identity = torch.zeros(self.n_prototypes, n_classes)
        for i in range(self.n_prototypes):
            class_identity[i, prototype_class[i]] = 1.0
        self.register_buffer('class_identity', class_identity)

    def get_prototype_at_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get prototype positions at normalized time t.

        Args:
            t: Time values, shape (batch_size, n_segments) or (n_segments,)
               Values should be in [0, 1]

        Returns:
            prototypes: (batch_size, n_segments, n_prototypes, latent_dim)
                       or (n_segments, n_prototypes, latent_dim)
        """
        # Handle different input shapes
        if t.dim() == 1:
            # (n_segments,) -> (n_segments, 1, 1)
            t = t.view(-1, 1, 1)
            squeeze_batch = True
        else:
            # (batch_size, n_segments) -> (batch_size, n_segments, 1, 1)
            t = t.unsqueeze(-1).unsqueeze(-1)
            squeeze_batch = False

        # origins: (1, 1, n_prototypes, latent_dim) or (n_prototypes, latent_dim)
        # velocities: (1, 1, n_prototypes, latent_dim)
        origins = self.prototype_origins.unsqueeze(0).unsqueeze(0)
        velocities = self.prototype_velocities.unsqueeze(0).unsqueeze(0)

        # P_k(t) = p_k^0 + t * v_k
        prototypes = origins + t * velocities

        if self.trajectory_type == 'polynomial':
            accelerations = self.prototype_accelerations.unsqueeze(0).unsqueeze(0)
            prototypes = prototypes + (t ** 2) * accelerations

        if squeeze_batch:
            prototypes = prototypes.squeeze(0)  # Remove batch dim if not present

        return prototypes

    def forward(
        self,
        z_segments: torch.Tensor,  # (batch_size, n_segments, latent_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute trajectory similarity and class logits.

        Args:
            z_segments: Encoded segments (batch_size, n_segments, latent_dim)

        Returns:
            distances: Per-segment distances (batch_size, n_segments, n_prototypes)
            similarities: Per-segment similarities (batch_size, n_segments, n_prototypes)
            trajectory_similarities: Aggregated similarities (batch_size, n_prototypes)
            logits: Class logits (batch_size, n_classes)
        """
        batch_size, n_segments, latent_dim = z_segments.shape

        # Generate normalized time values for each segment
        # t_values: (n_segments,) with values [0, 1/(n-1), 2/(n-1), ..., 1]
        if n_segments > 1:
            t_values = torch.linspace(0, 1, n_segments, device=z_segments.device)
        else:
            t_values = torch.tensor([0.5], device=z_segments.device)

        # Get prototype positions at each time point
        # prototypes: (n_segments, n_prototypes, latent_dim)
        prototypes = self.get_prototype_at_time(t_values)

        # Compute distances between each segment encoding and prototypes at that time
        # z_segments: (batch_size, n_segments, latent_dim)
        # prototypes: (n_segments, n_prototypes, latent_dim)

        # Expand for broadcasting
        z_expanded = z_segments.unsqueeze(2)  # (batch, n_seg, 1, latent)
        proto_expanded = prototypes.unsqueeze(0)  # (1, n_seg, n_proto, latent)

        # Squared L2 distance
        distances = torch.sum((z_expanded - proto_expanded) ** 2, dim=-1)
        # distances: (batch_size, n_segments, n_prototypes)

        # Convert distances to similarities
        similarities = torch.log(1 + 1 / (distances + 1e-6))
        # similarities: (batch_size, n_segments, n_prototypes)

        # Aggregate across time: mean similarity
        trajectory_similarities = similarities.mean(dim=1)
        # trajectory_similarities: (batch_size, n_prototypes)

        # Class logits: weighted sum of similarities for each class
        logits = torch.matmul(trajectory_similarities, self.class_identity)
        # logits: (batch_size, n_classes)

        return distances, similarities, trajectory_similarities, logits

    def get_prototype_classes(self) -> torch.Tensor:
        """Get the class assignment for each prototype."""
        return self.prototype_class

    def compute_velocity_norms(self) -> torch.Tensor:
        """Compute L2 norms of velocity vectors."""
        return torch.norm(self.prototype_velocities, dim=1)

    def compute_trajectory_diversity(self) -> Dict[str, float]:
        """
        Compute metrics about trajectory diversity.

        Returns:
            Dict containing velocity norms and direction similarities
        """
        velocities = self.prototype_velocities.detach()

        # Velocity norms
        velocity_norms = torch.norm(velocities, dim=1)
        mean_velocity_norm = velocity_norms.mean().item()
        min_velocity_norm = velocity_norms.min().item()

        # Velocity direction similarities (cosine)
        normalized_v = F.normalize(velocities, dim=1)
        cosine_sim = torch.mm(normalized_v, normalized_v.t())

        # Exclude diagonal
        n = velocities.size(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=velocities.device)
        off_diagonal = cosine_sim[mask]
        mean_cosine_sim = off_diagonal.mean().item()

        # Per-class velocity similarity
        intra_class_sims = []
        inter_class_sims = []

        for i in range(n):
            for j in range(i+1, n):
                sim = cosine_sim[i, j].item()
                if self.prototype_class[i] == self.prototype_class[j]:
                    intra_class_sims.append(sim)
                else:
                    inter_class_sims.append(sim)

        return {
            'mean_velocity_norm': mean_velocity_norm,
            'min_velocity_norm': min_velocity_norm,
            'mean_velocity_cosine_similarity': mean_cosine_sim,
            'intra_class_velocity_similarity': np.mean(intra_class_sims) if intra_class_sims else 0.0,
            'inter_class_velocity_similarity': np.mean(inter_class_sims) if inter_class_sims else 0.0,
        }


class ProgressivePrototypeTrajectories(BaseTimeSeriesModel):
    """
    Progressive Prototype Trajectories (PPT) for explainable MG detection.

    Key innovation: Prototypes are trajectory functions P_k(t) that capture
    expected temporal evolution patterns, enabling the model to distinguish
    between stable patterns (HC) and deteriorating patterns (MG).

    Architecture:
        Input (batch, seq_len, input_dim)
            |
            v
        Segment-wise Encoder -> (batch, n_segments, latent_dim)
            |
            v
        Trajectory Prototype Layer -> Trajectory similarities
            |
            v
        Classification -> Logits (batch, n_classes)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        latent_dim: int = 64,
        n_prototypes_per_class: int = 5,
        n_segments: int = 8,
        encoder_hidden: int = 64,
        encoder_layers: int = 3,
        kernel_size: int = 7,
        dropout: float = 0.2,
        trajectory_type: str = 'linear',
        **kwargs,
    ):
        """
        Args:
            input_dim: Number of input features per timestep
            num_classes: Number of output classes (2 for binary)
            seq_len: Expected sequence length
            latent_dim: Dimension of latent space for prototypes
            n_prototypes_per_class: Number of prototypes per class
            n_segments: Number of temporal segments to divide sequence into
            encoder_hidden: Hidden dimension in encoder
            encoder_layers: Number of encoder conv layers
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
            trajectory_type: 'linear' or 'polynomial'
        """
        super().__init__(input_dim, num_classes, seq_len)

        self.latent_dim = latent_dim
        self.n_prototypes_per_class = n_prototypes_per_class
        self.n_segments = n_segments
        self.encoder_hidden = encoder_hidden
        self.encoder_layers = encoder_layers
        self.kernel_size = kernel_size
        self.dropout_rate = dropout
        self.trajectory_type = trajectory_type

        # Segment length
        self.segment_len = seq_len // n_segments

        # Build segment encoder (same architecture as TempProtoNet encoder)
        self.encoder = self._build_encoder()

        # Trajectory prototype layer
        self.prototype_layer = TrajectoryPrototypeLayer(
            latent_dim=latent_dim,
            n_prototypes_per_class=n_prototypes_per_class,
            n_classes=num_classes,
            trajectory_type=trajectory_type,
        )

        # For storing training embeddings
        self._training_embeddings: Optional[torch.Tensor] = None
        self._training_labels: Optional[torch.Tensor] = None

    def _build_encoder(self) -> nn.Module:
        """Build the 1D CNN encoder for each segment."""
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

        encoder = nn.Sequential(*layers)

        # Projection head for each segment
        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, self.latent_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        return encoder

    def encode_segments(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input time series into segment-wise latent representations.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            z_segments: Segment encodings (batch_size, n_segments, latent_dim)
        """
        batch_size = x.size(0)
        segment_encodings = []

        for seg_idx in range(self.n_segments):
            start = seg_idx * self.segment_len
            end = start + self.segment_len

            # Handle last segment (may be longer if seq_len not divisible)
            if seg_idx == self.n_segments - 1:
                end = x.size(1)

            # Extract segment: (batch_size, segment_len, input_dim)
            segment = x[:, start:end, :]

            # Transpose for conv1d: (batch_size, input_dim, segment_len)
            segment = segment.transpose(1, 2)

            # Encode segment
            features = self.encoder(segment)
            z = self.projection_head(features)

            # Normalize
            z = F.normalize(z, p=2, dim=1)

            segment_encodings.append(z)

        # Stack: (batch_size, n_segments, latent_dim)
        z_segments = torch.stack(segment_encodings, dim=1)

        return z_segments

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            logits: Class logits (batch_size, num_classes)
        """
        z_segments = self.encode_segments(x)
        _, _, _, logits = self.prototype_layer(z_segments)
        return logits

    def forward_with_explanations(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with full explanation outputs.

        Returns:
            logits: Class logits
            z_segments: Segment encodings
            distances: Per-segment distances to prototypes
            per_segment_similarities: Per-segment similarities
            trajectory_similarities: Aggregated trajectory similarities
        """
        z_segments = self.encode_segments(x)
        distances, per_segment_sims, trajectory_sims, logits = self.prototype_layer(z_segments)
        return logits, z_segments, distances, per_segment_sims, trajectory_sims

    def compute_prototype_loss(
        self,
        z_segments: torch.Tensor,
        labels: torch.Tensor,
        distances: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute prototype-specific losses for training.

        Args:
            z_segments: Segment encodings (batch_size, n_segments, latent_dim)
            labels: Ground truth labels (batch_size,)
            distances: Per-segment distances (batch_size, n_segments, n_prototypes)

        Returns:
            cluster_loss: Trajectory clustering loss
            separation_loss: Trajectory separation loss
            diversity_loss: Velocity diversity loss (new for PPT)
        """
        batch_size = z_segments.size(0)
        prototype_classes = self.prototype_layer.prototype_class

        # 1. Cluster loss: trajectories should be close to same-class prototype trajectories
        cluster_losses = []
        for i in range(batch_size):
            label = labels[i].item()
            same_class_mask = (prototype_classes == label)

            # Mean distance across segments to same-class prototypes
            # distances[i]: (n_segments, n_prototypes)
            same_class_distances = distances[i, :, same_class_mask]  # (n_segments, n_same_class)

            # Min trajectory distance (to closest same-class prototype)
            min_traj_dist = same_class_distances.mean(dim=0).min()  # Mean over time, min over prototypes
            cluster_losses.append(min_traj_dist)

        cluster_loss = torch.stack(cluster_losses).mean()

        # 2. Separation loss: prototype trajectories of different classes should diverge
        # Compute mean separation between prototype origins of different classes
        origins = self.prototype_layer.prototype_origins
        sep_losses = []

        for c in range(self.num_classes):
            class_c_mask = (prototype_classes == c)
            other_class_mask = (prototype_classes != c)

            class_c_origins = origins[class_c_mask]
            other_origins = origins[other_class_mask]

            if len(class_c_origins) > 0 and len(other_origins) > 0:
                dists = torch.cdist(class_c_origins, other_origins)
                min_dist = dists.min()
                sep_losses.append(-min_dist)

        if sep_losses:
            separation_loss = torch.stack(sep_losses).mean()
        else:
            separation_loss = torch.tensor(0.0, device=z_segments.device)

        # 3. Diversity loss: encourage diverse velocity directions
        # Penalize high cosine similarity between velocity vectors
        velocities = self.prototype_layer.prototype_velocities
        normalized_v = F.normalize(velocities, dim=1)
        cosine_sim = torch.mm(normalized_v, normalized_v.t())

        # We want low similarity (diverse directions)
        n = velocities.size(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=velocities.device)
        diversity_loss = cosine_sim[mask].mean()  # Higher similarity = higher loss

        # Also penalize small velocity norms (prevent collapse to static)
        velocity_norms = torch.norm(velocities, dim=1)
        min_norm_penalty = F.relu(0.1 - velocity_norms).mean()  # Penalize norms < 0.1
        diversity_loss = diversity_loss + min_norm_penalty

        return cluster_loss, separation_loss, diversity_loss

    def get_prototype_similarities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get trajectory similarity scores to all prototypes.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            trajectory_similarities: (batch_size, n_prototypes)
        """
        z_segments = self.encode_segments(x)
        _, _, trajectory_sims, _ = self.prototype_layer(z_segments)
        return trajectory_sims

    def get_per_segment_similarities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get per-segment similarity scores for temporal analysis.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            per_segment_similarities: (batch_size, n_segments, n_prototypes)
        """
        z_segments = self.encode_segments(x)
        _, per_segment_sims, _, _ = self.prototype_layer(z_segments)
        return per_segment_sims

    def get_prototypes(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the learned prototype origins and velocities."""
        return (
            self.prototype_layer.prototype_origins.detach(),
            self.prototype_layer.prototype_velocities.detach(),
        )

    def get_prototype_classes(self) -> torch.Tensor:
        """Get the class assignment for each prototype."""
        return self.prototype_layer.prototype_class

    def explain_prediction(
        self,
        x: torch.Tensor,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Generate trajectory-based explanation for predictions.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            top_k: Number of most similar prototypes to include

        Returns:
            List of explanations, one per sample
        """
        with torch.no_grad():
            logits, z_segments, distances, per_seg_sims, traj_sims = self.forward_with_explanations(x)
            predictions = torch.argmax(logits, dim=1)
            probabilities = F.softmax(logits, dim=1)

            prototype_classes = self.prototype_layer.prototype_class
            origins, velocities = self.get_prototypes()

            explanations = []

            for i in range(x.size(0)):
                pred = predictions[i].item()
                prob = probabilities[i, pred].item()
                sample_traj_sims = traj_sims[i]
                sample_seg_sims = per_seg_sims[i]  # (n_segments, n_prototypes)

                # Get top-k most similar prototypes (by trajectory similarity)
                topk_sims, topk_indices = torch.topk(sample_traj_sims, top_k)

                similar_prototypes = []
                for j in range(top_k):
                    proto_idx = topk_indices[j].item()
                    proto_class = prototype_classes[proto_idx].item()
                    sim = topk_sims[j].item()

                    # Get per-segment similarity trend for this prototype
                    seg_sims = sample_seg_sims[:, proto_idx].cpu().numpy()

                    # Compute similarity trend (increasing/decreasing over time)
                    if len(seg_sims) > 1:
                        trend = np.polyfit(range(len(seg_sims)), seg_sims, 1)[0]
                    else:
                        trend = 0.0

                    similar_prototypes.append({
                        'prototype_idx': proto_idx,
                        'prototype_class': proto_class,
                        'prototype_class_name': 'MG' if proto_class == 1 else 'HC',
                        'trajectory_similarity': sim,
                        'per_segment_similarities': seg_sims.tolist(),
                        'similarity_trend': float(trend),
                        'velocity_norm': torch.norm(velocities[proto_idx]).item(),
                    })

                # Compute temporal discrimination: where in the sequence is most discriminative?
                # Compare MG vs HC prototype similarities per segment
                mg_mask = (prototype_classes == 1)
                hc_mask = (prototype_classes == 0)
                mg_seg_sims = sample_seg_sims[:, mg_mask].mean(dim=1).cpu().numpy()
                hc_seg_sims = sample_seg_sims[:, hc_mask].mean(dim=1).cpu().numpy()

                # Which segments favor the prediction most?
                if pred == 1:  # MG
                    segment_contributions = mg_seg_sims - hc_seg_sims
                else:  # HC
                    segment_contributions = hc_seg_sims - mg_seg_sims

                most_discriminative_segment = int(np.argmax(segment_contributions))

                explanation = {
                    'prediction': pred,
                    'prediction_name': 'MG' if pred == 1 else 'HC',
                    'confidence': prob,
                    'top_similar_prototypes': similar_prototypes,
                    'segment_contributions': segment_contributions.tolist(),
                    'most_discriminative_segment': most_discriminative_segment,
                    'most_discriminative_time': (most_discriminative_segment + 0.5) / self.n_segments,
                    'avg_mg_similarity': float(traj_sims[i][mg_mask].mean().item()),
                    'avg_hc_similarity': float(traj_sims[i][hc_mask].mean().item()),
                }

                explanations.append(explanation)

            return explanations

    def compute_prototype_diversity(self) -> Dict[str, float]:
        """
        Compute metrics about prototype and trajectory diversity.
        """
        # Origin diversity (same as TempProtoNet)
        origins = self.prototype_layer.prototype_origins.detach()
        pairwise_dists = torch.cdist(origins, origins)
        n = origins.size(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=origins.device)
        off_diagonal = pairwise_dists[mask]

        origin_diversity = {
            'mean_origin_distance': off_diagonal.mean().item(),
            'min_origin_distance': off_diagonal.min().item(),
        }

        # Trajectory diversity
        trajectory_diversity = self.prototype_layer.compute_trajectory_diversity()

        # Combine
        return {**origin_diversity, **trajectory_diversity}

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': 'ProgressivePrototypeTrajectories',
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'seq_len': self.seq_len,
            'latent_dim': self.latent_dim,
            'n_prototypes_per_class': self.n_prototypes_per_class,
            'n_segments': self.n_segments,
            'encoder_hidden': self.encoder_hidden,
            'encoder_layers': self.encoder_layers,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout_rate,
            'trajectory_type': self.trajectory_type,
            'n_total_prototypes': self.prototype_layer.n_prototypes,
        }


# Default configuration
DEFAULT_CONFIG = {
    'latent_dim': 64,
    'n_prototypes_per_class': 5,
    'n_segments': 8,
    'encoder_hidden': 64,
    'encoder_layers': 3,
    'kernel_size': 7,
    'dropout': 0.2,
    'trajectory_type': 'linear',
}
