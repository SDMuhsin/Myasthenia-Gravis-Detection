"""
CCECE Paper: Progressive Prototype Trajectories (PPT) - Fixed Version

This version fixes the padding artifact by using RELATIVE temporal segmentation
based on actual sequence length, not padded length.

Key Fix:
- Old: Segments based on padded length (late segments contain mostly zeros)
- New: Segments based on actual data length (late segments contain real late data)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from .base import BaseTimeSeriesModel


class TrajectoryPrototypeLayerFixed(nn.Module):
    """
    Trajectory prototype layer with support for variable-length sequences.
    """

    def __init__(
        self,
        latent_dim: int,
        n_prototypes_per_class: int,
        n_classes: int = 2,
        trajectory_type: str = 'linear',
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_prototypes_per_class = n_prototypes_per_class
        self.n_classes = n_classes
        self.n_prototypes = n_prototypes_per_class * n_classes
        self.trajectory_type = trajectory_type

        # Initial prototype positions
        self.prototype_origins = nn.Parameter(
            torch.randn(self.n_prototypes, latent_dim) * 0.1
        )

        # Trajectory velocities
        self.prototype_velocities = nn.Parameter(
            torch.randn(self.n_prototypes, latent_dim) * 0.05
        )

        if trajectory_type == 'polynomial':
            self.prototype_accelerations = nn.Parameter(
                torch.randn(self.n_prototypes, latent_dim) * 0.02
            )

        # Prototype-to-class mapping
        prototype_class = torch.zeros(self.n_prototypes, dtype=torch.long)
        for c in range(n_classes):
            start_idx = c * n_prototypes_per_class
            end_idx = start_idx + n_prototypes_per_class
            prototype_class[start_idx:end_idx] = c
        self.register_buffer('prototype_class', prototype_class)

        class_identity = torch.zeros(self.n_prototypes, n_classes)
        for i in range(self.n_prototypes):
            class_identity[i, prototype_class[i]] = 1.0
        self.register_buffer('class_identity', class_identity)

    def get_prototype_at_time(self, t: torch.Tensor) -> torch.Tensor:
        """Get prototype positions at normalized time t."""
        if t.dim() == 1:
            t = t.view(-1, 1, 1)
            squeeze_batch = True
        else:
            t = t.unsqueeze(-1).unsqueeze(-1)
            squeeze_batch = False

        origins = self.prototype_origins.unsqueeze(0).unsqueeze(0)
        velocities = self.prototype_velocities.unsqueeze(0).unsqueeze(0)

        prototypes = origins + t * velocities

        if self.trajectory_type == 'polynomial':
            accelerations = self.prototype_accelerations.unsqueeze(0).unsqueeze(0)
            prototypes = prototypes + (t ** 2) * accelerations

        if squeeze_batch:
            prototypes = prototypes.squeeze(0)

        return prototypes

    def forward(
        self,
        z_segments: torch.Tensor,
        segment_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute trajectory similarity with optional masking for padded segments.

        Args:
            z_segments: (batch_size, n_segments, latent_dim)
            segment_mask: (batch_size, n_segments) - 1 for real, 0 for padded

        Returns:
            distances, similarities, trajectory_similarities, logits
        """
        batch_size, n_segments, latent_dim = z_segments.shape

        # Normalized time values
        if n_segments > 1:
            t_values = torch.linspace(0, 1, n_segments, device=z_segments.device)
        else:
            t_values = torch.tensor([0.5], device=z_segments.device)

        # Get prototypes at each time
        prototypes = self.get_prototype_at_time(t_values)

        # Compute distances
        z_expanded = z_segments.unsqueeze(2)
        proto_expanded = prototypes.unsqueeze(0)
        distances = torch.sum((z_expanded - proto_expanded) ** 2, dim=-1)

        # Convert to similarities
        similarities = torch.log(1 + 1 / (distances + 1e-6))

        # Apply mask if provided (mask out padded segments)
        if segment_mask is not None:
            mask_expanded = segment_mask.unsqueeze(-1)  # (batch, n_seg, 1)
            similarities = similarities * mask_expanded
            # Compute mean over REAL segments only
            real_counts = segment_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (batch, 1)
            trajectory_similarities = similarities.sum(dim=1) / real_counts
        else:
            trajectory_similarities = similarities.mean(dim=1)

        logits = torch.matmul(trajectory_similarities, self.class_identity)

        return distances, similarities, trajectory_similarities, logits

    def get_prototype_classes(self) -> torch.Tensor:
        return self.prototype_class

    def compute_velocity_norms(self) -> torch.Tensor:
        return torch.norm(self.prototype_velocities, dim=1)

    def compute_trajectory_diversity(self) -> Dict[str, float]:
        velocities = self.prototype_velocities.detach()
        velocity_norms = torch.norm(velocities, dim=1)

        normalized_v = F.normalize(velocities, dim=1)
        cosine_sim = torch.mm(normalized_v, normalized_v.t())

        n = velocities.size(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=velocities.device)
        off_diagonal = cosine_sim[mask]

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
            'mean_velocity_norm': velocity_norms.mean().item(),
            'min_velocity_norm': velocity_norms.min().item(),
            'mean_velocity_cosine_similarity': off_diagonal.mean().item(),
            'intra_class_velocity_similarity': np.mean(intra_class_sims) if intra_class_sims else 0.0,
            'inter_class_velocity_similarity': np.mean(inter_class_sims) if inter_class_sims else 0.0,
        }


class PPTFixed(BaseTimeSeriesModel):
    """
    Progressive Prototype Trajectories - Fixed for padding artifact.

    Key fix: Uses RELATIVE temporal segmentation based on actual sequence length.
    Each segment represents a fixed percentage of the REAL data, not padded data.
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
        super().__init__(input_dim, num_classes, seq_len)

        self.latent_dim = latent_dim
        self.n_prototypes_per_class = n_prototypes_per_class
        self.n_segments = n_segments
        self.encoder_hidden = encoder_hidden
        self.encoder_layers = encoder_layers
        self.kernel_size = kernel_size
        self.dropout_rate = dropout
        self.trajectory_type = trajectory_type

        # Build encoder
        self.encoder = self._build_encoder()

        # Trajectory prototype layer
        self.prototype_layer = TrajectoryPrototypeLayerFixed(
            latent_dim=latent_dim,
            n_prototypes_per_class=n_prototypes_per_class,
            n_classes=num_classes,
            trajectory_type=trajectory_type,
        )

    def _build_encoder(self) -> nn.Module:
        """Build the 1D CNN encoder."""
        layers = []
        in_channels = self.input_dim

        for i in range(self.encoder_layers):
            out_channels = self.encoder_hidden * (2 ** min(i, 2))
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size,
                         padding=self.kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(self.dropout_rate),
            ])
            in_channels = out_channels

        self.encoder_output_dim = in_channels

        encoder = nn.Sequential(*layers)

        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, self.latent_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        return encoder

    def encode_segments_relative(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode using RELATIVE temporal segmentation.

        Each segment represents a fixed percentage of the ACTUAL sequence length,
        not the padded length.

        Args:
            x: (batch_size, seq_len, input_dim)
            lengths: (batch_size,) - actual lengths before padding

        Returns:
            z_segments: (batch_size, n_segments, latent_dim)
            segment_mask: (batch_size, n_segments) - 1 for real, 0 for padded
        """
        batch_size = x.size(0)
        device = x.device

        segment_encodings = []
        segment_masks = []

        for b in range(batch_size):
            actual_len = lengths[b].item()
            sample_encodings = []
            sample_mask = []

            for seg_idx in range(self.n_segments):
                # RELATIVE segment boundaries based on actual length
                rel_start = seg_idx / self.n_segments
                rel_end = (seg_idx + 1) / self.n_segments

                start = int(rel_start * actual_len)
                end = int(rel_end * actual_len)

                # Ensure at least 1 timestep per segment
                if end <= start:
                    end = start + 1
                if end > actual_len:
                    end = int(actual_len)
                    start = max(0, end - 1)

                # Check if this segment has real data
                if start < actual_len and end > start:
                    segment = x[b:b+1, start:end, :]  # (1, seg_len, input_dim)
                    segment = segment.transpose(1, 2)  # (1, input_dim, seg_len)
                    features = self.encoder(segment)
                    z = self.projection_head(features)
                    z = F.normalize(z, p=2, dim=1)
                    sample_encodings.append(z.squeeze(0))
                    sample_mask.append(1.0)
                else:
                    # This shouldn't happen with relative segmentation
                    sample_encodings.append(torch.zeros(self.latent_dim, device=device))
                    sample_mask.append(0.0)

            segment_encodings.append(torch.stack(sample_encodings))
            segment_masks.append(torch.tensor(sample_mask, device=device))

        z_segments = torch.stack(segment_encodings)  # (batch, n_segments, latent_dim)
        segment_mask = torch.stack(segment_masks)  # (batch, n_segments)

        return z_segments, segment_mask

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch_size, seq_len, input_dim)
            lengths: (batch_size,) - actual lengths. If None, assumes no padding.
        """
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), device=x.device)

        z_segments, segment_mask = self.encode_segments_relative(x, lengths)
        _, _, _, logits = self.prototype_layer(z_segments, segment_mask)
        return logits

    def forward_with_explanations(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward with full outputs including segment mask."""
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), device=x.device)

        z_segments, segment_mask = self.encode_segments_relative(x, lengths)
        distances, per_seg_sims, traj_sims, logits = self.prototype_layer(z_segments, segment_mask)
        return logits, z_segments, distances, per_seg_sims, traj_sims, segment_mask

    def compute_prototype_loss(
        self,
        z_segments: torch.Tensor,
        labels: torch.Tensor,
        distances: torch.Tensor,
        segment_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute losses with optional masking."""
        batch_size = z_segments.size(0)
        prototype_classes = self.prototype_layer.prototype_class

        # Cluster loss
        cluster_losses = []
        for i in range(batch_size):
            label = labels[i].item()
            same_class_mask = (prototype_classes == label)
            same_class_distances = distances[i, :, same_class_mask]

            if segment_mask is not None:
                mask = segment_mask[i].unsqueeze(-1)
                masked_dists = same_class_distances * mask
                real_count = segment_mask[i].sum().clamp(min=1)
                mean_dists = masked_dists.sum(dim=0) / real_count
            else:
                mean_dists = same_class_distances.mean(dim=0)

            min_traj_dist = mean_dists.min()
            cluster_losses.append(min_traj_dist)

        cluster_loss = torch.stack(cluster_losses).mean()

        # Separation loss
        origins = self.prototype_layer.prototype_origins
        sep_losses = []
        for c in range(self.num_classes):
            class_c_mask = (prototype_classes == c)
            other_class_mask = (prototype_classes != c)
            class_c_origins = origins[class_c_mask]
            other_origins = origins[other_class_mask]
            if len(class_c_origins) > 0 and len(other_origins) > 0:
                dists = torch.cdist(class_c_origins, other_origins)
                sep_losses.append(-dists.min())

        separation_loss = torch.stack(sep_losses).mean() if sep_losses else torch.tensor(0.0, device=z_segments.device)

        # Diversity loss
        velocities = self.prototype_layer.prototype_velocities
        normalized_v = F.normalize(velocities, dim=1)
        cosine_sim = torch.mm(normalized_v, normalized_v.t())
        n = velocities.size(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=velocities.device)
        diversity_loss = cosine_sim[mask].mean()

        velocity_norms = torch.norm(velocities, dim=1)
        min_norm_penalty = F.relu(0.1 - velocity_norms).mean()
        diversity_loss = diversity_loss + min_norm_penalty

        return cluster_loss, separation_loss, diversity_loss

    def get_prototype_similarities(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), device=x.device)
        z_segments, segment_mask = self.encode_segments_relative(x, lengths)
        _, _, traj_sims, _ = self.prototype_layer(z_segments, segment_mask)
        return traj_sims

    def get_per_segment_similarities(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), device=x.device)
        z_segments, segment_mask = self.encode_segments_relative(x, lengths)
        _, per_seg_sims, _, _ = self.prototype_layer(z_segments, segment_mask)
        return per_seg_sims, segment_mask

    def get_prototype_classes(self) -> torch.Tensor:
        return self.prototype_layer.prototype_class

    def compute_prototype_diversity(self) -> Dict[str, float]:
        origins = self.prototype_layer.prototype_origins.detach()
        pairwise_dists = torch.cdist(origins, origins)
        n = origins.size(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=origins.device)
        off_diagonal = pairwise_dists[mask]

        origin_diversity = {
            'mean_origin_distance': off_diagonal.mean().item(),
            'min_origin_distance': off_diagonal.min().item(),
        }

        trajectory_diversity = self.prototype_layer.compute_trajectory_diversity()
        return {**origin_diversity, **trajectory_diversity}

    def get_config(self) -> Dict[str, Any]:
        return {
            'model_type': 'PPTFixed',
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'seq_len': self.seq_len,
            'latent_dim': self.latent_dim,
            'n_prototypes_per_class': self.n_prototypes_per_class,
            'n_segments': self.n_segments,
            'trajectory_type': self.trajectory_type,
        }
