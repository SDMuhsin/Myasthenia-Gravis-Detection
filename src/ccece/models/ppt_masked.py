"""
CCECE Paper: Progressive Prototype Trajectories (PPT) - Masked Version

This version fixes the padding artifact using a MASKING approach:
1. Segments are computed using fixed absolute boundaries (efficient batch processing)
2. A mask indicates what fraction of each segment contains real data
3. Prototype similarities are weighted by the mask

Key advantages over relative segmentation:
- Preserves batch processing (BatchNorm works correctly)
- Efficient GPU utilization
- Correctly ignores padded regions in similarity computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import numpy as np

from .base import BaseTimeSeriesModel


class TrajectoryPrototypeLayerMasked(nn.Module):
    """
    Trajectory prototype layer with segment-level masking.

    Uses weighted similarity computation where padding contributes less.
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
        # t: (n_segments,) -> output: (n_segments, n_prototypes, latent_dim)
        t = t.view(-1, 1, 1)  # (n_segments, 1, 1)

        origins = self.prototype_origins.unsqueeze(0)  # (1, n_proto, latent_dim)
        velocities = self.prototype_velocities.unsqueeze(0)  # (1, n_proto, latent_dim)

        prototypes = origins + t * velocities

        if self.trajectory_type == 'polynomial':
            accelerations = self.prototype_accelerations.unsqueeze(0)
            prototypes = prototypes + (t ** 2) * accelerations

        return prototypes  # (n_segments, n_prototypes, latent_dim)

    def forward(
        self,
        z_segments: torch.Tensor,
        segment_weights: torch.Tensor,
        normalized_times: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute trajectory similarity with weighted segments.

        Args:
            z_segments: (batch_size, n_segments, latent_dim) - segment encodings
            segment_weights: (batch_size, n_segments) - weight for each segment (% real data)
            normalized_times: (batch_size, n_segments) - normalized time for each segment

        Returns:
            distances, per_segment_similarities, trajectory_similarities, logits
        """
        batch_size, n_segments, latent_dim = z_segments.shape
        device = z_segments.device

        # Default normalized times (mid-point of each segment)
        if normalized_times is None:
            # For masked approach, use RELATIVE times based on actual lengths
            # segment_weights indicate real data fraction, we adjust times accordingly
            default_t = torch.linspace(0, 1, n_segments + 1, device=device)
            t_values = (default_t[:-1] + default_t[1:]) / 2  # midpoints
            t_values = t_values.unsqueeze(0).expand(batch_size, -1)
        else:
            t_values = normalized_times

        # Get prototypes at mean time for batch (simplification for efficiency)
        # Use per-segment mean time across batch
        mean_t = t_values.mean(dim=0)  # (n_segments,)
        prototypes = self.get_prototype_at_time(mean_t)  # (n_segments, n_proto, latent_dim)

        # Compute distances: (batch, n_segments, n_prototypes)
        z_expanded = z_segments.unsqueeze(2)  # (batch, n_seg, 1, latent_dim)
        proto_expanded = prototypes.unsqueeze(0)  # (1, n_seg, n_proto, latent_dim)
        distances = torch.sum((z_expanded - proto_expanded) ** 2, dim=-1)

        # Convert to similarities
        similarities = torch.log(1 + 1 / (distances + 1e-6))

        # Weighted average across segments
        # segment_weights: (batch, n_segments) -> normalize to sum to 1
        weights_sum = segment_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        normalized_weights = segment_weights / weights_sum  # (batch, n_segments)

        # Weighted trajectory similarity
        weighted_sims = similarities * normalized_weights.unsqueeze(-1)
        trajectory_similarities = weighted_sims.sum(dim=1)  # (batch, n_prototypes)

        # Compute logits
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


class PPTMasked(BaseTimeSeriesModel):
    """
    Progressive Prototype Trajectories - Masked Version.

    Key approach:
    1. Use fixed absolute segment boundaries (efficient batch processing)
    2. Compute segment weights based on fraction of real data in each segment
    3. Weight prototype similarities by segment weights

    This preserves:
    - Batch processing (BatchNorm works)
    - GPU efficiency
    - Correct handling of padding (low weight for padded segments)
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

        # Compute segment boundaries (fixed, absolute)
        segment_size = seq_len // n_segments
        self.segment_boundaries = []
        for i in range(n_segments):
            start = i * segment_size
            end = (i + 1) * segment_size if i < n_segments - 1 else seq_len
            self.segment_boundaries.append((start, end))

        # Build encoder
        self.encoder = self._build_encoder()

        # Trajectory prototype layer
        self.prototype_layer = TrajectoryPrototypeLayerMasked(
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

    def compute_segment_weights(
        self,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute weights for each segment based on fraction of real data.

        Also computes normalized time for each segment based on actual length.

        Args:
            lengths: (batch_size,) - actual sequence lengths

        Returns:
            segment_weights: (batch_size, n_segments) - fraction of real data in each segment
            normalized_times: (batch_size, n_segments) - normalized time (0-1) for each segment
        """
        batch_size = lengths.size(0)
        device = lengths.device

        weights = torch.zeros(batch_size, self.n_segments, device=device)
        times = torch.zeros(batch_size, self.n_segments, device=device)

        for seg_idx, (start, end) in enumerate(self.segment_boundaries):
            seg_len = end - start

            # Compute overlap with real data for each sample
            real_end = lengths.float()  # (batch,)

            # How much of this segment is real data?
            overlap_start = torch.clamp(torch.tensor(start, device=device).float(), max=real_end)
            overlap_end = torch.clamp(torch.tensor(end, device=device).float(), max=real_end)
            overlap = torch.clamp(overlap_end - overlap_start, min=0)

            # Weight = fraction of segment that is real data
            weights[:, seg_idx] = overlap / seg_len

            # Normalized time = midpoint of real portion relative to actual length
            # If segment is fully padded, use the position it would have
            seg_mid = (start + end) / 2
            times[:, seg_idx] = torch.clamp(seg_mid / real_end.clamp(min=1), max=1.0)

        return weights, times

    def encode_segments(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode each segment using fixed boundaries.

        Batch processing is preserved - all segments for all samples are processed together.

        Args:
            x: (batch_size, seq_len, input_dim)

        Returns:
            z_segments: (batch_size, n_segments, latent_dim)
        """
        batch_size = x.size(0)
        device = x.device

        segment_encodings = []

        for start, end in self.segment_boundaries:
            segment = x[:, start:end, :]  # (batch, seg_len, input_dim)
            segment = segment.transpose(1, 2)  # (batch, input_dim, seg_len)

            features = self.encoder(segment)  # (batch, encoder_out_dim, reduced_len)
            z = self.projection_head(features)  # (batch, latent_dim)
            z = F.normalize(z, p=2, dim=1)

            segment_encodings.append(z)

        z_segments = torch.stack(segment_encodings, dim=1)  # (batch, n_segments, latent_dim)
        return z_segments

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
            lengths = torch.full((x.size(0),), x.size(1), device=x.device, dtype=torch.long)

        z_segments = self.encode_segments(x)
        segment_weights, normalized_times = self.compute_segment_weights(lengths)
        _, _, _, logits = self.prototype_layer(z_segments, segment_weights, normalized_times)
        return logits

    def forward_with_explanations(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward with full outputs."""
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), device=x.device, dtype=torch.long)

        z_segments = self.encode_segments(x)
        segment_weights, normalized_times = self.compute_segment_weights(lengths)
        distances, per_seg_sims, traj_sims, logits = self.prototype_layer(
            z_segments, segment_weights, normalized_times
        )
        return logits, z_segments, distances, per_seg_sims, traj_sims, segment_weights

    def compute_prototype_loss(
        self,
        z_segments: torch.Tensor,
        labels: torch.Tensor,
        distances: torch.Tensor,
        segment_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute prototype losses with segment weighting."""
        batch_size = z_segments.size(0)
        prototype_classes = self.prototype_layer.prototype_class
        device = z_segments.device

        # Normalize weights
        weights_sum = segment_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        norm_weights = segment_weights / weights_sum  # (batch, n_segments)

        # Cluster loss
        cluster_losses = []
        for i in range(batch_size):
            label = labels[i].item()
            same_class_mask = (prototype_classes == label)
            same_class_distances = distances[i, :, same_class_mask]  # (n_seg, n_same_class)

            # Weighted average over segments
            w = norm_weights[i].unsqueeze(-1)  # (n_seg, 1)
            weighted_dists = (same_class_distances * w).sum(dim=0)  # (n_same_class,)

            min_traj_dist = weighted_dists.min()
            cluster_losses.append(min_traj_dist)

        cluster_loss = torch.stack(cluster_losses).mean()

        # Separation loss (based on origins)
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

        separation_loss = torch.stack(sep_losses).mean() if sep_losses else torch.tensor(0.0, device=device)

        # Diversity loss
        velocities = self.prototype_layer.prototype_velocities
        normalized_v = F.normalize(velocities, dim=1)
        cosine_sim = torch.mm(normalized_v, normalized_v.t())
        n = velocities.size(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=device)
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
            lengths = torch.full((x.size(0),), x.size(1), device=x.device, dtype=torch.long)
        z_segments = self.encode_segments(x)
        segment_weights, normalized_times = self.compute_segment_weights(lengths)
        _, _, traj_sims, _ = self.prototype_layer(z_segments, segment_weights, normalized_times)
        return traj_sims

    def get_per_segment_similarities(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), device=x.device, dtype=torch.long)
        z_segments = self.encode_segments(x)
        segment_weights, normalized_times = self.compute_segment_weights(lengths)
        _, per_seg_sims, _, _ = self.prototype_layer(z_segments, segment_weights, normalized_times)
        return per_seg_sims, segment_weights

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
            'model_type': 'PPTMasked',
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'seq_len': self.seq_len,
            'latent_dim': self.latent_dim,
            'n_prototypes_per_class': self.n_prototypes_per_class,
            'n_segments': self.n_segments,
            'trajectory_type': self.trajectory_type,
        }
