"""
Multi-Head Trajectory Prototype Network (MHTPN)

A hybrid architecture combining:
1. MultiHeadProtoNet's structural constraint (2 prototypes per head, collapse impossible)
2. PPTMasked's temporal dynamics (trajectory prototypes with origin + velocity)
3. Segment-wise encoding with padding-aware weighting

Design principles:
- K independent heads, each with exactly 2 trajectory prototypes (1 per class)
- Each prototype traces a trajectory through time: p(t) = origin + t * velocity
- Segments are weighted by fraction of real data (handles padding)
- Average logits across heads for final prediction

Expected behavior:
- Structural constraint prevents prototype collapse (like MultiHeadProtoNet)
- Trajectory velocities capture temporal evolution (like PPT)
- Late segments should show more discriminative power for MG (clinical hypothesis)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from .base import BaseTimeSeriesModel


class TrajectoryPrototypeHead(nn.Module):
    """
    Single trajectory prototype head with exactly 2 prototypes (1 per class).

    Each prototype is a trajectory: p(t) = origin + t * velocity

    Structural constraint from MultiHeadProtoNet:
    - Only 2 prototypes, both MUST be used for classification
    - Collapse is impossible

    Temporal dynamics from PPT:
    - Prototypes evolve over time via velocities
    - Different prototypes can have different temporal patterns
    """

    def __init__(
        self,
        latent_dim: int,
        head_dim: int,
        n_classes: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.head_dim = head_dim
        self.n_classes = n_classes

        # Projection to this head's space
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, head_dim),
        )

        # Exactly 2 trajectory prototypes: one for HC (class 0), one for MG (class 1)
        # Each has origin and velocity
        self.prototype_origins = nn.Parameter(
            torch.randn(n_classes, head_dim) * 0.1
        )
        self.prototype_velocities = nn.Parameter(
            torch.randn(n_classes, head_dim) * 0.05
        )

        # Fixed class assignments
        prototype_class = torch.arange(n_classes, dtype=torch.long)
        self.register_buffer('prototype_class', prototype_class)

    def get_prototype_at_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get prototype positions at normalized time t.

        Args:
            t: (n_segments,) normalized time values

        Returns:
            prototypes: (n_segments, n_classes, head_dim)
        """
        t = t.view(-1, 1, 1)  # (n_segments, 1, 1)
        origins = self.prototype_origins.unsqueeze(0)  # (1, n_classes, head_dim)
        velocities = self.prototype_velocities.unsqueeze(0)  # (1, n_classes, head_dim)

        prototypes = origins + t * velocities
        return prototypes

    def forward(
        self,
        z_segments: torch.Tensor,
        segment_weights: torch.Tensor,
        t_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute trajectory similarity with weighted segments.

        Args:
            z_segments: (batch_size, n_segments, latent_dim) - segment encodings
            segment_weights: (batch_size, n_segments) - weight for each segment
            t_values: (n_segments,) - normalized time for each segment

        Returns:
            per_segment_distances: (batch, n_segments, n_classes)
            per_segment_similarities: (batch, n_segments, n_classes)
            trajectory_similarities: (batch, n_classes) - weighted average
            logits: (batch, n_classes)
        """
        batch_size, n_segments, latent_dim = z_segments.shape

        # Project segments to head space
        z_flat = z_segments.view(-1, latent_dim)
        h_flat = self.projection(z_flat)
        h_flat = F.normalize(h_flat, p=2, dim=1)
        h = h_flat.view(batch_size, n_segments, self.head_dim)

        # Get prototypes at each time point
        prototypes = self.get_prototype_at_time(t_values)  # (n_segments, n_classes, head_dim)

        # Compute distances: (batch, n_segments, n_classes)
        h_expanded = h.unsqueeze(2)  # (batch, n_seg, 1, head_dim)
        proto_expanded = prototypes.unsqueeze(0)  # (1, n_seg, n_classes, head_dim)
        distances = torch.sum((h_expanded - proto_expanded) ** 2, dim=-1)

        # Convert to similarities
        similarities = torch.log(1 + 1 / (distances + 1e-6))

        # Weighted average across segments
        weights_sum = segment_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        normalized_weights = segment_weights / weights_sum  # (batch, n_segments)

        weighted_sims = similarities * normalized_weights.unsqueeze(-1)
        trajectory_similarities = weighted_sims.sum(dim=1)  # (batch, n_classes)

        # Logits directly from similarities
        logits = trajectory_similarities

        return distances, similarities, trajectory_similarities, logits

    def get_velocity_norms(self) -> torch.Tensor:
        """Get L2 norm of prototype velocities."""
        return torch.norm(self.prototype_velocities, dim=1)


class MultiHeadTrajectoryProtoNet(BaseTimeSeriesModel):
    """
    Multi-Head Trajectory Prototype Network for explainable MG detection.

    Architecture:
        Input (batch, seq_len, input_dim)
            |
            v
        Segment-wise CNN Encoding -> (batch, n_segments, latent_dim)
            |
            v
        K Independent Trajectory Heads, each with 2 trajectory prototypes
            |
            v
        Average logits across heads -> Final prediction

    Why this combines the best of both:
    1. Structural constraint (from MultiHeadProtoNet):
       - 2 prototypes per head = collapse impossible
       - Alignment ≈ per-head accuracy

    2. Temporal dynamics (from PPT):
       - Segment-wise processing preserves temporal information
       - Trajectory prototypes (origin + velocity) model evolution
       - Segment weights handle padding correctly
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        latent_dim: int = 64,
        n_heads: int = 5,
        head_dim: int = 32,
        n_segments: int = 8,
        encoder_hidden: int = 64,
        encoder_layers: int = 3,
        kernel_size: int = 7,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__(input_dim, num_classes, seq_len)

        self.latent_dim = latent_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_segments = n_segments
        self.encoder_hidden = encoder_hidden
        self.encoder_layers = encoder_layers
        self.kernel_size = kernel_size
        self.dropout_rate = dropout

        # Compute segment boundaries (fixed, absolute)
        segment_size = seq_len // n_segments
        self.segment_boundaries = []
        for i in range(n_segments):
            start = i * segment_size
            end = (i + 1) * segment_size if i < n_segments - 1 else seq_len
            self.segment_boundaries.append((start, end))

        # Build shared CNN encoder
        self.encoder = self._build_encoder()

        # Create K independent trajectory prototype heads
        self.heads = nn.ModuleList([
            TrajectoryPrototypeHead(
                latent_dim=latent_dim,
                head_dim=head_dim,
                n_classes=num_classes,
            )
            for _ in range(n_heads)
        ])

        # Default normalized times (mid-point of each segment)
        t_default = torch.linspace(0, 1, n_segments + 1)
        t_midpoints = (t_default[:-1] + t_default[1:]) / 2
        self.register_buffer('t_default', t_midpoints)

    def _build_encoder(self) -> nn.Module:
        """Build the shared 1D CNN encoder for segments."""
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

        # Projection head: global pooling + linear to latent_dim
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
    ) -> torch.Tensor:
        """
        Compute weights for each segment based on fraction of real data.

        Args:
            lengths: (batch_size,) - actual sequence lengths

        Returns:
            segment_weights: (batch_size, n_segments) - fraction of real data
        """
        batch_size = lengths.size(0)
        device = lengths.device

        weights = torch.zeros(batch_size, self.n_segments, device=device)

        for seg_idx, (start, end) in enumerate(self.segment_boundaries):
            seg_len = end - start

            # How much of this segment is real data?
            real_end = lengths.float()
            overlap_start = torch.clamp(torch.tensor(start, device=device).float(), max=real_end)
            overlap_end = torch.clamp(torch.tensor(end, device=device).float(), max=real_end)
            overlap = torch.clamp(overlap_end - overlap_start, min=0)

            weights[:, seg_idx] = overlap / seg_len

        return weights

    def encode_segments(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode each segment using fixed boundaries.

        Args:
            x: (batch_size, seq_len, input_dim)

        Returns:
            z_segments: (batch_size, n_segments, latent_dim)
        """
        batch_size = x.size(0)
        segment_encodings = []

        for start, end in self.segment_boundaries:
            segment = x[:, start:end, :]  # (batch, seg_len, input_dim)
            segment = segment.transpose(1, 2)  # (batch, input_dim, seg_len)

            features = self.encoder(segment)
            z = self.projection_head(features)
            z = F.normalize(z, p=2, dim=1)

            segment_encodings.append(z)

        z_segments = torch.stack(segment_encodings, dim=1)
        return z_segments

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: average of all head logits.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            lengths: Optional actual sequence lengths (batch_size,)

        Returns:
            logits: Class logits (batch_size, num_classes)
        """
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), device=x.device, dtype=torch.long)

        z_segments = self.encode_segments(x)
        segment_weights = self.compute_segment_weights(lengths)

        # Collect logits from all heads
        all_logits = []
        for head in self.heads:
            _, _, _, head_logits = head(z_segments, segment_weights, self.t_default)
            all_logits.append(head_logits)

        # Average logits across heads
        logits = torch.stack(all_logits, dim=0).mean(dim=0)

        return logits

    def forward_with_explanations(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass with full explanation outputs.

        Returns:
            logits: (batch, n_classes)
            z_segments: (batch, n_segments, latent_dim)
            segment_weights: (batch, n_segments)
            all_per_seg_sims: List of per-segment similarities per head
            all_traj_sims: List of trajectory similarities per head
        """
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), device=x.device, dtype=torch.long)

        z_segments = self.encode_segments(x)
        segment_weights = self.compute_segment_weights(lengths)

        all_logits = []
        all_per_seg_sims = []
        all_traj_sims = []

        for head in self.heads:
            _, per_seg_sims, traj_sims, head_logits = head(
                z_segments, segment_weights, self.t_default
            )
            all_logits.append(head_logits)
            all_per_seg_sims.append(per_seg_sims)
            all_traj_sims.append(traj_sims)

        logits = torch.stack(all_logits, dim=0).mean(dim=0)

        return logits, z_segments, segment_weights, all_per_seg_sims, all_traj_sims

    def compute_prototype_loss(
        self,
        z_segments: torch.Tensor,
        labels: torch.Tensor,
        segment_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute prototype-specific losses for all heads.

        Returns:
            cluster_loss: Samples should be close to same-class trajectory
            separation_loss: Different-class prototypes should be far apart
            diversity_loss: Velocities should be diverse across heads
        """
        batch_size = z_segments.size(0)
        device = z_segments.device

        total_cluster_loss = torch.tensor(0.0, device=device)
        total_separation_loss = torch.tensor(0.0, device=device)

        # Normalize weights
        weights_sum = segment_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        norm_weights = segment_weights / weights_sum

        for head in self.heads:
            distances, _, _, _ = head(z_segments, segment_weights, self.t_default)

            # Cluster loss: weighted distance to same-class trajectory
            cluster_losses = []
            for i in range(batch_size):
                label = labels[i].item()
                same_class_dist = distances[i, :, label]  # (n_segments,)
                weighted_dist = (same_class_dist * norm_weights[i]).sum()
                cluster_losses.append(weighted_dist)

            cluster_loss = torch.stack(cluster_losses).mean()
            total_cluster_loss = total_cluster_loss + cluster_loss

            # Separation loss: prototype origins should be far apart
            origin_0 = head.prototype_origins[0]
            origin_1 = head.prototype_origins[1]
            inter_proto_dist = torch.sum((origin_0 - origin_1) ** 2)
            separation_loss = -inter_proto_dist
            total_separation_loss = total_separation_loss + separation_loss

        total_cluster_loss = total_cluster_loss / self.n_heads
        total_separation_loss = total_separation_loss / self.n_heads

        # Diversity loss: encourage diverse velocities across heads
        all_velocities = torch.stack([
            head.prototype_velocities for head in self.heads
        ], dim=0)  # (n_heads, n_classes, head_dim)

        # Flatten to (n_heads * n_classes, head_dim)
        velocities_flat = all_velocities.view(-1, self.head_dim)
        normalized_v = F.normalize(velocities_flat, dim=1)
        cosine_sim = torch.mm(normalized_v, normalized_v.t())

        n = velocities_flat.size(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=device)
        diversity_loss = cosine_sim[mask].mean()

        # Also encourage non-zero velocities
        velocity_norms = torch.norm(velocities_flat, dim=1)
        min_norm_penalty = F.relu(0.1 - velocity_norms).mean()
        diversity_loss = diversity_loss + min_norm_penalty

        return total_cluster_loss, total_separation_loss, diversity_loss

    def compute_temporal_discrimination(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute early vs late segment discrimination.

        This is the key metric for the temporal hypothesis:
        - MG should show MORE discrimination in late segments (fatigability)
        - Discrimination = difference between MG and HC similarities

        Returns:
            Dict with early/late discrimination metrics
        """
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), device=x.device, dtype=torch.long)

        self.eval()
        with torch.no_grad():
            z_segments = self.encode_segments(x)
            segment_weights = self.compute_segment_weights(lengths)

            # Average per-segment similarities across heads
            all_per_seg_sims = []
            for head in self.heads:
                _, per_seg_sims, _, _ = head(z_segments, segment_weights, self.t_default)
                all_per_seg_sims.append(per_seg_sims)

            # (batch, n_segments, n_classes)
            avg_sims = torch.stack(all_per_seg_sims, dim=0).mean(dim=0)

            # Split by label
            hc_mask = (labels == 0)
            mg_mask = (labels == 1)

            # Get similarities to correct class prototype per segment
            # HC samples -> similarity to HC prototype (index 0)
            # MG samples -> similarity to MG prototype (index 1)
            hc_sims = avg_sims[hc_mask, :, 0].mean(dim=0)  # (n_segments,)
            mg_sims = avg_sims[mg_mask, :, 1].mean(dim=0)  # (n_segments,)

            # Discrimination = MG-to-MG similarity - HC-to-HC similarity at each segment
            # Higher means that segment is more discriminative
            discrimination = mg_sims - hc_sims

            # Split into early and late
            mid = self.n_segments // 2
            early_discrim = discrimination[:mid].mean().item()
            late_discrim = discrimination[mid:].mean().item()

            return {
                'early_discrimination': early_discrim,
                'late_discrimination': late_discrim,
                'late_minus_early': late_discrim - early_discrim,
                'temporal_pattern_pass': late_discrim > early_discrim,
                'per_segment_discrimination': discrimination.cpu().numpy().tolist(),
            }

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': 'MultiHeadTrajectoryProtoNet',
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'seq_len': self.seq_len,
            'latent_dim': self.latent_dim,
            'n_heads': self.n_heads,
            'head_dim': self.head_dim,
            'n_segments': self.n_segments,
            'encoder_hidden': self.encoder_hidden,
            'encoder_layers': self.encoder_layers,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout_rate,
            'n_total_prototypes': self.n_heads * 2,
        }


# Default configuration
DEFAULT_CONFIG = {
    'latent_dim': 64,
    'n_heads': 5,
    'head_dim': 32,
    'n_segments': 8,
    'encoder_hidden': 64,
    'encoder_layers': 3,
    'kernel_size': 7,
    'dropout': 0.2,
}
