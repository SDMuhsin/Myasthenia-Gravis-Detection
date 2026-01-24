"""
Ablation-Configurable MultiHeadProtoNet

A flexible implementation that supports all ablation variants:
- Variable number of heads (Ablation 1)
- Prototype vs FC classification (Ablation 2)
- Configurable loss components (Ablation 3)
- Different fusion strategies (Ablation 4)
- Variable encoder architecture (Ablation 5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from ccece.models.base import BaseTimeSeriesModel
from ccece.experiments.ablation_study.configs import AblationConfig, FusionStrategy, ClassificationType


class PrototypeHead(nn.Module):
    """
    Single prototype head with exactly 2 prototypes (1 per class).
    Used for prototype-based classification.
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

        # Exactly 2 prototypes: one for HC (class 0), one for MG (class 1)
        self.prototypes = nn.Parameter(
            torch.randn(n_classes, head_dim) * 0.1
        )

        # Fixed class assignments
        prototype_class = torch.arange(n_classes, dtype=torch.long)
        self.register_buffer('prototype_class', prototype_class)

    def forward(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute distances, similarities, and logits for this head.

        Returns:
            distances: L2 distances to prototypes (batch_size, 2)
            similarities: Similarity scores (batch_size, 2)
            logits: Class logits (batch_size, 2)
        """
        # Project to this head's space
        h = self.projection(z)  # (batch, head_dim)
        h = F.normalize(h, p=2, dim=1)  # L2 normalize

        # Compute L2 distances to prototypes
        h_expanded = h.unsqueeze(1)  # (batch, 1, head_dim)
        proto_expanded = self.prototypes.unsqueeze(0)  # (1, 2, head_dim)
        distances = torch.sum((h_expanded - proto_expanded) ** 2, dim=2)  # (batch, 2)

        # Convert distances to similarities
        similarities = torch.log(1 + 1 / (distances + 1e-6))

        # Logits: directly use similarities
        logits = similarities

        return distances, similarities, logits

    def get_head_embedding(self, z: torch.Tensor) -> torch.Tensor:
        """Get the projected embedding for this head."""
        h = self.projection(z)
        h = F.normalize(h, p=2, dim=1)
        return h


class FCHead(nn.Module):
    """
    Standard fully-connected classification head.
    Used as baseline comparison against prototype-based classification.
    """

    def __init__(
        self,
        latent_dim: int,
        head_dim: int,
        n_classes: int = 2,
        use_dropout: bool = False,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.head_dim = head_dim
        self.n_classes = n_classes

        layers = [
            nn.Linear(latent_dim, head_dim),
            nn.ReLU(),
        ]

        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(head_dim, n_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute logits for this head.

        Returns distances and similarities as None placeholders for API compatibility.
        """
        logits = self.classifier(z)

        # Return None for distances/similarities (not applicable for FC)
        return None, None, logits


class AttentionFusion(nn.Module):
    """
    Learned attention-weighted fusion over heads.
    """

    def __init__(self, n_heads: int, latent_dim: int):
        super().__init__()
        self.n_heads = n_heads

        # Attention network that takes the encoded representation
        # and outputs attention weights for each head
        self.attention = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_heads),
        )

    def forward(
        self,
        head_logits: List[torch.Tensor],
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention-weighted combination of head logits.

        Args:
            head_logits: List of (batch, n_classes) tensors
            z: Encoded representation (batch, latent_dim)

        Returns:
            Combined logits (batch, n_classes)
        """
        # Compute attention weights
        weights = F.softmax(self.attention(z), dim=1)  # (batch, n_heads)

        # Stack head logits: (batch, n_heads, n_classes)
        stacked = torch.stack(head_logits, dim=1)

        # Weighted sum: (batch, n_classes)
        combined = (stacked * weights.unsqueeze(-1)).sum(dim=1)

        return combined


class AblationMultiHeadProtoNet(BaseTimeSeriesModel):
    """
    Ablation-configurable MultiHeadProtoNet.

    Supports all ablation variants:
    - Variable number of heads
    - Prototype vs FC classification
    - Different fusion strategies
    - Configurable encoder architecture
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        config: AblationConfig,
    ):
        super().__init__(input_dim, num_classes, seq_len)

        self.config = config
        self.latent_dim = config.latent_dim
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.encoder_hidden = config.encoder_hidden
        self.encoder_layers = config.encoder_layers
        self.kernel_size = config.kernel_size
        self.dropout_rate = config.dropout

        # Build shared CNN encoder
        self.encoder = self._build_encoder()

        # Create classification heads based on config
        if config.classification_type == ClassificationType.PROTOTYPE:
            self.heads = nn.ModuleList([
                PrototypeHead(
                    latent_dim=self.latent_dim,
                    head_dim=self.head_dim,
                    n_classes=num_classes,
                )
                for _ in range(self.n_heads)
            ])
            self.use_prototypes = True
        elif config.classification_type == ClassificationType.FC:
            self.heads = nn.ModuleList([
                FCHead(
                    latent_dim=self.latent_dim,
                    head_dim=self.head_dim,
                    n_classes=num_classes,
                    use_dropout=False,
                )
                for _ in range(self.n_heads)
            ])
            self.use_prototypes = False
        elif config.classification_type == ClassificationType.FC_DROPOUT:
            self.heads = nn.ModuleList([
                FCHead(
                    latent_dim=self.latent_dim,
                    head_dim=self.head_dim,
                    n_classes=num_classes,
                    use_dropout=True,
                    dropout_rate=config.fc_dropout,
                )
                for _ in range(self.n_heads)
            ])
            self.use_prototypes = False

        # Setup fusion strategy
        self.fusion_strategy = config.fusion_strategy
        if self.fusion_strategy == FusionStrategy.ATTENTION:
            self.attention_fusion = AttentionFusion(self.n_heads, self.latent_dim)

        # For storing training embeddings
        self._training_embeddings: Optional[torch.Tensor] = None
        self._training_labels: Optional[torch.Tensor] = None

    def _build_encoder(self) -> nn.Module:
        """Build the shared 1D CNN encoder."""
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input time series to shared latent representation."""
        # Transpose for conv1d: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)

        # Encode
        features = self.encoder(x)

        # Project to latent space
        z = self.projection_head(features)

        # L2 normalize
        z = F.normalize(z, p=2, dim=1)

        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with configured fusion strategy."""
        z = self.encode(x)

        # Collect logits from all heads
        all_logits = []
        for head in self.heads:
            _, _, head_logits = head(z)
            all_logits.append(head_logits)

        # Apply fusion strategy
        if self.fusion_strategy == FusionStrategy.AVERAGE:
            logits = torch.stack(all_logits, dim=0).mean(dim=0)

        elif self.fusion_strategy == FusionStrategy.MAX:
            stacked = torch.stack(all_logits, dim=0)  # (n_heads, batch, n_classes)
            logits = stacked.max(dim=0)[0]

        elif self.fusion_strategy == FusionStrategy.ATTENTION:
            logits = self.attention_fusion(all_logits, z)

        elif self.fusion_strategy == FusionStrategy.VOTING:
            # Soft voting: average probabilities then convert back to logits
            # This is differentiable unlike hard voting
            probs = torch.stack([F.softmax(l, dim=1) for l in all_logits], dim=0)
            avg_probs = probs.mean(dim=0)
            # Convert back to logits (log of probs)
            logits = torch.log(avg_probs + 1e-8)

        return logits

    def forward_with_explanations(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Optional[torch.Tensor]], List[torch.Tensor]]:
        """Forward pass with full explanation outputs."""
        z = self.encode(x)

        all_logits = []
        all_distances = []
        all_similarities = []

        for head in self.heads:
            distances, similarities, head_logits = head(z)
            all_logits.append(head_logits)
            all_distances.append(distances)
            all_similarities.append(similarities if similarities is not None else head_logits)

        # Apply fusion strategy
        if self.fusion_strategy == FusionStrategy.AVERAGE:
            logits = torch.stack(all_logits, dim=0).mean(dim=0)
        elif self.fusion_strategy == FusionStrategy.MAX:
            stacked = torch.stack(all_logits, dim=0)
            logits = stacked.max(dim=0)[0]
        elif self.fusion_strategy == FusionStrategy.ATTENTION:
            logits = self.attention_fusion(all_logits, z)
        elif self.fusion_strategy == FusionStrategy.VOTING:
            probs = torch.stack([F.softmax(l, dim=1) for l in all_logits], dim=0)
            avg_probs = probs.mean(dim=0)
            logits = torch.log(avg_probs + 1e-8)

        return logits, z, all_distances, all_similarities

    def forward_per_head(
        self,
        x: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Get per-head logits and predictions."""
        z = self.encode(x)

        head_logits = []
        head_predictions = []

        for head in self.heads:
            _, _, logits = head(z)
            head_logits.append(logits)
            head_predictions.append(logits.argmax(dim=1))

        return head_logits, head_predictions

    def compute_prototype_loss(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute prototype-specific losses for all heads.

        Only applicable when using prototype-based classification.
        Returns zero losses for FC classification.
        """
        device = z.device

        if not self.use_prototypes:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        batch_size = z.size(0)

        total_cluster_loss = torch.tensor(0.0, device=device)
        total_separation_loss = torch.tensor(0.0, device=device)

        for head in self.heads:
            distances, _, _ = head(z)  # (batch, 2)

            # Cluster loss: distance to same-class prototype
            cluster_losses = []
            for i in range(batch_size):
                label = labels[i].item()
                same_class_dist = distances[i, label]
                cluster_losses.append(same_class_dist)

            cluster_loss = torch.stack(cluster_losses).mean()
            total_cluster_loss = total_cluster_loss + cluster_loss

            # Separation loss: prototypes should be far apart
            proto_0 = head.prototypes[0]
            proto_1 = head.prototypes[1]
            inter_proto_dist = torch.sum((proto_0 - proto_1) ** 2)
            separation_loss = -inter_proto_dist
            total_separation_loss = total_separation_loss + separation_loss

        # Average across heads
        total_cluster_loss = total_cluster_loss / self.n_heads
        total_separation_loss = total_separation_loss / self.n_heads

        return total_cluster_loss, total_separation_loss

    def store_training_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Store training embeddings for alignment computation."""
        self._training_embeddings = embeddings.detach()
        self._training_labels = labels.detach()

    def compute_alignment_per_head(
        self,
        embeddings: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute alignment metrics for each head.

        Only meaningful for prototype-based classification.
        """
        if not self.use_prototypes:
            # Return dummy metrics for FC classification
            return {
                i: {
                    'alignment': 0.0,
                    'hc_alignment': 0.0,
                    'mg_alignment': 0.0,
                    'proto_distance': 0.0,
                }
                for i in range(self.n_heads)
            }

        if embeddings is None:
            embeddings = self._training_embeddings
        if labels is None:
            labels = self._training_labels

        if embeddings is None:
            raise ValueError("No embeddings available.")

        device = next(self.parameters()).device
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        results = {}

        for head_idx, head in enumerate(self.heads):
            distances, _, _ = head(embeddings)

            # For each sample, check if nearest prototype matches class
            nearest_proto = distances.argmin(dim=1)
            correct = (nearest_proto == labels).float()

            alignment = correct.mean().item()

            # Per-class alignment
            hc_mask = (labels == 0)
            mg_mask = (labels == 1)

            hc_alignment = correct[hc_mask].mean().item() if hc_mask.any() else 0.0
            mg_alignment = correct[mg_mask].mean().item() if mg_mask.any() else 0.0

            # Prototype distances
            proto_dist = torch.sum((head.prototypes[0] - head.prototypes[1]) ** 2).item()

            results[head_idx] = {
                'alignment': alignment,
                'hc_alignment': hc_alignment,
                'mg_alignment': mg_alignment,
                'proto_distance': proto_dist,
            }

        return results

    def compute_overall_alignment(
        self,
        embeddings: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> float:
        """Compute overall alignment across all heads."""
        if not self.use_prototypes:
            return 0.0

        per_head = self.compute_alignment_per_head(embeddings, labels)
        alignments = [per_head[i]['alignment'] for i in range(self.n_heads)]
        return np.mean(alignments)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': 'AblationMultiHeadProtoNet',
            'ablation_name': self.config.ablation_name,
            'variant_name': self.config.variant_name,
            **self.config.to_dict(),
        }


def create_model_for_ablation(
    input_dim: int,
    num_classes: int,
    seq_len: int,
    config: AblationConfig,
) -> AblationMultiHeadProtoNet:
    """
    Factory function to create a model for a specific ablation configuration.

    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        seq_len: Sequence length
        config: AblationConfig specifying the variant

    Returns:
        Configured AblationMultiHeadProtoNet
    """
    return AblationMultiHeadProtoNet(
        input_dim=input_dim,
        num_classes=num_classes,
        seq_len=seq_len,
        config=config,
    )
