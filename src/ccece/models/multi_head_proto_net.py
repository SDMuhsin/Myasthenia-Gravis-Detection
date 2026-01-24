"""
Multi-Head Prototype Network for MG Detection

This architecture addresses the prototype collapse problem through structural constraints:
- K independent heads, each with exactly 2 prototypes (1 per class)
- Each head MUST be discriminative to contribute to final prediction
- With only 2 prototypes per head, collapse is impossible
- Alignment ≈ per-head accuracy (structural guarantee)

Key insight from previous failures:
- 10 loss/architectural modifications failed because they didn't change the fundamental
  assignment mechanism or training dynamics
- This approach uses STRUCTURAL constraints: each head has only 2 prototypes,
  so BOTH must be used for classification

Expected behavior:
- Each head achieves ~70% accuracy
- With 2 prototypes per head, alignment ≈ head accuracy
- Multiple heads averaging together maintain or improve accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from .base import BaseTimeSeriesModel


class PrototypeHead(nn.Module):
    """
    Single prototype head with exactly 2 prototypes (1 per class).

    This structural constraint makes collapse impossible:
    - With only 2 prototypes, both MUST be used for classification
    - Alignment = head accuracy (by construction)
    """

    def __init__(
        self,
        latent_dim: int,
        head_dim: int,
        n_classes: int = 2,
    ):
        """
        Args:
            latent_dim: Input dimension from encoder
            head_dim: Dimension of this head's prototype space
            n_classes: Number of classes (2 for binary)
        """
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

        # Fixed class assignments: prototype 0 -> class 0, prototype 1 -> class 1
        prototype_class = torch.arange(n_classes, dtype=torch.long)
        self.register_buffer('prototype_class', prototype_class)

    def forward(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute distances, similarities, and logits for this head.

        Args:
            z: Encoded input (batch_size, latent_dim)

        Returns:
            distances: L2 distances to prototypes (batch_size, 2)
            similarities: Similarity scores (batch_size, 2)
            logits: Class logits (batch_size, 2) - direct from prototype similarities
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

        # Logits: directly use similarities as logits
        # prototype 0 similarity -> class 0 logit, prototype 1 similarity -> class 1 logit
        logits = similarities

        return distances, similarities, logits

    def get_head_embedding(self, z: torch.Tensor) -> torch.Tensor:
        """Get the projected embedding for this head."""
        h = self.projection(z)
        h = F.normalize(h, p=2, dim=1)
        return h


class MultiHeadProtoNet(BaseTimeSeriesModel):
    """
    Multi-Head Prototype Network for explainable MG detection.

    Architecture:
        Input (batch, seq_len, input_dim)
            |
            v
        Shared CNN Encoder -> Latent representation (batch, latent_dim)
            |
            v
        K Independent Prototype Heads, each with 2 prototypes
            |
            v
        Average logits across heads -> Final prediction

    Why this addresses prototype collapse:
        - Each head has exactly 2 prototypes (1 per class)
        - Both prototypes MUST be used for the head to classify correctly
        - No winner-take-all: with 2 prototypes, both are equally important
        - Alignment ≈ head accuracy (structural guarantee)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        latent_dim: int = 64,
        n_heads: int = 5,
        head_dim: int = 32,
        encoder_hidden: int = 64,
        encoder_layers: int = 3,
        kernel_size: int = 7,
        dropout: float = 0.2,
        **kwargs,
    ):
        """
        Args:
            input_dim: Number of input features per timestep
            num_classes: Number of output classes (2 for binary)
            seq_len: Expected sequence length
            latent_dim: Dimension of shared encoder output
            n_heads: Number of prototype heads (total prototypes = n_heads * 2)
            head_dim: Dimension of each head's prototype space
            encoder_hidden: Hidden dimension in encoder
            encoder_layers: Number of encoder conv layers
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
        """
        super().__init__(input_dim, num_classes, seq_len)

        self.latent_dim = latent_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.encoder_hidden = encoder_hidden
        self.encoder_layers = encoder_layers
        self.kernel_size = kernel_size
        self.dropout_rate = dropout

        # Build shared CNN encoder
        self.encoder = self._build_encoder()

        # Create K independent prototype heads
        self.heads = nn.ModuleList([
            PrototypeHead(
                latent_dim=latent_dim,
                head_dim=head_dim,
                n_classes=num_classes,
            )
            for _ in range(n_heads)
        ])

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
        """
        Encode input time series to shared latent representation.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            z: Latent representation (batch_size, latent_dim)
        """
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
        """
        Forward pass: average of all head logits.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            logits: Class logits (batch_size, num_classes)
        """
        z = self.encode(x)

        # Collect logits from all heads
        all_logits = []
        for head in self.heads:
            _, _, head_logits = head(z)
            all_logits.append(head_logits)

        # Average logits across heads
        logits = torch.stack(all_logits, dim=0).mean(dim=0)

        return logits

    def forward_with_explanations(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass with full explanation outputs.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            logits: Class logits (batch_size, num_classes)
            z: Shared latent representation (batch_size, latent_dim)
            all_distances: List of distances per head (each: batch_size, 2)
            all_similarities: List of similarities per head (each: batch_size, 2)
        """
        z = self.encode(x)

        all_logits = []
        all_distances = []
        all_similarities = []

        for head in self.heads:
            distances, similarities, head_logits = head(z)
            all_logits.append(head_logits)
            all_distances.append(distances)
            all_similarities.append(similarities)

        logits = torch.stack(all_logits, dim=0).mean(dim=0)

        return logits, z, all_distances, all_similarities

    def forward_per_head(
        self,
        x: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Get per-head logits and predictions.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            head_logits: List of logits per head (each: batch_size, 2)
            head_predictions: List of predictions per head (each: batch_size,)
        """
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

        For each head:
        - Cluster loss: samples should be close to their same-class prototype
        - Separation loss: different-class prototypes should be far apart

        Args:
            z: Encoded representations (batch_size, latent_dim)
            labels: Ground truth labels (batch_size,)

        Returns:
            total_cluster_loss: Sum of cluster losses across heads
            total_separation_loss: Sum of separation losses across heads
        """
        batch_size = z.size(0)
        device = z.device

        total_cluster_loss = torch.tensor(0.0, device=device)
        total_separation_loss = torch.tensor(0.0, device=device)

        for head in self.heads:
            distances, _, _ = head(z)  # (batch, 2)

            # Cluster loss: distance to same-class prototype
            # For each sample, we want it close to its class prototype
            cluster_losses = []
            for i in range(batch_size):
                label = labels[i].item()
                # Distance to same-class prototype (prototype index = class index)
                same_class_dist = distances[i, label]
                cluster_losses.append(same_class_dist)

            cluster_loss = torch.stack(cluster_losses).mean()
            total_cluster_loss = total_cluster_loss + cluster_loss

            # Separation loss: prototypes should be far apart
            proto_0 = head.prototypes[0]
            proto_1 = head.prototypes[1]
            inter_proto_dist = torch.sum((proto_0 - proto_1) ** 2)
            separation_loss = -inter_proto_dist  # Negative because we want to maximize distance
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
        """Store training embeddings for prototype projection."""
        self._training_embeddings = embeddings.detach()
        self._training_labels = labels.detach()

    def compute_alignment_per_head(
        self,
        embeddings: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute alignment metrics for each head.

        Alignment = % of samples whose nearest prototype matches their class

        With 2 prototypes per head:
        - Alignment ≈ head accuracy (by construction)

        Returns:
            Dict mapping head_idx to alignment metrics
        """
        if embeddings is None:
            embeddings = self._training_embeddings
        if labels is None:
            labels = self._training_labels

        if embeddings is None:
            raise ValueError("No embeddings available. Call store_training_embeddings first.")

        # Move embeddings to same device as model
        device = next(self.parameters()).device
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        results = {}

        for head_idx, head in enumerate(self.heads):
            distances, _, _ = head(embeddings)  # (n_samples, 2)

            # For each sample, check if nearest prototype matches class
            nearest_proto = distances.argmin(dim=1)  # (n_samples,)
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
        """
        Compute overall alignment across all heads.

        Overall alignment = average of per-head alignments
        """
        per_head = self.compute_alignment_per_head(embeddings, labels)
        alignments = [per_head[i]['alignment'] for i in range(self.n_heads)]
        return np.mean(alignments)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': 'MultiHeadProtoNet',
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'seq_len': self.seq_len,
            'latent_dim': self.latent_dim,
            'n_heads': self.n_heads,
            'head_dim': self.head_dim,
            'encoder_hidden': self.encoder_hidden,
            'encoder_layers': self.encoder_layers,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout_rate,
            'n_total_prototypes': self.n_heads * 2,
        }

    def explain_prediction(
        self,
        x: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """
        Generate explanation for predictions.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            List of explanations, one per sample
        """
        with torch.no_grad():
            logits, z, all_distances, all_similarities = self.forward_with_explanations(x)
            predictions = logits.argmax(dim=1)
            probabilities = F.softmax(logits, dim=1)

            # Get per-head predictions
            head_logits, head_preds = self.forward_per_head(x)

            explanations = []

            for i in range(x.size(0)):
                pred = predictions[i].item()
                prob = probabilities[i, pred].item()

                # Per-head info
                head_info = []
                for h_idx in range(self.n_heads):
                    head_pred = head_preds[h_idx][i].item()
                    head_sim = all_similarities[h_idx][i]

                    head_info.append({
                        'head_idx': h_idx,
                        'prediction': head_pred,
                        'prediction_name': 'MG' if head_pred == 1 else 'HC',
                        'hc_similarity': head_sim[0].item(),
                        'mg_similarity': head_sim[1].item(),
                        'agrees_with_final': head_pred == pred,
                    })

                n_agree = sum(1 for h in head_info if h['agrees_with_final'])

                explanation = {
                    'prediction': pred,
                    'prediction_name': 'MG' if pred == 1 else 'HC',
                    'confidence': prob,
                    'n_heads_agree': n_agree,
                    'agreement_rate': n_agree / self.n_heads,
                    'per_head_info': head_info,
                }

                explanations.append(explanation)

            return explanations


# Default configuration
DEFAULT_CONFIG = {
    'latent_dim': 64,
    'n_heads': 5,
    'head_dim': 32,
    'encoder_hidden': 64,
    'encoder_layers': 3,
    'kernel_size': 7,
    'dropout': 0.2,
}
