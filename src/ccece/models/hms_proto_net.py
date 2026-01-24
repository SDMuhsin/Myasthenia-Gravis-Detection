"""
CCECE Paper: Hierarchical Multi-Scale Prototype Network (HMSProtoNet)

A novel explainable architecture for MG detection that learns separate prototype
banks at multiple temporal resolutions with learned scale fusion.

Novelty: First hierarchical multi-scale prototype network for time series
classification with scale-specific explainability.

Explainability Mechanism:
- Model learns prototypes at 3 temporal scales (macro, meso, micro)
- Each scale captures different temporal characteristics:
  - Macro: Overall fatigue trajectory
  - Meso: Medium-term patterns
  - Micro: Fine motor fluctuations
- Predictions based on weighted fusion of scale-specific similarities
- Explanations: "Classified as MG due to macro-scale prototype 3 and micro-scale prototype 2"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from .base import BaseTimeSeriesModel


class ScalePrototypeBank(nn.Module):
    """
    Prototype bank for a single temporal scale.
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

        # Class identity for logit computation
        class_identity = torch.zeros(self.n_prototypes, n_classes)
        for i in range(self.n_prototypes):
            class_identity[i, prototype_class[i]] = 1.0
        self.register_buffer('class_identity', class_identity)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute prototype similarities and class logits.

        Args:
            z: Encoded input (batch_size, latent_dim)

        Returns:
            distances: L2 distances to prototypes (batch_size, n_prototypes)
            similarities: Similarity scores (batch_size, n_prototypes)
            logits: Class logits (batch_size, n_classes)
        """
        # Compute L2 distances
        z_expanded = z.unsqueeze(1)
        proto_expanded = self.prototypes.unsqueeze(0)
        distances = torch.sum((z_expanded - proto_expanded) ** 2, dim=2)

        # Convert distances to similarities
        similarities = torch.log(1 + 1 / (distances + 1e-6))

        # Class logits
        logits = torch.matmul(similarities, self.class_identity)

        return distances, similarities, logits


class HMSProtoNet(BaseTimeSeriesModel):
    """
    Hierarchical Multi-Scale Prototype Network for explainable MG detection.

    Architecture:
        Input (batch, seq_len, input_dim)
            |
            v
        Shared CNN Encoder -> Feature map (batch, hidden, T_reduced)
            |
            ├── Macro Pool (12 positions) -> Macro Prototypes -> macro_logits
            |
            ├── Meso Pool (48 positions) -> Meso Prototypes -> meso_logits
            |
            └── Micro Pool (96 positions) -> Micro Prototypes -> micro_logits
            |
            v
        Scale Fusion (learned weights) -> final_logits

    Explainability:
        - Predictions based on multi-scale prototype similarities
        - Each scale captures different temporal characteristics
        - Can explain which scales and prototypes drive the decision
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        latent_dim: int = 64,
        n_prototypes_per_class: int = 3,
        encoder_hidden: int = 64,
        encoder_layers: int = 3,
        kernel_size: int = 7,
        dropout: float = 0.2,
        macro_positions: int = 12,
        meso_positions: int = 48,
        micro_positions: int = 96,
        use_diversity_loss: bool = False,
        diversity_weight: float = 0.01,
        **kwargs,
    ):
        """
        Args:
            input_dim: Number of input features per timestep
            num_classes: Number of output classes (2 for binary)
            seq_len: Expected sequence length
            latent_dim: Dimension of latent space for prototypes
            n_prototypes_per_class: Number of prototypes per class per scale
            encoder_hidden: Hidden dimension in encoder
            encoder_layers: Number of encoder conv layers
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
            macro_positions: Number of positions for macro-scale pooling
            meso_positions: Number of positions for meso-scale pooling
            micro_positions: Number of positions for micro-scale pooling
            use_diversity_loss: Whether to use scale diversity loss
            diversity_weight: Weight for diversity loss
        """
        super().__init__(input_dim, num_classes, seq_len)

        self.latent_dim = latent_dim
        self.n_prototypes_per_class = n_prototypes_per_class
        self.encoder_hidden = encoder_hidden
        self.encoder_layers = encoder_layers
        self.kernel_size = kernel_size
        self.dropout_rate = dropout
        self.macro_positions = macro_positions
        self.meso_positions = meso_positions
        self.micro_positions = micro_positions
        self.use_diversity_loss = use_diversity_loss
        self.diversity_weight = diversity_weight

        # Build shared CNN encoder
        self.encoder = self._build_encoder()

        # Multi-scale pooling
        self.macro_pool = nn.AdaptiveAvgPool1d(macro_positions)
        self.meso_pool = nn.AdaptiveAvgPool1d(meso_positions)
        self.micro_pool = nn.AdaptiveAvgPool1d(micro_positions)

        # Scale-specific projection heads (shared encoder output -> latent dim)
        self.macro_proj = nn.Sequential(
            nn.Linear(self.encoder_output_dim * macro_positions, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )
        self.meso_proj = nn.Sequential(
            nn.Linear(self.encoder_output_dim * meso_positions, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )
        self.micro_proj = nn.Sequential(
            nn.Linear(self.encoder_output_dim * micro_positions, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )

        # Scale-specific prototype banks
        self.macro_prototypes = ScalePrototypeBank(
            latent_dim=latent_dim,
            n_prototypes_per_class=n_prototypes_per_class,
            n_classes=num_classes,
        )
        self.meso_prototypes = ScalePrototypeBank(
            latent_dim=latent_dim,
            n_prototypes_per_class=n_prototypes_per_class,
            n_classes=num_classes,
        )
        self.micro_prototypes = ScalePrototypeBank(
            latent_dim=latent_dim,
            n_prototypes_per_class=n_prototypes_per_class,
            n_classes=num_classes,
        )

        # Learned scale fusion weights
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)

        # For storing training embeddings (for prototype projection)
        self._training_embeddings = {
            'macro': None, 'meso': None, 'micro': None
        }
        self._training_labels: Optional[torch.Tensor] = None

    def _build_encoder(self) -> nn.Module:
        """Build the shared 1D CNN encoder (same as TempProtoNet)."""
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

    def encode_multi_scale(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input at multiple scales.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            macro_z: Macro-scale encoding (batch_size, latent_dim)
            meso_z: Meso-scale encoding (batch_size, latent_dim)
            micro_z: Micro-scale encoding (batch_size, latent_dim)
        """
        # Transpose for conv1d: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)

        # Shared encoder: (batch_size, encoder_output_dim, reduced_seq_len)
        features = self.encoder(x)

        # Multi-scale pooling
        macro_pooled = self.macro_pool(features)  # (batch, hidden, macro_positions)
        meso_pooled = self.meso_pool(features)    # (batch, hidden, meso_positions)
        micro_pooled = self.micro_pool(features)  # (batch, hidden, micro_positions)

        # Flatten pooled features
        batch_size = x.size(0)
        macro_flat = macro_pooled.view(batch_size, -1)
        meso_flat = meso_pooled.view(batch_size, -1)
        micro_flat = micro_pooled.view(batch_size, -1)

        # Project to latent space
        macro_z = self.macro_proj(macro_flat)
        meso_z = self.meso_proj(meso_flat)
        micro_z = self.micro_proj(micro_flat)

        # L2 normalize
        macro_z = F.normalize(macro_z, p=2, dim=1)
        meso_z = F.normalize(meso_z, p=2, dim=1)
        micro_z = F.normalize(micro_z, p=2, dim=1)

        return macro_z, meso_z, micro_z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            logits: Class logits (batch_size, num_classes)
        """
        # Encode at multiple scales
        macro_z, meso_z, micro_z = self.encode_multi_scale(x)

        # Compute scale-specific logits
        _, _, macro_logits = self.macro_prototypes(macro_z)
        _, _, meso_logits = self.meso_prototypes(meso_z)
        _, _, micro_logits = self.micro_prototypes(micro_z)

        # Fuse scales with learned weights
        weights = F.softmax(self.scale_weights, dim=0)
        logits = (
            weights[0] * macro_logits +
            weights[1] * meso_logits +
            weights[2] * micro_logits
        )

        return logits

    def forward_with_explanations(
        self,
        x: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Forward pass with full explanation outputs.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            Dict containing:
                - logits: Final class logits
                - scale_weights: Learned scale fusion weights
                - macro_*: Macro-scale outputs (z, distances, similarities, logits)
                - meso_*: Meso-scale outputs
                - micro_*: Micro-scale outputs
        """
        # Encode at multiple scales
        macro_z, meso_z, micro_z = self.encode_multi_scale(x)

        # Compute scale-specific outputs
        macro_dist, macro_sim, macro_logits = self.macro_prototypes(macro_z)
        meso_dist, meso_sim, meso_logits = self.meso_prototypes(meso_z)
        micro_dist, micro_sim, micro_logits = self.micro_prototypes(micro_z)

        # Fuse scales
        weights = F.softmax(self.scale_weights, dim=0)
        logits = (
            weights[0] * macro_logits +
            weights[1] * meso_logits +
            weights[2] * micro_logits
        )

        return {
            'logits': logits,
            'scale_weights': weights,
            # Macro
            'macro_z': macro_z,
            'macro_distances': macro_dist,
            'macro_similarities': macro_sim,
            'macro_logits': macro_logits,
            # Meso
            'meso_z': meso_z,
            'meso_distances': meso_dist,
            'meso_similarities': meso_sim,
            'meso_logits': meso_logits,
            # Micro
            'micro_z': micro_z,
            'micro_distances': micro_dist,
            'micro_similarities': micro_sim,
            'micro_logits': micro_logits,
        }

    def compute_prototype_loss(
        self,
        outputs: Dict[str, Any],
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute prototype-specific losses for training.

        Args:
            outputs: Output dict from forward_with_explanations
            labels: Ground truth labels (batch_size,)

        Returns:
            cluster_loss: Loss encouraging samples to be close to same-class prototypes
            separation_loss: Loss encouraging prototype separation
            diversity_loss: Loss encouraging scale diversity (optional)
        """
        batch_size = labels.size(0)

        # Compute losses for each scale
        scales = [
            ('macro', self.macro_prototypes),
            ('meso', self.meso_prototypes),
            ('micro', self.micro_prototypes),
        ]

        cluster_losses = []
        separation_losses = []

        for scale_name, proto_bank in scales:
            distances = outputs[f'{scale_name}_distances']
            prototype_classes = proto_bank.prototype_class

            # Cluster loss
            scale_cluster_losses = []
            for i in range(batch_size):
                label = labels[i].item()
                same_class_mask = (prototype_classes == label)
                same_class_distances = distances[i][same_class_mask]
                min_dist = same_class_distances.min()
                scale_cluster_losses.append(min_dist)
            cluster_losses.append(torch.stack(scale_cluster_losses).mean())

            # Separation loss
            prototypes = proto_bank.prototypes
            sep_losses_scale = []
            for c in range(self.num_classes):
                class_c_mask = (prototype_classes == c)
                other_mask = (prototype_classes != c)
                class_c_protos = prototypes[class_c_mask]
                other_protos = prototypes[other_mask]

                if len(class_c_protos) > 0 and len(other_protos) > 0:
                    dists = torch.cdist(class_c_protos, other_protos)
                    min_dist = dists.min()
                    sep_losses_scale.append(-min_dist)

            if sep_losses_scale:
                separation_losses.append(torch.stack(sep_losses_scale).mean())

        # Average across scales
        cluster_loss = torch.stack(cluster_losses).mean()
        separation_loss = torch.stack(separation_losses).mean() if separation_losses else torch.tensor(0.0, device=labels.device)

        # Diversity loss (optional)
        if self.use_diversity_loss:
            # Penalize if scale logits are too similar
            macro_logits = outputs['macro_logits']
            meso_logits = outputs['meso_logits']
            micro_logits = outputs['micro_logits']

            # Normalize logits for cosine similarity
            macro_norm = F.normalize(macro_logits, p=2, dim=1)
            meso_norm = F.normalize(meso_logits, p=2, dim=1)
            micro_norm = F.normalize(micro_logits, p=2, dim=1)

            # Cosine similarity between scale predictions
            sim_macro_meso = (macro_norm * meso_norm).sum(dim=1).mean()
            sim_meso_micro = (meso_norm * micro_norm).sum(dim=1).mean()

            diversity_loss = self.diversity_weight * (sim_macro_meso + sim_meso_micro)
        else:
            diversity_loss = torch.tensor(0.0, device=labels.device)

        return cluster_loss, separation_loss, diversity_loss

    def get_scale_weights(self) -> Dict[str, float]:
        """Get the learned scale fusion weights."""
        weights = F.softmax(self.scale_weights, dim=0)
        return {
            'macro': weights[0].item(),
            'meso': weights[1].item(),
            'micro': weights[2].item(),
        }

    def compute_prototype_diversity(self) -> Dict[str, Any]:
        """Compute prototype diversity metrics for each scale."""
        results = {}

        for scale_name, proto_bank in [
            ('macro', self.macro_prototypes),
            ('meso', self.meso_prototypes),
            ('micro', self.micro_prototypes),
        ]:
            prototypes = proto_bank.prototypes.detach()
            prototype_classes = proto_bank.prototype_class

            # Pairwise distances
            pairwise_dists = torch.cdist(prototypes, prototypes)
            n = prototypes.size(0)
            mask = ~torch.eye(n, dtype=torch.bool, device=prototypes.device)
            off_diagonal = pairwise_dists[mask]

            mean_dist = off_diagonal.mean().item()
            min_dist = off_diagonal.min().item()

            # Inter-class and intra-class
            inter_dists = []
            intra_dists = []
            for i in range(n):
                for j in range(i+1, n):
                    dist = pairwise_dists[i, j].item()
                    if prototype_classes[i] == prototype_classes[j]:
                        intra_dists.append(dist)
                    else:
                        inter_dists.append(dist)

            results[scale_name] = {
                'mean_pairwise_distance': mean_dist,
                'min_pairwise_distance': min_dist,
                'inter_class_distance': np.mean(inter_dists) if inter_dists else 0.0,
                'intra_class_distance': np.mean(intra_dists) if intra_dists else 0.0,
            }

        return results

    def store_training_embeddings(
        self,
        macro_embeddings: torch.Tensor,
        meso_embeddings: torch.Tensor,
        micro_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Store training embeddings for prototype projection."""
        self._training_embeddings['macro'] = macro_embeddings.detach()
        self._training_embeddings['meso'] = meso_embeddings.detach()
        self._training_embeddings['micro'] = micro_embeddings.detach()
        self._training_labels = labels.detach()

    def project_prototypes(
        self,
        k: int = 3,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Project prototypes to nearest training samples at each scale.

        Args:
            k: Number of nearest neighbors per prototype

        Returns:
            Dict mapping scale name to list of prototype projections
        """
        if self._training_labels is None:
            raise ValueError("Training embeddings not stored.")

        projections = {}

        for scale_name, proto_bank in [
            ('macro', self.macro_prototypes),
            ('meso', self.meso_prototypes),
            ('micro', self.micro_prototypes),
        ]:
            embeddings = self._training_embeddings[scale_name]
            prototypes = proto_bank.prototypes.detach()
            prototype_classes = proto_bank.prototype_class

            scale_projections = []
            for proto_idx in range(prototypes.size(0)):
                proto = prototypes[proto_idx:proto_idx+1]
                dists = torch.cdist(proto, embeddings).squeeze(0)
                topk_dists, topk_indices = torch.topk(dists, k, largest=False)

                projection = {
                    'prototype_idx': proto_idx,
                    'prototype_class': prototype_classes[proto_idx].item(),
                    'nearest_indices': topk_indices.cpu().numpy(),
                    'nearest_distances': topk_dists.cpu().numpy(),
                    'nearest_labels': self._training_labels[topk_indices].cpu().numpy(),
                }
                scale_projections.append(projection)

            projections[scale_name] = scale_projections

        return projections

    def explain_prediction(
        self,
        x: torch.Tensor,
        top_k: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for predictions.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            top_k: Number of most similar prototypes per scale

        Returns:
            List of explanations, one per sample
        """
        with torch.no_grad():
            outputs = self.forward_with_explanations(x)
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=1)
            probabilities = F.softmax(logits, dim=1)
            scale_weights = outputs['scale_weights']

            explanations = []

            for i in range(x.size(0)):
                pred = predictions[i].item()
                prob = probabilities[i, pred].item()

                explanation = {
                    'prediction': pred,
                    'prediction_name': 'MG' if pred == 1 else 'HC',
                    'confidence': prob,
                    'scale_weights': {
                        'macro': scale_weights[0].item(),
                        'meso': scale_weights[1].item(),
                        'micro': scale_weights[2].item(),
                    },
                    'scale_contributions': {},
                }

                # Per-scale explanations
                for scale_name, proto_bank in [
                    ('macro', self.macro_prototypes),
                    ('meso', self.meso_prototypes),
                    ('micro', self.micro_prototypes),
                ]:
                    sims = outputs[f'{scale_name}_similarities'][i]
                    scale_logits = outputs[f'{scale_name}_logits'][i]
                    proto_classes = proto_bank.prototype_class

                    topk_sims, topk_indices = torch.topk(sims, top_k)
                    top_prototypes = []
                    for j in range(top_k):
                        idx = topk_indices[j].item()
                        top_prototypes.append({
                            'prototype_idx': idx,
                            'prototype_class': proto_classes[idx].item(),
                            'prototype_class_name': 'MG' if proto_classes[idx].item() == 1 else 'HC',
                            'similarity': topk_sims[j].item(),
                        })

                    # Scale prediction
                    scale_pred = torch.argmax(scale_logits).item()

                    explanation['scale_contributions'][scale_name] = {
                        'prediction': scale_pred,
                        'prediction_name': 'MG' if scale_pred == 1 else 'HC',
                        'top_prototypes': top_prototypes,
                    }

                explanations.append(explanation)

        return explanations

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': 'HMSProtoNet',
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'seq_len': self.seq_len,
            'latent_dim': self.latent_dim,
            'n_prototypes_per_class': self.n_prototypes_per_class,
            'encoder_hidden': self.encoder_hidden,
            'encoder_layers': self.encoder_layers,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout_rate,
            'macro_positions': self.macro_positions,
            'meso_positions': self.meso_positions,
            'micro_positions': self.micro_positions,
            'use_diversity_loss': self.use_diversity_loss,
            'diversity_weight': self.diversity_weight,
            'n_total_prototypes': (
                self.macro_prototypes.n_prototypes +
                self.meso_prototypes.n_prototypes +
                self.micro_prototypes.n_prototypes
            ),
        }


# Default configuration
DEFAULT_CONFIG = {
    'latent_dim': 64,
    'n_prototypes_per_class': 3,
    'encoder_hidden': 64,
    'encoder_layers': 3,
    'kernel_size': 7,
    'dropout': 0.2,
    'macro_positions': 12,
    'meso_positions': 48,
    'micro_positions': 96,
    'use_diversity_loss': False,
    'diversity_weight': 0.01,
}
