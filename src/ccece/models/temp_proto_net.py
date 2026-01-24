"""
CCECE Paper: Temporal Prototype Network (TempProtoNet)

A genuinely explainable architecture for MG detection based on prototype learning.

Explainability Mechanism:
- Model learns K prototype vectors in a latent space
- Each prototype is associated with a class (MG or HC)
- Predictions are based on similarity to prototypes
- Explanations: "Classified as MG because similar to prototype P_3"

Key Features:
- Prototypes are LEARNED (not handcrafted formulas)
- Explainability is INTRINSIC (not post-hoc attribution)
- Clinician-friendly explanations via prototype projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from .base import BaseTimeSeriesModel


class PrototypeLayer(nn.Module):
    """
    Learnable prototype layer for prototype-based classification.

    Each prototype is a learnable vector in the latent space.
    Classification is based on similarity to class-specific prototypes.
    """

    def __init__(
        self,
        latent_dim: int,
        n_prototypes_per_class: int,
        n_classes: int = 2,
        classification_temperature: float = 0.1,
    ):
        """
        Args:
            latent_dim: Dimension of the latent space
            n_prototypes_per_class: Number of prototypes per class
            n_classes: Number of classes (default 2 for binary)
            classification_temperature: Temperature for softmax over prototype similarities.
                Low values (~0.1) concentrate weight on nearest prototype, forcing
                individual prototype alignment. High values (~1.0) approximate sum.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.n_prototypes_per_class = n_prototypes_per_class
        self.n_classes = n_classes
        self.n_prototypes = n_prototypes_per_class * n_classes
        self.classification_temperature = classification_temperature

        # Learnable prototypes: (n_prototypes, latent_dim)
        self.prototypes = nn.Parameter(
            torch.randn(self.n_prototypes, latent_dim) * 0.1
        )

        # Prototype-to-class mapping (not learnable, fixed assignment)
        # First n_prototypes_per_class belong to class 0 (HC)
        # Next n_prototypes_per_class belong to class 1 (MG)
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

    def forward(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute prototype similarities and class logits.

        Args:
            z: Encoded input (batch_size, latent_dim)

        Returns:
            distances: L2 distances to prototypes (batch_size, n_prototypes)
            similarities: Similarity scores (batch_size, n_prototypes)
            logits: Class logits (batch_size, n_classes)
        """
        # Compute L2 distances: (batch_size, n_prototypes)
        z_expanded = z.unsqueeze(1)
        proto_expanded = self.prototypes.unsqueeze(0)

        # Squared L2 distance
        distances = torch.sum((z_expanded - proto_expanded) ** 2, dim=2)

        # Convert distances to similarities (closer = higher similarity)
        similarities = torch.log(1 + 1 / (distances + 1e-6))

        # Temperature-controlled nearest-prototype classification
        # Low temperature concentrates weight on nearest prototype, requiring
        # individual prototype specialization for correct classification.
        # NOTE: This improves alignment slightly (52.6% vs 50.4% baseline) but
        # does not achieve the >65% target due to prototype collapse.
        assignment_logits = similarities / self.classification_temperature
        assignment_weights = F.softmax(assignment_logits, dim=1)  # (batch, n_proto)

        # Class logits: weighted vote from prototypes
        logits = torch.matmul(assignment_weights, self.class_identity)

        return distances, similarities, logits

    def get_prototype_classes(self) -> torch.Tensor:
        """Get the class assignment for each prototype."""
        return self.prototype_class


class TempProtoNet(BaseTimeSeriesModel):
    """
    Temporal Prototype Network for explainable MG detection.

    Architecture:
        Input (batch, seq_len, input_dim)
            |
            v
        CNN Encoder -> Latent representation (batch, latent_dim)
            |
            v
        Prototype Layer -> Similarities (batch, n_prototypes)
            |
            v
        Classification -> Logits (batch, n_classes)

    Explainability:
        - Predictions are based on similarity to learned prototypes
        - Each prototype can be visualized via projection to nearest training sample
        - No handcrafted formulas - prototypes are learned representations
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
        classification_temperature: float = 0.1,
        **kwargs,
    ):
        """
        Args:
            input_dim: Number of input features per timestep
            num_classes: Number of output classes (2 for binary)
            seq_len: Expected sequence length
            latent_dim: Dimension of latent space for prototypes
            n_prototypes_per_class: Number of prototypes per class
            encoder_hidden: Hidden dimension in encoder
            encoder_layers: Number of encoder conv layers
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
            classification_temperature: Temperature for prototype assignment softmax.
                Low values (0.1) force nearest-prototype classification.
                High values (1.0+) approximate sum-of-similarities.
        """
        super().__init__(input_dim, num_classes, seq_len)

        self.latent_dim = latent_dim
        self.n_prototypes_per_class = n_prototypes_per_class
        self.encoder_hidden = encoder_hidden
        self.encoder_layers = encoder_layers
        self.kernel_size = kernel_size
        self.dropout_rate = dropout
        self.classification_temperature = classification_temperature

        # Build 1D CNN encoder
        self.encoder = self._build_encoder()

        # Prototype layer with temperature-controlled classification
        self.prototype_layer = PrototypeLayer(
            latent_dim=latent_dim,
            n_prototypes_per_class=n_prototypes_per_class,
            n_classes=num_classes,
            classification_temperature=classification_temperature,
        )

        # For tracking training sample embeddings (for prototype projection)
        self._training_embeddings: Optional[torch.Tensor] = None
        self._training_labels: Optional[torch.Tensor] = None
        self._training_indices: Optional[torch.Tensor] = None

    def _build_encoder(self) -> nn.Module:
        """Build the 1D CNN encoder."""
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

        # Final projection to latent space
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
        Encode input time series to latent representation.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            z: Latent representation (batch_size, latent_dim)
        """
        # Transpose for conv1d: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)

        # Encode: (batch_size, encoder_output_dim, reduced_seq_len)
        features = self.encoder(x)

        # Project to latent space: (batch_size, latent_dim)
        z = self.projection_head(features)

        # L2 normalize for better distance computation
        z = F.normalize(z, p=2, dim=1)

        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            logits: Class logits (batch_size, num_classes)
        """
        # Encode
        z = self.encode(x)

        # Compute similarities and logits
        distances, similarities, logits = self.prototype_layer(z)

        return logits

    def forward_with_explanations(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with full explanation outputs.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            logits: Class logits (batch_size, num_classes)
            z: Latent representation (batch_size, latent_dim)
            distances: Distances to prototypes (batch_size, n_prototypes)
            similarities: Similarities to prototypes (batch_size, n_prototypes)
        """
        z = self.encode(x)
        distances, similarities, logits = self.prototype_layer(z)
        return logits, z, distances, similarities

    def get_prototype_similarities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get similarity scores to all prototypes.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            similarities: (batch_size, n_prototypes)
        """
        z = self.encode(x)
        _, similarities, _ = self.prototype_layer(z)
        return similarities

    def get_prototypes(self) -> torch.Tensor:
        """Get the learned prototype vectors."""
        return self.prototype_layer.prototypes.detach()

    def get_prototype_classes(self) -> torch.Tensor:
        """Get the class assignment for each prototype."""
        return self.prototype_layer.prototype_class

    def compute_prototype_loss(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        distances: torch.Tensor,
        softmin_temperature: float = 1.0,
        margin: float = 0.3,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute prototype-specific losses for training.

        FIX for prototype collapse (PARTIAL): Uses softmin instead of min() to ensure
        ALL same-class prototypes receive gradients, not just the closest one.
        Also includes contrastive and diversity losses.

        NOTE: Despite multiple attempts (softmin, diversity, contrastive, purity,
        prototype initialization, max classification), alignment remains near baseline
        (~50-52%) due to fundamental architectural limitations. The classification
        mechanism using sum-of-similarities doesn't require individual prototype
        specialization. See PROTOTYPE_ALIGNMENT_INVESTIGATION.md for details.

        Args:
            z: Encoded representations (batch_size, latent_dim)
            labels: Ground truth labels (batch_size,)
            distances: Distances to prototypes (batch_size, n_prototypes)
            softmin_temperature: Temperature for softmin
            margin: Margin for contrastive loss

        Returns:
            cluster_loss, separation_loss, diversity_loss, contrastive_loss
        """
        batch_size = z.size(0)
        prototype_classes = self.prototype_layer.prototype_class
        n_prototypes_per_class = self.n_prototypes_per_class

        # Cluster loss: use SOFTMIN to distribute gradients to ALL same-class prototypes
        cluster_losses = []
        contrastive_losses = []

        for i in range(batch_size):
            label = labels[i].item()

            same_class_mask = (prototype_classes == label)
            diff_class_mask = (prototype_classes != label)

            same_class_distances = distances[i][same_class_mask]
            diff_class_distances = distances[i][diff_class_mask]

            # Cluster loss: weighted distance to same-class prototypes
            weights = F.softmax(-same_class_distances / softmin_temperature, dim=0)
            weighted_same_dist = (same_class_distances * weights).sum()
            cluster_losses.append(weighted_same_dist)

            # Contrastive loss
            min_same_dist = same_class_distances.min()
            min_diff_dist = diff_class_distances.min()
            contrastive = F.relu(min_same_dist + margin - min_diff_dist)
            contrastive_losses.append(contrastive)

        cluster_loss = torch.stack(cluster_losses).mean()
        contrastive_loss = torch.stack(contrastive_losses).mean()

        # Separation loss: prototypes of different classes should be far apart
        prototypes = self.prototype_layer.prototypes

        sep_losses = []
        for c in range(self.num_classes):
            class_c_mask = (prototype_classes == c)
            class_c_protos = prototypes[class_c_mask]

            other_class_mask = (prototype_classes != c)
            other_protos = prototypes[other_class_mask]

            if len(class_c_protos) > 0 and len(other_protos) > 0:
                dists = torch.cdist(class_c_protos, other_protos)
                min_dist = dists.min()
                sep_losses.append(-min_dist)

        if sep_losses:
            separation_loss = torch.stack(sep_losses).mean()
        else:
            separation_loss = torch.tensor(0.0, device=z.device)

        # Diversity loss: encourage samples to be distributed evenly across prototypes
        diversity_losses = []
        for c in range(self.num_classes):
            class_mask = (labels == c)
            if not class_mask.any():
                continue

            class_distances = distances[class_mask]
            proto_mask = (prototype_classes == c)
            class_proto_distances = class_distances[:, proto_mask]

            soft_assignments = F.softmax(-class_proto_distances / softmin_temperature, dim=1)
            avg_assignment = soft_assignments.mean(dim=0)
            uniform = torch.ones_like(avg_assignment) / n_prototypes_per_class
            kl_div = (avg_assignment * (torch.log(avg_assignment + 1e-8) - torch.log(uniform + 1e-8))).sum()
            diversity_losses.append(kl_div)

        if diversity_losses:
            diversity_loss = torch.stack(diversity_losses).mean()
        else:
            diversity_loss = torch.tensor(0.0, device=z.device)

        return cluster_loss, separation_loss, diversity_loss, contrastive_loss

    def store_training_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ):
        """
        Store training embeddings for prototype projection.

        Args:
            embeddings: Encoded training samples (n_samples, latent_dim)
            labels: Training labels (n_samples,)
            indices: Original indices in dataset (n_samples,)
        """
        self._training_embeddings = embeddings.detach()
        self._training_labels = labels.detach()
        self._training_indices = indices

    def initialize_prototypes_from_data(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        Initialize prototypes from actual class embeddings using k-means-like selection.

        This ensures prototypes start in class-specific regions of the latent space,
        improving alignment from the beginning.

        Args:
            embeddings: Encoded training samples (n_samples, latent_dim)
            labels: Training labels (n_samples,)
        """
        prototype_classes = self.prototype_layer.prototype_class
        n_prototypes_per_class = self.n_prototypes_per_class
        device = self.prototype_layer.prototypes.device

        with torch.no_grad():
            for c in range(self.num_classes):
                # Get embeddings for this class
                class_mask = (labels == c)
                class_embeddings = embeddings[class_mask]

                if len(class_embeddings) < n_prototypes_per_class:
                    continue

                # Use k-means++ style initialization: spread prototypes across class
                # Start with random sample as first prototype
                selected_indices = []
                selected = [torch.randint(len(class_embeddings), (1,)).item()]
                selected_indices.append(selected[0])

                for _ in range(n_prototypes_per_class - 1):
                    # Compute distances from each sample to nearest selected prototype
                    selected_embeds = class_embeddings[selected_indices]
                    dists = torch.cdist(class_embeddings, selected_embeds)
                    min_dists = dists.min(dim=1)[0]

                    # Select sample with max min-distance (farthest from any selected)
                    # Add small noise to break ties
                    min_dists = min_dists + torch.randn_like(min_dists) * 0.01
                    next_idx = min_dists.argmax().item()
                    selected_indices.append(next_idx)

                # Set prototypes for this class
                proto_start = c * n_prototypes_per_class
                for i, sample_idx in enumerate(selected_indices):
                    proto_idx = proto_start + i
                    self.prototype_layer.prototypes.data[proto_idx] = class_embeddings[sample_idx].to(device)

    def project_prototypes_to_class_centroids(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        momentum: float = 0.5,
    ):
        """
        Project each prototype toward the centroid of its k-nearest same-class samples.

        This helps maintain prototype alignment during training.

        Args:
            embeddings: Current training embeddings (n_samples, latent_dim)
            labels: Training labels (n_samples,)
            momentum: How much to move prototype toward centroid (0=no move, 1=full move)
        """
        prototype_classes = self.prototype_layer.prototype_class
        device = self.prototype_layer.prototypes.device
        k = 20  # Number of nearest neighbors to use for centroid

        with torch.no_grad():
            for proto_idx in range(self.prototype_layer.n_prototypes):
                proto_class = prototype_classes[proto_idx].item()
                proto = self.prototype_layer.prototypes[proto_idx:proto_idx+1]

                # Get embeddings for this class only
                class_mask = (labels == proto_class)
                class_embeddings = embeddings[class_mask].to(device)

                if len(class_embeddings) < k:
                    continue

                # Find k nearest same-class samples
                dists = torch.cdist(proto, class_embeddings).squeeze(0)
                _, topk_indices = torch.topk(dists, k, largest=False)

                # Compute centroid of nearest samples
                centroid = class_embeddings[topk_indices].mean(dim=0)

                # Move prototype toward centroid with momentum
                new_proto = (1 - momentum) * proto.squeeze(0) + momentum * centroid
                new_proto = F.normalize(new_proto, p=2, dim=0)  # Keep normalized
                self.prototype_layer.prototypes.data[proto_idx] = new_proto

    def project_prototypes(
        self,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Project prototypes to nearest training samples.

        This enables visualization: show what actual data each prototype resembles.

        Args:
            k: Number of nearest neighbors per prototype

        Returns:
            List of dicts, one per prototype, containing:
                - prototype_idx: Index of the prototype
                - prototype_class: Class this prototype belongs to
                - nearest_indices: Indices of k nearest training samples
                - nearest_distances: Distances to k nearest samples
                - nearest_labels: Labels of k nearest samples
        """
        if self._training_embeddings is None:
            raise ValueError("Training embeddings not stored. Call store_training_embeddings first.")

        prototypes = self.prototype_layer.prototypes.detach()
        prototype_classes = self.prototype_layer.prototype_class

        projections = []

        for proto_idx in range(prototypes.size(0)):
            proto = prototypes[proto_idx:proto_idx+1]  # (1, latent_dim)

            # Compute distances to all training samples
            # training_embeddings: (n_samples, latent_dim)
            dists = torch.cdist(proto, self._training_embeddings).squeeze(0)  # (n_samples,)

            # Get k nearest
            topk_dists, topk_indices = torch.topk(dists, k, largest=False)

            projection = {
                'prototype_idx': proto_idx,
                'prototype_class': prototype_classes[proto_idx].item(),
                'nearest_indices': topk_indices.cpu().numpy(),
                'nearest_distances': topk_dists.cpu().numpy(),
                'nearest_labels': self._training_labels[topk_indices].cpu().numpy(),
            }

            if self._training_indices is not None:
                projection['original_indices'] = self._training_indices[topk_indices].cpu().numpy()

            projections.append(projection)

        return projections

    def explain_prediction(
        self,
        x: torch.Tensor,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Generate explanation for predictions.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            top_k: Number of most similar prototypes to include

        Returns:
            List of explanations, one per sample in batch
        """
        with torch.no_grad():
            logits, z, distances, similarities = self.forward_with_explanations(x)
            predictions = torch.argmax(logits, dim=1)
            probabilities = F.softmax(logits, dim=1)

            prototype_classes = self.prototype_layer.prototype_class

            explanations = []

            for i in range(x.size(0)):
                pred = predictions[i].item()
                prob = probabilities[i, pred].item()
                sample_sims = similarities[i]

                # Get top-k most similar prototypes
                topk_sims, topk_indices = torch.topk(sample_sims, top_k)

                similar_prototypes = []
                for j in range(top_k):
                    proto_idx = topk_indices[j].item()
                    proto_class = prototype_classes[proto_idx].item()
                    sim = topk_sims[j].item()

                    similar_prototypes.append({
                        'prototype_idx': proto_idx,
                        'prototype_class': proto_class,
                        'prototype_class_name': 'MG' if proto_class == 1 else 'HC',
                        'similarity': sim,
                    })

                # Compute average similarity per class
                class_sims = {}
                for c in range(self.num_classes):
                    class_mask = (prototype_classes == c)
                    class_sims[c] = sample_sims[class_mask].mean().item()

                explanation = {
                    'prediction': pred,
                    'prediction_name': 'MG' if pred == 1 else 'HC',
                    'confidence': prob,
                    'top_similar_prototypes': similar_prototypes,
                    'avg_similarity_to_HC': class_sims[0],
                    'avg_similarity_to_MG': class_sims[1],
                }

                explanations.append(explanation)

            return explanations

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': 'TempProtoNet',
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'seq_len': self.seq_len,
            'latent_dim': self.latent_dim,
            'n_prototypes_per_class': self.n_prototypes_per_class,
            'encoder_hidden': self.encoder_hidden,
            'encoder_layers': self.encoder_layers,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout_rate,
            'classification_temperature': self.classification_temperature,
            'n_total_prototypes': self.prototype_layer.n_prototypes,
        }

    def compute_prototype_diversity(self) -> Dict[str, float]:
        """
        Compute metrics about prototype diversity.

        Returns:
            Dict containing:
                - mean_pairwise_distance: Average distance between all prototypes
                - min_pairwise_distance: Minimum distance between any two prototypes
                - inter_class_distance: Average distance between prototypes of different classes
                - intra_class_distance: Average distance between prototypes of same class
        """
        prototypes = self.prototype_layer.prototypes.detach()
        prototype_classes = self.prototype_layer.prototype_class

        # Compute all pairwise distances
        pairwise_dists = torch.cdist(prototypes, prototypes)

        # Mean and min (excluding diagonal)
        n = prototypes.size(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=prototypes.device)
        off_diagonal = pairwise_dists[mask]

        mean_dist = off_diagonal.mean().item()
        min_dist = off_diagonal.min().item()

        # Inter-class and intra-class distances
        inter_dists = []
        intra_dists = []

        for i in range(n):
            for j in range(i+1, n):
                dist = pairwise_dists[i, j].item()
                if prototype_classes[i] == prototype_classes[j]:
                    intra_dists.append(dist)
                else:
                    inter_dists.append(dist)

        inter_class_dist = np.mean(inter_dists) if inter_dists else 0.0
        intra_class_dist = np.mean(intra_dists) if intra_dists else 0.0

        return {
            'mean_pairwise_distance': mean_dist,
            'min_pairwise_distance': min_dist,
            'inter_class_distance': inter_class_dist,
            'intra_class_distance': intra_class_dist,
        }


# Default configuration
DEFAULT_CONFIG = {
    'latent_dim': 64,
    'n_prototypes_per_class': 5,
    'encoder_hidden': 64,
    'encoder_layers': 3,
    'kernel_size': 7,
    'dropout': 0.2,
    'classification_temperature': 0.1,  # Low temperature for nearest-prototype classification
}
