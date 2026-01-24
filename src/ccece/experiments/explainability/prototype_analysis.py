"""
Component 1: Prototype Analysis

Analyzes whether learned prototypes are meaningful class representations.

Metrics computed:
- Inter-class distance: Mean distance between HC and MG prototypes
- Intra-class distance: Mean distance within same class
- Separability ratio: inter/intra distance ratio
- Per-head separability: Separability for each of 5 heads
- Prototype norm variance: Variance in prototype L2 norms

Visualizations:
- Prototype embeddings t-SNE
- Prototype distance matrix heatmap
- Per-head prototype quality bar chart
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


class PrototypeAnalyzer:
    """Analyzer for prototype quality and separability."""

    def __init__(
        self,
        model,
        device: torch.device,
        output_dir: str,
        quantitative_dir: str,
    ):
        """
        Args:
            model: Trained MultiHeadProtoNet
            device: Computation device
            output_dir: Directory for figure output
            quantitative_dir: Directory for quantitative JSON output
        """
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.quantitative_dir = quantitative_dir
        self.model.eval()

    def extract_prototypes(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract all prototypes from the model.

        Returns:
            prototypes: (n_heads * 2, head_dim) array of prototype vectors
            labels: (n_heads * 2,) array of class labels (0=HC, 1=MG)
        """
        prototypes = []
        labels = []

        with torch.no_grad():
            for head_idx, head in enumerate(self.model.heads):
                # Get prototypes for this head
                proto = head.prototypes.cpu().numpy()  # (2, head_dim)
                prototypes.append(proto[0])  # HC prototype
                prototypes.append(proto[1])  # MG prototype
                labels.extend([0, 1])  # HC, MG

        return np.array(prototypes), np.array(labels)

    def compute_distance_matrix(
        self,
        prototypes: np.ndarray
    ) -> np.ndarray:
        """Compute pairwise L2 distance matrix between prototypes."""
        n = len(prototypes)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(prototypes[i] - prototypes[j])

        return distances

    def compute_separability_metrics(
        self,
        prototypes: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute separability metrics for prototypes.

        Returns:
            Dict with inter_class_distance, intra_class_distance, separability_ratio
        """
        distance_matrix = self.compute_distance_matrix(prototypes)

        # Inter-class distances (HC-MG pairs)
        inter_class_dists = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if labels[i] != labels[j]:
                    inter_class_dists.append(distance_matrix[i, j])

        # Intra-class distances (same class pairs)
        intra_class_dists = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if labels[i] == labels[j]:
                    intra_class_dists.append(distance_matrix[i, j])

        inter_class_distance = np.mean(inter_class_dists) if inter_class_dists else 0.0
        intra_class_distance = np.mean(intra_class_dists) if intra_class_dists else 0.0

        # Avoid division by zero
        if intra_class_distance > 0:
            separability_ratio = inter_class_distance / intra_class_distance
        else:
            separability_ratio = float('inf') if inter_class_distance > 0 else 1.0

        return {
            'inter_class_distance': float(inter_class_distance),
            'intra_class_distance': float(intra_class_distance),
            'separability_ratio': float(separability_ratio),
            'mean_same_class': float(intra_class_distance),
            'mean_diff_class': float(inter_class_distance),
            'min_diff_class': float(min(inter_class_dists)) if inter_class_dists else 0.0,
        }

    def compute_per_head_quality(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute quality metrics for each head.

        Args:
            embeddings: Training embeddings (n_samples, latent_dim)
            labels: Training labels (n_samples,)

        Returns:
            Dict mapping head_idx to quality metrics
        """
        results = {}

        embeddings = embeddings.to(self.device)
        labels = labels.to(self.device)

        with torch.no_grad():
            for head_idx, head in enumerate(self.model.heads):
                # Get distances to prototypes
                distances, _, _ = head(embeddings)  # (n_samples, 2)

                # Compute separability for this head
                proto = head.prototypes.cpu().numpy()  # (2, head_dim)
                proto_dist = np.linalg.norm(proto[0] - proto[1])

                # Compute alignment
                nearest_proto = distances.argmin(dim=1)
                alignment = (nearest_proto == labels).float().mean().item()

                # Per-class alignment
                hc_mask = (labels == 0)
                mg_mask = (labels == 1)
                hc_alignment = (nearest_proto[hc_mask] == 0).float().mean().item() if hc_mask.any() else 0.0
                mg_alignment = (nearest_proto[mg_mask] == 1).float().mean().item() if mg_mask.any() else 0.0

                # Prototype norms
                hc_proto_norm = float(np.linalg.norm(proto[0]))
                mg_proto_norm = float(np.linalg.norm(proto[1]))

                results[f'head_{head_idx}'] = {
                    'separability': float(proto_dist),
                    'alignment': float(alignment),
                    'hc_alignment': float(hc_alignment),
                    'mg_alignment': float(mg_alignment),
                    'hc_prototype_norm': hc_proto_norm,
                    'mg_prototype_norm': mg_proto_norm,
                }

        return results

    def compute_sample_to_prototype_distances(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Compute statistics on sample-to-prototype distances.

        Returns distances to correct and incorrect prototypes.
        """
        embeddings = embeddings.to(self.device)
        labels = labels.to(self.device)

        all_correct_dists = []
        all_incorrect_dists = []

        with torch.no_grad():
            for head in self.model.heads:
                distances, _, _ = head(embeddings)  # (n_samples, 2)

                for i in range(len(labels)):
                    label = labels[i].item()
                    correct_dist = distances[i, label].item()
                    incorrect_dist = distances[i, 1 - label].item()

                    all_correct_dists.append(correct_dist)
                    all_incorrect_dists.append(incorrect_dist)

        return {
            'mean_correct_distance': float(np.mean(all_correct_dists)),
            'std_correct_distance': float(np.std(all_correct_dists)),
            'mean_incorrect_distance': float(np.mean(all_incorrect_dists)),
            'std_incorrect_distance': float(np.std(all_incorrect_dists)),
            'distance_ratio': float(np.mean(all_incorrect_dists) / (np.mean(all_correct_dists) + 1e-6)),
        }

    def generate_tsne_visualization(
        self,
        prototypes: np.ndarray,
        labels: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
        embedding_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Generate t-SNE visualization of prototypes and optionally sample embeddings.

        Returns quantitative metrics.
        """
        if embeddings is not None:
            # Combine prototypes and sample embeddings
            all_data = np.vstack([prototypes, embeddings])
            all_labels = np.concatenate([labels, embedding_labels])
            is_prototype = np.array([True] * len(prototypes) + [False] * len(embeddings))
        else:
            all_data = prototypes
            all_labels = labels
            is_prototype = np.ones(len(prototypes), dtype=bool)

        # Fit t-SNE
        perplexity = min(30, len(all_data) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embedded = tsne.fit_transform(all_data)

        # Separate prototypes and samples
        proto_embedded = embedded[is_prototype]
        proto_labels = all_labels[is_prototype]

        # Compute silhouette score for prototypes
        if len(proto_embedded) > 2:
            silhouette = silhouette_score(proto_embedded, proto_labels)
        else:
            silhouette = 0.0

        # Compute centroid distances
        hc_centroid = proto_embedded[proto_labels == 0].mean(axis=0)
        mg_centroid = proto_embedded[proto_labels == 1].mean(axis=0)
        centroid_distance = np.linalg.norm(hc_centroid - mg_centroid)

        # Compute within-class spread
        hc_spread = np.std(proto_embedded[proto_labels == 0])
        mg_spread = np.std(proto_embedded[proto_labels == 1])

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot samples if available
        if embeddings is not None:
            sample_embedded = embedded[~is_prototype]
            sample_labels = all_labels[~is_prototype]

            ax.scatter(
                sample_embedded[sample_labels == 0, 0],
                sample_embedded[sample_labels == 0, 1],
                c='lightblue', alpha=0.3, s=20, label='HC samples'
            )
            ax.scatter(
                sample_embedded[sample_labels == 1, 0],
                sample_embedded[sample_labels == 1, 1],
                c='lightsalmon', alpha=0.3, s=20, label='MG samples'
            )

        # Plot prototypes
        for i in range(len(proto_embedded)):
            head_idx = i // 2
            color = 'blue' if proto_labels[i] == 0 else 'red'
            marker = 'o' if proto_labels[i] == 0 else 's'
            label = f'HC Proto' if proto_labels[i] == 0 else 'MG Proto'
            if i < 2:  # Only label first of each type
                ax.scatter(
                    proto_embedded[i, 0],
                    proto_embedded[i, 1],
                    c=color, s=200, marker=marker, edgecolors='black',
                    linewidths=2, label=label, zorder=10
                )
            else:
                ax.scatter(
                    proto_embedded[i, 0],
                    proto_embedded[i, 1],
                    c=color, s=200, marker=marker, edgecolors='black',
                    linewidths=2, zorder=10
                )
            ax.annotate(f'H{head_idx}', (proto_embedded[i, 0], proto_embedded[i, 1]),
                       fontsize=8, ha='center', va='center', color='white', fontweight='bold')

        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title('Prototype Embeddings (t-SNE)')
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig_prototype_tsne.pdf'), dpi=150, bbox_inches='tight')
        plt.close()

        return {
            'cluster_silhouette_score': float(silhouette),
            'inter_class_centroid_distance': float(centroid_distance),
            'intra_class_spread_hc': float(hc_spread),
            'intra_class_spread_mg': float(mg_spread),
        }

    def generate_distance_matrix_heatmap(
        self,
        distance_matrix: np.ndarray,
        n_heads: int
    ):
        """Generate heatmap of pairwise prototype distances."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create labels
        labels = []
        for h in range(n_heads):
            labels.extend([f'H{h}_HC', f'H{h}_MG'])

        im = ax.imshow(distance_matrix, cmap='viridis')

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('L2 Distance', rotation=-90, va='bottom')

        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{distance_matrix[i, j]:.2f}',
                              ha='center', va='center', color='white' if distance_matrix[i, j] > distance_matrix.max()/2 else 'black',
                              fontsize=8)

        ax.set_title('Prototype Distance Matrix')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig_prototype_distances.pdf'), dpi=150, bbox_inches='tight')
        plt.close()

    def generate_per_head_quality_chart(
        self,
        per_head_quality: Dict[str, Dict[str, float]],
        n_heads: int
    ):
        """Generate bar chart of per-head quality metrics."""
        heads = [f'Head {i}' for i in range(n_heads)]
        separabilities = [per_head_quality[f'head_{i}']['separability'] for i in range(n_heads)]
        alignments = [per_head_quality[f'head_{i}']['alignment'] for i in range(n_heads)]

        x = np.arange(len(heads))
        width = 0.35

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Separability chart
        bars1 = ax1.bar(x, separabilities, width, color='steelblue')
        ax1.set_ylabel('Prototype Separation (L2 Distance)')
        ax1.set_xlabel('Head')
        ax1.set_title('Per-Head Prototype Separation')
        ax1.set_xticks(x)
        ax1.set_xticklabels(heads)
        ax1.axhline(y=np.mean(separabilities), color='red', linestyle='--', label=f'Mean: {np.mean(separabilities):.2f}')
        ax1.legend()

        # Alignment chart
        bars2 = ax2.bar(x, alignments, width, color='forestgreen')
        ax2.set_ylabel('Alignment (Accuracy)')
        ax2.set_xlabel('Head')
        ax2.set_title('Per-Head Alignment')
        ax2.set_xticks(x)
        ax2.set_xticklabels(heads)
        ax2.axhline(y=0.65, color='red', linestyle='--', label='Threshold (65%)')
        ax2.axhline(y=np.mean(alignments), color='orange', linestyle='--', label=f'Mean: {np.mean(alignments):.2%}')
        ax2.set_ylim(0, 1)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig_per_head_quality.pdf'), dpi=150, bbox_inches='tight')
        plt.close()

    def run_analysis(
        self,
        train_loader: DataLoader
    ) -> Dict[str, Any]:
        """
        Run the complete prototype analysis.

        Returns:
            Dict with all analysis results
        """
        # Get training embeddings from model
        if self.model._training_embeddings is None:
            # Collect embeddings
            all_embeddings = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels in train_loader:
                    inputs = inputs.to(self.device)
                    z = self.model.encode(inputs)
                    all_embeddings.append(z.cpu())
                    all_labels.append(labels)
            embeddings = torch.cat(all_embeddings, dim=0)
            labels = torch.cat(all_labels, dim=0)
            self.model.store_training_embeddings(embeddings, labels)
        else:
            embeddings = self.model._training_embeddings
            labels = self.model._training_labels

        # Extract prototypes
        prototypes, proto_labels = self.extract_prototypes()

        # Compute distance matrix
        distance_matrix = self.compute_distance_matrix(prototypes)

        # Compute separability metrics
        separability_metrics = self.compute_separability_metrics(prototypes, proto_labels)

        # Compute per-head quality
        per_head_quality = self.compute_per_head_quality(embeddings, labels)

        # Compute sample-to-prototype distances
        sample_distances = self.compute_sample_to_prototype_distances(embeddings, labels)

        # Compute prototype norm variance
        norms = np.linalg.norm(prototypes, axis=1)
        norm_variance = float(np.var(norms))

        # Generate visualizations
        # For t-SNE, get head embeddings for a sample of training data
        sample_idx = np.random.choice(len(embeddings), min(200, len(embeddings)), replace=False)
        sample_embeddings = embeddings[sample_idx]
        sample_labels_subset = labels[sample_idx]

        # Get head embeddings for samples
        head_embeddings = []
        with torch.no_grad():
            sample_embeddings_device = sample_embeddings.to(self.device)
            for head in self.model.heads:
                h = head.get_head_embedding(sample_embeddings_device)
                head_embeddings.append(h.cpu().numpy())

        # Use first head's embeddings for t-SNE
        first_head_embeddings = head_embeddings[0]

        tsne_metrics = self.generate_tsne_visualization(
            prototypes, proto_labels,
            first_head_embeddings, sample_labels_subset.numpy()
        )

        self.generate_distance_matrix_heatmap(distance_matrix, self.model.n_heads)
        self.generate_per_head_quality_chart(per_head_quality, self.model.n_heads)

        # Compile results
        results = {
            **separability_metrics,
            'per_head_quality': per_head_quality,
            'sample_distances': sample_distances,
            'prototype_norm_variance': norm_variance,
            'tsne_metrics': tsne_metrics,
            'distance_matrix': distance_matrix.tolist(),
        }

        # Save quantitative results
        with open(os.path.join(self.quantitative_dir, 'prototype_analysis.json'), 'w') as f:
            json.dump(results, f, indent=2)

        return results
