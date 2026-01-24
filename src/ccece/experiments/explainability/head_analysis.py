"""
Component 4: Head Specialization Analysis

Analyzes whether the 5 prototype heads specialize in different aspects.

Metrics computed:
- Head agreement matrix: Pairwise agreement rate between heads
- Head accuracy: Per-head classification accuracy
- Head confidence: Mean confidence (max softmax) per head
- Head disagreement rate: How often heads disagree on prediction
- Head mutual information: MI between head predictions
- Head feature sensitivity: Which features each head is most sensitive to

Visualizations:
- Head agreement heatmap
- Head performance comparison bar chart
- Head feature sensitivity heatmap
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, mutual_info_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class HeadSpecializationAnalyzer:
    """Analyzer for head specialization and agreement patterns."""

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
        self.n_heads = model.n_heads

    def get_per_head_predictions(
        self,
        val_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions from each head.

        Returns:
            head_predictions: (n_samples, n_heads) predictions per head
            head_confidences: (n_samples, n_heads) confidence per head
            true_labels: (n_samples,) true labels
        """
        self.model.eval()

        all_head_preds = []
        all_head_confs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)

                # Get per-head predictions
                head_logits, head_preds = self.model.forward_per_head(inputs)

                batch_head_preds = []
                batch_head_confs = []

                for h_idx in range(self.n_heads):
                    preds = head_preds[h_idx].cpu().numpy()
                    logits = head_logits[h_idx]
                    probs = F.softmax(logits, dim=1)
                    confs = probs.max(dim=1).values.cpu().numpy()

                    batch_head_preds.append(preds)
                    batch_head_confs.append(confs)

                # Stack: (batch, n_heads)
                batch_head_preds = np.stack(batch_head_preds, axis=1)
                batch_head_confs = np.stack(batch_head_confs, axis=1)

                all_head_preds.append(batch_head_preds)
                all_head_confs.append(batch_head_confs)
                all_labels.append(labels.numpy())

        head_predictions = np.vstack(all_head_preds)
        head_confidences = np.vstack(all_head_confs)
        true_labels = np.concatenate(all_labels)

        return head_predictions, head_confidences, true_labels

    def compute_agreement_matrix(
        self,
        head_predictions: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise agreement rate between heads.

        Args:
            head_predictions: (n_samples, n_heads) predictions

        Returns:
            agreement_matrix: (n_heads, n_heads) pairwise agreement rates
        """
        n_heads = head_predictions.shape[1]
        agreement_matrix = np.zeros((n_heads, n_heads))

        for i in range(n_heads):
            for j in range(n_heads):
                agreement = (head_predictions[:, i] == head_predictions[:, j]).mean()
                agreement_matrix[i, j] = agreement

        return agreement_matrix

    def compute_head_performance(
        self,
        head_predictions: np.ndarray,
        head_confidences: np.ndarray,
        true_labels: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-head performance metrics.

        Returns:
            Dict mapping head name to performance metrics
        """
        performance = {}

        for h_idx in range(self.n_heads):
            preds = head_predictions[:, h_idx]
            confs = head_confidences[:, h_idx]

            accuracy = accuracy_score(true_labels, preds)

            # Correct predictions confidence
            correct_mask = (preds == true_labels)
            correct_conf = confs[correct_mask].mean() if correct_mask.any() else 0.0

            # Wrong predictions confidence
            wrong_mask = (preds != true_labels)
            wrong_conf = confs[wrong_mask].mean() if wrong_mask.any() else 0.0

            performance[f'head_{h_idx}'] = {
                'accuracy': float(accuracy),
                'confidence_mean': float(confs.mean()),
                'confidence_std': float(confs.std()),
                'correct_confidence': float(correct_conf),
                'wrong_confidence': float(wrong_conf),
            }

        return performance

    def compute_disagreement_rate(
        self,
        head_predictions: np.ndarray
    ) -> float:
        """Compute fraction of samples where not all heads agree."""
        n_samples = head_predictions.shape[0]
        disagreements = 0

        for i in range(n_samples):
            if len(set(head_predictions[i, :])) > 1:
                disagreements += 1

        return float(disagreements / n_samples)

    def compute_mutual_information(
        self,
        head_predictions: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise mutual information between head predictions.

        Returns:
            mi_matrix: (n_heads, n_heads) mutual information values
        """
        n_heads = head_predictions.shape[1]
        mi_matrix = np.zeros((n_heads, n_heads))

        for i in range(n_heads):
            for j in range(n_heads):
                mi = mutual_info_score(head_predictions[:, i], head_predictions[:, j])
                mi_matrix[i, j] = mi

        return mi_matrix

    def compute_feature_sensitivity(
        self,
        val_loader: DataLoader,
        channel_names: List[str],
        n_repeats: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute feature sensitivity for each head using permutation.

        Args:
            val_loader: Validation data loader
            channel_names: Names of channels
            n_repeats: Number of permutation repeats

        Returns:
            Dict mapping head name to sensitivity info
        """
        self.model.eval()

        # Get baseline per-head predictions
        baseline_preds, _, true_labels = self.get_per_head_predictions(val_loader)

        baseline_accs = []
        for h_idx in range(self.n_heads):
            acc = accuracy_score(true_labels, baseline_preds[:, h_idx])
            baseline_accs.append(acc)

        # Compute sensitivity for each head and channel
        sensitivity = {f'head_{h}': {'sensitivity_vector': [], 'top_channels': []}
                      for h in range(self.n_heads)}

        for ch_idx, ch_name in enumerate(channel_names):
            drops_per_head = [[] for _ in range(self.n_heads)]

            for _ in range(n_repeats):
                # Get predictions with permuted channel
                permuted_preds = self._get_permuted_head_predictions(val_loader, ch_idx)

                for h_idx in range(self.n_heads):
                    perm_acc = accuracy_score(true_labels, permuted_preds[:, h_idx])
                    drop = baseline_accs[h_idx] - perm_acc
                    drops_per_head[h_idx].append(drop)

            # Store mean drop for each head
            for h_idx in range(self.n_heads):
                mean_drop = np.mean(drops_per_head[h_idx])
                sensitivity[f'head_{h_idx}']['sensitivity_vector'].append(float(mean_drop))

        # Find top channels for each head
        for h_idx in range(self.n_heads):
            sens_vector = sensitivity[f'head_{h_idx}']['sensitivity_vector']
            sorted_idx = np.argsort(sens_vector)[::-1]
            top_channels = [channel_names[i] for i in sorted_idx[:3]]
            sensitivity[f'head_{h_idx}']['top_channels'] = top_channels

        return sensitivity

    def _get_permuted_head_predictions(
        self,
        val_loader: DataLoader,
        feature_idx: int
    ) -> np.ndarray:
        """Get per-head predictions with a feature permuted."""
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                # Permute feature
                inputs_perm = inputs.clone()
                perm_idx = torch.randperm(inputs.size(0))
                inputs_perm[:, :, feature_idx] = inputs[perm_idx, :, feature_idx]
                inputs_perm = inputs_perm.to(self.device)

                _, head_preds = self.model.forward_per_head(inputs_perm)

                batch_preds = []
                for h_idx in range(self.n_heads):
                    batch_preds.append(head_preds[h_idx].cpu().numpy())

                all_preds.append(np.stack(batch_preds, axis=1))

        return np.vstack(all_preds)

    def generate_agreement_heatmap(
        self,
        agreement_matrix: np.ndarray
    ):
        """Generate heatmap of head agreement."""
        fig, ax = plt.subplots(figsize=(8, 7))

        labels = [f'Head {i}' for i in range(self.n_heads)]

        im = ax.imshow(agreement_matrix, cmap='YlGnBu', vmin=0.5, vmax=1.0)

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Agreement Rate', rotation=-90, va='bottom')

        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                color = 'white' if agreement_matrix[i, j] > 0.75 else 'black'
                ax.text(j, i, f'{agreement_matrix[i, j]:.2f}',
                       ha='center', va='center', color=color, fontsize=11)

        ax.set_title('Head Agreement Matrix')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig_head_agreement.pdf'), dpi=150, bbox_inches='tight')
        plt.close()

    def generate_performance_chart(
        self,
        head_performance: Dict[str, Dict[str, float]]
    ):
        """Generate bar chart of per-head performance."""
        heads = [f'Head {i}' for i in range(self.n_heads)]
        accuracies = [head_performance[f'head_{i}']['accuracy'] for i in range(self.n_heads)]
        confidences = [head_performance[f'head_{i}']['confidence_mean'] for i in range(self.n_heads)]

        x = np.arange(len(heads))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
        bars2 = ax.bar(x + width/2, confidences, width, label='Confidence', color='coral')

        ax.set_ylabel('Score')
        ax.set_xlabel('Head')
        ax.set_title('Per-Head Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(heads)
        ax.legend()
        ax.set_ylim(0, 1)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig_head_performance.pdf'), dpi=150, bbox_inches='tight')
        plt.close()

    def generate_feature_sensitivity_heatmap(
        self,
        sensitivity: Dict[str, Dict[str, Any]],
        channel_names: List[str]
    ):
        """Generate heatmap of per-head feature sensitivity."""
        # Build matrix: (n_heads, n_channels)
        matrix = np.array([
            sensitivity[f'head_{h}']['sensitivity_vector']
            for h in range(self.n_heads)
        ])

        fig, ax = plt.subplots(figsize=(14, 6))

        im = ax.imshow(matrix, cmap='Reds', aspect='auto')

        ax.set_xticks(np.arange(len(channel_names)))
        ax.set_yticks(np.arange(self.n_heads))
        ax.set_xticklabels(channel_names, rotation=45, ha='right')
        ax.set_yticklabels([f'Head {i}' for i in range(self.n_heads)])

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Sensitivity (Accuracy Drop)', rotation=-90, va='bottom')

        ax.set_title('Head Feature Sensitivity')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig_head_feature_sensitivity.pdf'), dpi=150, bbox_inches='tight')
        plt.close()

    def run_analysis(
        self,
        val_loader: DataLoader,
        channel_names: List[str]
    ) -> Dict[str, Any]:
        """
        Run the complete head specialization analysis.

        Args:
            val_loader: Validation data loader
            channel_names: Names of channels

        Returns:
            Dict with all analysis results
        """
        print("  Getting per-head predictions...")

        # Get predictions
        head_predictions, head_confidences, true_labels = self.get_per_head_predictions(val_loader)

        # Compute agreement matrix
        print("  Computing agreement matrix...")
        agreement_matrix = self.compute_agreement_matrix(head_predictions)

        # Compute head performance
        print("  Computing head performance...")
        head_performance = self.compute_head_performance(
            head_predictions, head_confidences, true_labels
        )

        # Compute disagreement rate
        disagreement_rate = self.compute_disagreement_rate(head_predictions)

        # Compute mutual information
        print("  Computing mutual information...")
        mi_matrix = self.compute_mutual_information(head_predictions)

        # Compute feature sensitivity
        print("  Computing feature sensitivity...")
        feature_sensitivity = self.compute_feature_sensitivity(val_loader, channel_names)

        # Generate visualizations
        print("  Generating visualizations...")
        self.generate_agreement_heatmap(agreement_matrix)
        self.generate_performance_chart(head_performance)
        self.generate_feature_sensitivity_heatmap(feature_sensitivity, channel_names)

        # Compile statistics
        agreement_off_diag = agreement_matrix[np.triu_indices(self.n_heads, k=1)]

        results = {
            'agreement_matrix': agreement_matrix.tolist(),
            'mean_pairwise_agreement': float(agreement_off_diag.mean()),
            'min_agreement': float(agreement_off_diag.min()),
            'max_agreement': float(agreement_off_diag.max()),
            'disagreement_rate': float(disagreement_rate),
            'head_performance': head_performance,
            'mutual_information_matrix': mi_matrix.tolist(),
            'mean_mutual_information': float(mi_matrix[np.triu_indices(self.n_heads, k=1)].mean()),
            'feature_sensitivity': feature_sensitivity,
        }

        # Save quantitative results
        def convert_to_serializable(obj):
            """Convert numpy types to Python native types."""
            if isinstance(obj, (np.floating, float)):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj

        with open(os.path.join(self.quantitative_dir, 'head_analysis.json'), 'w') as f:
            json.dump(convert_to_serializable(results), f, indent=2)

        return results
