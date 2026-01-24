"""
Component 5: Case Study Analysis

Analyzes what distinguishes high-confidence correct, low-confidence, and misclassified samples.

Sample categories (5 samples each):
- High-confidence correct MG: Top 5 MG samples by confidence, correctly classified
- High-confidence correct HC: Top 5 HC samples by confidence, correctly classified
- Low-confidence correct: 5 samples with lowest confidence among correct
- Misclassified MG→HC: MG samples misclassified as HC
- Misclassified HC→MG: HC samples misclassified as MG

Metrics per sample:
- confidence: Softmax probability of predicted class
- margin: Difference between top-2 class probabilities
- distance_to_correct_prototype: Distance to correct-class prototype
- distance_to_wrong_prototype: Distance to wrong-class prototype
- distance_ratio: wrong/correct distance (>1 = correct prediction)
- head_agreement: Fraction of heads agreeing on prediction
- saliency_entropy: How focused is attention for this sample
- peak_saliency_time: Where in sequence is attention highest
- top_channels: Most important channels for this sample

Visualizations:
- Case study panels (signal, saliency, distances)
- Category comparison box plots
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from torch.utils.data import DataLoader
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class CaseStudyAnalyzer:
    """Analyzer for case study samples."""

    def __init__(
        self,
        model,
        device: torch.device,
        output_dir: str,
        quantitative_dir: str,
        n_per_category: int = 5,
    ):
        """
        Args:
            model: Trained MultiHeadProtoNet
            device: Computation device
            output_dir: Directory for figure output
            quantitative_dir: Directory for quantitative JSON output
            n_per_category: Number of samples per category
        """
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.quantitative_dir = quantitative_dir
        self.n_per_category = n_per_category
        self.n_heads = model.n_heads

    def compute_sample_metrics(
        self,
        input_tensor: torch.Tensor,
        true_label: int,
        sample_idx: int,
        patient_id: str,
    ) -> Dict[str, Any]:
        """
        Compute all metrics for a single sample.

        Args:
            input_tensor: (1, seq_len, features) input
            true_label: True label (0=HC, 1=MG)
            sample_idx: Sample index in dataset
            patient_id: Patient identifier

        Returns:
            Dict with all metrics
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            # Forward with explanations
            logits, z, all_distances, all_similarities = self.model.forward_with_explanations(input_tensor)

            # Get predictions
            probs = F.softmax(logits, dim=1)
            predicted_label = logits.argmax(dim=1).item()
            confidence = probs[0, predicted_label].item()
            margin = abs(probs[0, 1].item() - probs[0, 0].item())

            # Per-head predictions
            _, head_preds = self.model.forward_per_head(input_tensor)
            head_agreements = sum(1 for hp in head_preds if hp[0].item() == predicted_label)
            head_agreement_rate = head_agreements / self.n_heads

            # Distance metrics (average across heads)
            correct_distances = []
            wrong_distances = []

            for h_idx in range(self.n_heads):
                dist = all_distances[h_idx][0]  # (2,)
                correct_dist = dist[true_label].item()
                wrong_dist = dist[1 - true_label].item()
                correct_distances.append(correct_dist)
                wrong_distances.append(wrong_dist)

            mean_correct_dist = np.mean(correct_distances)
            mean_wrong_dist = np.mean(wrong_distances)
            distance_ratio = mean_wrong_dist / (mean_correct_dist + 1e-6)

        # Compute saliency (simplified - use gradient magnitude)
        input_tensor.requires_grad = True
        logits = self.model(input_tensor)
        self.model.zero_grad()
        logits[0, predicted_label].backward()
        saliency = input_tensor.grad.abs().cpu().numpy()[0]  # (seq_len, features)

        # Saliency metrics
        temporal_saliency = saliency.mean(axis=1)
        if temporal_saliency.sum() > 0:
            temporal_saliency = temporal_saliency / temporal_saliency.sum()

        # Entropy
        temporal_saliency_safe = temporal_saliency + 1e-10
        temporal_saliency_safe = temporal_saliency_safe / temporal_saliency_safe.sum()
        saliency_entropy = -np.sum(temporal_saliency_safe * np.log(temporal_saliency_safe))

        # Peak saliency time (normalized 0-1)
        peak_saliency_time = np.argmax(temporal_saliency) / len(temporal_saliency)

        # Top channels
        feature_importance = saliency.mean(axis=0)
        top_channel_indices = np.argsort(feature_importance)[::-1][:3]

        return {
            'sample_idx': sample_idx,
            'patient_id': patient_id,
            'true_label': true_label,
            'true_label_name': 'MG' if true_label == 1 else 'HC',
            'predicted_label': predicted_label,
            'predicted_label_name': 'MG' if predicted_label == 1 else 'HC',
            'correct': predicted_label == true_label,
            'confidence': float(confidence),
            'margin': float(margin),
            'distance_to_correct_proto': float(mean_correct_dist),
            'distance_to_wrong_proto': float(mean_wrong_dist),
            'distance_ratio': float(distance_ratio),
            'head_agreement': float(head_agreement_rate),
            'saliency_entropy': float(saliency_entropy),
            'peak_saliency_time': float(peak_saliency_time),
            'top_channel_indices': top_channel_indices.tolist(),
        }

    def select_samples(
        self,
        all_metrics: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Select samples for each category.

        Categories:
        - high_confidence_correct_mg: Top N MG by confidence, correct
        - high_confidence_correct_hc: Top N HC by confidence, correct
        - low_confidence_correct: Lowest N confidence among correct
        - misclassified_mg_to_hc: MG samples predicted as HC
        - misclassified_hc_to_mg: HC samples predicted as MG
        """
        categories = {
            'high_confidence_correct_mg': [],
            'high_confidence_correct_hc': [],
            'low_confidence_correct': [],
            'misclassified_mg_to_hc': [],
            'misclassified_hc_to_mg': [],
        }

        # Filter by category
        for m in all_metrics:
            if m['correct']:
                if m['true_label'] == 1:  # MG
                    categories['high_confidence_correct_mg'].append(m)
                else:  # HC
                    categories['high_confidence_correct_hc'].append(m)
            else:
                if m['true_label'] == 1:  # MG misclassified as HC
                    categories['misclassified_mg_to_hc'].append(m)
                else:  # HC misclassified as MG
                    categories['misclassified_hc_to_mg'].append(m)

        # Select top N by confidence for high confidence categories
        categories['high_confidence_correct_mg'] = sorted(
            categories['high_confidence_correct_mg'],
            key=lambda x: x['confidence'],
            reverse=True
        )[:self.n_per_category]

        categories['high_confidence_correct_hc'] = sorted(
            categories['high_confidence_correct_hc'],
            key=lambda x: x['confidence'],
            reverse=True
        )[:self.n_per_category]

        # Combine all correct for low confidence selection
        all_correct = (
            categories['high_confidence_correct_mg'] +
            categories['high_confidence_correct_hc']
        )
        categories['low_confidence_correct'] = sorted(
            [m for m in all_metrics if m['correct']],
            key=lambda x: x['confidence']
        )[:self.n_per_category]

        # Limit misclassified
        categories['misclassified_mg_to_hc'] = categories['misclassified_mg_to_hc'][:self.n_per_category]
        categories['misclassified_hc_to_mg'] = categories['misclassified_hc_to_mg'][:self.n_per_category]

        return categories

    def compute_category_statistics(
        self,
        categories: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compute aggregate statistics for each category.

        Returns:
            Dict mapping metric name to category statistics
        """
        metrics_to_aggregate = [
            'confidence', 'margin', 'distance_ratio', 'head_agreement',
            'saliency_entropy', 'peak_saliency_time'
        ]

        stats_result = {}

        for metric in metrics_to_aggregate:
            stats_result[metric] = {}

            for cat_name, samples in categories.items():
                if not samples:
                    stats_result[metric][cat_name] = {
                        'mean': 0.0,
                        'std': 0.0,
                        'min': 0.0,
                        'max': 0.0,
                    }
                    continue

                values = [s[metric] for s in samples]
                stats_result[metric][cat_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }

        return stats_result

    def generate_case_study_panels(
        self,
        categories: Dict[str, List[Dict[str, Any]]],
        val_items: List[Dict],
        channel_names: List[str]
    ):
        """Generate visualization panels for case studies."""
        # Generate summary panel for each category
        for cat_name, samples in categories.items():
            if not samples:
                continue

            n_samples = min(len(samples), 3)  # Show up to 3 samples

            fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4*n_samples))
            if n_samples == 1:
                axes = axes.reshape(1, -1)

            for i, sample in enumerate(samples[:n_samples]):
                idx = sample['sample_idx']
                item = val_items[idx]
                data = item['data']

                # Plot 1: Raw signal (first 4 channels - eye positions)
                ax1 = axes[i, 0]
                time = np.arange(data.shape[0]) / data.shape[0]
                for ch in range(4):
                    ax1.plot(time, data[:, ch], alpha=0.7, label=channel_names[ch])
                ax1.set_title(f'Eye Position (True: {sample["true_label_name"]}, Pred: {sample["predicted_label_name"]})')
                ax1.set_xlabel('Normalized Time')
                ax1.set_ylabel('Position')
                ax1.legend(fontsize=8)

                # Plot 2: Confidence bar
                ax2 = axes[i, 1]
                ax2.barh(['Confidence', 'Head Agree', 'Dist Ratio'],
                        [sample['confidence'], sample['head_agreement'], min(sample['distance_ratio'], 5)/5],
                        color=['steelblue', 'forestgreen', 'coral'])
                ax2.set_xlim(0, 1)
                ax2.set_title(f'Metrics (Conf: {sample["confidence"]:.2f})')

                # Add text values
                for j, val in enumerate([sample['confidence'], sample['head_agreement'],
                                        sample['distance_ratio']]):
                    ax2.text(0.95, j, f'{val:.2f}', va='center', ha='right', fontsize=10)

                # Plot 3: Distance comparison
                ax3 = axes[i, 2]
                ax3.bar(['Correct\nProto', 'Wrong\nProto'],
                       [sample['distance_to_correct_proto'], sample['distance_to_wrong_proto']],
                       color=['green', 'red'])
                ax3.set_title(f'Prototype Distances (Ratio: {sample["distance_ratio"]:.2f})')
                ax3.set_ylabel('L2 Distance')

            plt.suptitle(f'Case Studies: {cat_name.replace("_", " ").title()}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'fig_case_study_{cat_name}.pdf'),
                       dpi=150, bbox_inches='tight')
            plt.close()

    def generate_category_comparison(
        self,
        category_stats: Dict[str, Dict[str, Dict[str, float]]]
    ):
        """Generate box plots comparing categories."""
        metrics_to_plot = ['confidence', 'head_agreement', 'distance_ratio', 'saliency_entropy']
        category_names = ['high_confidence_correct_mg', 'high_confidence_correct_hc',
                         'low_confidence_correct', 'misclassified_mg_to_hc', 'misclassified_hc_to_mg']
        short_names = ['HiConf MG', 'HiConf HC', 'LoConf', 'MG→HC', 'HC→MG']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]

            means = []
            stds = []
            for cat in category_names:
                if cat in category_stats[metric]:
                    means.append(category_stats[metric][cat]['mean'])
                    stds.append(category_stats[metric][cat]['std'])
                else:
                    means.append(0)
                    stds.append(0)

            x = np.arange(len(short_names))
            colors = ['#2ecc71', '#27ae60', '#f39c12', '#e74c3c', '#c0392b']

            bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(short_names, rotation=15, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} by Category')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig_category_comparison.pdf'), dpi=150, bbox_inches='tight')
        plt.close()

    def run_analysis(
        self,
        val_loader: DataLoader,
        val_items: List[Dict],
        channel_names: List[str]
    ) -> Dict[str, Any]:
        """
        Run the complete case study analysis.

        Args:
            val_loader: Validation data loader
            val_items: Original validation items (for patient IDs)
            channel_names: Names of channels

        Returns:
            Dict with all analysis results
        """
        print("  Computing sample metrics...")

        all_metrics = []
        sample_idx = 0

        for inputs, labels in val_loader:
            for i in range(inputs.size(0)):
                input_tensor = inputs[i:i+1]
                true_label = labels[i].item()
                patient_id = val_items[sample_idx]['patient_id']

                metrics = self.compute_sample_metrics(
                    input_tensor, true_label, sample_idx, patient_id
                )

                all_metrics.append(metrics)
                sample_idx += 1

                if sample_idx % 50 == 0:
                    print(f"    Processed {sample_idx} samples...")

        print(f"  Total samples: {len(all_metrics)}")

        # Select samples for each category
        print("  Selecting case study samples...")
        categories = self.select_samples(all_metrics)

        # Add channel names to samples
        for cat_samples in categories.values():
            for sample in cat_samples:
                sample['top_channels'] = [
                    channel_names[i] for i in sample['top_channel_indices']
                ]

        # Compute category statistics
        print("  Computing category statistics...")
        category_stats = self.compute_category_statistics(categories)

        # Generate visualizations
        print("  Generating visualizations...")
        self.generate_case_study_panels(categories, val_items, channel_names)
        self.generate_category_comparison(category_stats)

        # Compile results
        results = {
            **categories,
            'category_comparison': category_stats,
            'total_samples': len(all_metrics),
            'n_correct': sum(1 for m in all_metrics if m['correct']),
            'n_incorrect': sum(1 for m in all_metrics if not m['correct']),
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

        with open(os.path.join(self.quantitative_dir, 'case_studies.json'), 'w') as f:
            json.dump(convert_to_serializable(results), f, indent=2)

        return results
