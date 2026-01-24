"""
Component 3: Feature (Channel) Importance Analysis

Analyzes which of the 14 channels are most important for classification.

Method: Permutation Importance - model-agnostic, reliable method that shuffles
each channel independently and measures accuracy drop.

Metrics computed:
- Feature ranking with 95% confidence intervals
- Top 3 cumulative importance
- Importance Gini coefficient (inequality)
- Category-level importance (velocity, error, position, target)
- Statistical tests: velocity vs position, error vs raw

Visualizations:
- Feature importance bar chart with error bars
- Feature importance by category grouped bar chart
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from torch.utils.data import DataLoader
from scipy import stats
from sklearn.metrics import accuracy_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureImportanceAnalyzer:
    """Analyzer for feature importance using permutation importance."""

    def __init__(
        self,
        model,
        device: torch.device,
        output_dir: str,
        quantitative_dir: str,
        n_repeats: int = 10,
    ):
        """
        Args:
            model: Trained MultiHeadProtoNet
            device: Computation device
            output_dir: Directory for figure output
            quantitative_dir: Directory for quantitative JSON output
            n_repeats: Number of permutation repeats per feature
        """
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.quantitative_dir = quantitative_dir
        self.n_repeats = n_repeats

    def compute_baseline_accuracy(
        self,
        val_loader: DataLoader
    ) -> float:
        """Compute baseline accuracy on validation data."""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        return accuracy_score(all_labels, all_preds)

    def compute_permuted_accuracy(
        self,
        val_loader: DataLoader,
        feature_idx: int
    ) -> float:
        """Compute accuracy with a single feature permuted."""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                # Clone and permute the feature
                inputs_permuted = inputs.clone()

                # Permute across the batch dimension
                batch_size = inputs.size(0)
                perm_idx = torch.randperm(batch_size)
                inputs_permuted[:, :, feature_idx] = inputs[perm_idx, :, feature_idx]

                inputs_permuted = inputs_permuted.to(self.device)
                outputs = self.model(inputs_permuted)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        return accuracy_score(all_labels, all_preds)

    def compute_permutation_importance(
        self,
        val_loader: DataLoader,
        channel_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute permutation importance for all features.

        Returns:
            Dict mapping channel names to importance stats (mean, std, ci_low, ci_high)
        """
        baseline_acc = self.compute_baseline_accuracy(val_loader)
        print(f"    Baseline accuracy: {baseline_acc:.4f}")

        importance = {}

        for idx, name in enumerate(channel_names):
            print(f"    Computing importance for {name} ({idx+1}/{len(channel_names)})...")

            drops = []
            for _ in range(self.n_repeats):
                permuted_acc = self.compute_permuted_accuracy(val_loader, idx)
                drop = baseline_acc - permuted_acc
                drops.append(drop)

            mean_drop = np.mean(drops)
            std_drop = np.std(drops)

            # 95% confidence interval
            ci_low = mean_drop - 1.96 * std_drop / np.sqrt(self.n_repeats)
            ci_high = mean_drop + 1.96 * std_drop / np.sqrt(self.n_repeats)

            importance[name] = {
                'mean': float(mean_drop),
                'std': float(std_drop),
                'ci_low': float(ci_low),
                'ci_high': float(ci_high),
            }

        return importance

    def compute_category_importance(
        self,
        importance: Dict[str, Dict[str, float]],
        channel_categories: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Aggregate importance by category.

        Args:
            importance: Per-channel importance
            channel_categories: Mapping of category names to channel names

        Returns:
            Dict mapping category to aggregate importance
        """
        category_importance = {}

        for category, channels in channel_categories.items():
            total_importance = sum(
                importance.get(ch, {'mean': 0})['mean']
                for ch in channels
            )
            category_importance[category] = float(total_importance)

        return category_importance

    def compute_statistical_tests(
        self,
        importance: Dict[str, Dict[str, float]],
        channel_categories: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute statistical tests for feature importance.

        Returns:
            Dict with test results
        """
        tests = {}

        # Get importance values by category
        velocity_values = [importance[ch]['mean'] for ch in channel_categories['velocity']]
        position_values = [importance[ch]['mean'] for ch in channel_categories['position']]
        error_values = [importance[ch]['mean'] for ch in channel_categories['error']]
        target_values = [importance[ch]['mean'] for ch in channel_categories['target']]

        # Raw = position (eye positions)
        raw_values = position_values

        # Velocity vs Position
        if len(velocity_values) >= 2 and len(position_values) >= 2:
            t_stat, p_value = stats.ttest_ind(velocity_values, position_values)
            tests['velocity_vs_position'] = {
                't_stat': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'velocity_mean': float(np.mean(velocity_values)),
                'position_mean': float(np.mean(position_values)),
            }
        else:
            tests['velocity_vs_position'] = {
                't_stat': 0.0,
                'p_value': 1.0,
                'significant': False,
            }

        # Error vs Raw
        if len(error_values) >= 2 and len(raw_values) >= 2:
            t_stat, p_value = stats.ttest_ind(error_values, raw_values)
            tests['error_vs_raw'] = {
                't_stat': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'error_mean': float(np.mean(error_values)),
                'raw_mean': float(np.mean(raw_values)),
            }
        else:
            tests['error_vs_raw'] = {
                't_stat': 0.0,
                'p_value': 1.0,
                'significant': False,
            }

        # Left vs Right eye
        left_channels = ['LH', 'LV', 'LH_Velocity', 'LV_Velocity', 'ErrorH_L', 'ErrorV_L']
        right_channels = ['RH', 'RV', 'RH_Velocity', 'RV_Velocity', 'ErrorH_R', 'ErrorV_R']

        left_values = [importance[ch]['mean'] for ch in left_channels if ch in importance]
        right_values = [importance[ch]['mean'] for ch in right_channels if ch in importance]

        if len(left_values) >= 2 and len(right_values) >= 2:
            t_stat, p_value = stats.ttest_ind(left_values, right_values)
            tests['left_vs_right'] = {
                't_stat': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'left_mean': float(np.mean(left_values)),
                'right_mean': float(np.mean(right_values)),
            }
        else:
            tests['left_vs_right'] = {
                't_stat': 0.0,
                'p_value': 1.0,
                'significant': False,
            }

        return tests

    def compute_importance_gini(
        self,
        importance: Dict[str, Dict[str, float]]
    ) -> float:
        """
        Compute Gini coefficient for importance distribution.

        Higher Gini = more unequal distribution (few features dominate)
        Lower Gini = more equal distribution
        """
        values = np.array([v['mean'] for v in importance.values()])
        values = np.abs(values)  # Use absolute values

        if values.sum() == 0:
            return 0.0

        # Sort values
        sorted_values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)

        # Gini coefficient formula
        gini = (np.sum((2 * index - n - 1) * sorted_values)) / (n * np.sum(sorted_values))

        return float(gini)

    def generate_importance_bar_chart(
        self,
        importance: Dict[str, Dict[str, float]],
        channel_names: List[str]
    ):
        """Generate horizontal bar chart of feature importance with error bars."""
        # Sort by importance
        sorted_channels = sorted(
            channel_names,
            key=lambda x: importance[x]['mean'],
            reverse=True
        )

        values = [importance[ch]['mean'] for ch in sorted_channels]
        errors = [importance[ch]['std'] for ch in sorted_channels]

        fig, ax = plt.subplots(figsize=(12, 8))

        y_pos = np.arange(len(sorted_channels))

        # Color by category
        colors = []
        for ch in sorted_channels:
            if 'Velocity' in ch:
                colors.append('#e74c3c')  # Red for velocity
            elif 'Error' in ch:
                colors.append('#3498db')  # Blue for error
            elif 'Target' in ch:
                colors.append('#95a5a6')  # Gray for target
            else:
                colors.append('#2ecc71')  # Green for position

        bars = ax.barh(y_pos, values, xerr=errors, align='center', color=colors, alpha=0.8, capsize=3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_channels)
        ax.invert_yaxis()
        ax.set_xlabel('Importance (Accuracy Drop)')
        ax.set_title('Feature Importance (Permutation)')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', label='Velocity'),
            Patch(facecolor='#3498db', label='Error'),
            Patch(facecolor='#2ecc71', label='Position'),
            Patch(facecolor='#95a5a6', label='Target'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig_feature_importance.pdf'), dpi=150, bbox_inches='tight')
        plt.close()

    def generate_category_bar_chart(
        self,
        category_importance: Dict[str, float]
    ):
        """Generate grouped bar chart by category."""
        categories = list(category_importance.keys())
        values = list(category_importance.values())

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(categories))
        colors = ['#2ecc71', '#95a5a6', '#e74c3c', '#3498db']  # position, target, velocity, error

        bars = ax.bar(x, values, color=colors)

        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in categories])
        ax.set_ylabel('Cumulative Importance (Accuracy Drop)')
        ax.set_title('Feature Importance by Category')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig_feature_importance_by_category.pdf'), dpi=150, bbox_inches='tight')
        plt.close()

    def run_analysis(
        self,
        val_loader: DataLoader,
        channel_names: List[str],
        channel_categories: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Run the complete feature importance analysis.

        Args:
            val_loader: Validation data loader
            channel_names: Names of channels
            channel_categories: Mapping of category names to channel names

        Returns:
            Dict with all analysis results
        """
        print("  Computing permutation importance...")

        # Compute importance
        importance = self.compute_permutation_importance(val_loader, channel_names)

        # Create ranking
        ranking = sorted(
            [{'channel': ch, **stats} for ch, stats in importance.items()],
            key=lambda x: x['mean'],
            reverse=True
        )

        # Compute top 3 cumulative
        top3_cumulative = sum(r['mean'] for r in ranking[:3])

        # Compute Gini coefficient
        importance_gini = self.compute_importance_gini(importance)

        # Compute category importance
        category_importance = self.compute_category_importance(importance, channel_categories)

        # Statistical tests
        statistical_tests = self.compute_statistical_tests(importance, channel_categories)

        # Generate visualizations
        print("  Generating visualizations...")
        self.generate_importance_bar_chart(importance, channel_names)
        self.generate_category_bar_chart(category_importance)

        # Compile results
        results = {
            'ranking': ranking,
            'top3_cumulative': float(top3_cumulative),
            'importance_gini': float(importance_gini),
            'category_importance': category_importance,
            **statistical_tests,
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

        with open(os.path.join(self.quantitative_dir, 'feature_importance.json'), 'w') as f:
            json.dump(convert_to_serializable(results), f, indent=2)

        return results
