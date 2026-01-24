"""
Component 2: Temporal Saliency Analysis

Analyzes which time regions in the recordings drive classification decisions.

Method: Integrated Gradients - more reliable than vanilla gradients for attribution.

Metrics computed:
- Peak location mean/std: Where in the sequence is attention highest
- Attention entropy: How spread out is the attention
- Top 10% coverage: Fraction of total attention in top 10% regions
- Early vs late ratio: Attention in first half vs second half
- Cross-sample consistency: Correlation of saliency patterns

Statistical tests:
- Peak location t-test: Do MG and HC have different peak locations?
- Entropy t-test: Do classes have different attention spread?
- Early/late ratio t-test: Fatigability pattern detection

Visualizations:
- Aggregate saliency heatmap (time x channels)
- Per-channel temporal importance
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
from torch.utils.data import DataLoader
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class TemporalSaliencyAnalyzer:
    """Analyzer for temporal saliency patterns using Integrated Gradients.

    IMPORTANT FIX (2026-01-24): This version correctly handles the padding artifact
    discovered in PPT_INVESTIGATION_FINDINGS.md. Zero-padded regions at the end of
    sequences were causing the model to appear to focus on early regions, when in
    reality the late segment discrimination was being diluted by padding.

    The fix:
    1. Detect actual sequence length (before padding)
    2. Compute saliency metrics only on real data
    3. Use relative temporal positions (0-1 normalized to actual length)
    """

    def __init__(
        self,
        model,
        device: torch.device,
        output_dir: str,
        quantitative_dir: str,
        ig_steps: int = 50,
        n_samples: int = 100,
    ):
        """
        Args:
            model: Trained MultiHeadProtoNet
            device: Computation device
            output_dir: Directory for figure output
            quantitative_dir: Directory for quantitative JSON output
            ig_steps: Number of integration steps for IG
            n_samples: Maximum number of samples to analyze
        """
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.quantitative_dir = quantitative_dir
        self.ig_steps = ig_steps
        self.n_samples = n_samples

    def detect_actual_length(self, input_tensor: torch.Tensor) -> int:
        """
        Detect actual sequence length by finding where zero-padding starts.

        The preprocessing pipeline pads shorter sequences with zeros. We detect
        the actual length by finding the last non-zero timestep.

        Args:
            input_tensor: (1, seq_len, features) input

        Returns:
            Actual sequence length (before padding)
        """
        # Get the input as numpy
        x = input_tensor.detach().cpu().numpy()[0]  # (seq_len, features)

        # Find the last timestep that has any non-zero values
        # We check the sum of absolute values per timestep
        timestep_activity = np.abs(x).sum(axis=1)  # (seq_len,)

        # Find last non-zero timestep
        non_zero_mask = timestep_activity > 1e-6
        if not non_zero_mask.any():
            return len(timestep_activity)

        last_nonzero_idx = np.where(non_zero_mask)[0][-1]

        # Actual length is last_nonzero_idx + 1
        return int(last_nonzero_idx + 1)

    def integrated_gradients(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Compute Integrated Gradients attribution.

        Args:
            input_tensor: Input tensor (1, seq_len, features)
            target_class: Target class for gradient computation (None = predicted)
            baseline: Baseline input (None = zeros)

        Returns:
            Attribution map (seq_len, features)
        """
        self.model.eval()
        input_tensor = input_tensor.detach().to(self.device)

        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()

        # Create baseline (zeros by default)
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        baseline = baseline.to(self.device)

        # Compute integrated gradients
        all_gradients = []
        for alpha in np.linspace(0, 1, self.ig_steps):
            scaled = baseline + alpha * (input_tensor - baseline)
            scaled = scaled.clone().detach().requires_grad_(True)

            output = self.model(scaled)
            self.model.zero_grad()
            output[0, target_class].backward()

            gradients = scaled.grad.detach().cpu().numpy()[0]
            all_gradients.append(gradients)

        # Average gradients
        avg_gradients = np.mean(all_gradients, axis=0)

        # Multiply by input difference
        input_diff = (input_tensor - baseline).detach().cpu().numpy()[0]
        attributions = avg_gradients * input_diff

        return np.abs(attributions)

    def compute_saliency_metrics(
        self,
        saliency_map: np.ndarray,
        actual_length: int = None
    ) -> Dict[str, float]:
        """
        Compute metrics from a single saliency map.

        IMPORTANT: This method now correctly handles padded sequences by only
        analyzing the actual (non-padded) portion of the saliency map.

        Args:
            saliency_map: (seq_len, features) saliency values
            actual_length: Actual sequence length (before padding). If None,
                          uses full length (assumes no padding).

        Returns:
            Dict with computed metrics
        """
        full_seq_len = saliency_map.shape[0]

        # Use actual length if provided, otherwise use full length
        if actual_length is None:
            actual_length = full_seq_len

        # CRITICAL FIX: Only analyze the REAL (non-padded) portion
        # This fixes the artifact where late segments appeared less discriminative
        # due to zero-padding diluting the signal
        saliency_real = saliency_map[:actual_length, :]

        # Aggregate across features for temporal saliency
        temporal_saliency = np.mean(saliency_real, axis=1)

        # Normalize
        if temporal_saliency.sum() > 0:
            temporal_saliency = temporal_saliency / temporal_saliency.sum()

        real_len = len(temporal_saliency)

        # Peak location (normalized to 0-1 relative to ACTUAL length)
        peak_location = np.argmax(temporal_saliency) / real_len

        # Attention entropy
        # Add small epsilon to avoid log(0)
        temporal_saliency_safe = temporal_saliency + 1e-10
        temporal_saliency_safe = temporal_saliency_safe / temporal_saliency_safe.sum()
        entropy = -np.sum(temporal_saliency_safe * np.log(temporal_saliency_safe))

        # Top 10% coverage (relative to actual length)
        sorted_saliency = np.sort(temporal_saliency)[::-1]
        top_10_idx = max(1, int(real_len * 0.1))
        top_10_coverage = np.sum(sorted_saliency[:top_10_idx])

        # Early vs late ratio (first half vs second half of ACTUAL data)
        # This is the key metric for fatigability detection
        mid_point = real_len // 2
        early_sum = np.sum(temporal_saliency[:mid_point])
        late_sum = np.sum(temporal_saliency[mid_point:])
        early_late_ratio = early_sum / (late_sum + 1e-10)

        return {
            'peak_location': float(peak_location),
            'attention_entropy': float(entropy),
            'top10_coverage': float(top_10_coverage),
            'early_vs_late_ratio': float(early_late_ratio),
            'actual_length': int(actual_length),
            'padded_length': int(full_seq_len),
        }

    def compute_per_channel_metrics(
        self,
        saliency_map: np.ndarray,
        channel_names: List[str],
        actual_length: int = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-channel temporal metrics.

        Args:
            saliency_map: (seq_len, features) saliency values
            channel_names: Names of channels
            actual_length: Actual sequence length (before padding)

        Returns:
            Dict mapping channel names to metrics
        """
        results = {}

        # Use actual length if provided
        if actual_length is None:
            actual_length = saliency_map.shape[0]

        # Only analyze real (non-padded) portion
        saliency_real = saliency_map[:actual_length, :]
        real_len = actual_length

        for i, name in enumerate(channel_names):
            channel_saliency = saliency_real[:, i]

            if channel_saliency.sum() > 0:
                channel_saliency_norm = channel_saliency / channel_saliency.sum()
                peak_time = np.argmax(channel_saliency) / real_len
                importance_integral = float(channel_saliency.sum())
            else:
                peak_time = 0.5
                importance_integral = 0.0

            results[name] = {
                'peak_time': float(peak_time),
                'importance_integral': float(importance_integral),
            }

        return results

    def run_statistical_tests(
        self,
        mg_metrics: List[Dict[str, float]],
        hc_metrics: List[Dict[str, float]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run statistical tests comparing MG vs HC saliency patterns.

        Args:
            mg_metrics: List of metric dicts for MG samples
            hc_metrics: List of metric dicts for HC samples

        Returns:
            Dict of test results
        """
        tests = {}

        metrics_to_test = ['peak_location', 'attention_entropy', 'early_vs_late_ratio']

        for metric in metrics_to_test:
            mg_values = [m[metric] for m in mg_metrics]
            hc_values = [m[metric] for m in hc_metrics]

            # Skip if insufficient data
            if len(mg_values) < 2 or len(hc_values) < 2:
                tests[metric] = {
                    't_stat': 0.0,
                    'p_value': 1.0,
                    'significant': False,
                    'cohens_d': 0.0,
                }
                continue

            # Independent t-test
            t_stat, p_value = stats.ttest_ind(mg_values, hc_values)

            # Cohen's d effect size
            mg_mean = np.mean(mg_values)
            hc_mean = np.mean(hc_values)
            pooled_std = np.sqrt(
                ((len(mg_values) - 1) * np.var(mg_values) + (len(hc_values) - 1) * np.var(hc_values)) /
                (len(mg_values) + len(hc_values) - 2)
            )
            cohens_d = (mg_mean - hc_mean) / (pooled_std + 1e-10)

            tests[metric] = {
                't_stat': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'cohens_d': float(cohens_d),
            }

        return tests

    def generate_heatmap(
        self,
        mg_saliency: np.ndarray,
        hc_saliency: np.ndarray,
        channel_names: List[str],
        downsample_factor: int = 10
    ):
        """
        Generate aggregate saliency heatmap for both classes.

        Args:
            mg_saliency: (n_mg_samples, seq_len, features) averaged saliency
            hc_saliency: (n_hc_samples, seq_len, features) averaged saliency
            channel_names: Names of channels
            downsample_factor: Factor to downsample time dimension for visualization
        """
        # Average across samples
        mg_avg = np.mean(mg_saliency, axis=0)  # (seq_len, features)
        hc_avg = np.mean(hc_saliency, axis=0)

        # Downsample time dimension for visualization
        seq_len = mg_avg.shape[0]
        new_len = seq_len // downsample_factor
        mg_avg_ds = mg_avg[:new_len * downsample_factor].reshape(new_len, downsample_factor, -1).mean(axis=1)
        hc_avg_ds = hc_avg[:new_len * downsample_factor].reshape(new_len, downsample_factor, -1).mean(axis=1)

        # Create time labels (normalized 0-1)
        time_labels = np.linspace(0, 1, new_len)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # MG heatmap
        im1 = axes[0].imshow(
            mg_avg_ds.T, aspect='auto', cmap='hot',
            extent=[0, 1, len(channel_names) - 0.5, -0.5]
        )
        axes[0].set_title('MG Saliency Pattern')
        axes[0].set_xlabel('Normalized Time')
        axes[0].set_ylabel('Channel')
        axes[0].set_yticks(np.arange(len(channel_names)))
        axes[0].set_yticklabels(channel_names)
        plt.colorbar(im1, ax=axes[0], label='Saliency')

        # HC heatmap
        im2 = axes[1].imshow(
            hc_avg_ds.T, aspect='auto', cmap='hot',
            extent=[0, 1, len(channel_names) - 0.5, -0.5]
        )
        axes[1].set_title('HC Saliency Pattern')
        axes[1].set_xlabel('Normalized Time')
        axes[1].set_ylabel('Channel')
        axes[1].set_yticks(np.arange(len(channel_names)))
        axes[1].set_yticklabels(channel_names)
        plt.colorbar(im2, ax=axes[1], label='Saliency')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig_temporal_saliency_heatmap.pdf'), dpi=150, bbox_inches='tight')
        plt.close()

    def generate_per_channel_temporal_plot(
        self,
        mg_saliency: np.ndarray,
        hc_saliency: np.ndarray,
        channel_names: List[str],
        downsample_factor: int = 10
    ):
        """Generate per-channel temporal importance line plots."""
        # Average across samples
        mg_avg = np.mean(mg_saliency, axis=0)
        hc_avg = np.mean(hc_saliency, axis=0)

        # Downsample for visualization
        seq_len = mg_avg.shape[0]
        new_len = seq_len // downsample_factor
        mg_avg_ds = mg_avg[:new_len * downsample_factor].reshape(new_len, downsample_factor, -1).mean(axis=1)
        hc_avg_ds = hc_avg[:new_len * downsample_factor].reshape(new_len, downsample_factor, -1).mean(axis=1)

        time = np.linspace(0, 1, new_len)

        # Plot velocity and error channels (most clinically relevant)
        relevant_channels = [
            ('LH_Velocity', 6), ('RH_Velocity', 7),
            ('ErrorH_L', 10), ('ErrorV_L', 12)
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, (name, ch_idx) in enumerate(relevant_channels):
            ax = axes[idx]
            ax.plot(time, mg_avg_ds[:, ch_idx], 'r-', label='MG', linewidth=2)
            ax.plot(time, hc_avg_ds[:, ch_idx], 'b-', label='HC', linewidth=2)
            ax.set_xlabel('Normalized Time')
            ax.set_ylabel('Saliency')
            ax.set_title(f'{name}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle('Per-Channel Temporal Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig_per_channel_temporal.pdf'), dpi=150, bbox_inches='tight')
        plt.close()

    def run_analysis(
        self,
        val_loader: DataLoader,
        channel_names: List[str]
    ) -> Dict[str, Any]:
        """
        Run the complete temporal saliency analysis.

        IMPORTANT FIX: This method now correctly handles the padding artifact
        by detecting actual sequence lengths and computing metrics only on
        real (non-padded) data. This fixes the issue where late segments
        appeared less discriminative due to zero-padding dilution.

        Args:
            val_loader: Validation data loader
            channel_names: Names of channels

        Returns:
            Dict with all analysis results
        """
        print("  Computing saliency maps...")
        print("  NOTE: Using padding-aware analysis (fixing late-segment dilution artifact)")

        mg_saliency_maps = []
        hc_saliency_maps = []
        mg_metrics = []
        hc_metrics = []
        mg_actual_lengths = []
        hc_actual_lengths = []

        samples_processed = 0

        for inputs, labels in val_loader:
            for i in range(inputs.size(0)):
                if samples_processed >= self.n_samples:
                    break

                input_sample = inputs[i:i+1].to(self.device)
                label = labels[i].item()

                # CRITICAL FIX: Detect actual sequence length (before padding)
                actual_length = self.detect_actual_length(input_sample)

                # Compute integrated gradients
                saliency_map = self.integrated_gradients(input_sample)

                # Store saliency map and compute metrics with actual length
                metrics = self.compute_saliency_metrics(saliency_map, actual_length)

                if label == 1:  # MG
                    mg_saliency_maps.append(saliency_map)
                    mg_metrics.append(metrics)
                    mg_actual_lengths.append(actual_length)
                else:  # HC
                    hc_saliency_maps.append(saliency_map)
                    hc_metrics.append(metrics)
                    hc_actual_lengths.append(actual_length)

                samples_processed += 1

            if samples_processed >= self.n_samples:
                break

        print(f"  Processed {samples_processed} samples ({len(mg_saliency_maps)} MG, {len(hc_saliency_maps)} HC)")

        # Report padding statistics
        if mg_actual_lengths and hc_actual_lengths:
            padded_len = mg_saliency_maps[0].shape[0] if mg_saliency_maps else hc_saliency_maps[0].shape[0]
            mg_mean_len = np.mean(mg_actual_lengths)
            hc_mean_len = np.mean(hc_actual_lengths)
            mg_padding_frac = 1 - (mg_mean_len / padded_len)
            hc_padding_frac = 1 - (hc_mean_len / padded_len)
            print(f"  Padding statistics:")
            print(f"    MG: mean actual length = {mg_mean_len:.0f}/{padded_len} ({mg_padding_frac*100:.1f}% padding)")
            print(f"    HC: mean actual length = {hc_mean_len:.0f}/{padded_len} ({hc_padding_frac*100:.1f}% padding)")

        # Convert to arrays
        mg_saliency = np.array(mg_saliency_maps) if mg_saliency_maps else np.array([])
        hc_saliency = np.array(hc_saliency_maps) if hc_saliency_maps else np.array([])

        # Aggregate metrics
        def aggregate_metrics(metrics_list: List[Dict]) -> Dict[str, float]:
            if not metrics_list:
                return {
                    'peak_location_mean': 0.5,
                    'peak_location_std': 0.0,
                    'attention_entropy': 0.0,
                    'top10_coverage': 0.0,
                    'early_vs_late_ratio': 1.0,
                }
            return {
                'peak_location_mean': float(np.mean([m['peak_location'] for m in metrics_list])),
                'peak_location_std': float(np.std([m['peak_location'] for m in metrics_list])),
                'attention_entropy': float(np.mean([m['attention_entropy'] for m in metrics_list])),
                'top10_coverage': float(np.mean([m['top10_coverage'] for m in metrics_list])),
                'early_vs_late_ratio': float(np.mean([m['early_vs_late_ratio'] for m in metrics_list])),
            }

        mg_aggregated = aggregate_metrics(mg_metrics)
        hc_aggregated = aggregate_metrics(hc_metrics)

        # Statistical tests
        statistical_tests = self.run_statistical_tests(mg_metrics, hc_metrics)

        # Compute per-channel metrics on average saliency
        if len(mg_saliency_maps) > 0 and len(hc_saliency_maps) > 0:
            avg_saliency = np.mean(mg_saliency_maps + hc_saliency_maps, axis=0)
            per_channel_metrics = self.compute_per_channel_metrics(avg_saliency, channel_names)

            # Generate visualizations
            print("  Generating visualizations...")
            self.generate_heatmap(mg_saliency, hc_saliency, channel_names)
            self.generate_per_channel_temporal_plot(mg_saliency, hc_saliency, channel_names)
        else:
            per_channel_metrics = {}

        # Compile results
        results = {
            'mg': mg_aggregated,
            'hc': hc_aggregated,
            'statistical_tests': statistical_tests,
            'per_channel_temporal': per_channel_metrics,
            'n_samples_analyzed': {
                'mg': len(mg_saliency_maps),
                'hc': len(hc_saliency_maps),
            },
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

        with open(os.path.join(self.quantitative_dir, 'temporal_saliency_metrics.json'), 'w') as f:
            json.dump(convert_to_serializable(results), f, indent=2)

        return results
