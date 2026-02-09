"""
Figure Generator for MHTPN Ablation Study

Generates figures with quantitative JSON companions for verification.
All figures are saved as PDF with accompanying metrics JSON.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from typing import Dict, List, Any, Optional


# Consistent style for paper figures
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'default': '#2E86AB',  # Blue
    'variant': '#A23B72',  # Pink
    'pass': '#28A745',     # Green
    'fail': '#DC3545',     # Red
}


def generate_all_figures(results_path: str, output_dir: str) -> Dict[str, str]:
    """
    Generate all figures for the ablation study.

    Args:
        results_path: Path to full_results.json
        output_dir: Directory to save figures

    Returns:
        Dictionary mapping figure names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    with open(results_path, 'r') as f:
        results = json.load(f)

    generated = {}

    # Figure 1: n_segments ablation
    if 'n_segments' in results['ablations']:
        path = generate_n_segments_figure(
            results['ablations']['n_segments'],
            figures_dir
        )
        generated['ablation_n_segments'] = path

    # Figure 2: Loss components ablation
    if 'loss' in results['ablations']:
        path = generate_loss_components_figure(
            results['ablations']['loss'],
            figures_dir
        )
        generated['ablation_loss_components'] = path

    # Figure 3: Per-segment discrimination
    if 'n_segments' in results['ablations']:
        # Use default config (n_segments=8)
        default_result = None
        for r in results['ablations']['n_segments']:
            if r['config'].get('is_default', False):
                default_result = r
                break
        if default_result:
            path = generate_per_segment_discrimination_figure(
                default_result,
                figures_dir
            )
            generated['per_segment_discrimination'] = path

    # Figure 4: Trajectory vs Static comparison
    if 'trajectory' in results['ablations']:
        path = generate_trajectory_figure(
            results['ablations']['trajectory'],
            figures_dir
        )
        generated['ablation_trajectory'] = path

    # Figure 5: Segment weighting comparison
    if 'weighting' in results['ablations']:
        path = generate_weighting_figure(
            results['ablations']['weighting'],
            figures_dir
        )
        generated['ablation_weighting'] = path

    return generated


def generate_n_segments_figure(
    ablation_results: List[Dict],
    output_dir: str,
) -> str:
    """
    Generate bar chart for n_segments ablation with temporal pass rate overlay.
    """
    # Sort by n_segments
    sorted_results = sorted(ablation_results, key=lambda x: x['config']['n_segments'])

    n_segments_values = [r['config']['n_segments'] for r in sorted_results]
    accuracies = [r['aggregate']['mean_accuracy'] * 100 for r in sorted_results]
    stds = [r['aggregate']['std_accuracy'] * 100 for r in sorted_results]
    temporal_pass = [r['aggregate']['n_temporal_pass'] for r in sorted_results]
    is_default = [r['config'].get('is_default', False) for r in sorted_results]

    # Create figure
    fig, ax1 = plt.subplots(figsize=(8, 5))

    x = np.arange(len(n_segments_values))
    width = 0.6

    # Bar colors
    colors = [COLORS['default'] if d else COLORS['variant'] for d in is_default]

    # Accuracy bars
    bars = ax1.bar(x, accuracies, width, yerr=stds, capsize=4,
                   color=colors, edgecolor='black', linewidth=1)

    ax1.set_xlabel('Number of Temporal Segments', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(n_segments_values)
    ax1.set_ylim(60, 80)

    # Add value labels on bars
    for i, (bar, acc, std) in enumerate(zip(bars, accuracies, stds)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Secondary axis for temporal pass rate
    ax2 = ax1.twinx()
    ax2.plot(x, temporal_pass, 'o-', color=COLORS['pass'], linewidth=2,
             markersize=8, label='Temporal Pass (folds)')
    ax2.set_ylabel('Temporal Pattern Pass (out of 5 folds)', fontsize=12)
    ax2.set_ylim(0, 6)
    ax2.set_yticks([0, 1, 2, 3, 4, 5])

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=COLORS['default'], edgecolor='black', label='Default (n=8)'),
        Patch(facecolor=COLORS['variant'], edgecolor='black', label='Variant'),
        Line2D([0], [0], marker='o', color=COLORS['pass'], label='Temporal Pass',
               markersize=8, linewidth=2),
    ]
    ax1.legend(handles=legend_elements, loc='lower right')

    plt.title('Effect of Temporal Segmentation on MHTPN Performance', fontsize=14)
    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_dir, 'ablation_n_segments.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save quantitative companion
    metrics = {
        'n_segments_values': n_segments_values,
        'accuracies': accuracies,
        'std_accuracies': stds,
        'temporal_pass_counts': temporal_pass,
        'is_default': is_default,
        'best_n_segments': n_segments_values[np.argmax(accuracies)],
        'best_accuracy': float(max(accuracies)),
    }
    metrics_path = os.path.join(output_dir, 'ablation_n_segments_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return fig_path


def generate_loss_components_figure(
    ablation_results: List[Dict],
    output_dir: str,
) -> str:
    """
    Generate bar chart for loss components ablation.
    """
    # Define order
    order = ['CE only', 'CE + Cluster', 'CE + Separation', 'CE + Cl + Sep', 'Full']
    results_by_name = {r['config']['variant_name']: r for r in ablation_results}

    labels = []
    accuracies = []
    stds = []
    is_default = []

    for name in order:
        if name in results_by_name:
            r = results_by_name[name]
            labels.append(name)
            accuracies.append(r['aggregate']['mean_accuracy'] * 100)
            stds.append(r['aggregate']['std_accuracy'] * 100)
            is_default.append(r['config'].get('is_default', False))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(labels))
    width = 0.6

    colors = [COLORS['default'] if d else COLORS['variant'] for d in is_default]

    bars = ax.bar(x, accuracies, width, yerr=stds, capsize=4,
                  color=colors, edgecolor='black', linewidth=1)

    ax.set_xlabel('Loss Configuration', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylim(60, 80)

    # Add value labels
    for bar, acc, std in zip(bars, accuracies, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
               f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add horizontal line for default
    default_acc = accuracies[is_default.index(True)] if True in is_default else None
    if default_acc:
        ax.axhline(y=default_acc, color='gray', linestyle='--', alpha=0.5, label=f'Default: {default_acc:.1f}%')

    plt.title('Effect of Loss Components on MHTPN Performance', fontsize=14)
    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_dir, 'ablation_loss_components.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save quantitative companion
    metrics = {
        'labels': labels,
        'accuracies': accuracies,
        'std_accuracies': stds,
        'is_default': is_default,
        'best_loss': labels[np.argmax(accuracies)],
        'best_accuracy': float(max(accuracies)),
        'worst_loss': labels[np.argmin(accuracies)],
        'worst_accuracy': float(min(accuracies)),
    }
    metrics_path = os.path.join(output_dir, 'ablation_loss_components_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return fig_path


def generate_per_segment_discrimination_figure(
    default_result: Dict,
    output_dir: str,
) -> str:
    """
    Generate line plot showing discrimination across segments.
    """
    per_segment = default_result['aggregate']['per_segment_discrimination_avg']
    n_segments = len(per_segment)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(1, n_segments + 1)

    # Plot discrimination values
    ax.plot(x, per_segment, 'o-', color=COLORS['default'], linewidth=2, markersize=10)

    # Fill early vs late regions
    mid = n_segments // 2
    ax.axvspan(0.5, mid + 0.5, alpha=0.1, color='blue', label='Early segments')
    ax.axvspan(mid + 0.5, n_segments + 0.5, alpha=0.1, color='red', label='Late segments')

    # Add horizontal line at 0
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Calculate early and late means
    early_mean = np.mean(per_segment[:mid])
    late_mean = np.mean(per_segment[mid:])

    # Add annotations
    ax.axhline(y=early_mean, xmin=0.05, xmax=0.45, color='blue', linestyle='-', alpha=0.7)
    ax.axhline(y=late_mean, xmin=0.55, xmax=0.95, color='red', linestyle='-', alpha=0.7)

    ax.text(mid/2 + 0.5, early_mean + 0.02, f'Early mean: {early_mean:.3f}',
           ha='center', fontsize=10, color='blue')
    ax.text(mid + mid/2 + 0.5, late_mean + 0.02, f'Late mean: {late_mean:.3f}',
           ha='center', fontsize=10, color='red')

    ax.set_xlabel('Segment Number (Early -> Late)', fontsize=12)
    ax.set_ylabel('Discrimination (MG-MG sim - HC-HC sim)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Seg {i}' for i in x], rotation=45, ha='right')

    # Legend
    ax.legend(loc='upper left')

    plt.title(f'Per-Segment Discrimination (Late-Early = {late_mean - early_mean:+.3f})', fontsize=14)
    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_dir, 'per_segment_discrimination.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save quantitative companion
    metrics = {
        'n_segments': n_segments,
        'per_segment_discrimination': per_segment,
        'segment_labels': [f'Segment {i}' for i in range(1, n_segments + 1)],
        'early_segments': list(range(1, mid + 1)),
        'late_segments': list(range(mid + 1, n_segments + 1)),
        'early_mean': float(early_mean),
        'late_mean': float(late_mean),
        'late_minus_early': float(late_mean - early_mean),
        'temporal_pattern_pass': bool(late_mean > early_mean),
    }
    metrics_path = os.path.join(output_dir, 'per_segment_discrimination_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return fig_path


def generate_trajectory_figure(
    ablation_results: List[Dict],
    output_dir: str,
) -> str:
    """
    Generate comparison figure for trajectory vs static prototypes.
    """
    results_by_name = {r['config']['variant_name']: r for r in ablation_results}

    labels = ['Trajectory\n(default)', 'Static\n(velocity=0)']
    accuracies = []
    stds = []
    velocity_norms = []

    for name in ['Trajectory', 'Static']:
        if name in results_by_name:
            r = results_by_name[name]
            accuracies.append(r['aggregate']['mean_accuracy'] * 100)
            stds.append(r['aggregate']['std_accuracy'] * 100)
            velocity_norms.append(r['aggregate']['mean_velocity_norm'])

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    x = np.arange(len(labels))
    width = 0.5

    colors = [COLORS['default'], COLORS['variant']]

    # Subplot 1: Accuracy
    bars1 = ax1.bar(x, accuracies, width, yerr=stds, capsize=6,
                   color=colors, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(60, 80)
    ax1.set_title('Classification Accuracy', fontsize=12)

    for bar, acc, std in zip(bars1, accuracies, stds):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Subplot 2: Velocity norms
    bars2 = ax2.bar(x, velocity_norms, width, color=colors, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Mean Velocity Norm', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title('Prototype Velocity Magnitude', fontsize=12)

    for bar, vel in zip(bars2, velocity_norms):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{vel:.3f}', ha='center', va='bottom', fontsize=11)

    plt.suptitle('Trajectory vs Static Prototypes', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_dir, 'ablation_trajectory.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save quantitative companion
    metrics = {
        'labels': ['Trajectory', 'Static'],
        'accuracies': accuracies,
        'std_accuracies': stds,
        'velocity_norms': velocity_norms,
        'accuracy_diff': float(accuracies[0] - accuracies[1]) if len(accuracies) == 2 else None,
        'trajectory_better': accuracies[0] > accuracies[1] if len(accuracies) == 2 else None,
    }
    metrics_path = os.path.join(output_dir, 'ablation_trajectory_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return fig_path


def generate_weighting_figure(
    ablation_results: List[Dict],
    output_dir: str,
) -> str:
    """
    Generate comparison figure for segment weighting strategies.
    """
    order = ['Uniform', 'Padding-aware', 'Learned']
    results_by_name = {r['config']['variant_name']: r for r in ablation_results}

    labels = []
    accuracies = []
    stds = []
    temporal_pass = []
    is_default = []

    for name in order:
        if name in results_by_name:
            r = results_by_name[name]
            labels.append(name)
            accuracies.append(r['aggregate']['mean_accuracy'] * 100)
            stds.append(r['aggregate']['std_accuracy'] * 100)
            temporal_pass.append(r['aggregate']['n_temporal_pass'])
            is_default.append(r['config'].get('is_default', False))

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(labels))
    width = 0.5

    colors = [COLORS['default'] if d else COLORS['variant'] for d in is_default]

    bars = ax.bar(x, accuracies, width, yerr=stds, capsize=5,
                  color=colors, edgecolor='black', linewidth=1)

    ax.set_xlabel('Segment Weighting Strategy', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(60, 80)

    # Add value labels with temporal pass info
    for bar, acc, std, tp in zip(bars, accuracies, stds, temporal_pass):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
               f'{acc:.1f}%\n({tp}/5 pass)', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title('Effect of Segment Weighting on MHTPN Performance', fontsize=14)
    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_dir, 'ablation_weighting.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save quantitative companion
    metrics = {
        'labels': labels,
        'accuracies': accuracies,
        'std_accuracies': stds,
        'temporal_pass_counts': temporal_pass,
        'is_default': is_default,
        'best_strategy': labels[np.argmax(accuracies)],
        'best_accuracy': float(max(accuracies)),
    }
    metrics_path = os.path.join(output_dir, 'ablation_weighting_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return fig_path


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python figure_generator.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]
    results_path = os.path.join(results_dir, 'full_results.json')

    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found")
        sys.exit(1)

    figures = generate_all_figures(results_path, results_dir)
    print("Figures generated:")
    for name, path in figures.items():
        print(f"  - {name}: {path}")
