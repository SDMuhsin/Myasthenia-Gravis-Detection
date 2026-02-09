"""
Figure Generator for MHTPN Explainability Analysis

Generates all figures with quantitative JSON companions.
Every figure has a corresponding JSON file with metrics for verification.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from typing import Dict, Any, List, Optional


# Consistent style for paper figures
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'hc': '#2E86AB',      # Blue for HC
    'mg': '#E94F37',      # Red for MG
    'default': '#2E86AB',
    'pass': '#28A745',    # Green
    'fail': '#DC3545',    # Red
    'neutral': '#6C757D', # Gray
}


def convert_to_json_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(v) for v in obj)
    return obj


def save_json(data: Dict, path: str):
    """Save dictionary as JSON file, handling numpy types."""
    serializable = convert_to_json_serializable(data)
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)


# =============================================================================
# Component 1: Trajectory Prototype Figures
# =============================================================================

def generate_trajectory_evolution_2d(
    trajectory_results: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Generate 2D visualization of trajectory evolution.

    Shows HC and MG prototype trajectories in PCA space with time progression.
    """
    pca_data = trajectory_results['trajectory_pca']
    n_heads = len([k for k in pca_data['trajectories_2d'].keys() if k.startswith('head_')])

    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]

    for head_idx in range(n_heads):
        ax = axes[head_idx]
        head_data = pca_data['trajectories_2d'][f'head_{head_idx}']

        hc_traj = np.array(head_data['hc'])
        mg_traj = np.array(head_data['mg'])

        n_points = len(hc_traj)

        # Plot trajectories with color gradient for time
        for i in range(n_points - 1):
            alpha = 0.3 + 0.7 * (i / n_points)

            ax.plot(hc_traj[i:i+2, 0], hc_traj[i:i+2, 1],
                    color=COLORS['hc'], alpha=alpha, linewidth=2)
            ax.plot(mg_traj[i:i+2, 0], mg_traj[i:i+2, 1],
                    color=COLORS['mg'], alpha=alpha, linewidth=2)

        # Mark start and end points
        ax.scatter(hc_traj[0, 0], hc_traj[0, 1], c=COLORS['hc'], s=100,
                   marker='o', edgecolor='black', linewidth=1, label='HC start', zorder=5)
        ax.scatter(hc_traj[-1, 0], hc_traj[-1, 1], c=COLORS['hc'], s=100,
                   marker='s', edgecolor='black', linewidth=1, label='HC end', zorder=5)
        ax.scatter(mg_traj[0, 0], mg_traj[0, 1], c=COLORS['mg'], s=100,
                   marker='o', edgecolor='black', linewidth=1, label='MG start', zorder=5)
        ax.scatter(mg_traj[-1, 0], mg_traj[-1, 1], c=COLORS['mg'], s=100,
                   marker='s', edgecolor='black', linewidth=1, label='MG end', zorder=5)

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'Head {head_idx}')

        if head_idx == 0:
            ax.legend(fontsize=8)

    plt.suptitle('Prototype Trajectory Evolution (PCA)', fontsize=14, y=1.02)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, 'fig_trajectory_evolution_2d.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save metrics
    metrics = {
        'pca_explained_variance': pca_data['pca_explained_variance'],
        'hc_centroid': pca_data['hc_centroid'],
        'mg_centroid': pca_data['mg_centroid'],
        'class_separation_in_pca': pca_data['class_separation_in_pca'],
        'n_heads': n_heads,
    }
    metrics_path = os.path.join(output_dir, 'fig_trajectory_evolution_2d_metrics.json')
    save_json(metrics, metrics_path)

    return fig_path


def generate_interclass_distance_over_time(
    trajectory_results: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Generate plot showing inter-class distance over time for each head.
    """
    evolution_data = trajectory_results['trajectory_evolution']
    t_values = evolution_data['aggregate']['t_values']
    n_heads = len([k for k in evolution_data.keys() if k.startswith('head_')])

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.viridis(np.linspace(0, 0.8, n_heads))

    for head_idx in range(n_heads):
        head_data = evolution_data[f'head_{head_idx}']
        distances = head_data['inter_class_distance_over_time']

        ax.plot(t_values, distances, '-o', color=colors[head_idx],
                linewidth=2, markersize=4, label=f'Head {head_idx}')

    ax.set_xlabel('Normalized Time (t)', fontsize=12)
    ax.set_ylabel('Inter-class Distance (HC-MG)', fontsize=12)
    ax.set_title('Distance Between HC and MG Prototypes Over Time', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig_path = os.path.join(output_dir, 'fig_interclass_distance_over_time.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save metrics
    metrics = {
        't_values': t_values,
        'distances_by_head': {
            f'head_{i}': evolution_data[f'head_{i}']['inter_class_distance_over_time']
            for i in range(n_heads)
        },
        'mean_trajectory_divergence': evolution_data['aggregate']['mean_trajectory_divergence'],
        'n_heads_with_positive_divergence': evolution_data['aggregate']['n_heads_with_positive_divergence'],
    }
    metrics_path = os.path.join(output_dir, 'fig_interclass_distance_over_time_metrics.json')
    save_json(metrics, metrics_path)

    return fig_path


def generate_velocity_directions(
    trajectory_results: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Generate arrow plot showing velocity vectors for each head.
    """
    evolution_data = trajectory_results['trajectory_evolution']
    n_heads = len([k for k in evolution_data.keys() if k.startswith('head_')])

    fig, ax = plt.subplots(figsize=(10, 6))

    y_positions = np.arange(n_heads)

    for head_idx in range(n_heads):
        head_data = evolution_data[f'head_{head_idx}']

        hc_norm = head_data['hc_velocity_norm']
        mg_norm = head_data['mg_velocity_norm']

        # Draw arrows
        ax.arrow(0, head_idx + 0.15, hc_norm, 0, head_width=0.1, head_length=0.02,
                 fc=COLORS['hc'], ec=COLORS['hc'], linewidth=2)
        ax.arrow(0, head_idx - 0.15, mg_norm, 0, head_width=0.1, head_length=0.02,
                 fc=COLORS['mg'], ec=COLORS['mg'], linewidth=2)

        # Add labels
        ax.text(hc_norm + 0.05, head_idx + 0.15, f'{hc_norm:.2f}',
                va='center', fontsize=9, color=COLORS['hc'])
        ax.text(mg_norm + 0.05, head_idx - 0.15, f'{mg_norm:.2f}',
                va='center', fontsize=9, color=COLORS['mg'])

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f'Head {i}' for i in range(n_heads)])
    ax.set_xlabel('Velocity Magnitude', fontsize=12)
    ax.set_title('Prototype Velocity Magnitudes by Head', fontsize=14)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['hc'], label='HC prototype'),
        Patch(facecolor=COLORS['mg'], label='MG prototype'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    fig_path = os.path.join(output_dir, 'fig_velocity_directions.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save metrics
    metrics = {
        'velocity_norms': {
            f'head_{i}': {
                'hc': evolution_data[f'head_{i}']['hc_velocity_norm'],
                'mg': evolution_data[f'head_{i}']['mg_velocity_norm'],
            }
            for i in range(n_heads)
        },
        'mean_velocity_norm': evolution_data['aggregate']['mean_velocity_norm'],
        'all_heads_have_motion': evolution_data['aggregate']['all_heads_have_motion'],
    }
    metrics_path = os.path.join(output_dir, 'fig_velocity_directions_metrics.json')
    save_json(metrics, metrics_path)

    return fig_path


# =============================================================================
# Component 2: Per-Segment Decision Figures
# =============================================================================

def generate_per_segment_discrimination(
    segment_results: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Generate bar chart showing discrimination per segment.
    """
    discrimination = segment_results['per_segment_discrimination']
    n_segments = len(discrimination)
    mid = n_segments // 2

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(1, n_segments + 1)

    # Color gradient: blue (early) to red (late)
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, n_segments))

    bars = ax.bar(x, discrimination, color=colors, edgecolor='black', linewidth=1)

    # Add horizontal line at 0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Shade early vs late regions
    ax.axvspan(0.5, mid + 0.5, alpha=0.1, color='blue', label='Early segments')
    ax.axvspan(mid + 0.5, n_segments + 0.5, alpha=0.1, color='red', label='Late segments')

    # Add mean lines
    early_mean = np.mean(discrimination[:mid])
    late_mean = np.mean(discrimination[mid:])

    ax.axhline(y=early_mean, xmin=0.05, xmax=0.45, color='blue', linestyle='--',
               linewidth=2, alpha=0.8)
    ax.axhline(y=late_mean, xmin=0.55, xmax=0.95, color='red', linestyle='--',
               linewidth=2, alpha=0.8)

    ax.text(mid/2 + 0.5, early_mean + 0.02, f'Early: {early_mean:.3f}',
            ha='center', fontsize=10, color='blue', fontweight='bold')
    ax.text(mid + mid/2 + 0.5, late_mean + 0.02, f'Late: {late_mean:.3f}',
            ha='center', fontsize=10, color='red', fontweight='bold')

    ax.set_xlabel('Segment Number (Early -> Late)', fontsize=12)
    ax.set_ylabel('Discrimination (MG-MG sim - HC-HC sim)', fontsize=12)
    ax.set_xticks(x)
    ax.set_title(f'Per-Segment Discrimination (Late-Early = {late_mean - early_mean:+.3f})',
                 fontsize=14)
    ax.legend(loc='upper left')

    plt.tight_layout()

    fig_path = os.path.join(output_dir, 'fig_per_segment_discrimination.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save metrics
    metrics = {
        'per_segment_discrimination': discrimination,
        'early_discrimination': segment_results['early_discrimination'],
        'late_discrimination': segment_results['late_discrimination'],
        'late_minus_early': segment_results['late_minus_early'],
        'temporal_pattern_pass': segment_results['temporal_pattern_pass'],
        'most_discriminative_segment': segment_results['most_discriminative_segment'],
    }
    metrics_path = os.path.join(output_dir, 'fig_per_segment_discrimination_metrics.json')
    save_json(metrics, metrics_path)

    return fig_path


def generate_similarity_profiles(
    segment_results: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Generate plot showing HC and MG similarity profiles over segments.
    """
    hc_sims = segment_results['hc_mean_similarity_profile']
    mg_sims = segment_results['mg_mean_similarity_profile']
    n_segments = len(hc_sims)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(1, n_segments + 1)

    ax.plot(x, hc_sims, '-o', color=COLORS['hc'], linewidth=2, markersize=8,
            label='HC to HC prototype')
    ax.plot(x, mg_sims, '-s', color=COLORS['mg'], linewidth=2, markersize=8,
            label='MG to MG prototype')

    ax.set_xlabel('Segment Number', fontsize=12)
    ax.set_ylabel('Similarity to Correct Prototype', fontsize=12)
    ax.set_xticks(x)
    ax.set_title('Class-wise Similarity Profiles Over Time', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig_path = os.path.join(output_dir, 'fig_similarity_profiles.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save metrics
    metrics = {
        'hc_similarity_profile': hc_sims,
        'mg_similarity_profile': mg_sims,
        'hc_mean': float(np.mean(hc_sims)),
        'mg_mean': float(np.mean(mg_sims)),
        'hc_trend': float(hc_sims[-1] - hc_sims[0]),
        'mg_trend': float(mg_sims[-1] - mg_sims[0]),
    }
    metrics_path = os.path.join(output_dir, 'fig_similarity_profiles_metrics.json')
    save_json(metrics, metrics_path)

    return fig_path


def generate_per_head_segment_patterns(
    per_head_results: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Generate subplot for each head showing discrimination pattern.
    """
    n_heads = per_head_results['aggregate']['n_heads']

    fig, axes = plt.subplots(1, n_heads, figsize=(3 * n_heads, 4), sharey=True)
    if n_heads == 1:
        axes = [axes]

    for head_idx in range(n_heads):
        ax = axes[head_idx]
        head_data = per_head_results[f'head_{head_idx}']

        discrimination = head_data['per_segment_discrimination']
        n_segments = len(discrimination)
        mid = n_segments // 2

        x = np.arange(1, n_segments + 1)
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, n_segments))

        ax.bar(x, discrimination, color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        late_minus_early = head_data['late_minus_early']
        status = "PASS" if head_data['temporal_pattern_pass'] else "FAIL"
        color = COLORS['pass'] if head_data['temporal_pattern_pass'] else COLORS['fail']

        ax.set_title(f'Head {head_idx}\n({status}: {late_minus_early:+.3f})',
                     fontsize=10, color=color)
        ax.set_xlabel('Segment')
        if head_idx == 0:
            ax.set_ylabel('Discrimination')

    plt.suptitle('Per-Head Segment Discrimination Patterns', fontsize=14, y=1.02)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, 'fig_per_head_segment_patterns.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save metrics
    metrics = {
        f'head_{i}': {
            'late_minus_early': per_head_results[f'head_{i}']['late_minus_early'],
            'temporal_pattern_pass': per_head_results[f'head_{i}']['temporal_pattern_pass'],
            'peak_segment': per_head_results[f'head_{i}']['peak_segment'],
        }
        for i in range(n_heads)
    }
    metrics['aggregate'] = per_head_results['aggregate']
    metrics_path = os.path.join(output_dir, 'fig_per_head_segment_patterns_metrics.json')
    save_json(metrics, metrics_path)

    return fig_path


# =============================================================================
# Component 3: Velocity Diversity Figures
# =============================================================================

def generate_velocity_diversity_heatmap(
    diversity_results: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Generate heatmap of velocity cosine similarities.
    """
    matrix_data = diversity_results['cosine_similarity_matrix']
    matrix = np.array(matrix_data['matrix'])
    labels = matrix_data['labels']

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', fontsize=12)

    # Add value annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                           ha='center', va='center', fontsize=8,
                           color='white' if abs(matrix[i, j]) > 0.5 else 'black')

    ax.set_title(f'Velocity Diversity (score={diversity_results["diversity_score"]:.2f})',
                 fontsize=14)

    plt.tight_layout()

    fig_path = os.path.join(output_dir, 'fig_velocity_diversity_heatmap.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save metrics
    metrics = {
        'diversity_score': diversity_results['diversity_score'],
        'mean_off_diagonal_similarity': diversity_results['mean_off_diagonal_similarity'],
        'max_off_diagonal_similarity': diversity_results['max_off_diagonal_similarity'],
        'interpretation': diversity_results['interpretation'],
    }
    metrics_path = os.path.join(output_dir, 'fig_velocity_diversity_heatmap_metrics.json')
    save_json(metrics, metrics_path)

    return fig_path


def generate_velocity_magnitudes(
    diversity_results: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Generate grouped bar chart of velocity magnitudes per head.
    """
    norms_per_head = diversity_results['velocity_norms_per_head']
    n_heads = len(norms_per_head)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(n_heads)
    width = 0.35

    hc_norms = [norms_per_head[i]['hc_norm'] for i in range(n_heads)]
    mg_norms = [norms_per_head[i]['mg_norm'] for i in range(n_heads)]

    bars1 = ax.bar(x - width/2, hc_norms, width, label='HC', color=COLORS['hc'],
                   edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, mg_norms, width, label='MG', color=COLORS['mg'],
                   edgecolor='black', linewidth=1)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Head', fontsize=12)
    ax.set_ylabel('Velocity Magnitude', fontsize=12)
    ax.set_title('Prototype Velocity Magnitudes by Head and Class', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Head {i}' for i in range(n_heads)])
    ax.legend()
    ax.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='Min threshold (0.05)')

    plt.tight_layout()

    fig_path = os.path.join(output_dir, 'fig_velocity_magnitudes.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save metrics
    metrics = {
        'hc_norms': hc_norms,
        'mg_norms': mg_norms,
        'mean_velocity_norm': diversity_results['mean_velocity_norm'],
        'min_velocity_norm': diversity_results['min_velocity_norm'],
        'all_above_threshold': diversity_results['success_criteria']['all_velocities_nonzero']['passed'],
    }
    metrics_path = os.path.join(output_dir, 'fig_velocity_magnitudes_metrics.json')
    save_json(metrics, metrics_path)

    return fig_path


def generate_velocity_pca(
    diversity_results: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Generate PCA scatter plot of velocity vectors.
    """
    pca_data = diversity_results['velocity_pca']
    coords = np.array(pca_data['coordinates'])
    labels = pca_data['labels']

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot points
    for i, (coord, label) in enumerate(zip(coords, labels)):
        head_idx = int(label[1])
        class_name = label.split('-')[1]
        color = COLORS['hc'] if class_name == 'HC' else COLORS['mg']
        marker = 'o' if class_name == 'HC' else 's'

        ax.scatter(coord[0], coord[1], c=color, s=100, marker=marker,
                   edgecolor='black', linewidth=1, alpha=0.8)
        ax.annotate(label, (coord[0], coord[1]), fontsize=8,
                    xytext=(5, 5), textcoords='offset points')

    # Plot centroids
    hc_centroid = pca_data['hc_centroid']
    mg_centroid = pca_data['mg_centroid']

    ax.scatter(hc_centroid[0], hc_centroid[1], c=COLORS['hc'], s=200, marker='X',
               edgecolor='black', linewidth=2, label='HC centroid', zorder=10)
    ax.scatter(mg_centroid[0], mg_centroid[1], c=COLORS['mg'], s=200, marker='X',
               edgecolor='black', linewidth=2, label='MG centroid', zorder=10)

    ax.set_xlabel(f'PC1 ({pca_data["explained_variance"][0]:.1%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca_data["explained_variance"][1]:.1%})', fontsize=12)
    ax.set_title(f'Velocity Vectors in PCA Space (separation={pca_data["class_separation"]:.2f})',
                 fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig_path = os.path.join(output_dir, 'fig_velocity_pca.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save metrics
    metrics = {
        'pca_explained_variance': pca_data['explained_variance'],
        'hc_centroid': hc_centroid,
        'mg_centroid': mg_centroid,
        'class_separation': pca_data['class_separation'],
    }
    metrics_path = os.path.join(output_dir, 'fig_velocity_pca_metrics.json')
    save_json(metrics, metrics_path)

    return fig_path


# =============================================================================
# Component 4: Case Study Figures
# =============================================================================

def generate_gap_trend_comparison(
    case_study_results: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Generate box plot comparing gap trend slopes across categories.
    """
    gap_stats = case_study_results['gap_trend_statistics']

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['high_conf_correct_mg', 'high_conf_correct_hc',
                  'low_conf_correct', 'misclassified_mg_to_hc', 'misclassified_hc_to_mg']
    category_labels = ['High Conf\nCorrect MG', 'High Conf\nCorrect HC',
                       'Low Conf\nCorrect', 'Misclass\nMG->HC', 'Misclass\nHC->MG']

    data = []
    for cat in categories:
        case_studies = case_study_results['case_studies'].get(cat, [])
        slopes = [cs['gap_trend_slope'] for cs in case_studies]
        data.append(slopes)

    # Create box plot
    bp = ax.boxplot(data, labels=category_labels, patch_artist=True)

    # Color boxes
    colors = [COLORS['mg'], COLORS['hc'], COLORS['neutral'],
              COLORS['fail'], COLORS['fail']]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('Gap Trend Slope', fontsize=12)
    ax.set_title('Gap Trend Slope by Prediction Category', fontsize=14)

    # Add annotations
    comparison = gap_stats.get('mg_vs_hc_comparison', {})
    if comparison.get('p_value') is not None:
        sig_text = f"MG vs HC: p={comparison['p_value']:.4f}"
        if comparison.get('significant'):
            sig_text += " **"
        ax.text(0.95, 0.95, sig_text, transform=ax.transAxes,
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    fig_path = os.path.join(output_dir, 'fig_gap_trend_comparison.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save metrics
    metrics = {
        'gap_trend_statistics': gap_stats,
        'success_criteria': case_study_results['success_criteria'],
    }
    metrics_path = os.path.join(output_dir, 'fig_gap_trend_comparison_metrics.json')
    save_json(metrics, metrics_path)

    return fig_path


def generate_decision_segment_histogram(
    case_study_results: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Generate histogram of decision segments.
    """
    decision_stats = case_study_results['decision_segment_stats']

    fig, ax = plt.subplots(figsize=(8, 5))

    if decision_stats.get('distribution'):
        segments = sorted(decision_stats['distribution'].keys())
        counts = [decision_stats['distribution'].get(seg, 0) for seg in segments]

        ax.bar(segments, counts, color=COLORS['default'], edgecolor='black', linewidth=1)

        # Add vertical line at midpoint
        n_segments = max(segments)
        mid = n_segments // 2
        ax.axvline(x=mid + 0.5, color='red', linestyle='--', linewidth=2,
                   label=f'Mid-point (seg {mid})')

        ax.set_xlabel('Decision Segment', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xticks(segments)
        ax.set_title(f'Decision Segment Distribution (late frac={decision_stats.get("late_decision_fraction", 0):.1%})',
                     fontsize=14)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No decision segments identified',
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()

    fig_path = os.path.join(output_dir, 'fig_decision_segment_histogram.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save metrics
    metrics = decision_stats
    metrics_path = os.path.join(output_dir, 'fig_decision_segment_histogram_metrics.json')
    save_json(metrics, metrics_path)

    return fig_path


def generate_case_study_panel(
    case_study: Dict[str, Any],
    category: str,
    sample_idx_in_category: int,
    output_dir: str,
) -> str:
    """
    Generate a detailed panel for a single case study.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Per-segment similarities
    ax1 = axes[0]
    n_segments = len(case_study['per_segment_hc_similarity'])
    x = np.arange(1, n_segments + 1)

    ax1.plot(x, case_study['per_segment_hc_similarity'], '-o', color=COLORS['hc'],
             linewidth=2, markersize=6, label='HC similarity')
    ax1.plot(x, case_study['per_segment_mg_similarity'], '-s', color=COLORS['mg'],
             linewidth=2, markersize=6, label='MG similarity')

    ax1.set_xlabel('Segment')
    ax1.set_ylabel('Similarity')
    ax1.set_title('Per-Segment Prototype Similarities')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Gap over time with trend
    ax2 = axes[1]
    gap = case_study['similarity_gap']

    ax2.plot(x, gap, '-o', color=COLORS['neutral'], linewidth=2, markersize=6)

    # Add trend line
    slope = case_study['gap_trend_slope']
    intercept = case_study['gap_trend_intercept']
    trend_line = [intercept + slope * (i - 1) for i in x]
    ax2.plot(x, trend_line, '--', color='red', linewidth=2,
             label=f'Trend (slope={slope:.3f})')

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.fill_between(x, 0, gap, where=[g > 0 for g in gap],
                     color=COLORS['mg'], alpha=0.3, label='Toward MG')
    ax2.fill_between(x, 0, gap, where=[g <= 0 for g in gap],
                     color=COLORS['hc'], alpha=0.3, label='Toward HC')

    ax2.set_xlabel('Segment')
    ax2.set_ylabel('Gap (MG - HC similarity)')
    ax2.set_title(f'Similarity Gap: {case_study["trend_interpretation"]}')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Main title
    correctness = "CORRECT" if case_study['correct'] else "INCORRECT"
    correctness_color = COLORS['pass'] if case_study['correct'] else COLORS['fail']

    plt.suptitle(
        f'{category.replace("_", " ").title()} - Sample {case_study["sample_idx"]} | '
        f'True: {case_study["true_label_name"]}, Pred: {case_study["predicted_label_name"]} ({correctness}) | '
        f'Conf: {case_study["confidence"]:.2f}',
        fontsize=12, color=correctness_color
    )

    plt.tight_layout()

    fig_path = os.path.join(output_dir, f'fig_case_study_{category}_{sample_idx_in_category}.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    return fig_path


def generate_all_case_study_panels(
    case_study_results: Dict[str, Any],
    output_dir: str,
) -> List[str]:
    """
    Generate panel figures for all case studies.
    """
    fig_paths = []

    for category, studies in case_study_results['case_studies'].items():
        for idx, study in enumerate(studies):
            path = generate_case_study_panel(study, category, idx, output_dir)
            fig_paths.append(path)

    return fig_paths


# =============================================================================
# Main Figure Generation
# =============================================================================

def generate_all_figures(
    trajectory_results: Dict[str, Any],
    segment_results: Dict[str, Any],
    per_head_segment_results: Dict[str, Any],
    diversity_results: Dict[str, Any],
    case_study_results: Dict[str, Any],
    output_dir: str,
) -> Dict[str, List[str]]:
    """
    Generate all figures for the explainability analysis.

    Returns:
        Dict mapping component name to list of generated figure paths
    """
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    generated = {
        'trajectory': [],
        'segment': [],
        'diversity': [],
        'case_studies': [],
    }

    # Component 1: Trajectory figures
    generated['trajectory'].append(generate_trajectory_evolution_2d(trajectory_results, figures_dir))
    generated['trajectory'].append(generate_interclass_distance_over_time(trajectory_results, figures_dir))
    generated['trajectory'].append(generate_velocity_directions(trajectory_results, figures_dir))

    # Component 2: Segment figures
    generated['segment'].append(generate_per_segment_discrimination(segment_results, figures_dir))
    generated['segment'].append(generate_similarity_profiles(segment_results, figures_dir))
    generated['segment'].append(generate_per_head_segment_patterns(per_head_segment_results, figures_dir))

    # Component 3: Diversity figures
    generated['diversity'].append(generate_velocity_diversity_heatmap(diversity_results, figures_dir))
    generated['diversity'].append(generate_velocity_magnitudes(diversity_results, figures_dir))
    generated['diversity'].append(generate_velocity_pca(diversity_results, figures_dir))

    # Component 4: Case study figures
    generated['case_studies'].append(generate_gap_trend_comparison(case_study_results, figures_dir))
    generated['case_studies'].append(generate_decision_segment_histogram(case_study_results, figures_dir))
    generated['case_studies'].extend(generate_all_case_study_panels(case_study_results, figures_dir))

    return generated
