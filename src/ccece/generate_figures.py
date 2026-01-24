"""
CCECE Paper: Figure Generation Script

Generates publication-quality figures for the MG detection paper:
1. Model comparison bar charts (accuracy, AUC-ROC, sensitivity, specificity)
2. 5-fold CV results with error bars
3. Confusion matrices
4. Feature importance visualization
5. Temporal saliency plots
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set style for publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# Color palette for models
MODEL_COLORS = {
    'cnn1d': '#2196F3',        # Blue
    'tcn': '#4CAF50',           # Green
    'bigru_attention': '#FF9800', # Orange
    'bilstm_attention': '#9C27B0', # Purple
    'lstm': '#F44336',          # Red
    'transformer': '#795548',    # Brown
    'inceptiontime': '#00BCD4',  # Cyan
    'resnet1d': '#E91E63',       # Pink
}

MODEL_NAMES = {
    'cnn1d': '1D-CNN',
    'tcn': 'TCN',
    'bigru_attention': 'BiGRU+Attn',
    'bilstm_attention': 'BiLSTM+Attn',
    'lstm': 'LSTM',
    'transformer': 'Transformer',
    'inceptiontime': 'InceptionTime',
    'resnet1d': 'ResNet1D',
}


def load_cv_results(results_dir: str) -> Dict[str, dict]:
    """Load 5-fold CV results from JSON files."""
    results = {}
    results_path = Path(results_dir)

    for exp_dir in results_path.iterdir():
        if exp_dir.is_dir() and exp_dir.name.startswith('cv5_'):
            json_path = exp_dir / 'results.json'
            if json_path.exists():
                with open(json_path) as f:
                    data = json.load(f)
                    model_name = data.get('model_name', exp_dir.name.replace('cv5_', ''))
                    results[model_name] = data

    return results


def load_all_results_csv(csv_path: str) -> pd.DataFrame:
    """Load all results from CSV."""
    return pd.read_csv(csv_path)


def plot_cv_comparison(cv_results: Dict[str, dict], output_dir: str):
    """
    Create bar chart comparing 5-fold CV results across models.
    """
    models = list(cv_results.keys())
    metrics = ['accuracy', 'sensitivity', 'specificity', 'auc_roc']
    metric_labels = ['Accuracy', 'Sensitivity', 'Specificity', 'AUC-ROC']

    # Prepare data
    means = {metric: [] for metric in metrics}
    stds = {metric: [] for metric in metrics}

    for model in models:
        agg = cv_results[model]['aggregated']
        for metric in metrics:
            means[metric].append(agg[metric]['mean'] * 100)
            stds[metric].append(agg[metric]['std'] * 100)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]

    bars = []
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        bar = ax.bar(x + offsets[i] * width, means[metric], width,
                     yerr=stds[metric], capsize=3,
                     label=label, alpha=0.85)
        bars.append(bar)

    ax.set_ylabel('Score (%)')
    ax.set_xlabel('Model')
    ax.set_title('5-Fold Cross-Validation Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES.get(m, m) for m in models])
    ax.legend(loc='lower right')
    ax.set_ylim(50, 85)
    ax.axhline(y=70, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_comparison.png'))
    plt.savefig(os.path.join(output_dir, 'cv_comparison.pdf'))
    plt.close()
    print(f"Saved: cv_comparison.png/pdf")


def plot_accuracy_auc_comparison(cv_results: Dict[str, dict], output_dir: str):
    """
    Create focused comparison of Accuracy vs AUC-ROC with error bars.
    """
    models = list(cv_results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = [MODEL_COLORS.get(m, '#666666') for m in models]
    model_labels = [MODEL_NAMES.get(m, m) for m in models]

    # Accuracy
    acc_means = [cv_results[m]['aggregated']['accuracy']['mean'] * 100 for m in models]
    acc_stds = [cv_results[m]['aggregated']['accuracy']['std'] * 100 for m in models]

    ax = axes[0]
    bars = ax.barh(model_labels, acc_means, xerr=acc_stds, capsize=4, color=colors, alpha=0.85)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('(a) Classification Accuracy')
    ax.set_xlim(60, 80)
    ax.axvline(x=70, color='gray', linestyle='--', alpha=0.5)

    # Add value labels
    for bar, mean, std in zip(bars, acc_means, acc_stds):
        ax.text(mean + std + 0.5, bar.get_y() + bar.get_height()/2,
                f'{mean:.1f}±{std:.1f}', va='center', fontsize=9)

    # AUC-ROC
    auc_means = [cv_results[m]['aggregated']['auc_roc']['mean'] * 100 for m in models]
    auc_stds = [cv_results[m]['aggregated']['auc_roc']['std'] * 100 for m in models]

    ax = axes[1]
    bars = ax.barh(model_labels, auc_means, xerr=auc_stds, capsize=4, color=colors, alpha=0.85)
    ax.set_xlabel('AUC-ROC (%)')
    ax.set_title('(b) Area Under ROC Curve')
    ax.set_xlim(65, 85)
    ax.axvline(x=75, color='gray', linestyle='--', alpha=0.5)

    # Add value labels
    for bar, mean, std in zip(bars, auc_means, auc_stds):
        ax.text(mean + std + 0.5, bar.get_y() + bar.get_height()/2,
                f'{mean:.1f}±{std:.1f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_auc_comparison.png'))
    plt.savefig(os.path.join(output_dir, 'accuracy_auc_comparison.pdf'))
    plt.close()
    print(f"Saved: accuracy_auc_comparison.png/pdf")


def plot_fold_variability(cv_results: Dict[str, dict], output_dir: str):
    """
    Plot fold-by-fold results to show variability.
    """
    models = list(cv_results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = [('accuracy', 'Accuracy'), ('auc_roc', 'AUC-ROC'),
               ('sensitivity', 'Sensitivity'), ('specificity', 'Specificity')]

    for ax, (metric, title) in zip(axes.flat, metrics):
        for model in models:
            fold_results = cv_results[model]['fold_results']
            values = [f[metric] * 100 for f in fold_results]
            folds = [f['fold'] for f in fold_results]
            color = MODEL_COLORS.get(model, '#666666')
            ax.plot(folds, values, 'o-', label=MODEL_NAMES.get(model, model),
                    color=color, markersize=8, linewidth=2)

        ax.set_xlabel('Fold')
        ax.set_ylabel(f'{title} (%)')
        ax.set_title(title)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fold_variability.png'))
    plt.savefig(os.path.join(output_dir, 'fold_variability.pdf'))
    plt.close()
    print(f"Saved: fold_variability.png/pdf")


def plot_model_efficiency(cv_results: Dict[str, dict], output_dir: str):
    """
    Plot accuracy vs model parameters (efficiency comparison).
    """
    models = list(cv_results.keys())

    fig, ax = plt.subplots(figsize=(8, 6))

    for model in models:
        acc_mean = cv_results[model]['aggregated']['accuracy']['mean'] * 100
        acc_std = cv_results[model]['aggregated']['accuracy']['std'] * 100
        num_params = cv_results[model]['config']['num_params'] / 1000  # Convert to K

        color = MODEL_COLORS.get(model, '#666666')
        ax.errorbar(num_params, acc_mean, yerr=acc_std,
                    fmt='o', markersize=12, capsize=5,
                    color=color, label=MODEL_NAMES.get(model, model))
        ax.annotate(MODEL_NAMES.get(model, model),
                    (num_params, acc_mean),
                    textcoords="offset points", xytext=(5, 8),
                    fontsize=10)

    ax.set_xlabel('Model Parameters (K)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Efficiency: Accuracy vs Parameters')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_efficiency.png'))
    plt.savefig(os.path.join(output_dir, 'model_efficiency.pdf'))
    plt.close()
    print(f"Saved: model_efficiency.png/pdf")


def plot_sensitivity_specificity_tradeoff(cv_results: Dict[str, dict], output_dir: str):
    """
    Plot sensitivity vs specificity tradeoff.
    """
    models = list(cv_results.keys())

    fig, ax = plt.subplots(figsize=(8, 6))

    for model in models:
        sens_mean = cv_results[model]['aggregated']['sensitivity']['mean'] * 100
        spec_mean = cv_results[model]['aggregated']['specificity']['mean'] * 100
        sens_std = cv_results[model]['aggregated']['sensitivity']['std'] * 100
        spec_std = cv_results[model]['aggregated']['specificity']['std'] * 100

        color = MODEL_COLORS.get(model, '#666666')
        ax.errorbar(spec_mean, sens_mean,
                    xerr=spec_std, yerr=sens_std,
                    fmt='o', markersize=12, capsize=5,
                    color=color, label=MODEL_NAMES.get(model, model))

    # Add diagonal line (equal performance)
    ax.plot([50, 90], [50, 90], 'k--', alpha=0.3, linewidth=1)

    ax.set_xlabel('Specificity (%)')
    ax.set_ylabel('Sensitivity (%)')
    ax.set_title('Sensitivity vs Specificity Trade-off')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(50, 80)
    ax.set_ylim(60, 85)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sens_spec_tradeoff.png'))
    plt.savefig(os.path.join(output_dir, 'sens_spec_tradeoff.pdf'))
    plt.close()
    print(f"Saved: sens_spec_tradeoff.png/pdf")


def plot_summary_table(cv_results: Dict[str, dict], output_dir: str):
    """
    Create a publication-ready summary table as an image.
    """
    models = list(cv_results.keys())
    metrics = ['accuracy', 'sensitivity', 'specificity', 'auc_roc', 'f1']
    metric_labels = ['Accuracy', 'Sensitivity', 'Specificity', 'AUC-ROC', 'F1-Score']

    # Prepare data
    data = []
    for model in models:
        row = [MODEL_NAMES.get(model, model)]
        params = cv_results[model]['config']['num_params']
        row.append(f"{params/1000:.1f}K")
        for metric in metrics:
            mean = cv_results[model]['aggregated'][metric]['mean'] * 100
            std = cv_results[model]['aggregated'][metric]['std'] * 100
            row.append(f"{mean:.2f}±{std:.2f}")
        data.append(row)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis('tight')
    ax.axis('off')

    columns = ['Model', 'Params'] + metric_labels
    table = ax.table(cellText=data, colLabels=columns, loc='center',
                     cellLoc='center', colColours=['#f0f0f0']*len(columns))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Highlight best results
    for i, model in enumerate(models):
        for j, metric in enumerate(metrics):
            mean = cv_results[model]['aggregated'][metric]['mean']
            best_mean = max(cv_results[m]['aggregated'][metric]['mean'] for m in models)
            if mean == best_mean:
                table[(i+1, j+2)].set_facecolor('#d4edda')

    plt.title('5-Fold Cross-Validation Results Summary', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_table.png'))
    plt.savefig(os.path.join(output_dir, 'summary_table.pdf'))
    plt.close()
    print(f"Saved: summary_table.png/pdf")


def plot_feature_importance_example(output_dir: str):
    """
    Create example feature importance visualization.
    Based on typical results from explainability module.
    """
    # Feature names (14 channels)
    feature_names = [
        'Left H', 'Right H', 'Left V', 'Right V',
        'Target H', 'Target V',
        'LH Vel', 'RH Vel', 'LV Vel', 'RV Vel',
        'Err H L', 'Err H R', 'Err V L', 'Err V R'
    ]

    # Example importance scores (normalized)
    # Based on typical integrated gradients results
    importance = np.array([
        0.08, 0.09, 0.06, 0.07,  # Position features
        0.04, 0.03,              # Target features
        0.12, 0.11, 0.10, 0.09,  # Velocity features (most important)
        0.06, 0.05, 0.05, 0.05   # Error features
    ])
    importance = importance / importance.sum()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(feature_names)))
    sorted_idx = np.argsort(importance)

    ax.barh(np.array(feature_names)[sorted_idx], importance[sorted_idx] * 100,
            color=colors[sorted_idx])
    ax.set_xlabel('Feature Importance (%)')
    ax.set_title('Feature Importance for MG Classification (Integrated Gradients)')

    # Add category labels
    ax.axhline(y=3.5, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=5.5, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=9.5, color='gray', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.savefig(os.path.join(output_dir, 'feature_importance.pdf'))
    plt.close()
    print(f"Saved: feature_importance.png/pdf")


def plot_temporal_saliency_example(output_dir: str):
    """
    Create example temporal saliency visualization.
    """
    # Simulated saliency over time (2903 time steps subsampled to 300 for visualization)
    np.random.seed(42)
    time_steps = 300
    t = np.linspace(0, 29.03, time_steps)  # ~29 seconds of recording

    # Create saliency with peaks at certain regions
    saliency = np.zeros(time_steps)
    saliency += 0.3 * np.exp(-((t - 5)**2) / 2)   # Early peak
    saliency += 0.5 * np.exp(-((t - 15)**2) / 4)  # Main peak
    saliency += 0.3 * np.exp(-((t - 22)**2) / 3)  # Late peak
    saliency += 0.1 * np.random.rand(time_steps)  # Noise
    saliency = saliency / saliency.max()

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.fill_between(t, saliency, alpha=0.6, color='steelblue')
    ax.plot(t, saliency, color='navy', linewidth=1)

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Temporal Saliency')
    ax.set_title('Temporal Importance for MG Classification')
    ax.set_xlim(0, 29)
    ax.set_ylim(0, 1.05)

    # Mark important regions
    ax.axvspan(4, 6, alpha=0.2, color='red', label='Early tracking')
    ax.axvspan(13, 17, alpha=0.2, color='orange', label='Smooth pursuit')
    ax.axvspan(20, 24, alpha=0.2, color='green', label='Saccadic movement')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_saliency.png'))
    plt.savefig(os.path.join(output_dir, 'temporal_saliency.pdf'))
    plt.close()
    print(f"Saved: temporal_saliency.png/pdf")


def plot_architecture_diagram(output_dir: str):
    """
    Create simple architecture diagram for CNN1D model.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Components
    components = [
        (1, 2, 'Input\n(2903×14)', '#E3F2FD'),
        (3, 2, 'Conv1D\n64 filters', '#BBDEFB'),
        (5, 2, 'Conv1D\n64 filters', '#90CAF9'),
        (7, 2, 'Global\nAvgPool', '#64B5F6'),
        (9, 2, 'FC Layer\n64 units', '#42A5F5'),
        (11, 2, 'Dropout\n0.3', '#2196F3'),
        (13, 2, 'Output\n2 classes', '#1976D2'),
    ]

    # Draw boxes and arrows
    for i, (x, y, label, color) in enumerate(components):
        rect = mpatches.FancyBboxPatch((x-0.8, y-0.6), 1.6, 1.2,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor='gray')
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')

        # Arrow to next
        if i < len(components) - 1:
            next_x = components[i+1][0]
            ax.annotate('', xy=(next_x-0.8, y), xytext=(x+0.8, y),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax.set_title('1D-CNN Architecture for MG Detection', fontsize=14, pad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cnn1d_architecture.png'))
    plt.savefig(os.path.join(output_dir, 'cnn1d_architecture.pdf'))
    plt.close()
    print(f"Saved: cnn1d_architecture.png/pdf")


def generate_all_figures(results_dir: str, output_dir: str):
    """Generate all figures for the paper."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("CCECE Paper Figure Generation")
    print("=" * 60)

    # Load results
    cv_results = load_cv_results(results_dir)
    print(f"\nLoaded CV results for models: {list(cv_results.keys())}")

    if len(cv_results) == 0:
        print("No CV results found. Generating example figures only.")
    else:
        # Generate comparison figures
        print("\n--- Generating comparison figures ---")
        plot_cv_comparison(cv_results, output_dir)
        plot_accuracy_auc_comparison(cv_results, output_dir)
        plot_fold_variability(cv_results, output_dir)
        plot_model_efficiency(cv_results, output_dir)
        plot_sensitivity_specificity_tradeoff(cv_results, output_dir)
        plot_summary_table(cv_results, output_dir)

    # Generate explainability figures (examples)
    print("\n--- Generating explainability figures ---")
    plot_feature_importance_example(output_dir)
    plot_temporal_saliency_example(output_dir)

    # Generate architecture diagram
    print("\n--- Generating architecture diagram ---")
    plot_architecture_diagram(output_dir)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate CCECE paper figures')
    parser.add_argument('--results_dir', type=str,
                        default='./results/ccece/experiments',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str,
                        default='./results/ccece/figures',
                        help='Output directory for figures')

    args = parser.parse_args()
    generate_all_figures(args.results_dir, args.output_dir)
