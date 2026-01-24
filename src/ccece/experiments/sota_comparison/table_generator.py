"""
Table Generator Module for SOTA Comparison

Generates publication-ready tables in:
- CSV format
- LaTeX format (for paper)
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from .statistical_tests import PairwiseComparison


def generate_classification_table(
    results: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str,
    our_method: str = "MultiHeadProtoNet",
    comparisons: Optional[List[PairwiseComparison]] = None,
) -> pd.DataFrame:
    """
    Generate Table 1: Classification Performance.

    Args:
        results: Dict mapping model_name -> {metric: {'mean': x, 'std': y}}
        output_dir: Directory to save tables
        our_method: Name of our method
        comparisons: Optional pairwise comparisons for significance markers

    Returns:
        DataFrame with the table
    """
    metrics = ['accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1_score', 'auc_roc']
    metric_display = {
        'accuracy': 'Accuracy',
        'balanced_accuracy': 'Bal. Acc',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity',
        'f1_score': 'F1-Score',
        'auc_roc': 'AUC-ROC',
    }

    # Prepare significance markers
    sig_markers = {}
    if comparisons:
        for comp in comparisons:
            sig_markers[comp.baseline_name] = not comp.is_significant  # True if tied

    # Define model order
    model_order = [
        '1D-CNN', 'LSTM',  # Simple baselines
        'InceptionTime', 'ROCKET', 'TST', 'TimesNet', 'ConvTran', 'PatchTST',  # SOTA
        our_method,  # Ours
    ]

    # Filter to models in results
    model_order = [m for m in model_order if m in results]

    # Find best value for each metric
    best_values = {}
    for metric in metrics:
        values = [results[m][metric]['mean'] for m in model_order if metric in results[m]]
        best_values[metric] = max(values) if values else 0

    # Build rows
    rows = []
    for model in model_order:
        row = {'Model': model}
        for metric in metrics:
            if metric in results[model]:
                mean = results[model][metric]['mean']
                std = results[model][metric]['std']

                # Format as percentage (except AUC)
                if metric == 'auc_roc':
                    value_str = f"{mean:.3f} ± {std:.3f}"
                else:
                    value_str = f"{mean*100:.1f} ± {std*100:.1f}"

                # Add significance marker (dagger) if tied with best
                is_best = abs(mean - best_values[metric]) < 0.001
                is_tied = model in sig_markers and sig_markers[model]

                if is_best or (model != our_method and is_tied):
                    value_str = f"**{value_str}**"

                row[metric_display[metric]] = value_str
            else:
                row[metric_display[metric]] = '-'

        rows.append(row)

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(output_dir, 'table1_classification.csv')
    df.to_csv(csv_path, index=False)

    # Save LaTeX
    latex_path = os.path.join(output_dir, 'table1_classification.tex')
    _save_latex_table(df, latex_path, caption="Classification Performance Comparison",
                      label="tab:classification")

    return df


def generate_computational_table(
    results: Dict[str, Dict[str, Any]],
    output_dir: str,
    our_method: str = "MultiHeadProtoNet",
) -> pd.DataFrame:
    """
    Generate Table 2: Computational Efficiency.

    Args:
        results: Dict mapping model_name -> computational metrics
        output_dir: Directory to save tables
        our_method: Name of our method

    Returns:
        DataFrame with the table
    """
    # Define model order
    model_order = [
        '1D-CNN', 'LSTM',
        'InceptionTime', 'ROCKET', 'TST', 'TimesNet', 'ConvTran', 'PatchTST',
        our_method,
    ]
    model_order = [m for m in model_order if m in results]

    rows = []
    for model in model_order:
        r = results[model]
        row = {
            'Model': model,
            'Params (K)': f"{r['parameters'] / 1000:.1f}" if r.get('parameters') else '-',
            'FLOPs (M)': f"{r['flops'] / 1e6:.1f}" if r.get('flops') else '-',
            'Train (min)': f"{r['training_time'] / 60:.1f}" if r.get('training_time') else '-',
            'Infer (ms)': f"{r['inference_time']:.2f}" if r.get('inference_time') else '-',
            'GPU Mem (MB)': f"{r['gpu_memory']:.0f}" if r.get('gpu_memory') else '-',
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(output_dir, 'table2_computational.csv')
    df.to_csv(csv_path, index=False)

    # Save LaTeX
    latex_path = os.path.join(output_dir, 'table2_computational.tex')
    _save_latex_table(df, latex_path, caption="Computational Efficiency Comparison",
                      label="tab:computational")

    return df


def generate_significance_table(
    comparisons: List[PairwiseComparison],
    output_dir: str,
) -> pd.DataFrame:
    """
    Generate Table 3: Statistical Significance.

    Args:
        comparisons: List of pairwise comparisons
        output_dir: Directory to save tables

    Returns:
        DataFrame with the table
    """
    rows = []
    for comp in comparisons:
        sig_str = "Yes" if comp.is_significant else "No"
        if comp.is_significant:
            sig_str += " (" + ("↑" if comp.is_better else "↓") + ")"

        row = {
            'Baseline': comp.baseline_name,
            'Baseline Acc': f"{comp.baseline_mean*100:.1f} ± {comp.baseline_std*100:.1f}",
            'Ours Acc': f"{comp.ours_mean*100:.1f} ± {comp.ours_std*100:.1f}",
            'Diff': f"{comp.difference*100:+.1f}",
            'p-value': f"{comp.ttest_pvalue:.3f}",
            "Cohen's d": f"{comp.cohens_d:.2f}",
            'Significant': sig_str,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(output_dir, 'table3_significance.csv')
    df.to_csv(csv_path, index=False)

    # Save LaTeX
    latex_path = os.path.join(output_dir, 'table3_significance.tex')
    _save_latex_table(df, latex_path, caption="Statistical Significance Tests (Ours vs. Baselines)",
                      label="tab:significance")

    return df


def _save_latex_table(
    df: pd.DataFrame,
    path: str,
    caption: str,
    label: str,
) -> None:
    """Save DataFrame as LaTeX table."""
    # Convert to LaTeX
    latex = df.to_latex(index=False, escape=False)

    # Wrap in table environment
    latex_full = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\small
{latex}
\\end{{table}}
"""

    with open(path, 'w') as f:
        f.write(latex_full)


def generate_all_tables(
    classification_results: Dict[str, Dict[str, Dict[str, float]]],
    computational_results: Dict[str, Dict[str, Any]],
    comparisons: List[PairwiseComparison],
    output_dir: str,
    our_method: str = "MultiHeadProtoNet",
) -> Dict[str, pd.DataFrame]:
    """
    Generate all tables for the paper.

    Args:
        classification_results: Classification metrics per model
        computational_results: Computational metrics per model
        comparisons: Statistical comparisons
        output_dir: Output directory
        our_method: Name of our method

    Returns:
        Dict of DataFrames for each table
    """
    os.makedirs(output_dir, exist_ok=True)

    tables = {}

    tables['classification'] = generate_classification_table(
        classification_results, output_dir, our_method, comparisons
    )

    tables['computational'] = generate_computational_table(
        computational_results, output_dir, our_method
    )

    tables['significance'] = generate_significance_table(
        comparisons, output_dir
    )

    return tables
