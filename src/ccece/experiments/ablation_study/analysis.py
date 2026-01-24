"""
Analysis Module for Ablation Study

Generates:
- Statistical tests (paired t-tests comparing variants to default)
- Publication-ready tables (CSV + LaTeX)
- Summary statistics
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats


@dataclass
class AblationComparison:
    """Result of comparing a variant to the default."""
    variant_name: str
    variant_mean: float
    variant_std: float
    default_mean: float
    default_std: float
    difference: float
    difference_percent: float
    ttest_pvalue: float
    cohens_d: float
    is_significant: bool
    is_better: bool


def paired_ttest(
    variant_values: List[float],
    default_values: List[float],
    alpha: float = 0.05,
) -> Tuple[float, float, bool, bool]:
    """
    Perform paired t-test comparing variant to default.

    Returns:
        pvalue, cohen's d, is_significant, is_better
    """
    if len(variant_values) != len(default_values):
        raise ValueError("Must have same number of folds for paired test")

    n = len(variant_values)
    if n < 2:
        return 1.0, 0.0, False, False

    # Paired t-test
    t_stat, pvalue = stats.ttest_rel(variant_values, default_values)

    # Cohen's d for paired samples
    diff = np.array(variant_values) - np.array(default_values)
    cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)

    is_significant = pvalue < alpha
    is_better = np.mean(variant_values) > np.mean(default_values)

    return pvalue, cohens_d, is_significant, is_better


def compare_variant_to_default(
    variant_name: str,
    variant_results: Dict[str, Any],
    default_results: Dict[str, Any],
    metric: str = 'accuracy',
) -> AblationComparison:
    """
    Compare a variant to the default configuration.

    Args:
        variant_name: Name of the variant
        variant_results: Results for the variant
        default_results: Results for the default
        metric: Metric to compare

    Returns:
        AblationComparison object
    """
    variant_values = variant_results['aggregated'][metric]['values']
    default_values = default_results['aggregated'][metric]['values']

    variant_mean = variant_results['aggregated'][metric]['mean']
    variant_std = variant_results['aggregated'][metric]['std']
    default_mean = default_results['aggregated'][metric]['mean']
    default_std = default_results['aggregated'][metric]['std']

    difference = variant_mean - default_mean
    difference_percent = (difference / default_mean) * 100 if default_mean > 0 else 0

    pvalue, cohens_d, is_significant, is_better = paired_ttest(
        variant_values, default_values
    )

    return AblationComparison(
        variant_name=variant_name,
        variant_mean=variant_mean,
        variant_std=variant_std,
        default_mean=default_mean,
        default_std=default_std,
        difference=difference,
        difference_percent=difference_percent,
        ttest_pvalue=pvalue,
        cohens_d=cohens_d,
        is_significant=is_significant,
        is_better=is_better,
    )


def generate_ablation_table(
    ablation_name: str,
    variants: Dict[str, Any],
    output_dir: str,
    metrics_to_show: Optional[List[str]] = None,
    table_number: int = 4,
) -> pd.DataFrame:
    """
    Generate a table for a single ablation study.

    Args:
        ablation_name: Name of the ablation
        variants: Dict of variant_name -> results
        output_dir: Output directory
        metrics_to_show: Metrics to include in table
        table_number: Table number for filename

    Returns:
        DataFrame with the table
    """
    if metrics_to_show is None:
        metrics_to_show = ['accuracy', 'balanced_accuracy', 'f1_score']

    metric_display = {
        'accuracy': 'Accuracy',
        'balanced_accuracy': 'Bal. Acc',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity',
        'f1_score': 'F1-Score',
        'auc_roc': 'AUC-ROC',
        'alignment': 'Alignment',
    }

    # Find default for comparison
    default_results = None
    default_name = None
    for variant_name, data in variants.items():
        if data.get('is_default'):
            default_results = data
            default_name = variant_name
            break

    # Build rows
    rows = []
    comparisons = []

    for variant_name, data in variants.items():
        if 'error' in data.get('aggregated', {}):
            continue

        row = {'Variant': variant_name}

        for metric in metrics_to_show:
            if metric in data['aggregated']:
                mean = data['aggregated'][metric]['mean']
                std = data['aggregated'][metric]['std']

                if metric == 'auc_roc':
                    value_str = f"{mean:.3f}"
                else:
                    value_str = f"{mean*100:.1f} ± {std*100:.1f}"

                # Bold if default
                if data.get('is_default'):
                    value_str = f"**{value_str}**"

                row[metric_display[metric]] = value_str
            else:
                row[metric_display[metric]] = '-'

        # Compute difference vs default
        if default_results and not data.get('is_default'):
            comp = compare_variant_to_default(variant_name, data, default_results)
            comparisons.append(comp)

            delta = comp.difference_percent
            sig_marker = "*" if comp.is_significant else ""
            row['Δ vs Default'] = f"{delta:+.1f}%{sig_marker}"
        else:
            row['Δ vs Default'] = '—'

        rows.append(row)

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(output_dir, f'table{table_number}_{ablation_name}.csv')
    df.to_csv(csv_path, index=False)

    # Save LaTeX
    latex_path = os.path.join(output_dir, f'table{table_number}_{ablation_name}.tex')
    _save_latex_table(df, latex_path,
                      caption=f"Ablation Study: {_format_ablation_name(ablation_name)}",
                      label=f"tab:ablation_{ablation_name}")

    # Save comparisons
    if comparisons:
        _save_comparisons(comparisons, output_dir, ablation_name, table_number)

    return df


def _format_ablation_name(ablation_name: str) -> str:
    """Format ablation name for display."""
    name_map = {
        'n_heads': 'Number of Heads',
        'classification': 'Classification Method',
        'loss': 'Loss Components',
        'fusion': 'Fusion Strategy',
        'encoder': 'Encoder Architecture',
    }
    return name_map.get(ablation_name, ablation_name.title())


def _save_latex_table(
    df: pd.DataFrame,
    path: str,
    caption: str,
    label: str,
) -> None:
    """Save DataFrame as LaTeX table."""
    latex = df.to_latex(index=False, escape=False)

    latex_full = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\small
{latex}
\\vspace{{2pt}}
\\footnotesize{{* indicates statistically significant difference (p < 0.05)}}
\\end{{table}}
"""

    with open(path, 'w') as f:
        f.write(latex_full)


def _save_comparisons(
    comparisons: List[AblationComparison],
    output_dir: str,
    ablation_name: str,
    table_number: int,
) -> None:
    """Save detailed comparison results."""
    rows = []
    for comp in comparisons:
        row = {
            'Variant': comp.variant_name,
            'Variant Acc': f"{comp.variant_mean*100:.1f} ± {comp.variant_std*100:.1f}",
            'Default Acc': f"{comp.default_mean*100:.1f} ± {comp.default_std*100:.1f}",
            'Diff': f"{comp.difference*100:+.2f}",
            'Diff %': f"{comp.difference_percent:+.1f}%",
            'p-value': f"{comp.ttest_pvalue:.4f}",
            "Cohen's d": f"{comp.cohens_d:.3f}",
            'Significant': 'Yes' if comp.is_significant else 'No',
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f'table{table_number}_{ablation_name}_significance.csv')
    df.to_csv(csv_path, index=False)


def generate_n_heads_table(
    variants: Dict[str, Any],
    output_dir: str,
) -> pd.DataFrame:
    """
    Generate Table 4: Number of Heads Ablation.

    Includes: Accuracy, Bal. Acc, F1-Score, Alignment, Δ vs Default
    """
    metrics = ['accuracy', 'balanced_accuracy', 'f1_score']

    # Add alignment if available
    has_alignment = any(
        'alignment' in v['aggregated']
        for v in variants.values()
        if 'aggregated' in v
    )

    if has_alignment:
        metrics.append('alignment')

    return generate_ablation_table(
        ablation_name='n_heads',
        variants=variants,
        output_dir=output_dir,
        metrics_to_show=metrics,
        table_number=4,
    )


def generate_classification_table(
    variants: Dict[str, Any],
    output_dir: str,
) -> pd.DataFrame:
    """
    Generate Table 5: Prototype vs Standard Classification.
    """
    return generate_ablation_table(
        ablation_name='classification',
        variants=variants,
        output_dir=output_dir,
        metrics_to_show=['accuracy', 'balanced_accuracy', 'f1_score'],
        table_number=5,
    )


def generate_loss_table(
    variants: Dict[str, Any],
    output_dir: str,
) -> pd.DataFrame:
    """
    Generate Table 6: Loss Component Analysis.
    """
    metrics = ['accuracy', 'balanced_accuracy']

    # Add alignment if available
    has_alignment = any(
        'alignment' in v['aggregated']
        for v in variants.values()
        if 'aggregated' in v
    )

    if has_alignment:
        metrics.append('alignment')

    return generate_ablation_table(
        ablation_name='loss',
        variants=variants,
        output_dir=output_dir,
        metrics_to_show=metrics,
        table_number=6,
    )


def generate_fusion_table(
    variants: Dict[str, Any],
    output_dir: str,
) -> pd.DataFrame:
    """
    Generate Table 7: Fusion Strategy Comparison.
    """
    return generate_ablation_table(
        ablation_name='fusion',
        variants=variants,
        output_dir=output_dir,
        metrics_to_show=['accuracy', 'balanced_accuracy', 'f1_score'],
        table_number=7,
    )


def generate_encoder_table(
    variants: Dict[str, Any],
    output_dir: str,
) -> pd.DataFrame:
    """
    Generate Table 8: Encoder Architecture Sensitivity.

    Includes parameter count.
    """
    df = generate_ablation_table(
        ablation_name='encoder',
        variants=variants,
        output_dir=output_dir,
        metrics_to_show=['accuracy', 'balanced_accuracy'],
        table_number=8,
    )

    # Add parameter count column
    params = []
    for variant_name, data in variants.items():
        if 'aggregated' in data and 'num_params' in data['aggregated']:
            p = data['aggregated']['num_params']
            params.append(f"{p/1000:.1f}K")
        else:
            params.append('-')

    # Insert params column after Variant
    df.insert(1, 'Params', params[:len(df)])

    # Re-save with params
    csv_path = os.path.join(output_dir, 'table8_encoder.csv')
    df.to_csv(csv_path, index=False)

    return df


def generate_all_ablation_tables(
    all_results: Dict[str, Dict[str, Any]],
    output_dir: str,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Generate all ablation study tables.

    Args:
        all_results: Results from run_ablation_study()
        output_dir: Output directory
        verbose: Print progress

    Returns:
        Dict of DataFrames for each table
    """
    os.makedirs(output_dir, exist_ok=True)

    tables = {}

    # Table 4: Number of Heads
    if 'n_heads' in all_results:
        if verbose:
            print("  Generating Table 4: Number of Heads...")
        tables['n_heads'] = generate_n_heads_table(all_results['n_heads'], output_dir)

    # Table 5: Classification Method
    if 'classification' in all_results:
        if verbose:
            print("  Generating Table 5: Classification Method...")
        tables['classification'] = generate_classification_table(
            all_results['classification'], output_dir
        )

    # Table 6: Loss Components
    if 'loss' in all_results:
        if verbose:
            print("  Generating Table 6: Loss Components...")
        tables['loss'] = generate_loss_table(all_results['loss'], output_dir)

    # Table 7: Fusion Strategy
    if 'fusion' in all_results:
        if verbose:
            print("  Generating Table 7: Fusion Strategy...")
        tables['fusion'] = generate_fusion_table(all_results['fusion'], output_dir)

    # Table 8: Encoder Architecture
    if 'encoder' in all_results:
        if verbose:
            print("  Generating Table 8: Encoder Architecture...")
        tables['encoder'] = generate_encoder_table(all_results['encoder'], output_dir)

    # Generate summary table
    if verbose:
        print("  Generating Summary Table...")
    tables['summary'] = generate_summary_table(all_results, output_dir)

    return tables


def generate_summary_table(
    all_results: Dict[str, Dict[str, Any]],
    output_dir: str,
) -> pd.DataFrame:
    """
    Generate a summary table showing key findings from each ablation.
    """
    rows = []

    ablation_questions = {
        'n_heads': 'Is multi-head necessary?',
        'classification': 'Does prototype learning help?',
        'loss': 'Which losses are essential?',
        'fusion': 'Best fusion strategy?',
        'encoder': 'Sensitivity to architecture?',
    }

    for ablation_name, variants in all_results.items():
        # Find default and best non-default
        default_acc = None
        best_variant = None
        best_acc = -1

        for variant_name, data in variants.items():
            if 'error' in data.get('aggregated', {}):
                continue

            acc = data['aggregated']['accuracy']['mean']

            if data.get('is_default'):
                default_acc = acc
                default_name = variant_name
            else:
                if acc > best_acc:
                    best_acc = acc
                    best_variant = variant_name

        if default_acc is None:
            continue

        # Determine key finding
        if best_variant and best_acc > default_acc:
            finding = f"{best_variant} slightly better (+{(best_acc-default_acc)*100:.1f}%)"
        else:
            finding = f"Default ({default_name}) is optimal"

        row = {
            'Ablation': _format_ablation_name(ablation_name),
            'Question': ablation_questions.get(ablation_name, ''),
            'Default': default_name,
            'Default Acc': f"{default_acc*100:.1f}%",
            'Key Finding': finding,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    csv_path = os.path.join(output_dir, 'ablation_summary.csv')
    df.to_csv(csv_path, index=False)

    return df


def compute_effect_sizes(
    all_results: Dict[str, Dict[str, Any]],
) -> Dict[str, List[AblationComparison]]:
    """
    Compute effect sizes for all ablations.

    Returns:
        Dict mapping ablation_name -> list of comparisons
    """
    all_comparisons = {}

    for ablation_name, variants in all_results.items():
        # Find default
        default_results = None
        for variant_name, data in variants.items():
            if data.get('is_default'):
                default_results = data
                break

        if default_results is None:
            continue

        comparisons = []
        for variant_name, data in variants.items():
            if data.get('is_default') or 'error' in data.get('aggregated', {}):
                continue

            comp = compare_variant_to_default(variant_name, data, default_results)
            comparisons.append(comp)

        all_comparisons[ablation_name] = comparisons

    return all_comparisons
