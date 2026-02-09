"""
Statistical Analysis for MHTPN Ablation Study

Generates:
- Tables 4-7 in CSV and LaTeX format
- Statistical tests (paired t-test, Cohen's d, p-values)
- Summary statistics for paper
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def paired_t_test(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    """Perform paired t-test, return t-statistic and p-value."""
    t_stat, p_value = stats.ttest_rel(group1, group2)
    return float(t_stat), float(p_value)


def analyze_results(results_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Analyze ablation results and generate summary statistics.

    Args:
        results_path: Path to full_results.json
        output_dir: Directory to save analysis outputs

    Returns:
        Dictionary with analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)

    with open(results_path, 'r') as f:
        results = json.load(f)

    analysis = {
        'metadata': results['metadata'],
        'tables': {},
        'statistical_tests': {},
    }

    # Analyze each ablation
    for ablation_name in ['n_segments', 'trajectory', 'loss', 'weighting']:
        if ablation_name not in results['ablations']:
            continue

        ablation_results = results['ablations'][ablation_name]

        if ablation_name == 'n_segments':
            table, tests = analyze_n_segments(ablation_results, output_dir)
            analysis['tables']['table4_n_segments'] = table
            analysis['statistical_tests']['n_segments'] = tests

        elif ablation_name == 'trajectory':
            table, tests = analyze_trajectory(ablation_results, output_dir)
            analysis['tables']['table5_trajectory'] = table
            analysis['statistical_tests']['trajectory'] = tests

        elif ablation_name == 'loss':
            table, tests = analyze_loss(ablation_results, output_dir)
            analysis['tables']['table6_loss'] = table
            analysis['statistical_tests']['loss'] = tests

        elif ablation_name == 'weighting':
            table, tests = analyze_weighting(ablation_results, output_dir)
            analysis['tables']['table7_weighting'] = table
            analysis['statistical_tests']['weighting'] = tests

    # Save analysis summary
    summary_path = os.path.join(output_dir, 'analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    # Generate summary CSV
    generate_summary_csv(results, output_dir)

    return analysis


def get_fold_accuracies(result: Dict) -> np.ndarray:
    """Extract fold accuracies from a variant result."""
    return np.array([r['accuracy'] for r in result['fold_results']])


def find_default(results: List[Dict]) -> Optional[Dict]:
    """Find the default configuration in a list of results."""
    for r in results:
        if r['config'].get('is_default', False):
            return r
    return None


def analyze_n_segments(
    ablation_results: List[Dict],
    output_dir: str,
) -> Tuple[Dict, Dict]:
    """
    Analyze Ablation 1: n_segments.

    Generates Table 4.
    """
    # Find default (n_segments=8)
    default = find_default(ablation_results)
    default_accs = get_fold_accuracies(default)

    rows = []
    tests = {}

    for result in ablation_results:
        config = result['config']
        n_seg = config['n_segments']
        agg = result['aggregate']

        accs = get_fold_accuracies(result)

        # Statistical test vs default
        if not config.get('is_default', False):
            t_stat, p_value = paired_t_test(accs, default_accs)
            d = cohens_d(accs, default_accs)
            delta = agg['mean_accuracy'] - default['aggregate']['mean_accuracy']
        else:
            t_stat, p_value, d = None, None, None
            delta = 0.0

        rows.append({
            'n_segments': n_seg,
            'Accuracy (%)': f"{agg['mean_accuracy']*100:.1f} +/- {agg['std_accuracy']*100:.1f}",
            'accuracy_mean': agg['mean_accuracy'],
            'accuracy_std': agg['std_accuracy'],
            'Temporal Pass': agg['temporal_pass_rate'],
            'Late-Early': f"{agg['mean_late_minus_early']:+.3f}",
            'late_early_value': agg['mean_late_minus_early'],
            'Delta vs Default': f"{delta*100:+.1f}%" if delta != 0 else "-",
            'delta_value': delta,
            'p-value': p_value,
            'Cohen_d': d,
            'is_default': config.get('is_default', False),
        })

        tests[f'n_segments={n_seg}'] = {
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': d,
            'significant': p_value < 0.05 if p_value else None,
        }

    # Sort by n_segments
    rows.sort(key=lambda x: x['n_segments'])

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save CSV
    csv_cols = ['n_segments', 'Accuracy (%)', 'Temporal Pass', 'Late-Early', 'Delta vs Default', 'p-value', 'Cohen_d']
    df[csv_cols].to_csv(
        os.path.join(output_dir, 'tables', 'table4_n_segments.csv'),
        index=False
    )

    # Generate LaTeX
    latex = generate_latex_table(
        df,
        caption="Effect of Temporal Segmentation (n\\_segments)",
        label="tab:ablation_n_segments",
        columns=['n_segments', 'Accuracy (%)', 'Temporal Pass', 'Late-Early', 'Delta vs Default'],
        highlight_default=True,
    )
    with open(os.path.join(output_dir, 'tables', 'table4_n_segments.tex'), 'w') as f:
        f.write(latex)

    return rows, tests


def analyze_trajectory(
    ablation_results: List[Dict],
    output_dir: str,
) -> Tuple[Dict, Dict]:
    """
    Analyze Ablation 2: Trajectory vs Static.

    Generates Table 5.
    """
    default = find_default(ablation_results)
    default_accs = get_fold_accuracies(default)

    rows = []
    tests = {}

    for result in ablation_results:
        config = result['config']
        variant = config['variant_name']
        agg = result['aggregate']

        accs = get_fold_accuracies(result)

        if not config.get('is_default', False):
            t_stat, p_value = paired_t_test(accs, default_accs)
            d = cohens_d(accs, default_accs)
            delta = agg['mean_accuracy'] - default['aggregate']['mean_accuracy']
        else:
            t_stat, p_value, d = None, None, None
            delta = 0.0

        rows.append({
            'Type': variant,
            'Accuracy (%)': f"{agg['mean_accuracy']*100:.1f} +/- {agg['std_accuracy']*100:.1f}",
            'accuracy_mean': agg['mean_accuracy'],
            'accuracy_std': agg['std_accuracy'],
            'Velocity Norm': f"{agg['mean_velocity_norm']:.3f}",
            'velocity_norm': agg['mean_velocity_norm'],
            'Temporal Pass': agg['temporal_pass_rate'],
            'Delta vs Default': f"{delta*100:+.1f}%" if delta != 0 else "-",
            'delta_value': delta,
            'p-value': p_value,
            'Cohen_d': d,
            'is_default': config.get('is_default', False),
        })

        tests[variant] = {
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': d,
            'significant': p_value < 0.05 if p_value else None,
        }

    df = pd.DataFrame(rows)

    # Save CSV
    csv_cols = ['Type', 'Accuracy (%)', 'Velocity Norm', 'Temporal Pass', 'Delta vs Default', 'p-value', 'Cohen_d']
    df[csv_cols].to_csv(
        os.path.join(output_dir, 'tables', 'table5_trajectory.csv'),
        index=False
    )

    # Generate LaTeX
    latex = generate_latex_table(
        df,
        caption="Trajectory vs Static Prototypes",
        label="tab:ablation_trajectory",
        columns=['Type', 'Accuracy (%)', 'Velocity Norm', 'Temporal Pass', 'Delta vs Default'],
        highlight_default=True,
    )
    with open(os.path.join(output_dir, 'tables', 'table5_trajectory.tex'), 'w') as f:
        f.write(latex)

    return rows, tests


def analyze_loss(
    ablation_results: List[Dict],
    output_dir: str,
) -> Tuple[Dict, Dict]:
    """
    Analyze Ablation 3: Loss Components.

    Generates Table 6.
    """
    default = find_default(ablation_results)
    default_accs = get_fold_accuracies(default)

    rows = []
    tests = {}

    # Define order
    order = ['CE only', 'CE + Cluster', 'CE + Separation', 'CE + Cl + Sep', 'Full']
    results_by_name = {r['config']['variant_name']: r for r in ablation_results}

    for variant_name in order:
        if variant_name not in results_by_name:
            continue

        result = results_by_name[variant_name]
        config = result['config']
        agg = result['aggregate']

        accs = get_fold_accuracies(result)

        if not config.get('is_default', False):
            t_stat, p_value = paired_t_test(accs, default_accs)
            d = cohens_d(accs, default_accs)
            delta = agg['mean_accuracy'] - default['aggregate']['mean_accuracy']
        else:
            t_stat, p_value, d = None, None, None
            delta = 0.0

        # Mark significant with asterisks
        sig_marker = ""
        if p_value is not None:
            if p_value < 0.001:
                sig_marker = "***"
            elif p_value < 0.01:
                sig_marker = "**"
            elif p_value < 0.05:
                sig_marker = "*"

        rows.append({
            'Loss': variant_name,
            'Accuracy (%)': f"{agg['mean_accuracy']*100:.1f} +/- {agg['std_accuracy']*100:.1f}",
            'accuracy_mean': agg['mean_accuracy'],
            'accuracy_std': agg['std_accuracy'],
            'Temporal Pass': agg['temporal_pass_rate'],
            'Delta vs Full': f"{delta*100:+.1f}%{sig_marker}" if delta != 0 else "-",
            'delta_value': delta,
            'p-value': p_value,
            'Cohen_d': d,
            'is_default': config.get('is_default', False),
        })

        tests[variant_name] = {
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': d,
            'significant': p_value < 0.05 if p_value else None,
        }

    df = pd.DataFrame(rows)

    # Save CSV
    csv_cols = ['Loss', 'Accuracy (%)', 'Temporal Pass', 'Delta vs Full', 'p-value', 'Cohen_d']
    df[csv_cols].to_csv(
        os.path.join(output_dir, 'tables', 'table6_loss.csv'),
        index=False
    )

    # Generate LaTeX
    latex = generate_latex_table(
        df,
        caption="Effect of Loss Components",
        label="tab:ablation_loss",
        columns=['Loss', 'Accuracy (%)', 'Temporal Pass', 'Delta vs Full'],
        highlight_default=True,
    )
    with open(os.path.join(output_dir, 'tables', 'table6_loss.tex'), 'w') as f:
        f.write(latex)

    return rows, tests


def analyze_weighting(
    ablation_results: List[Dict],
    output_dir: str,
) -> Tuple[Dict, Dict]:
    """
    Analyze Ablation 4: Segment Weighting.

    Generates Table 7.
    """
    default = find_default(ablation_results)
    default_accs = get_fold_accuracies(default)

    rows = []
    tests = {}

    # Define order
    order = ['Uniform', 'Padding-aware', 'Learned']
    results_by_name = {r['config']['variant_name']: r for r in ablation_results}

    for variant_name in order:
        if variant_name not in results_by_name:
            continue

        result = results_by_name[variant_name]
        config = result['config']
        agg = result['aggregate']

        accs = get_fold_accuracies(result)

        if not config.get('is_default', False):
            t_stat, p_value = paired_t_test(accs, default_accs)
            d = cohens_d(accs, default_accs)
            delta = agg['mean_accuracy'] - default['aggregate']['mean_accuracy']
        else:
            t_stat, p_value, d = None, None, None
            delta = 0.0

        rows.append({
            'Strategy': variant_name,
            'Accuracy (%)': f"{agg['mean_accuracy']*100:.1f} +/- {agg['std_accuracy']*100:.1f}",
            'accuracy_mean': agg['mean_accuracy'],
            'accuracy_std': agg['std_accuracy'],
            'Temporal Pass': agg['temporal_pass_rate'],
            'Delta vs Default': f"{delta*100:+.1f}%" if delta != 0 else "-",
            'delta_value': delta,
            'p-value': p_value,
            'Cohen_d': d,
            'is_default': config.get('is_default', False),
        })

        tests[variant_name] = {
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': d,
            'significant': p_value < 0.05 if p_value else None,
        }

    df = pd.DataFrame(rows)

    # Save CSV
    csv_cols = ['Strategy', 'Accuracy (%)', 'Temporal Pass', 'Delta vs Default', 'p-value', 'Cohen_d']
    df[csv_cols].to_csv(
        os.path.join(output_dir, 'tables', 'table7_weighting.csv'),
        index=False
    )

    # Generate LaTeX
    latex = generate_latex_table(
        df,
        caption="Effect of Segment Weighting Strategy",
        label="tab:ablation_weighting",
        columns=['Strategy', 'Accuracy (%)', 'Temporal Pass', 'Delta vs Default'],
        highlight_default=True,
    )
    with open(os.path.join(output_dir, 'tables', 'table7_weighting.tex'), 'w') as f:
        f.write(latex)

    return rows, tests


def generate_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    columns: List[str],
    highlight_default: bool = True,
) -> str:
    """Generate LaTeX table code."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")

    # Column specification
    col_spec = 'l' + 'c' * (len(columns) - 1)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header
    header = " & ".join(columns)
    lines.append(f"{header} \\\\")
    lines.append("\\midrule")

    # Data rows
    for _, row in df.iterrows():
        values = []
        for col in columns:
            val = str(row[col])
            # Bold default row
            if highlight_default and row.get('is_default', False):
                val = f"\\textbf{{{val}}}"
            values.append(val)
        lines.append(" & ".join(values) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_summary_csv(results: Dict, output_dir: str):
    """Generate a summary CSV with all variants."""
    rows = []

    for ablation_name, ablation_results in results['ablations'].items():
        for result in ablation_results:
            config = result['config']
            agg = result['aggregate']

            rows.append({
                'Ablation': ablation_name,
                'Variant ID': config['ablation_id'],
                'Variant': config['variant_name'],
                'Is Default': config.get('is_default', False),
                'Mean Accuracy': agg['mean_accuracy'],
                'Std Accuracy': agg['std_accuracy'],
                'Mean Sensitivity': agg['mean_sensitivity'],
                'Mean Specificity': agg['mean_specificity'],
                'Mean F1': agg['mean_f1'],
                'Mean AUC-ROC': agg['mean_auc_roc'],
                'Temporal Pass': agg['n_temporal_pass'],
                'Mean Late-Early': agg['mean_late_minus_early'],
                'Mean Velocity Norm': agg['mean_velocity_norm'],
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'ablation_summary.csv'), index=False)


def statistical_tests(results_path: str) -> Dict[str, Any]:
    """
    Run comprehensive statistical tests on ablation results.

    Returns dictionary with all test results.
    """
    with open(results_path, 'r') as f:
        results = json.load(f)

    all_tests = {}

    for ablation_name, ablation_results in results['ablations'].items():
        default = find_default(ablation_results)
        if default is None:
            continue

        default_accs = get_fold_accuracies(default)
        tests = {}

        for result in ablation_results:
            if result['config'].get('is_default', False):
                continue

            variant_name = result['config']['variant_name']
            accs = get_fold_accuracies(result)

            t_stat, p_value = paired_t_test(accs, default_accs)
            d = cohens_d(accs, default_accs)

            tests[variant_name] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': d,
                'significant_0.05': p_value < 0.05,
                'significant_0.01': p_value < 0.01,
                'significant_0.001': p_value < 0.001,
                'effect_size': (
                    'large' if abs(d) >= 0.8 else
                    'medium' if abs(d) >= 0.5 else
                    'small' if abs(d) >= 0.2 else
                    'negligible'
                ),
                'mean_diff': float(np.mean(accs) - np.mean(default_accs)),
            }

        all_tests[ablation_name] = tests

    return all_tests


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analysis.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]
    results_path = os.path.join(results_dir, 'full_results.json')

    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found")
        sys.exit(1)

    analysis = analyze_results(results_path, results_dir)
    print("Analysis complete. Tables saved to:", os.path.join(results_dir, 'tables'))
