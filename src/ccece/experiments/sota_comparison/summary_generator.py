"""
Cross-Dataset Summary Generator

Generates comprehensive summaries across all datasets for the SOTA comparison.
Produces tables, figures, and statistics suitable for paper inclusion.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import csv


@dataclass
class DatasetSummary:
    """Summary of results for a single dataset."""
    name: str
    domain: str
    n_samples: int
    n_classes: int
    mhtpn_acc: float
    mhtpn_std: float
    best_baseline: str
    baseline_acc: float
    baseline_std: float
    improvement: float
    rank: int
    all_model_results: Dict[str, Dict[str, float]]


def load_dataset_results(results_path: str) -> Dict[str, Any]:
    """Load results from a dataset's full_results.json."""
    with open(results_path, 'r') as f:
        return json.load(f)


def compute_dataset_summary(results: Dict[str, Any]) -> DatasetSummary:
    """
    Compute summary statistics for a single dataset's results.

    Args:
        results: Loaded results dict from full_results.json

    Returns:
        DatasetSummary with key metrics
    """
    dataset_info = results['dataset']
    classification = results['classification']

    # Get MHTPN performance
    if 'MHTPN' in classification:
        mhtpn_acc = classification['MHTPN']['accuracy']['mean']
        mhtpn_std = classification['MHTPN']['accuracy']['std']
    else:
        mhtpn_acc = 0.0
        mhtpn_std = 0.0

    # Find best baseline and collect all results
    best_baseline = None
    best_baseline_acc = 0.0
    best_baseline_std = 0.0
    all_model_results = {}

    for model_name, metrics in classification.items():
        acc = metrics['accuracy']['mean']
        std = metrics['accuracy']['std']
        all_model_results[model_name] = {'accuracy': acc, 'std': std}

        if model_name != 'MHTPN' and acc > best_baseline_acc:
            best_baseline_acc = acc
            best_baseline_std = std
            best_baseline = model_name

    # Calculate improvement over best baseline
    improvement = (mhtpn_acc - best_baseline_acc) * 100

    # Determine MHTPN rank
    all_accs = [m['accuracy']['mean'] for m in classification.values()]
    rank = sorted(all_accs, reverse=True).index(mhtpn_acc) + 1

    return DatasetSummary(
        name=dataset_info['name'],
        domain=dataset_info['domain'],
        n_samples=dataset_info['n_samples'],
        n_classes=dataset_info['n_classes'],
        mhtpn_acc=mhtpn_acc,
        mhtpn_std=mhtpn_std,
        best_baseline=best_baseline or 'N/A',
        baseline_acc=best_baseline_acc,
        baseline_std=best_baseline_std,
        improvement=improvement,
        rank=rank,
        all_model_results=all_model_results,
    )


def generate_paper_table(
    summaries: List[DatasetSummary],
    output_path: str,
    format: str = 'latex',
) -> str:
    """
    Generate a publication-ready table of results.

    Args:
        summaries: List of DatasetSummary objects
        output_path: Path to save the table
        format: 'latex' or 'csv'

    Returns:
        Table as string
    """
    if format == 'latex':
        # Generate LaTeX table
        lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            r'\caption{MHTPN Performance Across Datasets}',
            r'\label{tab:cross_dataset}',
            r'\begin{tabular}{lcrcccc}',
            r'\toprule',
            r'Dataset & Domain & Samples & MHTPN Acc. & Best Baseline & Improvement & Rank \\',
            r'\midrule',
        ]

        for s in summaries:
            mhtpn_str = f"${s.mhtpn_acc*100:.1f} \\pm {s.mhtpn_std*100:.1f}$"
            baseline_str = f"{s.best_baseline} (${s.baseline_acc*100:.1f}\\%$)"
            imp_str = f"+{s.improvement:.1f}\\%" if s.improvement >= 0 else f"{s.improvement:.1f}\\%"

            lines.append(
                f"{s.name} & {s.domain} & {s.n_samples:,} & {mhtpn_str} & {baseline_str} & {imp_str} & \\#{s.rank} \\\\"
            )

        lines.extend([
            r'\bottomrule',
            r'\end{tabular}',
            r'\end{table}',
        ])

        table_str = '\n'.join(lines)

    else:
        # Generate CSV
        rows = []
        for s in summaries:
            rows.append({
                'Dataset': s.name,
                'Domain': s.domain,
                'Samples': s.n_samples,
                'Classes': s.n_classes,
                'MHTPN_Accuracy': f"{s.mhtpn_acc*100:.1f}",
                'MHTPN_Std': f"{s.mhtpn_std*100:.1f}",
                'Best_Baseline': s.best_baseline,
                'Baseline_Accuracy': f"{s.baseline_acc*100:.1f}",
                'Baseline_Std': f"{s.baseline_std*100:.1f}",
                'Improvement_pp': f"{s.improvement:.1f}",
                'Rank': s.rank,
            })

        table_str = ','.join(rows[0].keys()) + '\n'
        for row in rows:
            table_str += ','.join(str(v) for v in row.values()) + '\n'

    # Save
    with open(output_path, 'w') as f:
        f.write(table_str)

    return table_str


def generate_model_comparison_table(
    summaries: List[DatasetSummary],
    output_path: str,
) -> str:
    """
    Generate a table comparing all models across datasets.

    Args:
        summaries: List of DatasetSummary objects
        output_path: Path to save the table

    Returns:
        Table as CSV string
    """
    # Collect all models
    all_models = set()
    for s in summaries:
        all_models.update(s.all_model_results.keys())

    all_models = sorted(all_models)

    # Build rows
    header = ['Model'] + [s.name for s in summaries] + ['Average']
    rows = [header]

    for model in all_models:
        row = [model]
        accs = []
        for s in summaries:
            if model in s.all_model_results:
                acc = s.all_model_results[model]['accuracy']
                row.append(f"{acc*100:.1f}")
                accs.append(acc)
            else:
                row.append('-')
        # Average across datasets
        if accs:
            row.append(f"{np.mean(accs)*100:.1f}")
        else:
            row.append('-')
        rows.append(row)

    # Save as CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return '\n'.join([','.join(r) for r in rows])


def generate_aggregate_statistics(
    summaries: List[DatasetSummary],
) -> Dict[str, Any]:
    """
    Compute aggregate statistics across all datasets.

    Args:
        summaries: List of DatasetSummary objects

    Returns:
        Dict with aggregate statistics
    """
    # MHTPN statistics
    mhtpn_accs = [s.mhtpn_acc for s in summaries]
    improvements = [s.improvement for s in summaries]
    ranks = [s.rank for s in summaries]

    # Count wins
    num_first_place = sum(1 for r in ranks if r == 1)

    return {
        'n_datasets': len(summaries),
        'mhtpn_mean_accuracy': np.mean(mhtpn_accs),
        'mhtpn_std_accuracy': np.std(mhtpn_accs),
        'mean_improvement_pp': np.mean(improvements),
        'max_improvement_pp': max(improvements),
        'min_improvement_pp': min(improvements),
        'mean_rank': np.mean(ranks),
        'num_first_place': num_first_place,
        'first_place_rate': num_first_place / len(summaries),
        'datasets_summary': [
            {
                'name': s.name,
                'accuracy': s.mhtpn_acc,
                'improvement': s.improvement,
                'rank': s.rank,
            }
            for s in summaries
        ],
    }


def generate_full_summary(
    experiment_dir: str,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Generate full cross-dataset summary from an experiment directory.

    Args:
        experiment_dir: Root experiment directory containing dataset subdirs
        output_dir: Output directory (defaults to experiment_dir/combined)
        verbose: Print progress

    Returns:
        Dict with all summary data
    """
    if output_dir is None:
        output_dir = os.path.join(experiment_dir, 'combined')
    os.makedirs(output_dir, exist_ok=True)

    # Find all dataset result files
    summaries = []

    for item in os.listdir(experiment_dir):
        item_path = os.path.join(experiment_dir, item)
        if os.path.isdir(item_path) and item not in ['combined', 'per_fold']:
            results_file = os.path.join(item_path, 'full_results.json')
            if os.path.exists(results_file):
                if verbose:
                    print(f"Loading results from {item}...")
                results = load_dataset_results(results_file)
                summary = compute_dataset_summary(results)
                summaries.append(summary)

    if not summaries:
        if verbose:
            print("No dataset results found.")
        return {}

    if verbose:
        print(f"\nFound {len(summaries)} datasets.")

    # Generate outputs
    # 1. Cross-dataset summary CSV
    csv_path = os.path.join(output_dir, 'cross_dataset_summary.csv')
    generate_paper_table(summaries, csv_path, format='csv')
    if verbose:
        print(f"Saved: {csv_path}")

    # 2. LaTeX table
    latex_path = os.path.join(output_dir, 'cross_dataset_summary.tex')
    generate_paper_table(summaries, latex_path, format='latex')
    if verbose:
        print(f"Saved: {latex_path}")

    # 3. Model comparison table
    comparison_path = os.path.join(output_dir, 'model_comparison.csv')
    generate_model_comparison_table(summaries, comparison_path)
    if verbose:
        print(f"Saved: {comparison_path}")

    # 4. Aggregate statistics
    agg_stats = generate_aggregate_statistics(summaries)
    stats_path = os.path.join(output_dir, 'aggregate_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(agg_stats, f, indent=2)
    if verbose:
        print(f"Saved: {stats_path}")

    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("AGGREGATE STATISTICS")
        print("=" * 70)
        print(f"Datasets evaluated: {agg_stats['n_datasets']}")
        print(f"MHTPN mean accuracy: {agg_stats['mhtpn_mean_accuracy']*100:.1f}%")
        print(f"Mean improvement: {agg_stats['mean_improvement_pp']:+.1f} pp")
        print(f"First place rate: {agg_stats['num_first_place']}/{agg_stats['n_datasets']} "
              f"({agg_stats['first_place_rate']*100:.0f}%)")

    return {
        'summaries': [
            {
                'name': s.name,
                'domain': s.domain,
                'n_samples': s.n_samples,
                'mhtpn_acc': s.mhtpn_acc,
                'improvement': s.improvement,
                'rank': s.rank,
            }
            for s in summaries
        ],
        'aggregate': agg_stats,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate cross-dataset summary')
    parser.add_argument('experiment_dir', type=str, help='Experiment directory')
    parser.add_argument('--output', type=str, default=None, help='Output directory')

    args = parser.parse_args()

    generate_full_summary(
        experiment_dir=args.experiment_dir,
        output_dir=args.output,
        verbose=True,
    )
