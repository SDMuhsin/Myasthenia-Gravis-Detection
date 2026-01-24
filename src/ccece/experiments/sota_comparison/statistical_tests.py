"""
Statistical Tests Module for SOTA Comparison

Performs statistical significance testing between models:
- Paired t-test
- Wilcoxon signed-rank test
- Cohen's d effect size
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass


@dataclass
class PairwiseComparison:
    """Result of comparing two models."""
    baseline_name: str
    baseline_mean: float
    baseline_std: float
    ours_mean: float
    ours_std: float
    difference: float
    ttest_pvalue: float
    wilcoxon_pvalue: Optional[float]
    cohens_d: float
    is_significant: bool  # At alpha=0.05
    is_better: bool  # Our method is better


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.

    Uses pooled standard deviation formula.

    Returns:
        Cohen's d (positive if y > x)
    """
    nx, ny = len(x), len(y)
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))

    if pooled_std == 0:
        return 0.0

    return (my - mx) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def paired_ttest(x: np.ndarray, y: np.ndarray) -> float:
    """
    Perform paired t-test.

    Args:
        x: Baseline scores (n_folds,)
        y: Our method scores (n_folds,)

    Returns:
        Two-sided p-value
    """
    if len(x) != len(y):
        raise ValueError("Arrays must have same length for paired test")

    if len(x) < 2:
        return 1.0  # Cannot perform test with < 2 samples

    _, pvalue = stats.ttest_rel(x, y)
    return pvalue


def wilcoxon_test(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative).

    Args:
        x: Baseline scores (n_folds,)
        y: Our method scores (n_folds,)

    Returns:
        Two-sided p-value, or None if test cannot be performed
    """
    if len(x) != len(y):
        raise ValueError("Arrays must have same length for paired test")

    if len(x) < 6:
        return None  # Wilcoxon needs at least 6 samples for meaningful result

    try:
        # Use 'wilcox' method for exact p-value when possible
        _, pvalue = stats.wilcoxon(y - x, alternative='two-sided')
        return pvalue
    except ValueError:
        return None


def compare_to_baseline(
    baseline_name: str,
    baseline_scores: np.ndarray,
    ours_scores: np.ndarray,
    alpha: float = 0.05,
) -> PairwiseComparison:
    """
    Compare our method to a baseline.

    Args:
        baseline_name: Name of baseline method
        baseline_scores: Per-fold scores for baseline (n_folds,)
        ours_scores: Per-fold scores for our method (n_folds,)
        alpha: Significance level

    Returns:
        PairwiseComparison result
    """
    baseline_mean = np.mean(baseline_scores)
    baseline_std = np.std(baseline_scores, ddof=1) if len(baseline_scores) > 1 else 0.0
    ours_mean = np.mean(ours_scores)
    ours_std = np.std(ours_scores, ddof=1) if len(ours_scores) > 1 else 0.0

    # Effect size
    d = cohens_d(baseline_scores, ours_scores)

    # Statistical tests
    ttest_p = paired_ttest(baseline_scores, ours_scores)
    wilcoxon_p = wilcoxon_test(baseline_scores, ours_scores)

    return PairwiseComparison(
        baseline_name=baseline_name,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        ours_mean=ours_mean,
        ours_std=ours_std,
        difference=ours_mean - baseline_mean,
        ttest_pvalue=ttest_p,
        wilcoxon_pvalue=wilcoxon_p,
        cohens_d=d,
        is_significant=ttest_p < alpha,
        is_better=ours_mean > baseline_mean,
    )


def run_all_comparisons(
    all_results: Dict[str, Dict[str, List[float]]],
    our_method_name: str = "MultiHeadProtoNet",
    metric: str = "accuracy",
    alpha: float = 0.05,
) -> List[PairwiseComparison]:
    """
    Compare our method to all baselines.

    Args:
        all_results: Dict mapping model_name -> {metric: [fold_scores]}
        our_method_name: Name of our method in results
        metric: Metric to compare on
        alpha: Significance level

    Returns:
        List of PairwiseComparison results
    """
    if our_method_name not in all_results:
        raise ValueError(f"Our method '{our_method_name}' not found in results")

    ours_scores = np.array(all_results[our_method_name][metric])
    comparisons = []

    for baseline_name, baseline_results in all_results.items():
        if baseline_name == our_method_name:
            continue

        baseline_scores = np.array(baseline_results[metric])
        comparison = compare_to_baseline(baseline_name, baseline_scores, ours_scores, alpha)
        comparisons.append(comparison)

    return comparisons


def format_significance(comparison: PairwiseComparison) -> str:
    """Format significance result for display."""
    if comparison.is_significant:
        direction = "better" if comparison.is_better else "worse"
        return f"** (p={comparison.ttest_pvalue:.3f}, d={comparison.cohens_d:.2f}, {direction})"
    else:
        return f"ns (p={comparison.ttest_pvalue:.3f})"
