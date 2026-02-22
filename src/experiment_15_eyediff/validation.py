"""
Validation module for eye difference equations.

The validation approach:
- MG patients: Should show SIGNIFICANT difference between left and right eyes
- HC patients: Should show NO significant difference between left and right eyes

We use statistical tests to quantify this.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def validate_equation(hc_differences, mg_differences, equation_name="Unknown",
                     results_dir="./results", alpha=0.05):
    """
    Validate an equation by comparing eye differences in HC vs MG.

    The ideal equation should:
    1. Show small eye differences in HC (median close to 0)
    2. Show large eye differences in MG (median significantly > 0)
    3. MG differences should be significantly larger than HC differences

    Args:
        hc_differences: List of eye differences for HC patients
        mg_differences: List of eye differences for MG patients
        equation_name: Name of the equation being tested
        results_dir: Directory to save results
        alpha: Significance level for statistical tests

    Returns:
        Dictionary with validation metrics and test results
    """
    # Remove NaN values
    hc_diffs = np.array([d for d in hc_differences if not np.isnan(d)])
    mg_diffs = np.array([d for d in mg_differences if not np.isnan(d)])

    print(f"\n{'='*80}")
    print(f"Validation: {equation_name}")
    print(f"{'='*80}")
    print(f"HC differences: {len(hc_diffs)} samples (after NaN removal)")
    print(f"MG differences: {len(mg_diffs)} samples (after NaN removal)")

    if len(hc_diffs) < 2 or len(mg_diffs) < 2:
        print("ERROR: Insufficient data for statistical testing")
        return None

    # Descriptive statistics
    hc_mean = np.mean(hc_diffs)
    hc_median = np.median(hc_diffs)
    hc_std = np.std(hc_diffs)

    mg_mean = np.mean(mg_diffs)
    mg_median = np.median(mg_diffs)
    mg_std = np.std(mg_diffs)

    print(f"\nDescriptive Statistics:")
    print(f"  HC: mean={hc_mean:.4f}, median={hc_median:.4f}, std={hc_std:.4f}")
    print(f"  MG: mean={mg_mean:.4f}, median={mg_median:.4f}, std={mg_std:.4f}")

    # Test 1: Are MG differences larger than HC differences?
    # Use Mann-Whitney U test (non-parametric, no normality assumption)
    u_statistic, p_value_mw = stats.mannwhitneyu(mg_diffs, hc_diffs, alternative='greater')

    print(f"\nTest 1: Mann-Whitney U Test (MG > HC)")
    print(f"  U-statistic: {u_statistic:.4f}")
    print(f"  p-value: {p_value_mw:.6f}")
    print(f"  Significant at α={alpha}: {p_value_mw < alpha}")

    # Test 2: Are HC differences close to zero? (Wilcoxon signed-rank test)
    # Tests if median is significantly different from 0
    if len(hc_diffs) >= 3:
        w_stat_hc, p_value_hc = stats.wilcoxon(hc_diffs, alternative='two-sided')
        print(f"\nTest 2: Wilcoxon Test (HC median ≠ 0)")
        print(f"  W-statistic: {w_stat_hc:.4f}")
        print(f"  p-value: {p_value_hc:.6f}")
        print(f"  HC median significantly ≠ 0: {p_value_hc < alpha}")
        print(f"  Ideal: Should NOT be significant (we want HC ≈ 0)")
    else:
        p_value_hc = np.nan
        print(f"\nTest 2: Insufficient HC data for Wilcoxon test")

    # Test 3: Are MG differences significantly > 0? (Wilcoxon signed-rank test)
    if len(mg_diffs) >= 3:
        w_stat_mg, p_value_mg = stats.wilcoxon(mg_diffs, alternative='greater')
        print(f"\nTest 3: Wilcoxon Test (MG median > 0)")
        print(f"  W-statistic: {w_stat_mg:.4f}")
        print(f"  p-value: {p_value_mg:.6f}")
        print(f"  MG median significantly > 0: {p_value_mg < alpha}")
        print(f"  Ideal: Should be significant (we want MG > 0)")
    else:
        p_value_mg = np.nan
        print(f"\nTest 3: Insufficient MG data for Wilcoxon test")

    # Effect size: Cohen's d
    pooled_std = np.sqrt((hc_std**2 + mg_std**2) / 2)
    cohens_d = (mg_mean - hc_mean) / pooled_std if pooled_std > 0 else np.nan

    print(f"\nEffect Size:")
    print(f"  Cohen's d: {cohens_d:.4f}")
    print(f"  Interpretation: ", end="")
    if np.isnan(cohens_d):
        print("N/A")
    elif abs(cohens_d) < 0.2:
        print("Negligible")
    elif abs(cohens_d) < 0.5:
        print("Small")
    elif abs(cohens_d) < 0.8:
        print("Medium")
    else:
        print("Large")

    # Overall validation score
    # Good equation should: (1) MG > HC (p<0.05), (2) HC ≈ 0 (p>0.05), (3) MG > 0 (p<0.05)
    validation_score = 0
    if p_value_mw < alpha:
        validation_score += 1  # MG > HC
    if not np.isnan(p_value_hc) and p_value_hc >= alpha:
        validation_score += 1  # HC ≈ 0
    if not np.isnan(p_value_mg) and p_value_mg < alpha:
        validation_score += 1  # MG > 0

    print(f"\nValidation Score: {validation_score}/3")
    print(f"  1 point: MG > HC (Mann-Whitney p < {alpha}): {'✓' if p_value_mw < alpha else '✗'}")
    if not np.isnan(p_value_hc):
        print(f"  1 point: HC ≈ 0 (Wilcoxon p >= {alpha}): {'✓' if p_value_hc >= alpha else '✗'}")
    else:
        print(f"  1 point: HC ≈ 0: N/A")
    if not np.isnan(p_value_mg):
        print(f"  1 point: MG > 0 (Wilcoxon p < {alpha}): {'✓' if p_value_mg < alpha else '✗'}")
    else:
        print(f"  1 point: MG > 0: N/A")

    print(f"{'='*80}\n")

    # Package results
    results = {
        'equation_name': equation_name,
        'n_hc': len(hc_diffs),
        'n_mg': len(mg_diffs),
        'hc_mean': hc_mean,
        'hc_median': hc_median,
        'hc_std': hc_std,
        'mg_mean': mg_mean,
        'mg_median': mg_median,
        'mg_std': mg_std,
        'mann_whitney_u': u_statistic,
        'mann_whitney_p': p_value_mw,
        'wilcoxon_hc_p': p_value_hc,
        'wilcoxon_mg_p': p_value_mg,
        'cohens_d': cohens_d,
        'validation_score': validation_score,
        'hc_differences': hc_diffs,
        'mg_differences': mg_diffs
    }

    return results


def plot_validation_results(validation_results, results_dir, equation_name="Unknown"):
    """
    Create visualization plots for validation results.

    Args:
        validation_results: Dictionary from validate_equation()
        results_dir: Directory to save plots
        equation_name: Name for plot titles
    """
    import os
    os.makedirs(results_dir, exist_ok=True)

    hc_diffs = validation_results['hc_differences']
    mg_diffs = validation_results['mg_differences']

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Validation Results: {equation_name}', fontsize=16, fontweight='bold')

    # Plot 1: Box plots
    ax1 = axes[0, 0]
    data_to_plot = [hc_diffs, mg_diffs]
    bp = ax1.boxplot(data_to_plot, labels=['HC', 'MG'], patch_artist=True,
                     showmeans=True, meanline=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax1.set_ylabel('Eye Difference (metric units)', fontsize=12)
    ax1.set_title('Box Plot: Eye Differences by Group', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero difference')
    ax1.legend()

    # Plot 2: Histograms
    ax2 = axes[0, 1]
    ax2.hist(hc_diffs, bins=30, alpha=0.6, label='HC', color='blue', density=True)
    ax2.hist(mg_diffs, bins=30, alpha=0.6, label='MG', color='red', density=True)
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Zero')
    ax2.set_xlabel('Eye Difference (metric units)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Distribution of Eye Differences', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Violin plots
    ax3 = axes[1, 0]
    parts = ax3.violinplot([hc_diffs, mg_diffs], positions=[1, 2],
                          showmeans=True, showmedians=True)
    ax3.set_xticks([1, 2])
    ax3.set_xticklabels(['HC', 'MG'])
    ax3.set_ylabel('Eye Difference (metric units)', fontsize=12)
    ax3.set_title('Violin Plot: Distribution Shape', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # Plot 4: Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    VALIDATION SUMMARY
    {'='*40}

    Sample Sizes:
      HC: {validation_results['n_hc']}
      MG: {validation_results['n_mg']}

    Descriptive Statistics:
      HC: mean={validation_results['hc_mean']:.4f},
          median={validation_results['hc_median']:.4f}
      MG: mean={validation_results['mg_mean']:.4f},
          median={validation_results['mg_median']:.4f}

    Statistical Tests:
      Mann-Whitney (MG>HC): p={validation_results['mann_whitney_p']:.6f}
      Wilcoxon HC≈0: p={validation_results['wilcoxon_hc_p']:.6f}
      Wilcoxon MG>0: p={validation_results['wilcoxon_mg_p']:.6f}

    Effect Size:
      Cohen's d: {validation_results['cohens_d']:.4f}

    Validation Score: {validation_results['validation_score']}/3
    """

    ax4.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'{equation_name.replace(" ", "_")}_validation.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Validation plot saved to: {plot_path}")
