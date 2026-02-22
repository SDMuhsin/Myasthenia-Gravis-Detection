#!/usr/bin/env python3
"""
Experiment 19: MG vs CNP Temporal Asymmetry Dynamics

Based on medical team feedback:
- MG: Fatigability - asymmetry should INCREASE over the session
- CNP: Static deficit - asymmetry should be CONSTANT (present from start, no progression)

This experiment computes per-saccade asymmetry and analyzes temporal dynamics
to distinguish MG from CNP based on how asymmetry evolves over time.

Key Metrics:
- Asymmetry Slope: Rate of change of asymmetry over session
- Asymmetry Delta: (late asymmetry) - (early asymmetry)
- Initial Asymmetry: Mean asymmetry in first third
- Final Asymmetry: Mean asymmetry in last third
- Growth Ratio: Final / Initial asymmetry
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
from tqdm import tqdm

# Configuration
BASE_DIR = './data'
RESULTS_DIR = './results/exp_19_temporal_asymmetry'
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']
CSV_ENCODING = 'utf-16-le'
CSV_SEPARATOR = ','
MIN_SEQ_LEN_THRESHOLD = 50
SAMPLE_RATE = 120  # Hz
SACCADE_THRESHOLD = 5.0  # degrees
MIN_SACCADES_REQUIRED = 6  # Need enough saccades to compute early/late


def create_results_dir():
    """Create results directory if it doesn't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Results will be saved to: {RESULTS_DIR}")


def load_data_from_folder(folder_path, class_name, label):
    """Load all CSV files from a folder structure."""
    items = []

    if not os.path.isdir(folder_path):
        print(f"  Warning: Directory not found: {folder_path}")
        return items

    patient_dirs = [d for d in os.listdir(folder_path)
                    if os.path.isdir(os.path.join(folder_path, d))]

    if patient_dirs:
        for patient_folder in tqdm(patient_dirs, desc=f"  Loading {class_name}", leave=False):
            patient_path = os.path.join(folder_path, patient_folder)
            csv_files = glob.glob(os.path.join(patient_path, '*.csv'))

            for csv_file in csv_files:
                item = load_single_csv(csv_file, class_name, label, patient_folder)
                if item is not None:
                    items.append(item)
    else:
        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        for csv_file in tqdm(csv_files, desc=f"  Loading {class_name}", leave=False):
            patient_id = os.path.splitext(os.path.basename(csv_file))[0]
            item = load_single_csv(csv_file, class_name, label, patient_id)
            if item is not None:
                items.append(item)

    return items


def load_single_csv(csv_path, class_name, label, patient_id):
    """Load a single CSV file."""
    try:
        df = pd.read_csv(csv_path, encoding=CSV_ENCODING, sep=CSV_SEPARATOR)
        df.columns = [col.strip() for col in df.columns]

        if any(col not in df.columns for col in FEATURE_COLUMNS):
            return None

        if len(df) < MIN_SEQ_LEN_THRESHOLD:
            return None

        df_features = df[FEATURE_COLUMNS].copy()
        for col in df_features.columns:
            df_features.loc[:, col] = pd.to_numeric(df_features[col], errors='coerce')

        if df_features.isnull().sum().sum() > 0.1 * df_features.size:
            return None

        df_features = df_features.fillna(0)

        return {
            'data': df_features.values.astype(np.float32),
            'label': label,
            'patient_id': patient_id,
            'filename': os.path.basename(csv_path),
            'class_name': class_name,
        }
    except Exception:
        return None


def load_all_data():
    """Load HC, MG, and CNP data."""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    all_items = []

    # Load HC
    print("\nLoading Healthy Controls...")
    hc_items = load_data_from_folder(
        os.path.join(BASE_DIR, 'Healthy control'), 'HC', 0
    )
    all_items.extend(hc_items)
    print(f"  Loaded {len(hc_items)} HC sequences")

    # Load MG (Definite + Probable)
    print("\nLoading MG patients...")
    mg_definite = load_data_from_folder(
        os.path.join(BASE_DIR, 'Definite MG'), 'MG', 1
    )
    mg_probable = load_data_from_folder(
        os.path.join(BASE_DIR, 'Probable MG'), 'MG', 1
    )
    all_items.extend(mg_definite)
    all_items.extend(mg_probable)
    print(f"  Loaded {len(mg_definite) + len(mg_probable)} MG sequences")

    # Load CNP subtypes
    print("\nLoading CNP patients...")
    cnp_base = os.path.join(BASE_DIR, 'Non-MG diplopia (CNP, etc)')

    cnp_3rd = load_data_from_folder(os.path.join(cnp_base, '3rd'), 'CNP_3rd', 2)
    cnp_4th = load_data_from_folder(os.path.join(cnp_base, '4th'), 'CNP_4th', 3)
    cnp_6th = load_data_from_folder(os.path.join(cnp_base, '6th'), 'CNP_6th', 4)
    cnp_tao = load_data_from_folder(os.path.join(cnp_base, 'TAO'), 'CNP_TAO', 5)

    all_items.extend(cnp_3rd)
    all_items.extend(cnp_4th)
    all_items.extend(cnp_6th)
    all_items.extend(cnp_tao)

    print(f"  CNP 3rd: {len(cnp_3rd)} sequences")
    print(f"  CNP 4th: {len(cnp_4th)} sequences")
    print(f"  CNP 6th: {len(cnp_6th)} sequences")
    print(f"  CNP TAO: {len(cnp_tao)} sequences")
    print(f"  Total CNP: {len(cnp_3rd) + len(cnp_4th) + len(cnp_6th) + len(cnp_tao)} sequences")

    return all_items


# =============================================================================
# SACCADE DETECTION
# =============================================================================

def detect_saccades(target_signal, direction='positive', threshold=SACCADE_THRESHOLD):
    """Detect saccade onset indices from target signal."""
    target_diff = np.diff(target_signal)

    if direction == 'positive':
        indices = np.where(target_diff > threshold)[0] + 1
    elif direction == 'negative':
        indices = np.where(target_diff < -threshold)[0] + 1
    else:
        indices = np.where(np.abs(target_diff) > threshold)[0] + 1

    return indices.tolist()


# =============================================================================
# PER-SACCADE ASYMMETRY COMPUTATION
# =============================================================================

def compute_per_saccade_asymmetry(eye_l, eye_r, target, saccade_indices):
    """
    Compute asymmetry for each saccade trial.

    Returns:
        List of (saccade_number, asymmetry_value) tuples
    """
    n_samples = len(eye_l)
    asymmetry_series = []

    for i, idx in enumerate(saccade_indices):
        if idx >= n_samples:
            continue

        # Settling window: 200-400ms after saccade onset (24-48 samples at 120Hz)
        start = min(idx + 24, n_samples)
        end = min(idx + 48, n_samples)

        if end > start + 5:  # Need enough samples
            # Compute mean absolute error for each eye
            error_l = np.mean(np.abs(eye_l[start:end] - target[start:end]))
            error_r = np.mean(np.abs(eye_r[start:end] - target[start:end]))

            # Asymmetry = absolute difference between eyes
            asymmetry = np.abs(error_l - error_r)

            asymmetry_series.append((i, asymmetry))

    return asymmetry_series


def compute_temporal_dynamics(asymmetry_series):
    """
    Compute temporal dynamics metrics from asymmetry time series.

    Returns dict with:
        - slope: Linear regression slope of asymmetry vs trial number
        - delta: (late mean) - (early mean)
        - initial: Mean of first third
        - final: Mean of last third
        - growth_ratio: final / initial
        - normalized_slope: slope / initial
        - n_saccades: Number of saccades
    """
    if len(asymmetry_series) < MIN_SACCADES_REQUIRED:
        return None

    trial_nums = np.array([x[0] for x in asymmetry_series])
    asymmetry_vals = np.array([x[1] for x in asymmetry_series])

    n = len(asymmetry_vals)
    third = max(1, n // 3)

    # Early (first third) and Late (last third)
    early_asymmetry = asymmetry_vals[:third]
    late_asymmetry = asymmetry_vals[-third:]

    initial_mean = np.mean(early_asymmetry)
    final_mean = np.mean(late_asymmetry)

    # Slope: linear regression of asymmetry vs trial number
    if len(trial_nums) >= 3:
        slope, intercept, r_value, p_value, std_err = stats.linregress(trial_nums, asymmetry_vals)
    else:
        slope = np.nan
        r_value = np.nan

    # Delta: late - early
    delta = final_mean - initial_mean

    # Growth ratio: final / initial (with epsilon to avoid division by zero)
    epsilon = 0.01  # Small value in degrees
    growth_ratio = final_mean / (initial_mean + epsilon)

    # Normalized slope: slope relative to initial level
    normalized_slope = slope / (initial_mean + epsilon) if not np.isnan(slope) else np.nan

    return {
        'slope': slope,
        'delta': delta,
        'initial': initial_mean,
        'final': final_mean,
        'growth_ratio': growth_ratio,
        'normalized_slope': normalized_slope,
        'r_squared': r_value**2 if not np.isnan(r_value) else np.nan,
        'n_saccades': n
    }


# =============================================================================
# GROUP ANALYSIS
# =============================================================================

def compute_dynamics_for_group(items, axis='vertical', direction='positive'):
    """Compute temporal dynamics metrics for a group of items."""
    dynamics_list = []

    for item in items:
        data = item['data']

        if axis == 'horizontal':
            eye_l, eye_r, target = data[:, 0], data[:, 1], data[:, 4]
        else:
            eye_l, eye_r, target = data[:, 2], data[:, 3], data[:, 5]

        saccade_indices = detect_saccades(target, direction=direction)

        if len(saccade_indices) < MIN_SACCADES_REQUIRED:
            continue

        asymmetry_series = compute_per_saccade_asymmetry(eye_l, eye_r, target, saccade_indices)

        if len(asymmetry_series) < MIN_SACCADES_REQUIRED:
            continue

        dynamics = compute_temporal_dynamics(asymmetry_series)

        if dynamics is not None:
            dynamics['patient_id'] = item['patient_id']
            dynamics['class_name'] = item['class_name']
            dynamics_list.append(dynamics)

    return dynamics_list


def compute_cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan

    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    if pooled_std == 0:
        return np.nan

    return (np.mean(group2) - np.mean(group1)) / pooled_std


def compare_groups(dynamics_a, dynamics_b, name_a, name_b):
    """Compare two groups across all temporal dynamics metrics."""
    metrics = ['slope', 'delta', 'initial', 'final', 'growth_ratio', 'normalized_slope']
    results = {}

    for metric in metrics:
        vals_a = [d[metric] for d in dynamics_a if not np.isnan(d[metric])]
        vals_b = [d[metric] for d in dynamics_b if not np.isnan(d[metric])]

        if len(vals_a) < 10 or len(vals_b) < 10:
            results[metric] = {
                'n_a': len(vals_a), 'n_b': len(vals_b),
                'cohens_d': np.nan, 'p_value': np.nan,
                'mean_a': np.nan, 'mean_b': np.nan,
                'std_a': np.nan, 'std_b': np.nan,
            }
            continue

        d = compute_cohens_d(vals_a, vals_b)
        _, p = stats.mannwhitneyu(vals_b, vals_a, alternative='two-sided')

        results[metric] = {
            'n_a': len(vals_a), 'n_b': len(vals_b),
            'cohens_d': d, 'p_value': p,
            'mean_a': np.mean(vals_a), 'mean_b': np.mean(vals_b),
            'std_a': np.std(vals_a), 'std_b': np.std(vals_b),
        }

    return results


def main():
    print("="*80)
    print("EXPERIMENT 19: TEMPORAL ASYMMETRY DYNAMICS")
    print("MG (Fatigability) vs CNP (Static Deficit)")
    print("="*80)

    create_results_dir()

    # Load data
    all_items = load_all_data()

    # Group items by class
    hc_items = [x for x in all_items if x['class_name'] == 'HC']
    mg_items = [x for x in all_items if x['class_name'] == 'MG']
    cnp_3rd = [x for x in all_items if x['class_name'] == 'CNP_3rd']
    cnp_4th = [x for x in all_items if x['class_name'] == 'CNP_4th']
    cnp_6th = [x for x in all_items if x['class_name'] == 'CNP_6th']
    cnp_all = cnp_3rd + cnp_4th + cnp_6th  # Exclude TAO (n too small)

    print(f"\nFinal counts:")
    print(f"  HC: {len(hc_items)}")
    print(f"  MG: {len(mg_items)}")
    print(f"  CNP (all): {len(cnp_all)}")
    print(f"    - 3rd: {len(cnp_3rd)}")
    print(f"    - 4th: {len(cnp_4th)}")
    print(f"    - 6th: {len(cnp_6th)}")

    # Compute temporal dynamics for each group (using upward vertical saccades)
    print("\n" + "="*80)
    print("COMPUTING TEMPORAL DYNAMICS (Upward Vertical Saccades)")
    print("="*80)

    print("\nComputing for HC...")
    dynamics_hc = compute_dynamics_for_group(hc_items, axis='vertical', direction='positive')
    print(f"  Valid samples: {len(dynamics_hc)}")

    print("\nComputing for MG...")
    dynamics_mg = compute_dynamics_for_group(mg_items, axis='vertical', direction='positive')
    print(f"  Valid samples: {len(dynamics_mg)}")

    print("\nComputing for CNP (all)...")
    dynamics_cnp_all = compute_dynamics_for_group(cnp_all, axis='vertical', direction='positive')
    print(f"  Valid samples: {len(dynamics_cnp_all)}")

    print("\nComputing for CNP subtypes...")
    dynamics_cnp_3rd = compute_dynamics_for_group(cnp_3rd, axis='vertical', direction='positive')
    dynamics_cnp_4th = compute_dynamics_for_group(cnp_4th, axis='vertical', direction='positive')
    dynamics_cnp_6th = compute_dynamics_for_group(cnp_6th, axis='vertical', direction='positive')
    print(f"  CNP_3rd valid: {len(dynamics_cnp_3rd)}")
    print(f"  CNP_4th valid: {len(dynamics_cnp_4th)}")
    print(f"  CNP_6th valid: {len(dynamics_cnp_6th)}")

    # Descriptive statistics
    print("\n" + "="*80)
    print("DESCRIPTIVE STATISTICS")
    print("="*80)

    def print_group_stats(name, dynamics):
        if not dynamics:
            print(f"\n{name}: No valid data")
            return
        slopes = [d['slope'] for d in dynamics if not np.isnan(d['slope'])]
        deltas = [d['delta'] for d in dynamics if not np.isnan(d['delta'])]
        initials = [d['initial'] for d in dynamics if not np.isnan(d['initial'])]
        growth_ratios = [d['growth_ratio'] for d in dynamics if not np.isnan(d['growth_ratio'])]

        print(f"\n{name} (n={len(dynamics)}):")
        print(f"  Slope:        mean={np.mean(slopes):.4f}, std={np.std(slopes):.4f}")
        print(f"  Delta:        mean={np.mean(deltas):.4f}, std={np.std(deltas):.4f}")
        print(f"  Initial:      mean={np.mean(initials):.4f}, std={np.std(initials):.4f}")
        print(f"  Growth Ratio: mean={np.mean(growth_ratios):.4f}, std={np.std(growth_ratios):.4f}")

    print_group_stats("HC", dynamics_hc)
    print_group_stats("MG", dynamics_mg)
    print_group_stats("CNP (all)", dynamics_cnp_all)
    print_group_stats("CNP_3rd", dynamics_cnp_3rd)
    print_group_stats("CNP_4th", dynamics_cnp_4th)
    print_group_stats("CNP_6th", dynamics_cnp_6th)

    # Perform comparisons
    print("\n" + "="*80)
    print("STATISTICAL COMPARISONS")
    print("="*80)

    comparisons = [
        ('MG', 'HC', dynamics_mg, dynamics_hc),
        ('CNP_all', 'HC', dynamics_cnp_all, dynamics_hc),
        ('CNP_3rd', 'HC', dynamics_cnp_3rd, dynamics_hc),
        ('CNP_4th', 'HC', dynamics_cnp_4th, dynamics_hc),
        ('CNP_6th', 'HC', dynamics_cnp_6th, dynamics_hc),
        ('MG', 'CNP_all', dynamics_mg, dynamics_cnp_all),
        ('MG', 'CNP_3rd', dynamics_mg, dynamics_cnp_3rd),
        ('MG', 'CNP_4th', dynamics_mg, dynamics_cnp_4th),
        ('MG', 'CNP_6th', dynamics_mg, dynamics_cnp_6th),
    ]

    all_results = []

    for name_b, name_a, dynamics_b, dynamics_a in comparisons:
        result = compare_groups(dynamics_a, dynamics_b, name_a, name_b)

        for metric, stats_dict in result.items():
            all_results.append({
                'Comparison': f'{name_b} vs {name_a}',
                'Group_A': name_a,
                'Group_B': name_b,
                'Metric': metric,
                'n_A': stats_dict['n_a'],
                'n_B': stats_dict['n_b'],
                'Mean_A': stats_dict['mean_a'],
                'Mean_B': stats_dict['mean_b'],
                'Cohen_d': stats_dict['cohens_d'],
                'p_value': stats_dict['p_value'],
            })

    df_results = pd.DataFrame(all_results)

    # Print results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    # Key metrics for the hypothesis
    key_metrics = ['slope', 'delta', 'initial', 'growth_ratio']

    for comparison in df_results['Comparison'].unique():
        df_comp = df_results[df_results['Comparison'] == comparison]
        df_comp = df_comp[df_comp['Metric'].isin(key_metrics)]

        print(f"\n--- {comparison} ---")
        for _, row in df_comp.iterrows():
            d = row['Cohen_d']
            p = row['p_value']
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            d_str = f"{d:.3f}" if not np.isnan(d) else "N/A"
            p_str = f"{p:.4f}" if not np.isnan(p) else "N/A"
            print(f"  {row['Metric']:15s}: d={d_str:>7s}, p={p_str} {sig}")

    # THE KEY COMPARISON: MG vs CNP
    print("\n" + "="*80)
    print("KEY COMPARISON: MG vs CNP (all)")
    print("Testing the Medical Team's Hypothesis")
    print("="*80)

    df_key = df_results[(df_results['Comparison'] == 'MG vs CNP_all') &
                        (df_results['Metric'].isin(key_metrics))]

    print("\nHypothesis: MG shows MORE growth (fatigability) than CNP (static)")
    print("\n| Metric       | MG Mean  | CNP Mean | Cohen's d | p-value  | Interpretation |")
    print("|--------------|----------|----------|-----------|----------|----------------|")

    for _, row in df_key.iterrows():
        metric = row['Metric']
        mean_cnp = row['Mean_A']
        mean_mg = row['Mean_B']
        d = row['Cohen_d']
        p = row['p_value']

        # Interpretation based on hypothesis
        if metric == 'slope':
            interp = "MG > CNP?" if d > 0 else "CNP >= MG"
        elif metric == 'delta':
            interp = "MG > CNP?" if d > 0 else "CNP >= MG"
        elif metric == 'initial':
            interp = "CNP > MG?" if d < 0 else "MG >= CNP"
        elif metric == 'growth_ratio':
            interp = "MG > CNP?" if d > 0 else "CNP >= MG"
        else:
            interp = "-"

        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        mean_mg_str = f"{mean_mg:.4f}" if not np.isnan(mean_mg) else "N/A"
        mean_cnp_str = f"{mean_cnp:.4f}" if not np.isnan(mean_cnp) else "N/A"
        d_str = f"{d:.3f}" if not np.isnan(d) else "N/A"
        p_str = f"{p:.4f}{sig}" if not np.isnan(p) else "N/A"

        print(f"| {metric:12s} | {mean_mg_str:>8s} | {mean_cnp_str:>8s} | {d_str:>9s} | {p_str:>8s} | {interp:14s} |")

    # Save results
    df_results.to_csv(os.path.join(RESULTS_DIR, 'detailed_results.csv'), index=False)

    # Save per-patient dynamics for further analysis
    all_dynamics = []
    for d in dynamics_hc:
        d_copy = d.copy()
        d_copy['group'] = 'HC'
        all_dynamics.append(d_copy)
    for d in dynamics_mg:
        d_copy = d.copy()
        d_copy['group'] = 'MG'
        all_dynamics.append(d_copy)
    for d in dynamics_cnp_all:
        d_copy = d.copy()
        d_copy['group'] = 'CNP'
        all_dynamics.append(d_copy)

    df_dynamics = pd.DataFrame(all_dynamics)
    df_dynamics.to_csv(os.path.join(RESULTS_DIR, 'per_patient_dynamics.csv'), index=False)

    # Generate summary
    print("\n" + "="*80)
    print("SUMMARY COMPARISON TABLE")
    print("="*80)

    summary_rows = []
    for comparison in df_results['Comparison'].unique():
        df_comp = df_results[(df_results['Comparison'] == comparison) &
                            (df_results['Metric'].isin(key_metrics))]
        if df_comp.empty:
            continue

        mean_d = df_comp['Cohen_d'].mean()
        n_sig = (df_comp['p_value'] < 0.05).sum()

        summary_rows.append({
            'Comparison': comparison,
            'Mean_d': mean_d,
            'N_Significant': f"{n_sig}/4",
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(RESULTS_DIR, 'summary.csv'), index=False)

    print("\n| Comparison         | Mean Cohen's d | Significant Metrics |")
    print("|--------------------|----------------|---------------------|")
    for _, row in df_summary.iterrows():
        print(f"| {row['Comparison']:18s} | {row['Mean_d']:>14.3f} | {row['N_Significant']:>19s} |")

    # Generate report
    print("\n" + "="*80)
    print("GENERATING REPORT")
    print("="*80)

    report_path = os.path.join(RESULTS_DIR, 'REPORT.md')
    with open(report_path, 'w') as f:
        f.write("# Experiment 19: Temporal Asymmetry Dynamics\n\n")
        f.write("## Objective\n\n")
        f.write("Test whether MG and CNP can be distinguished by **how asymmetry changes over time**:\n")
        f.write("- **MG hypothesis**: Fatigability - asymmetry increases during the session\n")
        f.write("- **CNP hypothesis**: Static deficit - asymmetry is constant throughout\n\n")

        f.write("## Methodology\n\n")
        f.write("1. Compute per-saccade asymmetry: |error_L - error_R| for each saccade\n")
        f.write("2. Analyze temporal dynamics:\n")
        f.write("   - **Slope**: Rate of asymmetry change (linear regression)\n")
        f.write("   - **Delta**: Late asymmetry minus early asymmetry\n")
        f.write("   - **Initial**: Mean asymmetry in first third of session\n")
        f.write("   - **Growth Ratio**: Final/Initial asymmetry\n\n")

        f.write("## Key Results: MG vs CNP\n\n")
        f.write("| Metric | MG Mean | CNP Mean | Cohen's d | p-value | Interpretation |\n")
        f.write("|--------|---------|----------|-----------|---------|----------------|\n")

        for _, row in df_key.iterrows():
            metric = row['Metric']
            mean_cnp = row['Mean_A']
            mean_mg = row['Mean_B']
            d = row['Cohen_d']
            p = row['p_value']

            if metric in ['slope', 'delta', 'growth_ratio']:
                interp = "MG shows more growth" if d > 0.2 else "Similar" if abs(d) < 0.2 else "CNP shows more growth"
            elif metric == 'initial':
                interp = "CNP higher initial" if d < -0.2 else "Similar" if abs(d) < 0.2 else "MG higher initial"
            else:
                interp = "-"

            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            f.write(f"| {metric} | {mean_mg:.4f} | {mean_cnp:.4f} | {d:.3f} | {p:.4f}{sig} | {interp} |\n")

        f.write("\n## All Comparisons Summary\n\n")
        f.write("| Comparison | Mean Cohen's d | Significant Metrics |\n")
        f.write("|------------|----------------|---------------------|\n")
        for _, row in df_summary.iterrows():
            f.write(f"| {row['Comparison']} | {row['Mean_d']:.3f} | {row['N_Significant']} |\n")

        f.write("\n## Interpretation\n\n")

        # Get MG vs CNP results
        mg_cnp_d = df_summary[df_summary['Comparison'] == 'MG vs CNP_all']['Mean_d'].values
        if len(mg_cnp_d) > 0:
            mg_cnp_d = mg_cnp_d[0]
            if mg_cnp_d > 0.3:
                f.write("**The temporal dynamics metrics show BETTER discrimination between MG and CNP** ")
                f.write(f"(Mean d = {mg_cnp_d:.3f}) compared to the magnitude-based approach in Experiment 18 (d = 0.081).\n\n")
                f.write("This supports the medical team's hypothesis that fatigability (dynamic worsening) ")
                f.write("is a distinguishing feature of MG.\n")
            elif mg_cnp_d > 0.1:
                f.write("**The temporal dynamics metrics show modest improvement in MG vs CNP discrimination** ")
                f.write(f"(Mean d = {mg_cnp_d:.3f}) compared to Experiment 18 (d = 0.081).\n\n")
                f.write("There is some support for the fatigability hypothesis, but the effect is small.\n")
            else:
                f.write("**The temporal dynamics approach does not substantially improve MG vs CNP discrimination** ")
                f.write(f"(Mean d = {mg_cnp_d:.3f}).\n\n")
                f.write("The patterns of asymmetry change over time are similar between MG and CNP in this data.\n")

        f.write("\n---\n")
        f.write(f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*\n")

    print(f"\nReport saved to: {report_path}")
    print(f"Detailed results: {os.path.join(RESULTS_DIR, 'detailed_results.csv')}")
    print(f"Per-patient data: {os.path.join(RESULTS_DIR, 'per_patient_dynamics.csv')}")

    print("\n" + "="*80)
    print("EXPERIMENT 19 COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
