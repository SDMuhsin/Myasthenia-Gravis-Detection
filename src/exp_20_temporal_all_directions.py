#!/usr/bin/env python3
"""
Experiment 20: Temporal Asymmetry Dynamics - All Saccade Directions

Extension of Experiment 19 to test ALL 6 saccade directions:
1. vertical_up - Upward vertical saccades
2. vertical_down - Downward vertical saccades
3. vertical_both - Both vertical directions
4. horizontal_left - Leftward horizontal saccades
5. horizontal_right - Rightward horizontal saccades
6. horizontal_both - Both horizontal directions

Tests the fatigability hypothesis (MG shows increasing asymmetry vs CNP static)
across all directions to find the optimal signal.

Key Questions:
- Does any direction show stronger MG vs CNP discrimination than vertical_up?
- Does horizontal work better for CNP_6th (6th nerve = lateral gaze)?
- Which direction best captures the fatigability pattern?
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = './data'
RESULTS_DIR = './results/exp_20_temporal_all_directions'
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']
CSV_ENCODING = 'utf-16-le'
CSV_SEPARATOR = ','
MIN_SEQ_LEN_THRESHOLD = 50
SAMPLE_RATE = 120  # Hz
SACCADE_THRESHOLD = 5.0  # degrees
MIN_SACCADES_REQUIRED = 6  # Need enough saccades to compute early/late

# Saccade direction configurations
SACCADE_DIRECTIONS = {
    'vertical_up': {'axis': 'vertical', 'direction': 'positive'},
    'vertical_down': {'axis': 'vertical', 'direction': 'negative'},
    'vertical_both': {'axis': 'vertical', 'direction': 'both'},
    'horizontal_right': {'axis': 'horizontal', 'direction': 'positive'},
    'horizontal_left': {'axis': 'horizontal', 'direction': 'negative'},
    'horizontal_both': {'axis': 'horizontal', 'direction': 'both'},
}


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
    print(f"  CNP TAO: {len(cnp_tao)} sequences (excluded, n too small)")
    print(f"  Total CNP (excl TAO): {len(cnp_3rd) + len(cnp_4th) + len(cnp_6th)} sequences")

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

        if len(vals_a) < 5 or len(vals_b) < 5:
            results[metric] = {
                'n_a': len(vals_a), 'n_b': len(vals_b),
                'cohens_d': np.nan, 'p_value': np.nan,
                'mean_a': np.nan, 'mean_b': np.nan,
                'std_a': np.nan, 'std_b': np.nan,
            }
            continue

        d = compute_cohens_d(vals_a, vals_b)
        try:
            _, p = stats.mannwhitneyu(vals_b, vals_a, alternative='two-sided')
        except ValueError:
            p = np.nan

        results[metric] = {
            'n_a': len(vals_a), 'n_b': len(vals_b),
            'cohens_d': d, 'p_value': p,
            'mean_a': np.mean(vals_a), 'mean_b': np.mean(vals_b),
            'std_a': np.std(vals_a), 'std_b': np.std(vals_b),
        }

    return results


def run_analysis_for_direction(direction_name, direction_config, grouped_items):
    """Run full analysis for a single saccade direction."""
    axis = direction_config['axis']
    direction = direction_config['direction']

    print(f"\n  Computing dynamics for {direction_name}...")

    # Compute dynamics for each group
    dynamics = {}
    for group_name, items in grouped_items.items():
        dynamics[group_name] = compute_dynamics_for_group(items, axis=axis, direction=direction)

    # Print sample counts
    for group_name, dyn_list in dynamics.items():
        print(f"    {group_name}: {len(dyn_list)} valid samples")

    return dynamics


def run_comparisons(dynamics, direction_name):
    """Run all group comparisons for a direction."""
    comparisons = [
        ('MG', 'HC'),
        ('CNP_all', 'HC'),
        ('CNP_3rd', 'HC'),
        ('CNP_4th', 'HC'),
        ('CNP_6th', 'HC'),
        ('MG', 'CNP_all'),
        ('MG', 'CNP_3rd'),
        ('MG', 'CNP_4th'),
        ('MG', 'CNP_6th'),
    ]

    all_results = []

    for name_b, name_a in comparisons:
        if name_a not in dynamics or name_b not in dynamics:
            continue
        if len(dynamics[name_a]) < 5 or len(dynamics[name_b]) < 5:
            continue

        result = compare_groups(dynamics[name_a], dynamics[name_b], name_a, name_b)

        for metric, stats_dict in result.items():
            all_results.append({
                'Direction': direction_name,
                'Comparison': f'{name_b} vs {name_a}',
                'Group_A': name_a,
                'Group_B': name_b,
                'Metric': metric,
                'n_A': stats_dict['n_a'],
                'n_B': stats_dict['n_b'],
                'Mean_A': stats_dict['mean_a'],
                'Mean_B': stats_dict['mean_b'],
                'Std_A': stats_dict['std_a'],
                'Std_B': stats_dict['std_b'],
                'Cohen_d': stats_dict['cohens_d'],
                'p_value': stats_dict['p_value'],
            })

    return all_results


def main():
    print("="*80)
    print("EXPERIMENT 20: TEMPORAL ASYMMETRY DYNAMICS - ALL SACCADE DIRECTIONS")
    print("="*80)
    print("\nTesting fatigability hypothesis across 6 saccade directions")

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

    grouped_items = {
        'HC': hc_items,
        'MG': mg_items,
        'CNP_all': cnp_all,
        'CNP_3rd': cnp_3rd,
        'CNP_4th': cnp_4th,
        'CNP_6th': cnp_6th,
    }

    print(f"\nFinal counts:")
    for name, items in grouped_items.items():
        print(f"  {name}: {len(items)} sequences")

    # Run analysis for each saccade direction
    print("\n" + "="*80)
    print("RUNNING ANALYSIS FOR ALL SACCADE DIRECTIONS")
    print("="*80)

    all_results = []
    all_dynamics = {}  # Store dynamics for detailed output

    for direction_name, direction_config in SACCADE_DIRECTIONS.items():
        print(f"\n{'='*40}")
        print(f"Direction: {direction_name}")
        print(f"{'='*40}")

        # Compute dynamics for this direction
        dynamics = run_analysis_for_direction(direction_name, direction_config, grouped_items)
        all_dynamics[direction_name] = dynamics

        # Run comparisons
        results = run_comparisons(dynamics, direction_name)
        all_results.extend(results)

    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)

    # Save detailed results
    df_results.to_csv(os.path.join(RESULTS_DIR, 'detailed_results.csv'), index=False)
    print(f"\nDetailed results saved: {os.path.join(RESULTS_DIR, 'detailed_results.csv')}")

    # =============================================================================
    # ANALYSIS: SUMMARY BY DIRECTION
    # =============================================================================
    print("\n" + "="*80)
    print("SUMMARY BY DIRECTION")
    print("="*80)

    key_metrics = ['slope', 'delta', 'initial', 'growth_ratio']

    summary_rows = []
    for direction in SACCADE_DIRECTIONS.keys():
        df_dir = df_results[df_results['Direction'] == direction]

        for comparison in df_dir['Comparison'].unique():
            df_comp = df_dir[(df_dir['Comparison'] == comparison) &
                           (df_dir['Metric'].isin(key_metrics))]

            if df_comp.empty:
                continue

            mean_d = df_comp['Cohen_d'].mean()
            max_d = df_comp['Cohen_d'].max()
            n_sig = (df_comp['p_value'] < 0.05).sum()

            summary_rows.append({
                'Direction': direction,
                'Comparison': comparison,
                'Mean_d': mean_d,
                'Max_d': max_d,
                'N_Significant': f"{n_sig}/{len(key_metrics)}",
                'N_Sig_Int': n_sig,
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(RESULTS_DIR, 'summary_by_direction.csv'), index=False)

    # =============================================================================
    # KEY ANALYSIS: MG vs HC - Best Direction
    # =============================================================================
    print("\n" + "-"*60)
    print("MG vs HC: Which direction shows strongest temporal dynamics?")
    print("-"*60)

    df_mg_hc = df_summary[df_summary['Comparison'] == 'MG vs HC'].copy()
    if not df_mg_hc.empty:
        df_mg_hc = df_mg_hc.sort_values('Mean_d', ascending=False)

        print("\n| Direction        | Mean Cohen's d | Max d  | Significant |")
        print("|------------------|----------------|--------|-------------|")
        for _, row in df_mg_hc.iterrows():
            print(f"| {row['Direction']:16s} | {row['Mean_d']:>14.3f} | {row['Max_d']:>6.3f} | {row['N_Significant']:>11s} |")

        best_mg_hc = df_mg_hc.iloc[0]['Direction']
        best_mg_hc_d = df_mg_hc.iloc[0]['Mean_d']
        print(f"\nBest for MG vs HC: {best_mg_hc} (Mean d = {best_mg_hc_d:.3f})")

    # =============================================================================
    # KEY ANALYSIS: MG vs CNP - Best Direction
    # =============================================================================
    print("\n" + "-"*60)
    print("MG vs CNP_all: Which direction best distinguishes fatigability?")
    print("-"*60)

    df_mg_cnp = df_summary[df_summary['Comparison'] == 'MG vs CNP_all'].copy()
    if not df_mg_cnp.empty:
        df_mg_cnp = df_mg_cnp.sort_values('Mean_d', ascending=False)

        print("\n| Direction        | Mean Cohen's d | Max d  | Significant |")
        print("|------------------|----------------|--------|-------------|")
        for _, row in df_mg_cnp.iterrows():
            d_val = row['Mean_d']
            d_str = f"{d_val:>14.3f}" if not np.isnan(d_val) else "           N/A"
            max_d = row['Max_d']
            max_str = f"{max_d:>6.3f}" if not np.isnan(max_d) else "   N/A"
            print(f"| {row['Direction']:16s} | {d_str} | {max_str} | {row['N_Significant']:>11s} |")

        best_mg_cnp = df_mg_cnp.iloc[0]['Direction']
        best_mg_cnp_d = df_mg_cnp.iloc[0]['Mean_d']
        print(f"\nBest for MG vs CNP: {best_mg_cnp} (Mean d = {best_mg_cnp_d:.3f})")
        print(f"Exp 19 baseline (vertical_up only): Mean d = 0.066")

    # =============================================================================
    # KEY ANALYSIS: CNP_6th with Horizontal Saccades
    # =============================================================================
    print("\n" + "-"*60)
    print("CNP_6th Analysis: Does horizontal help? (6th nerve = lateral gaze)")
    print("-"*60)

    df_cnp6 = df_summary[df_summary['Comparison'] == 'MG vs CNP_6th'].copy()
    if not df_cnp6.empty:
        df_cnp6 = df_cnp6.sort_values('Mean_d', ascending=False)

        print("\n| Direction        | Mean Cohen's d | Max d  | Significant |")
        print("|------------------|----------------|--------|-------------|")
        for _, row in df_cnp6.iterrows():
            d_val = row['Mean_d']
            d_str = f"{d_val:>14.3f}" if not np.isnan(d_val) else "           N/A"
            max_d = row['Max_d']
            max_str = f"{max_d:>6.3f}" if not np.isnan(max_d) else "   N/A"
            print(f"| {row['Direction']:16s} | {d_str} | {max_str} | {row['N_Significant']:>11s} |")

    # =============================================================================
    # BEST DIRECTION PER COMPARISON
    # =============================================================================
    print("\n" + "-"*60)
    print("Best Direction for Each Comparison")
    print("-"*60)

    best_per_comparison = []
    for comparison in df_summary['Comparison'].unique():
        df_comp = df_summary[df_summary['Comparison'] == comparison]
        if not df_comp.empty and not df_comp['Mean_d'].isna().all():
            best_row = df_comp.loc[df_comp['Mean_d'].idxmax()]
            best_per_comparison.append({
                'Comparison': comparison,
                'Best_Direction': best_row['Direction'],
                'Mean_d': best_row['Mean_d'],
                'N_Significant': best_row['N_Significant'],
            })

    df_best = pd.DataFrame(best_per_comparison)
    df_best.to_csv(os.path.join(RESULTS_DIR, 'best_direction_per_comparison.csv'), index=False)

    print("\n| Comparison         | Best Direction   | Mean d | Sig    |")
    print("|--------------------|------------------|--------|--------|")
    for _, row in df_best.iterrows():
        d_val = row['Mean_d']
        d_str = f"{d_val:>6.3f}" if not np.isnan(d_val) else "   N/A"
        print(f"| {row['Comparison']:18s} | {row['Best_Direction']:16s} | {d_str} | {row['N_Significant']:>6s} |")

    # =============================================================================
    # DETAILED METRIC COMPARISON FOR KEY COMPARISONS
    # =============================================================================
    print("\n" + "="*80)
    print("DETAILED METRIC COMPARISON: MG vs CNP_all")
    print("="*80)

    df_mg_cnp_detail = df_results[(df_results['Comparison'] == 'MG vs CNP_all') &
                                  (df_results['Metric'].isin(key_metrics))]

    if not df_mg_cnp_detail.empty:
        print("\n| Direction        | Metric       | MG Mean  | CNP Mean | Cohen's d | p-value  |")
        print("|------------------|--------------|----------|----------|-----------|----------|")

        for direction in SACCADE_DIRECTIONS.keys():
            df_dir = df_mg_cnp_detail[df_mg_cnp_detail['Direction'] == direction]
            for _, row in df_dir.iterrows():
                d = row['Cohen_d']
                p = row['p_value']
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                d_str = f"{d:>9.3f}" if not np.isnan(d) else "      N/A"
                p_str = f"{p:>7.4f}{sig}" if not np.isnan(p) else "     N/A"
                mean_b = row['Mean_B']
                mean_a = row['Mean_A']
                mean_b_str = f"{mean_b:>8.4f}" if not np.isnan(mean_b) else "     N/A"
                mean_a_str = f"{mean_a:>8.4f}" if not np.isnan(mean_a) else "     N/A"
                print(f"| {direction:16s} | {row['Metric']:12s} | {mean_b_str} | {mean_a_str} | {d_str} | {p_str:>8s} |")

    # =============================================================================
    # GENERATE REPORT
    # =============================================================================
    print("\n" + "="*80)
    print("GENERATING REPORT")
    print("="*80)

    report_path = os.path.join(RESULTS_DIR, 'REPORT.md')
    with open(report_path, 'w') as f:
        f.write("# Experiment 20: Temporal Asymmetry Dynamics - All Saccade Directions\n\n")
        f.write("## Objective\n\n")
        f.write("Extend Experiment 19 to test the fatigability hypothesis across **all 6 saccade directions**:\n")
        f.write("- vertical_up, vertical_down, vertical_both\n")
        f.write("- horizontal_left, horizontal_right, horizontal_both\n\n")
        f.write("**Key Questions:**\n")
        f.write("1. Does any direction show stronger MG vs CNP discrimination than vertical_up?\n")
        f.write("2. Does horizontal work better for CNP_6th (6th nerve = lateral gaze)?\n")
        f.write("3. Which direction best captures the fatigability pattern?\n\n")

        f.write("## Results Summary\n\n")

        # MG vs HC
        f.write("### MG vs HC: Best Direction for Temporal Dynamics\n\n")
        f.write("| Direction | Mean Cohen's d | Max d | Significant |\n")
        f.write("|-----------|----------------|-------|-------------|\n")
        if not df_mg_hc.empty:
            for _, row in df_mg_hc.iterrows():
                d_val = row['Mean_d']
                d_str = f"{d_val:.3f}" if not np.isnan(d_val) else "N/A"
                max_d = row['Max_d']
                max_str = f"{max_d:.3f}" if not np.isnan(max_d) else "N/A"
                f.write(f"| {row['Direction']} | {d_str} | {max_str} | {row['N_Significant']} |\n")

        # MG vs CNP
        f.write("\n### MG vs CNP_all: Best Direction for Distinguishing Fatigability\n\n")
        f.write("| Direction | Mean Cohen's d | Max d | Significant |\n")
        f.write("|-----------|----------------|-------|-------------|\n")
        if not df_mg_cnp.empty:
            for _, row in df_mg_cnp.iterrows():
                d_val = row['Mean_d']
                d_str = f"{d_val:.3f}" if not np.isnan(d_val) else "N/A"
                max_d = row['Max_d']
                max_str = f"{max_d:.3f}" if not np.isnan(max_d) else "N/A"
                f.write(f"| {row['Direction']} | {d_str} | {max_str} | {row['N_Significant']} |\n")

        f.write("\n**Baseline (Exp 19, vertical_up only):** Mean d = 0.066\n\n")

        # Best per comparison
        f.write("### Best Direction Per Comparison\n\n")
        f.write("| Comparison | Best Direction | Mean d | Significant |\n")
        f.write("|------------|----------------|--------|-------------|\n")
        for _, row in df_best.iterrows():
            d_val = row['Mean_d']
            d_str = f"{d_val:.3f}" if not np.isnan(d_val) else "N/A"
            f.write(f"| {row['Comparison']} | {row['Best_Direction']} | {d_str} | {row['N_Significant']} |\n")

        # Key findings
        f.write("\n## Key Findings\n\n")

        if not df_mg_hc.empty and not df_mg_cnp.empty:
            best_hc = df_mg_hc.iloc[0]
            best_cnp = df_mg_cnp.iloc[0]

            f.write(f"1. **Best for MG vs HC**: {best_hc['Direction']} (Mean d = {best_hc['Mean_d']:.3f})\n")
            f.write(f"2. **Best for MG vs CNP**: {best_cnp['Direction']} (Mean d = {best_cnp['Mean_d']:.3f})\n")

            if best_cnp['Mean_d'] > 0.066:
                improvement = ((best_cnp['Mean_d'] - 0.066) / 0.066) * 100
                f.write(f"3. **Improvement over Exp 19**: +{improvement:.1f}% for MG vs CNP discrimination\n")
            else:
                f.write("3. **No improvement** over Exp 19 baseline for MG vs CNP\n")

        f.write("\n## Interpretation\n\n")
        f.write("*To be filled based on results analysis*\n\n")

        f.write("---\n")
        f.write(f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*\n")

    print(f"\nReport saved to: {report_path}")
    print(f"Summary by direction: {os.path.join(RESULTS_DIR, 'summary_by_direction.csv')}")
    print(f"Best direction per comparison: {os.path.join(RESULTS_DIR, 'best_direction_per_comparison.csv')}")

    print("\n" + "="*80)
    print("EXPERIMENT 20 COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
