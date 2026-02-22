#!/usr/bin/env python3
"""
Experiment 18: CNP (Cranial Nerve Palsy) Analysis

Apply the analytical metrics from Experiment 15 to CNP patients and compare:
1. CNP vs HC - Can metrics detect CNP?
2. CNP subtypes vs HC - Which subtypes are detectable?
3. MG vs CNP - Can metrics distinguish MG from CNP?

Using top 5 analytical metrics:
- H38b (composite)
- FAT1 (error degradation)
- FAT3 (error slope)
- TTT2 (sustained latency)
- TTT3 (first entry latency)
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
RESULTS_DIR = './results/exp_18_cnp_analysis'
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']
CSV_ENCODING = 'utf-16-le'
CSV_SEPARATOR = ','
MIN_SEQ_LEN_THRESHOLD = 50
SAMPLE_RATE = 120  # Hz
SACCADE_THRESHOLD = 5.0  # degrees


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

    # Check if folder has patient subdirectories or direct CSV files
    patient_dirs = [d for d in os.listdir(folder_path)
                    if os.path.isdir(os.path.join(folder_path, d))]

    if patient_dirs:
        # Folder has patient subdirectories
        for patient_folder in tqdm(patient_dirs, desc=f"  Loading {class_name}", leave=False):
            patient_path = os.path.join(folder_path, patient_folder)
            csv_files = glob.glob(os.path.join(patient_path, '*.csv'))

            for csv_file in csv_files:
                item = load_single_csv(csv_file, class_name, label, patient_folder)
                if item is not None:
                    items.append(item)
    else:
        # Direct CSV files in folder
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

        # Check required columns
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
# METRIC COMPUTATION (same as Exp 17)
# =============================================================================

def compute_fat1_metric(eye_l, eye_r, target, saccade_indices):
    """FAT1: Error degradation (late - early error asymmetry)."""
    n_samples = len(eye_l)
    errors_l, errors_r = [], []

    for idx in saccade_indices:
        if idx >= n_samples:
            continue
        start = min(idx + 24, n_samples)
        end = min(idx + 48, n_samples)

        if end > start + 5:
            errors_l.append(np.mean(np.abs(eye_l[start:end] - target[start:end])))
            errors_r.append(np.mean(np.abs(eye_r[start:end] - target[start:end])))

    if len(errors_l) < 3:
        return np.nan

    third = max(1, len(errors_l) // 3)
    deg_l = np.mean(errors_l[-third:]) - np.mean(errors_l[:third])
    deg_r = np.mean(errors_r[-third:]) - np.mean(errors_r[:third])

    return np.abs(deg_l - deg_r)


def compute_fat3_metric(eye_l, eye_r, target, saccade_indices):
    """FAT3: Error slope asymmetry."""
    n_samples = len(eye_l)
    errors_l, errors_r = [], []

    for idx in saccade_indices:
        if idx >= n_samples:
            continue
        start = min(idx + 24, n_samples)
        end = min(idx + 48, n_samples)

        if end > start + 5:
            errors_l.append(np.mean(np.abs(eye_l[start:end] - target[start:end])))
            errors_r.append(np.mean(np.abs(eye_r[start:end] - target[start:end])))

    if len(errors_l) < 3:
        return np.nan

    x = np.arange(len(errors_l))
    slope_l, _, _, _, _ = stats.linregress(x, errors_l)
    slope_r, _, _, _, _ = stats.linregress(x, errors_r)

    return np.abs(slope_l - slope_r)


def compute_ttt2_metric(eye_l, eye_r, target, saccade_indices):
    """TTT2: Sustained 100ms latency asymmetry."""
    n_samples = len(eye_l)
    tolerance = 3.0
    sustain_samples = 12

    latencies_l, latencies_r = [], []

    for idx in saccade_indices:
        if idx >= n_samples:
            continue
        window_end = min(idx + 120, n_samples)

        lat_l = np.nan
        for t in range(idx, window_end - sustain_samples):
            if all(np.abs(eye_l[t:t+sustain_samples] - target[t:t+sustain_samples]) <= tolerance):
                lat_l = (t - idx) / SAMPLE_RATE * 1000
                break

        lat_r = np.nan
        for t in range(idx, window_end - sustain_samples):
            if all(np.abs(eye_r[t:t+sustain_samples] - target[t:t+sustain_samples]) <= tolerance):
                lat_r = (t - idx) / SAMPLE_RATE * 1000
                break

        if not np.isnan(lat_l) and not np.isnan(lat_r):
            latencies_l.append(lat_l)
            latencies_r.append(lat_r)

    if len(latencies_l) < 3:
        return np.nan

    third = max(1, len(latencies_l) // 3)
    return np.abs(np.mean(latencies_l[:third]) - np.mean(latencies_r[:third]))


def compute_ttt3_metric(eye_l, eye_r, target, saccade_indices):
    """TTT3: First entry 4° tolerance latency asymmetry."""
    n_samples = len(eye_l)
    tolerance = 4.0

    latencies_l, latencies_r = [], []

    for idx in saccade_indices:
        if idx >= n_samples:
            continue
        window_end = min(idx + 60, n_samples)

        lat_l = np.nan
        for t in range(idx, window_end):
            if np.abs(eye_l[t] - target[t]) <= tolerance:
                lat_l = (t - idx) / SAMPLE_RATE * 1000
                break

        lat_r = np.nan
        for t in range(idx, window_end):
            if np.abs(eye_r[t] - target[t]) <= tolerance:
                lat_r = (t - idx) / SAMPLE_RATE * 1000
                break

        if not np.isnan(lat_l) and not np.isnan(lat_r):
            latencies_l.append(lat_l)
            latencies_r.append(lat_r)

    if len(latencies_l) < 3:
        return np.nan

    third = max(1, len(latencies_l) // 3)
    return np.abs(np.mean(latencies_l[:third]) - np.mean(latencies_r[:third]))


def compute_h38b_metric(eye_l, eye_r, target, saccade_indices):
    """H38b: Composite asymmetry metric."""
    n_samples = len(eye_l)
    tolerance = 3.0

    errors_l, errors_r = [], []
    latencies_l, latencies_r = [], []

    for idx in saccade_indices:
        if idx >= n_samples:
            continue

        # Error
        start = min(idx + 24, n_samples)
        end = min(idx + 48, n_samples)
        if end > start + 5:
            errors_l.append(np.mean(np.abs(eye_l[start:end] - target[start:end])))
            errors_r.append(np.mean(np.abs(eye_r[start:end] - target[start:end])))

        # Latency
        window_end = min(idx + 60, n_samples)
        lat_l = np.nan
        for t in range(idx, window_end):
            if np.abs(eye_l[t] - target[t]) <= tolerance:
                lat_l = (t - idx) / SAMPLE_RATE * 1000
                break
        lat_r = np.nan
        for t in range(idx, window_end):
            if np.abs(eye_r[t] - target[t]) <= tolerance:
                lat_r = (t - idx) / SAMPLE_RATE * 1000
                break

        if not np.isnan(lat_l):
            latencies_l.append(lat_l)
        if not np.isnan(lat_r):
            latencies_r.append(lat_r)

    if len(errors_l) < 3 or len(latencies_l) == 0:
        return np.nan

    # MAD
    mad_l = np.median(np.abs(np.array(errors_l) - np.median(errors_l)))
    mad_r = np.median(np.abs(np.array(errors_r) - np.median(errors_r)))

    # Degradation
    n_err = len(errors_l)
    early_n = max(1, int(n_err * 0.2))
    late_n = max(1, int(n_err * 0.2))
    deg_l = np.mean(errors_l[-late_n:]) - np.mean(errors_l[:early_n])
    deg_r = np.mean(errors_r[-late_n:]) - np.mean(errors_r[:early_n])

    # Latency
    n_fast = max(1, len(latencies_l) // 4)
    lat_l = np.mean(sorted(latencies_l)[:n_fast])
    lat_r = np.mean(sorted(latencies_r)[:n_fast]) if latencies_r else lat_l

    # Composite
    cv_asym = np.abs(mad_l - mad_r) / ((mad_l + mad_r) / 2 + 1e-6)
    deg_asym = np.abs(deg_l - deg_r)
    lat_asym = np.abs(lat_l - lat_r)

    return 0.5 * (0.30 * cv_asym + 0.70 * deg_asym) + 0.5 * (lat_asym / 100)


# =============================================================================
# ANALYSIS
# =============================================================================

def compute_metrics_for_group(items, axis='vertical', direction='positive'):
    """Compute all 5 metrics for a group of items."""
    metrics = {'H38b': [], 'FAT1': [], 'FAT3': [], 'TTT2': [], 'TTT3': []}

    for item in items:
        data = item['data']

        if axis == 'horizontal':
            eye_l, eye_r, target = data[:, 0], data[:, 1], data[:, 4]
        else:
            eye_l, eye_r, target = data[:, 2], data[:, 3], data[:, 5]

        saccade_indices = detect_saccades(target, direction=direction)
        if len(saccade_indices) < 3:
            continue

        h38b = compute_h38b_metric(eye_l, eye_r, target, saccade_indices)
        fat1 = compute_fat1_metric(eye_l, eye_r, target, saccade_indices)
        fat3 = compute_fat3_metric(eye_l, eye_r, target, saccade_indices)
        ttt2 = compute_ttt2_metric(eye_l, eye_r, target, saccade_indices)
        ttt3 = compute_ttt3_metric(eye_l, eye_r, target, saccade_indices)

        if not np.isnan(h38b): metrics['H38b'].append(h38b)
        if not np.isnan(fat1): metrics['FAT1'].append(fat1)
        if not np.isnan(fat3): metrics['FAT3'].append(fat3)
        if not np.isnan(ttt2): metrics['TTT2'].append(ttt2)
        if not np.isnan(ttt3): metrics['TTT3'].append(ttt3)

    return metrics


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


def compare_groups(metrics_a, metrics_b, name_a, name_b):
    """Compare two groups across all metrics."""
    results = {}

    for metric in ['H38b', 'FAT1', 'FAT3', 'TTT2', 'TTT3']:
        vals_a = metrics_a[metric]
        vals_b = metrics_b[metric]

        if len(vals_a) < 10 or len(vals_b) < 10:
            results[metric] = {
                'n_a': len(vals_a), 'n_b': len(vals_b),
                'cohens_d': np.nan, 'p_value': np.nan,
                'mean_a': np.nan, 'mean_b': np.nan
            }
            continue

        d = compute_cohens_d(vals_a, vals_b)
        _, p = stats.mannwhitneyu(vals_b, vals_a, alternative='two-sided')

        results[metric] = {
            'n_a': len(vals_a), 'n_b': len(vals_b),
            'cohens_d': d, 'p_value': p,
            'mean_a': np.mean(vals_a), 'mean_b': np.mean(vals_b)
        }

    return results


def main():
    print("="*80)
    print("EXPERIMENT 18: CNP (CRANIAL NERVE PALSY) ANALYSIS")
    print("Applying Analytical Metrics to CNP Patients")
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
    cnp_tao = [x for x in all_items if x['class_name'] == 'CNP_TAO']
    cnp_all = cnp_3rd + cnp_4th + cnp_6th + cnp_tao

    print(f"\nFinal counts:")
    print(f"  HC: {len(hc_items)}")
    print(f"  MG: {len(mg_items)}")
    print(f"  CNP (all): {len(cnp_all)}")
    print(f"    - 3rd: {len(cnp_3rd)}")
    print(f"    - 4th: {len(cnp_4th)}")
    print(f"    - 6th: {len(cnp_6th)}")
    print(f"    - TAO: {len(cnp_tao)}")

    # Compute metrics for each group (using upward vertical saccades)
    print("\n" + "="*80)
    print("COMPUTING METRICS (Upward Vertical Saccades)")
    print("="*80)

    print("\nComputing for HC...")
    metrics_hc = compute_metrics_for_group(hc_items, axis='vertical', direction='positive')

    print("Computing for MG...")
    metrics_mg = compute_metrics_for_group(mg_items, axis='vertical', direction='positive')

    print("Computing for CNP (all)...")
    metrics_cnp_all = compute_metrics_for_group(cnp_all, axis='vertical', direction='positive')

    print("Computing for CNP subtypes...")
    metrics_cnp_3rd = compute_metrics_for_group(cnp_3rd, axis='vertical', direction='positive')
    metrics_cnp_4th = compute_metrics_for_group(cnp_4th, axis='vertical', direction='positive')
    metrics_cnp_6th = compute_metrics_for_group(cnp_6th, axis='vertical', direction='positive')
    metrics_cnp_tao = compute_metrics_for_group(cnp_tao, axis='vertical', direction='positive')

    # Perform comparisons
    print("\n" + "="*80)
    print("PERFORMING COMPARISONS")
    print("="*80)

    comparisons = [
        ('MG', 'HC', metrics_mg, metrics_hc),
        ('CNP_all', 'HC', metrics_cnp_all, metrics_hc),
        ('CNP_3rd', 'HC', metrics_cnp_3rd, metrics_hc),
        ('CNP_4th', 'HC', metrics_cnp_4th, metrics_hc),
        ('CNP_6th', 'HC', metrics_cnp_6th, metrics_hc),
        ('CNP_TAO', 'HC', metrics_cnp_tao, metrics_hc),
        ('MG', 'CNP_all', metrics_mg, metrics_cnp_all),
        ('MG', 'CNP_3rd', metrics_mg, metrics_cnp_3rd),
        ('MG', 'CNP_4th', metrics_mg, metrics_cnp_4th),
        ('MG', 'CNP_6th', metrics_mg, metrics_cnp_6th),
    ]

    all_results = []

    for name_b, name_a, metrics_b, metrics_a in comparisons:
        result = compare_groups(metrics_a, metrics_b, name_a, name_b)

        for metric, stats_dict in result.items():
            all_results.append({
                'Comparison': f'{name_b} vs {name_a}',
                'Group_A': name_a,
                'Group_B': name_b,
                'Metric': metric,
                'n_A': stats_dict['n_a'],
                'n_B': stats_dict['n_b'],
                'Cohen_d': stats_dict['cohens_d'],
                'p_value': stats_dict['p_value'],
                'Mean_A': stats_dict['mean_a'],
                'Mean_B': stats_dict['mean_b'],
            })

    df_results = pd.DataFrame(all_results)

    # Print results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    # Create pivot table for each comparison
    for comparison in df_results['Comparison'].unique():
        df_comp = df_results[df_results['Comparison'] == comparison]
        print(f"\n--- {comparison} ---")

        for _, row in df_comp.iterrows():
            d = row['Cohen_d']
            p = row['p_value']
            sig = '*' if p < 0.05 else ''
            d_str = f"{d:.3f}" if not np.isnan(d) else "N/A"
            p_str = f"{p:.4f}" if not np.isnan(p) else "N/A"
            print(f"  {row['Metric']}: d={d_str}, p={p_str} {sig}")

        # Mean Cohen's d
        mean_d = df_comp['Cohen_d'].mean()
        print(f"  Mean d: {mean_d:.3f}")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: Mean Cohen's d by Comparison")
    print("="*80)

    summary_rows = []
    for comparison in df_results['Comparison'].unique():
        df_comp = df_results[df_results['Comparison'] == comparison]
        mean_d = df_comp['Cohen_d'].mean()
        best_metric = df_comp.loc[df_comp['Cohen_d'].idxmax(), 'Metric'] if not df_comp['Cohen_d'].isna().all() else 'N/A'
        best_d = df_comp['Cohen_d'].max()
        n_sig = (df_comp['p_value'] < 0.05).sum()

        summary_rows.append({
            'Comparison': comparison,
            'Mean_d': mean_d,
            'Best_Metric': best_metric,
            'Best_d': best_d,
            'N_Significant': n_sig,
        })

        print(f"{comparison:25s}: Mean d = {mean_d:.3f}, Best = {best_metric} ({best_d:.3f}), {n_sig}/5 sig")

    df_summary = pd.DataFrame(summary_rows)

    # Save results
    df_results.to_csv(os.path.join(RESULTS_DIR, 'detailed_results.csv'), index=False)
    df_summary.to_csv(os.path.join(RESULTS_DIR, 'summary.csv'), index=False)

    # Generate report
    print("\n" + "="*80)
    print("GENERATING REPORT")
    print("="*80)

    report_path = os.path.join(RESULTS_DIR, 'REPORT.md')
    with open(report_path, 'w') as f:
        f.write("# Experiment 18: CNP Analysis Report\n\n")
        f.write("## Summary\n\n")
        f.write("This experiment applies the analytical metrics from Experiment 15 to CNP patients.\n\n")

        f.write("## Key Findings\n\n")
        f.write("### vs HC (Healthy Control) Comparisons\n\n")
        f.write("| Comparison | Mean d | Best Metric | Best d | Significant |\n")
        f.write("|------------|--------|-------------|--------|-------------|\n")

        for _, row in df_summary.iterrows():
            if 'HC' in row['Comparison']:
                f.write(f"| {row['Comparison']} | {row['Mean_d']:.3f} | {row['Best_Metric']} | {row['Best_d']:.3f} | {row['N_Significant']}/5 |\n")

        f.write("\n### MG vs CNP Comparisons\n\n")
        f.write("| Comparison | Mean d | Best Metric | Best d | Significant |\n")
        f.write("|------------|--------|-------------|--------|-------------|\n")

        for _, row in df_summary.iterrows():
            if 'MG vs CNP' in row['Comparison']:
                f.write(f"| {row['Comparison']} | {row['Mean_d']:.3f} | {row['Best_Metric']} | {row['Best_d']:.3f} | {row['N_Significant']}/5 |\n")

        f.write("\n## Interpretation\n\n")

        # MG vs HC baseline
        mg_hc_d = df_summary[df_summary['Comparison'] == 'MG vs HC']['Mean_d'].values[0]
        f.write(f"**Baseline (MG vs HC):** Mean Cohen's d = {mg_hc_d:.3f}\n\n")

        # CNP vs HC
        cnp_hc_d = df_summary[df_summary['Comparison'] == 'CNP_all vs HC']['Mean_d'].values[0]
        f.write(f"**CNP vs HC:** Mean Cohen's d = {cnp_hc_d:.3f}\n\n")

        if cnp_hc_d > 0.2:
            f.write("- CNP patients show detectable differences from HC using these metrics.\n")
        else:
            f.write("- CNP patients show minimal differences from HC using these metrics.\n")

        # MG vs CNP
        mg_cnp_d = df_summary[df_summary['Comparison'] == 'MG vs CNP_all']['Mean_d'].values[0]
        f.write(f"\n**MG vs CNP:** Mean Cohen's d = {mg_cnp_d:.3f}\n\n")

        if mg_cnp_d > 0.2:
            f.write("- The metrics can distinguish between MG and CNP patients.\n")
        else:
            f.write("- The metrics show limited ability to distinguish MG from CNP.\n")

        f.write("\n---\n")
        f.write(f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*\n")

    print(f"\nReport saved to: {report_path}")
    print(f"Detailed results saved to: {os.path.join(RESULTS_DIR, 'detailed_results.csv')}")
    print(f"Summary saved to: {os.path.join(RESULTS_DIR, 'summary.csv')}")

    print("\n" + "="*80)
    print("EXPERIMENT 18 COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
