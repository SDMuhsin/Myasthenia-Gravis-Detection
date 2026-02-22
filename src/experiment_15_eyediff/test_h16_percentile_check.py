#!/usr/bin/env python3
"""
H16 Empirical Check: Does p90 reduce or amplify HC asymmetry?
Test whether percentile-based aggregation helps with HC baseline problem.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_timeseries_data, merge_mg_classes

BASE_DIR = './data'
CLASS_DEFINITIONS = {
    'HC': {'path': 'Healthy control', 'label': 0},
    'MG': {'path': 'Definite MG', 'label': 1},
    'Probable_MG': {'path': 'Probable MG', 'label': 1},
}
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']

print("="*80)
print("H16 Empirical Check: Percentile vs Mean Aggregation")
print("="*80)

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 100)
sequences = merge_mg_classes(raw_sequences)

def compute_windowed_degradation_percentiles(eye_pos, target_pos, n_windows=10):
    """Compute degradation across sliding windows, return percentiles"""
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    if np.sum(valid) < 50:
        return np.nan, np.nan, np.nan

    eye_clean = eye_pos[valid]
    target_clean = target_pos[valid]
    error = np.abs(eye_clean - target_clean)
    n = len(error)

    # Baseline = first 20%
    baseline_n = int(n * 0.2)
    baseline_error = np.mean(error[:baseline_n])

    # Compute degradation for multiple windows
    window_size = n // n_windows
    degradations = []

    for i in range(1, n_windows):
        start = i * window_size
        end = min((i+1) * window_size, n)
        if end - start < 10:  # Skip very small windows
            continue
        window_error = np.mean(error[start:end])
        degradation = window_error - baseline_error
        degradations.append(degradation)

    if len(degradations) < 3:
        return np.nan, np.nan, np.nan

    degradations = np.array(degradations)
    return np.mean(degradations), np.percentile(degradations, 90), np.percentile(degradations, 50)

# Compute metrics
results = []

for seq in sequences[:200]:  # First 200 for speed
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    left_mean, left_p90, left_p50 = compute_windowed_degradation_percentiles(
        df['LV'].values, df['TargetV'].values)
    right_mean, right_p90, right_p50 = compute_windowed_degradation_percentiles(
        df['RV'].values, df['TargetV'].values)

    if not (np.isnan(left_mean) or np.isnan(right_mean)):
        results.append({
            'class': 1 if seq['class_name'] == 'MG' else 0,
            'asym_mean': abs(left_mean - right_mean),
            'asym_p90': abs(left_p90 - right_p90),
            'asym_p50': abs(left_p50 - right_p50)
        })

results_df = pd.DataFrame(results)
hc_results = results_df[results_df['class'] == 0]
mg_results = results_df[results_df['class'] == 1]

print(f"\nAnalyzed {len(hc_results)} HC, {len(mg_results)} MG")

print(f"\n1. HC ASYMMETRY COMPARISON:")
print(f"   Mean degradation asymmetry: {hc_results['asym_mean'].mean():.3f} ± {hc_results['asym_mean'].std():.3f}")
print(f"   P90 degradation asymmetry:  {hc_results['asym_p90'].mean():.3f} ± {hc_results['asym_p90'].std():.3f}")
print(f"   P50 degradation asymmetry:  {hc_results['asym_p50'].mean():.3f} ± {hc_results['asym_p50'].std():.3f}")

amplification_p90 = hc_results['asym_p90'].mean() / hc_results['asym_mean'].mean()
amplification_p50 = hc_results['asym_p50'].mean() / hc_results['asym_mean'].mean()
print(f"\n   P90 amplification factor: {amplification_p90:.2f}x")
print(f"   P50 amplification factor: {amplification_p50:.2f}x")

print(f"\n2. MG ASYMMETRY COMPARISON:")
print(f"   Mean degradation asymmetry: {mg_results['asym_mean'].mean():.3f} ± {mg_results['asym_mean'].std():.3f}")
print(f"   P90 degradation asymmetry:  {mg_results['asym_p90'].mean():.3f} ± {mg_results['asym_p90'].std():.3f}")
print(f"   P50 degradation asymmetry:  {mg_results['asym_p50'].mean():.3f} ± {mg_results['asym_p50'].std():.3f}")

print(f"\n3. EFFECT SIZE COMPARISON:")
from scipy import stats

# Mean-based
hc_mean_asym = hc_results['asym_mean'].values
mg_mean_asym = mg_results['asym_mean'].values
pooled_std_mean = np.sqrt((np.std(hc_mean_asym)**2 + np.std(mg_mean_asym)**2) / 2)
d_mean = (mg_mean_asym.mean() - hc_mean_asym.mean()) / pooled_std_mean

# P90-based
hc_p90_asym = hc_results['asym_p90'].values
mg_p90_asym = mg_results['asym_p90'].values
pooled_std_p90 = np.sqrt((np.std(hc_p90_asym)**2 + np.std(mg_p90_asym)**2) / 2)
d_p90 = (mg_p90_asym.mean() - hc_p90_asym.mean()) / pooled_std_p90

# P50-based
hc_p50_asym = hc_results['asym_p50'].values
mg_p50_asym = mg_results['asym_p50'].values
pooled_std_p50 = np.sqrt((np.std(hc_p50_asym)**2 + np.std(mg_p50_asym)**2) / 2)
d_p50 = (mg_p50_asym.mean() - hc_p50_asym.mean()) / pooled_std_p50

print(f"   Mean-based: d={d_mean:.3f}")
print(f"   P90-based:  d={d_p90:.3f} ({'+' if d_p90 > d_mean else ''}{d_p90-d_mean:.3f})")
print(f"   P50-based:  d={d_p50:.3f} ({'+' if d_p50 > d_mean else ''}{d_p50-d_mean:.3f})")

print(f"\n4. HC≈0 TEST:")
_, p_hc_mean = stats.wilcoxon(hc_mean_asym)
_, p_hc_p90 = stats.wilcoxon(hc_p90_asym)
_, p_hc_p50 = stats.wilcoxon(hc_p50_asym)

print(f"   Mean-based: p={p_hc_mean:.6f} {'✓ PASS' if p_hc_mean >= 0.05 else '✗ FAIL'}")
print(f"   P90-based:  p={p_hc_p90:.6f} {'✓ PASS' if p_hc_p90 >= 0.05 else '✗ FAIL'}")
print(f"   P50-based:  p={p_hc_p50:.6f} {'✓ PASS' if p_hc_p50 >= 0.05 else '✗ FAIL'}")

print(f"\n5. VERDICT:")
if d_p90 > d_mean and p_hc_p90 >= 0.05:
    print("   ✓✓✓ P90 IMPROVES both effect size AND passes HC≈0!")
elif d_p90 > d_mean:
    print(f"   ✓ P90 improves effect size but still fails HC≈0")
elif p_hc_p90 >= 0.05:
    print(f"   ✓ P90 passes HC≈0 but decreases effect size")
else:
    print(f"   ✗ P90 makes it WORSE on both metrics - ABANDON hypothesis")

print("="*80)
