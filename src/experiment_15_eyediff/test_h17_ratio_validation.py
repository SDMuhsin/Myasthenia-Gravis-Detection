#!/usr/bin/env python3
"""
H17 Empirical Validation: Asymmetry Ratio on Full Dataset
Test if worse/better ratio works on ALL patients, not just high-asymmetry subgroup.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_timeseries_data, merge_mg_classes
from equations import h2_mad_variability

BASE_DIR = './data'
CLASS_DEFINITIONS = {
    'HC': {'path': 'Healthy control', 'label': 0},
    'MG': {'path': 'Definite MG', 'label': 1},
    'Probable_MG': {'path': 'Probable MG', 'label': 1},
}
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']

print("="*80)
print("H17 Empirical Validation: Asymmetry Ratio")
print("="*80)

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

def compute_early_late_degradation(eye_pos, target_pos):
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    if np.sum(valid) < 50:
        return np.nan
    eye_clean = eye_pos[valid]
    target_clean = target_pos[valid]
    error = np.abs(eye_clean - target_clean)
    n = len(error)
    early_n = int(n * 0.2)
    late_n = int(n * 0.2)
    early_mean = np.mean(error[:early_n])
    late_mean = np.mean(error[-late_n:])
    return late_mean - early_mean

# Compute metrics for ALL sequences
results = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    left_mad_dict = h2_mad_variability(df['LV'].values, df['TargetV'].values)
    right_mad_dict = h2_mad_variability(df['RV'].values, df['TargetV'].values)
    left_mad_val = left_mad_dict['mad_position']
    right_mad_val = right_mad_dict['mad_position']

    left_deg_val = compute_early_late_degradation(df['LV'].values, df['TargetV'].values)
    right_deg_val = compute_early_late_degradation(df['RV'].values, df['TargetV'].values)

    if not (np.isnan(left_mad_val) or np.isnan(right_mad_val) or
            np.isnan(left_deg_val) or np.isnan(right_deg_val)):

        combined_left = 0.45 * left_mad_val + 0.55 * left_deg_val
        combined_right = 0.45 * right_mad_val + 0.55 * right_deg_val

        # H11 absolute asymmetry
        asymmetry_abs = abs(combined_left - combined_right)

        # H17 ratio-based metrics
        better = min(abs(combined_left), abs(combined_right))
        worse = max(abs(combined_left), abs(combined_right))

        ratio = worse / (better + 0.5)
        ratio_minus_1 = ratio - 1.0

        results.append({
            'class': 1 if seq['class_name'] == 'MG' else 0,
            'asymmetry_abs': asymmetry_abs,
            'ratio': ratio,
            'ratio_minus_1': ratio_minus_1,
            'better': better,
            'worse': worse
        })

results_df = pd.DataFrame(results)
hc_df = results_df[results_df['class'] == 0]
mg_df = results_df[results_df['class'] == 1]

print(f"\nDataset: {len(hc_df)} HC, {len(mg_df)} MG")

# Test different formulations
metrics = {
    'H11 (absolute)': 'asymmetry_abs',
    'H17 ratio': 'ratio',
    'H17 ratio-1': 'ratio_minus_1'
}

print(f"\n{'Metric':<20} {'HC mean':<12} {'MG mean':<12} {'Cohen d':<10} {'p_MW':<10} {'p_HC≈0':<10} {'Val'}")
print("="*90)

for name, col in metrics.items():
    hc_vals = hc_df[col].values
    mg_vals = mg_df[col].values

    hc_mean = np.mean(hc_vals)
    hc_std = np.std(hc_vals)
    mg_mean = np.mean(mg_vals)
    mg_std = np.std(mg_vals)

    pooled_std = np.sqrt((hc_std**2 + mg_std**2) / 2)
    d = (mg_mean - hc_mean) / pooled_std

    _, p_mw = stats.mannwhitneyu(mg_vals, hc_vals, alternative='greater')
    _, p_hc = stats.wilcoxon(hc_vals)
    _, p_mg = stats.wilcoxon(mg_vals)

    val_score = (p_mw < 0.05) + (p_hc >= 0.05) + (p_mg < 0.05)

    print(f"{name:<20} {hc_mean:<12.3f} {mg_mean:<12.3f} {d:<10.3f} {p_mw:<10.6f} {p_hc:<10.6f} {val_score}/3")

# Check if ratio improves on absolute
hc_abs = hc_df['asymmetry_abs'].values
mg_abs = mg_df['asymmetry_abs'].values
hc_ratio = hc_df['ratio_minus_1'].values
mg_ratio = mg_df['ratio_minus_1'].values

d_abs = (np.mean(mg_abs) - np.mean(hc_abs)) / np.sqrt((np.std(hc_abs)**2 + np.std(mg_abs)**2) / 2)
d_ratio = (np.mean(mg_ratio) - np.mean(hc_ratio)) / np.sqrt((np.std(hc_ratio)**2 + np.std(mg_ratio)**2) / 2)

print(f"\n{'='*90}")
print(f"VERDICT:")
if d_ratio >= 0.5 and d_ratio > d_abs:
    print(f"  ✓✓✓ Ratio achieves d≥0.5 AND improves over H11!")
    print(f"  Improvement: {d_abs:.3f} → {d_ratio:.3f} ({100*(d_ratio-d_abs)/d_abs:+.1f}%)")
elif d_ratio > d_abs:
    print(f"  ✓ Ratio improves over H11 but d<0.5")
    print(f"  Improvement: {d_abs:.3f} → {d_ratio:.3f} ({100*(d_ratio-d_abs)/d_abs:+.1f}%)")
else:
    print(f"  ✗ Ratio WORSE than or equal to H11")
    print(f"  Change: {d_abs:.3f} → {d_ratio:.3f} ({100*(d_ratio-d_abs)/d_abs:+.1f}%)")

print("="*90)
