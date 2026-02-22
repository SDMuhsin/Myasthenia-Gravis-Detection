#!/usr/bin/env python3
"""
Phase 4 empirical check: Are MAD and degradation correlated?
If r < 0.5, they capture independent information → combination will help
If r > 0.7, they're redundant → combination won't help
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_timeseries_data, merge_mg_classes
from equations import h2_mad_variability, compute_eye_difference

# Configuration
BASE_DIR = './data'
CLASS_DEFINITIONS = {
    'HC': {'path': 'Healthy control', 'label': 0},
    'MG': {'path': 'Definite MG', 'label': 1},
    'Probable_MG': {'path': 'Probable MG', 'label': 1},
}
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']

print("Phase 4: Correlation check between MAD and Degradation")

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

mad_asymmetries = []
degradation_asymmetries = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    # MAD asymmetry
    left_mad = h2_mad_variability(df['LV'].values, df['TargetV'].values)
    right_mad = h2_mad_variability(df['RV'].values, df['TargetV'].values)
    mad_asym = compute_eye_difference(left_mad, right_mad, 'mad_position')

    # Degradation asymmetry
    left_deg = compute_early_late_degradation(df['LV'].values, df['TargetV'].values)
    right_deg = compute_early_late_degradation(df['RV'].values, df['TargetV'].values)

    if not np.isnan(mad_asym) and not np.isnan(left_deg) and not np.isnan(right_deg):
        deg_asym = abs(left_deg - right_deg)
        mad_asymmetries.append(mad_asym)
        degradation_asymmetries.append(deg_asym)

# Correlation
correlation, p_value = stats.pearsonr(mad_asymmetries, degradation_asymmetries)

print(f"\nAnalyzed {len(mad_asymmetries)} sequences")
print(f"\nPearson correlation between MAD and Degradation asymmetries:")
print(f"  r = {correlation:.4f}")
print(f"  p = {p_value:.6f}")
print(f"  r² = {correlation**2:.4f} (shared variance)")

print("\n" + "="*80)
if abs(correlation) < 0.3:
    print("✓✓ INDEPENDENT METRICS (r < 0.3)")
    print("Combination will capture orthogonal information → GO")
elif abs(correlation) < 0.5:
    print("✓ WEAKLY CORRELATED (0.3 ≤ r < 0.5)")
    print("Some overlap but still useful to combine → GO")
elif abs(correlation) < 0.7:
    print("→ MODERATELY CORRELATED (0.5 ≤ r < 0.7)")
    print("Significant overlap, combination may help moderately → CAUTIOUS GO")
else:
    print("✗ HIGHLY CORRELATED (r ≥ 0.7)")
    print("Redundant metrics, combination won't help → NO-GO")

# Also test simple combination
combined = [(m + d) / 2 for m, d in zip(mad_asymmetries, degradation_asymmetries)]

# Split by class for effect size
mg_indices = []
hc_indices = []

for i, seq in enumerate(sequences):
    if i < len(mad_asymmetries):
        if seq['class_name'] == 'MG':
            mg_indices.append(i)
        else:
            hc_indices.append(i)

hc_mad = [mad_asymmetries[i] for i in hc_indices if i < len(mad_asymmetries)]
mg_mad = [mad_asymmetries[i] for i in mg_indices if i < len(mad_asymmetries)]

hc_deg = [degradation_asymmetries[i] for i in hc_indices if i < len(degradation_asymmetries)]
mg_deg = [degradation_asymmetries[i] for i in mg_indices if i < len(degradation_asymmetries)]

hc_combined = [combined[i] for i in hc_indices if i < len(combined)]
mg_combined = [combined[i] for i in mg_indices if i < len(combined)]

# Cohen's d for combined
hc_mean = np.mean(hc_combined)
hc_std = np.std(hc_combined)
mg_mean = np.mean(mg_combined)
mg_std = np.std(mg_combined)
pooled_std = np.sqrt((hc_std**2 + mg_std**2) / 2)
d_combined = (mg_mean - hc_mean) / pooled_std

# Compare to individual
hc_mad_mean = np.mean(hc_mad)
hc_mad_std = np.std(hc_mad)
mg_mad_mean = np.mean(mg_mad)
mg_mad_std = np.std(mg_mad)
pooled_mad = np.sqrt((hc_mad_std**2 + mg_mad_std**2) / 2)
d_mad = (mg_mad_mean - hc_mad_mean) / pooled_mad

hc_deg_mean = np.mean(hc_deg)
hc_deg_std = np.std(hc_deg)
mg_deg_mean = np.mean(mg_deg)
mg_deg_std = np.std(mg_deg)
pooled_deg = np.sqrt((hc_deg_std**2 + mg_deg_std**2) / 2)
d_deg = (mg_deg_mean - hc_deg_mean) / pooled_deg

print("\n" + "="*80)
print("QUICK EFFECT SIZE COMPARISON")
print("="*80)
print(f"  MAD alone:         d = {d_mad:.4f}")
print(f"  Degradation alone: d = {d_deg:.4f}")
print(f"  Combined (avg):    d = {d_combined:.4f}")

if d_combined >= 0.5:
    print("\n✓✓✓ COMBINED REACHES TARGET d ≥ 0.5!")
elif d_combined > max(d_mad, d_deg):
    print(f"\n✓ SYNERGY: Combined ({d_combined:.2f}) > best individual ({max(d_mad, d_deg):.2f})")
else:
    print(f"\n✗ NO IMPROVEMENT: Combined ({d_combined:.2f}) ≤ best individual ({max(d_mad, d_deg):.2f})")

print("="*80)
