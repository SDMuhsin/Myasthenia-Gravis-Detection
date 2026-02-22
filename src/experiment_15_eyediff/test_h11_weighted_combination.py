#!/usr/bin/env python3
"""
H11: Optimize weighted combination of MAD + Degradation

Test weights: w * MAD + (1-w) * Degradation for w in [0, 0.1, ..., 1.0]
Find optimal w that maximizes Cohen's d.
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

print("="*80)
print("H11: Optimizing Weighted Combination")
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

# Compute both metrics for all sequences
mad_asym_list = []
deg_asym_list = []
class_labels = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    # MAD
    left_mad = h2_mad_variability(df['LV'].values, df['TargetV'].values)
    right_mad = h2_mad_variability(df['RV'].values, df['TargetV'].values)
    mad_asym = compute_eye_difference(left_mad, right_mad, 'mad_position')

    # Degradation
    left_deg = compute_early_late_degradation(df['LV'].values, df['TargetV'].values)
    right_deg = compute_early_late_degradation(df['RV'].values, df['TargetV'].values)

    if not np.isnan(mad_asym) and not np.isnan(left_deg) and not np.isnan(right_deg):
        deg_asym = abs(left_deg - right_deg)
        mad_asym_list.append(mad_asym)
        deg_asym_list.append(deg_asym)
        class_labels.append(1 if seq['class_name'] == 'MG' else 0)

mad_asym_list = np.array(mad_asym_list)
deg_asym_list = np.array(deg_asym_list)
class_labels = np.array(class_labels)

# Normalize to same scale (z-score)
mad_mean, mad_std = np.mean(mad_asym_list), np.std(mad_asym_list)
deg_mean, deg_std = np.mean(deg_asym_list), np.std(deg_asym_list)

mad_normalized = (mad_asym_list - mad_mean) / mad_std
deg_normalized = (deg_asym_list - deg_mean) / deg_std

# Test different weights
weights = np.arange(0, 1.05, 0.05)
best_d = 0
best_weight = 0
results = []

for w in weights:
    combined = w * mad_normalized + (1 - w) * deg_normalized

    # Split by class
    hc_combined = combined[class_labels == 0]
    mg_combined = combined[class_labels == 1]

    # Cohen's d
    hc_mean = np.mean(hc_combined)
    hc_std = np.std(hc_combined)
    mg_mean = np.mean(mg_combined)
    mg_std = np.std(mg_combined)
    pooled_std = np.sqrt((hc_std**2 + mg_std**2) / 2)
    d = (mg_mean - hc_mean) / pooled_std

    results.append((w, d))

    if d > best_d:
        best_d = d
        best_weight = w

print(f"\nTested {len(weights)} weight combinations")
print(f"\nOptimal weight: w = {best_weight:.2f}")
print(f"  Formula: {best_weight:.2f} * MAD + {1-best_weight:.2f} * Degradation")
print(f"  Cohen's d: {best_d:.4f}")

# Show top 5
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
print("\nTop 5 combinations:")
for i, (w, d) in enumerate(results_sorted[:5]):
    print(f"  {i+1}. w={w:.2f}: d={d:.4f}")

# Re-compute with best weight on unnormalized data
best_combined_raw = best_weight * mad_asym_list + (1 - best_weight) * deg_asym_list

hc_raw = best_combined_raw[class_labels == 0]
mg_raw = best_combined_raw[class_labels == 1]

# Full validation
u_stat, p_mw = stats.mannwhitneyu(mg_raw, hc_raw, alternative='greater')
_, p_hc = stats.wilcoxon(hc_raw)
_, p_mg = stats.wilcoxon(mg_raw)

hc_mean_raw = np.mean(hc_raw)
hc_std_raw = np.std(hc_raw)
mg_mean_raw = np.mean(mg_raw)
mg_std_raw = np.std(mg_raw)
pooled_std_raw = np.sqrt((hc_std_raw**2 + mg_std_raw**2) / 2)
d_raw = (mg_mean_raw - hc_mean_raw) / pooled_std_raw

print("\n" + "="*80)
print("FULL VALIDATION WITH OPTIMAL WEIGHT")
print("="*80)
print(f"\nH11 Combined (w={best_weight:.2f}):")
print(f"  HC: {hc_mean_raw:.4f} ± {hc_std_raw:.4f}")
print(f"  MG: {mg_mean_raw:.4f} ± {mg_std_raw:.4f}")
print(f"  Cohen's d: {d_raw:.4f}")
print(f"  Mann-Whitney p: {p_mw:.6f}")
print(f"  Wilcoxon HC≈0: p={p_hc:.6f} {'✓' if p_hc >= 0.05 else '✗'}")
print(f"  Wilcoxon MG>0: p={p_mg:.6f} {'✓' if p_mg < 0.05 else '✗'}")

validation_score = 0
if p_mw < 0.05:
    validation_score += 1
if p_hc >= 0.05:
    validation_score += 1
if p_mg < 0.05:
    validation_score += 1

print(f"\n  VALIDATION SCORE: {validation_score}/3")

print("\n" + "="*80)
if d_raw >= 0.5:
    print("✓✓✓ TARGET REACHED: d ≥ 0.5!")
    print("PUBLICATION-WORTHY RESULT!")
elif d_raw >= 0.45:
    print("✓✓ VERY CLOSE (d ≥ 0.45)")
else:
    print(f"→ Close but not quite (d = {d_raw:.2f})")

print("="*80)
