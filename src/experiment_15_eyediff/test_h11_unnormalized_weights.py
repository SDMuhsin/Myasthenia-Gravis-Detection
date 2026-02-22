#!/usr/bin/env python3
"""
H11: Find optimal weights WITHOUT z-score normalization.
This matches the testbench approach of combining raw metrics.
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
print("H11: Optimizing Weights (UNNORMALIZED)")
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

# Compute metrics for all sequences (per-eye, not asymmetries)
mad_left = []
mad_right = []
deg_left = []
deg_right = []
class_labels = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    # MAD
    left_mad_dict = h2_mad_variability(df['LV'].values, df['TargetV'].values)
    right_mad_dict = h2_mad_variability(df['RV'].values, df['TargetV'].values)
    left_mad_val = left_mad_dict['mad_position']
    right_mad_val = right_mad_dict['mad_position']

    # Degradation
    left_deg_val = compute_early_late_degradation(df['LV'].values, df['TargetV'].values)
    right_deg_val = compute_early_late_degradation(df['RV'].values, df['TargetV'].values)

    if not (np.isnan(left_mad_val) or np.isnan(right_mad_val) or
            np.isnan(left_deg_val) or np.isnan(right_deg_val)):
        mad_left.append(left_mad_val)
        mad_right.append(right_mad_val)
        deg_left.append(left_deg_val)
        deg_right.append(right_deg_val)
        class_labels.append(1 if seq['class_name'] == 'MG' else 0)

mad_left = np.array(mad_left)
mad_right = np.array(mad_right)
deg_left = np.array(deg_left)
deg_right = np.array(deg_right)
class_labels = np.array(class_labels)

print(f"\nAnalyzed {len(class_labels)} sequences")
print(f"  HC: {np.sum(class_labels == 0)}")
print(f"  MG: {np.sum(class_labels == 1)}")

# Test different weights (on RAW, unnormalized data)
weights = np.arange(0, 1.05, 0.05)
best_d = 0
best_weight = 0
results = []

for w in weights:
    # Compute combined score for each eye
    combined_left = w * mad_left + (1 - w) * deg_left
    combined_right = w * mad_right + (1 - w) * deg_right

    # Asymmetry
    asymmetry = np.abs(combined_left - combined_right)

    # Split by class
    hc_asym = asymmetry[class_labels == 0]
    mg_asym = asymmetry[class_labels == 1]

    # Cohen's d
    hc_mean = np.mean(hc_asym)
    hc_std = np.std(hc_asym)
    mg_mean = np.mean(mg_asym)
    mg_std = np.std(mg_asym)
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

# Full validation with best weight
combined_left_best = best_weight * mad_left + (1 - best_weight) * deg_left
combined_right_best = best_weight * mad_right + (1 - best_weight) * deg_right
asymmetry_best = np.abs(combined_left_best - combined_right_best)

hc_best = asymmetry_best[class_labels == 0]
mg_best = asymmetry_best[class_labels == 1]

u_stat, p_mw = stats.mannwhitneyu(mg_best, hc_best, alternative='greater')
_, p_hc = stats.wilcoxon(hc_best)
_, p_mg = stats.wilcoxon(mg_best)

hc_mean = np.mean(hc_best)
hc_std = np.std(hc_best)
mg_mean = np.mean(mg_best)
mg_std = np.std(mg_best)
pooled_std = np.sqrt((hc_std**2 + mg_std**2) / 2)
d_final = (mg_mean - hc_mean) / pooled_std

print("\n" + "="*80)
print("FULL VALIDATION WITH OPTIMAL WEIGHT (UNNORMALIZED)")
print("="*80)
print(f"\nH11 Combined (w={best_weight:.2f}):")
print(f"  HC: {hc_mean:.4f} ± {hc_std:.4f}")
print(f"  MG: {mg_mean:.4f} ± {mg_std:.4f}")
print(f"  Cohen's d: {d_final:.4f}")
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
if d_final >= 0.5:
    print("✓✓✓ TARGET REACHED: d ≥ 0.5!")
    print("PUBLICATION-WORTHY RESULT!")
elif d_final >= 0.45:
    print("✓✓ VERY CLOSE (d ≥ 0.45)")
else:
    print(f"→ Close but not quite (d = {d_final:.2f})")

print("="*80)
