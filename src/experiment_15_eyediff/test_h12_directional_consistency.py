#!/usr/bin/env python3
"""
H12 Empirical Pre-Check: Directional Consistency of Degradation

Test if MG shows consistent degradation direction across temporal windows
while HC shows random/balanced directions.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_timeseries_data, merge_mg_classes

# Configuration
BASE_DIR = './data'
CLASS_DEFINITIONS = {
    'HC': {'path': 'Healthy control', 'label': 0},
    'MG': {'path': 'Definite MG', 'label': 1},
    'Probable_MG': {'path': 'Probable MG', 'label': 1},
}
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']

print("="*80)
print("H12: Directional Consistency Pre-Check")
print("="*80)

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

def compute_windowed_degradation_consistency(eye_pos, target_pos, n_windows=5):
    """
    Compute directional consistency of degradation across temporal windows.

    Returns:
        consistency: Value in [0, 1], where 1 = perfect consistency
    """
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    if np.sum(valid) < 100:  # Need minimum data
        return np.nan

    eye_clean = eye_pos[valid]
    target_clean = target_pos[valid]
    error = np.abs(eye_clean - target_clean)

    n = len(error)
    window_size = n // n_windows

    if window_size < 20:  # Need minimum samples per window
        return np.nan

    degradations = []

    for w in range(n_windows):
        start_idx = w * window_size
        end_idx = (w + 1) * window_size if w < n_windows - 1 else n

        window_error = error[start_idx:end_idx]

        # Split window into early/late halves
        half = len(window_error) // 2
        if half < 5:
            continue

        early_error = np.mean(window_error[:half])
        late_error = np.mean(window_error[half:])

        degradation = late_error - early_error
        degradations.append(degradation)

    if len(degradations) < 3:
        return np.nan

    return np.array(degradations)

# Analyze directional consistency
hc_consistencies = []
mg_consistencies = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    # Vertical saccades
    left_deg = compute_windowed_degradation_consistency(df['LV'].values, df['TargetV'].values)
    right_deg = compute_windowed_degradation_consistency(df['RV'].values, df['TargetV'].values)

    if not (isinstance(left_deg, float) and np.isnan(left_deg)):
        # Compute directional consistency
        signs = np.sign(left_deg - right_deg)
        consistency = np.abs(np.sum(signs)) / len(signs)

        if seq['class_name'] == 'HC':
            hc_consistencies.append(consistency)
        else:
            mg_consistencies.append(consistency)

hc_consistencies = np.array(hc_consistencies)
mg_consistencies = np.array(mg_consistencies)

print(f"\nAnalyzed sequences:")
print(f"  HC: {len(hc_consistencies)}")
print(f"  MG: {len(mg_consistencies)}")

print(f"\nDirectional Consistency:")
print(f"  HC: {np.mean(hc_consistencies):.4f} ± {np.std(hc_consistencies):.4f}")
print(f"  MG: {np.mean(mg_consistencies):.4f} ± {np.std(mg_consistencies):.4f}")
print(f"  Ratio MG/HC: {np.mean(mg_consistencies) / np.mean(hc_consistencies):.2f}x")

# Statistical tests
u_stat, p_mw = stats.mannwhitneyu(mg_consistencies, hc_consistencies, alternative='greater')
_, p_hc = stats.wilcoxon(hc_consistencies) if len(hc_consistencies) > 0 else (np.nan, np.nan)
_, p_mg = stats.wilcoxon(mg_consistencies) if len(mg_consistencies) > 0 else (np.nan, np.nan)

# Cohen's d
hc_mean = np.mean(hc_consistencies)
hc_std = np.std(hc_consistencies)
mg_mean = np.mean(mg_consistencies)
mg_std = np.std(mg_consistencies)
pooled_std = np.sqrt((hc_std**2 + mg_std**2) / 2)
d = (mg_mean - hc_mean) / pooled_std

print("\n" + "="*80)
print("VALIDATION")
print("="*80)
print(f"  Mann-Whitney MG>HC: p={p_mw:.6f} {'✓' if p_mw < 0.05 else '✗'}")
print(f"  Wilcoxon HC≈0: p={p_hc:.6f} {'✓' if p_hc >= 0.05 else '✗'}")
print(f"  Wilcoxon MG>0: p={p_mg:.6f} {'✓' if p_mg < 0.05 else '✗'}")
print(f"  Cohen's d: {d:.4f}")

validation_score = 0
if p_mw < 0.05:
    validation_score += 1
if p_hc >= 0.05:
    validation_score += 1
if p_mg < 0.05:
    validation_score += 1

print(f"\n  VALIDATION SCORE: {validation_score}/3")

print("\n" + "="*80)
print("DECISION:")
if d >= 0.5 and validation_score == 3:
    print("✓✓✓ SUCCESS: Meets all criteria (d≥0.5 AND 3/3 validation)")
elif d >= 0.5:
    print(f"✓✓ High effect size (d={d:.2f}) but validation {validation_score}/3")
elif validation_score == 3:
    print(f"✓ Perfect validation but d={d:.2f} < 0.50")
else:
    print(f"✗ Does not meet criteria: d={d:.2f}, validation {validation_score}/3")

if d > 0.41:
    print(f"→ IMPROVEMENT over H11 (d=0.41)!")
else:
    print(f"→ No improvement over H11 (d=0.41)")

print("="*80)
