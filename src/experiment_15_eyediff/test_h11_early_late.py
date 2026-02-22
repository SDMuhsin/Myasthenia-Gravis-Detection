#!/usr/bin/env python3
"""
H11: Early vs Late Performance Degradation Asymmetry

Compare first 20% vs last 20% of sequence.
Affected eye should show larger degradation (late - early).
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
print("H11: Early-Late Degradation Asymmetry")
print("="*80)

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

def compute_early_late_degradation(eye_pos, target_pos, early_pct=0.2, late_pct=0.2):
    """
    Compare performance in first 20% vs last 20% of sequence.
    Returns degradation = late_error - early_error (positive = worse over time).
    """
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    if np.sum(valid) < 50:
        return np.nan

    eye_clean = eye_pos[valid]
    target_clean = target_pos[valid]
    error = np.abs(eye_clean - target_clean)

    n = len(error)
    early_n = int(n * early_pct)
    late_n = int(n * late_pct)

    early_segment = error[:early_n]
    late_segment = error[-late_n:]

    early_mean = np.mean(early_segment)
    late_mean = np.mean(late_segment)

    degradation = late_mean - early_mean  # Positive = getting worse

    return degradation

# Compute for all sequences
hc_left_deg = []
hc_right_deg = []
mg_left_deg = []
mg_right_deg = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    # Vertical
    left_deg = compute_early_late_degradation(df['LV'].values, df['TargetV'].values)
    right_deg = compute_early_late_degradation(df['RV'].values, df['TargetV'].values)

    if np.isnan(left_deg) or np.isnan(right_deg):
        continue

    if seq['class_name'] == 'HC':
        hc_left_deg.append(left_deg)
        hc_right_deg.append(right_deg)
    else:
        mg_left_deg.append(left_deg)
        mg_right_deg.append(right_deg)

# Compute asymmetry in degradation
hc_deg_asym = [abs(l - r) for l, r in zip(hc_left_deg, hc_right_deg)]
mg_deg_asym = [abs(l - r) for l, r in zip(mg_left_deg, mg_right_deg)]

hc_mean = np.mean(hc_deg_asym)
hc_std = np.std(hc_deg_asym)
mg_mean = np.mean(mg_deg_asym)
mg_std = np.std(mg_deg_asym)

pooled_std = np.sqrt((hc_std**2 + mg_std**2) / 2)
cohens_d = (mg_mean - hc_mean) / pooled_std if pooled_std > 0 else 0

print(f"\nAnalyzed: HC={len(hc_deg_asym)}, MG={len(mg_deg_asym)}")

print("\n" + "="*80)
print("H11: EARLY-LATE DEGRADATION ASYMMETRY")
print("="*80)
print(f"\n|Degradation_L - Degradation_R| where degradation = late_error - early_error:")
print(f"  HC: {hc_mean:.4f} ± {hc_std:.4f}")
print(f"  MG: {mg_mean:.4f} ± {mg_std:.4f}")
print(f"  Cohen's d: {cohens_d:.4f}")
print(f"  MG/HC ratio: {mg_mean/hc_mean:.2f}x")

u_stat, p_val = stats.mannwhitneyu(mg_deg_asym, hc_deg_asym, alternative='greater')
print(f"  Mann-Whitney p: {p_val:.6f}")

# Wilcoxon tests
_, hc_wilcox_p = stats.wilcoxon(hc_deg_asym)
_, mg_wilcoxon_p = stats.wilcoxon(mg_deg_asym)

print(f"\n  Wilcoxon HC≈0: p={hc_wilcox_p:.6f} {'✓' if hc_wilcox_p >= 0.05 else '✗'}")
print(f"  Wilcoxon MG>0: p={mg_wilcoxon_p:.6f} {'✓' if mg_wilcoxon_p < 0.05 else '✗'}")

validation_score = 0
if p_val < 0.05:
    validation_score += 1
    print("\n  Test 1 (MG>HC): ✓ PASS")
else:
    print("\n  Test 1 (MG>HC): ✗ FAIL")

if hc_wilcox_p >= 0.05:
    validation_score += 1
    print("  Test 2 (HC≈0): ✓ PASS")
else:
    print("  Test 2 (HC≈0): ✗ FAIL")

if mg_wilcoxon_p < 0.05:
    validation_score += 1
    print("  Test 3 (MG>0): ✓ PASS")
else:
    print("  Test 3 (MG>0): ✗ FAIL")

print(f"\n  VALIDATION SCORE: {validation_score}/3")

print("\n" + "="*80)
if cohens_d >= 0.5:
    print("✓✓✓ TARGET REACHED: d ≥ 0.5!")
    print("PUBLICATION-WORTHY EFFECT SIZE!")
elif cohens_d > 0.40:
    print("✓✓ STRONG IMPROVEMENT (d > 0.40)")
elif cohens_d > 0.29:
    print("✓ Improvement over previous best (d=0.29)")
else:
    print(f"→ Weak (d={cohens_d:.2f})")

print("="*80)

# Also check which eye degrades more
mg_left_mean_deg = np.mean(mg_left_deg)
mg_right_mean_deg = np.mean(mg_right_deg)

print(f"\nMean degradation (late - early) in MG:")
print(f"  Left:  {mg_left_mean_deg:.4f}°")
print(f"  Right: {mg_right_mean_deg:.4f}°")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print(f"This metric measures how much DIFFERENTLY the two eyes")
print(f"degrade from beginning to end of the 25s recording.")
print(f"MG shows {mg_mean/hc_mean:.2f}x more asymmetric degradation than HC.")
print("="*80)
