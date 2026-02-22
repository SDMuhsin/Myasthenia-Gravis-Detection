#!/usr/bin/env python3
"""
BREAKTHROUGH HYPOTHESIS: MG affected eye shows PROGRESSIVE WORSENING
within the 25s recording, while unaffected eye stays stable.

Clinicians may observe: "Left eye performance degrades over the task"
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
print("HYPOTHESIS: Progressive Fatigue Asymmetry")
print("="*80)
print("\nAffected eye degrades over time, unaffected eye stays stable")
print("="*80)

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

def compute_progressive_slope(eye_pos, target_pos, n_bins=5):
    """
    Divide recording into bins and measure error trend.
    Positive slope = worsening performance over time.
    """
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    if np.sum(valid) < 50:
        return np.nan

    eye_clean = eye_pos[valid]
    target_clean = target_pos[valid]
    error = np.abs(eye_clean - target_clean)

    # Divide into bins
    bin_size = len(error) // n_bins
    bin_means = []

    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(error)
        bin_means.append(np.mean(error[start:end]))

    # Fit linear trend
    x = np.arange(n_bins)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, bin_means)

    return slope  # Positive = degrading

# Test on all sequences
hc_left_slopes = []
hc_right_slopes = []
mg_left_slopes = []
mg_right_slopes = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    # Vertical (best direction)
    left_slope = compute_progressive_slope(df['LV'].values, df['TargetV'].values)
    right_slope = compute_progressive_slope(df['RV'].values, df['TargetV'].values)

    if np.isnan(left_slope) or np.isnan(right_slope):
        continue

    if seq['class_name'] == 'HC':
        hc_left_slopes.append(left_slope)
        hc_right_slopes.append(right_slope)
    else:
        mg_left_slopes.append(left_slope)
        mg_right_slopes.append(right_slope)

print(f"\nAnalyzed sequences:")
print(f"  HC: {len(hc_left_slopes)}")
print(f"  MG: {len(mg_left_slopes)}")

# Compute asymmetry in slope
hc_slope_asym = [abs(l - r) for l, r in zip(hc_left_slopes, hc_right_slopes)]
mg_slope_asym = [abs(l - r) for l, r in zip(mg_left_slopes, mg_right_slopes)]

hc_mean = np.mean(hc_slope_asym)
hc_std = np.std(hc_slope_asym)
mg_mean = np.mean(mg_slope_asym)
mg_std = np.std(mg_slope_asym)

pooled_std = np.sqrt((hc_std**2 + mg_std**2) / 2)
cohens_d = (mg_mean - hc_mean) / pooled_std if pooled_std > 0 else 0

print("\n" + "="*80)
print("PROGRESSIVE SLOPE ASYMMETRY")
print("="*80)
print(f"\nAsymmetry in degradation slopes |slope_L - slope_R|:")
print(f"  HC: {hc_mean:.4f} ± {hc_std:.4f}")
print(f"  MG: {mg_mean:.4f} ± {mg_std:.4f}")
print(f"  Cohen's d: {cohens_d:.4f}")
print(f"  MG/HC ratio: {mg_mean/hc_mean:.2f}x")

u_stat, p_val = stats.mannwhitneyu(mg_slope_asym, hc_slope_asym, alternative='greater')
print(f"  Mann-Whitney p: {p_val:.6f}")

# Also check: Which eye degrades more in MG?
mg_left_mean = np.mean(mg_left_slopes)
mg_right_mean = np.mean(mg_right_slopes)

print(f"\nMean slopes (positive = degrading):")
print(f"  MG Left:  {mg_left_mean:.4f}")
print(f"  MG Right: {mg_right_mean:.4f}")

if abs(mg_left_mean) > abs(mg_right_mean):
    print(f"  → Left eye degrades more")
else:
    print(f"  → Right eye degrades more")

# For each MG patient, identify which eye degrades more
mg_steeper_eye_left = 0
mg_steeper_eye_right = 0

for l_slope, r_slope in zip(mg_left_slopes, mg_right_slopes):
    if abs(l_slope) > abs(r_slope):
        mg_steeper_eye_left += 1
    else:
        mg_steeper_eye_right += 1

print(f"\nMG patients with steeper degradation:")
print(f"  Left eye:  {mg_steeper_eye_left} ({100*mg_steeper_eye_left/len(mg_left_slopes):.1f}%)")
print(f"  Right eye: {mg_steeper_eye_right} ({100*mg_steeper_eye_right/len(mg_left_slopes):.1f}%)")

print("\n" + "="*80)
if cohens_d >= 0.5:
    print("✓✓✓ BREAKTHROUGH: d ≥ 0.5!")
    print("MG shows SIGNIFICANTLY greater asymmetry in progressive fatigue!")
elif cohens_d > 0.29:
    print("✓ IMPROVEMENT over previous best (d=0.29)")
else:
    print(f"→ Still weak (d={cohens_d:.2f})")

print("="*80)
