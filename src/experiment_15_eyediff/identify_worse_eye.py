#!/usr/bin/env python3
"""
BREAKTHROUGH INSIGHT: Don't measure |L-R| asymmetry.
Identify WHICH eye (L or R) performs worse, and see if MG patients have a clear worse eye.

Clinicians ask: "Which eye is affected?" not "How different are they?"
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
print("FRESH APPROACH: Identify WHICH Eye Performs Worse")
print("="*80)

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

def compute_tracking_error(eye_pos, target_pos):
    """Mean absolute tracking error."""
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    if np.sum(valid) < 10:
        return np.nan
    return np.mean(np.abs(eye_pos[valid] - target_pos[valid]))

# For each sequence, determine which eye is worse
hc_worse_eye_clarity = []
mg_worse_eye_clarity = []

hc_left_worse_count = 0
hc_right_worse_count = 0
mg_left_worse_count = 0
mg_right_worse_count = 0

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    # Vertical (best direction from prior analysis)
    left_error = compute_tracking_error(df['LV'].values, df['TargetV'].values)
    right_error = compute_tracking_error(df['RV'].values, df['TargetV'].values)

    if np.isnan(left_error) or np.isnan(right_error):
        continue

    # Which eye has LARGER error (worse performance)?
    worse_eye_difference = np.abs(left_error - right_error)

    # Normalize by average error (relative difference)
    avg_error = (left_error + right_error) / 2
    if avg_error > 0:
        clarity = worse_eye_difference / avg_error  # How CLEARLY worse is the worse eye?
    else:
        clarity = 0

    if seq['class_name'] == 'HC':
        hc_worse_eye_clarity.append(clarity)
        if left_error > right_error:
            hc_left_worse_count += 1
        else:
            hc_right_worse_count += 1
    else:
        mg_worse_eye_clarity.append(clarity)
        if left_error > right_error:
            mg_left_worse_count += 1
        else:
            mg_right_worse_count += 1

print("\n" + "="*80)
print("WORSE EYE IDENTIFICATION CLARITY")
print("="*80)

hc_mean_clarity = np.mean(hc_worse_eye_clarity)
hc_std_clarity = np.std(hc_worse_eye_clarity)
mg_mean_clarity = np.mean(mg_worse_eye_clarity)
mg_std_clarity = np.std(mg_worse_eye_clarity)

print(f"\nWorse Eye Clarity (|L_error - R_error| / avg_error):")
print(f"  HC: {hc_mean_clarity:.4f} ± {hc_std_clarity:.4f}")
print(f"  MG: {mg_mean_clarity:.4f} ± {mg_std_clarity:.4f}")

# Cohen's d
pooled_std = np.sqrt((hc_std_clarity**2 + mg_std_clarity**2) / 2)
cohens_d = (mg_mean_clarity - hc_mean_clarity) / pooled_std if pooled_std > 0 else 0

print(f"  Cohen's d: {cohens_d:.4f}")
print(f"  MG/HC ratio: {mg_mean_clarity/hc_mean_clarity:.2f}x")

# Mann-Whitney test
u_stat, p_val = stats.mannwhitneyu(mg_worse_eye_clarity, hc_worse_eye_clarity, alternative='greater')
print(f"  Mann-Whitney p-value: {p_val:.6f}")

print("\n" + "="*80)
print("WORSE EYE DISTRIBUTION")
print("="*80)

hc_total = hc_left_worse_count + hc_right_worse_count
mg_total = mg_left_worse_count + mg_right_worse_count

print(f"\nHC (n={hc_total}):")
print(f"  Left worse:  {hc_left_worse_count} ({100*hc_left_worse_count/hc_total:.1f}%)")
print(f"  Right worse: {hc_right_worse_count} ({100*hc_right_worse_count/hc_total:.1f}%)")
print(f"  Balance: {abs(hc_left_worse_count - hc_right_worse_count)/hc_total:.1%} from 50/50")

print(f"\nMG (n={mg_total}):")
print(f"  Left worse:  {mg_left_worse_count} ({100*mg_left_worse_count/mg_total:.1f}%)")
print(f"  Right worse: {mg_right_worse_count} ({100*mg_right_worse_count/mg_total:.1f}%)")
print(f"  Balance: {abs(mg_left_worse_count - mg_right_worse_count)/mg_total:.1%} from 50/50")

print("\n" + "="*80)
print("HYPOTHESIS: MG patients have CLEARER worse eye identification")
print("="*80)

if cohens_d >= 0.5:
    print("\n✓✓✓ BREAKTHROUGH: d ≥ 0.5!")
    print("MG patients show SIGNIFICANTLY clearer identification of worse-performing eye")
elif cohens_d > 0.29:
    print("\n✓ IMPROVEMENT over previous best (d=0.29)")
elif cohens_d > 0:
    print(f"\n→ Positive direction but weak effect (d={cohens_d:.2f})")
else:
    print("\n✗ Wrong direction or no effect")

print("\n" + "="*80)
print("INSIGHT: The MAGNITUDE of performance difference between eyes")
print("may be the key discriminator, not which specific eye (L vs R)")
print("="*80)
