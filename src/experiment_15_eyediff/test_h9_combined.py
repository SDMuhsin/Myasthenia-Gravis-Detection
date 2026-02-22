#!/usr/bin/env python3
"""Test H9: Combined H+V asymmetry."""

import os
import sys
import numpy as np
import pandas as pd

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

print("Testing H9: Combined H+V MAD asymmetry...")
raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

hc_diffs_mean = []
hc_diffs_max = []
mg_diffs_mean = []
mg_diffs_max = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    # Horizontal
    left_h = h2_mad_variability(df['LH'].values, df['TargetH'].values)
    right_h = h2_mad_variability(df['RH'].values, df['TargetH'].values)
    diff_h = compute_eye_difference(left_h, right_h, 'mad_position')

    # Vertical
    left_v = h2_mad_variability(df['LV'].values, df['TargetV'].values)
    right_v = h2_mad_variability(df['RV'].values, df['TargetV'].values)
    diff_v = compute_eye_difference(left_v, right_v, 'mad_position')

    # Combine
    if not np.isnan(diff_h) and not np.isnan(diff_v):
        combined_mean = (diff_h + diff_v) / 2
        combined_max = max(diff_h, diff_v)

        if seq['class_name'] == 'HC':
            hc_diffs_mean.append(combined_mean)
            hc_diffs_max.append(combined_max)
        else:
            mg_diffs_mean.append(combined_mean)
            mg_diffs_max.append(combined_max)

print(f"Valid combined differences: HC={len(hc_diffs_mean)}, MG={len(mg_diffs_mean)}")

# MEAN combination
hc_mean = np.mean(hc_diffs_mean)
hc_std = np.std(hc_diffs_mean)
mg_mean = np.mean(mg_diffs_mean)
mg_std = np.std(mg_diffs_mean)
pooled_std = np.sqrt((hc_std**2 + mg_std**2) / 2)
cohens_d_mean = (mg_mean - hc_mean) / pooled_std if pooled_std > 0 else 0

print(f"\nH9 (MEAN of H+V):")
print(f"  HC: {hc_mean:.4f} ± {hc_std:.4f}")
print(f"  MG: {mg_mean:.4f} ± {mg_std:.4f}")
print(f"  Cohen's d: {cohens_d_mean:.4f}")

# MAX combination
hc_mean_max = np.mean(hc_diffs_max)
hc_std_max = np.std(hc_diffs_max)
mg_mean_max = np.mean(mg_diffs_max)
mg_std_max = np.std(mg_diffs_max)
pooled_std_max = np.sqrt((hc_std_max**2 + mg_std_max**2) / 2)
cohens_d_max = (mg_mean_max - hc_mean_max) / pooled_std_max if pooled_std_max > 0 else 0

print(f"\nH9 (MAX of H,V):")
print(f"  HC: {hc_mean_max:.4f} ± {hc_std_max:.4f}")
print(f"  MG: {mg_mean_max:.4f} ± {mg_std_max:.4f}")
print(f"  Cohen's d: {cohens_d_max:.4f}")

print("\nComparison:")
print(f"  H only: d=0.14")
print(f"  V only: d=0.29")
print(f"  MEAN(H,V): d={cohens_d_mean:.2f}")
print(f"  MAX(H,V): d={cohens_d_max:.2f}")

if max(cohens_d_mean, cohens_d_max) >= 0.5:
    print("\n✓✓✓ TARGET REACHED: d ≥ 0.5!")
elif max(cohens_d_mean, cohens_d_max) > 0.29:
    print("\n✓ IMPROVEMENT over V-only")
else:
    print("\n✗ NO improvement - reject H9")
