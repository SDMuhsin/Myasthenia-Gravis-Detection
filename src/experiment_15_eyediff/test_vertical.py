#!/usr/bin/env python3
"""Test H2 (MAD) on VERTICAL saccades instead of horizontal."""

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

print("Testing H2 (MAD) on VERTICAL saccades...")
print("Loading all sequences...")

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

hc_diffs = []
mg_diffs = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    # VERTICAL saccades
    left_metrics = h2_mad_variability(df['LV'].values, df['TargetV'].values)
    right_metrics = h2_mad_variability(df['RV'].values, df['TargetV'].values)

    diff = compute_eye_difference(left_metrics, right_metrics, 'mad_position')

    if seq['class_name'] == 'HC':
        hc_diffs.append(diff)
    else:
        mg_diffs.append(diff)

# Remove NaNs
hc_clean = [d for d in hc_diffs if not np.isnan(d)]
mg_clean = [d for d in mg_diffs if not np.isnan(d)]

print(f"\nValid differences:")
print(f"  HC: {len(hc_clean)}/{len(hc_diffs)}")
print(f"  MG: {len(mg_clean)}/{len(mg_diffs)}")

hc_mean = np.mean(hc_clean)
hc_std = np.std(hc_clean)
mg_mean = np.mean(mg_clean)
mg_std = np.std(mg_clean)

pooled_std = np.sqrt((hc_std**2 + mg_std**2) / 2)
cohens_d = (mg_mean - hc_mean) / pooled_std if pooled_std > 0 else 0

print(f"\nH2 MAD on VERTICAL saccades:")
print(f"  HC: {hc_mean:.4f} ± {hc_std:.4f}")
print(f"  MG: {mg_mean:.4f} ± {mg_std:.4f}")
print(f"  Cohen's d: {cohens_d:.4f}")
print(f"  MG/HC ratio: {mg_mean/hc_mean:.2f}x")

print("\nComparison to HORIZONTAL:")
print("  Horizontal d=0.14, MG/HC=1.25x")
print(f"  Vertical d={cohens_d:.2f}, MG/HC={mg_mean/hc_mean:.2f}x")

if abs(cohens_d) > 0.14:
    print("\n✓ Vertical shows BETTER discrimination than horizontal!")
else:
    print("\n✗ Vertical NO better than horizontal")
