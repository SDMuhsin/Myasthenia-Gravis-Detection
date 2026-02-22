#!/usr/bin/env python3
"""
H14: Test relative asymmetry ratio
asymmetry = |L - R| / (|L| + |R|)
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
print("H14: Relative Asymmetry Ratio")
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

# Compute per-eye metrics
mad_left = []
mad_right = []
deg_left = []
deg_right = []
class_labels = []

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

# Compute H11 combined metric for each eye
combined_left = 0.45 * mad_left + 0.55 * deg_left
combined_right = 0.45 * mad_right + 0.55 * deg_right

# Test different relative asymmetry formulations
formulations = [
    ("Absolute asymmetry (H11)", lambda l, r: np.abs(l - r)),
    ("Relative ratio", lambda l, r: np.abs(l - r) / (np.abs(l) + np.abs(r) + 1e-10)),
    ("Normalized difference", lambda l, r: np.abs(l - r) / np.maximum(np.abs(l), np.abs(r))),
    ("Coefficient of variation", lambda l, r: np.abs(l - r) / (0.5 * (np.abs(l) + np.abs(r)) + 1e-10)),
]

results = []

for name, asym_func in formulations:
    asymmetry = asym_func(combined_left, combined_right)

    hc_asym = asymmetry[class_labels == 0]
    mg_asym = asymmetry[class_labels == 1]

    hc_mean = np.mean(hc_asym)
    hc_std = np.std(hc_asym)
    mg_mean = np.mean(mg_asym)
    mg_std = np.std(mg_asym)
    pooled_std = np.sqrt((hc_std**2 + mg_std**2) / 2)
    d = (mg_mean - hc_mean) / pooled_std

    _, p_mw = stats.mannwhitneyu(mg_asym, hc_asym, alternative='greater')
    _, p_hc = stats.wilcoxon(hc_asym)
    _, p_mg = stats.wilcoxon(mg_asym)

    val_score = (p_mw < 0.05) + (p_hc >= 0.05) + (p_mg < 0.05)

    results.append({
        'name': name,
        'd': d,
        'val_score': val_score,
        'p_hc': p_hc,
        'hc_mean': hc_mean,
        'mg_mean': mg_mean
    })

    print(f"\n{name}:")
    print(f"  HC: {hc_mean:.4f} ± {hc_std:.4f}")
    print(f"  MG: {mg_mean:.4f} ± {mg_std:.4f}")
    print(f"  Cohen's d: {d:.4f}")
    print(f"  Validation: {val_score}/3")
    print(f"  HC≈0 test: p={p_hc:.6f} {'✓' if p_hc >= 0.05 else '✗'}")

best = max(results, key=lambda x: x['d'])

print("\n" + "="*80)
print("BEST FORMULATION:")
print("="*80)
print(f"  {best['name']}")
print(f"  Cohen's d: {best['d']:.4f}")
print(f"  Validation: {best['val_score']}/3")
print(f"  HC≈0 test: p={best['p_hc']:.6f} {'✓' if best['p_hc'] >= 0.05 else '✗'}")

if best['d'] >= 0.5 and best['val_score'] == 3:
    print("\n✓✓✓ SUCCESS!")
elif best['d'] > 0.41:
    print(f"\n✓ IMPROVEMENT over H11 (d=0.41 → d={best['d']:.2f})")
elif best['val_score'] > 2:
    print(f"\n✓ VALIDATION IMPROVEMENT (2/3 → {best['val_score']}/3)")
else:
    print(f"\n✗ No improvement over H11")

print("="*80)
