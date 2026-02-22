#!/usr/bin/env python3
"""
H18: Degradation Acceleration (temporal second derivative)
Measure how fast fatigue is ACCELERATING, not just magnitude of degradation.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_timeseries_data, merge_mg_classes

BASE_DIR = './data'
CLASS_DEFINITIONS = {
    'HC': {'path': 'Healthy control', 'label': 0},
    'MG': {'path': 'Definite MG', 'label': 1},
    'Probable_MG': {'path': 'Probable MG', 'label': 1},
}
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']

print("="*80)
print("H18: Degradation Acceleration")
print("="*80)

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 100)
sequences = merge_mg_classes(raw_sequences)

def compute_degradation_acceleration(eye_pos, target_pos):
    """Fit quadratic to error over time, return acceleration coefficient"""
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    if np.sum(valid) < 50:
        return np.nan

    eye_clean = eye_pos[valid]
    target_clean = target_pos[valid]
    error = np.abs(eye_clean - target_clean)
    n = len(error)

    # Time indices (normalized to [0, 1])
    t = np.linspace(0, 1, n)

    # Fit quadratic: error(t) = a + b*t + c*t^2
    # c is the acceleration coefficient
    try:
        coeffs = np.polyfit(t, error, 2)
        acceleration = coeffs[0]  # Coefficient of t^2
        return acceleration
    except:
        return np.nan

# Compute for all sequences
results = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    left_accel = compute_degradation_acceleration(df['LV'].values, df['TargetV'].values)
    right_accel = compute_degradation_acceleration(df['RV'].values, df['TargetV'].values)

    if not (np.isnan(left_accel) or np.isnan(right_accel)):
        asymmetry = abs(left_accel - right_accel)
        results.append({
            'class': 1 if seq['class_name'] == 'MG' else 0,
            'asymmetry': asymmetry,
            'left_accel': left_accel,
            'right_accel': right_accel
        })

results_df = pd.DataFrame(results)
hc_df = results_df[results_df['class'] == 0]
mg_df = results_df[results_df['class'] == 1]

print(f"\nDataset: {len(hc_df)} HC, {len(mg_df)} MG")

hc_asym = hc_df['asymmetry'].values
mg_asym = mg_df['asymmetry'].values

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

print(f"\nRESULTS:")
print(f"  HC: {hc_mean:.4f} ± {hc_std:.4f}")
print(f"  MG: {mg_mean:.4f} ± {mg_std:.4f}")
print(f"  Cohen's d: {d:.4f}")
print(f"  Validation: {val_score}/3")
print(f"  Mann-Whitney p: {p_mw:.6f} {'✓' if p_mw < 0.05 else '✗'}")
print(f"  HC≈0 p: {p_hc:.6f} {'✓' if p_hc >= 0.05 else '✗'}")
print(f"  MG>0 p: {p_mg:.6f} {'✓' if p_mg < 0.05 else '✗'}")

print(f"\nVERDICT:")
if d >= 0.5 and val_score == 3:
    print(f"  ✓✓✓ SUCCESS! d={d:.3f} ≥ 0.5, validation 3/3")
elif d > 0.41:
    print(f"  ✓ Improvement over H11 (d=0.41 → {d:.3f})")
else:
    print(f"  ✗ No improvement (H11 d=0.41, H18 d={d:.3f})")

print("="*80)
