#!/usr/bin/env python3
"""
CYCLE 48 - PHASE 4: Error Distribution Shape Analysis
Outside-the-box approach #2: Analyze SHAPE not just location/scale

HYPOTHESIS: MG shows different error distribution shapes (skewness, kurtosis, bimodality)
due to intermittent fatigue creating mixed performance states.

RATIONALE: Standard metrics (mean, MAD) average over distribution. If MG has
bimodal distribution (good saccades + fatigued saccades), shape metrics may detect this.
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
print("CYCLE 48: ERROR DISTRIBUTION SHAPE ANALYSIS")
print("="*80)

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

def compute_shape_metrics(LV, RV, TargetV, sample_rate_hz=120):
    """
    Extract distribution shape metrics (skewness, kurtosis) alongside degradation.
    """
    def process_eye(eye_pos, target_pos):
        valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
        eye = eye_pos[valid]
        target = target_pos[valid]
        if len(eye) < 50:
            return None

        # Detect upward saccades
        target_diff = np.diff(target)
        up_indices = np.where(target_diff > 5.0)[0] + 1

        if len(up_indices) < 20:  # Need at least 20 for shape estimation
            return None

        # Extract per-saccade errors
        saccade_errors = []
        for idx in up_indices:
            start = idx + 20
            end = min(idx + 50, len(eye))
            if end - start < 10:
                continue
            error = np.mean(np.abs(eye[start:end] - target[start:end]))
            saccade_errors.append(error)

        if len(saccade_errors) < 20:
            return None

        errors = np.array(saccade_errors)

        # SHAPE METRICS
        skewness = stats.skew(errors)  # >0 = right tail (outliers on high side)
        kurtosis = stats.kurtosis(errors)  # >0 = heavy tails

        # Robust skewness (Bowley skewness using quartiles)
        q1, q2, q3 = np.percentile(errors, [25, 50, 75])
        robust_skew = ((q3 - q2) - (q2 - q1)) / (q3 - q1) if (q3 - q1) > 0 else 0

        # Bimodality coefficient: (skew^2 + 1) / (kurt + 3)
        # Values > 0.555 suggest bimodal distribution
        bimodality = (skewness**2 + 1) / (kurtosis + 3) if (kurtosis + 3) > 0 else 0

        # DEGRADATION (for comparison)
        n = len(errors)
        third = max(2, n // 3)
        early_err = np.mean(errors[:third])
        late_err = np.mean(errors[-third:])
        deg = late_err - early_err

        return {
            'deg': deg,
            'skewness': skewness,
            'robust_skew': robust_skew,
            'kurtosis': kurtosis,
            'bimodality': bimodality,
            'n_saccades': len(errors),
        }

    met_L = process_eye(LV, TargetV)
    met_R = process_eye(RV, TargetV)

    if met_L is None or met_R is None:
        return None

    return {
        'deg_L': met_L['deg'],
        'deg_R': met_R['deg'],
        'skew_L': met_L['skewness'],
        'skew_R': met_R['skewness'],
        'kurt_L': met_L['kurtosis'],
        'kurt_R': met_R['kurtosis'],
        'bimod_L': met_L['bimodality'],
        'bimod_R': met_R['bimodality'],
    }

hc_metrics = []
mg_metrics = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)
    metrics = compute_shape_metrics(df['LV'].values, df['RV'].values, df['TargetV'].values)
    if metrics is not None:
        if seq['label'] == 0:
            hc_metrics.append(metrics)
        else:
            mg_metrics.append(metrics)

print(f"Valid sequences: HC={len(hc_metrics)}, MG={len(mg_metrics)}\n")

# Extract arrays
hc_deg_L = np.array([m['deg_L'] for m in hc_metrics])
hc_deg_R = np.array([m['deg_R'] for m in hc_metrics])
hc_skew_L = np.array([m['skew_L'] for m in hc_metrics])
hc_skew_R = np.array([m['skew_R'] for m in hc_metrics])
hc_kurt_L = np.array([m['kurt_L'] for m in hc_metrics])
hc_kurt_R = np.array([m['kurt_R'] for m in hc_metrics])

mg_deg_L = np.array([m['deg_L'] for m in mg_metrics])
mg_deg_R = np.array([m['deg_R'] for m in mg_metrics])
mg_skew_L = np.array([m['skew_L'] for m in mg_metrics])
mg_skew_R = np.array([m['skew_R'] for m in mg_metrics])
mg_kurt_L = np.array([m['kurt_L'] for m in mg_metrics])
mg_kurt_R = np.array([m['kurt_R'] for m in mg_metrics])

print("="*80)
print("ANALYSIS 1: DISTRIBUTION SHAPE CHARACTERISTICS")
print("="*80)

print(f"\nSkewness (>0 = right tail):")
print(f"  HC Left:  {np.mean(hc_skew_L):>7.3f} ± {np.std(hc_skew_L):.3f}")
print(f"  HC Right: {np.mean(hc_skew_R):>7.3f} ± {np.std(hc_skew_R):.3f}")
print(f"  MG Left:  {np.mean(mg_skew_L):>7.3f} ± {np.std(mg_skew_L):.3f}")
print(f"  MG Right: {np.mean(mg_skew_R):>7.3f} ± {np.std(mg_skew_R):.3f}")

print(f"\nKurtosis (>0 = heavy tails):")
print(f"  HC Left:  {np.mean(hc_kurt_L):>7.3f} ± {np.std(hc_kurt_L):.3f}")
print(f"  HC Right: {np.mean(hc_kurt_R):>7.3f} ± {np.std(hc_kurt_R):.3f}")
print(f"  MG Left:  {np.mean(mg_kurt_L):>7.3f} ± {np.std(mg_kurt_L):.3f}")
print(f"  MG Right: {np.mean(mg_kurt_R):>7.3f} ± {np.std(mg_kurt_R):.3f}")

print("\n" + "="*80)
print("ANALYSIS 2: SHAPE ASYMMETRY DISCRIMINATION")
print("="*80)

def cohens_d(mg_vals, hc_vals):
    pooled_std = np.sqrt(((len(hc_vals)-1)*np.var(hc_vals, ddof=1) +
                          (len(mg_vals)-1)*np.var(mg_vals, ddof=1)) /
                         (len(hc_vals) + len(mg_vals) - 2))
    return (np.mean(mg_vals) - np.mean(hc_vals)) / pooled_std if pooled_std > 0 else 0

# Asymmetries
hc_deg_asym = np.abs(hc_deg_L - hc_deg_R)
mg_deg_asym = np.abs(mg_deg_L - mg_deg_R)

hc_skew_asym = np.abs(hc_skew_L - hc_skew_R)
mg_skew_asym = np.abs(mg_skew_L - mg_skew_R)

hc_kurt_asym = np.abs(hc_kurt_L - hc_kurt_R)
mg_kurt_asym = np.abs(mg_kurt_L - mg_kurt_R)

d_deg = cohens_d(mg_deg_asym, hc_deg_asym)
d_skew = cohens_d(mg_skew_asym, hc_skew_asym)
d_kurt = cohens_d(mg_kurt_asym, hc_kurt_asym)

print(f"\nDegradation Asymmetry:")
print(f"  HC: {np.mean(hc_deg_asym):.4f} ± {np.std(hc_deg_asym):.4f}°")
print(f"  MG: {np.mean(mg_deg_asym):.4f} ± {np.std(mg_deg_asym):.4f}°")
print(f"  Cohen's d = {d_deg:.3f}")

print(f"\nSkewness Asymmetry:")
print(f"  HC: {np.mean(hc_skew_asym):.4f} ± {np.std(hc_skew_asym):.4f}")
print(f"  MG: {np.mean(mg_skew_asym):.4f} ± {np.std(mg_skew_asym):.4f}")
print(f"  Cohen's d = {d_skew:.3f}")

print(f"\nKurtosis Asymmetry:")
print(f"  HC: {np.mean(hc_kurt_asym):.4f} ± {np.std(hc_kurt_asym):.4f}")
print(f"  MG: {np.mean(mg_kurt_asym):.4f} ± {np.std(mg_kurt_asym):.4f}")
print(f"  Cohen's d = {d_kurt:.3f}")

best_shape_d = max(d_skew, d_kurt)
best_shape = "Skewness" if d_skew > d_kurt else "Kurtosis"

print(f"\nBest shape metric: {best_shape} with d={best_shape_d:.3f}")
print(f"  vs Degradation: {((best_shape_d - d_deg)/d_deg*100):+.1f}%")

if best_shape_d >= 0.45:
    print(f"  ✓ PASS: Shape discrimination strong")
    shape_discriminative = True
elif best_shape_d < 0.30:
    print(f"  ✗ FAIL: Shape discrimination too weak")
    shape_discriminative = False
else:
    print(f"  ~ MARGINAL")
    shape_discriminative = True

print("\n" + "="*80)
print("GO/NO-GO DECISION")
print("="*80)

print(f"\nCriteria:")
print(f"1. Shape discrimination (d≥0.45): {'✓ PASS' if shape_discriminative else '✗ FAIL'} (d={best_shape_d:.3f})")
print(f"2. Improvement over degradation: {'✓ PASS' if best_shape_d > d_deg * 1.05 else '✗ FAIL'}")

if best_shape_d >= 0.45 or best_shape_d > d_deg * 1.05:
    print(f"\nDECISION: GO")
    print(f"  Shape metrics provide discriminative signal")
    print(f"  Best: {best_shape} asymmetry d={best_shape_d:.3f}")
else:
    print(f"\nDECISION: NO-GO")
    print(f"  Shape metrics too weak (d={best_shape_d:.3f})")
    print(f"  Distribution shape doesn't capture MG asymmetry")

print("="*80)
