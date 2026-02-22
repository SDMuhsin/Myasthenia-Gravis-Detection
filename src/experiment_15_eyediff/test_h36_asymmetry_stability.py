#!/usr/bin/env python3
"""
H36: Temporal Stability of Asymmetry

CRITICAL INSIGHT: HC has ~0.5° baseline asymmetry that is STABLE over time.
MG has asymmetry that CHANGES due to progressive fatigue.

Hypothesis: Measure VARIABILITY of asymmetry over time (std of |L-R| across windows),
not absolute asymmetry.
- HC: Stable baseline → low variability
- MG: Fluctuating fatigue → high variability

This CANCELS HC baseline by measuring its stability, not magnitude.
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

print("Loading sequences...")
raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

def compute_asymmetry_stability(eye_L, eye_R, target, n_windows=5):
    """
    Divide upward saccade period into windows, compute |L-R| error in each,
    measure std of these asymmetries.
    """
    valid = ~(np.isnan(eye_L) | np.isnan(eye_R) | np.isnan(target))
    eye_L_clean = eye_L[valid]
    eye_R_clean = eye_R[valid]
    target_clean = target[valid]

    if len(target_clean) < 100:
        return None

    # Extract upward saccade period
    target_diff = np.diff(target_clean)
    up_indices = np.where(target_diff > 5.0)[0] + 1
    mask = np.zeros(len(eye_L_clean), dtype=bool)
    for idx in up_indices:
        mask[idx:min(idx+50, len(eye_L_clean))] = True

    eye_L_up = eye_L_clean[mask]
    eye_R_up = eye_R_clean[mask]
    target_up = target_clean[mask]

    if len(eye_L_up) < 50:
        return None

    # Divide into windows
    window_size = len(eye_L_up) // n_windows
    if window_size < 10:
        return None

    asymmetries = []
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size if i < n_windows - 1 else len(eye_L_up)

        error_L = eye_L_up[start_idx:end_idx] - target_up[start_idx:end_idx]
        error_R = eye_R_up[start_idx:end_idx] - target_up[start_idx:end_idx]

        # MAD asymmetry in this window
        mad_L = np.median(np.abs(error_L - np.median(error_L)))
        mad_R = np.median(np.abs(error_R - np.median(error_R)))
        cv_asym = abs(mad_L - mad_R) / ((mad_L + mad_R) / 2) if (mad_L + mad_R) > 0 else 0

        asymmetries.append(cv_asym)

    # Stability = std of asymmetry across windows
    # Lower std = stable baseline (HC)
    # Higher std = fluctuating asymmetry (MG fatigue)
    stability = np.std(asymmetries)

    return {
        'asymmetry_std': stability,
        'asymmetry_mean': np.mean(asymmetries),
        'asymmetries': asymmetries,
    }

print("Analyzing asymmetry stability...")
hc_results = []
mg_results = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    LV = df['LV'].values
    RV = df['RV'].values
    TargetV = df['TargetV'].values

    result = compute_asymmetry_stability(LV, RV, TargetV, n_windows=5)

    if result is None:
        continue

    if seq['label'] == 0:
        hc_results.append(result)
    else:
        mg_results.append(result)

hc_std = np.array([r['asymmetry_std'] for r in hc_results])
mg_std = np.array([r['asymmetry_std'] for r in mg_results])
hc_mean = np.array([r['asymmetry_mean'] for r in hc_results])
mg_mean = np.array([r['asymmetry_mean'] for r in mg_results])

print(f"Valid samples: HC={len(hc_std)}, MG={len(mg_std)}\n")

print("="*80)
print("H36: TEMPORAL STABILITY OF ASYMMETRY")
print("="*80)

print("\n1. ASYMMETRY VARIABILITY (std across 5 windows):")
print(f"   HC std(asymmetry): {np.median(hc_std):.4f} (mean={np.mean(hc_std):.4f})")
print(f"   MG std(asymmetry): {np.median(mg_std):.4f} (mean={np.mean(mg_std):.4f})")
print(f"   Ratio (MG/HC): {np.median(mg_std) / np.median(hc_std):.2f}x")

u_std, p_std = stats.mannwhitneyu(mg_std, hc_std, alternative='greater')
pooled_std_std = np.sqrt(((len(hc_std)-1)*np.std(hc_std, ddof=1)**2 + (len(mg_std)-1)*np.std(mg_std, ddof=1)**2) / (len(hc_std) + len(mg_std) - 2))
d_std = (np.mean(mg_std) - np.mean(hc_std)) / pooled_std_std

print(f"   Mann-Whitney U: p={p_std:.4f} ({'PASS' if p_std < 0.05 else 'FAIL'})")
print(f"   Cohen's d: {d_std:.3f}")

print("\n2. BASELINE ASYMMETRY MAGNITUDE (mean across windows):")
print(f"   HC mean(asymmetry): {np.median(hc_mean):.4f}")
print(f"   MG mean(asymmetry): {np.median(mg_mean):.4f}")
print(f"   Ratio: {np.median(mg_mean) / np.median(hc_mean):.2f}x")

u_mean, p_mean = stats.mannwhitneyu(mg_mean, hc_mean, alternative='greater')
pooled_std_mean = np.sqrt(((len(hc_mean)-1)*np.std(hc_mean, ddof=1)**2 + (len(mg_mean)-1)*np.std(mg_mean, ddof=1)**2) / (len(hc_mean) + len(mg_mean) - 2))
d_mean = (np.mean(mg_mean) - np.mean(hc_mean)) / pooled_std_mean

print(f"   Cohen's d: {d_mean:.3f} (for reference)")

print("\n3. COMPARISON:")
print(f"   H30 (absolute asymmetry): d=0.606")
print(f"   H36 (asymmetry stability): d={d_std:.3f} ({(d_std/0.606 - 1)*100:+.1f}%)")

print("\n" + "="*80)
if d_std > 0.606:
    print(f"✓ PROCEED: Stability metric d={d_std:.3f} > H30")
    print("  Temporal stability successfully cancels HC baseline asymmetry")
else:
    print(f"✗ REJECT: Stability metric d={d_std:.3f} ≤ H30")
    print("  Temporal approach does not improve discrimination")
