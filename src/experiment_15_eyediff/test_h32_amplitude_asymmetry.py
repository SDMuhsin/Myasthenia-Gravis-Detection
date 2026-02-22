#!/usr/bin/env python3
"""
H32 Pre-Analysis: Saccade Amplitude Asymmetry

Hypothesis: MG patients show asymmetric saccade amplitude (one eye undershoots target more than the other)
due to differential muscle weakness.

Analysis: Measure peak-to-peak amplitude of upward saccades for each eye.
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
print(f"Total sequences: {len(sequences)}\n")

def compute_saccade_amplitude_upward(eye_pos, target_pos):
    """
    Compute saccade amplitudes for upward jumps.
    Amplitude = peak_position - starting_position for each saccade.
    """
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    eye = eye_pos[valid]
    target = target_pos[valid]

    if len(target) < 20:
        return None

    target_diff = np.diff(target)
    jump_indices = np.where(target_diff > 5.0)[0] + 1

    amplitudes = []

    for jump_idx in jump_indices:
        if jump_idx >= len(eye) - 10 or jump_idx < 5:
            continue

        # Starting position (before jump)
        start_pos = np.median(eye[max(0, jump_idx-5):jump_idx])

        # Peak position (within 0.5s after jump)
        window_end = min(jump_idx + 60, len(eye))  # 0.5s at 120Hz
        peak_pos = np.max(eye[jump_idx:window_end])

        amplitude = peak_pos - start_pos
        amplitudes.append(amplitude)

    if len(amplitudes) < 3:
        return None

    return {
        'mean_amplitude': np.mean(amplitudes),
        'median_amplitude': np.median(amplitudes),
        'std_amplitude': np.std(amplitudes),
        'amplitudes': amplitudes,
    }

print("Analyzing saccade amplitudes...")
hc_results = []
mg_results = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    LV = df['LV'].values
    RV = df['RV'].values
    TargetV = df['TargetV'].values

    amp_L = compute_saccade_amplitude_upward(LV, TargetV)
    amp_R = compute_saccade_amplitude_upward(RV, TargetV)

    if amp_L is None or amp_R is None:
        continue

    result = {
        'mean_L': amp_L['mean_amplitude'],
        'mean_R': amp_R['mean_amplitude'],
        'median_L': amp_L['median_amplitude'],
        'median_R': amp_R['median_amplitude'],
        'std_L': amp_L['std_amplitude'],
        'std_R': amp_R['std_amplitude'],
    }

    if seq['label'] == 0:
        hc_results.append(result)
    else:
        mg_results.append(result)

hc_mean_L = np.array([r['mean_L'] for r in hc_results])
hc_mean_R = np.array([r['mean_R'] for r in hc_results])
mg_mean_L = np.array([r['mean_L'] for r in mg_results])
mg_mean_R = np.array([r['mean_R'] for r in mg_results])

print(f"Valid samples: HC={len(hc_results)}, MG={len(mg_results)}\n")

print("="*80)
print("SACCADE AMPLITUDE ANALYSIS")
print("="*80)

print("\n1. ABSOLUTE AMPLITUDE (Per-Eye Comparison):")
print(f"   HC Left: {np.median(hc_mean_L):.2f}° ± {np.std(hc_mean_L):.2f}°")
print(f"   HC Right: {np.median(hc_mean_R):.2f}° ± {np.std(hc_mean_R):.2f}°")
print(f"   MG Left: {np.median(mg_mean_L):.2f}° ± {np.std(mg_mean_L):.2f}°")
print(f"   MG Right: {np.median(mg_mean_R):.2f}° ± {np.std(mg_mean_R):.2f}°")

print("\n2. AMPLITUDE ASYMMETRY (|Left - Right|):")
hc_asym = np.abs(hc_mean_L - hc_mean_R)
mg_asym = np.abs(mg_mean_L - mg_mean_R)

print(f"   HC asymmetry: {np.median(hc_asym):.2f}° (mean={np.mean(hc_asym):.2f}°, std={np.std(hc_asym):.2f}°)")
print(f"   MG asymmetry: {np.median(mg_asym):.2f}° (mean={np.mean(mg_asym):.2f}°, std={np.std(mg_asym):.2f}°)")
print(f"   Ratio (MG/HC): {np.median(mg_asym) / np.median(hc_asym):.2f}x")

# Statistical tests
u_stat, p_val = stats.mannwhitneyu(mg_asym, hc_asym, alternative='greater')
pooled_std = np.sqrt(((len(hc_asym)-1)*np.std(hc_asym, ddof=1)**2 + (len(mg_asym)-1)*np.std(mg_asym, ddof=1)**2) / (len(hc_asym) + len(mg_asym) - 2))
d = (np.mean(mg_asym) - np.mean(hc_asym)) / pooled_std

print(f"\n   Mann-Whitney U test: p={p_val:.4f} ({'PASS' if p_val < 0.05 else 'FAIL'})")
print(f"   Cohen's d: {d:.3f}")

# HC≈0 test
_, p_hc = stats.wilcoxon(hc_asym, alternative='greater')
print(f"   HC≈0 test (Wilcoxon): p={p_hc:.4f} ({'FAIL' if p_hc < 0.05 else 'PASS'})")

print("\n3. COMPARISON TO H30:")
print(f"   H30 baseline: d=0.606")
print(f"   H32 amplitude: d={d:.3f} ({(d/0.606 - 1)*100:+.1f}%)")

print("\n4. AMPLITUDE STD ASYMMETRY (Variability):")
hc_std_asym = np.abs(np.array([r['std_L'] for r in hc_results]) - np.array([r['std_R'] for r in hc_results]))
mg_std_asym = np.abs(np.array([r['std_L'] for r in mg_results]) - np.array([r['std_R'] for r in mg_results]))

print(f"   HC std_asym: {np.median(hc_std_asym):.2f}°")
print(f"   MG std_asym: {np.median(mg_std_asym):.2f}°")
print(f"   Ratio (MG/HC): {np.median(mg_std_asym) / np.median(hc_std_asym):.2f}x")

u_std, p_std = stats.mannwhitneyu(mg_std_asym, hc_std_asym, alternative='greater')
pooled_std_std = np.sqrt(((len(hc_std_asym)-1)*np.std(hc_std_asym, ddof=1)**2 + (len(mg_std_asym)-1)*np.std(mg_std_asym, ddof=1)**2) / (len(hc_std_asym) + len(mg_std_asym) - 2))
d_std = (np.mean(mg_std_asym) - np.mean(hc_std_asym)) / pooled_std_std

print(f"   Mann-Whitney U test: p={p_std:.4f}")
print(f"   Cohen's d: {d_std:.3f}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

if d > 0.606:
    print(f"✓ PROCEED: Amplitude asymmetry shows d={d:.3f} > H30")
    print("  Amplitude-based metric warrants full implementation")
elif d > 0.550:
    print(f"⚠ MARGINAL: Amplitude asymmetry shows d={d:.3f}")
    print(f"  Close to H30 but not exceeding - consider combination approach")
else:
    print(f"✗ REJECT: Amplitude asymmetry shows d={d:.3f} < H30")
    print(f"  Amplitude does not improve over existing metrics")
