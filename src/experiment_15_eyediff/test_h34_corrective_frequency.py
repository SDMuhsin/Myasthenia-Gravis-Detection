#!/usr/bin/env python3
"""
H34: Corrective Saccade Frequency Asymmetry

Hypothesis: Weak eye (MG-affected) requires more frequent micro-corrections during
attempted fixation, creating asymmetry in corrective saccade rate.

Measurement: Count small saccades (<3°) during "steady-state" periods between
large target jumps. Asymmetry = |freq_L - freq_R|
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

def count_corrective_saccades(eye_pos, target_pos, sample_rate_hz=120):
    """Count small corrective saccades between target jumps"""
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    eye = eye_pos[valid]
    target = target_pos[valid]

    if len(target) < 50:
        return None

    # Identify large target jumps
    target_diff = np.diff(target)
    jump_indices = np.where(np.abs(target_diff) > 5.0)[0] + 1

    if len(jump_indices) < 2:
        return None

    # Between jumps = "steady state" periods
    corrective_count = 0
    total_steady_samples = 0

    for i in range(len(jump_indices) - 1):
        start_idx = jump_indices[i] + int(0.3 * sample_rate_hz)  # Skip 300ms after jump
        end_idx = jump_indices[i+1]

        if end_idx - start_idx < 20:
            continue

        steady_eye = eye[start_idx:end_idx]
        eye_diff = np.abs(np.diff(steady_eye))

        # Corrective saccade = movement >0.5° in single sample
        correctives = np.sum(eye_diff > 0.5)
        corrective_count += correctives
        total_steady_samples += len(steady_eye)

    if total_steady_samples < 50:
        return None

    # Frequency (corrections per second)
    duration_sec = total_steady_samples / sample_rate_hz
    freq_hz = corrective_count / duration_sec

    return {
        'corrective_freq_hz': freq_hz,
        'total_corrections': corrective_count,
        'duration_sec': duration_sec,
    }

print("Analyzing corrective saccade frequency...")
hc_results = []
mg_results = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    LV = df['LV'].values
    RV = df['RV'].values
    TargetV = df['TargetV'].values

    result_L = count_corrective_saccades(LV, TargetV)
    result_R = count_corrective_saccades(RV, TargetV)

    if result_L is None or result_R is None:
        continue

    freq_asym = abs(result_L['corrective_freq_hz'] - result_R['corrective_freq_hz'])

    if seq['label'] == 0:
        hc_results.append(freq_asym)
    else:
        mg_results.append(freq_asym)

hc_asym = np.array(hc_results)
mg_asym = np.array(mg_results)

print(f"Valid samples: HC={len(hc_asym)}, MG={len(mg_asym)}\n")

print("="*80)
print("H34: CORRECTIVE SACCADE FREQUENCY ASYMMETRY")
print("="*80)

print(f"\nHC frequency asymmetry: {np.median(hc_asym):.3f} Hz (mean={np.mean(hc_asym):.3f}, std={np.std(hc_asym):.3f})")
print(f"MG frequency asymmetry: {np.median(mg_asym):.3f} Hz (mean={np.mean(mg_asym):.3f}, std={np.std(mg_asym):.3f})")
print(f"Ratio (MG/HC): {np.median(mg_asym) / np.median(hc_asym):.2f}x")

u_stat, p_val = stats.mannwhitneyu(mg_asym, hc_asym, alternative='greater')
pooled_std = np.sqrt(((len(hc_asym)-1)*np.std(hc_asym, ddof=1)**2 + (len(mg_asym)-1)*np.std(mg_asym, ddof=1)**2) / (len(hc_asym) + len(mg_asym) - 2))
d = (np.mean(mg_asym) - np.mean(hc_asym)) / pooled_std

print(f"\nMann-Whitney U test: p={p_val:.4f} ({'PASS' if p_val < 0.05 else 'FAIL'})")
print(f"Cohen's d: {d:.3f}")

print(f"\nComparison to H30: d=0.606")
print(f"H34: d={d:.3f} ({(d/0.606 - 1)*100:+.1f}%)")

print("\n" + "="*80)
if d > 0.606:
    print(f"✓ PROCEED: d={d:.3f} exceeds H30")
elif d > 0.550:
    print(f"⚠ MARGINAL: d={d:.3f} close to H30")
else:
    print(f"✗ REJECT: d={d:.3f} < H30")
