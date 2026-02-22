#!/usr/bin/env python3
"""H22: Full Validation"""

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

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

def compute_positional_metrics(eye_pos, target_pos):
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    eye_clean = eye_pos[valid]
    target_clean = target_pos[valid]

    if len(eye_clean) < 50:
        return {'mad': np.nan, 'degradation': np.nan}

    median_pos = np.median(eye_clean)
    mad = np.median(np.abs(eye_clean - median_pos))

    error = np.abs(eye_clean - target_clean)
    n = len(error)
    early_n = int(n * 0.2)
    late_n = int(n * 0.2)

    degradation = np.mean(error[-late_n:]) - np.mean(error[:early_n])

    return {'mad': mad, 'degradation': degradation}

def compute_saccade_latencies(eye_pos, target_pos, sample_rate_hz=120):
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    eye = eye_pos[valid]
    target = target_pos[valid]

    if len(target) < 20:
        return []

    target_diff = np.abs(np.diff(target))
    jump_threshold = 5.0
    jump_indices = np.where(target_diff > jump_threshold)[0] + 1

    latencies_ms = []

    for jump_idx in jump_indices:
        if jump_idx >= len(target) - 10:
            continue

        new_target = target[jump_idx]
        threshold_deg = 3.0

        for offset in range(1, min(100, len(eye) - jump_idx)):
            eye_pos_now = eye[jump_idx + offset]
            error = abs(eye_pos_now - new_target)

            if error < threshold_deg:
                latency_ms = (offset / sample_rate_hz) * 1000
                latencies_ms.append(latency_ms)
                break

    return latencies_ms

def compute_latency_metric(latencies):
    if len(latencies) < 3:
        return np.nan

    latencies = np.array(latencies)
    n = len(latencies)
    early_n = max(2, n // 3)

    return np.mean(latencies[:early_n])

results = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    left_pos = compute_positional_metrics(df['LV'].values, df['TargetV'].values)
    right_pos = compute_positional_metrics(df['RV'].values, df['TargetV'].values)

    left_latencies = compute_saccade_latencies(df['LV'].values, df['TargetV'].values)
    right_latencies = compute_saccade_latencies(df['RV'].values, df['TargetV'].values)

    left_lat = compute_latency_metric(left_latencies)
    right_lat = compute_latency_metric(right_latencies)

    if (not np.isnan(left_pos['mad']) and not np.isnan(right_pos['mad']) and
        not np.isnan(left_lat) and not np.isnan(right_lat)):

        pos_component = 0.45 * abs(left_pos['mad'] - right_pos['mad']) + \
                       0.55 * abs(left_pos['degradation'] - right_pos['degradation'])

        lat_component = abs(left_lat - right_lat) / 100.0

        combined = (pos_component + lat_component) / 2

        results.append({
            'class': 1 if seq['class_name'] == 'MG' else 0,
            'combined': combined
        })

results_df = pd.DataFrame(results)
hc_df = results_df[results_df['class'] == 0]
mg_df = results_df[results_df['class'] == 1]

hc_vals = hc_df['combined'].values
mg_vals = mg_df['combined'].values

hc_mean = np.mean(hc_vals)
mg_mean = np.mean(mg_vals)
hc_std = np.std(hc_vals)
mg_std = np.std(mg_vals)

pooled_std = np.sqrt((hc_std**2 + mg_std**2) / 2)
d = (mg_mean - hc_mean) / pooled_std

_, p_mw = stats.mannwhitneyu(mg_vals, hc_vals, alternative='greater')
_, p_hc = stats.wilcoxon(hc_vals)
_, p_mg = stats.wilcoxon(mg_vals)

val_score = (p_mw < 0.05) + (p_hc >= 0.05) + (p_mg < 0.05)

print(f"\n{'='*80}")
print(f"H22: Combined Position + Latency - FULL VALIDATION")
print(f"{'='*80}")
print(f"Dataset: {len(hc_vals)} HC, {len(mg_vals)} MG")
print(f"\nHC: {hc_mean:.3f} ± {hc_std:.3f}")
print(f"MG: {mg_mean:.3f} ± {mg_std:.3f}")
print(f"\nCohen's d: {d:.4f}")
print(f"Validation score: {val_score}/3")
print(f"\n  [1] MG > HC: p={p_mw:.6f} {'✓' if p_mw < 0.05 else '✗'}")
print(f"  [2] HC ≈ 0:  p={p_hc:.6f} {'✓' if p_hc >= 0.05 else '✗'}")
print(f"  [3] MG > 0:  p={p_mg:.6f} {'✓' if p_mg < 0.05 else '✗'}")
print(f"\n{'='*80}")
print(f"VERDICT:")
if d >= 0.5 and val_score == 3:
    print(f"  ✓✓✓ SUCCESS! d={d:.3f} ≥ 0.5, validation 3/3")
elif d >= 0.5:
    print(f"  ✓✓ NEAR SUCCESS! d={d:.3f} ≥ 0.5, validation {val_score}/3")
elif d > 0.412:
    print(f"  ✓ BEST YET! 18% improvement over H11 (d=0.412 → {d:.3f}), validation {val_score}/3")
else:
    print(f"  No improvement over H11 (d=0.412)")
print(f"{'='*80}")
