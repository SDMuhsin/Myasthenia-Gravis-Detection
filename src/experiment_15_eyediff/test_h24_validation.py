#!/usr/bin/env python3
"""H24: Upward Saccades - Full Validation"""

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

def compute_positional_metrics_upward(eye_pos, target_pos):
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    eye_clean = eye_pos[valid]
    target_clean = target_pos[valid]

    if len(eye_clean) < 50:
        return {'mad': np.nan, 'degradation': np.nan}

    # Filter for upward saccades
    target_diff = np.diff(target_clean)
    up_indices = np.where(target_diff > 5.0)[0] + 1
    mask = np.zeros(len(eye_clean), dtype=bool)
    for idx in up_indices:
        mask[idx:min(idx+50, len(eye_clean))] = True

    eye_clean = eye_clean[mask]
    target_clean = target_clean[mask]

    if len(eye_clean) < 30:
        return {'mad': np.nan, 'degradation': np.nan}

    median_pos = np.median(eye_clean)
    mad = np.median(np.abs(eye_clean - median_pos))

    error = np.abs(eye_clean - target_clean)
    n = len(error)
    early_n = max(5, int(n * 0.2))
    late_n = max(5, int(n * 0.2))

    degradation = np.mean(error[-late_n:]) - np.mean(error[:early_n])

    return {'mad': mad, 'degradation': degradation}

def compute_saccade_latencies_upward(eye_pos, target_pos, sample_rate_hz=120):
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    eye = eye_pos[valid]
    target = target_pos[valid]

    if len(target) < 20:
        return []

    target_diff = np.diff(target)
    jump_indices = np.where(target_diff > 5.0)[0] + 1  # Upward only

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

    pos_l = compute_positional_metrics_upward(df['LV'].values, df['TargetV'].values)
    pos_r = compute_positional_metrics_upward(df['RV'].values, df['TargetV'].values)

    lat_l = compute_latency_metric(compute_saccade_latencies_upward(df['LV'].values, df['TargetV'].values))
    lat_r = compute_latency_metric(compute_saccade_latencies_upward(df['RV'].values, df['TargetV'].values))

    if (not np.isnan(pos_l['mad']) and not np.isnan(pos_r['mad']) and
        not np.isnan(lat_l) and not np.isnan(lat_r)):

        pos_comp = 0.45 * abs(pos_l['mad'] - pos_r['mad']) + \
                   0.55 * abs(pos_l['degradation'] - pos_r['degradation'])
        lat_comp = abs(lat_l - lat_r) / 100.0

        combined = (pos_comp + lat_comp) / 2

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
print(f"H24: UPWARD VERTICAL SACCADES - FULL VALIDATION")
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
    print(f"  ✓✓✓ COMPLETE SUCCESS! d={d:.3f} ≥ 0.5, validation 3/3")
    print(f"  Publication-worthy metric achieved!")
elif d >= 0.5:
    print(f"  ✓✓ MAJOR BREAKTHROUGH! d={d:.3f} ≥ 0.5, validation {val_score}/3")
    print(f"  Effect size target met, but validation test {3-val_score} failed")
else:
    print(f"  Effect size: d={d:.3f}")
print(f"{'='*80}")
