#!/usr/bin/env python3
"""H19: Full validation check"""

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

def analyze_latency_fatigue(latencies):
    if len(latencies) < 3:
        return {'mean_latency': np.nan, 'early_latency': np.nan}

    latencies = np.array(latencies)
    n = len(latencies)
    early_n = max(2, n // 3)

    return {
        'mean_latency': np.mean(latencies),
        'early_latency': np.mean(latencies[:early_n])
    }

results = []
for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    left_latencies = compute_saccade_latencies(df['LV'].values, df['TargetV'].values)
    right_latencies = compute_saccade_latencies(df['RV'].values, df['TargetV'].values)

    if len(left_latencies) >= 3 and len(right_latencies) >= 3:
        left_metrics = analyze_latency_fatigue(left_latencies)
        right_metrics = analyze_latency_fatigue(right_latencies)

        results.append({
            'class': 1 if seq['class_name'] == 'MG' else 0,
            'early_asymmetry': abs(left_metrics['early_latency'] - right_metrics['early_latency']),
        })

results_df = pd.DataFrame(results)
hc_df = results_df[results_df['class'] == 0]
mg_df = results_df[results_df['class'] == 1]

hc_vals = hc_df['early_asymmetry'].values
mg_vals = mg_df['early_asymmetry'].values

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
print(f"H19: Early Latency Asymmetry Validation")
print(f"{'='*80}")
print(f"Dataset: {len(hc_vals)} HC, {len(mg_vals)} MG")
print(f"\nHC: {hc_mean:.2f} ± {hc_std:.2f} ms")
print(f"MG: {mg_mean:.2f} ± {mg_std:.2f} ms")
print(f"\nCohen's d: {d:.4f}")
print(f"Validation score: {val_score}/3")
print(f"\n  [1] MG > HC: p={p_mw:.6f} {'✓' if p_mw < 0.05 else '✗'}")
print(f"  [2] HC ≈ 0:  p={p_hc:.6f} {'✓' if p_hc >= 0.05 else '✗'}")
print(f"  [3] MG > 0:  p={p_mg:.6f} {'✓' if p_mg < 0.05 else '✗'}")
print(f"\n{'='*80}")
print(f"VERDICT: d={d:.3f} vs H11 d=0.412")
if d > 0.412:
    print(f"✓ Improvement!")
else:
    print(f"✗ No improvement (18% worse)")
print(f"{'='*80}")
