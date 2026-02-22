#!/usr/bin/env python3
"""
H21: Maximum/P90 Latency Asymmetry

Hypothesis: MG shows fatigue in WORST-CASE performance. Compare the SLOWEST
saccades (p90 or max) between eyes, not mean latency.

Fatigued muscle shows up most clearly when pushed to limits.
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
print("H21: Maximum Latency Asymmetry")
print("="*80)

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

def analyze_max_latency(latencies):
    if len(latencies) < 5:
        return {'p90': np.nan, 'p95': np.nan, 'max': np.nan, 'mean': np.nan}

    latencies = np.array(latencies)

    return {
        'p90': np.percentile(latencies, 90),
        'p95': np.percentile(latencies, 95),
        'max': np.max(latencies),
        'mean': np.mean(latencies)
    }

results = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    left_latencies = compute_saccade_latencies(df['LV'].values, df['TargetV'].values)
    right_latencies = compute_saccade_latencies(df['RV'].values, df['TargetV'].values)

    if len(left_latencies) >= 5 and len(right_latencies) >= 5:
        left_metrics = analyze_max_latency(left_latencies)
        right_metrics = analyze_max_latency(right_latencies)

        results.append({
            'class': 1 if seq['class_name'] == 'MG' else 0,
            'p90_asym': abs(left_metrics['p90'] - right_metrics['p90']),
            'p95_asym': abs(left_metrics['p95'] - right_metrics['p95']),
            'max_asym': abs(left_metrics['max'] - right_metrics['max']),
            'mean_asym': abs(left_metrics['mean'] - right_metrics['mean']),
        })

results_df = pd.DataFrame(results)
hc_df = results_df[results_df['class'] == 0]
mg_df = results_df[results_df['class'] == 1]

print(f"\nDataset: {len(hc_df)} HC, {len(mg_df)} MG")

print(f"\n{'Metric':<35} {'HC':<12} {'MG':<12} {'d':<10} {'HC≈0 p':<12}")
print("="*85)

metrics = {
    'Mean latency asymmetry (ms)': lambda df: df['mean_asym'].values,
    'P90 latency asymmetry (ms)': lambda df: df['p90_asym'].values,
    'P95 latency asymmetry (ms)': lambda df: df['p95_asym'].values,
    'Max latency asymmetry (ms)': lambda df: df['max_asym'].values,
}

best_d = 0
best_name = None

for name, func in metrics.items():
    hc_vals = func(hc_df)
    mg_vals = func(mg_df)

    hc_mean = np.mean(hc_vals)
    mg_mean = np.mean(mg_vals)
    hc_std = np.std(hc_vals)
    mg_std = np.std(mg_vals)

    pooled_std = np.sqrt((hc_std**2 + mg_std**2) / 2)
    d = (mg_mean - hc_mean) / pooled_std

    _, p_hc = stats.wilcoxon(hc_vals)

    print(f"{name:<35} {hc_mean:<12.2f} {mg_mean:<12.2f} {d:<10.3f} {p_hc:<12.6f}")

    if d > best_d:
        best_d = d
        best_name = name

print(f"\n{'='*85}")
print(f"BEST: {best_name}")
print(f"Cohen's d: {best_d:.3f}")

if best_d > 0.5:
    print(f"\n✓✓✓ SUCCESS! d={best_d:.3f} ≥ 0.50")
elif best_d > 0.41:
    print(f"\n✓ IMPROVEMENT over H11! (d=0.41 → {best_d:.3f})")
else:
    print(f"\nNo improvement (H11 d=0.41, current d={best_d:.3f})")

print("="*85)
