#!/usr/bin/env python3
"""
H22: Combined Position + Latency

Hypothesis: Combine the best positional metric (MAD+degradation from H11)
with the best latency metric (early latency from H19).

These capture different dimensions - position = spatial accuracy, latency = temporal response.
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
print("H22: Combined Position + Latency")
print("="*80)

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

def compute_positional_metrics(eye_pos, target_pos):
    """MAD + degradation from H11"""
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    eye_clean = eye_pos[valid]
    target_clean = target_pos[valid]

    if len(eye_clean) < 50:
        return {'mad': np.nan, 'degradation': np.nan}

    # MAD
    median_pos = np.median(eye_clean)
    mad = np.median(np.abs(eye_clean - median_pos))

    # Degradation (early 20% vs late 20%)
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
    """Early latency from H19"""
    if len(latencies) < 3:
        return np.nan

    latencies = np.array(latencies)
    n = len(latencies)
    early_n = max(2, n // 3)

    return np.mean(latencies[:early_n])

results = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    # Positional metrics
    left_pos = compute_positional_metrics(df['LV'].values, df['TargetV'].values)
    right_pos = compute_positional_metrics(df['RV'].values, df['TargetV'].values)

    # Latency metrics
    left_latencies = compute_saccade_latencies(df['LV'].values, df['TargetV'].values)
    right_latencies = compute_saccade_latencies(df['RV'].values, df['TargetV'].values)

    left_lat = compute_latency_metric(left_latencies)
    right_lat = compute_latency_metric(right_latencies)

    # Check validity
    if (not np.isnan(left_pos['mad']) and not np.isnan(right_pos['mad']) and
        not np.isnan(left_lat) and not np.isnan(right_lat)):

        # H11 positional component
        pos_component = 0.45 * abs(left_pos['mad'] - right_pos['mad']) + \
                       0.55 * abs(left_pos['degradation'] - right_pos['degradation'])

        # Latency component (normalize to ~same scale as positional)
        lat_component = abs(left_lat - right_lat) / 100.0  # Convert ms to deg-like scale

        results.append({
            'class': 1 if seq['class_name'] == 'MG' else 0,
            'pos_component': pos_component,
            'lat_component': lat_component,
            'combined_equal': (pos_component + lat_component) / 2,
            'combined_weighted': 0.7 * pos_component + 0.3 * lat_component,
        })

results_df = pd.DataFrame(results)
hc_df = results_df[results_df['class'] == 0]
mg_df = results_df[results_df['class'] == 1]

print(f"\nDataset: {len(hc_df)} HC, {len(mg_df)} MG")

print(f"\n{'Metric':<40} {'HC':<12} {'MG':<12} {'d':<10}")
print("="*80)

metrics = {
    'Positional only (H11)': lambda df: df['pos_component'].values,
    'Latency only (normalized)': lambda df: df['lat_component'].values,
    'Combined (equal weight)': lambda df: df['combined_equal'].values,
    'Combined (0.7 pos + 0.3 lat)': lambda df: df['combined_weighted'].values,
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

    print(f"{name:<40} {hc_mean:<12.3f} {mg_mean:<12.3f} {d:<10.3f}")

    if d > best_d:
        best_d = d
        best_name = name

print(f"\n{'='*80}")
print(f"BEST: {best_name}")
print(f"Cohen's d: {best_d:.3f}")

if best_d > 0.5:
    print(f"\n✓✓✓ SUCCESS! d={best_d:.3f} ≥ 0.50")
elif best_d > 0.412:
    print(f"\n✓ IMPROVEMENT over H11! (d=0.412 → {best_d:.3f})")
else:
    print(f"\nNo improvement (H11 d=0.412, current d={best_d:.3f})")

print("="*80)
