#!/usr/bin/env python3
"""
H23: Add Horizontal Dimension to H22

H22 achieved d=0.485 using vertical-only (position + latency).
Hypothesis: Adding horizontal saccades provides a third orthogonal dimension,
potentially pushing d≥0.5.

Test 3 combinations:
1. V-only (H22 baseline)
2. H-only
3. Combined H+V
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
print("H23: Horizontal + Vertical Combined")
print("="*80)

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

def compute_combined_metric(lh, rh, lv, rv, th, tv):
    """Compute H+V combined metric"""
    # Horizontal
    h_pos = compute_positional_metrics(lh, th)
    h_pos_r = compute_positional_metrics(rh, th)
    h_lat_l = compute_latency_metric(compute_saccade_latencies(lh, th))
    h_lat_r = compute_latency_metric(compute_saccade_latencies(rh, th))

    # Vertical
    v_pos = compute_positional_metrics(lv, tv)
    v_pos_r = compute_positional_metrics(rv, tv)
    v_lat_l = compute_latency_metric(compute_saccade_latencies(lv, tv))
    v_lat_r = compute_latency_metric(compute_saccade_latencies(rv, tv))

    # Check validity
    h_valid = not (np.isnan(h_pos['mad']) or np.isnan(h_pos_r['mad']) or
                   np.isnan(h_lat_l) or np.isnan(h_lat_r))
    v_valid = not (np.isnan(v_pos['mad']) or np.isnan(v_pos_r['mad']) or
                   np.isnan(v_lat_l) or np.isnan(v_lat_r))

    if not (h_valid or v_valid):
        return None

    # Compute H score
    h_score = np.nan
    if h_valid:
        h_pos_comp = 0.45 * abs(h_pos['mad'] - h_pos_r['mad']) + \
                     0.55 * abs(h_pos['degradation'] - h_pos_r['degradation'])
        h_lat_comp = abs(h_lat_l - h_lat_r) / 100.0
        h_score = (h_pos_comp + h_lat_comp) / 2

    # Compute V score
    v_score = np.nan
    if v_valid:
        v_pos_comp = 0.45 * abs(v_pos['mad'] - v_pos_r['mad']) + \
                     0.55 * abs(v_pos['degradation'] - v_pos_r['degradation'])
        v_lat_comp = abs(v_lat_l - v_lat_r) / 100.0
        v_score = (v_pos_comp + v_lat_comp) / 2

    return {
        'h_score': h_score,
        'v_score': v_score,
        'combined': np.nanmean([h_score, v_score])  # Average of H and V
    }

results = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    metrics = compute_combined_metric(
        df['LH'].values, df['RH'].values,
        df['LV'].values, df['RV'].values,
        df['TargetH'].values, df['TargetV'].values
    )

    if metrics and not np.isnan(metrics['combined']):
        results.append({
            'class': 1 if seq['class_name'] == 'MG' else 0,
            'h_score': metrics['h_score'],
            'v_score': metrics['v_score'],
            'combined': metrics['combined']
        })

results_df = pd.DataFrame(results)
hc_df = results_df[results_df['class'] == 0]
mg_df = results_df[results_df['class'] == 1]

print(f"\nDataset: {len(hc_df)} HC, {len(mg_df)} MG")

print(f"\n{'Metric':<30} {'HC':<12} {'MG':<12} {'d':<10}")
print("="*70)

for metric_name in ['v_score', 'h_score', 'combined']:
    hc_vals = hc_df[metric_name].dropna().values
    mg_vals = mg_df[metric_name].dropna().values

    if len(hc_vals) == 0 or len(mg_vals) == 0:
        continue

    hc_mean = np.mean(hc_vals)
    mg_mean = np.mean(mg_vals)
    hc_std = np.std(hc_vals)
    mg_std = np.std(mg_vals)

    pooled_std = np.sqrt((hc_std**2 + mg_std**2) / 2)
    d = (mg_mean - hc_mean) / pooled_std

    label = {'v_score': 'Vertical only (H22)', 'h_score': 'Horizontal only', 'combined': 'Combined H+V'}[metric_name]
    print(f"{label:<30} {hc_mean:<12.3f} {mg_mean:<12.3f} {d:<10.3f}")

print("="*70)
