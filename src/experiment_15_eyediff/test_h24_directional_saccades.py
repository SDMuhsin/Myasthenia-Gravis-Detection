#!/usr/bin/env python3
"""
H24: Directional Saccade Analysis (Upward vs Downward)

Clinical insight: Ocular MG commonly affects levator palpebrae and superior rectus
(muscles controlling upward gaze and eyelid elevation). Ptosis is a hallmark sign.

Hypothesis: UPWARD vertical saccades may show stronger MG signature than downward
or mixed vertical saccades.

Test separately:
1. All vertical (H22 baseline)
2. Upward only (target increases)
3. Downward only (target decreases)
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
print("H24: Directional Saccade Analysis")
print("="*80)

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

def compute_positional_metrics_directional(eye_pos, target_pos, direction='all'):
    """
    Compute MAD + degradation for specific saccade directions
    direction: 'all', 'up', 'down'
    """
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    eye_clean = eye_pos[valid]
    target_clean = target_pos[valid]

    if len(eye_clean) < 50:
        return {'mad': np.nan, 'degradation': np.nan}

    # Filter by direction if specified
    if direction != 'all':
        target_diff = np.diff(target_clean)

        if direction == 'up':
            # Keep indices where target increased (upward saccade)
            up_indices = np.where(target_diff > 5.0)[0] + 1  # +1 to get post-jump index
            mask = np.zeros(len(eye_clean), dtype=bool)
            for idx in up_indices:
                # Include ~50 samples after each upward jump
                mask[idx:min(idx+50, len(eye_clean))] = True
        elif direction == 'down':
            # Keep indices where target decreased (downward saccade)
            down_indices = np.where(target_diff < -5.0)[0] + 1
            mask = np.zeros(len(eye_clean), dtype=bool)
            for idx in down_indices:
                mask[idx:min(idx+50, len(eye_clean))] = True

        eye_clean = eye_clean[mask]
        target_clean = target_clean[mask]

        if len(eye_clean) < 30:
            return {'mad': np.nan, 'degradation': np.nan}

    # MAD
    median_pos = np.median(eye_clean)
    mad = np.median(np.abs(eye_clean - median_pos))

    # Degradation
    error = np.abs(eye_clean - target_clean)
    n = len(error)
    early_n = max(5, int(n * 0.2))
    late_n = max(5, int(n * 0.2))

    if n < 30:
        return {'mad': np.nan, 'degradation': np.nan}

    degradation = np.mean(error[-late_n:]) - np.mean(error[:early_n])

    return {'mad': mad, 'degradation': degradation}

def compute_saccade_latencies_directional(eye_pos, target_pos, direction='all', sample_rate_hz=120):
    """Compute latencies for specific directions"""
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    eye = eye_pos[valid]
    target = target_pos[valid]

    if len(target) < 20:
        return []

    target_diff = np.diff(target)

    # Filter by direction
    if direction == 'up':
        jump_indices = np.where(target_diff > 5.0)[0] + 1
    elif direction == 'down':
        jump_indices = np.where(target_diff < -5.0)[0] + 1
    else:  # 'all'
        jump_indices = np.where(np.abs(target_diff) > 5.0)[0] + 1

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

def compute_combined_metric_directional(lv, rv, tv, direction='all'):
    """Compute H22-style combined metric for specific direction"""
    pos_l = compute_positional_metrics_directional(lv, tv, direction)
    pos_r = compute_positional_metrics_directional(rv, tv, direction)

    lat_l = compute_latency_metric(compute_saccade_latencies_directional(lv, tv, direction))
    lat_r = compute_latency_metric(compute_saccade_latencies_directional(rv, tv, direction))

    if (np.isnan(pos_l['mad']) or np.isnan(pos_r['mad']) or
        np.isnan(lat_l) or np.isnan(lat_r)):
        return np.nan

    pos_comp = 0.45 * abs(pos_l['mad'] - pos_r['mad']) + \
               0.55 * abs(pos_l['degradation'] - pos_r['degradation'])
    lat_comp = abs(lat_l - lat_r) / 100.0

    return (pos_comp + lat_comp) / 2

results = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    all_score = compute_combined_metric_directional(
        df['LV'].values, df['RV'].values, df['TargetV'].values, 'all')
    up_score = compute_combined_metric_directional(
        df['LV'].values, df['RV'].values, df['TargetV'].values, 'up')
    down_score = compute_combined_metric_directional(
        df['LV'].values, df['RV'].values, df['TargetV'].values, 'down')

    results.append({
        'class': 1 if seq['class_name'] == 'MG' else 0,
        'all': all_score,
        'up': up_score,
        'down': down_score
    })

results_df = pd.DataFrame(results)
hc_df = results_df[results_df['class'] == 0]
mg_df = results_df[results_df['class'] == 1]

print(f"\nDataset: {len(hc_df)} HC, {len(mg_df)} MG")

print(f"\n{'Direction':<25} {'HC':<12} {'MG':<12} {'d':<10} {'N_HC':<8} {'N_MG':<8}")
print("="*80)

best_d = 0
best_name = None

for direction in ['all', 'up', 'down']:
    hc_vals = hc_df[direction].dropna().values
    mg_vals = mg_df[direction].dropna().values

    if len(hc_vals) < 10 or len(mg_vals) < 10:
        print(f"{direction:<25} {'N/A':<12} {'N/A':<12} {'N/A':<10} {len(hc_vals):<8} {len(mg_vals):<8}")
        continue

    hc_mean = np.mean(hc_vals)
    mg_mean = np.mean(mg_vals)
    hc_std = np.std(hc_vals)
    mg_std = np.std(mg_vals)

    pooled_std = np.sqrt((hc_std**2 + mg_std**2) / 2)
    d = (mg_mean - hc_mean) / pooled_std

    label = {'all': 'All vertical (H22)', 'up': 'Upward only', 'down': 'Downward only'}[direction]
    print(f"{label:<25} {hc_mean:<12.3f} {mg_mean:<12.3f} {d:<10.3f} {len(hc_vals):<8} {len(mg_vals):<8}")

    if d > best_d:
        best_d = d
        best_name = label

print(f"\n{'='*80}")
print(f"BEST: {best_name}")
print(f"Cohen's d: {best_d:.3f}")

if best_d > 0.5:
    print(f"\n✓✓✓ BREAKTHROUGH! d={best_d:.3f} ≥ 0.50")
elif best_d > 0.485:
    print(f"\n✓ Improvement over H22 (d=0.485 → {best_d:.3f})")
else:
    print(f"\nNo improvement over H22 (d=0.485)")

print("="*80)
