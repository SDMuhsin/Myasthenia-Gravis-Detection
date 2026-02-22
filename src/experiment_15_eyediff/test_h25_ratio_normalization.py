#!/usr/bin/env python3
"""
H25: Ratio-Based Normalization

Hypothesis: (worse - better) / (worse + better) may reduce HC baseline asymmetry
by normalizing out symmetric anatomical variation.

Rationale: Absolute differences |L-R| are affected by overall scale.
Relative differences may cancel out baseline asymmetry if it's proportional.

Expected: Improved HC≈0 test, potentially lower d due to information loss
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

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

# Reuse H24 helper functions
def compute_positional_metrics_upward(eye_pos, target_pos):
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    eye_clean = eye_pos[valid]
    target_clean = target_pos[valid]

    if len(eye_clean) < 50:
        return {'mad': np.nan, 'degradation': np.nan}

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
    jump_indices = np.where(target_diff > 5.0)[0] + 1

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

# New H25 ratio-based metric
results = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    pos_l = compute_positional_metrics_upward(df['LV'].values, df['TargetV'].values)
    pos_r = compute_positional_metrics_upward(df['RV'].values, df['TargetV'].values)

    lat_l = compute_latency_metric(compute_saccade_latencies_upward(df['LV'].values, df['TargetV'].values))
    lat_r = compute_latency_metric(compute_saccade_latencies_upward(df['RV'].values, df['TargetV'].values))

    if (not np.isnan(pos_l['mad']) and not np.isnan(pos_r['mad']) and
        not np.isnan(lat_l) and not np.isnan(lat_r)):

        # Compute per-eye scores (as in H24)
        score_l = 0.45 * pos_l['mad'] + 0.55 * pos_l['degradation'] + lat_l / 100
        score_r = 0.45 * pos_r['mad'] + 0.55 * pos_r['degradation'] + lat_r / 100

        # H24 baseline (absolute difference)
        h24_absolute = abs(score_l - score_r)

        # H25 variants (ratio-based)
        worse = max(score_l, score_r)
        better = min(score_l, score_r)
        
        if worse + better > 0.01:  # Avoid division by zero
            h25_ratio = (worse - better) / (worse + better)  # Normalized difference
        else:
            h25_ratio = np.nan

        if worse > 0.01:
            h25_ratio_worse = (worse - better) / worse  # Relative to worse eye
        else:
            h25_ratio_worse = np.nan

        if better > 0.01 and worse > 0.01:
            h25_log_ratio = np.log(worse / better)  # Log ratio (symmetric)
        else:
            h25_log_ratio = np.nan

        results.append({
            'class': seq['class_name'],
            'h24_absolute': h24_absolute,
            'h25_ratio': h25_ratio,
            'h25_ratio_worse': h25_ratio_worse,
            'h25_log_ratio': h25_log_ratio,
        })

results_df = pd.DataFrame(results)
hc_df = results_df[results_df['class'] == 'HC']
mg_df = results_df[results_df['class'] == 'MG']

print("\n" + "="*80)
print("H25: RATIO-BASED NORMALIZATION")
print("="*80)
print(f"\nDataset: {len(hc_df)} HC, {len(mg_df)} MG")

print("\n" + "-"*80)
print("Comparing 4 metrics:")
print("  [H24] Absolute:     |score_L - score_R|")
print("  [H25a] Ratio:       (worse - better) / (worse + better)")
print("  [H25b] Ratio/worse: (worse - better) / worse")
print("  [H25c] Log ratio:   log(worse / better)")
print("-"*80)

metrics = ['h24_absolute', 'h25_ratio', 'h25_ratio_worse', 'h25_log_ratio']
names = ['H24 Absolute', 'H25a Ratio', 'H25b Ratio/Worse', 'H25c Log Ratio']

print(f"\n{'Metric':<20} {'HC Mean':<12} {'MG Mean':<12} {'d':<10} {'Val':<6} {'HC≈0 p':<12}")
print("="*80)

best_d = 0
best_metric = None

for metric, name in zip(metrics, names):
    hc_vals = hc_df[metric].dropna().values
    mg_vals = mg_df[metric].dropna().values

    if len(hc_vals) < 10 or len(mg_vals) < 10:
        print(f"{name:<20} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'N/A':<6} {'N/A':<12}")
        continue

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

    print(f"{name:<20} {hc_mean:<12.4f} {mg_mean:<12.4f} {d:<10.4f} {val_score:<6}/3 {p_hc:<12.6f}")

    if d > best_d:
        best_d = d
        best_metric = name

print("\n" + "="*80)
print(f"BEST: {best_metric}")
print(f"Cohen's d: {best_d:.4f}")

if best_d >= 0.65:
    print("\n✓✓✓ BREAKTHROUGH! d ≥ 0.65 target achieved!")
elif best_d > 0.577:
    print(f"\n✓ IMPROVEMENT over H24 (d=0.577 → {best_d:.3f})")
elif best_d >= 0.55:
    print(f"\n~ MARGINAL improvement (d={best_d:.3f})")
else:
    print(f"\n✗ No improvement over H24 (d={best_d:.3f} ≤ 0.577)")

print("="*80)
