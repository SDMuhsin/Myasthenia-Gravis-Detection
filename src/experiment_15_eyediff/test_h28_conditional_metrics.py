#!/usr/bin/env python3
"""
H28: Conditional Metrics

Hypothesis: Only compute asymmetry when EITHER eye shows degradation.
If both eyes are stable (no degradation), ignore baseline asymmetry.

Rationale: HC baseline asymmetry (~0.58°) exists even when both eyes are healthy.
By filtering out cases where NEITHER eye shows degradation, we may reduce HC false positives.

Expected: Higher specificity by filtering stable HC cases, potentially lower sensitivity.
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

def compute_upward_metrics(eye_pos, target_pos):
    """Extract upward saccade data and compute metrics"""
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    eye_clean = eye_pos[valid]
    target_clean = target_pos[valid]

    if len(eye_clean) < 50:
        return None

    target_diff = np.diff(target_clean)
    up_indices = np.where(target_diff > 5.0)[0] + 1
    mask = np.zeros(len(eye_clean), dtype=bool)
    for idx in up_indices:
        mask[idx:min(idx+50, len(eye_clean))] = True

    eye_upward = eye_clean[mask]
    target_upward = target_clean[mask]

    if len(eye_upward) < 30:
        return None

    # Tracking error time series
    error = eye_upward - target_upward

    # MAD
    mad = np.median(np.abs(eye_upward - np.median(eye_upward)))

    # Degradation
    n = len(error)
    early_n = max(5, int(n * 0.2))
    late_n = max(5, int(n * 0.2))
    degradation = np.mean(np.abs(error[-late_n:])) - np.mean(np.abs(error[:early_n]))

    return {
        'mad': mad,
        'degradation': degradation,
        'error': error
    }

def compute_saccade_latencies_upward(eye_pos, target_pos, sample_rate_hz=120):
    """Compute saccade latencies for upward jumps"""
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
    """Compute early latency metric"""
    if len(latencies) < 3:
        return np.nan
    latencies = np.array(latencies)
    n = len(latencies)
    early_n = max(2, n // 3)
    return np.mean(latencies[:early_n])

results = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    metrics_l = compute_upward_metrics(df['LV'].values, df['TargetV'].values)
    metrics_r = compute_upward_metrics(df['RV'].values, df['TargetV'].values)

    if metrics_l is None or metrics_r is None:
        continue

    lat_l = compute_latency_metric(compute_saccade_latencies_upward(df['LV'].values, df['TargetV'].values))
    lat_r = compute_latency_metric(compute_saccade_latencies_upward(df['RV'].values, df['TargetV'].values))

    if np.isnan(lat_l) or np.isnan(lat_r):
        continue

    # H24 baseline components
    mad_asym = abs(metrics_l['mad'] - metrics_r['mad'])
    deg_asym = abs(metrics_l['degradation'] - metrics_r['degradation'])
    lat_asym = abs(lat_l - lat_r) / 100

    h24_positional = 0.45 * mad_asym + 0.55 * deg_asym
    h24_full = 0.5 * h24_positional + 0.5 * lat_asym

    # H28 NEW: Conditional asymmetry
    # Criteria for "degradation present":
    # 1. Either eye shows degradation > threshold
    # 2. Either eye shows high MAD
    # 3. Either eye shows high latency

    deg_threshold = 0.5  # degrees (significant degradation)
    mad_threshold = 1.0  # degrees (high variability)
    lat_threshold = 250  # ms (slow response)

    # Check if EITHER eye shows dysfunction
    has_degradation = (metrics_l['degradation'] > deg_threshold or
                      metrics_r['degradation'] > deg_threshold)

    has_high_mad = (metrics_l['mad'] > mad_threshold or
                   metrics_r['mad'] > mad_threshold)

    has_high_latency = (lat_l > lat_threshold or
                       lat_r > lat_threshold)

    # H28 variants:
    # a) Only use asymmetry if degradation detected
    h28_deg_only = h24_full if has_degradation else 0.0

    # b) Only use if ANY dysfunction detected
    h28_any_dysfunction = h24_full if (has_degradation or has_high_mad or has_high_latency) else 0.0

    # c) Only use if degradation asymmetry exceeds threshold
    h28_deg_asym_threshold = h24_full if deg_asym > 0.3 else 0.0

    # d) Weighted by degradation magnitude (soft threshold)
    max_deg = max(abs(metrics_l['degradation']), abs(metrics_r['degradation']))
    weight = min(1.0, max_deg / deg_threshold)  # 0 to 1
    h28_weighted = h24_full * weight

    results.append({
        'class': seq['class_name'],
        'h24_baseline': h24_full,
        'h28_deg_only': h28_deg_only,
        'h28_any_dysfunction': h28_any_dysfunction,
        'h28_deg_asym_threshold': h28_deg_asym_threshold,
        'h28_weighted': h28_weighted,
        'has_degradation': has_degradation,
        'has_high_mad': has_high_mad,
        'has_high_latency': has_high_latency,
    })

results_df = pd.DataFrame(results)
hc_df = results_df[results_df['class'] == 'HC']
mg_df = results_df[results_df['class'] == 'MG']

print("\n" + "="*80)
print("H28: CONDITIONAL METRICS")
print("="*80)
print(f"\nDataset: {len(hc_df)} HC, {len(mg_df)} MG")
print("\nGoal: Only compute asymmetry when dysfunction is detected")
print("Hypothesis: Filter stable HC cases to reduce false positives")

# Check filtering statistics
print("\n" + "-"*80)
print("FILTERING STATISTICS")
print(f"HC with degradation: {hc_df['has_degradation'].sum()}/{len(hc_df)} ({100*hc_df['has_degradation'].mean():.1f}%)")
print(f"MG with degradation: {mg_df['has_degradation'].sum()}/{len(mg_df)} ({100*mg_df['has_degradation'].mean():.1f}%)")
print(f"HC with ANY dysfunction: {hc_df[['has_degradation', 'has_high_mad', 'has_high_latency']].any(axis=1).sum()}/{len(hc_df)} ({100*hc_df[['has_degradation', 'has_high_mad', 'has_high_latency']].any(axis=1).mean():.1f}%)")
print(f"MG with ANY dysfunction: {mg_df[['has_degradation', 'has_high_mad', 'has_high_latency']].any(axis=1).sum()}/{len(mg_df)} ({100*mg_df[['has_degradation', 'has_high_mad', 'has_high_latency']].any(axis=1).mean():.1f}%)")
print("-"*80)

metrics = ['h24_baseline', 'h28_deg_only', 'h28_any_dysfunction', 'h28_deg_asym_threshold', 'h28_weighted']
names = ['H24 Baseline', 'H28a Deg Only', 'H28b Any Dysfunc', 'H28c Deg Asym Thresh', 'H28d Weighted']

print(f"\n{'Metric':<22} {'HC Mean':<12} {'MG Mean':<12} {'d':<10} {'Val':<6} {'Spec@0.4':<10}")
print("="*90)

best_d = 0
best_metric = None

for metric, name in zip(metrics, names):
    hc_vals = hc_df[metric].values
    mg_vals = mg_df[metric].values

    # Handle zeros from filtering
    hc_nonzero = hc_vals[hc_vals > 0]
    mg_nonzero = mg_vals[mg_vals > 0]

    if len(hc_nonzero) < 10 or len(mg_nonzero) < 10:
        print(f"{name:<22} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'N/A':<6} {'N/A':<10}")
        continue

    hc_mean = np.mean(hc_vals)
    mg_mean = np.mean(mg_vals)
    hc_std = np.std(hc_vals)
    mg_std = np.std(mg_vals)

    pooled_std = np.sqrt((hc_std**2 + mg_std**2) / 2)
    d = (mg_mean - hc_mean) / pooled_std if pooled_std > 0 else 0.0

    _, p_mw = stats.mannwhitneyu(mg_vals, hc_vals, alternative='greater')
    _, p_hc = stats.wilcoxon(hc_vals[hc_vals != 0]) if len(hc_vals[hc_vals != 0]) > 0 else (0, 1.0)
    _, p_mg = stats.wilcoxon(mg_vals[mg_vals != 0]) if len(mg_vals[mg_vals != 0]) > 0 else (0, 1.0)

    val_score = (p_mw < 0.05) + (p_hc >= 0.05) + (p_mg < 0.05)

    # Specificity at threshold 0.4
    spec = 100 * np.sum(hc_vals < 0.4) / len(hc_vals)

    print(f"{name:<22} {hc_mean:<12.4f} {mg_mean:<12.4f} {d:<10.4f} {val_score:<6}/3 {spec:<10.1f}%")

    if d > best_d:
        best_d = d
        best_metric = name

print("\n" + "="*90)
print(f"BEST: {best_metric}, d={best_d:.4f}")

if best_d >= 0.65:
    print("\n✓✓✓ BREAKTHROUGH! d ≥ 0.65 target achieved!")
elif best_d > 0.577:
    print(f"\n✓ IMPROVEMENT over H24 (d=0.577 → {best_d:.3f})")
else:
    print(f"\n✗ No improvement over H24 (best d={best_d:.3f} ≤ 0.577)")

print("="*90)
