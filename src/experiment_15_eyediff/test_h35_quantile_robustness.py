#!/usr/bin/env python3
"""
H35: Quantile-Based Robust Metrics

Hypothesis: H30 uses robust MAD but non-robust mean for degradation/latency.
Replace means with quantiles (e.g., 75th percentile) for outlier resistance.

Rationale: Outliers in MG may be attention lapses, not fatigue. Quantiles
focus on typical performance, not extreme values.
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

def compute_upward_metrics_quantile(eye_pos, target_pos):
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
    error = eye_upward - target_upward

    # MAD (same as H30)
    mad = np.median(np.abs(error - np.median(error)))

    # Degradation: Use 75th percentile of late error - 25th percentile of early error
    n = len(error)
    early_n = max(5, int(n * 0.2))
    late_n = max(5, int(n * 0.2))

    # H30 style (mean-based)
    degradation_mean = np.mean(np.abs(error[-late_n:])) - np.mean(np.abs(error[:early_n]))

    # Quantile-based (75p late - 25p early)
    degradation_q75 = np.percentile(np.abs(error[-late_n:]), 75) - np.percentile(np.abs(error[:early_n]), 25)

    # Median-based
    degradation_median = np.median(np.abs(error[-late_n:])) - np.median(np.abs(error[:early_n]))

    return {
        'mad': mad,
        'degradation_mean': degradation_mean,
        'degradation_q75': degradation_q75,
        'degradation_median': degradation_median,
    }

def compute_latencies_upward(eye_pos, target_pos, sample_rate_hz=120):
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
        for i in range(jump_idx, min(jump_idx + int(0.5 * sample_rate_hz), len(eye))):
            if abs(eye[i] - new_target) <= threshold_deg:
                latency_samples = i - jump_idx
                latency_ms = (latency_samples / sample_rate_hz) * 1000
                latencies_ms.append(latency_ms)
                break
    return latencies_ms

print("Computing quantile-robust metrics...")
hc_data = {'cv': [], 'deg_mean': [], 'deg_q75': [], 'deg_median': [], 'lat_mean': [], 'lat_median': [], 'lat_q25': []}
mg_data = {'cv': [], 'deg_mean': [], 'deg_q75': [], 'deg_median': [], 'lat_mean': [], 'lat_median': [], 'lat_q25': []}

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)
    LV = df['LV'].values
    RV = df['RV'].values
    TargetV = df['TargetV'].values

    metrics_L = compute_upward_metrics_quantile(LV, TargetV)
    metrics_R = compute_upward_metrics_quantile(RV, TargetV)
    if metrics_L is None or metrics_R is None:
        continue

    latencies_L = compute_latencies_upward(LV, TargetV)
    latencies_R = compute_latencies_upward(RV, TargetV)
    if len(latencies_L) < 3 or len(latencies_R) < 3:
        continue

    mad_L = metrics_L['mad']
    mad_R = metrics_R['mad']
    cv_asym = abs(mad_L - mad_R) / ((mad_L + mad_R) / 2)

    # Degradation variants
    deg_mean_asym = abs(metrics_L['degradation_mean'] - metrics_R['degradation_mean'])
    deg_q75_asym = abs(metrics_L['degradation_q75'] - metrics_R['degradation_q75'])
    deg_median_asym = abs(metrics_L['degradation_median'] - metrics_R['degradation_median'])

    # Latency variants
    lat_mean = abs(np.mean(sorted(latencies_L)[:max(3, len(latencies_L)//4)]) - np.mean(sorted(latencies_R)[:max(3, len(latencies_R)//4)])) / 100
    lat_median = abs(np.median(latencies_L) - np.median(latencies_R)) / 100
    lat_q25 = abs(np.percentile(latencies_L, 25) - np.percentile(latencies_R, 25)) / 100

    data = hc_data if seq['label'] == 0 else mg_data
    data['cv'].append(cv_asym)
    data['deg_mean'].append(deg_mean_asym)
    data['deg_q75'].append(deg_q75_asym)
    data['deg_median'].append(deg_median_asym)
    data['lat_mean'].append(lat_mean)
    data['lat_median'].append(lat_median)
    data['lat_q25'].append(lat_q25)

for key in hc_data:
    hc_data[key] = np.array(hc_data[key])
    mg_data[key] = np.array(mg_data[key])

print(f"Valid samples: HC={len(hc_data['cv'])}, MG={len(mg_data['cv'])}\n")

print("="*80)
print("H35: QUANTILE-BASED ROBUST METRICS")
print("="*80)

# Test variants
variants = [
    ('H30 (baseline)', 0.5*(0.45*hc_data['cv'] + 0.55*hc_data['deg_mean']) + 0.5*hc_data['lat_mean'],
                        0.5*(0.45*mg_data['cv'] + 0.55*mg_data['deg_mean']) + 0.5*mg_data['lat_mean']),
    ('H35a (q75 deg)', 0.5*(0.45*hc_data['cv'] + 0.55*hc_data['deg_q75']) + 0.5*hc_data['lat_mean'],
                       0.5*(0.45*mg_data['cv'] + 0.55*mg_data['deg_q75']) + 0.5*mg_data['lat_mean']),
    ('H35b (median deg)', 0.5*(0.45*hc_data['cv'] + 0.55*hc_data['deg_median']) + 0.5*hc_data['lat_mean'],
                          0.5*(0.45*mg_data['cv'] + 0.55*mg_data['deg_median']) + 0.5*mg_data['lat_mean']),
    ('H35c (median lat)', 0.5*(0.45*hc_data['cv'] + 0.55*hc_data['deg_mean']) + 0.5*hc_data['lat_median'],
                          0.5*(0.45*mg_data['cv'] + 0.55*mg_data['deg_mean']) + 0.5*mg_data['lat_median']),
    ('H35d (q25 lat)', 0.5*(0.45*hc_data['cv'] + 0.55*hc_data['deg_mean']) + 0.5*hc_data['lat_q25'],
                       0.5*(0.45*mg_data['cv'] + 0.55*mg_data['deg_mean']) + 0.5*mg_data['lat_q25']),
    ('H35e (all robust)', 0.5*(0.45*hc_data['cv'] + 0.55*hc_data['deg_median']) + 0.5*hc_data['lat_median'],
                          0.5*(0.45*mg_data['cv'] + 0.55*mg_data['deg_median']) + 0.5*mg_data['lat_median']),
]

results = []
for name, hc_scores, mg_scores in variants:
    pooled_std = np.sqrt(((len(hc_scores)-1)*np.var(hc_scores, ddof=1) + (len(mg_scores)-1)*np.var(mg_scores, ddof=1)) / (len(hc_scores) + len(mg_scores) - 2))
    d = (np.mean(mg_scores) - np.mean(hc_scores)) / pooled_std
    u, p = stats.mannwhitneyu(mg_scores, hc_scores, alternative='greater')
    results.append((name, d, p))

print("\nRanked by Cohen's d:")
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
h30_d = results_sorted[0][1]

for i, (name, d_val, p_val) in enumerate(results_sorted, 1):
    improvement = (d_val / h30_d - 1) * 100
    print(f"{i}. {name:20} d={d_val:.3f} p={p_val:.4f} ({improvement:+.1f}%)")

best_name, best_d, best_p = results_sorted[0]

print("\n" + "="*80)
if best_d > h30_d * 1.02:
    print(f"✓ PROCEED with {best_name}")
else:
    print(f"✗ REJECT: Best ({best_name}) shows {(best_d/h30_d - 1)*100:+.1f}% vs H30")
    print("  Quantile-based robustness does not improve discrimination")
