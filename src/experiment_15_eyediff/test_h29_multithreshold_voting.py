#!/usr/bin/env python3
"""
H29: Multi-threshold Voting

Hypothesis: Require multiple sub-metrics to exceed thresholds for positive detection.
Reduces false positives by requiring consensus across components.

Rationale: HC baseline asymmetry may show in ONE component (e.g., MAD) but not MULTIPLE.
MG dysfunction should be evident across MAD, degradation, AND latency simultaneously.

Expected: Higher specificity, potentially lower sensitivity, improved precision.
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

    # Component asymmetries
    mad_asym = abs(metrics_l['mad'] - metrics_r['mad'])
    deg_asym = abs(metrics_l['degradation'] - metrics_r['degradation'])
    lat_asym = abs(lat_l - lat_r)

    # H24 baseline
    h24_positional = 0.45 * mad_asym + 0.55 * deg_asym
    h24_full = 0.5 * h24_positional + 0.5 * (lat_asym / 100)

    # H29 NEW: Multi-threshold voting
    # Define component thresholds (empirically set based on HC median values)
    mad_threshold = 0.5  # degrees
    deg_threshold = 0.4  # degrees
    lat_threshold = 20   # ms

    # Count how many components exceed thresholds
    vote_mad = mad_asym > mad_threshold
    vote_deg = deg_asym > deg_threshold
    vote_lat = lat_asym > lat_threshold

    votes = int(vote_mad) + int(vote_deg) + int(vote_lat)

    # H29 variants:
    # a) Require ALL 3 components (strict consensus)
    h29_all = h24_full if votes == 3 else 0.0

    # b) Require ANY 2 components (majority vote)
    h29_majority = h24_full if votes >= 2 else 0.0

    # c) Weighted by vote count (soft voting)
    h29_weighted_votes = h24_full * (votes / 3.0)

    # d) Binary flag + magnitude: vote count as multiplier
    h29_vote_multiplier = h24_full * votes  # 0x, 1x, 2x, or 3x

    # e) Threshold on COMBINED metric only (for comparison)
    h29_combined_threshold = h24_full if h24_full > 0.4 else 0.0

    results.append({
        'class': seq['class_name'],
        'h24_baseline': h24_full,
        'h29_all': h29_all,
        'h29_majority': h29_majority,
        'h29_weighted_votes': h29_weighted_votes,
        'h29_vote_multiplier': h29_vote_multiplier,
        'h29_combined_threshold': h29_combined_threshold,
        'votes': votes,
        'mad_asym': mad_asym,
        'deg_asym': deg_asym,
        'lat_asym': lat_asym,
    })

results_df = pd.DataFrame(results)
hc_df = results_df[results_df['class'] == 'HC']
mg_df = results_df[results_df['class'] == 'MG']

print("\n" + "="*80)
print("H29: MULTI-THRESHOLD VOTING")
print("="*80)
print(f"\nDataset: {len(hc_df)} HC, {len(mg_df)} MG")
print("\nGoal: Require consensus across components to reduce false positives")
print("Hypothesis: HC baseline affects one component, MG affects all")

# Check voting statistics
print("\n" + "-"*80)
print("VOTING STATISTICS")
for n_votes in range(4):
    hc_count = (hc_df['votes'] == n_votes).sum()
    mg_count = (mg_df['votes'] == n_votes).sum()
    print(f"{n_votes} votes: HC={hc_count}/{len(hc_df)} ({100*hc_count/len(hc_df):.1f}%), MG={mg_count}/{len(mg_df)} ({100*mg_count/len(mg_df):.1f}%)")
print("-"*80)

metrics = ['h24_baseline', 'h29_all', 'h29_majority', 'h29_weighted_votes', 'h29_vote_multiplier', 'h29_combined_threshold']
names = ['H24 Baseline', 'H29a All (3/3)', 'H29b Majority (2/3)', 'H29c Weighted Votes', 'H29d Vote Multiplier', 'H29e Combined Thresh']

print(f"\n{'Metric':<22} {'HC Mean':<12} {'MG Mean':<12} {'d':<10} {'Val':<6} {'Spec@0.4':<10}")
print("="*90)

best_d = 0
best_metric = None

for metric, name in zip(metrics, names):
    hc_vals = hc_df[metric].values
    mg_vals = mg_df[metric].values

    hc_mean = np.mean(hc_vals)
    mg_mean = np.mean(mg_vals)
    hc_std = np.std(hc_vals)
    mg_std = np.std(mg_vals)

    pooled_std = np.sqrt((hc_std**2 + mg_std**2) / 2)
    d = (mg_mean - hc_mean) / pooled_std if pooled_std > 0 else 0.0

    # Handle zeros from filtering
    hc_nonzero = hc_vals[hc_vals > 0]
    mg_nonzero = mg_vals[mg_vals > 0]

    if len(hc_nonzero) >= 10 and len(mg_nonzero) >= 10:
        _, p_mw = stats.mannwhitneyu(mg_vals, hc_vals, alternative='greater')
        _, p_hc = stats.wilcoxon(hc_vals[hc_vals != 0]) if len(hc_vals[hc_vals != 0]) > 0 else (0, 1.0)
        _, p_mg = stats.wilcoxon(mg_vals[mg_vals != 0]) if len(mg_vals[mg_vals != 0]) > 0 else (0, 1.0)
        val_score = (p_mw < 0.05) + (p_hc >= 0.05) + (p_mg < 0.05)
    else:
        val_score = 0

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
