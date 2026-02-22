#!/usr/bin/env python3
"""
H19: Proper Time-to-Target Analysis

Clinician insight: Affected eye takes LONGER to reach target after target changes.
Measure latency/reaction time for each saccade, not positional error.

Key: Detect target changes (square wave transitions), measure how long until
eye position reaches within threshold of new target.
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
print("H19: Time-to-Target Latency Analysis")
print("="*80)

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

def detect_target_changes(target_signal, min_change=5.0):
    """Detect when target changes position (square wave transitions)"""
    changes = []
    target_clean = target_signal[~np.isnan(target_signal)]

    if len(target_clean) < 10:
        return changes

    for i in range(1, len(target_clean)):
        delta = abs(target_clean[i] - target_clean[i-1])
        if delta > min_change:  # Significant position change
            changes.append({
                'index': i,
                'from': target_clean[i-1],
                'to': target_clean[i]
            })

    return changes

def measure_time_to_target(eye_signal, target_signal, threshold=2.0):
    """
    For each target change, measure how long until eye reaches new target.

    threshold: degrees - how close eye must get to be considered "reached"
    """
    changes = detect_target_changes(target_signal)
    latencies = []

    for change in changes:
        start_idx = change['index']
        target_pos = change['to']

        # Look forward to find when eye reaches target
        for offset in range(100):  # Max 100 samples (~100ms at 1kHz)
            idx = start_idx + offset
            if idx >= len(eye_signal):
                break

            if np.isnan(eye_signal[idx]):
                continue

            # Check if eye is within threshold of target
            error = abs(eye_signal[idx] - target_pos)
            if error < threshold:
                latencies.append(offset)  # Time steps until reached
                break

    return latencies

def compute_latency_metrics(eye_pos, target_pos):
    """Compute latency-based metrics"""
    latencies = measure_time_to_target(eye_pos, target_pos, threshold=3.0)

    if len(latencies) < 3:  # Need at least 3 saccades
        return {
            'mean_latency': np.nan,
            'p90_latency': np.nan,
            'early_latency': np.nan,
            'late_latency': np.nan,
            'latency_degradation': np.nan
        }

    latencies = np.array(latencies)

    # Early vs late comparison
    n = len(latencies)
    early_n = max(2, n // 3)
    late_n = max(2, n // 3)

    early_latency = np.mean(latencies[:early_n])
    late_latency = np.mean(latencies[-late_n:])

    return {
        'mean_latency': np.mean(latencies),
        'p90_latency': np.percentile(latencies, 90),
        'early_latency': early_latency,
        'late_latency': late_latency,
        'latency_degradation': late_latency - early_latency
    }

# Test on first 100 sequences
print("\nAnalyzing latency metrics...")

results = []
for seq in sequences[:200]:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    left_metrics = compute_latency_metrics(df['LV'].values, df['TargetV'].values)
    right_metrics = compute_latency_metrics(df['RV'].values, df['TargetV'].values)

    if not (np.isnan(left_metrics['mean_latency']) or np.isnan(right_metrics['mean_latency'])):
        results.append({
            'class': 1 if seq['class_name'] == 'MG' else 0,
            'left_mean': left_metrics['mean_latency'],
            'right_mean': right_metrics['mean_latency'],
            'left_p90': left_metrics['p90_latency'],
            'right_p90': right_metrics['p90_latency'],
            'left_degrad': left_metrics['latency_degradation'],
            'right_degrad': right_metrics['latency_degradation'],
        })

results_df = pd.DataFrame(results)
hc_df = results_df[results_df['class'] == 0]
mg_df = results_df[results_df['class'] == 1]

print(f"\nProcessed {len(hc_df)} HC, {len(mg_df)} MG")

# Test different asymmetry formulations
print(f"\n{'Metric':<30} {'HC mean':<12} {'MG mean':<12} {'d':<10} {'HC≈0 p':<12}")
print("="*80)

metrics = {
    'Mean latency asymmetry': lambda df: np.abs(df['left_mean'] - df['right_mean']),
    'P90 latency asymmetry': lambda df: np.abs(df['left_p90'] - df['right_p90']),
    'Degradation asymmetry': lambda df: np.abs(df['left_degrad'] - df['right_degrad']),
}

best_d = 0
best_name = None

for name, metric_func in metrics.items():
    hc_vals = metric_func(hc_df).values
    mg_vals = metric_func(mg_df).values

    hc_mean = np.mean(hc_vals)
    mg_mean = np.mean(mg_vals)

    d = (mg_mean - hc_mean) / np.sqrt((np.std(hc_vals)**2 + np.std(mg_vals)**2) / 2)

    _, p_hc = stats.wilcoxon(hc_vals)

    print(f"{name:<30} {hc_mean:<12.3f} {mg_mean:<12.3f} {d:<10.3f} {p_hc:<12.6f}")

    if d > best_d:
        best_d = d
        best_name = name

print(f"\n{'='*80}")
print(f"BEST: {best_name}")
print(f"Cohen's d: {best_d:.3f}")

if best_d > 0.41:
    print(f"✓ IMPROVEMENT over H11 (d=0.41 → {best_d:.3f})")
else:
    print(f"Current approach: d={best_d:.3f} vs H11 d=0.41")

print(f"\n{'='*80}")
print("\nNext: If promising, run on full dataset and test degradation of latency over time")
print("="*80)
