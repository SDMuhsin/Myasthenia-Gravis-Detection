#!/usr/bin/env python3
"""
H19 v2: Proper saccade latency measurement

Key insight from clinicians: Measure TIME TO REACH TARGET after target jumps.
The target is a square wave - it jumps between positions.
Measure how long each eye takes to catch up to the new position.
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
print("H19 v2: Saccade Latency (Time-to-Target)")
print("="*80)

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

def compute_saccade_latencies(eye_pos, target_pos, sample_rate_hz=120):
    """
    Measure latency for each saccade.

    Approach:
    1. Find target transitions (square wave jumps)
    2. For each jump, measure time until eye gets within 3° of new target
    3. Return list of latencies in milliseconds
    """
    # Clean data
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    eye = eye_pos[valid]
    target = target_pos[valid]

    if len(target) < 20:
        return []

    # Detect target jumps (derivative > threshold)
    target_diff = np.abs(np.diff(target))
    jump_threshold = 5.0  # degrees
    jump_indices = np.where(target_diff > jump_threshold)[0] + 1

    latencies_ms = []

    for jump_idx in jump_indices:
        if jump_idx >= len(target) - 10:  # Need lookahead
            continue

        new_target = target[jump_idx]

        # Find when eye reaches within 3° of new target
        threshold_deg = 3.0

        for offset in range(1, min(100, len(eye) - jump_idx)):
            eye_pos_now = eye[jump_idx + offset]
            error = abs(eye_pos_now - new_target)

            if error < threshold_deg:
                # Convert samples to milliseconds
                latency_ms = (offset / sample_rate_hz) * 1000
                latencies_ms.append(latency_ms)
                break

    return latencies_ms

def analyze_latency_fatigue(latencies):
    """Measure if latencies increase over time (fatigue)"""
    if len(latencies) < 3:  # Need at least 3 saccades
        return {
            'mean_latency': np.nan,
            'median_latency': np.nan,
            'early_latency': np.nan,
            'late_latency': np.nan,
            'fatigue_slope': np.nan
        }

    latencies = np.array(latencies)
    n = len(latencies)

    # Early vs late (first 1/3 vs last 1/3)
    early_n = max(2, n // 3)
    late_n = max(2, n // 3)

    early = np.mean(latencies[:early_n])
    late = np.mean(latencies[-late_n:])

    # Linear trend
    t = np.arange(n)
    if len(t) >= 3:  # Need at least 3 points for polyfit
        slope = np.polyfit(t, latencies, 1)[0]
    else:
        slope = np.nan

    return {
        'mean_latency': np.mean(latencies),
        'median_latency': np.median(latencies),
        'early_latency': early,
        'late_latency': late,
        'fatigue_slope': slope
    }

# Process sequences
print("\nProcessing sequences...")

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
            'left_mean': left_metrics['mean_latency'],
            'right_mean': right_metrics['mean_latency'],
            'left_early': left_metrics['early_latency'],
            'right_early': right_metrics['early_latency'],
            'left_late': left_metrics['late_latency'],
            'right_late': right_metrics['late_latency'],
            'left_slope': left_metrics['fatigue_slope'],
            'right_slope': right_metrics['fatigue_slope'],
        })

results_df = pd.DataFrame(results)
hc_df = results_df[results_df['class'] == 0]
mg_df = results_df[results_df['class'] == 1]

print(f"Valid sequences: {len(hc_df)} HC, {len(mg_df)} MG")

if len(hc_df) < 10 or len(mg_df) < 10:
    print("\n✗ Not enough valid sequences. Latency detection may have failed.")
    print("Debugging: Check if target transitions are being detected correctly.")
else:
    # Test metrics
    print(f"\n{'Metric':<35} {'HC':<12} {'MG':<12} {'d':<10} {'HC≈0 p':<12}")
    print("="*85)

    test_metrics = {
        'Mean latency asymmetry (ms)': lambda df: np.abs(df['left_mean'] - df['right_mean']),
        'Early latency asymmetry (ms)': lambda df: np.abs(df['left_early'] - df['right_early']),
        'Late latency asymmetry (ms)': lambda df: np.abs(df['left_late'] - df['right_late']),
        'Fatigue slope asymmetry (ms/saccade)': lambda df: np.abs(df['left_slope'] - df['right_slope']),
    }

    best_d = 0
    best_name = None

    for name, func in test_metrics.items():
        hc_vals = func(hc_df).values
        mg_vals = func(mg_df).values

        hc_mean = np.mean(hc_vals)
        mg_mean = np.mean(mg_vals)

        d = (mg_mean - hc_mean) / np.sqrt((np.std(hc_vals)**2 + np.std(mg_vals)**2) / 2)

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
