#!/usr/bin/env python3
"""
Cycle 8 Phase 1: Deep dive into individual saccade dynamics.

Instead of whole-sequence metrics, analyze INDIVIDUAL SACCADES to find
what differentiates MG from HC at the micro level.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_timeseries_data, merge_mg_classes

# Configuration
BASE_DIR = './data'
CLASS_DEFINITIONS = {
    'HC': {'path': 'Healthy control', 'label': 0},
    'MG': {'path': 'Definite MG', 'label': 1},
    'Probable_MG': {'path': 'Probable MG', 'label': 1},
}
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']


def detect_saccades(eye_pos, target_pos, threshold_degrees=5.0):
    """
    Detect individual saccades based on target changes.

    Returns list of (start_idx, end_idx, target_value) tuples.
    """
    target_diff = np.abs(np.diff(target_pos))
    target_changes = np.where(target_diff > threshold_degrees)[0] + 1

    saccade_starts = np.concatenate([[0], target_changes])
    saccade_ends = np.concatenate([target_changes, [len(target_pos)]])

    saccades = []
    for start_idx, end_idx in zip(saccade_starts, saccade_ends):
        if end_idx - start_idx >= 10:  # Minimum 10ms saccade
            target_value = np.median(target_pos[start_idx:end_idx])
            saccades.append((start_idx, end_idx, target_value))

    return saccades


def analyze_saccade_dynamics(eye_pos, target_pos, saccades, sampling_rate=1000):
    """
    For each saccade, compute detailed dynamics metrics.

    Returns list of dicts with per-saccade features.
    """
    saccade_features = []

    for start_idx, end_idx, target_value in saccades:
        eye_segment = eye_pos[start_idx:end_idx]

        # Skip if too many NaNs
        if np.sum(~np.isnan(eye_segment)) < 5:
            continue

        # Clean NaNs
        eye_clean = eye_segment[~np.isnan(eye_segment)]

        if len(eye_clean) < 5:
            continue

        # Duration
        duration_ms = (end_idx - start_idx) / sampling_rate * 1000

        # Initial and final positions
        initial_pos = eye_clean[0]
        final_pos = eye_clean[-1]

        # Amplitude (movement magnitude)
        amplitude = np.abs(final_pos - initial_pos)

        # Accuracy (error at end)
        final_error = np.abs(final_pos - target_value)

        # Velocity (approximate)
        velocity = np.abs(np.diff(eye_clean))
        peak_velocity = np.max(velocity) if len(velocity) > 0 else 0
        mean_velocity = np.mean(velocity) if len(velocity) > 0 else 0

        # Steadiness (variability in last 30% of saccade)
        settling_idx = int(len(eye_clean) * 0.7)
        settling_segment = eye_clean[settling_idx:]
        if len(settling_segment) > 2:
            settling_std = np.std(settling_segment)
        else:
            settling_std = 0

        # Overshoot/undershoot
        overshoot = final_pos - target_value

        saccade_features.append({
            'duration_ms': duration_ms,
            'amplitude': amplitude,
            'final_error': final_error,
            'peak_velocity': peak_velocity,
            'mean_velocity': mean_velocity,
            'settling_std': settling_std,
            'overshoot': overshoot,
        })

    return saccade_features


def main():
    print("="*80)
    print("CYCLE 8 PHASE 1: Individual Saccade Dynamics Analysis")
    print("="*80)

    # Load data
    print("\nLoading data...")
    raw_sequences = load_timeseries_data(
        BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS,
        'utf-16-le', ',', 50
    )
    sequences = merge_mg_classes(raw_sequences)

    # Sample balanced subset
    hc_seqs = [s for s in sequences if s['class_name'] == 'HC'][:200]
    mg_seqs = [s for s in sequences if s['class_name'] == 'MG'][:200]

    print(f"Analyzing {len(hc_seqs)} HC and {len(mg_seqs)} MG sequences...")

    # Analyze horizontal saccades
    hc_saccade_features = []
    mg_saccade_features = []

    for seq in hc_seqs:
        df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

        # Left eye
        saccades_left = detect_saccades(df['LH'].values, df['TargetH'].values)
        features_left = analyze_saccade_dynamics(df['LH'].values, df['TargetH'].values, saccades_left)

        # Right eye
        saccades_right = detect_saccades(df['RH'].values, df['TargetH'].values)
        features_right = analyze_saccade_dynamics(df['RH'].values, df['TargetH'].values, saccades_right)

        hc_saccade_features.extend(features_left + features_right)

    for seq in mg_seqs:
        df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

        # Left eye
        saccades_left = detect_saccades(df['LH'].values, df['TargetH'].values)
        features_left = analyze_saccade_dynamics(df['LH'].values, df['TargetH'].values, saccades_left)

        # Right eye
        saccades_right = detect_saccades(df['RH'].values, df['TargetH'].values)
        features_right = analyze_saccade_dynamics(df['RH'].values, df['TargetH'].values, saccades_right)

        mg_saccade_features.extend(features_left + features_right)

    print(f"\nTotal saccades detected:")
    print(f"  HC: {len(hc_saccade_features)}")
    print(f"  MG: {len(mg_saccade_features)}")

    # Convert to DataFrames
    hc_df = pd.DataFrame(hc_saccade_features)
    mg_df = pd.DataFrame(mg_saccade_features)

    print("\n" + "="*80)
    print("PER-SACCADE FEATURE COMPARISON")
    print("="*80)

    for feature in ['duration_ms', 'amplitude', 'final_error', 'peak_velocity',
                    'mean_velocity', 'settling_std', 'overshoot']:

        hc_vals = hc_df[feature].values
        mg_vals = mg_df[feature].values

        # Remove NaNs
        hc_vals_clean = hc_vals[~np.isnan(hc_vals)]
        mg_vals_clean = mg_vals[~np.isnan(mg_vals)]

        if len(hc_vals_clean) == 0 or len(mg_vals_clean) == 0:
            continue

        hc_mean = np.mean(hc_vals_clean)
        hc_std = np.std(hc_vals_clean)
        mg_mean = np.mean(mg_vals_clean)
        mg_std = np.std(mg_vals_clean)

        # Compute Cohen's d
        pooled_std = np.sqrt((hc_std**2 + mg_std**2) / 2)
        cohens_d = (mg_mean - hc_mean) / pooled_std if pooled_std > 0 else 0

        # Ratio
        ratio = mg_mean / hc_mean if hc_mean != 0 else 0

        print(f"\n{feature}:")
        print(f"  HC: {hc_mean:.4f} ± {hc_std:.4f}")
        print(f"  MG: {mg_mean:.4f} ± {mg_std:.4f}")
        print(f"  Cohen's d: {cohens_d:.4f}")
        print(f"  MG/HC ratio: {ratio:.2f}x")

    print("\n" + "="*80)
    print("KEY INSIGHT:")
    print("="*80)
    print("Individual saccade features reveal micro-level differences")
    print("that may be hidden when aggregating across entire sequence.")
    print("="*80)


if __name__ == "__main__":
    main()
