#!/usr/bin/env python3
"""
FRESH PERSPECTIVE: What do clinicians SEE in the raw data?

Stop aggregating. Look at actual eye movement patterns.
"""

import os
import sys
import numpy as np
import pandas as pd
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

print("="*80)
print("FRESH PERSPECTIVE: Visual Pattern Analysis")
print("="*80)

# Load data
raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

# Get a few examples from each class
hc_examples = [s for s in sequences if s['class_name'] == 'HC'][:5]
mg_examples = [s for s in sequences if s['class_name'] == 'MG'][:5]

# Create visualization
fig, axes = plt.subplots(5, 2, figsize=(16, 20))
fig.suptitle('Raw Eye Movement Patterns: HC vs MG', fontsize=16, fontweight='bold')

for idx, (hc_seq, mg_seq) in enumerate(zip(hc_examples, mg_examples)):
    # HC
    hc_df = pd.DataFrame(hc_seq['data'], columns=FEATURE_COLUMNS)
    ax_hc = axes[idx, 0]

    time = np.arange(len(hc_df)) / 1000  # Convert to seconds

    ax_hc.plot(time, hc_df['TargetH'], 'k--', linewidth=2, label='Target', alpha=0.5)
    ax_hc.plot(time, hc_df['LH'], 'b-', linewidth=1.5, label='Left Eye', alpha=0.8)
    ax_hc.plot(time, hc_df['RH'], 'r-', linewidth=1.5, label='Right Eye', alpha=0.8)

    ax_hc.set_ylabel('Position (degrees)', fontsize=10)
    ax_hc.set_title(f'HC Example {idx+1}: {hc_seq["filename"][:30]}...', fontsize=9)
    ax_hc.legend(loc='upper right', fontsize=8)
    ax_hc.grid(True, alpha=0.3)

    if idx == 4:
        ax_hc.set_xlabel('Time (seconds)', fontsize=10)

    # MG
    mg_df = pd.DataFrame(mg_seq['data'], columns=FEATURE_COLUMNS)
    ax_mg = axes[idx, 1]

    time = np.arange(len(mg_df)) / 1000

    ax_mg.plot(time, mg_df['TargetH'], 'k--', linewidth=2, label='Target', alpha=0.5)
    ax_mg.plot(time, mg_df['LH'], 'b-', linewidth=1.5, label='Left Eye', alpha=0.8)
    ax_mg.plot(time, mg_df['RH'], 'r-', linewidth=1.5, label='Right Eye', alpha=0.8)

    ax_mg.set_ylabel('Position (degrees)', fontsize=10)
    ax_mg.set_title(f'MG Example {idx+1}: {mg_seq["filename"][:30]}...', fontsize=9)
    ax_mg.legend(loc='upper right', fontsize=8)
    ax_mg.grid(True, alpha=0.3)

    if idx == 4:
        ax_mg.set_xlabel('Time (seconds)', fontsize=10)

plt.tight_layout()
plt.savefig('./results/exp_15_eyediff/raw_pattern_comparison.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved to: ./results/exp_15_eyediff/raw_pattern_comparison.png")

# Now let's compute what I SEE visually
print("\n" + "="*80)
print("PATTERN OBSERVATIONS")
print("="*80)

def analyze_tracking_quality(eye_pos, target_pos):
    """
    What patterns would a clinician notice?
    - Does eye reach target?
    - How closely does it track?
    - Are there lags/delays?
    - Overshoots/undershoots?
    """
    # Remove NaNs
    valid_idx = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    eye_clean = eye_pos[valid_idx]
    target_clean = target_pos[valid_idx]

    if len(eye_clean) < 10:
        return None

    # Tracking error
    error = eye_clean - target_clean

    # How often does eye FAIL to reach within 2 degrees of target?
    within_threshold = np.abs(error) < 2.0
    failure_rate = 1.0 - (np.sum(within_threshold) / len(within_threshold))

    # Average absolute error
    mean_error = np.mean(np.abs(error))

    # Lag: cross-correlation to detect delays
    # Simplified: just check if eye is consistently behind or ahead
    lag_direction = np.mean(error)  # Positive = undershoot, negative = overshoot

    return {
        'failure_rate': failure_rate,
        'mean_error': mean_error,
        'lag': lag_direction
    }

# Compare tracking quality for HC vs MG
hc_left_failures = []
hc_right_failures = []
mg_left_failures = []
mg_right_failures = []

for seq in hc_examples + [s for s in sequences if s['class_name']=='HC'][5:100]:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    left_metrics = analyze_tracking_quality(df['LH'].values, df['TargetH'].values)
    right_metrics = analyze_tracking_quality(df['RH'].values, df['TargetH'].values)

    if left_metrics and right_metrics:
        hc_left_failures.append(left_metrics['failure_rate'])
        hc_right_failures.append(right_metrics['failure_rate'])

for seq in mg_examples + [s for s in sequences if s['class_name']=='MG'][5:100]:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    left_metrics = analyze_tracking_quality(df['LH'].values, df['TargetH'].values)
    right_metrics = analyze_tracking_quality(df['RH'].values, df['TargetH'].values)

    if left_metrics and right_metrics:
        mg_left_failures.append(left_metrics['failure_rate'])
        mg_right_failures.append(right_metrics['failure_rate'])

print("\nTracking Failure Rates (proportion of time >2° from target):")
print(f"HC Left:  {np.mean(hc_left_failures):.3f} ± {np.std(hc_left_failures):.3f}")
print(f"HC Right: {np.mean(hc_right_failures):.3f} ± {np.std(hc_right_failures):.3f}")
print(f"MG Left:  {np.mean(mg_left_failures):.3f} ± {np.std(mg_left_failures):.3f}")
print(f"MG Right: {np.mean(mg_right_failures):.3f} ± {np.std(mg_right_failures):.3f}")

print("\n" + "="*80)
print("KEY QUESTION: What pattern discriminates affected from unaffected eye?")
print("="*80)
print("\nLook at the plots and identify visual differences...")
print("- Does one eye lag behind the other?")
print("- Does one eye fail to reach target more often?")
print("- Does one eye show more erratic movement?")
print("- Does one eye have larger position errors?")
print("\nVisualize and observe!")
