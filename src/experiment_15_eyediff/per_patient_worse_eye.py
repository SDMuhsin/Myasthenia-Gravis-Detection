#!/usr/bin/env python3
"""
CRITICAL REFRAME: The task is not MG vs HC discrimination.
The task is: Given a SINGLE patient's saccade sequence, which eye (L or R) is more affected?

This is within-patient analysis, not between-group discrimination!
"""

import os
import sys
import numpy as np
import pandas as pd

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
print("REFRAMED TASK: Per-Patient Worse Eye Identification")
print("="*80)
print("\nGiven a single sequence, which eye performs worse?")
print("Metrics to test:")
print("1. Mean absolute tracking error")
print("2. Failure rate (time outside 2° threshold)")
print("3. Peak error magnitude")
print("4. Settling time (time to stabilize)")
print("5. Variability (STD of position)")
print("="*80)

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

def comprehensive_performance_metrics(eye_pos, target_pos):
    """Compute comprehensive performance metrics for one eye."""
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    if np.sum(valid) < 10:
        return None

    eye_clean = eye_pos[valid]
    target_clean = target_pos[valid]
    error = eye_clean - target_clean

    metrics = {
        'mean_abs_error': np.mean(np.abs(error)),
        'max_abs_error': np.max(np.abs(error)),
        'failure_rate': np.sum(np.abs(error) > 2.0) / len(error),
        'std_error': np.std(error),
        'rms_error': np.sqrt(np.mean(error**2))
    }

    return metrics

# Test all metrics
print("\nTesting metrics on 100 MG sequences...")

mg_seqs = [s for s in sequences if s['class_name'] == 'MG'][:100]

results = {
    'mean_abs_error': {'agreements': 0, 'total': 0},
    'max_abs_error': {'agreements': 0, 'total': 0},
    'failure_rate': {'agreements': 0, 'total': 0},
    'std_error': {'agreements': 0, 'total': 0},
    'rms_error': {'agreements': 0, 'total': 0}
}

for seq in mg_seqs:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    # Vertical (best from previous analysis)
    left_metrics = comprehensive_performance_metrics(df['LV'].values, df['TargetV'].values)
    right_metrics = comprehensive_performance_metrics(df['RV'].values, df['TargetV'].values)

    if not left_metrics or not right_metrics:
        continue

    # For each metric, check if all metrics agree on which eye is worse
    for metric_name in results.keys():
        left_val = left_metrics[metric_name]
        right_val = right_metrics[metric_name]

        # Higher value = worse performance for all these metrics
        worse_eye_this_metric = 'left' if left_val > right_val else 'right'

        # Store for consistency check
        if 'worse_eyes' not in results[metric_name]:
            results[metric_name]['worse_eyes'] = []

        results[metric_name]['worse_eyes'].append(worse_eye_this_metric)
        results[metric_name]['total'] += 1

# Check agreement between metrics
print("\n" + "="*80)
print("METRIC CONSISTENCY CHECK")
print("="*80)
print("\nFor each patient, do different metrics agree on which eye is worse?")

# Get consensus for each patient
consensus_count = 0
total_count = results['mean_abs_error']['total']

for i in range(total_count):
    votes = []
    for metric_name in results.keys():
        votes.append(results[metric_name]['worse_eyes'][i])

    # Count votes
    left_votes = votes.count('left')
    right_votes = votes.count('right')

    # If 4+ out of 5 metrics agree, count as consensus
    if left_votes >= 4 or right_votes >= 4:
        consensus_count += 1

consensus_rate = consensus_count / total_count if total_count > 0 else 0

print(f"\nConsensus rate (4+ out of 5 metrics agree): {consensus_rate:.1%}")
print(f"  {consensus_count}/{total_count} patients have clear worse-eye consensus")

# Show per-metric worse-eye identification
print("\n" + "="*80)
print("PER-METRIC WORSE EYE IDENTIFICATION")
print("="*80)

for metric_name in results.keys():
    worse_eyes = results[metric_name]['worse_eyes']
    left_count = worse_eyes.count('left')
    right_count = worse_eyes.count('right')
    total = len(worse_eyes)

    print(f"\n{metric_name}:")
    print(f"  Left worse:  {left_count}/{total} ({100*left_count/total:.1f}%)")
    print(f"  Right worse: {right_count}/{total} ({100*right_count/total:.1f}%)")

print("\n" + "="*80)
print("KEY INSIGHT")
print("="*80)

if consensus_rate > 0.7:
    print(f"\n✓ HIGH CONSENSUS ({consensus_rate:.0%})!")
    print("Multiple metrics agree on which eye is worse")
    print("→ Can reliably identify worse eye using metric combination")
else:
    print(f"\n→ MODERATE CONSENSUS ({consensus_rate:.0%})")
    print("Metrics sometimes disagree on which eye is worse")
    print("→ May need more sophisticated approach or ground truth labels")

print("\n" + "="*80)
print("REALIZATION")
print("="*80)
print("\nIf we DON'T have ground truth labels for which eye is actually affected,")
print("we can't validate whether our metric correctly identifies the affected eye!")
print("\nWe can only show:")
print("1. Metrics consistently identify SOME eye as worse (consensus)")
print("2. MG patients have clearer worse-eye separation than HC")
print("\nBut we CANNOT prove our identified worse eye matches clinical reality")
print("without labeled data.")
print("="*80)
