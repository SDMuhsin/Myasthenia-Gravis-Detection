#!/usr/bin/env python3
"""
H15: Optimize degradation metric
Test different time windows and aggregations for measuring fatigue
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
print("H15: Degradation Optimization")
print("="*80)

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

def compute_degradation_variant(eye_pos, target_pos, variant):
    """Test different degradation formulations"""
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    if np.sum(valid) < 50:
        return np.nan

    eye_clean = eye_pos[valid]
    target_clean = target_pos[valid]
    error = np.abs(eye_clean - target_clean)
    n = len(error)

    if variant == "10vs10":
        # First 10% vs last 10%
        early_n = int(n * 0.1)
        late_n = int(n * 0.1)
        return np.mean(error[-late_n:]) - np.mean(error[:early_n])

    elif variant == "20vs20":
        # First 20% vs last 20% (current H11)
        early_n = int(n * 0.2)
        late_n = int(n * 0.2)
        return np.mean(error[-late_n:]) - np.mean(error[:early_n])

    elif variant == "30vs30":
        # First 30% vs last 30%
        early_n = int(n * 0.3)
        late_n = int(n * 0.3)
        return np.mean(error[-late_n:]) - np.mean(error[:early_n])

    elif variant == "max_degradation":
        # Maximum degradation across sliding windows
        window_size = n // 5
        max_deg = -np.inf
        for i in range(5):
            start = i * window_size
            end = min((i+1) * window_size, n)
            window_error = np.mean(error[start:end])
            if i == 0:
                baseline = window_error
            else:
                deg = window_error - baseline
                max_deg = max(max_deg, deg)
        return max_deg

    elif variant == "cumulative":
        # Cumulative degradation (area under degradation curve)
        window_size = n // 10
        baseline = np.mean(error[:window_size])
        cumulative_deg = 0
        for i in range(1, 10):
            start = i * window_size
            end = min((i+1) * window_size, n)
            window_error = np.mean(error[start:end])
            cumulative_deg += (window_error - baseline)
        return cumulative_deg / 9

    return np.nan

# Test variants
variants = ["10vs10", "20vs20", "30vs30", "max_degradation", "cumulative"]
results = []

for variant in variants:
    deg_left = []
    deg_right = []
    class_labels = []

    for seq in sequences:
        df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

        left_deg = compute_degradation_variant(df['LV'].values, df['TargetV'].values, variant)
        right_deg = compute_degradation_variant(df['RV'].values, df['TargetV'].values, variant)

        if not (np.isnan(left_deg) or np.isnan(right_deg)):
            deg_left.append(left_deg)
            deg_right.append(right_deg)
            class_labels.append(1 if seq['class_name'] == 'MG' else 0)

    deg_left = np.array(deg_left)
    deg_right = np.array(deg_right)
    class_labels = np.array(class_labels)

    # Asymmetry
    asymmetry = np.abs(deg_left - deg_right)

    hc_asym = asymmetry[class_labels == 0]
    mg_asym = asymmetry[class_labels == 1]

    hc_mean = np.mean(hc_asym)
    hc_std = np.std(hc_asym)
    mg_mean = np.mean(mg_asym)
    mg_std = np.std(mg_asym)
    pooled_std = np.sqrt((hc_std**2 + mg_std**2) / 2)
    d = (mg_mean - hc_mean) / pooled_std

    _, p_mw = stats.mannwhitneyu(mg_asym, hc_asym, alternative='greater')
    _, p_hc = stats.wilcoxon(hc_asym)
    _, p_mg = stats.wilcoxon(mg_asym)

    val_score = (p_mw < 0.05) + (p_hc >= 0.05) + (p_mg < 0.05)

    results.append({
        'variant': variant,
        'd': d,
        'val_score': val_score,
        'p_hc': p_hc
    })

    print(f"\n{variant}:")
    print(f"  HC: {hc_mean:.4f} ± {hc_std:.4f}")
    print(f"  MG: {mg_mean:.4f} ± {mg_std:.4f}")
    print(f"  Cohen's d: {d:.4f}")
    print(f"  Validation: {val_score}/3")
    print(f"  HC≈0: p={p_hc:.6f} {'✓' if p_hc >= 0.05 else '✗'}")

best = max(results, key=lambda x: x['d'])

print("\n" + "="*80)
print("BEST VARIANT:")
print("="*80)
print(f"  {best['variant']}")
print(f"  Cohen's d: {best['d']:.4f}")
print(f"  Validation: {best['val_score']}/3")

if best['d'] >= 0.5 and best['val_score'] == 3:
    print("\n✓✓✓ SUCCESS!")
elif best['d'] > 0.41:
    print(f"\n✓ IMPROVEMENT over H11 (d=0.41 → d={best['d']:.2f})")
else:
    print(f"\n✗ No improvement (best d={best['d']:.2f} vs H11 d=0.41)")

print("="*80)
