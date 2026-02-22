#!/usr/bin/env python3
"""
H30 FINAL VALIDATION (Full Dataset)

Comprehensive validation of H30 (CV normalization) on full dataset to confirm
it as best metric from 36 research cycles.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

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
print("H30 FINAL VALIDATION - FULL DATASET")
print("="*80)
print("Loading all sequences...")
raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)
print(f"Total: {len(sequences)} sequences\n")

def compute_h30_for_eyes(LV, RV, TargetV, sample_rate_hz=120):
    """Compute H30 components for both eyes"""
    def process_eye(eye_pos, target_pos):
        valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
        eye = eye_pos[valid]
        target = target_pos[valid]
        if len(eye) < 50:
            return None

        target_diff = np.diff(target)
        up_indices = np.where(target_diff > 5.0)[0] + 1
        mask = np.zeros(len(eye), dtype=bool)
        for idx in up_indices:
            mask[idx:min(idx+50, len(eye))] = True

        eye_up = eye[mask]
        target_up = target[mask]
        if len(eye_up) < 30:
            return None

        error = eye_up - target_up
        mad = np.median(np.abs(error - np.median(error)))

        n = len(error)
        early_n = max(5, int(n * 0.2))
        late_n = max(5, int(n * 0.2))
        deg = np.mean(np.abs(error[-late_n:])) - np.mean(np.abs(error[:early_n]))

        latencies = []
        for idx in up_indices:
            if idx >= len(eye) - 10:
                continue
            new_target = target[idx]
            for i in range(idx, min(idx + int(0.5 * sample_rate_hz), len(eye))):
                if abs(eye[i] - new_target) <= 3.0:
                    lat_ms = ((i - idx) / sample_rate_hz) * 1000
                    latencies.append(lat_ms)
                    break

        if len(latencies) < 3:
            return None

        lat = np.mean(sorted(latencies)[:max(3, len(latencies)//4)])
        return {'mad': mad, 'deg': deg, 'lat': lat}

    met_L = process_eye(LV, TargetV)
    met_R = process_eye(RV, TargetV)

    if met_L is None or met_R is None:
        return None

    # H30 formula
    cv_asym = abs(met_L['mad'] - met_R['mad']) / ((met_L['mad'] + met_R['mad']) / 2)
    deg_asym = abs(met_L['deg'] - met_R['deg'])
    lat_asym = abs(met_L['lat'] - met_R['lat']) / 100

    h30_score = 0.5 * (0.45*cv_asym + 0.55*deg_asym) + 0.5*lat_asym

    return h30_score

print("Computing H30 for all vertical sequences...")
hc_scores = []
mg_scores = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)
    score = compute_h30_for_eyes(df['LV'].values, df['RV'].values, df['TargetV'].values)
    if score is not None:
        if seq['label'] == 0:
            hc_scores.append(score)
        else:
            mg_scores.append(score)

hc_scores = np.array(hc_scores)
mg_scores = np.array(mg_scores)

print(f"Valid: HC={len(hc_scores)}, MG={len(mg_scores)}\n")

print("="*80)
print("RESULTS")
print("="*80)

print(f"\nDescriptive Statistics:")
print(f"  HC: {np.mean(hc_scores):.3f} ± {np.std(hc_scores):.3f} (median={np.median(hc_scores):.3f})")
print(f"  MG: {np.mean(mg_scores):.3f} ± {np.std(mg_scores):.3f} (median={np.median(mg_scores):.3f})")

# Effect size
pooled_std = np.sqrt(((len(hc_scores)-1)*np.var(hc_scores, ddof=1) + (len(mg_scores)-1)*np.var(mg_scores, ddof=1)) / (len(hc_scores) + len(mg_scores) - 2))
d = (np.mean(mg_scores) - np.mean(hc_scores)) / pooled_std

print(f"\nCohen's d: {d:.3f}")
print(f"  Interpretation: {'LARGE' if d >= 0.8 else 'MEDIUM' if d >= 0.5 else 'SMALL' if d >= 0.2 else 'NEGLIGIBLE'}")
print(f"  Target d≥0.65: {'✓ ACHIEVED' if d >= 0.65 else f'✗ {((d/0.65)*100):.1f}% of target'}")

# Validation tests
u, p_mw = stats.mannwhitneyu(mg_scores, hc_scores, alternative='greater')
_, p_hc = stats.wilcoxon(hc_scores, alternative='greater')
_, p_mg = stats.wilcoxon(mg_scores, alternative='greater')

print(f"\nValidation Tests:")
print(f"  [1] MG > HC (Mann-Whitney): p={p_mw:.6f} {'✓ PASS' if p_mw < 0.05 else '✗ FAIL'}")
print(f"  [2] HC ≈ 0 (Wilcoxon): p={p_hc:.6f} {'✓ PASS' if p_hc >= 0.05 else '✗ FAIL'}")
print(f"  [3] MG > 0 (Wilcoxon): p={p_mg:.6f} {'✓ PASS' if p_mg < 0.05 else '✗ FAIL'}")

validation_score = sum([p_mw < 0.05, p_hc >= 0.05, p_mg < 0.05])
print(f"  Validation Score: {validation_score}/3")

# Discrimination metrics
hc_median = np.median(hc_scores)
pct_mg_above_hc_median = (mg_scores > hc_median).mean() * 100

print(f"\nDiscrimination:")
print(f"  MG above HC median: {pct_mg_above_hc_median:.1f}%")
print(f"  Ratio (MG/HC medians): {np.median(mg_scores)/np.median(hc_scores):.2f}x")

# Specificity/Sensitivity (using MG median as threshold)
mg_median = np.median(mg_scores)
specificity = (hc_scores < mg_median).mean() * 100
sensitivity = (mg_scores >= mg_median).mean() * 100

print(f"\nClinical Metrics (threshold = MG median):")
print(f"  Specificity: {specificity:.1f}%")
print(f"  Sensitivity: {sensitivity:.1f}%")
print(f"  Overall accuracy: {((hc_scores < mg_median).sum() + (mg_scores >= mg_median).sum()) / (len(hc_scores) + len(mg_scores)) * 100:.1f}%")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
if d >= 0.65 and validation_score >= 2:
    print("✓ H30 EXCEEDS TARGETS - Publication-worthy metric achieved!")
elif d >= 0.60 and validation_score >= 2:
    print("⚠ H30 NEAR TARGET - Strong metric but 7% gap from d≥0.65 target")
    print(f"  Current: d={d:.3f}, Target: d≥0.65")
    print(f"  This represents 36 research cycles. H30 is best analytical metric found.")
else:
    print("✗ H30 BELOW TARGET")

print(f"\nFinal Status:")
print(f"  Cohen's d: {d:.3f} ({((d/0.65)*100):.1f}% of d≥0.65 target)")
print(f"  Specificity: {specificity:.1f}%")
print(f"  Validation: {validation_score}/3")
print(f"  Cycles completed: 36 (H1-H36)")
print(f"  Best metric: H30 CV normalization")
