#!/usr/bin/env python3
"""
H39: Z-Score Normalization Against HC Distribution

HYPOTHESIS: The "neither" gap exists because we use absolute thresholds (0.2°, 0.45°).
HC natural variation is ~0.5-1.0° asymmetry. MG overlap is where MG asymmetry < 1σ above HC mean.

SOLUTION: Normalize asymmetry scores as z-scores relative to HC distribution:
  z = (asymmetry - HC_mean) / HC_std

Expected behavior:
- HC z-scores centered at 0, most within ±2σ (95%)
- MG z-scores shifted positive, outliers beyond +2σ are clear MG
- "Neither" threshold at z < 1.0 (captures 84% of HC, rejects low-asymmetry MG)

Target achievement:
- HC 50-60% neither → z < 1.0 captures ~84% HC as neither (EXCEEDS target!)
- MG 10-20% neither → MG with z < 1.0 are subtle cases (SHOULD be ~15-20%)

This approach ADAPTS to HC natural variation instead of fighting it.
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
print("H39: Z-SCORE NORMALIZATION")
print("="*80)

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

def compute_h30_score(LV, RV, TargetV, sample_rate_hz=120):
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

    # H30 formula (best from 38 cycles)
    mad_asym = abs(met_L['mad'] - met_R['mad'])
    deg_asym = abs(met_L['deg'] - met_R['deg'])
    lat_asym = abs(met_L['lat'] - met_R['lat'])

    cv_asym = mad_asym / ((met_L['mad'] + met_R['mad']) / 2) if (met_L['mad'] + met_R['mad']) > 0 else 0

    score = 0.5 * (0.45*cv_asym + 0.55*deg_asym) + 0.5*(lat_asym / 100)
    return score

hc_scores = []
mg_scores = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)
    score = compute_h30_score(df['LV'].values, df['RV'].values, df['TargetV'].values)
    if score is not None:
        if seq['label'] == 0:
            hc_scores.append(score)
        else:
            mg_scores.append(score)

hc_scores = np.array(hc_scores)
mg_scores = np.array(mg_scores)

print(f"Valid: HC={len(hc_scores)}, MG={len(mg_scores)}\n")

# Compute HC distribution parameters
hc_mean = np.mean(hc_scores)
hc_std = np.std(hc_scores, ddof=1)

print("="*80)
print("HC DISTRIBUTION PARAMETERS")
print("="*80)
print(f"\nHC mean: {hc_mean:.3f}")
print(f"HC std: {hc_std:.3f}")
print(f"HC median: {np.median(hc_scores):.3f}")
print(f"HC 25th-75th percentile: {np.percentile(hc_scores, 25):.3f} - {np.percentile(hc_scores, 75):.3f}")

# Z-score normalization
hc_zscores = (hc_scores - hc_mean) / hc_std
mg_zscores = (mg_scores - hc_mean) / hc_std

print("\n" + "="*80)
print("Z-SCORE DISTRIBUTIONS")
print("="*80)
print(f"\nHC z-scores:")
print(f"  Mean: {np.mean(hc_zscores):.3f} (should be ~0)")
print(f"  Std: {np.std(hc_zscores, ddof=1):.3f} (should be ~1)")
print(f"  Within ±1σ: {100*(np.abs(hc_zscores) < 1).mean():.1f}% (expect 68%)")
print(f"  Within ±2σ: {100*(np.abs(hc_zscores) < 2).mean():.1f}% (expect 95%)")

print(f"\nMG z-scores:")
print(f"  Mean: {np.mean(mg_zscores):.3f}")
print(f"  Std: {np.std(mg_zscores, ddof=1):.3f}")
print(f"  Beyond +1σ: {100*(mg_zscores > 1).mean():.1f}%")
print(f"  Beyond +2σ: {100*(mg_zscores > 2).mean():.1f}%")

# Effect size on z-scores
pooled_std = np.sqrt(((len(hc_zscores)-1)*np.var(hc_zscores, ddof=1) + (len(mg_zscores)-1)*np.var(mg_zscores, ddof=1)) /
                      (len(hc_zscores) + len(mg_zscores) - 2))
d_z = (np.mean(mg_zscores) - np.mean(hc_zscores)) / pooled_std

print(f"\nCohen's d (z-scores): {d_z:.3f}")

# Test different z-score thresholds for "neither" classification
print("\n" + "="*80)
print("Z-SCORE THRESHOLD ANALYSIS")
print("="*80)

z_thresholds = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

print("\nThreshold | MG neither | HC neither | Gap from target (MG 15%, HC 55%)")
print("-" * 70)

for z_thresh in z_thresholds:
    mg_neither_pct = 100 * (mg_zscores < z_thresh).mean()
    hc_neither_pct = 100 * (hc_zscores < z_thresh).mean()

    gap = abs(mg_neither_pct - 15) + abs(hc_neither_pct - 55)

    marker = " ← BEST" if gap < 20 else ""
    print(f"z < {z_thresh:4.2f} | {mg_neither_pct:10.1f}% | {hc_neither_pct:10.1f}% | {gap:6.1f}{marker}")

# Find optimal z-threshold
best_z = None
best_gap_z = 999
for z in np.arange(0.3, 2.5, 0.1):
    mg_neither = 100 * (mg_zscores < z).mean()
    hc_neither = 100 * (hc_zscores < z).mean()
    gap = abs(mg_neither - 15) + abs(hc_neither - 55)
    if gap < best_gap_z:
        best_gap_z = gap
        best_z = z

mg_neither_best = 100 * (mg_zscores < best_z).mean()
hc_neither_best = 100 * (hc_zscores < best_z).mean()

print(f"\nOptimal z-threshold: {best_z:.2f}")
print(f"  MG neither: {mg_neither_best:.1f}% (target 10-20%)")
print(f"  HC neither: {hc_neither_best:.1f}% (target 50-60%)")
print(f"  Gap: {best_gap_z:.1f}")

print("\n" + "="*80)
print("COMPARISON TO H30/H38")
print("="*80)

print(f"\nH30 (raw scores, threshold=0.45°):")
print(f"  MG neither: 32.7%, HC neither: 54.6%, Gap: 18.1")

print(f"\nH38b (upweighted deg, threshold=0.45°):")
print(f"  MG neither: 30.9%, HC neither: 55.9%, Gap: 16.8")

print(f"\nH39 (z-score, threshold={best_z:.2f}σ):")
print(f"  MG neither: {mg_neither_best:.1f}%, HC neither: {hc_neither_best:.1f}%, Gap: {best_gap_z:.1f}")

improvement_vs_h30 = ((18.1 - best_gap_z) / 18.1) * 100
improvement_vs_h38 = ((16.8 - best_gap_z) / 16.8) * 100

print(f"\nImprovement:")
print(f"  vs H30: {improvement_vs_h30:+.1f}%")
print(f"  vs H38: {improvement_vs_h38:+.1f}%")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if best_gap_z < 15:
    print(f"\n✓ H39 ACHIEVES BREAKTHROUGH: Gap={best_gap_z:.1f} < 15")
    print(f"  Z-score normalization successfully adapts to HC natural variation")
    print(f"  MG {mg_neither_best:.1f}% neither (target 10-20%): {'✓ ACHIEVED' if 10 <= mg_neither_best <= 20 else '✗ CLOSE'}")
    print(f"  HC {hc_neither_best:.1f}% neither (target 50-60%): {'✓ ACHIEVED' if 50 <= hc_neither_best <= 60 else '✗ CLOSE'}")
    print("\n→ PROCEED to implement H39 as full metric")
elif best_gap_z < 17:
    print(f"\n⚠ H39 MARGINAL IMPROVEMENT: Gap={best_gap_z:.1f} (vs H38 gap=16.8)")
    print(f"  Z-score normalization helps but doesn't solve fundamental overlap")
    print("\n→ Consider combining z-score with other approaches")
else:
    print(f"\n✗ H39 NO IMPROVEMENT: Gap={best_gap_z:.1f} ≥ H38 gap=16.8")
    print(f"  Z-score normalization doesn't address root cause")
    print("\n→ Need fundamentally different approach (temporal dynamics, ensemble methods)")
