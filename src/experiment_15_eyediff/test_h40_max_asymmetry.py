#!/usr/bin/env python3
"""
H40: Maximum Asymmetry Across Temporal Windows

HYPOTHESIS: Instead of averaging asymmetry over entire sequence,
capture the PEAK asymmetry across multiple temporal windows.

RATIONALE:
- MG has intermittent dysfunction (variability d=0.467)
- Taking MAX captures "worst moment" of asymmetry
- HC fluctuations are noise → low max values
- MG dysfunction moments → high max values

Formula:
- Split sequence into 3 non-overlapping windows (early, mid, late)
- Compute H30 score in each window
- score_final = MAX(score_early, score_mid, score_late)

Expected: This should reduce HC "neither" gap by ignoring HC's random fluctuations
while capturing MG's peak dysfunction moments.
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
print("H40: MAXIMUM ASYMMETRY ACROSS TEMPORAL WINDOWS")
print("="*80)

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

def compute_h30_score(LV, RV, TargetV, sample_rate_hz=120):
    """H30 asymmetry formula"""
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
    mad_asym = abs(met_L['mad'] - met_R['mad'])
    deg_asym = abs(met_L['deg'] - met_R['deg'])
    lat_asym = abs(met_L['lat'] - met_R['lat'])

    cv_asym = mad_asym / ((met_L['mad'] + met_R['mad']) / 2) if (met_L['mad'] + met_R['mad']) > 0 else 0

    score = 0.5 * (0.45*cv_asym + 0.55*deg_asym) + 0.5*(lat_asym / 100)
    return score

def compute_max_asymmetry(LV, RV, TargetV, n_windows=3):
    """
    Compute H30 in n_windows and return MAXIMUM.
    Also return mean for comparison.
    """
    valid = ~(np.isnan(LV) | np.isnan(RV) | np.isnan(TargetV))
    LV_clean = LV[valid]
    RV_clean = RV[valid]
    TargetV_clean = TargetV[valid]

    if len(LV_clean) < 150:  # Need enough data
        return None

    window_size = len(LV_clean) // n_windows
    scores = []

    for i in range(n_windows):
        start = i * window_size
        end = start + window_size if i < n_windows - 1 else len(LV_clean)

        LV_win = LV_clean[start:end]
        RV_win = RV_clean[start:end]
        TargetV_win = TargetV_clean[start:end]

        score = compute_h30_score(LV_win, RV_win, TargetV_win)
        if score is not None:
            scores.append(score)

    if len(scores) == 0:
        return None

    return {
        'max': max(scores),
        'mean': np.mean(scores),
        'min': min(scores),
        'range': max(scores) - min(scores),
    }

hc_results = []
mg_results = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)
    result = compute_max_asymmetry(df['LV'].values, df['RV'].values, df['TargetV'].values)
    if result is not None:
        if seq['label'] == 0:
            hc_results.append(result)
        else:
            mg_results.append(result)

print(f"Valid: HC={len(hc_results)}, MG={len(mg_results)}\\n")

hc_max = np.array([r['max'] for r in hc_results])
mg_max = np.array([r['max'] for r in mg_results])

hc_mean = np.array([r['mean'] for r in hc_results])
mg_mean = np.array([r['mean'] for r in mg_results])

hc_range = np.array([r['range'] for r in hc_results])
mg_range = np.array([r['range'] for r in mg_results])

print("="*80)
print("METRIC COMPARISON")
print("="*80)

# MAX vs MEAN effect sizes
pooled_max = np.sqrt(((len(hc_max)-1)*np.var(hc_max, ddof=1) + (len(mg_max)-1)*np.var(mg_max, ddof=1)) /
                      (len(hc_max) + len(mg_max) - 2))
d_max = (np.mean(mg_max) - np.mean(hc_max)) / pooled_max

pooled_mean = np.sqrt(((len(hc_mean)-1)*np.var(hc_mean, ddof=1) + (len(mg_mean)-1)*np.var(mg_mean, ddof=1)) /
                       (len(hc_mean) + len(mg_mean) - 2))
d_mean = (np.mean(mg_mean) - np.mean(hc_mean)) / pooled_mean

pooled_range = np.sqrt(((len(hc_range)-1)*np.var(hc_range, ddof=1) + (len(mg_range)-1)*np.var(mg_range, ddof=1)) /
                        (len(hc_range) + len(mg_range) - 2))
d_range = (np.mean(mg_range) - np.mean(hc_range)) / pooled_range

print(f"\\nMAX asymmetry:")
print(f"  HC: {np.mean(hc_max):.3f} ± {np.std(hc_max):.3f}")
print(f"  MG: {np.mean(mg_max):.3f} ± {np.std(mg_max):.3f}")
print(f"  Cohen's d: {d_max:.3f}")

print(f"\\nMEAN asymmetry:")
print(f"  HC: {np.mean(hc_mean):.3f} ± {np.std(hc_mean):.3f}")
print(f"  MG: {np.mean(mg_mean):.3f} ± {np.std(mg_mean):.3f}")
print(f"  Cohen's d: {d_mean:.3f}")

print(f"\\nRANGE (max - min):")
print(f"  HC: {np.mean(hc_range):.3f} ± {np.std(hc_range):.3f}")
print(f"  MG: {np.mean(mg_range):.3f} ± {np.std(mg_range):.3f}")
print(f"  Cohen's d: {d_range:.3f}")

print("\\n" + "="*80)
print("THRESHOLD OPTIMIZATION: MAX vs MEAN")
print("="*80)

def find_best_threshold(scores_hc, scores_mg, name):
    best_thresh = None
    best_gap = 999
    for thresh in np.arange(0.2, 1.0, 0.05):
        mg_neither = 100 * (scores_mg < thresh).mean()
        hc_neither = 100 * (scores_hc < thresh).mean()
        gap = abs(mg_neither - 15) + abs(hc_neither - 55)
        if gap < best_gap:
            best_gap = gap
            best_thresh = thresh

    mg_neither_pct = 100 * (scores_mg < best_thresh).mean()
    hc_neither_pct = 100 * (scores_hc < best_thresh).mean()

    print(f"\\n{name}:")
    print(f"  Best threshold: {best_thresh:.2f}°")
    print(f"  MG neither: {mg_neither_pct:.1f}% (target 10-20%)")
    print(f"  HC neither: {hc_neither_pct:.1f}% (target 50-60%)")
    print(f"  Gap: {best_gap:.1f}")
    return best_gap

gap_max = find_best_threshold(hc_max, mg_max, "MAX asymmetry")
gap_mean = find_best_threshold(hc_mean, mg_mean, "MEAN asymmetry")
gap_range = find_best_threshold(hc_range, mg_range, "RANGE (variability)")

print("\\n" + "="*80)
print("CONCLUSION")
print("="*80)

print(f"\\nH30 (full sequence): gap=18.1")
print(f"H38 (degradation upweight): gap=16.8")
print(f"H40a (MAX across 3 windows): gap={gap_max:.1f}")
print(f"H40b (MEAN across 3 windows): gap={gap_mean:.1f}")
print(f"H40c (RANGE across 3 windows): gap={gap_range:.1f}")

if gap_max < 16.8:
    improvement = ((16.8 - gap_max) / 16.8) * 100
    print(f"\\n✓ H40a IMPROVEMENT: {improvement:.1f}% better than H38")
    print(f"  Cohen's d: {d_max:.3f}")
    print(f"\\n→ PROCEED to Phase 2: Formulate H40 with MAX operator")
else:
    print(f"\\n✗ H40a NO IMPROVEMENT: gap={gap_max:.1f} vs H38 gap=16.8")
    print(f"  MAX operator doesn't solve 'neither' problem")
    print(f"\\n→ Consider accepting H38 as realistic ceiling")
