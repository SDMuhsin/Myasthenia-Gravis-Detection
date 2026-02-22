#!/usr/bin/env python3
"""
CYCLE 40 - PHASE 1: Temporal Evolution of Asymmetry

HYPOTHESIS: Static metrics miss MG's key feature - FATIGABILITY.
MG asymmetry should INCREASE over time as neuromuscular junction fatigues.
HC asymmetry should remain CONSTANT (random noise, no fatigue).

ANALYSIS:
1. Split each sequence into 5 temporal windows (early → late)
2. Compute H30 asymmetry score in each window
3. Measure trend: Does asymmetry increase from early to late?
4. MG should show POSITIVE trends (increasing asymmetry)
5. HC should show NO trend (flat or random)

KEY INSIGHT: If MG shows increasing asymmetry trajectory but static total is low,
we might detect them by SLOPE rather than MAGNITUDE.

Example:
- HC: [0.3, 0.3, 0.3, 0.3, 0.3] → slope=0, mean=0.3
- MG subtle: [0.2, 0.3, 0.4, 0.5, 0.6] → slope=+0.1/window, mean=0.4
Both have low mean asymmetry, but MG shows clear PROGRESSION.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import linregress

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
print("CYCLE 40: TEMPORAL EVOLUTION OF ASYMMETRY")
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

def compute_temporal_trajectory(LV, RV, TargetV, n_windows=5):
    """
    Split sequence into n_windows temporal bins and compute H30 score in each.
    Returns: (scores_array, slope, mean_score)
    """
    # Remove NaNs
    valid = ~(np.isnan(LV) | np.isnan(RV) | np.isnan(TargetV))
    LV_clean = LV[valid]
    RV_clean = RV[valid]
    TargetV_clean = TargetV[valid]

    if len(LV_clean) < 200:  # Need enough data for splitting
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
        if score is None:
            return None
        scores.append(score)

    scores = np.array(scores)

    # Compute linear trend
    x = np.arange(n_windows)
    slope, intercept, r_value, p_value, std_err = linregress(x, scores)

    return {
        'scores': scores,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
    }

hc_trajectories = []
mg_trajectories = []

print("\nAnalyzing temporal trajectories...")
for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)
    result = compute_temporal_trajectory(df['LV'].values, df['RV'].values, df['TargetV'].values)
    if result is not None:
        if seq['label'] == 0:
            hc_trajectories.append(result)
        else:
            mg_trajectories.append(result)

print(f"Valid trajectories: HC={len(hc_trajectories)}, MG={len(mg_trajectories)}\n")

# Extract metrics
hc_slopes = np.array([t['slope'] for t in hc_trajectories])
mg_slopes = np.array([t['slope'] for t in mg_trajectories])

hc_means = np.array([t['mean_score'] for t in hc_trajectories])
mg_means = np.array([t['mean_score'] for t in mg_trajectories])

hc_stds = np.array([t['std_score'] for t in hc_trajectories])
mg_stds = np.array([t['std_score'] for t in mg_trajectories])

print("="*80)
print("TEMPORAL SLOPE ANALYSIS (asymmetry trend over time)")
print("="*80)

print(f"\nHC slopes (change in asymmetry per window):")
print(f"  Mean: {np.mean(hc_slopes):.4f}")
print(f"  Median: {np.median(hc_slopes):.4f}")
print(f"  Std: {np.std(hc_slopes):.4f}")
print(f"  Positive slopes: {100*(hc_slopes > 0).mean():.1f}%")
print(f"  Significant positive (p<0.05): {100*np.array([t['p_value'] < 0.05 and t['slope'] > 0 for t in hc_trajectories]).mean():.1f}%")

print(f"\nMG slopes (change in asymmetry per window):")
print(f"  Mean: {np.mean(mg_slopes):.4f}")
print(f"  Median: {np.median(mg_slopes):.4f}")
print(f"  Std: {np.std(mg_slopes):.4f}")
print(f"  Positive slopes: {100*(mg_slopes > 0).mean():.1f}%")
print(f"  Significant positive (p<0.05): {100*np.array([t['p_value'] < 0.05 and t['slope'] > 0 for t in mg_trajectories]).mean():.1f}%")

# Effect size on slopes
pooled_std_slope = np.sqrt(((len(hc_slopes)-1)*np.var(hc_slopes, ddof=1) + (len(mg_slopes)-1)*np.var(mg_slopes, ddof=1)) /
                            (len(hc_slopes) + len(mg_slopes) - 2))
d_slope = (np.mean(mg_slopes) - np.mean(hc_slopes)) / pooled_std_slope

print(f"\nCohen's d (slopes): {d_slope:.3f}")
u_slope, p_slope = stats.mannwhitneyu(mg_slopes, hc_slopes, alternative='greater')
print(f"Mann-Whitney U test (MG slopes > HC slopes): p={p_slope:.4f}")

print("\n" + "="*80)
print("COMPARISON TO STATIC MEAN ASYMMETRY")
print("="*80)

# Effect size on mean scores (static approach)
pooled_std_mean = np.sqrt(((len(hc_means)-1)*np.var(hc_means, ddof=1) + (len(mg_means)-1)*np.var(mg_means, ddof=1)) /
                          (len(hc_means) + len(mg_means) - 2))
d_mean = (np.mean(mg_means) - np.mean(hc_means)) / pooled_std_mean

print(f"\nStatic mean asymmetry:")
print(f"  HC: {np.mean(hc_means):.3f} ± {np.std(hc_means):.3f}")
print(f"  MG: {np.mean(mg_means):.3f} ± {np.std(mg_means):.3f}")
print(f"  Cohen's d: {d_mean:.3f}")

print(f"\nTemporal slope (rate of change):")
print(f"  HC: {np.mean(hc_slopes):.4f} ± {np.std(hc_slopes):.4f}")
print(f"  MG: {np.mean(mg_slopes):.4f} ± {np.std(mg_slopes):.4f}")
print(f"  Cohen's d: {d_slope:.3f}")

print("\n" + "="*80)
print("TEMPORAL VARIABILITY ANALYSIS")
print("="*80)

pooled_std_variability = np.sqrt(((len(hc_stds)-1)*np.var(hc_stds, ddof=1) + (len(mg_stds)-1)*np.var(mg_stds, ddof=1)) /
                                  (len(hc_stds) + len(mg_stds) - 2))
d_variability = (np.mean(mg_stds) - np.mean(hc_stds)) / pooled_std_variability

print(f"\nTemporal variability (std of scores across windows):")
print(f"  HC: {np.mean(hc_stds):.3f} ± {np.std(hc_stds):.3f}")
print(f"  MG: {np.mean(mg_stds):.3f} ± {np.std(mg_stds):.3f}")
print(f"  Cohen's d: {d_variability:.3f}")
print(f"  MG/HC ratio: {np.mean(mg_stds) / np.mean(hc_stds):.2f}x")

print("\n" + "="*80)
print("HYBRID METRIC: SLOPE + MEAN COMBINATION")
print("="*80)

# Test different combinations
print("\nTesting combined metrics: w*slope + (1-w)*mean")
print("Weight | Cohen's d | MG neither | HC neither | Gap")
print("-" * 60)

best_gap = 999
best_weight = None

for w in np.arange(0, 1.1, 0.1):
    # Normalize slope and mean to similar scales
    slope_norm = (mg_slopes - np.mean(hc_slopes)) / np.std(hc_slopes)
    mean_norm = (mg_means - np.mean(hc_means)) / np.std(hc_means)

    hc_slope_norm = (hc_slopes - np.mean(hc_slopes)) / np.std(hc_slopes)
    hc_mean_norm = (hc_means - np.mean(hc_means)) / np.std(hc_means)

    combined_mg = w * slope_norm + (1-w) * mean_norm
    combined_hc = w * hc_slope_norm + (1-w) * hc_mean_norm

    pooled = np.sqrt(((len(combined_hc)-1)*np.var(combined_hc, ddof=1) + (len(combined_mg)-1)*np.var(combined_mg, ddof=1)) /
                      (len(combined_hc) + len(combined_mg) - 2))
    d_combined = (np.mean(combined_mg) - np.mean(combined_hc)) / pooled

    # Find optimal threshold
    thresh_best = None
    gap_best = 999
    for thresh in np.percentile(combined_hc, np.arange(40, 70, 5)):
        mg_neither = 100 * (combined_mg < thresh).mean()
        hc_neither = 100 * (combined_hc < thresh).mean()
        gap = abs(mg_neither - 15) + abs(hc_neither - 55)
        if gap < gap_best:
            gap_best = gap
            thresh_best = thresh

    mg_neither_pct = 100 * (combined_mg < thresh_best).mean()
    hc_neither_pct = 100 * (combined_hc < thresh_best).mean()

    marker = " ← BEST" if gap_best < best_gap else ""
    if gap_best < best_gap:
        best_gap = gap_best
        best_weight = w

    print(f"{w:5.1f}  | {d_combined:9.3f} | {mg_neither_pct:10.1f}% | {hc_neither_pct:10.1f}% | {gap_best:6.1f}{marker}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

if d_slope > 0.3:
    print(f"\n✓ MG shows POSITIVE temporal slope (d={d_slope:.3f})")
    print(f"  Asymmetry increases over time due to fatigue")
else:
    print(f"\n✗ No significant temporal trend (d={d_slope:.3f})")
    print(f"  Fatigability hypothesis not supported")

if d_variability > 0.3:
    print(f"\n✓ MG shows higher temporal variability (d={d_variability:.3f})")
    print(f"  Asymmetry fluctuates more in MG than HC")
else:
    print(f"\n✗ No significant variability difference (d={d_variability:.3f})")

print(f"\nBest hybrid metric: {best_weight:.1f}*slope + {1-best_weight:.1f}*mean")
print(f"  Gap: {best_gap:.1f}")
print(f"  Compare to H38 gap: 16.8")

improvement = ((16.8 - best_gap) / 16.8) * 100 if best_gap < 16.8 else -((best_gap - 16.8) / 16.8) * 100

if best_gap < 16.8:
    print(f"\n→ IMPROVEMENT: {improvement:.1f}% better than H38")
    print(f"→ PROCEED to Phase 2: Formulate H40 with temporal dynamics")
else:
    print(f"\n→ NO IMPROVEMENT: {improvement:.1f}% worse than H38")
    print(f"→ Temporal dynamics do not solve 'neither' gap problem")
