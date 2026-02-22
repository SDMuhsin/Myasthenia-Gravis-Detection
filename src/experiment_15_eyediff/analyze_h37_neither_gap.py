#!/usr/bin/env python3
"""
CYCLE 38 - PHASE 1: Analyze H37 "Neither" Gap

Problem: H37 shows 29.9% MG "neither" (target: 10-20%) and 43.2% HC "neither" (target: 50-60%)

Goals:
1. Understand WHY H37 has too many MG "neither" (false negatives)
2. Understand WHY H37 has too few HC "neither" (false positives)
3. Identify what separates detected asymmetry from missed asymmetry in MG
4. Find features that better discriminate true asymmetry from noise

Analysis:
- Stratify MG by "neither" vs "detected" - what differs?
- Stratify HC by "neither" vs "detected" - what causes false positives?
- Test if thresholds need adjustment or if metric needs fundamental change
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
print("CYCLE 38: ANALYZING H37 'NEITHER' GAP")
print("="*80)
print("\nLoading full dataset...")
raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

def compute_h37_components(LV, RV, TargetV, sample_rate_hz=120):
    """Compute H37 with detailed component breakdown"""
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

    # Component asymmetries
    mad_asym = abs(met_L['mad'] - met_R['mad'])
    deg_asym = abs(met_L['deg'] - met_R['deg'])
    lat_asym = abs(met_L['lat'] - met_R['lat'])

    # CV-normalized (H30 style)
    cv_asym = mad_asym / ((met_L['mad'] + met_R['mad']) / 2) if (met_L['mad'] + met_R['mad']) > 0 else 0

    # Per-eye composite scores
    score_L = 0.5 * (0.45 * met_L['mad'] + 0.55 * met_L['deg']) + 0.5 * (met_L['lat'] / 100)
    score_R = 0.5 * (0.45 * met_R['mad'] + 0.55 * met_R['deg']) + 0.5 * (met_R['lat'] / 100)

    signed_diff = score_L - score_R
    mag = abs(signed_diff)

    # H37 classification (0.2° threshold)
    if mag < 0.2:
        category = 'Neither'
    elif signed_diff > 0:
        category = 'Left'
    else:
        category = 'Right'

    return {
        'category': category,
        'magnitude': mag,
        'score_L': score_L,
        'score_R': score_R,
        'mad_L': met_L['mad'],
        'mad_R': met_R['mad'],
        'deg_L': met_L['deg'],
        'deg_R': met_R['deg'],
        'lat_L': met_L['lat'],
        'lat_R': met_R['lat'],
        'mad_asym': mad_asym,
        'deg_asym': deg_asym,
        'lat_asym': lat_asym,
        'cv_asym': cv_asym,
    }

hc_results = []
mg_results = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)
    result = compute_h37_components(df['LV'].values, df['RV'].values, df['TargetV'].values)
    if result is None:
        continue
    if seq['label'] == 0:
        hc_results.append(result)
    else:
        mg_results.append(result)

print(f"Valid: HC={len(hc_results)}, MG={len(mg_results)}\n")

# Stratify results
mg_neither = [r for r in mg_results if r['category'] == 'Neither']
mg_detected = [r for r in mg_results if r['category'] != 'Neither']
hc_neither = [r for r in hc_results if r['category'] == 'Neither']
hc_detected = [r for r in hc_results if r['category'] != 'Neither']

print("="*80)
print("CURRENT H37 PERFORMANCE")
print("="*80)
print(f"\nMG: {len(mg_neither)} ({100*len(mg_neither)/len(mg_results):.1f}%) Neither, "
      f"{len(mg_detected)} ({100*len(mg_detected)/len(mg_results):.1f}%) Detected")
print(f"  TARGET: 10-20% Neither (currently {100*len(mg_neither)/len(mg_results):.1f}%)")
print(f"  GAP: Need to detect {len(mg_neither) - int(0.15*len(mg_results))} more MG patients")

print(f"\nHC: {len(hc_neither)} ({100*len(hc_neither)/len(hc_results):.1f}%) Neither, "
      f"{len(hc_detected)} ({100*len(hc_detected)/len(hc_results):.1f}%) Detected")
print(f"  TARGET: 50-60% Neither (currently {100*len(hc_neither)/len(hc_results):.1f}%)")
print(f"  GAP: Need to reclassify {int(0.55*len(hc_results)) - len(hc_neither)} HC as Neither")

print("\n" + "="*80)
print("ANALYSIS 1: MG 'NEITHER' vs 'DETECTED'")
print("="*80)
print("\nWhat differs between MG patients we detect vs miss?")

def compare_groups(group1, group2, name1, name2):
    features = ['magnitude', 'mad_asym', 'deg_asym', 'lat_asym', 'cv_asym',
                'score_L', 'score_R', 'mad_L', 'mad_R', 'deg_L', 'deg_R', 'lat_L', 'lat_R']

    for feat in features:
        vals1 = np.array([r[feat] for r in group1])
        vals2 = np.array([r[feat] for r in group2])

        mean1, mean2 = np.mean(vals1), np.mean(vals2)
        ratio = mean2 / mean1 if mean1 > 0 else 0

        u, p = stats.mannwhitneyu(vals2, vals1, alternative='two-sided')

        if p < 0.01:
            print(f"  {feat}: {name1}={mean1:.3f}, {name2}={mean2:.3f}, "
                  f"ratio={ratio:.2f}x, p={p:.4f} ***")

compare_groups(mg_neither, mg_detected, "MG_Neither", "MG_Detected")

print("\n" + "="*80)
print("ANALYSIS 2: HC 'NEITHER' vs 'DETECTED' (False Positives)")
print("="*80)
print("\nWhat causes HC false positives (should be 'neither' but detected as asymmetric)?")

compare_groups(hc_neither, hc_detected, "HC_Neither", "HC_Detected")

print("\n" + "="*80)
print("ANALYSIS 3: MAGNITUDE DISTRIBUTION")
print("="*80)

mg_mag = np.array([r['magnitude'] for r in mg_results])
hc_mag = np.array([r['magnitude'] for r in hc_results])

print(f"\nMagnitude percentiles:")
print(f"  MG: 25th={np.percentile(mg_mag, 25):.3f}, 50th={np.percentile(mg_mag, 50):.3f}, "
      f"75th={np.percentile(mg_mag, 75):.3f}")
print(f"  HC: 25th={np.percentile(hc_mag, 25):.3f}, 50th={np.percentile(hc_mag, 50):.3f}, "
      f"75th={np.percentile(hc_mag, 75):.3f}")

print(f"\nCurrent threshold: 0.2° (neither if mag < 0.2)")
print(f"  Below 0.2: MG={100*(mg_mag < 0.2).mean():.1f}%, HC={100*(hc_mag < 0.2).mean():.1f}%")

# Find optimal threshold
thresholds = np.arange(0.05, 0.5, 0.05)
for thresh in thresholds:
    mg_neither_pct = 100 * (mg_mag < thresh).mean()
    hc_neither_pct = 100 * (hc_mag < thresh).mean()

    mg_gap = abs(mg_neither_pct - 15)  # Target 15% (midpoint of 10-20%)
    hc_gap = abs(hc_neither_pct - 55)  # Target 55% (midpoint of 50-60%)
    total_gap = mg_gap + hc_gap

    marker = " ← BEST" if total_gap < 30 else ""
    print(f"  Threshold={thresh:.2f}: MG={mg_neither_pct:.1f}% neither, "
          f"HC={hc_neither_pct:.1f}% neither, gap={total_gap:.1f}{marker}")

print("\n" + "="*80)
print("ANALYSIS 4: COMPONENT CONTRIBUTIONS")
print("="*80)
print("\nWhich components (MAD, Degradation, Latency) best separate MG from HC?")

mg_neither_arr = np.array([[r['mad_asym'], r['deg_asym'], r['lat_asym']] for r in mg_neither])
mg_detected_arr = np.array([[r['mad_asym'], r['deg_asym'], r['lat_asym']] for r in mg_detected])
hc_neither_arr = np.array([[r['mad_asym'], r['deg_asym'], r['lat_asym']] for r in hc_neither])
hc_detected_arr = np.array([[r['mad_asym'], r['deg_asym'], r['lat_asym']] for r in hc_detected])

components = ['MAD asymmetry', 'Degradation asymmetry', 'Latency asymmetry']
for i, comp in enumerate(components):
    mg_det = mg_detected_arr[:, i]
    hc_det = hc_detected_arr[:, i]

    pooled = np.sqrt(((len(hc_det)-1)*np.var(hc_det, ddof=1) + (len(mg_det)-1)*np.var(mg_det, ddof=1)) /
                      (len(hc_det) + len(mg_det) - 2))
    d = (np.mean(mg_det) - np.mean(hc_det)) / pooled if pooled > 0 else 0

    print(f"  {comp}: d={d:.3f}, MG_det={np.mean(mg_det):.3f}, HC_det={np.mean(hc_det):.3f}")

print("\n" + "="*80)
print("KEY INSIGHTS FOR CYCLE 38")
print("="*80)

print("\n1. THRESHOLD ADJUSTMENT:")
print("   - Current 0.2° threshold may be suboptimal")
print("   - Explore adaptive thresholds or score-based classification")

print("\n2. COMPONENT REWEIGHTING:")
print("   - Check if MAD/Deg/Lat weights (0.45/0.55 and 0.5 lat) are optimal")
print("   - Consider boosting high-discrimination components")

print("\n3. METRIC SENSITIVITY:")
print("   - H37 may be too conservative (composite score dampens signal)")
print("   - Try max(mad_asym, deg_asym, lat_asym) instead of weighted average")

print("\n4. NORMALIZATION:")
print("   - CV normalization helps but may not be enough")
print("   - Explore z-score normalization against HC distribution")

print("\nNext: Formulate H38 hypothesis based on these insights")
