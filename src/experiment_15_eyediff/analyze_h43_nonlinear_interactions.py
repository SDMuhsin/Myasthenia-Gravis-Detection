#!/usr/bin/env python3
"""
CYCLE 43 - PHASE 1: Nonlinear Interactions Analysis

HYPOTHESIS: Linear combinations (H30, H38b, FAT1) hit a ceiling because they assume
independent, additive contributions. MG pathophysiology may involve INTERACTIONS:
- High variability (MAD) amplifies fatigue (Deg) impact
- Slow latency combined with high degradation indicates severe dysfunction
- Multiplicative effects capture synergistic impairment

ANALYSIS:
1. Compute interaction terms: MAD × Deg, MAD × Lat, Deg × Lat
2. Test ratio metrics: Deg/MAD, Lat/MAD
3. Test polynomial features: MAD², Deg²
4. Measure which interactions show stronger MG vs HC discrimination

Expected: Interaction terms may capture complex dysfunction patterns missed by linear models.
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
print("CYCLE 43: NONLINEAR INTERACTIONS ANALYSIS")
print("="*80)

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

def compute_base_metrics(LV, RV, TargetV, sample_rate_hz=120):
    """Extract MAD, Deg, Lat for each eye separately"""
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

    return {
        'mad_L': met_L['mad'],
        'mad_R': met_R['mad'],
        'deg_L': met_L['deg'],
        'deg_R': met_R['deg'],
        'lat_L': met_L['lat'],
        'lat_R': met_R['lat'],
    }

hc_metrics = []
mg_metrics = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)
    metrics = compute_base_metrics(df['LV'].values, df['RV'].values, df['TargetV'].values)
    if metrics is not None:
        if seq['label'] == 0:
            hc_metrics.append(metrics)
        else:
            mg_metrics.append(metrics)

print(f"Valid: HC={len(hc_metrics)}, MG={len(mg_metrics)}\n")

# Convert to arrays for analysis
hc_mad_L = np.array([m['mad_L'] for m in hc_metrics])
hc_mad_R = np.array([m['mad_R'] for m in hc_metrics])
hc_deg_L = np.array([m['deg_L'] for m in hc_metrics])
hc_deg_R = np.array([m['deg_R'] for m in hc_metrics])
hc_lat_L = np.array([m['lat_L'] for m in hc_metrics])
hc_lat_R = np.array([m['lat_R'] for m in hc_metrics])

mg_mad_L = np.array([m['mad_L'] for m in mg_metrics])
mg_mad_R = np.array([m['mad_R'] for m in mg_metrics])
mg_deg_L = np.array([m['deg_L'] for m in mg_metrics])
mg_deg_R = np.array([m['deg_R'] for m in mg_metrics])
mg_lat_L = np.array([m['lat_L'] for m in mg_metrics])
mg_lat_R = np.array([m['lat_R'] for m in mg_metrics])

print("="*80)
print("INTERACTION TERMS ANALYSIS")
print("="*80)

def test_asymmetry_feature(hc_L, hc_R, mg_L, mg_R, name):
    """Compute asymmetry and test discrimination"""
    hc_asym = np.abs(hc_L - hc_R)
    mg_asym = np.abs(mg_L - mg_R)

    # Cohen's d
    pooled_std = np.sqrt(((len(hc_asym)-1)*np.var(hc_asym, ddof=1) +
                          (len(mg_asym)-1)*np.var(mg_asym, ddof=1)) /
                         (len(hc_asym) + len(mg_asym) - 2))
    d = (np.mean(mg_asym) - np.mean(hc_asym)) / pooled_std if pooled_std > 0 else 0

    print(f"\n{name}:")
    print(f"  HC: {np.mean(hc_asym):.4f} ± {np.std(hc_asym):.4f}")
    print(f"  MG: {np.mean(mg_asym):.4f} ± {np.std(mg_asym):.4f}")
    print(f"  Cohen's d: {d:.3f}")

    return d

print("\n--- BASELINE LINEAR FEATURES ---")
d_mad = test_asymmetry_feature(hc_mad_L, hc_mad_R, mg_mad_L, mg_mad_R, "MAD asymmetry")
d_deg = test_asymmetry_feature(hc_deg_L, hc_deg_R, mg_deg_L, mg_deg_R, "Deg asymmetry")
d_lat = test_asymmetry_feature(hc_lat_L, hc_lat_R, mg_lat_L, mg_lat_R, "Lat asymmetry")

print("\n--- INTERACTION TERMS ---")

# MAD × Deg interaction
hc_mad_deg_L = hc_mad_L * np.abs(hc_deg_L)
hc_mad_deg_R = hc_mad_R * np.abs(hc_deg_R)
mg_mad_deg_L = mg_mad_L * np.abs(mg_deg_L)
mg_mad_deg_R = mg_mad_R * np.abs(mg_deg_R)
d_mad_deg = test_asymmetry_feature(hc_mad_deg_L, hc_mad_deg_R, mg_mad_deg_L, mg_mad_deg_R, "MAD × Deg")

# MAD × Lat interaction
hc_mad_lat_L = hc_mad_L * hc_lat_L
hc_mad_lat_R = hc_mad_R * hc_lat_R
mg_mad_lat_L = mg_mad_L * mg_lat_L
mg_mad_lat_R = mg_mad_R * mg_lat_R
d_mad_lat = test_asymmetry_feature(hc_mad_lat_L, hc_mad_lat_R, mg_mad_lat_L, mg_mad_lat_R, "MAD × Lat")

# Deg × Lat interaction
hc_deg_lat_L = np.abs(hc_deg_L) * hc_lat_L
hc_deg_lat_R = np.abs(hc_deg_R) * hc_lat_R
mg_deg_lat_L = np.abs(mg_deg_L) * mg_lat_L
mg_deg_lat_R = np.abs(mg_deg_R) * mg_lat_R
d_deg_lat = test_asymmetry_feature(hc_deg_lat_L, hc_deg_lat_R, mg_deg_lat_L, mg_deg_lat_R, "Deg × Lat")

print("\n--- RATIO FEATURES ---")

# Deg / MAD ratio (fatigue relative to variability)
hc_deg_mad_ratio_L = np.abs(hc_deg_L) / (hc_mad_L + 0.01)  # Add epsilon to avoid /0
hc_deg_mad_ratio_R = np.abs(hc_deg_R) / (hc_mad_R + 0.01)
mg_deg_mad_ratio_L = np.abs(mg_deg_L) / (mg_mad_L + 0.01)
mg_deg_mad_ratio_R = np.abs(mg_deg_R) / (mg_mad_R + 0.01)
d_deg_mad_ratio = test_asymmetry_feature(hc_deg_mad_ratio_L, hc_deg_mad_ratio_R,
                                          mg_deg_mad_ratio_L, mg_deg_mad_ratio_R, "Deg/MAD ratio")

# Lat / MAD ratio (speed relative to variability)
hc_lat_mad_ratio_L = hc_lat_L / (hc_mad_L + 0.01)
hc_lat_mad_ratio_R = hc_lat_R / (hc_mad_R + 0.01)
mg_lat_mad_ratio_L = mg_lat_L / (mg_mad_L + 0.01)
mg_lat_mad_ratio_R = mg_lat_R / (mg_mad_R + 0.01)
d_lat_mad_ratio = test_asymmetry_feature(hc_lat_mad_ratio_L, hc_lat_mad_ratio_R,
                                          mg_lat_mad_ratio_L, mg_lat_mad_ratio_R, "Lat/MAD ratio")

print("\n--- POLYNOMIAL FEATURES ---")

# MAD squared
d_mad_sq = test_asymmetry_feature(hc_mad_L**2, hc_mad_R**2, mg_mad_L**2, mg_mad_R**2, "MAD²")

# Deg squared
d_deg_sq = test_asymmetry_feature(hc_deg_L**2, hc_deg_R**2, mg_deg_L**2, mg_deg_R**2, "Deg²")

# Lat squared
d_lat_sq = test_asymmetry_feature(hc_lat_L**2, hc_lat_R**2, mg_lat_L**2, mg_lat_R**2, "Lat²")

print("\n" + "="*80)
print("SUMMARY: BEST FEATURES BY COHEN'S D")
print("="*80)

features = {
    'MAD (baseline)': d_mad,
    'Deg (baseline)': d_deg,
    'Lat (baseline)': d_lat,
    'MAD × Deg': d_mad_deg,
    'MAD × Lat': d_mad_lat,
    'Deg × Lat': d_deg_lat,
    'Deg/MAD ratio': d_deg_mad_ratio,
    'Lat/MAD ratio': d_lat_mad_ratio,
    'MAD²': d_mad_sq,
    'Deg²': d_deg_sq,
    'Lat²': d_lat_sq,
}

ranked = sorted(features.items(), key=lambda x: x[1], reverse=True)

print("\nRanking:")
for i, (name, d) in enumerate(ranked, 1):
    marker = " ← BEST" if i == 1 else ""
    improvement = ((d - d_deg) / d_deg * 100) if d_deg > 0 else 0
    print(f"{i:2d}. {name:<20} d={d:.3f} ({improvement:+.1f}% vs Deg){marker}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

best_feature = ranked[0][0]
best_d = ranked[0][1]

if best_d > d_deg * 1.1:  # 10% improvement
    print(f"\n✓ {best_feature} shows >10% improvement (d={best_d:.3f} vs Deg d={d_deg:.3f})")
    print(f"  Nonlinear/interaction terms capture additional discriminative signal")
    print(f"\n→ PROCEED to test {best_feature} in H43 metric")
elif best_d > d_deg * 1.05:  # 5% improvement
    print(f"\n≈ {best_feature} shows marginal 5-10% improvement (d={best_d:.3f})")
    print(f"  Small gain but may be worth testing")
    print(f"\n→ CONSIDER testing {best_feature}")
else:
    print(f"\n✗ No substantial improvement from nonlinear features")
    print(f"  Best: {best_feature} with d={best_d:.3f} vs Deg d={d_deg:.3f}")
    print(f"  Linear combinations appear optimal for this dataset")
    print(f"\n→ Need fundamentally different approach beyond feature engineering")

print("\n" + "="*80)
print("DONE")
print("="*80)
