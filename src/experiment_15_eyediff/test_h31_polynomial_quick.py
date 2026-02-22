#!/usr/bin/env python3
"""
H31 Quick Empirical Validation: Polynomial Features

Test whether polynomial/product combinations improve over H30's linear combination.
This is Phase 4 (pre-implementation empirical analysis) from research protocol.
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

print("Loading sequences...")
raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)
print(f"Total sequences: {len(sequences)}\n")

def compute_upward_metrics(eye_pos, target_pos):
    """Extract upward saccade data and compute MAD, degradation"""
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    eye_clean = eye_pos[valid]
    target_clean = target_pos[valid]

    if len(eye_clean) < 50:
        return None

    target_diff = np.diff(target_clean)
    up_indices = np.where(target_diff > 5.0)[0] + 1
    mask = np.zeros(len(eye_clean), dtype=bool)
    for idx in up_indices:
        mask[idx:min(idx+50, len(eye_clean))] = True

    eye_upward = eye_clean[mask]
    target_upward = target_clean[mask]

    if len(eye_upward) < 30:
        return None

    error = eye_upward - target_upward

    # MAD
    mad = np.median(np.abs(error - np.median(error)))

    # Degradation
    n = len(error)
    early_n = max(5, int(n * 0.2))
    late_n = max(5, int(n * 0.2))
    degradation = np.mean(np.abs(error[-late_n:])) - np.mean(np.abs(error[:early_n]))

    return {'mad': mad, 'degradation': degradation}

def compute_latencies_upward(eye_pos, target_pos, sample_rate_hz=120):
    """Compute saccade latencies for upward jumps"""
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    eye = eye_pos[valid]
    target = target_pos[valid]

    if len(target) < 20:
        return []

    target_diff = np.diff(target)
    jump_indices = np.where(target_diff > 5.0)[0] + 1

    latencies_ms = []

    for jump_idx in jump_indices:
        if jump_idx >= len(target) - 10:
            continue

        new_target = target[jump_idx]
        threshold_deg = 3.0

        for i in range(jump_idx, min(jump_idx + int(0.5 * sample_rate_hz), len(eye))):
            if abs(eye[i] - new_target) <= threshold_deg:
                latency_samples = i - jump_idx
                latency_ms = (latency_samples / sample_rate_hz) * 1000
                latencies_ms.append(latency_ms)
                break

    return latencies_ms

# Compute components for all vertical sequences
print("Computing H30 components for vertical sequences...")
hc_components = []
mg_components = []

for seq in sequences:
    # Create dataframe from raw data
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    LV = df['LV'].values
    RV = df['RV'].values
    TargetV = df['TargetV'].values

    metrics_L = compute_upward_metrics(LV, TargetV)
    metrics_R = compute_upward_metrics(RV, TargetV)

    if metrics_L is None or metrics_R is None:
        continue

    latencies_L = compute_latencies_upward(LV, TargetV)
    latencies_R = compute_latencies_upward(RV, TargetV)

    if len(latencies_L) < 3 or len(latencies_R) < 3:
        continue

    # H30-style components
    mad_L = metrics_L['mad']
    mad_R = metrics_R['mad']
    deg_L = metrics_L['degradation']
    deg_R = metrics_R['degradation']
    lat_L = np.mean(sorted(latencies_L)[:max(3, len(latencies_L)//4)])
    lat_R = np.mean(sorted(latencies_R)[:max(3, len(latencies_R)//4)])

    # CV normalization for MAD
    cv_asym = abs(mad_L - mad_R) / ((mad_L + mad_R) / 2)
    deg_asym = abs(deg_L - deg_R)
    lat_asym = abs(lat_L - lat_R) / 100

    components = {
        'cv_asym': cv_asym,
        'deg_asym': deg_asym,
        'lat_asym': lat_asym,
    }

    if seq['label'] == 0:  # HC
        hc_components.append(components)
    else:  # MG
        mg_components.append(components)

# Convert to arrays
hc_mad = np.array([c['cv_asym'] for c in hc_components])
hc_deg = np.array([c['deg_asym'] for c in hc_components])
hc_lat = np.array([c['lat_asym'] for c in hc_components])
mg_mad = np.array([c['cv_asym'] for c in mg_components])
mg_deg = np.array([c['deg_asym'] for c in mg_components])
mg_lat = np.array([c['lat_asym'] for c in mg_components])

print(f"Valid samples: HC={len(hc_mad)}, MG={len(mg_mad)}\n")

print("="*80)
print("H31 EMPIRICAL VALIDATION: POLYNOMIAL FEATURES")
print("="*80)

# Q1: Quadratic amplification
print("\n1. QUADRATIC AMPLIFICATION TEST:")
hc_mad_linear = np.median(hc_mad)
mg_mad_linear = np.median(mg_mad)
linear_ratio = mg_mad_linear / hc_mad_linear

hc_mad_sq = np.median(hc_mad**2)
mg_mad_sq = np.median(mg_mad**2)
quadratic_ratio = mg_mad_sq / hc_mad_sq

print(f"   Linear CV_asym: HC={hc_mad_linear:.3f}, MG={mg_mad_linear:.3f}, ratio={linear_ratio:.2f}x")
print(f"   Squared CV_asym²: HC={hc_mad_sq:.3f}, MG={mg_mad_sq:.3f}, ratio={quadratic_ratio:.2f}x")
print(f"   Amplification: {quadratic_ratio/linear_ratio:.2f}x")
print(f"   → {'✓ AMPLIFIES' if quadratic_ratio > linear_ratio*1.1 else '✗ NO AMPLIFICATION'} (need >10% increase)")

# Q2: Product term
print("\n2. PRODUCT TERM (CV_asym × Deg_asym) TEST:")
product_hc = hc_mad * hc_deg
product_mg = mg_mad * mg_deg
product_ratio = np.median(product_mg) / np.median(product_hc)
deg_ratio = np.median(mg_deg) / np.median(hc_deg)

print(f"   Product: HC={np.median(product_hc):.3f}, MG={np.median(product_mg):.3f}, ratio={product_ratio:.2f}x")
print(f"   Linear CV_asym: ratio={linear_ratio:.2f}x")
print(f"   Linear Deg_asym: ratio={deg_ratio:.2f}x")
print(f"   → Product is {'STRONGER' if product_ratio > max(linear_ratio, deg_ratio) else 'WEAKER'} than best linear")

# Q3: Numerical stability
print("\n3. NUMERICAL STABILITY:")
print(f"   CV_asym² max/min: HC={hc_mad.max()**2 / max(hc_mad.min()**2, 0.001):.1f}x")
print(f"   Product max/min: HC={product_hc.max() / max(product_hc.min(), 0.001):.1f}x")
print(f"   → {'⚠ WARNING' if product_hc.max() / max(product_hc.min(), 0.001) > 100 else '✓ OK'}")

# Q4: Outlier amplification
print("\n4. OUTLIER SENSITIVITY:")
hc_mad_sq_90p = np.percentile(hc_mad**2, 90)
hc_mad_sq_median = np.median(hc_mad**2)
print(f"   HC CV_asym² 90th/median: {hc_mad_sq_90p / hc_mad_sq_median:.2f}x")
print(f"   → {'⚠ HIGH' if hc_mad_sq_90p / hc_mad_sq_median > 3 else '✓ MODERATE'} outlier amplification")

# Q5: Effect size estimates
print("\n5. EFFECT SIZE ESTIMATES:")
print("   (Normalizing components to same scale for fair comparison)\n")

# Normalize to z-scores using HC distribution
hc_mad_norm = (hc_mad - np.mean(hc_mad)) / np.std(hc_mad)
hc_deg_norm = (hc_deg - np.mean(hc_deg)) / np.std(hc_deg)
hc_lat_norm = (hc_lat - np.mean(hc_lat)) / np.std(hc_lat)
mg_mad_norm = (mg_mad - np.mean(hc_mad)) / np.std(hc_mad)
mg_deg_norm = (mg_deg - np.mean(hc_deg)) / np.std(hc_deg)
mg_lat_norm = (mg_lat - np.mean(hc_lat)) / np.std(hc_lat)

# H30 baseline (linear only)
h30_hc = 0.5 * (0.45*hc_mad_norm + 0.55*hc_deg_norm) + 0.5*hc_lat_norm
h30_mg = 0.5 * (0.45*mg_mad_norm + 0.55*mg_deg_norm) + 0.5*mg_lat_norm
pooled_std_30 = np.sqrt(((len(h30_hc)-1)*np.std(h30_hc, ddof=1)**2 + (len(h30_mg)-1)*np.std(h30_mg, ddof=1)**2) / (len(h30_hc) + len(h30_mg) - 2))
d_h30 = (np.mean(h30_mg) - np.mean(h30_hc)) / pooled_std_30
u_30, p_30 = stats.mannwhitneyu(h30_mg, h30_hc, alternative='greater')
print(f"   H30 (baseline linear): d={d_h30:.3f}, p={p_30:.4f}")

# H31a: Add all polynomial terms (equal weight)
h31a_hc = (hc_mad_norm + hc_deg_norm + hc_lat_norm + hc_mad_norm**2 + hc_deg_norm**2 + hc_mad_norm*hc_deg_norm) / 6
h31a_mg = (mg_mad_norm + mg_deg_norm + mg_lat_norm + mg_mad_norm**2 + mg_deg_norm**2 + mg_mad_norm*mg_deg_norm) / 6
pooled_std_a = np.sqrt(((len(h31a_hc)-1)*np.std(h31a_hc, ddof=1)**2 + (len(h31a_mg)-1)*np.std(h31a_mg, ddof=1)**2) / (len(h31a_hc) + len(h31a_mg) - 2))
d_h31a = (np.mean(h31a_mg) - np.mean(h31a_hc)) / pooled_std_a
u_a, p_a = stats.mannwhitneyu(h31a_mg, h31a_hc, alternative='greater')
print(f"   H31a (equal-weight poly): d={d_h31a:.3f}, p={p_a:.4f} ({(d_h31a/d_h30 - 1)*100:+.1f}% vs H30)")

# H31b: Product-emphasized
h31b_hc = 0.4*(hc_mad_norm*hc_deg_norm) + 0.3*hc_mad_norm + 0.3*hc_deg_norm
h31b_mg = 0.4*(mg_mad_norm*mg_deg_norm) + 0.3*mg_mad_norm + 0.3*mg_deg_norm
pooled_std_b = np.sqrt(((len(h31b_hc)-1)*np.std(h31b_hc, ddof=1)**2 + (len(h31b_mg)-1)*np.std(h31b_mg, ddof=1)**2) / (len(h31b_hc) + len(h31b_mg) - 2))
d_h31b = (np.mean(h31b_mg) - np.mean(h31b_hc)) / pooled_std_b
u_b, p_b = stats.mannwhitneyu(h31b_mg, h31b_hc, alternative='greater')
print(f"   H31b (product-emphasis): d={d_h31b:.3f}, p={p_b:.4f} ({(d_h31b/d_h30 - 1)*100:+.1f}% vs H30)")

# H31c: Quadratic-only
h31c_hc = 0.5*hc_mad_norm**2 + 0.5*hc_deg_norm**2
h31c_mg = 0.5*mg_mad_norm**2 + 0.5*mg_deg_norm**2
pooled_std_c = np.sqrt(((len(h31c_hc)-1)*np.std(h31c_hc, ddof=1)**2 + (len(h31c_mg)-1)*np.std(h31c_mg, ddof=1)**2) / (len(h31c_hc) + len(h31c_mg) - 2))
d_h31c = (np.mean(h31c_mg) - np.mean(h31c_hc)) / pooled_std_c
u_c, p_c = stats.mannwhitneyu(h31c_mg, h31c_hc, alternative='greater')
print(f"   H31c (quadratic-only): d={d_h31c:.3f}, p={p_c:.4f} ({(d_h31c/d_h30 - 1)*100:+.1f}% vs H30)")

# H31d: Add just product term to H30
h31d_hc = h30_hc + 0.2*(hc_mad_norm*hc_deg_norm)
h31d_mg = h30_mg + 0.2*(mg_mad_norm*mg_deg_norm)
pooled_std_d = np.sqrt(((len(h31d_hc)-1)*np.std(h31d_hc, ddof=1)**2 + (len(h31d_mg)-1)*np.std(h31d_mg, ddof=1)**2) / (len(h31d_hc) + len(h31d_mg) - 2))
d_h31d = (np.mean(h31d_mg) - np.mean(h31d_hc)) / pooled_std_d
u_d, p_d = stats.mannwhitneyu(h31d_mg, h31d_hc, alternative='greater')
print(f"   H31d (H30 + product): d={d_h31d:.3f}, p={p_d:.4f} ({(d_h31d/d_h30 - 1)*100:+.1f}% vs H30)")

print("\n" + "="*80)
print("DECISION RECOMMENDATION")
print("="*80)

best_d = max(d_h31a, d_h31b, d_h31c, d_h31d)
best_name = ['H31a', 'H31b', 'H31c', 'H31d'][[d_h31a, d_h31b, d_h31c, d_h31d].index(best_d)]

if best_d > d_h30 * 1.02:  # At least 2% improvement
    print(f"✓ PROCEED with {best_name}")
    print(f"  Best result: d={best_d:.3f} ({(best_d/d_h30 - 1)*100:+.1f}% improvement over H30)")
    print(f"  This justifies full testbench validation")
elif best_d > d_h30:
    print(f"⚠ MARGINAL: {best_name} shows {(best_d/d_h30 - 1)*100:+.1f}% improvement")
    print(f"  Improvement too small (<2%) to justify polynomial complexity")
    print(f"  RECOMMEND: Skip H31, move to next hypothesis")
else:
    print(f"✗ REJECT: All polynomial variants WORSE than H30")
    print(f"  Best was {best_name} with d={best_d:.3f} ({(best_d/d_h30 - 1)*100:+.1f}%)")
    print(f"  Polynomial features do not improve discrimination")
    print(f"  RECOMMEND: Add to BLACKLIST, try different approach")

print("\nKEY INSIGHTS:")
if quadratic_ratio <= linear_ratio * 1.1:
    print("- Quadratic terms DO NOT amplify MG/HC separation")
if product_ratio <= max(linear_ratio, deg_ratio):
    print("- Product term WEAKER than best linear component")
if best_d <= d_h30:
    print("- Polynomial features ADD NOISE rather than signal")
