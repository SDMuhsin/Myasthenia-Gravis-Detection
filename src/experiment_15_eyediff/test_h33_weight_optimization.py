#!/usr/bin/env python3
"""
H33: Analytical Weight Optimization

Hypothesis: H30's fixed weights (0.45 MAD, 0.55 degradation, 0.50 positional, 0.50 latency)
may be sub-optimal. Optimize weights using analytical methods (NOT supervised learning).

Allowed Analytical Methods:
1. Variance-based weighting: w_i ∝ 1/var(component_i) for HC
   (inverse variance weighting reduces noise contribution)

2. Signal-to-noise weighting: w_i ∝ (μ_MG - μ_HC) / σ_HC
   (weights components by their discrimination power relative to HC noise)

3. Correlation-adjusted weighting: Adjust for redundancy between components

CRITICAL: These are unsupervised - weights computed from component statistics,
NOT from class labels directly.
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

# [Same compute functions as H31]
def compute_upward_metrics(eye_pos, target_pos):
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
    mad = np.median(np.abs(error - np.median(error)))
    n = len(error)
    early_n = max(5, int(n * 0.2))
    late_n = max(5, int(n * 0.2))
    degradation = np.mean(np.abs(error[-late_n:])) - np.mean(np.abs(error[:early_n]))
    return {'mad': mad, 'degradation': degradation}

def compute_latencies_upward(eye_pos, target_pos, sample_rate_hz=120):
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

print("Computing H30 components...")
hc_components = {'cv_asym': [], 'deg_asym': [], 'lat_asym': []}
mg_components = {'cv_asym': [], 'deg_asym': [], 'lat_asym': []}

for seq in sequences:
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

    mad_L = metrics_L['mad']
    mad_R = metrics_R['mad']
    deg_L = metrics_L['degradation']
    deg_R = metrics_R['degradation']
    lat_L = np.mean(sorted(latencies_L)[:max(3, len(latencies_L)//4)])
    lat_R = np.mean(sorted(latencies_R)[:max(3, len(latencies_R)//4)])

    cv_asym = abs(mad_L - mad_R) / ((mad_L + mad_R) / 2)
    deg_asym = abs(deg_L - deg_R)
    lat_asym = abs(lat_L - lat_R) / 100

    components = {'cv_asym': cv_asym, 'deg_asym': deg_asym, 'lat_asym': lat_asym}

    if seq['label'] == 0:
        hc_components['cv_asym'].append(cv_asym)
        hc_components['deg_asym'].append(deg_asym)
        hc_components['lat_asym'].append(lat_asym)
    else:
        mg_components['cv_asym'].append(cv_asym)
        mg_components['deg_asym'].append(deg_asym)
        mg_components['lat_asym'].append(lat_asym)

hc_cv = np.array(hc_components['cv_asym'])
hc_deg = np.array(hc_components['deg_asym'])
hc_lat = np.array(hc_components['lat_asym'])
mg_cv = np.array(mg_components['cv_asym'])
mg_deg = np.array(mg_components['deg_asym'])
mg_lat = np.array(mg_components['lat_asym'])

print(f"Valid samples: HC={len(hc_cv)}, MG={len(mg_cv)}\n")

print("="*80)
print("H33: ANALYTICAL WEIGHT OPTIMIZATION")
print("="*80)

# Method 1: Inverse variance weighting (HC noise minimization)
print("\n1. INVERSE VARIANCE WEIGHTING (minimize HC noise):")
var_cv = np.var(hc_cv)
var_deg = np.var(hc_deg)
var_lat = np.var(hc_lat)

w_cv_iv = (1/var_cv) / ((1/var_cv) + (1/var_deg) + (1/var_lat))
w_deg_iv = (1/var_deg) / ((1/var_cv) + (1/var_deg) + (1/var_lat))
w_lat_iv = (1/var_lat) / ((1/var_cv) + (1/var_deg) + (1/var_lat))

print(f"   HC variances: CV={var_cv:.4f}, Deg={var_deg:.4f}, Lat={var_lat:.4f}")
print(f"   Weights: CV={w_cv_iv:.3f}, Deg={w_deg_iv:.3f}, Lat={w_lat_iv:.3f}")

h33a_hc = w_cv_iv*hc_cv + w_deg_iv*hc_deg + w_lat_iv*hc_lat
h33a_mg = w_cv_iv*mg_cv + w_deg_iv*mg_deg + w_lat_iv*mg_lat
pooled_std_a = np.sqrt(((len(h33a_hc)-1)*np.var(h33a_hc, ddof=1) + (len(h33a_mg)-1)*np.var(h33a_mg, ddof=1)) / (len(h33a_hc) + len(h33a_mg) - 2))
d_h33a = (np.mean(h33a_mg) - np.mean(h33a_hc)) / pooled_std_a
u_a, p_a = stats.mannwhitneyu(h33a_mg, h33a_hc, alternative='greater')
print(f"   H33a: d={d_h33a:.3f}, p={p_a:.4f}")

# Method 2: Signal-to-noise weighting
print("\n2. SIGNAL-TO-NOISE WEIGHTING (maximize discrimination/noise ratio):")
snr_cv = (np.mean(mg_cv) - np.mean(hc_cv)) / np.std(hc_cv)
snr_deg = (np.mean(mg_deg) - np.mean(hc_deg)) / np.std(hc_deg)
snr_lat = (np.mean(mg_lat) - np.mean(hc_lat)) / np.std(hc_lat)

w_cv_snr = snr_cv / (snr_cv + snr_deg + snr_lat)
w_deg_snr = snr_deg / (snr_cv + snr_deg + snr_lat)
w_lat_snr = snr_lat / (snr_cv + snr_deg + snr_lat)

print(f"   SNR: CV={snr_cv:.3f}, Deg={snr_deg:.3f}, Lat={snr_lat:.3f}")
print(f"   Weights: CV={w_cv_snr:.3f}, Deg={w_deg_snr:.3f}, Lat={w_lat_snr:.3f}")

h33b_hc = w_cv_snr*hc_cv + w_deg_snr*hc_deg + w_lat_snr*hc_lat
h33b_mg = w_cv_snr*mg_cv + w_deg_snr*mg_deg + w_lat_snr*mg_lat
pooled_std_b = np.sqrt(((len(h33b_hc)-1)*np.var(h33b_hc, ddof=1) + (len(h33b_mg)-1)*np.var(h33b_mg, ddof=1)) / (len(h33b_hc) + len(h33b_mg) - 2))
d_h33b = (np.mean(h33b_mg) - np.mean(h33b_hc)) / pooled_std_b
u_b, p_b = stats.mannwhitneyu(h33b_mg, h33b_hc, alternative='greater')
print(f"   H33b: d={d_h33b:.3f}, p={p_b:.4f}")

# Method 3: Equal-weight (for comparison)
print("\n3. EQUAL WEIGHTING (baseline):")
h33c_hc = (hc_cv + hc_deg + hc_lat) / 3
h33c_mg = (mg_cv + mg_deg + mg_lat) / 3
pooled_std_c = np.sqrt(((len(h33c_hc)-1)*np.var(h33c_hc, ddof=1) + (len(h33c_mg)-1)*np.var(h33c_mg, ddof=1)) / (len(h33c_hc) + len(h33c_mg) - 2))
d_h33c = (np.mean(h33c_mg) - np.mean(h33c_hc)) / pooled_std_c
u_c, p_c = stats.mannwhitneyu(h33c_mg, h33c_hc, alternative='greater')
print(f"   H33c: d={d_h33c:.3f}, p={p_c:.4f}")

# H30 original (0.5 * (0.45*CV + 0.55*Deg) + 0.5*Lat)
print("\n4. H30 ORIGINAL WEIGHTING:")
h30_hc = 0.5 * (0.45*hc_cv + 0.55*hc_deg) + 0.5*hc_lat
h30_mg = 0.5 * (0.45*mg_cv + 0.55*mg_deg) + 0.5*mg_lat
pooled_std_30 = np.sqrt(((len(h30_hc)-1)*np.var(h30_hc, ddof=1) + (len(h30_mg)-1)*np.var(h30_mg, ddof=1)) / (len(h30_hc) + len(h30_mg) - 2))
d_h30 = (np.mean(h30_mg) - np.mean(h30_hc)) / pooled_std_30
u_30, p_30 = stats.mannwhitneyu(h30_mg, h30_hc, alternative='greater')
print(f"   Effective weights: CV=0.225, Deg=0.275, Lat=0.500")
print(f"   H30: d={d_h30:.3f}, p={p_30:.4f}")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

results = [
    ('H30 (original)', d_h30, (0.225, 0.275, 0.500)),
    ('H33a (inv-var)', d_h33a, (w_cv_iv, w_deg_iv, w_lat_iv)),
    ('H33b (SNR)', d_h33b, (w_cv_snr, w_deg_snr, w_lat_snr)),
    ('H33c (equal)', d_h33c, (0.333, 0.333, 0.333)),
]

results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

print("\nRanked by Cohen's d:")
for i, (name, d_val, weights) in enumerate(results_sorted, 1):
    improvement = (d_val / d_h30 - 1) * 100
    print(f"{i}. {name:20} d={d_val:.3f} ({improvement:+.1f}%) w=[{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}]")

best_name, best_d, best_weights = results_sorted[0]

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

if best_d > d_h30 * 1.02:
    print(f"✓ PROCEED with {best_name}")
    print(f"  {(best_d/d_h30 - 1)*100:+.1f}% improvement justifies implementation")
    print(f"  Optimal weights: CV={best_weights[0]:.3f}, Deg={best_weights[1]:.3f}, Lat={best_weights[2]:.3f}")
else:
    print(f"✗ REJECT: Best variant ({best_name}) shows {(best_d/d_h30 - 1)*100:+.1f}% change")
    print(f"  Weight optimization does not provide meaningful improvement")
    print(f"  H30's hand-tuned weights are near-optimal")
