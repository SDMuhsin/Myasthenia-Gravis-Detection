#!/usr/bin/env python3
"""
H26: Bilateral Symmetry Score

NOT a classification task - measuring asymmetry to identify MORE affected eye.

Hypothesis: HC should have high L-R correlation despite baseline position difference.
MG should show REDUCED correlation (desynchronization) in affected eye.

Rationale: Even if HC have |L-R| ≠ 0 in absolute position, their movements should
be synchronized (high correlation). MG asymmetric weakness breaks synchronization.

Expected: Better HC≈0 by measuring SYNCHRONY not just DIFFERENCE
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

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

def compute_upward_metrics(eye_pos, target_pos):
    """Extract upward saccade data"""
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
    
    # Tracking error time series
    error = eye_upward - target_upward
    
    return {
        'position': eye_upward,
        'target': target_upward,
        'error': error,
        'mad': np.median(np.abs(eye_upward - np.median(eye_upward))),
        'degradation': np.mean(np.abs(error[-int(len(error)*0.2):])) - np.mean(np.abs(error[:int(len(error)*0.2)]))
    }

results = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)
    
    metrics_l = compute_upward_metrics(df['LV'].values, df['TargetV'].values)
    metrics_r = compute_upward_metrics(df['RV'].values, df['TargetV'].values)
    
    if metrics_l is None or metrics_r is None:
        continue
    
    # Ensure same length for correlation (use shorter)
    n = min(len(metrics_l['error']), len(metrics_r['error']))
    error_l = metrics_l['error'][:n]
    error_r = metrics_r['error'][:n]
    
    if n < 20:
        continue
    
    # H24 baseline components
    mad_asym = abs(metrics_l['mad'] - metrics_r['mad'])
    deg_asym = abs(metrics_l['degradation'] - metrics_r['degradation'])
    h24_positional = 0.45 * mad_asym + 0.55 * deg_asym
    
    # H26 NEW: Correlation-based synchrony metrics
    
    # 1. Error correlation (should be high in HC - both eyes make similar tracking errors)
    corr_error = np.corrcoef(error_l, error_r)[0, 1] if n >= 20 else np.nan
    
    # 2. Desynchronization score: 1 - correlation (higher = more desynchronized)
    desync_score = 1 - corr_error if not np.isnan(corr_error) else np.nan
    
    # 3. Error variance asymmetry (different from MAD - captures variance ratio)
    var_l = np.var(error_l)
    var_r = np.var(error_r)
    var_ratio = max(var_l, var_r) / (min(var_l, var_r) + 1e-6) - 1  # Ratio - 1 (0 = symmetric)
    
    # 4. Combined: Desynchronization + positional asymmetry
    h26_desynch_only = desync_score
    h26_combined = 0.5 * desync_score + 0.5 * h24_positional if not np.isnan(desync_score) else np.nan
    
    results.append({
        'class': seq['class_name'],
        'h24_positional': h24_positional,
        'h26_desync': h26_desynch_only,
        'h26_combined': h26_combined,
        'corr_error': corr_error,
        'var_ratio': var_ratio
    })

results_df = pd.DataFrame(results)
hc_df = results_df[results_df['class'] == 'HC']
mg_df = results_df[results_df['class'] == 'MG']

print("\n" + "="*80)
print("H26: BILATERAL SYMMETRY SCORE (Correlation-Based)")
print("="*80)
print(f"\nDataset: {len(hc_df)} HC, {len(mg_df)} MG")
print("\nGoal: Measure SYNCHRONY (correlation) not just absolute difference")
print("Hypothesis: HC high correlation despite baseline difference")

metrics = ['h24_positional', 'h26_desync', 'h26_combined', 'var_ratio']
names = ['H24 Positional', 'H26 Desync Only', 'H26 Desync+Pos', 'Variance Ratio']

print(f"\n{'Metric':<20} {'HC Mean':<12} {'MG Mean':<12} {'d':<10} {'Val':<6} {'Spec@0.4':<10}")
print("="*90)

best_d = 0
best_metric = None

for metric, name in zip(metrics, names):
    hc_vals = hc_df[metric].dropna().values
    mg_vals = mg_df[metric].dropna().values
    
    if len(hc_vals) < 10 or len(mg_vals) < 10:
        continue
    
    hc_mean = np.mean(hc_vals)
    mg_mean = np.mean(mg_vals)
    hc_std = np.std(hc_vals)
    mg_std = np.std(mg_vals)
    
    pooled_std = np.sqrt((hc_std**2 + mg_std**2) / 2)
    d = (mg_mean - hc_mean) / pooled_std
    
    _, p_mw = stats.mannwhitneyu(mg_vals, hc_vals, alternative='greater')
    _, p_hc = stats.wilcoxon(hc_vals)
    _, p_mg = stats.wilcoxon(mg_vals)
    
    val_score = (p_mw < 0.05) + (p_hc >= 0.05) + (p_mg < 0.05)
    
    # Specificity at threshold 0.4
    spec = 100 * np.sum(hc_vals < 0.4) / len(hc_vals)
    
    print(f"{name:<20} {hc_mean:<12.4f} {mg_mean:<12.4f} {d:<10.4f} {val_score:<6}/3 {spec:<10.1f}%")
    
    if d > best_d:
        best_d = d
        best_metric = name

# Check correlation distribution
hc_corr = hc_df['corr_error'].dropna().values
mg_corr = mg_df['corr_error'].dropna().values

print("\n" + "-"*90)
print("ERROR CORRELATION ANALYSIS (raw correlation values)")
print(f"HC correlation: {np.mean(hc_corr):.3f} ± {np.std(hc_corr):.3f}")
print(f"MG correlation: {np.mean(mg_corr):.3f} ± {np.std(mg_corr):.3f}")
print(f"Expected: HC should have HIGHER correlation (more synchronized)")
print(f"Reality: {'HC > MG ✓' if np.mean(hc_corr) > np.mean(mg_corr) else 'MG > HC ✗ (hypothesis fails)'}")

print("\n" + "="*90)
print(f"BEST: {best_metric}, d={best_d:.4f}")

if best_d >= 0.65:
    print("\n✓✓✓ BREAKTHROUGH! d ≥ 0.65 target achieved!")
elif best_d > 0.577:
    print(f"\n✓ IMPROVEMENT over H24 (d=0.577 → {best_d:.3f})")
else:
    print(f"\n✗ No improvement over H24 (best d={best_d:.3f} ≤ 0.577)")

print("="*90)
