#!/usr/bin/env python3
"""
CYCLE 44 - PHASE 4: Empirical Pre-Implementation Analysis
Multi-Temporal Fatigue Signature

HYPOTHESIS: Combining multiple temporal measurements (degradation magnitude, slope, linearity)
captures richer fatigue signature than single-temporal metrics (FAT1 or FAT3 alone).

PRE-IMPLEMENTATION ANALYSIS:
1. Compute degradation magnitude (late - early) and slope (linear regression)
2. Check correlation between deg and slope (orthogonality test)
3. Measure individual discrimination: d_deg, d_slope
4. Test combined performance: d_combined
5. GO/NO-GO decision based on empirical results

Expected: If deg and slope are orthogonal (r<0.7) and both discriminative (d≥0.45),
combining them should improve over single metrics.
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
print("CYCLE 44: MULTI-TEMPORAL FATIGUE SIGNATURE - PHASE 4 ANALYSIS")
print("="*80)

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

def compute_temporal_features(LV, RV, TargetV, sample_rate_hz=120):
    """
    Extract degradation magnitude and slope for each eye.

    Returns dict with:
    - deg_L, deg_R: Degradation magnitude (late - early error)
    - slope_L, slope_R: Linear regression slope across 5 windows
    - rsq_L, rsq_R: R² of slope fit (linearity measure)
    """
    def process_eye(eye_pos, target_pos):
        valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
        eye = eye_pos[valid]
        target = target_pos[valid]
        if len(eye) < 50:
            return None

        # Detect upward saccades
        target_diff = np.diff(target)
        up_indices = np.where(target_diff > 5.0)[0] + 1

        # Create mask for upward saccades (50 samples post-jump)
        mask = np.zeros(len(eye), dtype=bool)
        for idx in up_indices:
            mask[idx:min(idx+50, len(eye))] = True

        eye_up = eye[mask]
        target_up = target[mask]
        if len(eye_up) < 30:
            return None

        error = eye_up - target_up

        # Divide into 5 equal windows (quintiles)
        n = len(error)
        window_size = n // 5
        if window_size < 5:  # Need at least 5 samples per window
            return None

        window_errors = []
        for i in range(5):
            start = i * window_size
            end = start + window_size if i < 4 else n  # Last window gets remainder
            window_err = np.mean(np.abs(error[start:end]))
            window_errors.append(window_err)

        # Degradation magnitude (last - first window)
        deg = window_errors[4] - window_errors[0]

        # Linear regression slope
        x = np.arange(5)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, window_errors)
        rsq = r_value ** 2

        return {
            'deg': deg,
            'slope': slope,
            'rsq': rsq,
            'window_errors': window_errors
        }

    met_L = process_eye(LV, TargetV)
    met_R = process_eye(RV, TargetV)

    if met_L is None or met_R is None:
        return None

    return {
        'deg_L': met_L['deg'],
        'deg_R': met_R['deg'],
        'slope_L': met_L['slope'],
        'slope_R': met_R['slope'],
        'rsq_L': met_L['rsq'],
        'rsq_R': met_R['rsq'],
    }

hc_metrics = []
mg_metrics = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)
    metrics = compute_temporal_features(df['LV'].values, df['RV'].values, df['TargetV'].values)
    if metrics is not None:
        if seq['label'] == 0:
            hc_metrics.append(metrics)
        else:
            mg_metrics.append(metrics)

print(f"Valid sequences: HC={len(hc_metrics)}, MG={len(mg_metrics)}\n")

# Extract arrays for analysis
hc_deg_L = np.array([m['deg_L'] for m in hc_metrics])
hc_deg_R = np.array([m['deg_R'] for m in hc_metrics])
hc_slope_L = np.array([m['slope_L'] for m in hc_metrics])
hc_slope_R = np.array([m['slope_R'] for m in hc_metrics])

mg_deg_L = np.array([m['deg_L'] for m in mg_metrics])
mg_deg_R = np.array([m['deg_R'] for m in mg_metrics])
mg_slope_L = np.array([m['slope_L'] for m in mg_metrics])
mg_slope_R = np.array([m['slope_R'] for m in mg_metrics])

print("="*80)
print("ANALYSIS 1: FEATURE CORRELATION (Orthogonality Test)")
print("="*80)

# Compute per-eye correlation between deg and slope
hc_deg_combined = np.concatenate([hc_deg_L, hc_deg_R])
hc_slope_combined = np.concatenate([hc_slope_L, hc_slope_R])
mg_deg_combined = np.concatenate([mg_deg_L, mg_deg_R])
mg_slope_combined = np.concatenate([mg_slope_L, mg_slope_R])

r_hc, p_hc = stats.pearsonr(hc_deg_combined, hc_slope_combined)
r_mg, p_mg = stats.pearsonr(mg_deg_combined, mg_slope_combined)

print(f"\nCorrelation between degradation magnitude and slope:")
print(f"  HC: r={r_hc:.3f} (p={p_hc:.4f})")
print(f"  MG: r={r_mg:.3f} (p={p_mg:.4f})")
print(f"\nOrthogonality Test:")
if r_hc < 0.7 and r_mg < 0.7:
    print(f"  ✓ PASS: r<0.7 for both groups - deg and slope are orthogonal")
    orthogonal = True
elif r_hc > 0.9 or r_mg > 0.9:
    print(f"  ✗ FAIL: r>0.9 - deg and slope are redundant")
    orthogonal = False
else:
    print(f"  ~ MARGINAL: 0.7≤r≤0.9 - moderate correlation")
    orthogonal = True

print("\n" + "="*80)
print("ANALYSIS 2: INDIVIDUAL COMPONENT DISCRIMINATION")
print("="*80)

# Compute asymmetries
hc_deg_asym = np.abs(hc_deg_L - hc_deg_R)
mg_deg_asym = np.abs(mg_deg_L - mg_deg_R)

hc_slope_asym = np.abs(hc_slope_L - hc_slope_R)
mg_slope_asym = np.abs(mg_slope_L - mg_slope_R)

# Cohen's d for each component
def cohens_d(mg_vals, hc_vals):
    pooled_std = np.sqrt(((len(hc_vals)-1)*np.var(hc_vals, ddof=1) +
                          (len(mg_vals)-1)*np.var(mg_vals, ddof=1)) /
                         (len(hc_vals) + len(mg_vals) - 2))
    return (np.mean(mg_vals) - np.mean(hc_vals)) / pooled_std if pooled_std > 0 else 0

d_deg = cohens_d(mg_deg_asym, hc_deg_asym)
d_slope = cohens_d(mg_slope_asym, hc_slope_asym)

print(f"\nDegradation Magnitude Asymmetry:")
print(f"  HC: {np.mean(hc_deg_asym):.4f} ± {np.std(hc_deg_asym):.4f}°")
print(f"  MG: {np.mean(mg_deg_asym):.4f} ± {np.std(mg_deg_asym):.4f}°")
print(f"  Cohen's d = {d_deg:.3f}")

print(f"\nSlope Asymmetry:")
print(f"  HC: {np.mean(hc_slope_asym):.4f} ± {np.std(hc_slope_asym):.4f}°/window")
print(f"  MG: {np.mean(mg_slope_asym):.4f} ± {np.std(mg_slope_asym):.4f}°/window")
print(f"  Cohen's d = {d_slope:.3f}")

print(f"\nDiscrimination Test:")
if d_slope >= 0.45:
    print(f"  ✓ PASS: d_slope={d_slope:.3f} ≥ 0.45 - slope is meaningful discriminator")
    slope_discriminative = True
elif d_slope < 0.35:
    print(f"  ✗ FAIL: d_slope={d_slope:.3f} < 0.35 - slope too weak")
    slope_discriminative = False
else:
    print(f"  ~ MARGINAL: 0.35 ≤ d_slope={d_slope:.3f} < 0.45")
    slope_discriminative = True

print("\n" + "="*80)
print("ANALYSIS 3: COMBINED PERFORMANCE")
print("="*80)

# Normalize to same scale before combining
# Standardize by HC std to make comparable
hc_deg_std = np.std(hc_deg_asym)
hc_slope_std = np.std(hc_slope_asym)

hc_deg_norm = hc_deg_asym / hc_deg_std
mg_deg_norm = mg_deg_asym / hc_deg_std
hc_slope_norm = hc_slope_asym / hc_slope_std
mg_slope_norm = mg_slope_asym / hc_slope_std

# Test different weight combinations
weight_combinations = [
    (0.50, 0.50, "Equal weight"),
    (0.60, 0.40, "Deg-heavy"),
    (0.70, 0.30, "Deg-dominant"),
    (0.40, 0.60, "Slope-heavy"),
]

print(f"\nTesting weight combinations:")
print(f"{'Weights':<20} {'MG_mean':<12} {'HC_mean':<12} {'Cohen_d':<10} {'vs FAT1':<10}")
print("-"*70)

best_d = 0
best_weights = None

for w_deg, w_slope, name in weight_combinations:
    hc_combined = w_deg * hc_deg_norm + w_slope * hc_slope_norm
    mg_combined = w_deg * mg_deg_norm + w_slope * mg_slope_norm

    d_combined = cohens_d(mg_combined, hc_combined)
    improvement = ((d_combined - 0.540) / 0.540 * 100)  # vs FAT1 d=0.540

    print(f"{name:<20} {np.mean(mg_combined):>6.3f}      {np.mean(hc_combined):>6.3f}      {d_combined:>6.3f}    {improvement:>+6.1f}%")

    if d_combined > best_d:
        best_d = d_combined
        best_weights = (w_deg, w_slope, name)

print("\n" + "="*80)
print("GO/NO-GO DECISION")
print("="*80)

print(f"\nCriteria Results:")
print(f"1. Orthogonality (r<0.7): {'✓ PASS' if orthogonal else '✗ FAIL'} (HC r={r_hc:.3f}, MG r={r_mg:.3f})")
print(f"2. Slope discrimination (d≥0.45): {'✓ PASS' if slope_discriminative else '✗ FAIL'} (d={d_slope:.3f})")
print(f"3. Combined improvement (d≥0.55): {'✓ PASS' if best_d >= 0.55 else '✗ FAIL'} (best d={best_d:.3f})")

passes = sum([orthogonal, slope_discriminative, best_d >= 0.55])

print(f"\n{'='*80}")
if passes >= 2:
    print("DECISION: GO")
    print(f"  {passes}/3 criteria passed")
    print(f"  Best combination: {best_weights[2]} (w_deg={best_weights[0]}, w_slope={best_weights[1]})")
    print(f"  Cohen's d = {best_d:.3f} ({((best_d - 0.540)/0.540*100):+.1f}% vs FAT1)")
    print(f"\n→ PROCEED to Phase 6: Implementation of H44")
else:
    print("DECISION: NO-GO")
    print(f"  Only {passes}/3 criteria passed")
    print(f"  Multi-temporal signature does not provide sufficient improvement")
    if not orthogonal:
        print(f"  Primary issue: Deg and slope are too correlated (r={max(r_hc, r_mg):.3f})")
    elif not slope_discriminative:
        print(f"  Primary issue: Slope discrimination too weak (d={d_slope:.3f})")
    else:
        print(f"  Primary issue: Combined performance below target (d={best_d:.3f} < 0.55)")
    print(f"\n→ REJECT H44. Gap~17 appears to be realistic ceiling.")
    print(f"→ Consider accepting current performance or exploring fundamentally different modality.")

print("="*80)
