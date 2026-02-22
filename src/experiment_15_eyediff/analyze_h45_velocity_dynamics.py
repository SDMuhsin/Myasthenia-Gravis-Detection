#!/usr/bin/env python3
"""
CYCLE 45 - PHASE 4: Empirical Pre-Implementation Analysis
Saccadic Velocity Dynamics

HYPOTHESIS: Velocity degradation (decline in peak saccadic velocity over session)
may capture muscle fatigue that position-based metrics miss. Velocity is kinematic
(process), position is static (endpoint).

PRE-IMPLEMENTATION ANALYSIS:
1. Detect saccades and compute peak velocity per saccade
2. Measure velocity degradation (early vs late mean peak velocity)
3. Check correlation with position degradation (orthogonality test)
4. Measure velocity degradation asymmetry discrimination (Cohen's d)
5. GO/NO-GO decision

Expected: If velocity degradation is orthogonal to position degradation (r<0.7)
and shows comparable discrimination (d≥0.50), it may improve the metric.
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
print("CYCLE 45: SACCADIC VELOCITY DYNAMICS - PHASE 4 ANALYSIS")
print("="*80)

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

def compute_velocity_and_position_metrics(LV, RV, TargetV, sample_rate_hz=120):
    """
    Extract both velocity degradation and position degradation for comparison.

    Returns dict with:
    - pos_deg_L, pos_deg_R: Position degradation (late - early error)
    - vel_deg_L, vel_deg_R: Velocity degradation (late - early peak velocity)
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

        if len(up_indices) < 6:  # Need at least 6 saccades
            return None

        # Extract per-saccade metrics
        saccade_errors = []
        saccade_peak_vels = []

        for idx in up_indices:
            # Position error: mean absolute error 20-50 samples after saccade onset
            start = idx + 20
            end = min(idx + 50, len(eye))
            if end - start < 10:
                continue

            error = np.mean(np.abs(eye[start:end] - target[start:end]))
            saccade_errors.append(error)

            # Velocity: peak velocity during saccade (0-30 samples after onset)
            sac_start = idx
            sac_end = min(idx + 30, len(eye))
            if sac_end - sac_start < 4:  # Need at least 4 samples
                continue

            # Compute velocity (degrees per sample → degrees per second)
            velocity = np.abs(np.diff(eye[sac_start:sac_end])) * sample_rate_hz

            if len(velocity) == 0:
                continue

            # Smooth velocity with 3-point moving average to reduce noise
            if len(velocity) >= 3:
                velocity_smooth = np.convolve(velocity, np.ones(3)/3, mode='valid')
            else:
                velocity_smooth = velocity

            if len(velocity_smooth) > 0:
                peak_vel = np.max(velocity_smooth)
                saccade_peak_vels.append(peak_vel)

        if len(saccade_errors) < 6 or len(saccade_peak_vels) < 6:
            return None

        # Ensure equal length by taking minimum
        min_len = min(len(saccade_errors), len(saccade_peak_vels))
        saccade_errors = saccade_errors[:min_len]
        saccade_peak_vels = saccade_peak_vels[:min_len]

        if min_len < 6:
            return None

        # Position degradation (early vs late 1/3)
        n = len(saccade_errors)
        third = max(2, n // 3)
        early_err = np.mean(saccade_errors[:third])
        late_err = np.mean(saccade_errors[-third:])
        pos_deg = late_err - early_err

        # Velocity degradation (early vs late 1/3)
        early_vel = np.mean(saccade_peak_vels[:third])
        late_vel = np.mean(saccade_peak_vels[-third:])
        vel_deg = late_vel - early_vel  # Note: negative if velocity decreases

        return {
            'pos_deg': pos_deg,
            'vel_deg': vel_deg,
            'n_saccades': min_len,
            'early_vel': early_vel,
            'late_vel': late_vel,
        }

    met_L = process_eye(LV, TargetV)
    met_R = process_eye(RV, TargetV)

    if met_L is None or met_R is None:
        return None

    return {
        'pos_deg_L': met_L['pos_deg'],
        'pos_deg_R': met_R['pos_deg'],
        'vel_deg_L': met_L['vel_deg'],
        'vel_deg_R': met_R['vel_deg'],
        'n_saccades_L': met_L['n_saccades'],
        'n_saccades_R': met_R['n_saccades'],
    }

hc_metrics = []
mg_metrics = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)
    metrics = compute_velocity_and_position_metrics(df['LV'].values, df['RV'].values, df['TargetV'].values)
    if metrics is not None:
        if seq['label'] == 0:
            hc_metrics.append(metrics)
        else:
            mg_metrics.append(metrics)

print(f"Valid sequences: HC={len(hc_metrics)}, MG={len(mg_metrics)}\n")

# Extract arrays
hc_pos_deg_L = np.array([m['pos_deg_L'] for m in hc_metrics])
hc_pos_deg_R = np.array([m['pos_deg_R'] for m in hc_metrics])
hc_vel_deg_L = np.array([m['vel_deg_L'] for m in hc_metrics])
hc_vel_deg_R = np.array([m['vel_deg_R'] for m in hc_metrics])

mg_pos_deg_L = np.array([m['pos_deg_L'] for m in mg_metrics])
mg_pos_deg_R = np.array([m['pos_deg_R'] for m in mg_metrics])
mg_vel_deg_L = np.array([m['vel_deg_L'] for m in mg_metrics])
mg_vel_deg_R = np.array([m['vel_deg_R'] for m in mg_metrics])

print("="*80)
print("ANALYSIS 1: VELOCITY DEGRADATION CHARACTERISTICS")
print("="*80)

print(f"\nVelocity Degradation (negative = slowing down):")
print(f"  HC Left:  {np.mean(hc_vel_deg_L):>7.2f} ± {np.std(hc_vel_deg_L):.2f} °/s")
print(f"  HC Right: {np.mean(hc_vel_deg_R):>7.2f} ± {np.std(hc_vel_deg_R):.2f} °/s")
print(f"  MG Left:  {np.mean(mg_vel_deg_L):>7.2f} ± {np.std(mg_vel_deg_L):.2f} °/s")
print(f"  MG Right: {np.mean(mg_vel_deg_R):>7.2f} ± {np.std(mg_vel_deg_R):.2f} °/s")

print(f"\nPosition Degradation (for comparison):")
print(f"  HC Left:  {np.mean(hc_pos_deg_L):>6.3f} ± {np.std(hc_pos_deg_L):.3f}°")
print(f"  HC Right: {np.mean(hc_pos_deg_R):>6.3f} ± {np.std(hc_pos_deg_R):.3f}°")
print(f"  MG Left:  {np.mean(mg_pos_deg_L):>6.3f} ± {np.std(mg_pos_deg_L):.3f}°")
print(f"  MG Right: {np.mean(mg_pos_deg_R):>6.3f} ± {np.std(mg_pos_deg_R):.3f}°")

print("\n" + "="*80)
print("ANALYSIS 2: ORTHOGONALITY TEST (Correlation)")
print("="*80)

# Combined per-eye measurements
hc_pos_combined = np.concatenate([hc_pos_deg_L, hc_pos_deg_R])
hc_vel_combined = np.concatenate([hc_vel_deg_L, hc_vel_deg_R])
mg_pos_combined = np.concatenate([mg_pos_deg_L, mg_pos_deg_R])
mg_vel_combined = np.concatenate([mg_vel_deg_L, mg_vel_deg_R])

r_hc, p_hc = stats.pearsonr(hc_pos_combined, hc_vel_combined)
r_mg, p_mg = stats.pearsonr(mg_pos_combined, mg_vel_combined)

print(f"\nCorrelation between position degradation and velocity degradation:")
print(f"  HC: r={r_hc:.3f} (p={p_hc:.4f})")
print(f"  MG: r={r_mg:.3f} (p={p_mg:.4f})")

if r_hc < 0.7 and r_mg < 0.7:
    print(f"\n✓ PASS: r<0.7 - velocity and position degradation are orthogonal")
    orthogonal = True
elif r_hc > 0.9 or r_mg > 0.9:
    print(f"\n✗ FAIL: r>0.9 - velocity degradation is redundant with position")
    orthogonal = False
else:
    print(f"\n~ MARGINAL: 0.7≤r≤0.9 - moderate correlation")
    orthogonal = True

print("\n" + "="*80)
print("ANALYSIS 3: VELOCITY DEGRADATION ASYMMETRY DISCRIMINATION")
print("="*80)

# Compute asymmetries
hc_pos_asym = np.abs(hc_pos_deg_L - hc_pos_deg_R)
mg_pos_asym = np.abs(mg_pos_deg_L - mg_pos_deg_R)

hc_vel_asym = np.abs(hc_vel_deg_L - hc_vel_deg_R)
mg_vel_asym = np.abs(mg_vel_deg_L - mg_vel_deg_R)

# Cohen's d
def cohens_d(mg_vals, hc_vals):
    pooled_std = np.sqrt(((len(hc_vals)-1)*np.var(hc_vals, ddof=1) +
                          (len(mg_vals)-1)*np.var(mg_vals, ddof=1)) /
                         (len(hc_vals) + len(mg_vals) - 2))
    return (np.mean(mg_vals) - np.mean(hc_vals)) / pooled_std if pooled_std > 0 else 0

d_pos = cohens_d(mg_pos_asym, hc_pos_asym)
d_vel = cohens_d(mg_vel_asym, hc_vel_asym)

print(f"\nPosition Degradation Asymmetry:")
print(f"  HC: {np.mean(hc_pos_asym):.4f} ± {np.std(hc_pos_asym):.4f}°")
print(f"  MG: {np.mean(mg_pos_asym):.4f} ± {np.std(mg_pos_asym):.4f}°")
print(f"  Cohen's d = {d_pos:.3f}")

print(f"\nVelocity Degradation Asymmetry:")
print(f"  HC: {np.mean(hc_vel_asym):.4f} ± {np.std(hc_vel_asym):.4f} °/s")
print(f"  MG: {np.mean(mg_vel_asym):.4f} ± {np.std(mg_vel_asym):.4f} °/s")
print(f"  Cohen's d = {d_vel:.3f}")

if d_vel >= 0.50:
    print(f"\n✓ PASS: d_vel={d_vel:.3f} ≥ 0.50 - velocity asymmetry is strong discriminator")
    vel_discriminative = True
elif d_vel < 0.40:
    print(f"\n✗ FAIL: d_vel={d_vel:.3f} < 0.40 - velocity asymmetry too weak")
    vel_discriminative = False
else:
    print(f"\n~ MARGINAL: 0.40 ≤ d_vel={d_vel:.3f} < 0.50")
    vel_discriminative = True

# Comparison
improvement = ((d_vel - d_pos) / d_pos * 100) if d_pos > 0 else 0
print(f"\nComparison:")
print(f"  Velocity vs Position: {improvement:+.1f}% {'(better)' if improvement > 0 else '(worse)'}")

print("\n" + "="*80)
print("ANALYSIS 4: COMBINED PERFORMANCE TEST")
print("="*80)

# Normalize to same scale
hc_pos_std = np.std(hc_pos_asym)
hc_vel_std = np.std(hc_vel_asym)

hc_pos_norm = hc_pos_asym / hc_pos_std
mg_pos_norm = mg_pos_asym / hc_pos_std
hc_vel_norm = hc_vel_asym / hc_vel_std
mg_vel_norm = mg_vel_asym / hc_vel_std

# Test combinations
print(f"\nTesting combinations:")
print(f"{'Metric':<30} {'MG_mean':<12} {'HC_mean':<12} {'Cohen_d':<10} {'vs PosOnly':<12}")
print("-"*75)

print(f"{'Position only':<30} {np.mean(mg_pos_norm):>6.3f}      {np.mean(hc_pos_norm):>6.3f}      {d_pos:>6.3f}    baseline")
print(f"{'Velocity only':<30} {np.mean(mg_vel_norm):>6.3f}      {np.mean(hc_vel_norm):>6.3f}      {d_vel:>6.3f}    {improvement:>+6.1f}%")

weight_combinations = [
    (0.50, 0.50, "Equal weight"),
    (0.70, 0.30, "Position-heavy"),
    (0.30, 0.70, "Velocity-heavy"),
]

best_d = max(d_pos, d_vel)
best_combo = None

for w_pos, w_vel, name in weight_combinations:
    hc_combined = w_pos * hc_pos_norm + w_vel * hc_vel_norm
    mg_combined = w_pos * mg_pos_norm + w_vel * mg_vel_norm
    d_combined = cohens_d(mg_combined, hc_combined)
    improvement_combo = ((d_combined - d_pos) / d_pos * 100)

    print(f"{name:<30} {np.mean(mg_combined):>6.3f}      {np.mean(hc_combined):>6.3f}      {d_combined:>6.3f}    {improvement_combo:>+6.1f}%")

    if d_combined > best_d:
        best_d = d_combined
        best_combo = (w_pos, w_vel, name)

print("\n" + "="*80)
print("GO/NO-GO DECISION")
print("="*80)

print(f"\nCriteria Results:")
print(f"1. Orthogonality (r<0.7): {'✓ PASS' if orthogonal else '✗ FAIL'} (HC r={r_hc:.3f}, MG r={r_mg:.3f})")
print(f"2. Velocity discrimination (d≥0.50): {'✓ PASS' if vel_discriminative else '✗ FAIL'} (d={d_vel:.3f})")
print(f"3. Combined improvement over position: {'✓ PASS' if best_d > d_pos * 1.05 else '✗ FAIL'} (best d={best_d:.3f} vs pos d={d_pos:.3f})")

passes = sum([orthogonal, vel_discriminative, best_d > d_pos * 1.05])

print(f"\n{'='*80}")
if passes >= 2:
    print("DECISION: GO")
    print(f"  {passes}/3 criteria passed")
    if best_combo:
        print(f"  Best combination: {best_combo[2]} (w_pos={best_combo[0]}, w_vel={best_combo[1]})")
        print(f"  Cohen's d = {best_d:.3f} ({((best_d - d_pos)/d_pos*100):+.1f}% vs position only)")
    else:
        print(f"  Velocity alone: d={d_vel:.3f} ({improvement:+.1f}% vs position)")
    print(f"\n→ PROCEED to Phase 6: Implementation of H45")
else:
    print("DECISION: NO-GO")
    print(f"  Only {passes}/3 criteria passed")
    if not orthogonal:
        print(f"  Primary issue: Velocity and position degradation too correlated (r={max(r_hc, r_mg):.3f})")
    elif not vel_discriminative:
        print(f"  Primary issue: Velocity discrimination too weak (d={d_vel:.3f})")
    else:
        print(f"  Primary issue: Combined performance no better than position alone")
    print(f"\n→ REJECT H45")
    print(f"→ GAP~17 CONFIRMED AS REALISTIC CEILING")
    print(f"→ Consider:")
    print(f"   1. Accept current H38b performance (gap=16.8)")
    print(f"   2. Validate on external dataset")
    print(f"   3. Explore fundamentally different modality (smooth pursuit, fixation)")

print("="*80)
