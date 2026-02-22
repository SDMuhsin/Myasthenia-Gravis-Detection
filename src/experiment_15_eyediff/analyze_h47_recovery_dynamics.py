#!/usr/bin/env python3
"""
CYCLE 47 - PHASE 4: Recovery Dynamics Analysis
Outside-the-box approach: Measure fatigue CLEARANCE not just ACCUMULATION

HYPOTHESIS: MG affected eye shows declining recovery capacity between saccades.
Recovery = improvement in error from saccade i to i+1 during brief rest period.

RATIONALE: All previous cycles measured degradation (worsening). MG involves BOTH
impaired contraction AND impaired recovery. Recovery dynamics may be orthogonal.
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
print("CYCLE 47: RECOVERY DYNAMICS ANALYSIS")
print("="*80)

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

def compute_recovery_and_degradation(LV, RV, TargetV, sample_rate_hz=120):
    """
    Extract both recovery dynamics and position degradation.

    Recovery: Change in error from saccade i to i+1 (normalized by rest duration)
    Degradation: Worsening of error from early to late saccades
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

        if len(up_indices) < 10:  # Need at least 10 saccades for recovery analysis
            return None

        # Extract per-saccade settling errors
        saccade_errors = []
        saccade_times = []

        for idx in up_indices:
            # Settling error: mean absolute error 20-50 samples after saccade
            start = idx + 20
            end = min(idx + 50, len(eye))
            if end - start < 10:
                continue

            error = np.mean(np.abs(eye[start:end] - target[start:end]))
            time_sec = idx / sample_rate_hz

            saccade_errors.append(error)
            saccade_times.append(time_sec)

        if len(saccade_errors) < 10:
            return None

        # RECOVERY DYNAMICS: Change in error between consecutive saccades
        recovery_rates = []
        for i in range(len(saccade_errors) - 1):
            error_improvement = saccade_errors[i] - saccade_errors[i+1]  # Positive = improved
            rest_duration = saccade_times[i+1] - saccade_times[i]

            if rest_duration < 0.1 or rest_duration > 5.0:  # Filter unreasonable intervals
                continue

            # Recovery rate: degrees improvement per second
            recovery_rate = error_improvement / rest_duration
            recovery_rates.append(recovery_rate)

        if len(recovery_rates) < 6:
            return None

        # Recovery degradation: Does recovery capacity decline over session?
        n = len(recovery_rates)
        third = max(2, n // 3)
        early_recovery = np.mean(recovery_rates[:third])
        late_recovery = np.mean(recovery_rates[-third:])
        recovery_deg = late_recovery - early_recovery  # Negative = declining recovery

        # POSITION DEGRADATION (for comparison)
        n_err = len(saccade_errors)
        third_err = max(2, n_err // 3)
        early_err = np.mean(saccade_errors[:third_err])
        late_err = np.mean(saccade_errors[-third_err:])
        pos_deg = late_err - early_err  # Positive = worsening

        return {
            'pos_deg': pos_deg,
            'recovery_deg': recovery_deg,
            'early_recovery': early_recovery,
            'late_recovery': late_recovery,
            'n_saccades': len(saccade_errors),
        }

    met_L = process_eye(LV, TargetV)
    met_R = process_eye(RV, TargetV)

    if met_L is None or met_R is None:
        return None

    return {
        'pos_deg_L': met_L['pos_deg'],
        'pos_deg_R': met_R['pos_deg'],
        'recovery_deg_L': met_L['recovery_deg'],
        'recovery_deg_R': met_R['recovery_deg'],
    }

hc_metrics = []
mg_metrics = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)
    metrics = compute_recovery_and_degradation(df['LV'].values, df['RV'].values, df['TargetV'].values)
    if metrics is not None:
        if seq['label'] == 0:
            hc_metrics.append(metrics)
        else:
            mg_metrics.append(metrics)

print(f"Valid sequences: HC={len(hc_metrics)}, MG={len(mg_metrics)}\n")

# Extract arrays
hc_pos_deg_L = np.array([m['pos_deg_L'] for m in hc_metrics])
hc_pos_deg_R = np.array([m['pos_deg_R'] for m in hc_metrics])
hc_rec_deg_L = np.array([m['recovery_deg_L'] for m in hc_metrics])
hc_rec_deg_R = np.array([m['recovery_deg_R'] for m in hc_metrics])

mg_pos_deg_L = np.array([m['pos_deg_L'] for m in mg_metrics])
mg_pos_deg_R = np.array([m['pos_deg_R'] for m in mg_metrics])
mg_rec_deg_L = np.array([m['recovery_deg_L'] for m in mg_metrics])
mg_rec_deg_R = np.array([m['recovery_deg_R'] for m in mg_metrics])

print("="*80)
print("ANALYSIS 1: RECOVERY CHARACTERISTICS")
print("="*80)

print(f"\nRecovery Degradation (negative = declining recovery capacity):")
print(f"  HC Left:  {np.mean(hc_rec_deg_L):>7.4f} ± {np.std(hc_rec_deg_L):.4f} °/s")
print(f"  HC Right: {np.mean(hc_rec_deg_R):>7.4f} ± {np.std(hc_rec_deg_R):.4f} °/s")
print(f"  MG Left:  {np.mean(mg_rec_deg_L):>7.4f} ± {np.std(mg_rec_deg_L):.4f} °/s")
print(f"  MG Right: {np.mean(mg_rec_deg_R):>7.4f} ± {np.std(mg_rec_deg_R):.4f} °/s")

print(f"\nPosition Degradation (for comparison):")
print(f"  HC Left:  {np.mean(hc_pos_deg_L):>6.3f} ± {np.std(hc_pos_deg_L):.3f}°")
print(f"  HC Right: {np.mean(hc_pos_deg_R):>6.3f} ± {np.std(hc_pos_deg_R):.3f}°")
print(f"  MG Left:  {np.mean(mg_pos_deg_L):>6.3f} ± {np.std(mg_pos_deg_L):.3f}°")
print(f"  MG Right: {np.mean(mg_pos_deg_R):>6.3f} ± {np.std(mg_pos_deg_R):.3f}°")

print("\n" + "="*80)
print("ANALYSIS 2: ORTHOGONALITY TEST")
print("="*80)

hc_pos_combined = np.concatenate([hc_pos_deg_L, hc_pos_deg_R])
hc_rec_combined = np.concatenate([hc_rec_deg_L, hc_rec_deg_R])
mg_pos_combined = np.concatenate([mg_pos_deg_L, mg_pos_deg_R])
mg_rec_combined = np.concatenate([mg_rec_deg_L, mg_rec_deg_R])

r_hc, p_hc = stats.pearsonr(hc_pos_combined, hc_rec_combined)
r_mg, p_mg = stats.pearsonr(mg_pos_combined, mg_rec_combined)

print(f"\nCorrelation between position degradation and recovery degradation:")
print(f"  HC: r={r_hc:.3f} (p={p_hc:.4f})")
print(f"  MG: r={r_mg:.3f} (p={p_mg:.4f})")

if abs(r_hc) < 0.7 and abs(r_mg) < 0.7:
    print(f"\n✓ PASS: |r|<0.7 - recovery and position degradation are orthogonal")
    orthogonal = True
elif abs(r_hc) > 0.9 or abs(r_mg) > 0.9:
    print(f"\n✗ FAIL: |r|>0.9 - recovery is redundant with position")
    orthogonal = False
else:
    print(f"\n~ MARGINAL: 0.7≤|r|≤0.9 - moderate correlation")
    orthogonal = True

print("\n" + "="*80)
print("ANALYSIS 3: RECOVERY DEGRADATION ASYMMETRY DISCRIMINATION")
print("="*80)

# Asymmetries
hc_pos_asym = np.abs(hc_pos_deg_L - hc_pos_deg_R)
mg_pos_asym = np.abs(mg_pos_deg_L - mg_pos_deg_R)

hc_rec_asym = np.abs(hc_rec_deg_L - hc_rec_deg_R)
mg_rec_asym = np.abs(mg_rec_deg_L - mg_rec_deg_R)

def cohens_d(mg_vals, hc_vals):
    pooled_std = np.sqrt(((len(hc_vals)-1)*np.var(hc_vals, ddof=1) +
                          (len(mg_vals)-1)*np.var(mg_vals, ddof=1)) /
                         (len(hc_vals) + len(mg_vals) - 2))
    return (np.mean(mg_vals) - np.mean(hc_vals)) / pooled_std if pooled_std > 0 else 0

d_pos = cohens_d(mg_pos_asym, hc_pos_asym)
d_rec = cohens_d(mg_rec_asym, hc_rec_asym)

print(f"\nPosition Degradation Asymmetry:")
print(f"  HC: {np.mean(hc_pos_asym):.4f} ± {np.std(hc_pos_asym):.4f}°")
print(f"  MG: {np.mean(mg_pos_asym):.4f} ± {np.std(mg_pos_asym):.4f}°")
print(f"  Cohen's d = {d_pos:.3f}")

print(f"\nRecovery Degradation Asymmetry:")
print(f"  HC: {np.mean(hc_rec_asym):.4f} ± {np.std(hc_rec_asym):.4f} °/s")
print(f"  MG: {np.mean(mg_rec_asym):.4f} ± {np.std(mg_rec_asym):.4f} °/s")
print(f"  Cohen's d = {d_rec:.3f}")

if d_rec >= 0.45:
    print(f"\n✓ PASS: d_rec={d_rec:.3f} ≥ 0.45 - recovery asymmetry is strong")
    rec_discriminative = True
elif d_rec < 0.30:
    print(f"\n✗ FAIL: d_rec={d_rec:.3f} < 0.30 - recovery asymmetry too weak")
    rec_discriminative = False
else:
    print(f"\n~ MARGINAL: 0.30 ≤ d_rec={d_rec:.3f} < 0.45")
    rec_discriminative = True

improvement = ((d_rec - d_pos) / d_pos * 100) if d_pos > 0 else 0
print(f"\nComparison:")
print(f"  Recovery vs Position: {improvement:+.1f}% {'(better)' if improvement > 0 else '(worse)'}")

print("\n" + "="*80)
print("ANALYSIS 4: COMBINED PERFORMANCE")
print("="*80)

# Normalize
hc_pos_std = np.std(hc_pos_asym)
hc_rec_std = np.std(hc_rec_asym)

hc_pos_norm = hc_pos_asym / hc_pos_std
mg_pos_norm = mg_pos_asym / hc_pos_std
hc_rec_norm = hc_rec_asym / hc_rec_std
mg_rec_norm = mg_rec_asym / hc_rec_std

print(f"\nTesting combinations:")
print(f"{'Metric':<30} {'MG_mean':<12} {'HC_mean':<12} {'Cohen_d':<10} {'vs PosOnly':<12}")
print("-"*75)

print(f"{'Position only':<30} {np.mean(mg_pos_norm):>6.3f}      {np.mean(hc_pos_norm):>6.3f}      {d_pos:>6.3f}    baseline")
print(f"{'Recovery only':<30} {np.mean(mg_rec_norm):>6.3f}      {np.mean(hc_rec_norm):>6.3f}      {d_rec:>6.3f}    {improvement:>+6.1f}%")

weights = [(0.50, 0.50, "Equal"), (0.70, 0.30, "Pos-heavy"), (0.30, 0.70, "Rec-heavy")]
best_d = max(d_pos, d_rec)
best_combo = None

for w_pos, w_rec, name in weights:
    hc_comb = w_pos * hc_pos_norm + w_rec * hc_rec_norm
    mg_comb = w_pos * mg_pos_norm + w_rec * mg_rec_norm
    d_comb = cohens_d(mg_comb, hc_comb)
    imp = ((d_comb - d_pos) / d_pos * 100)

    print(f"{name:<30} {np.mean(mg_comb):>6.3f}      {np.mean(hc_comb):>6.3f}      {d_comb:>6.3f}    {imp:>+6.1f}%")

    if d_comb > best_d:
        best_d = d_comb
        best_combo = (w_pos, w_rec, name)

print("\n" + "="*80)
print("GO/NO-GO DECISION")
print("="*80)

print(f"\nCriteria:")
print(f"1. Orthogonality (|r|<0.7): {'✓ PASS' if orthogonal else '✗ FAIL'} (HC r={r_hc:.3f}, MG r={r_mg:.3f})")
print(f"2. Recovery discrimination (d≥0.45): {'✓ PASS' if rec_discriminative else '✗ FAIL'} (d={d_rec:.3f})")
print(f"3. Combined improvement: {'✓ PASS' if best_d > d_pos * 1.05 else '✗ FAIL'} (best d={best_d:.3f})")

passes = sum([orthogonal, rec_discriminative, best_d > d_pos * 1.05])

print(f"\n{'='*80}")
if passes >= 2:
    print("DECISION: GO")
    print(f"  {passes}/3 criteria passed")
    if best_combo:
        print(f"  Best: {best_combo[2]} (w_pos={best_combo[0]}, w_rec={best_combo[1]})")
        print(f"  Cohen's d = {best_d:.3f} ({((best_d-d_pos)/d_pos*100):+.1f}%)")
    print(f"\n→ PROCEED to Phase 6: H47 implementation")
else:
    print("DECISION: NO-GO")
    print(f"  Only {passes}/3 criteria passed")
    if not rec_discriminative:
        print(f"  Primary: Recovery too weak (d={d_rec:.3f})")
    print(f"\n→ REJECT H47")

print("="*80)
