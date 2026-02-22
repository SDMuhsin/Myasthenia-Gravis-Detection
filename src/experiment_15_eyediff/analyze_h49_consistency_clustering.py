#!/usr/bin/env python3
"""
CYCLE 49 - PHASE 4: Saccade Consistency Clustering Analysis
Outside-the-box approach #3: Measure whether saccades form distinct quality clusters

HYPOTHESIS: MG shows bimodal saccade performance (good saccades + bad saccades),
while HC shows unimodal (all similar). Measured via k-means clustering silhouette score.

RATIONALE: Intermittent fatigue creates mixed populations. Standard metrics (mean, MAD)
average over mixture, losing information about cluster structure.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

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
print("CYCLE 49: SACCADE CONSISTENCY CLUSTERING ANALYSIS")
print("="*80)

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

def compute_clustering_metrics(LV, RV, TargetV, sample_rate_hz=120):
    """
    Extract clustering metrics (silhouette score) alongside degradation.
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

        if len(up_indices) < 20:  # Need at least 20 for clustering
            return None

        # Extract per-saccade errors
        saccade_errors = []
        for idx in up_indices:
            start = idx + 20
            end = min(idx + 50, len(eye))
            if end - start < 10:
                continue
            error = np.mean(np.abs(eye[start:end] - target[start:end]))
            saccade_errors.append(error)

        if len(saccade_errors) < 20:
            return None

        errors = np.array(saccade_errors).reshape(-1, 1)  # Sklearn expects 2D

        # K-means clustering (k=2)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(errors)

        # Silhouette score: How well-separated are clusters?
        # +1 = perfect clustering, 0 = overlapping, -1 = wrong
        try:
            silhouette = silhouette_score(errors, labels)
        except:
            silhouette = 0  # If clustering fails

        # Calinski-Harabasz score: Variance ratio (higher = better separated)
        try:
            calinski = calinski_harabasz_score(errors, labels)
        except:
            calinski = 0

        # DEGRADATION (for comparison)
        errors_flat = errors.flatten()
        n = len(errors_flat)
        third = max(2, n // 3)
        early_err = np.mean(errors_flat[:third])
        late_err = np.mean(errors_flat[-third:])
        deg = late_err - early_err

        # Cluster sizes (balance)
        cluster0_size = np.sum(labels == 0)
        cluster1_size = np.sum(labels == 1)
        cluster_balance = min(cluster0_size, cluster1_size) / len(labels)  # 0.5 = perfect balance

        return {
            'deg': deg,
            'silhouette': silhouette,
            'calinski': calinski,
            'cluster_balance': cluster_balance,
            'n_saccades': len(errors),
        }

    met_L = process_eye(LV, TargetV)
    met_R = process_eye(RV, TargetV)

    if met_L is None or met_R is None:
        return None

    return {
        'deg_L': met_L['deg'],
        'deg_R': met_R['deg'],
        'silh_L': met_L['silhouette'],
        'silh_R': met_R['silhouette'],
        'cal_L': met_L['calinski'],
        'cal_R': met_R['calinski'],
    }

hc_metrics = []
mg_metrics = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)
    metrics = compute_clustering_metrics(df['LV'].values, df['RV'].values, df['TargetV'].values)
    if metrics is not None:
        if seq['label'] == 0:
            hc_metrics.append(metrics)
        else:
            mg_metrics.append(metrics)

print(f"Valid sequences: HC={len(hc_metrics)}, MG={len(mg_metrics)}\n")

# Extract arrays
hc_deg_L = np.array([m['deg_L'] for m in hc_metrics])
hc_deg_R = np.array([m['deg_R'] for m in hc_metrics])
hc_silh_L = np.array([m['silh_L'] for m in hc_metrics])
hc_silh_R = np.array([m['silh_R'] for m in hc_metrics])

mg_deg_L = np.array([m['deg_L'] for m in mg_metrics])
mg_deg_R = np.array([m['deg_R'] for m in mg_metrics])
mg_silh_L = np.array([m['silh_L'] for m in mg_metrics])
mg_silh_R = np.array([m['silh_R'] for m in mg_metrics])

print("="*80)
print("ANALYSIS 1: CLUSTERING CHARACTERISTICS")
print("="*80)

print(f"\nSilhouette Score (+1=perfect clustering, 0=overlapping):")
print(f"  HC Left:  {np.mean(hc_silh_L):>7.3f} ± {np.std(hc_silh_L):.3f}")
print(f"  HC Right: {np.mean(hc_silh_R):>7.3f} ± {np.std(hc_silh_R):.3f}")
print(f"  MG Left:  {np.mean(mg_silh_L):>7.3f} ± {np.std(mg_silh_L):.3f}")
print(f"  MG Right: {np.mean(mg_silh_R):>7.3f} ± {np.std(mg_silh_R):.3f}")

print(f"\nDegradation (for comparison):")
print(f"  HC Left:  {np.mean(hc_deg_L):>6.3f} ± {np.std(hc_deg_L):.3f}°")
print(f"  HC Right: {np.mean(hc_deg_R):>6.3f} ± {np.std(hc_deg_R):.3f}°")
print(f"  MG Left:  {np.mean(mg_deg_L):>6.3f} ± {np.std(mg_deg_L):.3f}°")
print(f"  MG Right: {np.mean(mg_deg_R):>6.3f} ± {np.std(mg_deg_R):.3f}°")

print("\n" + "="*80)
print("ANALYSIS 2: ORTHOGONALITY TEST")
print("="*80)

hc_deg_combined = np.concatenate([hc_deg_L, hc_deg_R])
hc_silh_combined = np.concatenate([hc_silh_L, hc_silh_R])
mg_deg_combined = np.concatenate([mg_deg_L, mg_deg_R])
mg_silh_combined = np.concatenate([mg_silh_L, mg_silh_R])

r_hc, p_hc = stats.pearsonr(hc_deg_combined, hc_silh_combined)
r_mg, p_mg = stats.pearsonr(mg_deg_combined, mg_silh_combined)

print(f"\nCorrelation between degradation and silhouette:")
print(f"  HC: r={r_hc:.3f} (p={p_hc:.4f})")
print(f"  MG: r={r_mg:.3f} (p={p_mg:.4f})")

if abs(r_hc) < 0.7 and abs(r_mg) < 0.7:
    print(f"\n✓ PASS: |r|<0.7 - silhouette and degradation orthogonal")
    orthogonal = True
else:
    print(f"\n✗ FAIL: |r|≥0.7 - silhouette correlated with degradation")
    orthogonal = False

print("\n" + "="*80)
print("ANALYSIS 3: SILHOUETTE ASYMMETRY DISCRIMINATION")
print("="*80)

def cohens_d(mg_vals, hc_vals):
    pooled_std = np.sqrt(((len(hc_vals)-1)*np.var(hc_vals, ddof=1) +
                          (len(mg_vals)-1)*np.var(mg_vals, ddof=1)) /
                         (len(hc_vals) + len(mg_vals) - 2))
    return (np.mean(mg_vals) - np.mean(hc_vals)) / pooled_std if pooled_std > 0 else 0

hc_deg_asym = np.abs(hc_deg_L - hc_deg_R)
mg_deg_asym = np.abs(mg_deg_L - mg_deg_R)

hc_silh_asym = np.abs(hc_silh_L - hc_silh_R)
mg_silh_asym = np.abs(mg_silh_L - mg_silh_R)

d_deg = cohens_d(mg_deg_asym, hc_deg_asym)
d_silh = cohens_d(mg_silh_asym, hc_silh_asym)

print(f"\nDegradation Asymmetry:")
print(f"  HC: {np.mean(hc_deg_asym):.4f} ± {np.std(hc_deg_asym):.4f}°")
print(f"  MG: {np.mean(mg_deg_asym):.4f} ± {np.std(mg_deg_asym):.4f}°")
print(f"  Cohen's d = {d_deg:.3f}")

print(f"\nSilhouette Asymmetry:")
print(f"  HC: {np.mean(hc_silh_asym):.4f} ± {np.std(hc_silh_asym):.4f}")
print(f"  MG: {np.mean(mg_silh_asym):.4f} ± {np.std(mg_silh_asym):.4f}")
print(f"  Cohen's d = {d_silh:.3f}")

improvement = ((d_silh - d_deg) / d_deg * 100) if d_deg > 0 else 0
print(f"\nComparison:")
print(f"  Silhouette vs Degradation: {improvement:+.1f}% {'(better)' if improvement > 0 else '(worse)'}")

if d_silh >= 0.40:
    print(f"  ✓ PASS: Silhouette discrimination strong")
    silh_discriminative = True
elif d_silh < 0.30:
    print(f"  ✗ FAIL: Silhouette discrimination too weak")
    silh_discriminative = False
else:
    print(f"  ~ MARGINAL")
    silh_discriminative = True

print("\n" + "="*80)
print("GO/NO-GO DECISION")
print("="*80)

print(f"\nCriteria:")
print(f"1. Orthogonality (|r|<0.7): {'✓ PASS' if orthogonal else '✗ FAIL'}")
print(f"2. Silhouette discrimination (d≥0.40): {'✓ PASS' if silh_discriminative else '✗ FAIL'} (d={d_silh:.3f})")
print(f"3. Improvement over degradation: {'✓ PASS' if d_silh > d_deg * 1.05 else '✗ FAIL'}")

passes = sum([orthogonal, silh_discriminative, d_silh > d_deg * 1.05])

if passes >= 2:
    print(f"\nDECISION: GO")
    print(f"  {passes}/3 criteria passed")
    print(f"  Clustering structure provides discriminative signal")
else:
    print(f"\nDECISION: NO-GO")
    print(f"  Only {passes}/3 criteria passed")
    print(f"  Saccade clustering doesn't capture MG asymmetry")

print("="*80)
