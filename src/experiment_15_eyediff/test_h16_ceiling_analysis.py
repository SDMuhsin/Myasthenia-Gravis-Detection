#!/usr/bin/env python3
"""
H16 Pre-Analysis: Understand the ceiling at d=0.41

Why are we stuck? Analyze the fundamental limitation:
1. What is the distribution of HC asymmetry?
2. What percentage of HC patients have asymmetry > 0?
3. Is there a subpopulation of HC with high natural asymmetry?
4. Can we identify and filter outlier HC cases?
5. What is the best theoretically achievable effect size given HC baseline?
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_timeseries_data, merge_mg_classes
from equations import h2_mad_variability

BASE_DIR = './data'
CLASS_DEFINITIONS = {
    'HC': {'path': 'Healthy control', 'label': 0},
    'MG': {'path': 'Definite MG', 'label': 1},
    'Probable_MG': {'path': 'Probable MG', 'label': 1},
}
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']

print("="*80)
print("H16 Pre-Analysis: Understanding the Effect Size Ceiling")
print("="*80)

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

def compute_early_late_degradation(eye_pos, target_pos):
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    if np.sum(valid) < 50:
        return np.nan
    eye_clean = eye_pos[valid]
    target_clean = target_pos[valid]
    error = np.abs(eye_clean - target_clean)
    n = len(error)
    early_n = int(n * 0.2)
    late_n = int(n * 0.2)
    early_mean = np.mean(error[:early_n])
    late_mean = np.mean(error[-late_n:])
    return late_mean - early_mean

# Compute H11 combined metric
mad_left = []
mad_right = []
deg_left = []
deg_right = []
class_labels = []
sequence_ids = []

for idx, seq in enumerate(sequences):
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    left_mad_dict = h2_mad_variability(df['LV'].values, df['TargetV'].values)
    right_mad_dict = h2_mad_variability(df['RV'].values, df['TargetV'].values)
    left_mad_val = left_mad_dict['mad_position']
    right_mad_val = right_mad_dict['mad_position']

    left_deg_val = compute_early_late_degradation(df['LV'].values, df['TargetV'].values)
    right_deg_val = compute_early_late_degradation(df['RV'].values, df['TargetV'].values)

    if not (np.isnan(left_mad_val) or np.isnan(right_mad_val) or
            np.isnan(left_deg_val) or np.isnan(right_deg_val)):
        mad_left.append(left_mad_val)
        mad_right.append(right_mad_val)
        deg_left.append(left_deg_val)
        deg_right.append(right_deg_val)
        class_labels.append(1 if seq['class_name'] == 'MG' else 0)
        sequence_ids.append(idx)

mad_left = np.array(mad_left)
mad_right = np.array(mad_right)
deg_left = np.array(deg_left)
deg_right = np.array(deg_right)
class_labels = np.array(class_labels)

# H11 combined metric
combined_left = 0.45 * mad_left + 0.55 * deg_left
combined_right = 0.45 * mad_right + 0.55 * deg_right
asymmetry = np.abs(combined_left - combined_right)

hc_asym = asymmetry[class_labels == 0]
mg_asym = asymmetry[class_labels == 1]

print(f"\n1. DISTRIBUTION ANALYSIS")
print(f"   HC asymmetry: mean={np.mean(hc_asym):.3f}, median={np.median(hc_asym):.3f}, std={np.std(hc_asym):.3f}")
print(f"   MG asymmetry: mean={np.mean(mg_asym):.3f}, median={np.median(mg_asym):.3f}, std={np.std(mg_asym):.3f}")
print(f"   HC range: [{np.min(hc_asym):.3f}, {np.max(hc_asym):.3f}]")
print(f"   MG range: [{np.min(mg_asym):.3f}, {np.max(mg_asym):.3f}]")

print(f"\n2. HC BASELINE ASYMMETRY ANALYSIS")
hc_near_zero = np.sum(hc_asym < 0.1)
hc_small = np.sum((hc_asym >= 0.1) & (hc_asym < 0.5))
hc_moderate = np.sum((hc_asym >= 0.5) & (hc_asym < 1.0))
hc_large = np.sum(hc_asym >= 1.0)
print(f"   HC near-zero (<0.1): {hc_near_zero}/{len(hc_asym)} ({100*hc_near_zero/len(hc_asym):.1f}%)")
print(f"   HC small (0.1-0.5): {hc_small}/{len(hc_asym)} ({100*hc_small/len(hc_asym):.1f}%)")
print(f"   HC moderate (0.5-1.0): {hc_moderate}/{len(hc_asym)} ({100*hc_moderate/len(hc_asym):.1f}%)")
print(f"   HC large (≥1.0): {hc_large}/{len(hc_asym)} ({100*hc_large/len(hc_asym):.1f}%)")

print(f"\n3. MG ASYMMETRY DISTRIBUTION")
mg_near_zero = np.sum(mg_asym < 0.1)
mg_small = np.sum((mg_asym >= 0.1) & (mg_asym < 0.5))
mg_moderate = np.sum((mg_asym >= 0.5) & (mg_asym < 1.0))
mg_large = np.sum(mg_asym >= 1.0)
print(f"   MG near-zero (<0.1): {mg_near_zero}/{len(mg_asym)} ({100*mg_near_zero/len(mg_asym):.1f}%)")
print(f"   MG small (0.1-0.5): {mg_small}/{len(mg_asym)} ({100*mg_small/len(mg_asym):.1f}%)")
print(f"   MG moderate (0.5-1.0): {mg_moderate}/{len(mg_asym)} ({100*mg_moderate/len(mg_asym):.1f}%)")
print(f"   MG large (≥1.0): {mg_large}/{len(mg_asym)} ({100*mg_large/len(mg_asym):.1f}%)")

print(f"\n4. OVERLAP ANALYSIS")
hc_percentiles = [25, 50, 75, 90, 95]
mg_percentiles = [5, 10, 25, 50]
print(f"   HC percentiles:")
for p in hc_percentiles:
    val = np.percentile(hc_asym, p)
    print(f"      {p}th: {val:.3f}")
print(f"   MG percentiles:")
for p in mg_percentiles:
    val = np.percentile(mg_asym, p)
    print(f"      {p}th: {val:.3f}")

# Check if low-MG overlaps with high-HC
hc_75 = np.percentile(hc_asym, 75)
mg_25 = np.percentile(mg_asym, 25)
print(f"\n   HC 75th percentile: {hc_75:.3f}")
print(f"   MG 25th percentile: {mg_25:.3f}")
print(f"   Overlap: {'YES' if mg_25 < hc_75 else 'NO'}")

print(f"\n5. THEORETICAL MAXIMUM EFFECT SIZE")
# If we could perfectly separate distributions
# Best case: shift MG distribution up while keeping spread constant
hc_std = np.std(hc_asym)
mg_std = np.std(mg_asym)
pooled_std = np.sqrt((hc_std**2 + mg_std**2) / 2)
current_d = (np.mean(mg_asym) - np.mean(hc_asym)) / pooled_std

# What d would we get if MG mean was at MG 75th percentile?
mg_75 = np.percentile(mg_asym, 75)
theoretical_d_75 = (mg_75 - np.mean(hc_asym)) / pooled_std
print(f"   Current d: {current_d:.3f}")
print(f"   Theoretical d if MG mean = MG p75: {theoretical_d_75:.3f}")

# What d would we get if we could remove HC baseline (shift HC down by median)?
hc_shifted_mean = 0.0
theoretical_d_no_hc_baseline = (np.mean(mg_asym) - hc_shifted_mean) / pooled_std
print(f"   Theoretical d if HC baseline = 0: {theoretical_d_no_hc_baseline:.3f}")

print(f"\n6. COMPONENT ANALYSIS")
# Which component contributes more to the ceiling?
mad_asym = np.abs(mad_left - mad_right)
deg_asym = np.abs(deg_left - deg_right)

hc_mad_asym = mad_asym[class_labels == 0]
hc_deg_asym = deg_asym[class_labels == 0]

print(f"   HC MAD asymmetry: {np.mean(hc_mad_asym):.3f} ± {np.std(hc_mad_asym):.3f}")
print(f"   HC Degradation asymmetry: {np.mean(hc_deg_asym):.3f} ± {np.std(hc_deg_asym):.3f}")
print(f"   HC Combined asymmetry: {np.mean(hc_asym):.3f} ± {np.std(hc_asym):.3f}")

# Correlation between components in HC
from scipy.stats import pearsonr
r_mad_deg_hc, p_corr_hc = pearsonr(hc_mad_asym, hc_deg_asym)
print(f"   HC: Correlation(MAD_asym, Deg_asym) = {r_mad_deg_hc:.3f} (p={p_corr_hc:.4f})")

print(f"\n7. PERCENTAGE OF ZERO HC")
# What percentage of HC have asymmetry effectively zero?
hc_zero_count = np.sum(hc_asym < 0.05)
print(f"   HC with asymmetry < 0.05: {hc_zero_count}/{len(hc_asym)} ({100*hc_zero_count/len(hc_asym):.1f}%)")

print("="*80)
