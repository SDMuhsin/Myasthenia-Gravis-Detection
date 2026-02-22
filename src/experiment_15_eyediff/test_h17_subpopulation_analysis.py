#!/usr/bin/env python3
"""
H17 Pre-Analysis: Analyze MG subpopulations
Are there qualitative differences between high-asymmetry MG vs high-asymmetry HC?
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

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
print("H17 Pre-Analysis: MG Subpopulation Characteristics")
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
results = []

for seq in sequences:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    left_mad_dict = h2_mad_variability(df['LV'].values, df['TargetV'].values)
    right_mad_dict = h2_mad_variability(df['RV'].values, df['TargetV'].values)
    left_mad_val = left_mad_dict['mad_position']
    right_mad_val = right_mad_dict['mad_position']

    left_deg_val = compute_early_late_degradation(df['LV'].values, df['TargetV'].values)
    right_deg_val = compute_early_late_degradation(df['RV'].values, df['TargetV'].values)

    if not (np.isnan(left_mad_val) or np.isnan(right_mad_val) or
            np.isnan(left_deg_val) or np.isnan(right_deg_val)):

        combined_left = 0.45 * left_mad_val + 0.55 * left_deg_val
        combined_right = 0.45 * right_mad_val + 0.55 * right_deg_val
        asymmetry = abs(combined_left - combined_right)

        # Additional features
        worse_eye_combined = max(combined_left, combined_right)
        better_eye_combined = min(combined_left, combined_right)
        ratio = worse_eye_combined / (better_eye_combined + 0.01)

        # Degradation features
        worse_deg = max(abs(left_deg_val), abs(right_deg_val))
        both_degrade = (left_deg_val > 0) and (right_deg_val > 0)

        results.append({
            'class': 1 if seq['class_name'] == 'MG' else 0,
            'asymmetry': asymmetry,
            'worse_eye': worse_eye_combined,
            'better_eye': better_eye_combined,
            'ratio': ratio,
            'worse_deg': worse_deg,
            'both_degrade': both_degrade,
            'left_mad': left_mad_val,
            'right_mad': right_mad_val,
            'left_deg': left_deg_val,
            'right_deg': right_deg_val
        })

results_df = pd.DataFrame(results)
hc_df = results_df[results_df['class'] == 0]
mg_df = results_df[results_df['class'] == 1]

# Define subpopulations
hc_high_asym = hc_df[hc_df['asymmetry'] > hc_df['asymmetry'].quantile(0.75)]
hc_low_asym = hc_df[hc_df['asymmetry'] < hc_df['asymmetry'].quantile(0.25)]
mg_high_asym = mg_df[mg_df['asymmetry'] > mg_df['asymmetry'].quantile(0.75)]
mg_low_asym = mg_df[mg_df['asymmetry'] < mg_df['asymmetry'].quantile(0.25)]

print(f"\nSubpopulation sizes:")
print(f"  HC low asymmetry (p25): {len(hc_low_asym)}")
print(f"  HC high asymmetry (p75): {len(hc_high_asym)}")
print(f"  MG low asymmetry (p25): {len(mg_low_asym)}")
print(f"  MG high asymmetry (p75): {len(mg_high_asym)}")

print(f"\n1. ASYMMETRY CHARACTERISTICS:")
print(f"\n   HC High-Asymmetry:")
print(f"     Asymmetry: {hc_high_asym['asymmetry'].mean():.3f} ± {hc_high_asym['asymmetry'].std():.3f}")
print(f"     Worse eye: {hc_high_asym['worse_eye'].mean():.3f}")
print(f"     Better eye: {hc_high_asym['better_eye'].mean():.3f}")
print(f"     Ratio: {hc_high_asym['ratio'].mean():.3f}")
print(f"     Both degrade: {100*hc_high_asym['both_degrade'].mean():.1f}%")

print(f"\n   MG High-Asymmetry:")
print(f"     Asymmetry: {mg_high_asym['asymmetry'].mean():.3f} ± {mg_high_asym['asymmetry'].std():.3f}")
print(f"     Worse eye: {mg_high_asym['worse_eye'].mean():.3f}")
print(f"     Better eye: {mg_high_asym['better_eye'].mean():.3f}")
print(f"     Ratio: {mg_high_asym['ratio'].mean():.3f}")
print(f"     Both degrade: {100*mg_high_asym['both_degrade'].mean():.1f}%")

print(f"\n2. KEY DISCRIMINATORS (High-Asym MG vs High-Asym HC):")
print(f"   Worse eye performance:")
print(f"     HC: {hc_high_asym['worse_eye'].mean():.3f}")
print(f"     MG: {mg_high_asym['worse_eye'].mean():.3f}")
print(f"     MG/HC ratio: {mg_high_asym['worse_eye'].mean() / hc_high_asym['worse_eye'].mean():.2f}x")

print(f"\n   Better eye performance:")
print(f"     HC: {hc_high_asym['better_eye'].mean():.3f}")
print(f"     MG: {mg_high_asym['better_eye'].mean():.3f}")
print(f"     MG/HC ratio: {mg_high_asym['better_eye'].mean() / hc_high_asym['better_eye'].mean():.2f}x")

print(f"\n   Asymmetry ratio:")
print(f"     HC: {hc_high_asym['ratio'].mean():.3f}")
print(f"     MG: {mg_high_asym['ratio'].mean():.3f}")
print(f"     MG/HC ratio: {mg_high_asym['ratio'].mean() / hc_high_asym['ratio'].mean():.2f}x")

print(f"\n   Worst degradation:")
print(f"     HC: {hc_high_asym['worse_deg'].mean():.3f}")
print(f"     MG: {mg_high_asym['worse_deg'].mean():.3f}")
print(f"     MG/HC ratio: {mg_high_asym['worse_deg'].mean() / hc_high_asym['worse_deg'].mean():.2f}x")

print(f"\n3. UNILATERAL vs BILATERAL PATTERN:")
print(f"   Both eyes degrade:")
print(f"     HC high-asym: {100*hc_high_asym['both_degrade'].mean():.1f}%")
print(f"     MG high-asym: {100*mg_high_asym['both_degrade'].mean():.1f}%")
print(f"     MG low-asym: {100*mg_low_asym['both_degrade'].mean():.1f}%")

print(f"\n4. ABSOLUTE PERFORMANCE LEVELS:")
# Check if high-asym MG has worse absolute performance
print(f"   Average of both eyes (MAD):")
print(f"     HC high-asym: {0.5*(hc_high_asym['left_mad'].mean() + hc_high_asym['right_mad'].mean()):.3f}")
print(f"     MG high-asym: {0.5*(mg_high_asym['left_mad'].mean() + mg_high_asym['right_mad'].mean()):.3f}")
print(f"     MG low-asym: {0.5*(mg_low_asym['left_mad'].mean() + mg_low_asym['right_mad'].mean()):.3f}")

print(f"\n5. HYPOTHESIS: Can we distinguish high-asym MG from high-asym HC?")
# Test if worse_eye metric differs
_, p_worse = stats.mannwhitneyu(mg_high_asym['worse_eye'], hc_high_asym['worse_eye'], alternative='greater')
_, p_ratio = stats.mannwhitneyu(mg_high_asym['ratio'], hc_high_asym['ratio'], alternative='greater')

print(f"   Worse eye test: MG > HC, p={p_worse:.4f} {'✓' if p_worse < 0.05 else '✗'}")
print(f"   Ratio test: MG > HC, p={p_ratio:.4f} {'✓' if p_ratio < 0.05 else '✗'}")

# Effect size for high-asymmetry subpopulations only
d_high_asym = (mg_high_asym['asymmetry'].mean() - hc_high_asym['asymmetry'].mean()) / \
               np.sqrt((mg_high_asym['asymmetry'].std()**2 + hc_high_asym['asymmetry'].std()**2) / 2)
print(f"\n   Effect size within high-asymmetry groups: d={d_high_asym:.3f}")

print("="*80)
