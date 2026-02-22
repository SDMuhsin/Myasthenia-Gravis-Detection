#!/usr/bin/env python3
"""Test H8 assumption: Does MG show fluctuating asymmetry?"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_timeseries_data, merge_mg_classes

# Configuration
BASE_DIR = './data'
CLASS_DEFINITIONS = {
    'HC': {'path': 'Healthy control', 'label': 0},
    'MG': {'path': 'Definite MG', 'label': 1},
    'Probable_MG': {'path': 'Probable MG', 'label': 1},
}
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']

def compute_cv_of_asymmetry(lh, rh):
    """Compute CV of |LH - RH| time series."""
    asymmetry = np.abs(lh - rh)
    asymmetry_clean = asymmetry[~np.isnan(asymmetry)]

    if len(asymmetry_clean) < 10:
        return np.nan

    mean_asym = np.mean(asymmetry_clean)
    std_asym = np.std(asymmetry_clean)

    if mean_asym < 0.1:  # Too small, CV unstable
        return np.nan

    cv = std_asym / mean_asym
    return cv


# Load data
print("Loading 200 sequences per class...")
raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

hc_seqs = [s for s in sequences if s['class_name'] == 'HC'][:200]
mg_seqs = [s for s in sequences if s['class_name'] == 'MG'][:200]

hc_cvs = []
mg_cvs = []

for seq in hc_seqs:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)
    cv = compute_cv_of_asymmetry(df['LH'].values, df['RH'].values)
    if not np.isnan(cv):
        hc_cvs.append(cv)

for seq in mg_seqs:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)
    cv = compute_cv_of_asymmetry(df['LH'].values, df['RH'].values)
    if not np.isnan(cv):
        mg_cvs.append(cv)

print(f"\nValid CV measurements:")
print(f"  HC: {len(hc_cvs)}/{len(hc_seqs)}")
print(f"  MG: {len(mg_cvs)}/{len(mg_seqs)}")

hc_mean = np.mean(hc_cvs)
hc_std = np.std(hc_cvs)
mg_mean = np.mean(mg_cvs)
mg_std = np.std(mg_cvs)

pooled_std = np.sqrt((hc_std**2 + mg_std**2) / 2)
cohens_d = (mg_mean - hc_mean) / pooled_std if pooled_std > 0 else 0

print(f"\nCV of Asymmetry:")
print(f"  HC: {hc_mean:.4f} ± {hc_std:.4f}")
print(f"  MG: {mg_mean:.4f} ± {mg_std:.4f}")
print(f"  Cohen's d: {cohens_d:.4f}")
print(f"  MG/HC ratio: {mg_mean/hc_mean:.2f}x")

# Check mean asymmetry distribution
mean_asyms_hc = []
mean_asyms_mg = []

for seq in hc_seqs:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)
    asymmetry = np.abs(df['LH'].values - df['RH'].values)
    asymmetry_clean = asymmetry[~np.isnan(asymmetry)]
    if len(asymmetry_clean) > 0:
        mean_asyms_hc.append(np.mean(asymmetry_clean))

for seq in mg_seqs:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)
    asymmetry = np.abs(df['LH'].values - df['RH'].values)
    asymmetry_clean = asymmetry[~np.isnan(asymmetry)]
    if len(asymmetry_clean) > 0:
        mean_asyms_mg.append(np.mean(asymmetry_clean))

print(f"\nMean asymmetry (for stability check):")
print(f"  HC: {np.mean(mean_asyms_hc):.2f}° (median: {np.median(mean_asyms_hc):.2f}°)")
print(f"  MG: {np.mean(mean_asyms_mg):.2f}° (median: {np.median(mean_asyms_mg):.2f}°)")
print(f"  % with mean < 0.1°: HC={100*np.sum(np.array(mean_asyms_hc)<0.1)/len(mean_asyms_hc):.1f}%, MG={100*np.sum(np.array(mean_asyms_mg)<0.1)/len(mean_asyms_mg):.1f}%")

print("\n" + "="*80)
if cohens_d < 0.2:
    print("VERDICT: REJECT H8 - Negligible effect size")
elif cohens_d < 0.5:
    print("VERDICT: WEAK - Small effect, unlikely to reach d≥0.5 target")
else:
    print("VERDICT: PROMISING - Medium+ effect, proceed to implementation")
print("="*80)
