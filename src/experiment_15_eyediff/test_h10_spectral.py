#!/usr/bin/env python3
"""
H10: Spectral power asymmetry

Hypothesis: MG micro-tremor/instability shows in frequency domain
even if masked in time domain.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq

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

def spectral_power(signal_data, sampling_rate=1000, freq_band=(1, 50)):
    """Compute spectral power in frequency band."""
    # Remove NaN and detrend
    clean = signal_data[~np.isnan(signal_data)]
    if len(clean) < 100:
        return np.nan

    detrended = clean - np.mean(clean)

    # FFT
    n = len(detrended)
    fft_vals = fft(detrended)
    freqs = fftfreq(n, 1/sampling_rate)

    # Power spectral density
    psd = np.abs(fft_vals)**2

    # Integrate power in band
    freq_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    power = np.sum(psd[freq_mask])

    return power

print("Testing H10: Spectral power asymmetry...")
print("Loading 300 sequences per class...")

raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 50)
sequences = merge_mg_classes(raw_sequences)

hc_seqs = [s for s in sequences if s['class_name'] == 'HC'][:300]
mg_seqs = [s for s in sequences if s['class_name'] == 'MG'][:300]

hc_diffs = []
mg_diffs = []

for seq in hc_seqs + mg_seqs:
    df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)

    # Compute spectral power for each eye (vertical, since it's best)
    power_left = spectral_power(df['LV'].values)
    power_right = spectral_power(df['RV'].values)

    if not np.isnan(power_left) and not np.isnan(power_right):
        diff = np.abs(power_left - power_right)

        if seq['class_name'] == 'HC':
            hc_diffs.append(diff)
        else:
            mg_diffs.append(diff)

print(f"Valid spectral differences: HC={len(hc_diffs)}, MG={len(mg_diffs)}")

hc_mean = np.mean(hc_diffs)
hc_std = np.std(hc_diffs)
mg_mean = np.mean(mg_diffs)
mg_std = np.std(mg_diffs)

pooled_std = np.sqrt((hc_std**2 + mg_std**2) / 2)
cohens_d = (mg_mean - hc_mean) / pooled_std if pooled_std > 0 else 0

print(f"\nH10 Spectral Power Asymmetry (1-50 Hz):")
print(f"  HC: {hc_mean:.2e} ± {hc_std:.2e}")
print(f"  MG: {mg_mean:.2e} ± {mg_std:.2e}")
print(f"  Cohen's d: {cohens_d:.4f}")
print(f"  MG/HC ratio: {mg_mean/hc_mean:.2f}x")

print("\nComparison to best (V-MAD):")
print(f"  V-MAD: d=0.29")
print(f"  Spectral: d={cohens_d:.2f}")

if cohens_d >= 0.5:
    print("\n✓✓✓ BREAKTHROUGH: d ≥ 0.5!")
elif cohens_d > 0.29:
    print("\n✓ IMPROVEMENT")
else:
    print("\n✗ NO improvement")
