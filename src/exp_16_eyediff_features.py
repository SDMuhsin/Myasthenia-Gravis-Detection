#!/usr/bin/env python3
"""
Experiment 16: Eye Difference Feature Discriminatory Power Study
Binary Classification (HC vs MG including Probable MG)

Tests the 11 metrics from Experiment 15 (Eye Difference Detection) as neural network features,
plus additional frequency domain features.
Each metric is added as additional channel(s) to the baseline and tested independently.

Methodology (same as Experiment 14):
1. Train baseline ONCE, reuse for all feature comparisons
2. Set ALL random seeds (numpy, torch, torch.cuda)
3. Use 5-fold CV
4. Run 3 independent trials per experiment

Metrics tested:
- TTT1: First entry 3° latency asymmetry
- TTT2: Sustained 100ms latency asymmetry
- TTT3: First entry 4° latency asymmetry
- TTT4: Peak velocity latency asymmetry
- TTT5: 90% settling latency asymmetry
- FAT1: Error degradation asymmetry
- FAT2: Latency degradation asymmetry
- FAT3: Error slope asymmetry
- FAT4: Latency slope asymmetry
- FAT5: Variability increase asymmetry
- H38b: Composite metric asymmetry
- FFT: Frequency domain features (tremor band power asymmetry)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats
from tqdm import tqdm
import pandas as pd

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.data_loading import load_raw_sequences_and_labels
from utils.modeling import create_results_directory
from utils.deep_learning import (SaccadeRNN_Small, SaccadeStandardScaler, EarlyStopper,
                                subsample_data, train_epoch, evaluate_epoch)

# --- Configuration ---
BASE_DIR = './data'
BINARY_CLASS_DEFINITIONS = {
    'HC': {'path': 'Healthy control', 'label': 0},
    'MG': {'path': 'Definite MG', 'label': 1},
    'Probable_MG': {'path': 'Probable MG', 'label': 1},
}

FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']
CSV_ENCODING = 'utf-16-le'
CSV_SEPARATOR = ','
MIN_SEQ_LEN_THRESHOLD = 50
RESULTS_DIR = './results/exp_16_eyediff_features'
RANDOM_STATE = 42

# Deep Learning Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUBSAMPLE_FACTOR = 10
TARGET_SEQ_LEN_PERCENTILE = 90
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 8
NUM_FOLDS = 5
NUM_TRIALS = 3

# Saccade detection parameters
SACCADE_THRESHOLD = 5.0  # degrees - target jump threshold
SAMPLE_RATE = 120  # Hz


def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_binary_data(raw_items):
    """Prepare data for binary classification."""
    binary_items = []
    for item in raw_items:
        if item['class_name'] in ['MG', 'Probable_MG']:
            new_item = item.copy()
            new_item['class_name'] = 'MG'
            new_item['label'] = 1
            binary_items.append(new_item)
        elif item['class_name'] == 'HC':
            binary_items.append(item)
    return binary_items


def add_baseline_channels(binary_items):
    """Add 8 engineered channels to 6 raw = 14 total."""
    enhanced_items = []
    for item in binary_items:
        data = item['data']

        # Velocities
        lh_vel = np.gradient(data[:, 0])
        rh_vel = np.gradient(data[:, 1])
        lv_vel = np.gradient(data[:, 2])
        rv_vel = np.gradient(data[:, 3])

        # Errors
        errorh_l = data[:, 0] - data[:, 4]
        errorh_r = data[:, 1] - data[:, 4]
        errorv_l = data[:, 2] - data[:, 5]
        errorv_r = data[:, 3] - data[:, 5]

        enhanced_data = np.column_stack([
            data,
            lh_vel, rh_vel, lv_vel, rv_vel,
            errorh_l, errorh_r, errorv_l, errorv_r
        ])

        new_item = item.copy()
        new_item['data'] = enhanced_data
        enhanced_items.append(new_item)

    return enhanced_items


def detect_vertical_saccades(target_v):
    """Detect upward saccade onset indices from vertical target position."""
    target_diff = np.diff(target_v)
    up_indices = np.where(target_diff > SACCADE_THRESHOLD)[0] + 1
    return up_indices


def compute_ttt1_features(data):
    """
    TTT1: First entry 3° tolerance latency.
    Returns time-series channels for L/R latency and asymmetry.
    """
    lv = data[:, 2]  # Left vertical
    rv = data[:, 3]  # Right vertical
    target_v = data[:, 5]  # Target vertical

    saccade_indices = detect_vertical_saccades(target_v)
    n_samples = len(lv)

    # Initialize output channels
    lat_l_channel = np.zeros(n_samples)
    lat_r_channel = np.zeros(n_samples)
    asym_channel = np.zeros(n_samples)

    tolerance = 3.0

    for i, idx in enumerate(saccade_indices):
        if idx >= n_samples:
            continue

        # Search window: 500ms after saccade onset
        window_end = min(idx + 60, n_samples)  # 60 samples = 500ms at 120Hz

        # Find first entry for left eye
        lat_l = np.nan
        for t in range(idx, window_end):
            if abs(lv[t] - target_v[t]) <= tolerance:
                lat_l = (t - idx) / SAMPLE_RATE * 1000  # ms
                break

        # Find first entry for right eye
        lat_r = np.nan
        for t in range(idx, window_end):
            if abs(rv[t] - target_v[t]) <= tolerance:
                lat_r = (t - idx) / SAMPLE_RATE * 1000  # ms
                break

        # Fill channel values for this saccade region
        next_idx = saccade_indices[i+1] if i+1 < len(saccade_indices) else n_samples
        if not np.isnan(lat_l):
            lat_l_channel[idx:next_idx] = lat_l
        if not np.isnan(lat_r):
            lat_r_channel[idx:next_idx] = lat_r
        if not np.isnan(lat_l) and not np.isnan(lat_r):
            asym_channel[idx:next_idx] = abs(lat_l - lat_r)

    return lat_l_channel, lat_r_channel, asym_channel


def compute_ttt2_features(data):
    """
    TTT2: Sustained presence (100ms within 3°) latency.
    """
    lv = data[:, 2]
    rv = data[:, 3]
    target_v = data[:, 5]

    saccade_indices = detect_vertical_saccades(target_v)
    n_samples = len(lv)

    lat_l_channel = np.zeros(n_samples)
    lat_r_channel = np.zeros(n_samples)
    asym_channel = np.zeros(n_samples)

    tolerance = 3.0
    sustain_samples = 12  # 100ms at 120Hz

    for i, idx in enumerate(saccade_indices):
        if idx >= n_samples:
            continue

        window_end = min(idx + 120, n_samples)  # 1s window

        # Find sustained entry for left eye
        lat_l = np.nan
        for t in range(idx, window_end - sustain_samples):
            if all(abs(lv[t:t+sustain_samples] - target_v[t:t+sustain_samples]) <= tolerance):
                lat_l = (t - idx) / SAMPLE_RATE * 1000
                break

        # Find sustained entry for right eye
        lat_r = np.nan
        for t in range(idx, window_end - sustain_samples):
            if all(abs(rv[t:t+sustain_samples] - target_v[t:t+sustain_samples]) <= tolerance):
                lat_r = (t - idx) / SAMPLE_RATE * 1000
                break

        next_idx = saccade_indices[i+1] if i+1 < len(saccade_indices) else n_samples
        if not np.isnan(lat_l):
            lat_l_channel[idx:next_idx] = lat_l
        if not np.isnan(lat_r):
            lat_r_channel[idx:next_idx] = lat_r
        if not np.isnan(lat_l) and not np.isnan(lat_r):
            asym_channel[idx:next_idx] = abs(lat_l - lat_r)

    return lat_l_channel, lat_r_channel, asym_channel


def compute_ttt3_features(data):
    """
    TTT3: First entry 4° tolerance latency.
    """
    lv = data[:, 2]
    rv = data[:, 3]
    target_v = data[:, 5]

    saccade_indices = detect_vertical_saccades(target_v)
    n_samples = len(lv)

    lat_l_channel = np.zeros(n_samples)
    lat_r_channel = np.zeros(n_samples)
    asym_channel = np.zeros(n_samples)

    tolerance = 4.0

    for i, idx in enumerate(saccade_indices):
        if idx >= n_samples:
            continue

        window_end = min(idx + 60, n_samples)

        lat_l = np.nan
        for t in range(idx, window_end):
            if abs(lv[t] - target_v[t]) <= tolerance:
                lat_l = (t - idx) / SAMPLE_RATE * 1000
                break

        lat_r = np.nan
        for t in range(idx, window_end):
            if abs(rv[t] - target_v[t]) <= tolerance:
                lat_r = (t - idx) / SAMPLE_RATE * 1000
                break

        next_idx = saccade_indices[i+1] if i+1 < len(saccade_indices) else n_samples
        if not np.isnan(lat_l):
            lat_l_channel[idx:next_idx] = lat_l
        if not np.isnan(lat_r):
            lat_r_channel[idx:next_idx] = lat_r
        if not np.isnan(lat_l) and not np.isnan(lat_r):
            asym_channel[idx:next_idx] = abs(lat_l - lat_r)

    return lat_l_channel, lat_r_channel, asym_channel


def compute_ttt4_features(data):
    """
    TTT4: Peak velocity latency.
    """
    lv = data[:, 2]
    rv = data[:, 3]
    target_v = data[:, 5]

    lv_vel = np.abs(np.gradient(lv))
    rv_vel = np.abs(np.gradient(rv))

    saccade_indices = detect_vertical_saccades(target_v)
    n_samples = len(lv)

    lat_l_channel = np.zeros(n_samples)
    lat_r_channel = np.zeros(n_samples)
    asym_channel = np.zeros(n_samples)

    for i, idx in enumerate(saccade_indices):
        if idx >= n_samples:
            continue

        window_end = min(idx + 30, n_samples)  # 250ms window

        if window_end > idx:
            peak_l_idx = idx + np.argmax(lv_vel[idx:window_end])
            peak_r_idx = idx + np.argmax(rv_vel[idx:window_end])

            lat_l = (peak_l_idx - idx) / SAMPLE_RATE * 1000
            lat_r = (peak_r_idx - idx) / SAMPLE_RATE * 1000

            next_idx = saccade_indices[i+1] if i+1 < len(saccade_indices) else n_samples
            lat_l_channel[idx:next_idx] = lat_l
            lat_r_channel[idx:next_idx] = lat_r
            asym_channel[idx:next_idx] = abs(lat_l - lat_r)

    return lat_l_channel, lat_r_channel, asym_channel


def compute_ttt5_features(data):
    """
    TTT5: 90% settling time (within 10% of amplitude).
    """
    lv = data[:, 2]
    rv = data[:, 3]
    target_v = data[:, 5]

    saccade_indices = detect_vertical_saccades(target_v)
    n_samples = len(lv)

    lat_l_channel = np.zeros(n_samples)
    lat_r_channel = np.zeros(n_samples)
    asym_channel = np.zeros(n_samples)

    for i, idx in enumerate(saccade_indices):
        if idx >= n_samples or idx < 1:
            continue

        # Compute saccade amplitude
        amplitude = abs(target_v[idx] - target_v[idx-1])
        tolerance = 0.10 * amplitude

        window_end = min(idx + 60, n_samples)

        lat_l = np.nan
        for t in range(idx, window_end):
            if abs(lv[t] - target_v[t]) <= tolerance:
                lat_l = (t - idx) / SAMPLE_RATE * 1000
                break

        lat_r = np.nan
        for t in range(idx, window_end):
            if abs(rv[t] - target_v[t]) <= tolerance:
                lat_r = (t - idx) / SAMPLE_RATE * 1000
                break

        next_idx = saccade_indices[i+1] if i+1 < len(saccade_indices) else n_samples
        if not np.isnan(lat_l):
            lat_l_channel[idx:next_idx] = lat_l
        if not np.isnan(lat_r):
            lat_r_channel[idx:next_idx] = lat_r
        if not np.isnan(lat_l) and not np.isnan(lat_r):
            asym_channel[idx:next_idx] = abs(lat_l - lat_r)

    return lat_l_channel, lat_r_channel, asym_channel


def compute_fat1_features(data):
    """
    FAT1: Error degradation (late error - early error).
    """
    lv = data[:, 2]
    rv = data[:, 3]
    target_v = data[:, 5]

    saccade_indices = detect_vertical_saccades(target_v)
    n_samples = len(lv)

    # Compute per-saccade errors
    errors_l = []
    errors_r = []
    saccade_ranges = []

    for i, idx in enumerate(saccade_indices):
        if idx >= n_samples:
            continue

        # Settling window: 200-400ms after saccade
        start = min(idx + 24, n_samples)  # 200ms
        end = min(idx + 48, n_samples)    # 400ms

        if end > start:
            err_l = np.mean(np.abs(lv[start:end] - target_v[start:end]))
            err_r = np.mean(np.abs(rv[start:end] - target_v[start:end]))
            errors_l.append(err_l)
            errors_r.append(err_r)

            next_idx = saccade_indices[i+1] if i+1 < len(saccade_indices) else n_samples
            saccade_ranges.append((idx, next_idx))

    # Compute degradation
    deg_l_channel = np.zeros(n_samples)
    deg_r_channel = np.zeros(n_samples)
    asym_channel = np.zeros(n_samples)

    if len(errors_l) >= 3:
        third = len(errors_l) // 3
        early_l = np.mean(errors_l[:third])
        late_l = np.mean(errors_l[-third:])
        early_r = np.mean(errors_r[:third])
        late_r = np.mean(errors_r[-third:])

        deg_l = late_l - early_l
        deg_r = late_r - early_r
        asym = abs(deg_l - deg_r)

        # Fill channels with degradation values
        for start, end in saccade_ranges:
            deg_l_channel[start:end] = deg_l
            deg_r_channel[start:end] = deg_r
            asym_channel[start:end] = asym

    return deg_l_channel, deg_r_channel, asym_channel


def compute_fat2_features(data):
    """
    FAT2: Latency degradation (late latency - early latency).
    """
    lv = data[:, 2]
    rv = data[:, 3]
    target_v = data[:, 5]

    saccade_indices = detect_vertical_saccades(target_v)
    n_samples = len(lv)

    latencies_l = []
    latencies_r = []
    saccade_ranges = []

    tolerance = 4.0

    for i, idx in enumerate(saccade_indices):
        if idx >= n_samples:
            continue

        window_end = min(idx + 60, n_samples)

        lat_l = np.nan
        for t in range(idx, window_end):
            if abs(lv[t] - target_v[t]) <= tolerance:
                lat_l = (t - idx) / SAMPLE_RATE * 1000
                break

        lat_r = np.nan
        for t in range(idx, window_end):
            if abs(rv[t] - target_v[t]) <= tolerance:
                lat_r = (t - idx) / SAMPLE_RATE * 1000
                break

        if not np.isnan(lat_l) and not np.isnan(lat_r):
            latencies_l.append(lat_l)
            latencies_r.append(lat_r)
            next_idx = saccade_indices[i+1] if i+1 < len(saccade_indices) else n_samples
            saccade_ranges.append((idx, next_idx))

    deg_l_channel = np.zeros(n_samples)
    deg_r_channel = np.zeros(n_samples)
    asym_channel = np.zeros(n_samples)

    if len(latencies_l) >= 3:
        third = len(latencies_l) // 3
        early_l = np.mean(latencies_l[:third])
        late_l = np.mean(latencies_l[-third:])
        early_r = np.mean(latencies_r[:third])
        late_r = np.mean(latencies_r[-third:])

        deg_l = late_l - early_l
        deg_r = late_r - early_r
        asym = abs(deg_l - deg_r)

        for start, end in saccade_ranges:
            deg_l_channel[start:end] = deg_l
            deg_r_channel[start:end] = deg_r
            asym_channel[start:end] = asym

    return deg_l_channel, deg_r_channel, asym_channel


def compute_fat3_features(data):
    """
    FAT3: Error slope (linear regression slope of errors over session).
    """
    lv = data[:, 2]
    rv = data[:, 3]
    target_v = data[:, 5]

    saccade_indices = detect_vertical_saccades(target_v)
    n_samples = len(lv)

    errors_l = []
    errors_r = []
    saccade_ranges = []

    for i, idx in enumerate(saccade_indices):
        if idx >= n_samples:
            continue

        start = min(idx + 24, n_samples)
        end = min(idx + 48, n_samples)

        if end > start:
            err_l = np.mean(np.abs(lv[start:end] - target_v[start:end]))
            err_r = np.mean(np.abs(rv[start:end] - target_v[start:end]))
            errors_l.append(err_l)
            errors_r.append(err_r)

            next_idx = saccade_indices[i+1] if i+1 < len(saccade_indices) else n_samples
            saccade_ranges.append((idx, next_idx))

    slope_l_channel = np.zeros(n_samples)
    slope_r_channel = np.zeros(n_samples)
    asym_channel = np.zeros(n_samples)

    if len(errors_l) >= 3:
        x = np.arange(len(errors_l))
        slope_l, _, _, _, _ = stats.linregress(x, errors_l)
        slope_r, _, _, _, _ = stats.linregress(x, errors_r)
        asym = abs(slope_l - slope_r)

        for start, end in saccade_ranges:
            slope_l_channel[start:end] = slope_l
            slope_r_channel[start:end] = slope_r
            asym_channel[start:end] = asym

    return slope_l_channel, slope_r_channel, asym_channel


def compute_fat4_features(data):
    """
    FAT4: Latency slope (linear regression slope of latencies over session).
    """
    lv = data[:, 2]
    rv = data[:, 3]
    target_v = data[:, 5]

    saccade_indices = detect_vertical_saccades(target_v)
    n_samples = len(lv)

    latencies_l = []
    latencies_r = []
    saccade_ranges = []

    tolerance = 4.0

    for i, idx in enumerate(saccade_indices):
        if idx >= n_samples:
            continue

        window_end = min(idx + 60, n_samples)

        lat_l = np.nan
        for t in range(idx, window_end):
            if abs(lv[t] - target_v[t]) <= tolerance:
                lat_l = (t - idx) / SAMPLE_RATE * 1000
                break

        lat_r = np.nan
        for t in range(idx, window_end):
            if abs(rv[t] - target_v[t]) <= tolerance:
                lat_r = (t - idx) / SAMPLE_RATE * 1000
                break

        if not np.isnan(lat_l) and not np.isnan(lat_r):
            latencies_l.append(lat_l)
            latencies_r.append(lat_r)
            next_idx = saccade_indices[i+1] if i+1 < len(saccade_indices) else n_samples
            saccade_ranges.append((idx, next_idx))

    slope_l_channel = np.zeros(n_samples)
    slope_r_channel = np.zeros(n_samples)
    asym_channel = np.zeros(n_samples)

    if len(latencies_l) >= 3:
        x = np.arange(len(latencies_l))
        slope_l, _, _, _, _ = stats.linregress(x, latencies_l)
        slope_r, _, _, _, _ = stats.linregress(x, latencies_r)
        asym = abs(slope_l - slope_r)

        for start, end in saccade_ranges:
            slope_l_channel[start:end] = slope_l
            slope_r_channel[start:end] = slope_r
            asym_channel[start:end] = asym

    return slope_l_channel, slope_r_channel, asym_channel


def compute_fat5_features(data):
    """
    FAT5: Variability increase (late std - early std).
    """
    lv = data[:, 2]
    rv = data[:, 3]
    target_v = data[:, 5]

    saccade_indices = detect_vertical_saccades(target_v)
    n_samples = len(lv)

    errors_l = []
    errors_r = []
    saccade_ranges = []

    for i, idx in enumerate(saccade_indices):
        if idx >= n_samples:
            continue

        start = min(idx + 24, n_samples)
        end = min(idx + 48, n_samples)

        if end > start:
            err_l = np.mean(np.abs(lv[start:end] - target_v[start:end]))
            err_r = np.mean(np.abs(rv[start:end] - target_v[start:end]))
            errors_l.append(err_l)
            errors_r.append(err_r)

            next_idx = saccade_indices[i+1] if i+1 < len(saccade_indices) else n_samples
            saccade_ranges.append((idx, next_idx))

    var_l_channel = np.zeros(n_samples)
    var_r_channel = np.zeros(n_samples)
    asym_channel = np.zeros(n_samples)

    if len(errors_l) >= 3:
        third = len(errors_l) // 3
        early_std_l = np.std(errors_l[:third])
        late_std_l = np.std(errors_l[-third:])
        early_std_r = np.std(errors_r[:third])
        late_std_r = np.std(errors_r[-third:])

        var_inc_l = late_std_l - early_std_l
        var_inc_r = late_std_r - early_std_r
        asym = abs(var_inc_l - var_inc_r)

        for start, end in saccade_ranges:
            var_l_channel[start:end] = var_inc_l
            var_r_channel[start:end] = var_inc_r
            asym_channel[start:end] = asym

    return var_l_channel, var_r_channel, asym_channel


def compute_fft_features(data, window_size=60, hop_size=30):
    """
    FFT: Frequency domain features from eye position signals.

    Extracts spectral features that may capture:
    - Tremor frequencies (typically 3-8 Hz in MG)
    - Saccade dynamics in frequency domain
    - Eye movement instability patterns

    For each window, computes:
    - Dominant frequency (Hz)
    - Power in low band (0-2 Hz): fixation stability
    - Power in mid band (2-6 Hz): tremor range
    - Power in high band (6-15 Hz): saccade dynamics

    Returns 3 channels: spectral_power_L, spectral_power_R, spectral_asymmetry
    """
    lv = data[:, 2]  # Left vertical position
    rv = data[:, 3]  # Right vertical position
    n_samples = len(lv)

    # Initialize output channels
    spectral_l_channel = np.zeros(n_samples)
    spectral_r_channel = np.zeros(n_samples)
    asym_channel = np.zeros(n_samples)

    # Frequency bands (at 120 Hz sample rate, Nyquist = 60 Hz)
    # Low: 0-2 Hz (fixation), Mid: 2-6 Hz (tremor), High: 6-15 Hz (saccade)
    freq_resolution = SAMPLE_RATE / window_size  # Hz per bin

    low_band_end = int(2.0 / freq_resolution)
    mid_band_start = int(2.0 / freq_resolution)
    mid_band_end = int(6.0 / freq_resolution)
    high_band_start = int(6.0 / freq_resolution)
    high_band_end = int(15.0 / freq_resolution)

    # Sliding window FFT
    for start in range(0, n_samples - window_size, hop_size):
        end = start + window_size

        # Extract windows
        window_l = lv[start:end]
        window_r = rv[start:end]

        # Apply Hanning window to reduce spectral leakage
        hanning = np.hanning(window_size)
        window_l = window_l * hanning
        window_r = window_r * hanning

        # Compute FFT
        fft_l = np.abs(np.fft.rfft(window_l))
        fft_r = np.abs(np.fft.rfft(window_r))

        # Compute power in mid band (tremor range - most relevant for MG)
        # This is where neuromuscular fatigue/tremor would manifest
        mid_power_l = np.sum(fft_l[mid_band_start:mid_band_end+1]**2)
        mid_power_r = np.sum(fft_r[mid_band_start:mid_band_end+1]**2)

        # Total power for normalization
        total_power_l = np.sum(fft_l**2) + 1e-10
        total_power_r = np.sum(fft_r**2) + 1e-10

        # Relative mid-band power (tremor proportion)
        rel_mid_l = mid_power_l / total_power_l
        rel_mid_r = mid_power_r / total_power_r

        # Asymmetry in spectral characteristics
        asym = np.abs(rel_mid_l - rel_mid_r)

        # Fill channels for this window
        spectral_l_channel[start:end] = rel_mid_l
        spectral_r_channel[start:end] = rel_mid_r
        asym_channel[start:end] = asym

    return spectral_l_channel, spectral_r_channel, asym_channel


def compute_h38b_features(data):
    """
    H38b: Composite metric (MAD, Degradation, Latency weighted combination).
    """
    lv = data[:, 2]
    rv = data[:, 3]
    target_v = data[:, 5]

    saccade_indices = detect_vertical_saccades(target_v)
    n_samples = len(lv)

    # Compute per-saccade errors and latencies
    errors_l = []
    errors_r = []
    latencies_l = []
    latencies_r = []
    saccade_ranges = []

    tolerance = 3.0

    for i, idx in enumerate(saccade_indices):
        if idx >= n_samples:
            continue

        # Error
        start = min(idx + 24, n_samples)
        end = min(idx + 48, n_samples)

        if end > start:
            err_l = np.mean(np.abs(lv[start:end] - target_v[start:end]))
            err_r = np.mean(np.abs(rv[start:end] - target_v[start:end]))
            errors_l.append(err_l)
            errors_r.append(err_r)

        # Latency
        window_end = min(idx + 60, n_samples)
        lat_l = np.nan
        for t in range(idx, window_end):
            if abs(lv[t] - target_v[t]) <= tolerance:
                lat_l = (t - idx) / SAMPLE_RATE * 1000
                break
        lat_r = np.nan
        for t in range(idx, window_end):
            if abs(rv[t] - target_v[t]) <= tolerance:
                lat_r = (t - idx) / SAMPLE_RATE * 1000
                break

        if not np.isnan(lat_l):
            latencies_l.append(lat_l)
        if not np.isnan(lat_r):
            latencies_r.append(lat_r)

        next_idx = saccade_indices[i+1] if i+1 < len(saccade_indices) else n_samples
        saccade_ranges.append((idx, next_idx))

    score_l_channel = np.zeros(n_samples)
    score_r_channel = np.zeros(n_samples)
    asym_channel = np.zeros(n_samples)

    if len(errors_l) >= 3 and len(latencies_l) > 0:
        # MAD
        mad_l = np.median(np.abs(np.array(errors_l) - np.median(errors_l)))
        mad_r = np.median(np.abs(np.array(errors_r) - np.median(errors_r)))

        # Degradation
        third = len(errors_l) // 3
        early_l = np.mean(errors_l[:max(1, int(len(errors_l)*0.2))])
        late_l = np.mean(errors_l[-max(1, int(len(errors_l)*0.2)):])
        early_r = np.mean(errors_r[:max(1, int(len(errors_r)*0.2))])
        late_r = np.mean(errors_r[-max(1, int(len(errors_r)*0.2)):])
        deg_l = late_l - early_l
        deg_r = late_r - early_r

        # Latency (fastest 25%)
        n_fast = max(1, len(latencies_l) // 4)
        lat_l = np.mean(sorted(latencies_l)[:n_fast])
        lat_r = np.mean(sorted(latencies_r)[:n_fast]) if latencies_r else lat_l

        # Composite scores
        score_l = 0.5 * (0.30 * mad_l + 0.70 * deg_l) + 0.5 * (lat_l / 100)
        score_r = 0.5 * (0.30 * mad_r + 0.70 * deg_r) + 0.5 * (lat_r / 100)

        # Asymmetry components
        cv_asym = abs(mad_l - mad_r) / ((mad_l + mad_r) / 2 + 1e-6)
        deg_asym = abs(deg_l - deg_r)
        lat_asym = abs(lat_l - lat_r)

        final_asym = 0.5 * (0.30 * cv_asym + 0.70 * deg_asym) + 0.5 * (lat_asym / 100)

        for start, end in saccade_ranges:
            score_l_channel[start:end] = score_l
            score_r_channel[start:end] = score_r
            asym_channel[start:end] = final_asym

    return score_l_channel, score_r_channel, asym_channel


# Feature addition functions
def add_ttt1_channels(baseline_items):
    """Add TTT1 channels to baseline."""
    enhanced_items = []
    for item in baseline_items:
        data = item['data']
        lat_l, lat_r, asym = compute_ttt1_features(data[:, :6])  # Use raw columns
        enhanced_data = np.column_stack([data, lat_l, lat_r, asym])
        new_item = item.copy()
        new_item['data'] = enhanced_data
        enhanced_items.append(new_item)
    return enhanced_items


def add_ttt2_channels(baseline_items):
    """Add TTT2 channels to baseline."""
    enhanced_items = []
    for item in baseline_items:
        data = item['data']
        lat_l, lat_r, asym = compute_ttt2_features(data[:, :6])
        enhanced_data = np.column_stack([data, lat_l, lat_r, asym])
        new_item = item.copy()
        new_item['data'] = enhanced_data
        enhanced_items.append(new_item)
    return enhanced_items


def add_ttt3_channels(baseline_items):
    """Add TTT3 channels to baseline."""
    enhanced_items = []
    for item in baseline_items:
        data = item['data']
        lat_l, lat_r, asym = compute_ttt3_features(data[:, :6])
        enhanced_data = np.column_stack([data, lat_l, lat_r, asym])
        new_item = item.copy()
        new_item['data'] = enhanced_data
        enhanced_items.append(new_item)
    return enhanced_items


def add_ttt4_channels(baseline_items):
    """Add TTT4 channels to baseline."""
    enhanced_items = []
    for item in baseline_items:
        data = item['data']
        lat_l, lat_r, asym = compute_ttt4_features(data[:, :6])
        enhanced_data = np.column_stack([data, lat_l, lat_r, asym])
        new_item = item.copy()
        new_item['data'] = enhanced_data
        enhanced_items.append(new_item)
    return enhanced_items


def add_ttt5_channels(baseline_items):
    """Add TTT5 channels to baseline."""
    enhanced_items = []
    for item in baseline_items:
        data = item['data']
        lat_l, lat_r, asym = compute_ttt5_features(data[:, :6])
        enhanced_data = np.column_stack([data, lat_l, lat_r, asym])
        new_item = item.copy()
        new_item['data'] = enhanced_data
        enhanced_items.append(new_item)
    return enhanced_items


def add_fat1_channels(baseline_items):
    """Add FAT1 channels to baseline."""
    enhanced_items = []
    for item in baseline_items:
        data = item['data']
        deg_l, deg_r, asym = compute_fat1_features(data[:, :6])
        enhanced_data = np.column_stack([data, deg_l, deg_r, asym])
        new_item = item.copy()
        new_item['data'] = enhanced_data
        enhanced_items.append(new_item)
    return enhanced_items


def add_fat2_channels(baseline_items):
    """Add FAT2 channels to baseline."""
    enhanced_items = []
    for item in baseline_items:
        data = item['data']
        deg_l, deg_r, asym = compute_fat2_features(data[:, :6])
        enhanced_data = np.column_stack([data, deg_l, deg_r, asym])
        new_item = item.copy()
        new_item['data'] = enhanced_data
        enhanced_items.append(new_item)
    return enhanced_items


def add_fat3_channels(baseline_items):
    """Add FAT3 channels to baseline."""
    enhanced_items = []
    for item in baseline_items:
        data = item['data']
        slope_l, slope_r, asym = compute_fat3_features(data[:, :6])
        enhanced_data = np.column_stack([data, slope_l, slope_r, asym])
        new_item = item.copy()
        new_item['data'] = enhanced_data
        enhanced_items.append(new_item)
    return enhanced_items


def add_fat4_channels(baseline_items):
    """Add FAT4 channels to baseline."""
    enhanced_items = []
    for item in baseline_items:
        data = item['data']
        slope_l, slope_r, asym = compute_fat4_features(data[:, :6])
        enhanced_data = np.column_stack([data, slope_l, slope_r, asym])
        new_item = item.copy()
        new_item['data'] = enhanced_data
        enhanced_items.append(new_item)
    return enhanced_items


def add_fat5_channels(baseline_items):
    """Add FAT5 channels to baseline."""
    enhanced_items = []
    for item in baseline_items:
        data = item['data']
        var_l, var_r, asym = compute_fat5_features(data[:, :6])
        enhanced_data = np.column_stack([data, var_l, var_r, asym])
        new_item = item.copy()
        new_item['data'] = enhanced_data
        enhanced_items.append(new_item)
    return enhanced_items


def add_h38b_channels(baseline_items):
    """Add H38b channels to baseline."""
    enhanced_items = []
    for item in baseline_items:
        data = item['data']
        score_l, score_r, asym = compute_h38b_features(data[:, :6])
        enhanced_data = np.column_stack([data, score_l, score_r, asym])
        new_item = item.copy()
        new_item['data'] = enhanced_data
        enhanced_items.append(new_item)
    return enhanced_items


def add_fft_channels(baseline_items):
    """Add FFT frequency domain channels to baseline."""
    enhanced_items = []
    for item in baseline_items:
        data = item['data']
        spectral_l, spectral_r, asym = compute_fft_features(data[:, :6])
        enhanced_data = np.column_stack([data, spectral_l, spectral_r, asym])
        new_item = item.copy()
        new_item['data'] = enhanced_data
        enhanced_items.append(new_item)
    return enhanced_items


def train_model_cv(items, num_channels, trial_seed):
    """Train model with cross-validation."""
    set_all_seeds(trial_seed)

    # Subsample
    subsampled_items = subsample_data(items, SUBSAMPLE_FACTOR)

    # Calculate target sequence length
    seq_lens = [item['data'].shape[0] for item in subsampled_items]
    target_seq_len = int(np.percentile(seq_lens, TARGET_SEQ_LEN_PERCENTILE))

    # Prepare labels and patient IDs
    labels = np.array([item['label'] for item in subsampled_items])
    patient_ids = np.array([item['patient_id'] for item in subsampled_items])

    # Cross-validation
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=trial_seed)
    fold_accs = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(patient_ids, labels)):
        train_items = [subsampled_items[i] for i in train_idx]
        val_items = [subsampled_items[i] for i in val_idx]

        # Prepare sequences
        X_train = [item['data'] for item in train_items]
        y_train = [item['label'] for item in train_items]
        X_val = [item['data'] for item in val_items]
        y_val = [item['label'] for item in val_items]

        # Standardize
        scaler = SaccadeStandardScaler()
        scaler.fit(train_items)
        X_train_scaled = [scaler.transform(seq) for seq in X_train]
        X_val_scaled = [scaler.transform(seq) for seq in X_val]

        # Pad/truncate
        X_train_padded = np.array([
            seq[:target_seq_len] if len(seq) >= target_seq_len
            else np.vstack([seq, np.zeros((target_seq_len - len(seq), num_channels))])
            for seq in X_train_scaled
        ])
        X_val_padded = np.array([
            seq[:target_seq_len] if len(seq) >= target_seq_len
            else np.vstack([seq, np.zeros((target_seq_len - len(seq), num_channels))])
            for seq in X_val_scaled
        ])

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_padded)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val_padded)
        y_val_tensor = torch.LongTensor(y_val)

        # Datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Model
        model = SaccadeRNN_Small(
            input_dim=num_channels,
            hidden_dim=128,
            output_dim=2,
            n_layers=2,
            seq_len=target_seq_len,
            dropout_prob=0.3
        ).to(DEVICE)

        # Class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        early_stopper = EarlyStopper(patience=EARLY_STOPPING_PATIENCE, min_delta=0.001)

        # Training loop
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc, _, _ = evaluate_epoch(model, val_loader, criterion, DEVICE)

            if early_stopper.early_stop(val_loss):
                break

        # Final validation accuracy
        _, final_val_acc, _, _ = evaluate_epoch(model, val_loader, criterion, DEVICE)
        fold_accs.append(final_val_acc)

    return np.mean(fold_accs), np.std(fold_accs)


def main():
    """Main experiment runner."""
    print("="*80)
    print("Starting Experiment 16: Eye Difference Feature Discriminatory Power Study")
    print("Testing 11 metrics from Experiment 15 as neural network features")
    print("="*80)

    # Create results directory
    create_results_directory(RESULTS_DIR)

    # Load data ONCE
    print("\n" + "="*50)
    print("Loading Data (ONCE for all experiments)...")
    print("="*50)

    with open(os.devnull, 'w') as f_null:
        raw_items = load_raw_sequences_and_labels(
            BASE_DIR, BINARY_CLASS_DEFINITIONS, FEATURE_COLUMNS,
            CSV_ENCODING, CSV_SEPARATOR, MIN_SEQ_LEN_THRESHOLD, f_null
        )

    binary_items = prepare_binary_data(raw_items)
    baseline_items = add_baseline_channels(binary_items)

    print(f"Data loaded: {len(binary_items)} sequences")
    print(f"HC: {sum(1 for x in binary_items if x['label']==0)}, MG: {sum(1 for x in binary_items if x['label']==1)}")

    # Prepare enhanced datasets
    print("\n" + "="*50)
    print("Preparing Eye Difference Feature Sets...")
    print("="*50)

    feature_configs = [
        ("TTT1 (3deg latency)", add_ttt1_channels, 17),
        ("TTT2 (sustained)", add_ttt2_channels, 17),
        ("TTT3 (4deg latency)", add_ttt3_channels, 17),
        ("TTT4 (peak velocity)", add_ttt4_channels, 17),
        ("TTT5 (90% settling)", add_ttt5_channels, 17),
        ("FAT1 (error deg)", add_fat1_channels, 17),
        ("FAT2 (latency deg)", add_fat2_channels, 17),
        ("FAT3 (error slope)", add_fat3_channels, 17),
        ("FAT4 (latency slope)", add_fat4_channels, 17),
        ("FAT5 (var increase)", add_fat5_channels, 17),
        ("H38b (composite)", add_h38b_channels, 17),
        ("FFT (tremor freq)", add_fft_channels, 17),
    ]

    enhanced_datasets = {}
    for name, add_func, num_channels in feature_configs:
        print(f"  Preparing {name}...", end=" ")
        enhanced_datasets[name] = (add_func(baseline_items), num_channels)
        print(f"({num_channels} channels)")

    # Train baseline ONCE with multiple trials
    print("\n" + "="*50)
    print(f"Training SHARED Baseline ({NUM_TRIALS} trials, {NUM_FOLDS}-fold CV)...")
    print("="*50)

    baseline_results = []
    for trial in range(NUM_TRIALS):
        trial_seed = RANDOM_STATE + trial
        print(f"  Baseline Trial {trial+1}/{NUM_TRIALS} (seed={trial_seed})...", end=" ", flush=True)
        mean_acc, std_acc = train_model_cv(baseline_items, num_channels=14, trial_seed=trial_seed)
        baseline_results.append(mean_acc)
        print(f"{mean_acc:.4f} +/- {std_acc:.4f}")

    baseline_mean = np.mean(baseline_results)
    baseline_std = np.std(baseline_results)
    print(f"\n  Baseline Overall: {baseline_mean:.4f} +/- {baseline_std:.4f}")

    # Train enhanced models
    results = {}

    for feature_name, (feature_items, num_channels) in enhanced_datasets.items():
        print("\n" + "="*50)
        print(f"Training {feature_name} ({NUM_TRIALS} trials, {NUM_FOLDS}-fold CV)...")
        print("="*50)

        enhanced_results = []
        for trial in range(NUM_TRIALS):
            trial_seed = RANDOM_STATE + trial
            print(f"  {feature_name} Trial {trial+1}/{NUM_TRIALS} (seed={trial_seed})...", end=" ", flush=True)
            mean_acc, std_acc = train_model_cv(feature_items, num_channels=num_channels, trial_seed=trial_seed)
            enhanced_results.append(mean_acc)
            print(f"{mean_acc:.4f} +/- {std_acc:.4f}")

        enhanced_mean = np.mean(enhanced_results)
        enhanced_std = np.std(enhanced_results)
        delta = enhanced_mean - baseline_mean
        delta_pct = delta * 100

        print(f"\n  {feature_name} Overall: {enhanced_mean:.4f} +/- {enhanced_std:.4f}")
        print(f"  Delta: {delta:+.4f} ({delta_pct:+.2f}%)")

        results[feature_name] = {
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'enhanced_mean': enhanced_mean,
            'enhanced_std': enhanced_std,
            'delta': delta,
            'delta_pct': delta_pct
        }

    # Write summary
    summary_path = os.path.join(RESULTS_DIR, 'exp16_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXPERIMENT 16: EYE DIFFERENCE FEATURE STUDY SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Configuration:\n")
        f.write(f"  - Epochs: {EPOCHS}\n")
        f.write(f"  - Folds: {NUM_FOLDS}\n")
        f.write(f"  - Trials: {NUM_TRIALS}\n")
        f.write(f"  - Random State: {RANDOM_STATE}\n")
        f.write(f"  - Early Stopping Patience: {EARLY_STOPPING_PATIENCE}\n")
        f.write("\n")
        f.write(f"Shared Baseline (14 channels):\n")
        f.write(f"  Accuracy: {baseline_mean:.4f} +/- {baseline_std:.4f}\n")
        f.write("\n")

        for feature_name in [name for name, _, _ in feature_configs]:
            res = results[feature_name]
            f.write(f"{feature_name}:\n")
            f.write(f"  Enhanced Accuracy: {res['enhanced_mean']:.4f} +/- {res['enhanced_std']:.4f}\n")
            f.write(f"  Delta: {res['delta']:+.4f} ({res['delta_pct']:+.2f}%)\n")
            f.write("\n")

        # Ranking
        f.write("="*80 + "\n")
        f.write("RANKING BY DELTA:\n")
        f.write("="*80 + "\n")
        sorted_features = sorted(results.items(), key=lambda x: x[1]['delta'], reverse=True)
        for rank, (name, res) in enumerate(sorted_features, 1):
            symbol = "+" if res['delta'] > 0 else "-" if res['delta'] < 0 else "="
            f.write(f"  {rank:2d}. {name:25s}: {res['delta_pct']:+.2f}% ({res['enhanced_mean']:.4f})\n")
        f.write("="*80 + "\n")

    print("\n" + "="*80)
    print("Experiment 16 completed successfully!")
    print(f"Results saved to: {summary_path}")
    print("\nFinal Ranking:")
    for rank, (name, res) in enumerate(sorted_features, 1):
        color = "\033[92m" if res['delta'] > 0 else "\033[91m" if res['delta'] < 0 else "\033[0m"
        print(f"  {rank:2d}. {name:25s}: {res['delta_pct']:+.2f}%")
    print("="*80)


if __name__ == "__main__":
    main()
