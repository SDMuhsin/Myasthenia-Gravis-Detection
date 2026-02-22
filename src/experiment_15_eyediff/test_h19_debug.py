#!/usr/bin/env python3
"""Debug H19 latency calculation"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_timeseries_data, merge_mg_classes

BASE_DIR = './data'
CLASS_DEFINITIONS = {
    'HC': {'path': 'Healthy control', 'label': 0},
    'MG': {'path': 'Definite MG', 'label': 1},
    'Probable_MG': {'path': 'Probable MG', 'label': 1},
}
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']

print("Loading data...")
raw_sequences = load_timeseries_data(BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, 'utf-16-le', ',', 10)
sequences = merge_mg_classes(raw_sequences)

def compute_saccade_latencies(eye_pos, target_pos, sample_rate_hz=120):
    valid = ~(np.isnan(eye_pos) | np.isnan(target_pos))
    eye = eye_pos[valid]
    target = target_pos[valid]

    if len(target) < 20:
        return []

    # Detect target jumps
    target_diff = np.abs(np.diff(target))
    jump_threshold = 5.0
    jump_indices = np.where(target_diff > jump_threshold)[0] + 1

    latencies_ms = []

    for jump_idx in jump_indices:
        if jump_idx >= len(target) - 10:
            continue

        new_target = target[jump_idx]
        threshold_deg = 3.0

        for offset in range(1, min(100, len(eye) - jump_idx)):
            eye_pos_now = eye[jump_idx + offset]
            error = abs(eye_pos_now - new_target)

            if error < threshold_deg:
                latency_ms = (offset / sample_rate_hz) * 1000
                latencies_ms.append(latency_ms)
                break

    return latencies_ms

# Test on first few sequences of each class
hc_count = 0
mg_count = 0

for seq in sequences:
    if seq['class_name'] == 'HC' and hc_count < 3:
        hc_count += 1
        df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)
        latencies = compute_saccade_latencies(df['LV'].values, df['TargetV'].values)
        print(f"HC sample {hc_count}: {len(latencies)} saccades detected, latencies: {latencies[:5]}")

    if seq['class_name'] == 'MG' and mg_count < 3:
        mg_count += 1
        df = pd.DataFrame(seq['data'], columns=FEATURE_COLUMNS)
        latencies = compute_saccade_latencies(df['LV'].values, df['TargetV'].values)
        print(f"MG sample {mg_count}: {len(latencies)} saccades detected, latencies: {latencies[:5]}")

    if hc_count >= 3 and mg_count >= 3:
        break
