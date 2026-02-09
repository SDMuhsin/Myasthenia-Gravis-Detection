#!/usr/bin/env python3
"""
Generate minimal time series figures for the MHTPN architecture diagram.

Creates:
- 1 full signal PNG (2 channels: Target and Eye position)
- 8 segment PNGs (same signal split into 8 parts)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Output directory
OUTPUT_DIR = './paper/ccece/figures/architecture'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style settings for minimal plots
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5

FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']


def load_horizontal_saccade_sample():
    """Load a Horizontal Saccade sample where TargetH varies."""
    # Find Horizontal Saccade files
    patterns = [
        './data/Healthy control/**/*Horizontal*.csv',
        './data/Definite MG/**/*Horizontal*.csv',
        './data/Probable MG/**/*Horizontal*.csv',
    ]

    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern, recursive=True))

    print(f"Found {len(all_files)} Horizontal Saccade files")

    for csv_path in all_files:
        try:
            df = pd.read_csv(csv_path, encoding='utf-16-le')
            df.columns = [c.strip() for c in df.columns]

            if not all(col in df.columns for col in FEATURE_COLUMNS):
                continue

            # Extract features
            data = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors='coerce').fillna(0).values

            # Check that TargetH actually varies (not constant)
            target_h = data[:, 4]
            if target_h.std() > 1.0:  # Has variation
                print(f"Selected: {os.path.basename(csv_path)}")
                print(f"  Shape: {data.shape}")
                print(f"  TargetH range: [{target_h.min():.1f}, {target_h.max():.1f}]")
                return data

        except Exception as e:
            continue

    print("Warning: No suitable Horizontal Saccade file found")
    return None


def create_minimal_plot(data, start_idx, end_idx, output_path, figsize=(1.5, 0.8)):
    """
    Create a minimal 2-channel time series plot.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    # Extract segment
    segment = data[start_idx:end_idx]
    n_points = len(segment)

    # Time axis (normalized to segment)
    t = np.linspace(0, 1, n_points)

    # Channel indices: TargetH=4, LH=0
    target_h = segment[:, 4]  # Target horizontal
    eye_h = segment[:, 0]     # Left eye horizontal

    # Normalize to same scale for visualization
    all_vals = np.concatenate([target_h, eye_h])
    vmin, vmax = all_vals.min(), all_vals.max()
    margin = (vmax - vmin) * 0.1

    # Plot with distinct colors
    ax.plot(t, target_h, color='#2E86AB', linewidth=0.8, label='Target')
    ax.plot(t, eye_h, color='#E94F37', linewidth=0.8, alpha=0.8, label='Eye')

    # Minimal styling
    ax.set_xlim(0, 1)
    ax.set_ylim(vmin - margin, vmax + margin)

    # Remove all decorations
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout(pad=0.1)
    plt.savefig(output_path, bbox_inches='tight', facecolor='white',
                edgecolor='none', transparent=False)
    plt.close()
    print(f"Saved: {output_path}")


def create_full_signal_plot(data, output_path, target_len=2903):
    """Create the full signal plot with both channels."""
    # Pad or truncate to target length
    if len(data) > target_len:
        data = data[:target_len]
    elif len(data) < target_len:
        padding = np.zeros((target_len - len(data), data.shape[1]))
        data = np.vstack([data, padding])

    fig, ax = plt.subplots(figsize=(6, 1.2), dpi=150)

    n_points = len(data)
    t = np.linspace(0, 1, n_points)

    # Extract channels
    target_h = data[:, 4]
    eye_h = data[:, 0]

    # Plot (no normalization - show actual values)
    ax.plot(t, target_h, color='#2E86AB', linewidth=0.6, label='Target')
    ax.plot(t, eye_h, color='#E94F37', linewidth=0.6, alpha=0.8, label='Eye')

    # Add segment boundaries as subtle vertical lines
    n_segments = 8
    for i in range(1, n_segments):
        ax.axvline(x=i/n_segments, color='gray', linewidth=0.3, linestyle='--', alpha=0.5)

    # Set y limits based on data
    all_vals = np.concatenate([target_h, eye_h])
    vmin, vmax = all_vals.min(), all_vals.max()
    margin = (vmax - vmin) * 0.1
    ax.set_ylim(vmin - margin, vmax + margin)

    # Minimal styling
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    # Simple legend
    ax.legend(loc='upper right', fontsize=6, frameon=False)

    plt.tight_layout(pad=0.1)
    plt.savefig(output_path, bbox_inches='tight', facecolor='white',
                edgecolor='none', transparent=False)
    plt.close()
    print(f"Saved: {output_path}")

    return data


def main():
    print("Loading Horizontal Saccade sample data...")
    data = load_horizontal_saccade_sample()

    if data is None:
        print("ERROR: Could not load data. Exiting.")
        return

    target_len = 2903

    # Pad/truncate data
    if len(data) > target_len:
        data = data[:target_len]
    elif len(data) < target_len:
        padding = np.zeros((target_len - len(data), data.shape[1]))
        data = np.vstack([data, padding])

    # 1. Create full signal plot
    print("\nCreating full signal plot...")
    full_path = os.path.join(OUTPUT_DIR, 'signal_full.png')
    data = create_full_signal_plot(data, full_path, target_len)

    # 2. Create 8 segment plots
    print("\nCreating segment plots...")
    n_segments = 8
    segment_size = target_len // n_segments

    for seg_idx in range(n_segments):
        start_idx = seg_idx * segment_size
        end_idx = (seg_idx + 1) * segment_size if seg_idx < n_segments - 1 else target_len

        seg_path = os.path.join(OUTPUT_DIR, f'signal_segment_{seg_idx + 1}.png')
        create_minimal_plot(data, start_idx, end_idx, seg_path, figsize=(1.5, 0.8))

    print(f"\nDone! All figures saved to {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - signal_full.png (full 2-channel signal with segment boundaries)")
    print("  - signal_segment_1.png through signal_segment_8.png (8 segments)")


if __name__ == "__main__":
    main()
