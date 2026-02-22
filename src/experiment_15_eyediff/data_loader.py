"""
Data loader for Experiment 15: Eye Difference Analysis
Loads time-series saccade data without aggregation.
"""

import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm


def load_timeseries_data(base_dir, class_definitions, feature_columns,
                         encoding='utf-16-le', separator=',', min_seq_len=50):
    """
    Load saccade data as time series (no aggregation).

    Args:
        base_dir: Base directory containing class folders
        class_definitions: Dict with class names and their labels
        feature_columns: List of column names to extract
        encoding: CSV file encoding
        separator: CSV separator
        min_seq_len: Minimum sequence length to include

    Returns:
        List of dicts with keys: 'data', 'label', 'patient_id', 'filename', 'class_name'
    """
    print("="*80)
    print("Loading Time-Series Saccade Data")
    print("="*80)

    raw_sequences = []

    for class_name, class_info in class_definitions.items():
        label = class_info['label']
        class_dir = os.path.join(base_dir, class_info['path'])

        print(f"\nProcessing class: {class_name} (Label: {label})")
        print(f"Directory: {class_dir}")

        if not os.path.isdir(class_dir):
            print(f"WARNING: Directory not found: {class_dir}")
            continue

        # Get all patient directories
        patient_dirs = [d for d in os.listdir(class_dir)
                       if os.path.isdir(os.path.join(class_dir, d))]

        print(f"Found {len(patient_dirs)} patient directories")

        for patient_folder in tqdm(patient_dirs, desc=f"  Loading {class_name}"):
            patient_path = os.path.join(class_dir, patient_folder)
            csv_files = glob.glob(os.path.join(patient_path, '*.csv'))

            for csv_file in csv_files:
                try:
                    # Read CSV
                    df = pd.read_csv(csv_file, encoding=encoding, sep=separator)
                    df.columns = [col.strip() for col in df.columns]

                    # Check if all required columns exist
                    if not all(col in df.columns for col in feature_columns):
                        continue

                    # Check minimum length
                    if len(df) < min_seq_len:
                        continue

                    # Extract features
                    df_features = df[feature_columns].copy()

                    # Convert to numeric
                    for col in df_features.columns:
                        df_features[col] = pd.to_numeric(df_features[col], errors='coerce')

                    # Skip if too many NaNs
                    if df_features.isnull().sum().sum() > 0.1 * df_features.size:
                        continue

                    # Fill remaining NaNs with 0
                    df_features = df_features.fillna(0)

                    # Store as dict
                    raw_sequences.append({
                        'data': df_features.values.astype(np.float32),
                        'label': label,
                        'patient_id': patient_folder,
                        'filename': os.path.basename(csv_file),
                        'class_name': class_name
                    })

                except Exception as e:
                    # Silently skip problematic files
                    pass

    print(f"\n{'='*80}")
    print(f"Data loading complete!")
    print(f"Total sequences loaded: {len(raw_sequences)}")

    # Print class distribution
    for class_name in class_definitions.keys():
        count = sum(1 for seq in raw_sequences if seq['class_name'] == class_name)
        print(f"  {class_name}: {count} sequences")

    print("="*80 + "\n")

    return raw_sequences


def merge_mg_classes(raw_sequences):
    """
    Merge 'MG' and 'Probable_MG' into single 'MG' class.

    Args:
        raw_sequences: List of sequence dicts

    Returns:
        List of sequence dicts with merged MG classes
    """
    merged = []

    for seq in raw_sequences:
        new_seq = seq.copy()
        if seq['class_name'] in ['MG', 'Probable_MG']:
            new_seq['class_name'] = 'MG'
            new_seq['label'] = 1
        merged.append(new_seq)

    print("Merged MG classes:")
    hc_count = sum(1 for s in merged if s['class_name'] == 'HC')
    mg_count = sum(1 for s in merged if s['class_name'] == 'MG')
    print(f"  HC: {hc_count}")
    print(f"  MG: {mg_count}")
    print(f"  Total: {len(merged)}\n")

    return merged
