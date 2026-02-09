"""
Preprocess GazeBase dataset for MHTPN external validation.

Task: Session 1 vs Session 2 classification (within-visit fatigue detection)

The hypothesis is that Session 2 (performed after Session 1 in the same visit)
may show subtle fatigue effects in eye movement patterns, similar to how MG patients
show fatigability compared to healthy controls.

This tests MHTPN's ability to capture temporal dynamics where later portions
of recordings may differ from earlier portions.
"""

import os
import sys
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


def extract_gazebase(zip_path: str, output_dir: str, task: str = 'TEX') -> Dict:
    """
    Extract GazeBase data from nested zip structure.

    Args:
        zip_path: Path to GazeBase_v2_0.zip
        output_dir: Where to extract data
        task: Task type to extract (TEX=reading, HSS=horizontal saccade, RAN=random saccade)

    Returns:
        Dict with metadata about extracted files
    """
    task_names = {
        'TEX': 'Reading',
        'HSS': 'Horizontal_Saccades',
        'RAN': 'Random_Saccades',
        'FXS': 'Fixations',
        'VD1': 'Video_1',
        'VD2': 'Video_2',
        'BLG': 'Balura_Game'
    }

    if task not in task_names:
        raise ValueError(f"Unknown task: {task}. Available: {list(task_names.keys())}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_files = []

    print(f"Extracting GazeBase {task} data...")

    with zipfile.ZipFile(zip_path, 'r') as outer_zip:
        # List all subject zips
        subject_zips = [f for f in outer_zip.namelist() if f.endswith('.zip') and 'Subject_' in f]

        for subject_zip_path in tqdm(subject_zips, desc="Extracting subjects"):
            # Parse round and subject from path like "Round_1/Subject_1001.zip"
            parts = subject_zip_path.split('/')
            round_num = int(parts[0].replace('Round_', ''))
            subject_id = int(parts[1].replace('Subject_', '').replace('.zip', ''))

            # Extract subject zip to memory
            with outer_zip.open(subject_zip_path) as subject_zip_file:
                subject_zip_bytes = subject_zip_file.read()

            # Open subject zip from bytes
            import io
            with zipfile.ZipFile(io.BytesIO(subject_zip_bytes), 'r') as subject_zip:
                # Look for the specific task files
                for session in [1, 2]:
                    # Filename pattern: S{session}/S{session}_{TaskFolder}/S_{subject}_S{session}_{task}.csv
                    # Note: paths inside subject zip don't include Round_X/
                    csv_name = f"S{session}/S{session}_{task_names[task]}/S_{subject_id}_S{session}_{task}.csv"

                    if csv_name in subject_zip.namelist():
                        # Extract to output directory
                        out_file = output_dir / f"R{round_num}_S{subject_id}_S{session}_{task}.csv"

                        with subject_zip.open(csv_name) as csv_file:
                            with open(out_file, 'wb') as f:
                                f.write(csv_file.read())

                        extracted_files.append({
                            'file': str(out_file),
                            'round': round_num,
                            'subject_id': subject_id,
                            'session': session,
                            'task': task
                        })

    print(f"Extracted {len(extracted_files)} files")
    return {'files': extracted_files}


def compute_velocity(x: np.ndarray, y: np.ndarray, dt_ms: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute velocity from position using central differences."""
    vx = np.zeros_like(x)
    vy = np.zeros_like(y)

    # Central differences for interior points
    vx[1:-1] = (x[2:] - x[:-2]) / (2 * dt_ms)
    vy[1:-1] = (y[2:] - y[:-2]) / (2 * dt_ms)

    # Forward/backward differences for endpoints
    vx[0] = (x[1] - x[0]) / dt_ms
    vx[-1] = (x[-1] - x[-2]) / dt_ms
    vy[0] = (y[1] - y[0]) / dt_ms
    vy[-1] = (y[-1] - y[-2]) / dt_ms

    return vx, vy


def load_and_preprocess_file(file_path: str, target_length: int = 30000) -> Optional[np.ndarray]:
    """
    Load a GazeBase CSV and preprocess to fixed-length feature array.

    Args:
        file_path: Path to CSV file
        target_length: Desired sequence length (samples to use)

    Returns:
        Array of shape (target_length, n_features) or None if file invalid
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # Required columns
    required = ['n', 'x', 'y', 'val']
    if not all(col in df.columns for col in required):
        print(f"Missing columns in {file_path}")
        return None

    # Remove invalid samples (val > 0 means blink/missing)
    valid_mask = df['val'] == 0

    # Need at least 50% valid data
    if valid_mask.sum() < target_length * 0.5:
        print(f"Too few valid samples in {file_path}: {valid_mask.sum()}")
        return None

    # Get position data
    x = df['x'].values.astype(np.float32)
    y = df['y'].values.astype(np.float32)

    # Interpolate over invalid samples
    invalid_idx = np.where(~valid_mask)[0]
    valid_idx = np.where(valid_mask)[0]

    if len(invalid_idx) > 0 and len(valid_idx) > 0:
        x[invalid_idx] = np.interp(invalid_idx, valid_idx, x[valid_idx])
        y[invalid_idx] = np.interp(invalid_idx, valid_idx, y[valid_idx])

    # Handle NaN values that might remain
    x = np.nan_to_num(x, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)

    # Truncate or pad to target length
    n_samples = len(x)
    if n_samples >= target_length:
        # Take first target_length samples (start of recording, same as MG)
        x = x[:target_length]
        y = y[:target_length]
    else:
        # Pad with last value
        pad_len = target_length - n_samples
        x = np.pad(x, (0, pad_len), mode='edge')
        y = np.pad(y, (0, pad_len), mode='edge')

    # Compute velocities
    vx, vy = compute_velocity(x, y, dt_ms=1.0)

    # Compute speed
    speed = np.sqrt(vx**2 + vy**2)

    # Compute acceleration
    ax, ay = compute_velocity(vx, vy, dt_ms=1.0)
    acc_mag = np.sqrt(ax**2 + ay**2)

    # Get pupil diameter if available
    if 'dP' in df.columns:
        dp = df['dP'].values.astype(np.float32)
        dp = np.nan_to_num(dp, nan=dp[~np.isnan(dp)].mean() if (~np.isnan(dp)).any() else 0)
        if len(dp) >= target_length:
            dp = dp[:target_length]
        else:
            dp = np.pad(dp, (0, target_length - len(dp)), mode='edge')
    else:
        dp = np.zeros(target_length, dtype=np.float32)

    # Get event label if available
    if 'lab' in df.columns:
        lab = df['lab'].values.astype(np.float32)
        if len(lab) >= target_length:
            lab = lab[:target_length]
        else:
            lab = np.pad(lab, (0, target_length - len(lab)), mode='edge')
    else:
        lab = np.zeros(target_length, dtype=np.float32)

    # Stack features: x, y, vx, vy, speed, ax, ay, acc_mag, pupil, label
    features = np.stack([x, y, vx, vy, speed, ax, ay, acc_mag, dp, lab], axis=1)

    return features


def create_session_classification_dataset(
    extracted_dir: str,
    output_path: str,
    task: str = 'TEX',
    target_length: int = 30000,  # 30 seconds at 1000Hz
    min_rounds_per_subject: int = 1
) -> Dict:
    """
    Create dataset for Session 1 vs Session 2 classification.

    This task tests whether MHTPN can detect temporal/fatigue patterns
    that might differ between the first and second session of a visit.

    Args:
        extracted_dir: Directory with extracted CSV files
        output_path: Where to save preprocessed data
        task: Task type (TEX, HSS, RAN)
        target_length: Sequence length
        min_rounds_per_subject: Minimum rounds needed for a subject

    Returns:
        Dict with dataset statistics
    """
    extracted_dir = Path(extracted_dir)

    # Find all extracted files
    csv_files = list(extracted_dir.glob(f"R*_S*_S*_{task}.csv"))
    print(f"Found {len(csv_files)} {task} files")

    # Parse file metadata
    file_info = []
    for f in csv_files:
        # Parse filename: R{round}_S{subject}_S{session}_{task}.csv
        name = f.stem
        parts = name.split('_')
        round_num = int(parts[0][1:])
        subject_id = int(parts[1][1:])
        session = int(parts[2][1:])

        file_info.append({
            'file': str(f),
            'round': round_num,
            'subject_id': subject_id,
            'session': session
        })

    df_meta = pd.DataFrame(file_info)

    # Group by subject and count rounds
    subject_rounds = df_meta.groupby('subject_id')['round'].nunique()
    valid_subjects = subject_rounds[subject_rounds >= min_rounds_per_subject].index.tolist()
    print(f"Subjects with >= {min_rounds_per_subject} rounds: {len(valid_subjects)}")

    # Filter to valid subjects
    df_meta = df_meta[df_meta['subject_id'].isin(valid_subjects)]

    # Process files
    X_list = []
    y_list = []
    groups = []
    metadata = []

    # Group by (subject, round) to ensure we have both sessions
    for (subject_id, round_num), group in tqdm(df_meta.groupby(['subject_id', 'round']),
                                                desc="Processing recordings"):
        sessions = group['session'].unique()

        # Need both sessions for this round
        if len(sessions) < 2:
            continue

        for _, row in group.iterrows():
            features = load_and_preprocess_file(row['file'], target_length)

            if features is None:
                continue

            X_list.append(features)
            y_list.append(row['session'] - 1)  # Session 1 -> 0, Session 2 -> 1
            groups.append(subject_id)  # Group by subject for CV
            metadata.append({
                'subject_id': int(subject_id),
                'round': int(round_num),
                'session': int(row['session']),
                'file': row['file']
            })

    if len(X_list) == 0:
        raise ValueError("No valid samples found!")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    groups = np.array(groups, dtype=np.int64)

    print(f"\nDataset shape: {X.shape}")
    print(f"Labels: Session 1 (y=0): {(y==0).sum()}, Session 2 (y=1): {(y==1).sum()}")
    print(f"Unique subjects: {len(np.unique(groups))}")

    # Save dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        groups=groups,
        feature_names=['x', 'y', 'vx', 'vy', 'speed', 'ax', 'ay', 'acc_mag', 'pupil', 'label']
    )

    # Save metadata
    meta_path = output_path.parent / f"{output_path.stem}_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump({
            'task': task,
            'target_length': target_length,
            'n_samples': len(y),
            'n_features': X.shape[2],
            'seq_length': X.shape[1],
            'class_counts': {'session_1': int((y==0).sum()), 'session_2': int((y==1).sum())},
            'n_subjects': len(np.unique(groups)),
            'samples': metadata
        }, f, indent=2)

    print(f"\nSaved dataset to {output_path}")
    print(f"Saved metadata to {meta_path}")

    return {
        'n_samples': len(y),
        'n_features': X.shape[2],
        'seq_length': X.shape[1],
        'class_balance': {'session_1': int((y==0).sum()), 'session_2': int((y==1).sum())},
        'n_subjects': len(np.unique(groups))
    }


def main():
    """Main preprocessing pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess GazeBase for MHTPN validation')
    parser.add_argument('--zip-path', type=str,
                       default='/workspace/Myasthenia-Gravis-Detection/data/external/downloads/GazeBase_v2_0.zip',
                       help='Path to GazeBase zip file')
    parser.add_argument('--task', type=str, default='TEX',
                       choices=['TEX', 'HSS', 'RAN', 'FXS', 'VD1', 'VD2', 'BLG'],
                       help='Task type to extract')
    parser.add_argument('--target-length', type=int, default=30000,
                       help='Target sequence length (default: 30000 = 30s at 1000Hz)')
    parser.add_argument('--output-dir', type=str,
                       default='/workspace/Myasthenia-Gravis-Detection/data/external/gazebase',
                       help='Output directory')

    args = parser.parse_args()

    # Paths
    extract_dir = Path(args.output_dir) / 'extracted' / args.task
    output_file = Path(args.output_dir) / 'processed' / f'gazebase_{args.task.lower()}_session_clf.npz'

    # Step 1: Extract from zip if not already done
    if not extract_dir.exists() or len(list(extract_dir.glob('*.csv'))) == 0:
        print("Step 1: Extracting data from zip...")
        extract_gazebase(args.zip_path, str(extract_dir), task=args.task)
    else:
        print(f"Step 1: Skipping extraction, found existing files in {extract_dir}")

    # Step 2: Create classification dataset
    print("\nStep 2: Creating classification dataset...")
    stats = create_session_classification_dataset(
        str(extract_dir),
        str(output_file),
        task=args.task,
        target_length=args.target_length
    )

    print("\n" + "="*50)
    print("Preprocessing complete!")
    print("="*50)
    print(f"Dataset: {output_file}")
    print(f"Samples: {stats['n_samples']}")
    print(f"Features: {stats['n_features']}")
    print(f"Sequence length: {stats['seq_length']}")
    print(f"Subjects: {stats['n_subjects']}")
    print(f"Class balance: {stats['class_balance']}")

    return stats


if __name__ == '__main__':
    main()
