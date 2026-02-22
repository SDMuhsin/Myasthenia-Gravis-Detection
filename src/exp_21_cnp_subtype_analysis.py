#!/usr/bin/env python3
"""
Experiment 21: CNP Subtype Analysis

Clinical team requested two analyses:

Part A: Directional Signatures for CNP Subtypes
- CNP_3rd: Should show asymmetry in BOTH horizontal AND vertical
- CNP_4th: Should show asymmetry primarily in VERTICAL
- CNP_6th: Should show asymmetry primarily in HORIZONTAL

Part B: CNP Multiclass Classification (3rd vs 4th vs 6th)
- Build classifier to distinguish CNP subtypes
- Use direction-specific features
- Report accuracy, confusion matrix, per-class metrics
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ML imports for Part B
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
BASE_DIR = './data'
RESULTS_DIR = './results/exp_21_cnp_subtype_analysis'
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']
CSV_ENCODING = 'utf-16-le'
CSV_SEPARATOR = ','
MIN_SEQ_LEN_THRESHOLD = 50
SAMPLE_RATE = 120  # Hz
SACCADE_THRESHOLD = 5.0  # degrees
MIN_SACCADES_REQUIRED = 3

# Saccade direction configurations
DIRECTIONS = {
    'vertical_up': {'axis': 'vertical', 'direction': 'positive'},
    'vertical_down': {'axis': 'vertical', 'direction': 'negative'},
    'horizontal_right': {'axis': 'horizontal', 'direction': 'positive'},
    'horizontal_left': {'axis': 'horizontal', 'direction': 'negative'},
}


def create_results_dir():
    """Create results directory if it doesn't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Results will be saved to: {RESULTS_DIR}")


# =============================================================================
# DATA LOADING (reused from exp_18/exp_20)
# =============================================================================

def load_data_from_folder(folder_path, class_name, label):
    """Load all CSV files from a folder structure."""
    items = []

    if not os.path.isdir(folder_path):
        print(f"  Warning: Directory not found: {folder_path}")
        return items

    patient_dirs = [d for d in os.listdir(folder_path)
                    if os.path.isdir(os.path.join(folder_path, d))]

    if patient_dirs:
        for patient_folder in tqdm(patient_dirs, desc=f"  Loading {class_name}", leave=False):
            patient_path = os.path.join(folder_path, patient_folder)
            csv_files = glob.glob(os.path.join(patient_path, '*.csv'))

            for csv_file in csv_files:
                item = load_single_csv(csv_file, class_name, label, patient_folder)
                if item is not None:
                    items.append(item)
    else:
        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        for csv_file in tqdm(csv_files, desc=f"  Loading {class_name}", leave=False):
            patient_id = os.path.splitext(os.path.basename(csv_file))[0]
            item = load_single_csv(csv_file, class_name, label, patient_id)
            if item is not None:
                items.append(item)

    return items


def load_single_csv(csv_path, class_name, label, patient_id):
    """Load a single CSV file."""
    try:
        df = pd.read_csv(csv_path, encoding=CSV_ENCODING, sep=CSV_SEPARATOR)
        df.columns = [col.strip() for col in df.columns]

        if any(col not in df.columns for col in FEATURE_COLUMNS):
            return None

        if len(df) < MIN_SEQ_LEN_THRESHOLD:
            return None

        df_features = df[FEATURE_COLUMNS].copy()
        for col in df_features.columns:
            df_features.loc[:, col] = pd.to_numeric(df_features[col], errors='coerce')

        if df_features.isnull().sum().sum() > 0.1 * df_features.size:
            return None

        df_features = df_features.fillna(0)

        return {
            'data': df_features.values.astype(np.float32),
            'label': label,
            'patient_id': patient_id,
            'filename': os.path.basename(csv_path),
            'class_name': class_name,
        }
    except Exception:
        return None


def load_cnp_data():
    """Load CNP subtype data only (3rd, 4th, 6th)."""
    print("\n" + "="*80)
    print("LOADING CNP DATA")
    print("="*80)

    cnp_base = os.path.join(BASE_DIR, 'Non-MG diplopia (CNP, etc)')

    print("\nLoading CNP subtypes...")
    cnp_3rd = load_data_from_folder(os.path.join(cnp_base, '3rd'), 'CNP_3rd', 0)
    cnp_4th = load_data_from_folder(os.path.join(cnp_base, '4th'), 'CNP_4th', 1)
    cnp_6th = load_data_from_folder(os.path.join(cnp_base, '6th'), 'CNP_6th', 2)

    print(f"\n  CNP 3rd: {len(cnp_3rd)} sequences")
    print(f"  CNP 4th: {len(cnp_4th)} sequences")
    print(f"  CNP 6th: {len(cnp_6th)} sequences")
    print(f"  Total: {len(cnp_3rd) + len(cnp_4th) + len(cnp_6th)} sequences")

    return cnp_3rd, cnp_4th, cnp_6th


def load_all_data_with_hc():
    """Load HC and CNP data for comparison."""
    print("\n" + "="*80)
    print("LOADING ALL DATA")
    print("="*80)

    # Load HC
    print("\nLoading Healthy Controls...")
    hc_items = load_data_from_folder(
        os.path.join(BASE_DIR, 'Healthy control'), 'HC', -1
    )
    print(f"  Loaded {len(hc_items)} HC sequences")

    # Load CNP
    cnp_3rd, cnp_4th, cnp_6th = load_cnp_data()

    return hc_items, cnp_3rd, cnp_4th, cnp_6th


# =============================================================================
# SACCADE DETECTION AND METRIC COMPUTATION
# =============================================================================

def detect_saccades(target_signal, direction='positive', threshold=SACCADE_THRESHOLD):
    """Detect saccade onset indices from target signal."""
    target_diff = np.diff(target_signal)

    if direction == 'positive':
        indices = np.where(target_diff > threshold)[0] + 1
    elif direction == 'negative':
        indices = np.where(target_diff < -threshold)[0] + 1
    else:
        indices = np.where(np.abs(target_diff) > threshold)[0] + 1

    return indices.tolist()


def compute_asymmetry_metric(eye_l, eye_r, target, saccade_indices):
    """
    Compute asymmetry metric for a set of saccades.
    Returns the mean absolute error asymmetry between eyes.
    """
    n_samples = len(eye_l)
    asymmetries = []

    for idx in saccade_indices:
        if idx >= n_samples:
            continue

        # Settling window: 200-400ms after saccade onset (24-48 samples at 120Hz)
        start = min(idx + 24, n_samples)
        end = min(idx + 48, n_samples)

        if end > start + 5:
            error_l = np.mean(np.abs(eye_l[start:end] - target[start:end]))
            error_r = np.mean(np.abs(eye_r[start:end] - target[start:end]))
            asymmetry = np.abs(error_l - error_r)
            asymmetries.append(asymmetry)

    if len(asymmetries) < MIN_SACCADES_REQUIRED:
        return np.nan

    return np.mean(asymmetries)


def compute_fat1_metric(eye_l, eye_r, target, saccade_indices):
    """FAT1: Error degradation (late - early error asymmetry)."""
    n_samples = len(eye_l)
    errors_l, errors_r = [], []

    for idx in saccade_indices:
        if idx >= n_samples:
            continue
        start = min(idx + 24, n_samples)
        end = min(idx + 48, n_samples)

        if end > start + 5:
            errors_l.append(np.mean(np.abs(eye_l[start:end] - target[start:end])))
            errors_r.append(np.mean(np.abs(eye_r[start:end] - target[start:end])))

    if len(errors_l) < 3:
        return np.nan

    third = max(1, len(errors_l) // 3)
    deg_l = np.mean(errors_l[-third:]) - np.mean(errors_l[:third])
    deg_r = np.mean(errors_r[-third:]) - np.mean(errors_r[:third])

    return np.abs(deg_l - deg_r)


def compute_metrics_for_item(item, axis='vertical', direction='positive'):
    """Compute multiple asymmetry metrics for a single item."""
    data = item['data']

    if axis == 'horizontal':
        eye_l, eye_r, target = data[:, 0], data[:, 1], data[:, 4]
    else:
        eye_l, eye_r, target = data[:, 2], data[:, 3], data[:, 5]

    saccade_indices = detect_saccades(target, direction=direction)

    if len(saccade_indices) < MIN_SACCADES_REQUIRED:
        return None

    asymmetry = compute_asymmetry_metric(eye_l, eye_r, target, saccade_indices)
    fat1 = compute_fat1_metric(eye_l, eye_r, target, saccade_indices)

    if np.isnan(asymmetry):
        return None

    return {
        'asymmetry': asymmetry,
        'fat1': fat1 if not np.isnan(fat1) else 0.0,
        'n_saccades': len(saccade_indices),
    }


# =============================================================================
# PART A: DIRECTIONAL SIGNATURES ANALYSIS
# =============================================================================

def run_part_a(cnp_3rd, cnp_4th, cnp_6th, hc_items):
    """
    Part A: Directional Signatures for CNP Subtypes

    For each CNP subtype, compare horizontal vs vertical asymmetry.
    Expected patterns:
    - CNP_3rd: Both horizontal AND vertical
    - CNP_4th: Primarily VERTICAL
    - CNP_6th: Primarily HORIZONTAL
    """
    print("\n" + "="*80)
    print("PART A: DIRECTIONAL SIGNATURES FOR CNP SUBTYPES")
    print("="*80)

    groups = {
        'HC': hc_items,
        'CNP_3rd': cnp_3rd,
        'CNP_4th': cnp_4th,
        'CNP_6th': cnp_6th,
    }

    # Compute metrics for each group and direction
    results = []

    for group_name, items in groups.items():
        print(f"\nProcessing {group_name} ({len(items)} sequences)...")

        for dir_name, dir_config in DIRECTIONS.items():
            axis = dir_config['axis']
            direction = dir_config['direction']

            asymmetries = []
            for item in items:
                metrics = compute_metrics_for_item(item, axis=axis, direction=direction)
                if metrics is not None:
                    asymmetries.append(metrics['asymmetry'])

            if len(asymmetries) >= 10:
                results.append({
                    'Group': group_name,
                    'Direction': dir_name,
                    'Axis': axis.upper(),
                    'Mean_Asymmetry': np.mean(asymmetries),
                    'Std_Asymmetry': np.std(asymmetries),
                    'Median_Asymmetry': np.median(asymmetries),
                    'N_Samples': len(asymmetries),
                })

    df_results = pd.DataFrame(results)

    # Compute H vs V ratios for each CNP subtype
    print("\n" + "-"*60)
    print("HORIZONTAL vs VERTICAL ASYMMETRY BY CNP SUBTYPE")
    print("-"*60)

    ratio_results = []

    for group in ['CNP_3rd', 'CNP_4th', 'CNP_6th', 'HC']:
        df_group = df_results[df_results['Group'] == group]

        # Get vertical asymmetry (mean of up and down)
        v_up = df_group[df_group['Direction'] == 'vertical_up']['Mean_Asymmetry'].values
        v_down = df_group[df_group['Direction'] == 'vertical_down']['Mean_Asymmetry'].values

        # Get horizontal asymmetry (mean of left and right)
        h_left = df_group[df_group['Direction'] == 'horizontal_left']['Mean_Asymmetry'].values
        h_right = df_group[df_group['Direction'] == 'horizontal_right']['Mean_Asymmetry'].values

        if len(v_up) > 0 and len(h_left) > 0:
            v_mean = (v_up[0] + v_down[0]) / 2 if len(v_down) > 0 else v_up[0]
            h_mean = (h_left[0] + h_right[0]) / 2 if len(h_right) > 0 else h_left[0]

            h_v_ratio = h_mean / (v_mean + 1e-6)

            ratio_results.append({
                'Group': group,
                'Vertical_Mean': v_mean,
                'Horizontal_Mean': h_mean,
                'H_V_Ratio': h_v_ratio,
                'Dominant_Direction': 'HORIZONTAL' if h_v_ratio > 1.0 else 'VERTICAL',
            })

            print(f"\n{group}:")
            print(f"  Vertical asymmetry:   {v_mean:.4f}")
            print(f"  Horizontal asymmetry: {h_mean:.4f}")
            print(f"  H/V Ratio: {h_v_ratio:.3f}")
            print(f"  Dominant: {'HORIZONTAL' if h_v_ratio > 1.0 else 'VERTICAL'}")

    df_ratios = pd.DataFrame(ratio_results)

    # Statistical comparison: Does the pattern match expectations?
    print("\n" + "-"*60)
    print("PATTERN MATCHING ANALYSIS")
    print("-"*60)

    pattern_matches = []

    # CNP_3rd should show BOTH (ratio close to 1)
    if 'CNP_3rd' in df_ratios['Group'].values:
        cnp3_ratio = df_ratios[df_ratios['Group'] == 'CNP_3rd']['H_V_Ratio'].values[0]
        cnp3_match = 0.5 < cnp3_ratio < 2.0  # Both directions
        pattern_matches.append({
            'Subtype': 'CNP_3rd',
            'Expected': 'BOTH (H+V)',
            'H_V_Ratio': cnp3_ratio,
            'Match': 'YES' if cnp3_match else 'NO',
        })
        print(f"\nCNP_3rd (3rd nerve):")
        print(f"  Expected: Both H and V (ratio 0.5-2.0)")
        print(f"  Observed: H/V = {cnp3_ratio:.3f}")
        print(f"  Match: {'YES' if cnp3_match else 'NO'}")

    # CNP_4th should show VERTICAL (ratio < 1)
    if 'CNP_4th' in df_ratios['Group'].values:
        cnp4_ratio = df_ratios[df_ratios['Group'] == 'CNP_4th']['H_V_Ratio'].values[0]
        cnp4_match = cnp4_ratio < 1.0  # Vertical dominant
        pattern_matches.append({
            'Subtype': 'CNP_4th',
            'Expected': 'VERTICAL (V > H)',
            'H_V_Ratio': cnp4_ratio,
            'Match': 'YES' if cnp4_match else 'NO',
        })
        print(f"\nCNP_4th (4th nerve):")
        print(f"  Expected: Primarily vertical (ratio < 1.0)")
        print(f"  Observed: H/V = {cnp4_ratio:.3f}")
        print(f"  Match: {'YES' if cnp4_match else 'NO'}")

    # CNP_6th should show HORIZONTAL (ratio > 1)
    if 'CNP_6th' in df_ratios['Group'].values:
        cnp6_ratio = df_ratios[df_ratios['Group'] == 'CNP_6th']['H_V_Ratio'].values[0]
        cnp6_match = cnp6_ratio > 1.0  # Horizontal dominant
        pattern_matches.append({
            'Subtype': 'CNP_6th',
            'Expected': 'HORIZONTAL (H > V)',
            'H_V_Ratio': cnp6_ratio,
            'Match': 'YES' if cnp6_match else 'NO',
        })
        print(f"\nCNP_6th (6th nerve):")
        print(f"  Expected: Primarily horizontal (ratio > 1.0)")
        print(f"  Observed: H/V = {cnp6_ratio:.3f}")
        print(f"  Match: {'YES' if cnp6_match else 'NO'}")

    df_patterns = pd.DataFrame(pattern_matches)

    # Compute effect sizes: CNP subtype vs HC for each direction
    print("\n" + "-"*60)
    print("EFFECT SIZES: CNP SUBTYPES vs HC")
    print("-"*60)

    effect_results = []

    for dir_name, dir_config in DIRECTIONS.items():
        axis = dir_config['axis']
        direction = dir_config['direction']

        # Get HC asymmetries
        hc_asymmetries = []
        for item in hc_items:
            metrics = compute_metrics_for_item(item, axis=axis, direction=direction)
            if metrics is not None:
                hc_asymmetries.append(metrics['asymmetry'])

        if len(hc_asymmetries) < 10:
            continue

        for cnp_name, cnp_items in [('CNP_3rd', cnp_3rd), ('CNP_4th', cnp_4th), ('CNP_6th', cnp_6th)]:
            cnp_asymmetries = []
            for item in cnp_items:
                metrics = compute_metrics_for_item(item, axis=axis, direction=direction)
                if metrics is not None:
                    cnp_asymmetries.append(metrics['asymmetry'])

            if len(cnp_asymmetries) < 10:
                continue

            # Cohen's d
            n1, n2 = len(hc_asymmetries), len(cnp_asymmetries)
            var1, var2 = np.var(hc_asymmetries, ddof=1), np.var(cnp_asymmetries, ddof=1)
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
            cohens_d = (np.mean(cnp_asymmetries) - np.mean(hc_asymmetries)) / (pooled_std + 1e-6)

            # Mann-Whitney U test
            _, p_value = stats.mannwhitneyu(cnp_asymmetries, hc_asymmetries, alternative='two-sided')

            effect_results.append({
                'Comparison': f'{cnp_name} vs HC',
                'Direction': dir_name,
                'Axis': axis.upper(),
                'Cohen_d': cohens_d,
                'p_value': p_value,
                'Significant': 'YES' if p_value < 0.05 else 'NO',
                'n_CNP': n2,
                'n_HC': n1,
            })

    df_effects = pd.DataFrame(effect_results)

    print("\n| Comparison     | Direction       | Cohen's d | p-value  | Sig? |")
    print("|----------------|-----------------|-----------|----------|------|")
    for _, row in df_effects.iterrows():
        sig = '*' if row['Significant'] == 'YES' else ''
        print(f"| {row['Comparison']:14s} | {row['Direction']:15s} | {row['Cohen_d']:>9.3f} | {row['p_value']:.4f}{sig:>2s} | {row['Significant']:>4s} |")

    # Save Part A results
    df_results.to_csv(os.path.join(RESULTS_DIR, 'part_a_directional_metrics.csv'), index=False)
    df_ratios.to_csv(os.path.join(RESULTS_DIR, 'part_a_hv_ratios.csv'), index=False)
    df_patterns.to_csv(os.path.join(RESULTS_DIR, 'part_a_pattern_matching.csv'), index=False)
    df_effects.to_csv(os.path.join(RESULTS_DIR, 'part_a_effect_sizes.csv'), index=False)

    print(f"\nPart A results saved to {RESULTS_DIR}")

    return df_results, df_ratios, df_patterns, df_effects


# =============================================================================
# PART B: CNP MULTICLASS CLASSIFICATION
# =============================================================================

def extract_features_for_item(item):
    """Extract direction-specific features for classification."""
    features = {}

    for dir_name, dir_config in DIRECTIONS.items():
        axis = dir_config['axis']
        direction = dir_config['direction']

        metrics = compute_metrics_for_item(item, axis=axis, direction=direction)

        if metrics is not None:
            features[f'{dir_name}_asymmetry'] = metrics['asymmetry']
            features[f'{dir_name}_fat1'] = metrics['fat1']
            features[f'{dir_name}_n_saccades'] = metrics['n_saccades']
        else:
            features[f'{dir_name}_asymmetry'] = np.nan
            features[f'{dir_name}_fat1'] = np.nan
            features[f'{dir_name}_n_saccades'] = 0

    # Add ratio features
    v_asym = np.nanmean([features.get('vertical_up_asymmetry', np.nan),
                         features.get('vertical_down_asymmetry', np.nan)])
    h_asym = np.nanmean([features.get('horizontal_left_asymmetry', np.nan),
                         features.get('horizontal_right_asymmetry', np.nan)])

    features['v_mean_asymmetry'] = v_asym
    features['h_mean_asymmetry'] = h_asym
    features['h_v_ratio'] = h_asym / (v_asym + 1e-6) if not np.isnan(v_asym) and not np.isnan(h_asym) else np.nan

    return features


def run_part_b(cnp_3rd, cnp_4th, cnp_6th):
    """
    Part B: CNP Multiclass Classification (3rd vs 4th vs 6th)
    """
    print("\n" + "="*80)
    print("PART B: CNP MULTICLASS CLASSIFICATION (3rd vs 4th vs 6th)")
    print("="*80)

    # Extract features for all CNP samples
    print("\nExtracting features...")

    all_features = []
    all_labels = []
    all_patient_ids = []

    for label, (items, name) in enumerate([(cnp_3rd, 'CNP_3rd'), (cnp_4th, 'CNP_4th'), (cnp_6th, 'CNP_6th')]):
        for item in tqdm(items, desc=f"  {name}", leave=False):
            features = extract_features_for_item(item)
            features['class_name'] = name
            features['label'] = label
            features['patient_id'] = item['patient_id']

            all_features.append(features)
            all_labels.append(label)
            all_patient_ids.append(item['patient_id'])

    df_features = pd.DataFrame(all_features)

    # Get feature columns
    feature_cols = [col for col in df_features.columns if col not in ['class_name', 'label', 'patient_id']]

    # Drop rows with too many NaN values (more than half)
    df_features = df_features.dropna(subset=feature_cols, thresh=len(feature_cols)//2)

    # Fill remaining NaN with column median, then 0 if still NaN
    for col in feature_cols:
        if df_features[col].isna().any():
            median_val = df_features[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df_features[col] = df_features[col].fillna(median_val)

    # Final safety check: replace any remaining NaN/inf with 0
    df_features[feature_cols] = df_features[feature_cols].replace([np.inf, -np.inf], 0.0)
    df_features[feature_cols] = df_features[feature_cols].fillna(0.0)

    print(f"\nTotal samples after filtering: {len(df_features)}")
    print(f"  CNP_3rd: {(df_features['label'] == 0).sum()}")
    print(f"  CNP_4th: {(df_features['label'] == 1).sum()}")
    print(f"  CNP_6th: {(df_features['label'] == 2).sum()}")

    # Prepare data for classification
    X = df_features[feature_cols].values
    y = df_features['label'].values
    class_names = ['CNP_3rd', 'CNP_4th', 'CNP_6th']

    # Final check for NaN values
    if np.isnan(X).any():
        print("Warning: Replacing remaining NaN values with 0")
        X = np.nan_to_num(X, nan=0.0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define classifiers to test
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42, class_weight='balanced'),
    }

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    best_acc = 0
    best_clf_name = None
    best_y_pred = None

    print("\n" + "-"*60)
    print("CLASSIFICATION RESULTS (5-Fold Cross-Validation)")
    print("-"*60)

    for clf_name, clf in classifiers.items():
        print(f"\n{clf_name}:")

        y_pred = cross_val_predict(clf, X_scaled, y, cv=cv)

        acc = accuracy_score(y, y_pred)
        f1_macro = f1_score(y, y_pred, average='macro')
        f1_weighted = f1_score(y, y_pred, average='weighted')
        precision_macro = precision_score(y, y_pred, average='macro')
        recall_macro = recall_score(y, y_pred, average='macro')

        results.append({
            'Classifier': clf_name,
            'Accuracy': acc,
            'F1_Macro': f1_macro,
            'F1_Weighted': f1_weighted,
            'Precision_Macro': precision_macro,
            'Recall_Macro': recall_macro,
        })

        print(f"  Accuracy:    {acc:.3f}")
        print(f"  F1 (macro):  {f1_macro:.3f}")
        print(f"  F1 (weighted): {f1_weighted:.3f}")

        if acc > best_acc:
            best_acc = acc
            best_clf_name = clf_name
            best_y_pred = y_pred

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Accuracy', ascending=False)

    print("\n" + "-"*60)
    print(f"BEST CLASSIFIER: {best_clf_name} (Accuracy: {best_acc:.3f})")
    print("-"*60)

    # Detailed classification report for best classifier
    print("\nClassification Report:")
    print(classification_report(y, best_y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y, best_y_pred)

    print("\nConfusion Matrix:")
    print(f"             {'  '.join(class_names)}")
    for i, row in enumerate(cm):
        print(f"{class_names[i]:10s}  {row}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'CNP Subtype Classification\n{best_clf_name} (Accuracy: {best_acc:.3f})')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'part_b_confusion_matrix.png'), dpi=150)
    plt.close()

    # Per-class metrics
    per_class_results = []
    for i, name in enumerate(class_names):
        mask = y == i
        class_acc = (best_y_pred[mask] == y[mask]).mean()

        # One-vs-rest metrics
        y_binary = (y == i).astype(int)
        y_pred_binary = (best_y_pred == i).astype(int)

        precision = precision_score(y_binary, y_pred_binary)
        recall = recall_score(y_binary, y_pred_binary)
        f1 = f1_score(y_binary, y_pred_binary)

        per_class_results.append({
            'Class': name,
            'N_Samples': mask.sum(),
            'Accuracy': class_acc,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
        })

    df_per_class = pd.DataFrame(per_class_results)

    print("\n" + "-"*60)
    print("PER-CLASS METRICS")
    print("-"*60)
    print("\n| Class    | N    | Accuracy | Precision | Recall | F1     |")
    print("|----------|------|----------|-----------|--------|--------|")
    for _, row in df_per_class.iterrows():
        print(f"| {row['Class']:8s} | {row['N_Samples']:>4d} | {row['Accuracy']:.3f}    | {row['Precision']:.3f}     | {row['Recall']:.3f}  | {row['F1']:.3f}  |")

    # Feature importance (if using Random Forest)
    if 'Random Forest' in classifiers:
        print("\n" + "-"*60)
        print("FEATURE IMPORTANCE (Random Forest)")
        print("-"*60)

        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X_scaled, y)

        importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)

        print("\nTop 10 features:")
        for _, row in importance.head(10).iterrows():
            print(f"  {row['Feature']:30s}: {row['Importance']:.4f}")

        importance.to_csv(os.path.join(RESULTS_DIR, 'part_b_feature_importance.csv'), index=False)

    # Save Part B results
    df_results.to_csv(os.path.join(RESULTS_DIR, 'part_b_classifier_comparison.csv'), index=False)
    df_per_class.to_csv(os.path.join(RESULTS_DIR, 'part_b_per_class_metrics.csv'), index=False)
    df_features.to_csv(os.path.join(RESULTS_DIR, 'part_b_feature_matrix.csv'), index=False)

    print(f"\nPart B results saved to {RESULTS_DIR}")

    return df_results, df_per_class, best_acc, best_clf_name


# =============================================================================
# MAIN
# =============================================================================

def generate_report(part_a_results, part_b_results):
    """Generate final report."""
    df_ratios, df_patterns, df_effects = part_a_results[1], part_a_results[2], part_a_results[3]
    df_classifiers, df_per_class, best_acc, best_clf = part_b_results

    report_path = os.path.join(RESULTS_DIR, 'REPORT.md')

    with open(report_path, 'w') as f:
        f.write("# Experiment 21: CNP Subtype Analysis\n\n")

        f.write("## Overview\n\n")
        f.write("This experiment addresses clinical team feedback with two analyses:\n")
        f.write("1. **Part A:** Directional signatures for CNP subtypes\n")
        f.write("2. **Part B:** Multiclass classification (3rd vs 4th vs 6th)\n\n")

        # Part A
        f.write("---\n\n")
        f.write("## Part A: Directional Signatures\n\n")

        f.write("### Clinical Expectation\n\n")
        f.write("Based on neuroanatomy:\n")
        f.write("- **CNP_3rd (oculomotor):** Both horizontal AND vertical\n")
        f.write("- **CNP_4th (trochlear):** Primarily VERTICAL\n")
        f.write("- **CNP_6th (abducens):** Primarily HORIZONTAL\n\n")

        f.write("### Results: H/V Ratio by Subtype\n\n")
        f.write("| Subtype  | Vertical | Horizontal | H/V Ratio | Dominant |\n")
        f.write("|----------|----------|------------|-----------|----------|\n")
        for _, row in df_ratios.iterrows():
            f.write(f"| {row['Group']} | {row['Vertical_Mean']:.4f} | {row['Horizontal_Mean']:.4f} | {row['H_V_Ratio']:.3f} | {row['Dominant_Direction']} |\n")

        f.write("\n### Pattern Matching\n\n")
        f.write("| Subtype  | Expected | H/V Ratio | Match |\n")
        f.write("|----------|----------|-----------|-------|\n")
        for _, row in df_patterns.iterrows():
            f.write(f"| {row['Subtype']} | {row['Expected']} | {row['H_V_Ratio']:.3f} | {row['Match']} |\n")

        matches = df_patterns['Match'].value_counts().get('YES', 0)
        total = len(df_patterns)
        f.write(f"\n**Summary:** {matches}/{total} subtypes match expected directional patterns.\n\n")

        # Part B
        f.write("---\n\n")
        f.write("## Part B: Multiclass Classification\n\n")

        f.write("### Classifier Comparison (5-Fold CV)\n\n")
        f.write("| Classifier | Accuracy | F1 (Macro) | F1 (Weighted) |\n")
        f.write("|------------|----------|------------|---------------|\n")
        for _, row in df_classifiers.iterrows():
            f.write(f"| {row['Classifier']} | {row['Accuracy']:.3f} | {row['F1_Macro']:.3f} | {row['F1_Weighted']:.3f} |\n")

        f.write(f"\n**Best Classifier:** {best_clf} (Accuracy: {best_acc:.3f})\n\n")

        f.write("### Per-Class Metrics\n\n")
        f.write("| Class | N | Accuracy | Precision | Recall | F1 |\n")
        f.write("|-------|---|----------|-----------|--------|----|\n")
        for _, row in df_per_class.iterrows():
            f.write(f"| {row['Class']} | {row['N_Samples']} | {row['Accuracy']:.3f} | {row['Precision']:.3f} | {row['Recall']:.3f} | {row['F1']:.3f} |\n")

        # Conclusions
        f.write("\n---\n\n")
        f.write("## Key Findings\n\n")

        f.write("### Part A: Directional Signatures\n\n")
        f.write("*Results indicate whether CNP subtypes show expected directional patterns.*\n\n")

        f.write("### Part B: Classification\n\n")
        f.write(f"- Best accuracy: **{best_acc:.1%}** using {best_clf}\n")
        f.write(f"- Baseline (random): 33.3%\n")
        f.write(f"- Improvement over baseline: **{((best_acc - 0.333) / 0.333) * 100:.1f}%**\n\n")

        f.write("---\n")
        f.write(f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*\n")

    print(f"\nReport saved to: {report_path}")


def main():
    print("="*80)
    print("EXPERIMENT 21: CNP SUBTYPE ANALYSIS")
    print("="*80)

    create_results_dir()

    # Load data
    hc_items, cnp_3rd, cnp_4th, cnp_6th = load_all_data_with_hc()

    # Part A: Directional Signatures
    part_a_results = run_part_a(cnp_3rd, cnp_4th, cnp_6th, hc_items)

    # Part B: Multiclass Classification
    part_b_results = run_part_b(cnp_3rd, cnp_4th, cnp_6th)

    # Generate Report
    print("\n" + "="*80)
    print("GENERATING FINAL REPORT")
    print("="*80)

    generate_report(part_a_results, part_b_results)

    print("\n" + "="*80)
    print("EXPERIMENT 21 COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
