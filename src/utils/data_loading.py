import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from scipy.stats import iqr as scipy_iqr

def parse_frequency_from_filename(filename):
    """Extracts saccade frequency (e.g., 0.5, 0.75, 1) from a filename."""
    match = re.search(r'\((\d+(\.\d+)?)\s*Hz\)', filename, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 'N/A'

def load_raw_sequences_and_labels(base_dir, class_definitions_dict, feature_columns_expected, 
                                encoding, separator, min_seq_len_threshold, f_out):
    """Loads data and extracts saccade frequency from filenames."""
    f_out.write("=" * 80 + "\n")
    f_out.write("Phase: Data Loading and Frequency Extraction\n")
    f_out.write("=" * 80 + "\n")
    print("\n" + "="*50 + "\nStarting Data Loading and Frequency Extraction...\n" + "="*50)
    raw_items = []

    for class_name_key, class_details in class_definitions_dict.items():
        label = class_details['label']
        class_dir_abs = os.path.join(base_dir, class_details['path'])
        msg = f"\nProcessing Class: '{class_name_key}' (Label: {label}) from path: {class_dir_abs}\n"
        print(msg.strip())
        f_out.write(msg)
        if not os.path.isdir(class_dir_abs):
            f_out.write(f"WARNING: Class directory not found: {class_dir_abs}\n")
            continue

        patient_dirs = [d for d in os.listdir(class_dir_abs) if os.path.isdir(os.path.join(class_dir_abs, d))]
        for patient_folder_name in tqdm(patient_dirs, desc=f"  Patients in {class_name_key}"):
            patient_dir_path = os.path.join(class_dir_abs, patient_folder_name)
            csv_files = glob.glob(os.path.join(patient_dir_path, '*.csv'))
            if not csv_files: continue

            for csv_file_path in csv_files:
                try:
                    df_full = pd.read_csv(csv_file_path, encoding=encoding, sep=separator)
                    df_full.columns = [col.strip() for col in df_full.columns]
                    if any(col not in df_full.columns for col in feature_columns_expected) or len(df_full) < min_seq_len_threshold:
                        continue
                    
                    df_features = df_full[feature_columns_expected].copy()
                    for col in df_features.columns:
                        df_features.loc[:, col] = pd.to_numeric(df_features[col], errors='coerce')
                    
                    if df_features.isnull().sum().sum() > 0.1 * df_features.size: continue
                    df_features = df_features.fillna(0)

                    # Extract frequency
                    frequency = parse_frequency_from_filename(os.path.basename(csv_file_path))

                    raw_items.append({
                        'data': df_features.values.astype(np.float32), 'label': label,
                        'patient_id': patient_folder_name, 'filename': os.path.basename(csv_file_path),
                        'class_name': class_name_key, 'frequency': frequency
                    })
                except Exception as e:
                    pass # Silently skip problematic files

    summary_msg = f"\nData loading complete. Loaded {len(raw_items)} raw sequences.\n"
    print(summary_msg.strip())
    f_out.write(summary_msg + "-" * 80 + "\n\n")
    return raw_items

def engineer_and_aggregate_features(raw_items_dicts, original_feature_names, f_out):
    """Combines feature engineering and aggregation into one step for efficiency."""
    f_out.write("="*80 + "\n")
    f_out.write("Phase: Feature Engineering and Aggregation\n")
    f_out.write("="*80 + "\n")
    print("\n" + "="*50 + "\nStarting Feature Engineering and Aggregation...\n" + "="*50)
    
    if not raw_items_dicts:
        f_out.write("No raw items to process. Returning empty DataFrame.\n")
        return pd.DataFrame(), []

    aggregated_data_rows = []
    engineered_feature_names = []

    for item in tqdm(raw_items_dicts, desc="  Engineering & Aggregating"):
        df_original = pd.DataFrame(item['data'], columns=original_feature_names)
        
        # --- Engineering ---
        df_engineered_parts = []
        for pos_col in ['LH', 'RH', 'LV', 'RV']:
            velocity_series = df_original[pos_col].diff().fillna(0)
            df_engineered_parts.append(velocity_series.rename(f'{pos_col}_Vel'))
        
        error_definitions = [('LH', 'TargetH', 'ErrorH_L'), ('RH', 'TargetH', 'ErrorH_R'),
                             ('LV', 'TargetV', 'ErrorV_L'), ('RV', 'TargetV', 'ErrorV_R')]
        for eye_col, target_col, error_col_name in error_definitions:
            df_engineered_parts.append((df_original[eye_col] - df_original[target_col]).rename(error_col_name))
        
        df_all_features = pd.concat([df_original] + df_engineered_parts, axis=1)
        if not engineered_feature_names: engineered_feature_names = df_all_features.columns.tolist()

        # --- Aggregation ---
        current_row = {
            'patient_id': item['patient_id'], 'filename': item['filename'],
            'class_name': item['class_name'], 'label': item['label'],
            'frequency': item['frequency']
        }
        for feature_name in engineered_feature_names:
            ft_ts = df_all_features[feature_name]
            current_row[f'{feature_name}_mean'] = np.mean(ft_ts)
            current_row[f'{feature_name}_std'] = np.std(ft_ts)
            current_row[f'{feature_name}_median'] = np.median(ft_ts)
            current_row[f'{feature_name}_iqr'] = scipy_iqr(ft_ts)
        aggregated_data_rows.append(current_row)

    agg_df = pd.DataFrame(aggregated_data_rows)
    f_out.write(f"Aggregated features DataFrame created with shape: {agg_df.shape}\n")
    f_out.write(f"Aggregated numerical features list: {[f for f in agg_df.columns if f not in ['patient_id', 'filename', 'class_name', 'label', 'frequency']]}\n\n")
    
    return agg_df, [f for f in agg_df.columns if '_mean' in f or '_std' in f or '_median' in f or '_iqr' in f]
