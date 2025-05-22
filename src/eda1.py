import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import iqr as scipy_iqr, mannwhitneyu
from datetime import datetime

# --- Configuration ---
BASE_DIR = './data' 

CLASS_DEFINITIONS = {
    'HC': {'path': 'Healthy control', 'label': 0},
    'MG': {'path': 'Definite MG', 'label': 1},
    'CNP3': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '3rd'), 'label': 2},
    'CNP4': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '4th'), 'label': 3},
    'CNP6': {'path': os.path.join('Non-MG diplopia (CNP, etc)', '6th'), 'label': 4},
    'TAO': {'path': os.path.join('Non-MG diplopia (CNP, etc)', 'TAO'), 'label': 5},
}
CLASS_MAPPING = {name: details['label'] for name, details in CLASS_DEFINITIONS.items()}
INV_CLASS_MAPPING = {details['label']: name for name, details in CLASS_DEFINITIONS.items()}
ORDERED_CLASS_NAMES = [INV_CLASS_MAPPING[i] for i in sorted(INV_CLASS_MAPPING.keys())]

FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV'] 
CSV_ENCODING = 'utf-16-le' 
CSV_SEPARATOR = ',' 
TARGET_SEQ_LEN_PERCENTILE = 95 
MIN_SEQ_LEN_THRESHOLD = 50 
RESULTS_DIR = './results'
NUMERICAL_SUMMARY_FILENAME = 'eda_numerical_summary.txt'

# --- Helper for descriptive stats ---
def get_descriptive_stats_df(data_series, name="Value"):
    if data_series.empty:
        return pd.Series({
            'Count': 0, 'Mean': np.nan, 'Std': np.nan, 'Min': np.nan, 
            'Q1 (25%)': np.nan, 'Median (50%)': np.nan, 'Q3 (75%)': np.nan, 'Max': np.nan, 'IQR': np.nan
        }, name=name)
    
    stats = {
        'Count': data_series.count(),
        'Mean': data_series.mean(),
        'Std': data_series.std(),
        'Min': data_series.min(),
        'Q1 (25%)': data_series.quantile(0.25),
        'Median (50%)': data_series.median(),
        'Q3 (75%)': data_series.quantile(0.75),
        'Max': data_series.max(),
    }
    stats['IQR'] = stats['Q3 (75%)'] - stats['Q1 (25%)']
    return pd.Series(stats, name=name)

# --- Data Loading and Feature Engineering (Adapted from user's code) ---
def load_raw_sequences_and_labels(base_dir, class_definitions_dict, feature_columns_expected, encoding, separator, min_seq_len_threshold, f_out):
    f_out.write("="*70 + "\n")
    f_out.write("Phase: Data Loading\n")
    f_out.write("="*70 + "\n")
    print("="*50)
    print("Starting Data Loading (with robust column handling and new class structure)...")
    print("="*50)
    raw_items = []
    
    for class_name_key, class_details in class_definitions_dict.items():
        label = class_details['label']
        current_class_path_segment = class_details['path']
        class_dir_abs = os.path.join(base_dir, current_class_path_segment)
        
        msg = f"\nProcessing Class: '{class_name_key}' (Label: {label}) from path: {class_dir_abs}\n"
        print(msg.strip())
        f_out.write(msg)

        if not os.path.isdir(class_dir_abs):
            warning_msg = f"WARNING: Class directory not found: {class_dir_abs}\n"
            print(warning_msg.strip())
            f_out.write(warning_msg)
            continue
            
        patient_dirs = [d for d in os.listdir(class_dir_abs) if os.path.isdir(os.path.join(class_dir_abs, d))]
        if not patient_dirs:
            info_msg = f"INFO: No patient directories found in {class_dir_abs} for class '{class_name_key}'\n"
            print(info_msg.strip())
            f_out.write(info_msg)
            continue
            
        for patient_folder_name in tqdm(patient_dirs, desc=f"  Patients in {class_name_key}"):
            patient_id = patient_folder_name 
            patient_dir_path = os.path.join(class_dir_abs, patient_folder_name)
            csv_files = glob.glob(os.path.join(patient_dir_path, '*.csv'))
            if not csv_files: continue

            for csv_file_path in csv_files:
                try:
                    df_full = pd.read_csv(csv_file_path, encoding=encoding, sep=separator)
                    original_columns = df_full.columns.tolist()
                    df_full.columns = [col.strip() for col in original_columns]
                    
                    missing_cols = [col for col in feature_columns_expected if col not in df_full.columns]
                    if missing_cols: continue
                        
                    df_features = df_full[feature_columns_expected]
                    if df_features.empty or len(df_features) < min_seq_len_threshold: continue 
                    
                    for col in df_features.columns:
                        df_features.loc[:, col] = pd.to_numeric(df_features[col], errors='coerce')
                    
                    if df_features.isnull().sum().sum() > 0.1 * df_features.size: continue
                    
                    df_features = df_features.fillna(0) 
                    sequence_data = df_features.values.astype(np.float32)
                    raw_items.append({
                        'data': sequence_data, 'label': label, 'patient_id': patient_id, 
                        'filename': os.path.basename(csv_file_path),
                        'class_name': class_name_key, 
                        'original_length': len(sequence_data)
                    })
                except Exception as e:
                    error_msg = f"ERROR processing {os.path.basename(csv_file_path)} (Patient: {patient_id}, Class: {class_name_key}): {type(e).__name__} - {e}. Skipping.\n"
                    print(error_msg.strip())
                    # f_out.write(error_msg) # Optional: log errors to file too
    
    summary_msg = f"\nData loading complete. Loaded {len(raw_items)} raw sequences meeting criteria.\n"
    print(summary_msg.strip())
    f_out.write(summary_msg)
    if not raw_items:
        crit_warning_msg = "CRITICAL WARNING: No raw sequences were loaded. Please check `BASE_DIR`, `CLASS_DEFINITIONS`, CSV files, and `MIN_SEQ_LEN_THRESHOLD`.\n"
        print(crit_warning_msg.strip())
        f_out.write(crit_warning_msg)
    f_out.write("-" * 70 + "\n\n")
    return raw_items

def engineer_features_from_raw_data(raw_items_dicts, original_feature_names, f_out):
    f_out.write("="*70 + "\n")
    f_out.write("Phase: Feature Engineering\n")
    f_out.write("="*70 + "\n")
    print("\n" + "="*50)
    print("Starting Feature Engineering (velocities, errors)...")
    print("="*50)

    if not raw_items_dicts:
        no_items_msg = "INFO: No raw items to engineer features from.\n"
        print(no_items_msg.strip())
        f_out.write(no_items_msg)
        f_out.write("-" * 70 + "\n\n")
        return [], 0, []

    engineered_items = []
    final_num_channels = 0
    final_feature_names = [] 

    for item in tqdm(raw_items_dicts, desc="  Engineering Features"):
        sequence_data_original = item['data'] 
        df_original = pd.DataFrame(sequence_data_original, columns=original_feature_names)
        df_engineered_parts = []
        for pos_col in ['LH', 'RH', 'LV', 'RV']:
            if pos_col in df_original.columns:
                velocity_series = df_original[pos_col].diff().fillna(0) 
                df_engineered_parts.append(velocity_series.rename(f'{pos_col}_Vel'))
            else: 
                df_engineered_parts.append(pd.Series(np.zeros(len(df_original)), name=f'{pos_col}_Vel'))
        target_h_exists = 'TargetH' in df_original.columns
        target_v_exists = 'TargetV' in df_original.columns
        error_definitions = [
            ('LH', 'TargetH', 'ErrorH_L', target_h_exists), ('RH', 'TargetH', 'ErrorH_R', target_h_exists),
            ('LV', 'TargetV', 'ErrorV_L', target_v_exists), ('RV', 'TargetV', 'ErrorV_R', target_v_exists)
        ]
        for eye_col, target_col, error_col_name, target_exists in error_definitions:
            if target_exists and eye_col in df_original.columns:
                df_engineered_parts.append((df_original[eye_col] - df_original[target_col]).rename(error_col_name))
            else: 
                df_engineered_parts.append(pd.Series(np.zeros(len(df_original)), name=error_col_name))
        if df_engineered_parts:
            df_engineered_combined = pd.concat(df_engineered_parts, axis=1)
            df_all_features = pd.concat([df_original, df_engineered_combined], axis=1)
        else: df_all_features = df_original
        if not final_feature_names: final_feature_names = df_all_features.columns.tolist()
        engineered_sequence_data = df_all_features.values.astype(np.float32) 
        engineered_items.append({
            'data': engineered_sequence_data, 'label': item['label'],
            'patient_id': item['patient_id'], 'filename': item['filename'],
            'class_name': item['class_name'], 'original_length': item['original_length']
        })
        if final_num_channels == 0: 
            final_num_channels = engineered_sequence_data.shape[1]
            if final_num_channels != len(final_feature_names):
                 crit_warn_msg = f"CRITICAL WARNING: Mismatch in derived channel count ({final_num_channels}) and feature name list length ({len(final_feature_names)}).\n"
                 print(crit_warn_msg.strip())
                 f_out.write(crit_warn_msg)

    summary_msg1 = f"\nFeature engineering complete.\n"
    summary_msg2 = f"  Number of features (channels) per sequence: {final_num_channels}\n"
    summary_msg3 = f"  Feature names: {final_feature_names}\n"
    print(summary_msg1.strip())
    print(summary_msg2.strip())
    print(summary_msg3.strip())
    f_out.write(summary_msg1 + summary_msg2 + summary_msg3)

    if not engineered_items: 
        no_items_warn = "WARNING: No items after feature engineering.\n"
        print(no_items_warn.strip())
        f_out.write(no_items_warn)
    f_out.write("-" * 70 + "\n\n")
    return engineered_items, final_num_channels, final_feature_names

def pad_or_truncate(sequence_data_features_cols, target_len, num_input_channels):
    current_len = sequence_data_features_cols.shape[0]
    if current_len == target_len: return sequence_data_features_cols
    if current_len > target_len: return sequence_data_features_cols[:target_len, :]
    padding = np.zeros((target_len - current_len, num_input_channels), dtype=np.float32)
    return np.vstack((sequence_data_features_cols, padding))

def preprocess_data_for_avg_ts(engineered_items_dicts, target_seq_len_percentile, num_actual_channels, f_out):
    f_out.write("="*70 + "\n")
    f_out.write(f"Phase: Preprocessing Data for Average Time Series (Padding/Truncating to {target_seq_len_percentile}th percentile length)\n")
    f_out.write("="*70 + "\n")
    if not engineered_items_dicts:
        no_items_msg = "INFO: No engineered items to preprocess for average time series.\n"
        print(no_items_msg.strip())
        f_out.write(no_items_msg)
        f_out.write("-" * 70 + "\n\n")
        return [], [], 0
        
    print("\n" + "="*50)
    print(f"Preprocessing Data for Average Time Series Plots (Padding/Truncating)...")
    print(f"  (Target length based on {target_seq_len_percentile}th percentile of original lengths)")
    print("="*50)
    lengths = [item['data'].shape[0] for item in engineered_items_dicts]

    if not lengths:
        no_len_warn = "WARNING: No sequence lengths found. Cannot determine target length for padding.\n"
        print(no_len_warn.strip())
        f_out.write(no_len_warn)
        f_out.write("-" * 70 + "\n\n")
        return [], [], 0
        
    target_len = int(np.percentile(lengths, target_seq_len_percentile))
    
    stats_msg = (f"  Original sequence lengths stats (before padding/truncation): "
                 f"Min={np.min(lengths)}, Max={np.max(lengths)}, "
                 f"Mean={np.mean(lengths):.2f}, Median={np.median(lengths)}\n")
    target_len_msg = f"  Calculated target sequence length for average time series: {target_len} time steps.\n"
    print(stats_msg.strip())
    print(target_len_msg.strip())
    f_out.write(stats_msg + target_len_msg)

    if target_len == 0: 
        target_len = max(1, np.max(lengths) if lengths else 1)
        adj_msg = f"  ADJUSTMENT: Target sequence length was 0, adjusted to: {target_len}\n"
        print(adj_msg.strip())
        f_out.write(adj_msg)

    all_data_processed_transposed, all_labels_processed = [], []
    for item in tqdm(engineered_items_dicts, desc="  Padding/Truncating sequences"):
        sequence_data = item['data'] 
        if sequence_data.shape[1] != num_actual_channels: continue # Error already printed by caller
        processed_seq = pad_or_truncate(sequence_data, target_len, num_actual_channels)
        all_data_processed_transposed.append(np.transpose(processed_seq, (1, 0))) 
        all_labels_processed.append(item['label'])
    
    summary_msg = f"\nPreprocessing for average time series complete. Processed {len(all_data_processed_transposed)} samples to length {target_len}.\n"
    print(summary_msg.strip())
    f_out.write(summary_msg)

    if not all_data_processed_transposed: 
        no_data_warn = "WARNING: No data after padding/truncation for avg TS.\n"
        print(no_data_warn.strip())
        f_out.write(no_data_warn)
    f_out.write("-" * 70 + "\n\n")
    return all_data_processed_transposed, all_labels_processed, target_len

def create_results_directory(dir_path=RESULTS_DIR):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"INFO: Created directory for results: {dir_path}")

def analyze_general_statistics(items_list_of_dicts, ordered_class_names_list, f_out):
    f_out.write("="*70 + "\n")
    f_out.write("I. General Data Statistics\n")
    f_out.write("="*70 + "\n")
    print("\n" + "="*50)
    print("General Data Statistics")
    print("="*50)

    if not items_list_of_dicts:
        no_data_msg = "  No data items to analyze. Skipping general statistics.\n"
        print(no_data_msg.strip())
        f_out.write(no_data_msg)
        f_out.write("-" * 70 + "\n\n")
        return pd.DataFrame()

    df = pd.DataFrame(items_list_of_dicts)
    num_total_sequences = len(df)
    num_total_patients = df['patient_id'].nunique()

    f_out.write(f"Total number of sequences (trials) loaded: {num_total_sequences}\n")
    f_out.write(f"Total number of unique patients: {num_total_patients}\n\n")
    print(f"  Total number of sequences (trials) loaded: {num_total_sequences}")
    print(f"  Total number of unique patients: {num_total_patients}")
    
    f_out.write("--- Statistics Per Class ---\n")
    print("\n  --- Statistics Per Class ---")
    for class_name_str in ordered_class_names_list:
        class_df = df[df['class_name'] == class_name_str] # Filter by 'class_name' string
        num_sequences_class = len(class_df)
        num_patients_class = class_df['patient_id'].nunique()
        
        # Corrected lines for class_stats_msg:
        class_stats_msg = (
            f"Class '{class_name_str}':\n"
            f"  Number of sequences: {num_sequences_class} ({((num_sequences_class / num_total_sequences) if num_total_sequences > 0 else 0.0):.2%} of total)\n"
            f"  Number of unique patients: {num_patients_class}\n"
        )
        
        # Corrected lines for print statements:
        print(f"  Class '{class_name_str}':")
        print(f"    Number of sequences: {num_sequences_class} ({((num_sequences_class / num_total_sequences) if num_total_sequences > 0 else 0.0):.2%} of total)")
        print(f"    Number of unique patients: {num_patients_class}")
        
        f_out.write(class_stats_msg)
    
    f_out.write("\n--- Patient Summary (Number of Sequences per Patient) ---\n")
    print("\n  --- Patient Summary (Number of Sequences per Patient) ---")
    patient_summary = df.groupby(['patient_id', 'class_name'])['filename'].count().reset_index(name='num_sequences')
    f_out.write(patient_summary.to_string() + "\n")
    print(patient_summary.to_string())
    
    f_out.write("-" * 70 + "\n\n")
    return df

def analyze_sequence_lengths(items_dataframe, results_dir_path, ordered_class_names_list, f_out):
    f_out.write("="*70 + "\n")
    f_out.write("II. Sequence Length Analysis (Original Lengths)\n")
    f_out.write("="*70 + "\n")
    print("\n" + "="*50)
    print("Sequence Length Analysis (based on original lengths before padding/truncation)")
    print("="*50)

    if 'original_length' not in items_dataframe.columns or items_dataframe.empty:
        no_data_msg = "  'original_length' not found or DataFrame empty. Skipping sequence length analysis.\n"
        print(no_data_msg.strip())
        f_out.write(no_data_msg)
        f_out.write("-" * 70 + "\n\n")
        return

    lengths = items_dataframe['original_length']
    f_out.write("--- Overall Original Sequence Lengths ---\n")
    overall_stats_s = get_descriptive_stats_df(lengths, "Overall").drop("IQR") # IQR is less common for length summary
    overall_stats_s['95th Percentile'] = np.percentile(lengths, 95) # User's target base
    overall_stats_s['User Target Percentile ('+str(TARGET_SEQ_LEN_PERCENTILE)+'th)'] = np.percentile(lengths, TARGET_SEQ_LEN_PERCENTILE)

    f_out.write(overall_stats_s.to_string() + "\n\n")
    print("  --- Overall Original Sequence Lengths ---")
    print(overall_stats_s.to_string())


    plt.figure(figsize=(max(10,len(ordered_class_names_list)*1.5), 7)) # Dynamic width
    sns.histplot(data=items_dataframe, x='original_length', hue='class_name', hue_order=ordered_class_names_list, kde=True, element="step", stat="density", common_norm=False)
    plt.title('Distribution of Original Sequence Lengths by Class')
    plt.xlabel('Sequence Length (time steps)'); plt.ylabel('Density')
    plt.grid(True, linestyle='--', alpha=0.7); plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0,0,0.85,1]) # Adjust for legend
    plot_path = os.path.join(results_dir_path, 'EDA_sequence_length_distribution.png')
    plt.savefig(plot_path); plt.close()
    print(f"  Saved sequence length distribution plot to: {plot_path}")
    
    f_out.write("--- Original Sequence Lengths by Class ---\n")
    print("\n  --- Original Sequence Lengths by Class ---")
    all_class_stats = []
    for class_name_str in ordered_class_names_list:
        class_lengths = items_dataframe[items_dataframe['class_name'] == class_name_str]['original_length']
        if class_lengths.empty: 
            stats_series = get_descriptive_stats_df(class_lengths, class_name_str) # Will show NaNs
        else:
            stats_series = get_descriptive_stats_df(class_lengths, class_name_str)
        all_class_stats.append(stats_series)
    
    summary_table_df = pd.concat(all_class_stats, axis=1)
    f_out.write(summary_table_df.to_string() + "\n")
    print(summary_table_df)
    f_out.write("-" * 70 + "\n\n")


def build_aggregated_features_df(engineered_items_list_of_dicts, list_of_feature_names, f_out):
    f_out.write("="*70 + "\n")
    f_out.write("Phase: Building Aggregated Features DataFrame\n")
    f_out.write("="*70 + "\n")
    print("\n" + "="*50)
    print("Building DataFrame of Aggregated Features (mean, median, std, etc. per sequence)...")
    print("="*50)

    if not engineered_items_list_of_dicts:
        no_items_msg = "  No engineered items to process for aggregation. Returning empty DataFrame.\n"
        print(no_items_msg.strip())
        f_out.write(no_items_msg)
        f_out.write("-" * 70 + "\n\n")
        return pd.DataFrame()

    aggregated_data_rows = []
    for item in tqdm(engineered_items_list_of_dicts, desc="  Aggregating features for each sequence"):
        seq_data_np = item['data'] 
        current_row = {'patient_id': item['patient_id'], 'filename': item['filename'], 
                       'class_name': item['class_name'], 'label': item['label'], 
                       'original_length': item['original_length']}
        for i, feature_name_str in enumerate(list_of_feature_names):
            ft_ts = seq_data_np[:, i]
            current_row[f'{feature_name_str}_mean'] = np.mean(ft_ts)
            current_row[f'{feature_name_str}_median'] = np.median(ft_ts)
            current_row[f'{feature_name_str}_std'] = np.std(ft_ts)
            current_row[f'{feature_name_str}_min'] = np.min(ft_ts)
            current_row[f'{feature_name_str}_max'] = np.max(ft_ts)
            current_row[f'{feature_name_str}_range'] = np.max(ft_ts) - np.min(ft_ts)
            current_row[f'{feature_name_str}_iqr'] = scipy_iqr(ft_ts)
        aggregated_data_rows.append(current_row)
    agg_df = pd.DataFrame(aggregated_data_rows)
    
    summary_msg = f"  Aggregated features DataFrame created with shape: {agg_df.shape}\n"
    print(summary_msg.strip())
    f_out.write(summary_msg)

    if agg_df.empty: 
        warn_msg = "WARNING: Aggregated features DataFrame is empty.\n"
        print(warn_msg.strip())
        f_out.write(warn_msg)
    f_out.write("-" * 70 + "\n\n")
    return agg_df

def analyze_aggregated_feature_stats(aggregated_dataframe, list_of_feature_names, ordered_class_names_list, results_dir_path, f_out):
    f_out.write("="*70 + "\n")
    f_out.write("III. Aggregated Feature Statistics (Summary of per-sequence stats)\n")
    f_out.write("="*70 + "\n")
    f_out.write("For each feature, the following shows descriptive statistics of its aggregated values\n")
    f_out.write("(e.g., 'mean_per_sequence', 'median_per_sequence') for each class.\n")
    f_out.write("Mann-Whitney U tests compare each class against 'HC' (Healthy Control).\n\n")

    print("\n" + "="*50)
    print("Analyzing Aggregated Feature Statistics (comparing classes using boxplots & M-W U test vs HC)")
    print("="*50)

    if aggregated_dataframe.empty:
        no_data_msg = "  Aggregated features DataFrame is empty. Skipping this analysis.\n"
        print(no_data_msg.strip())
        f_out.write(no_data_msg)
        f_out.write("-" * 70 + "\n\n")
        return

    stats_to_plot_suffixes = ['mean', 'median', 'std', 'iqr']
    agg_plots_subdir = os.path.join(results_dir_path, 'EDA_aggregated_feature_plots')
    create_results_directory(agg_plots_subdir)
    hc_class_name = 'HC'

    for feature_name_original in tqdm(list_of_feature_names, desc="  Generating plots & stats for aggregated features"):
        f_out.write(f"--- Feature: {feature_name_original} ---\n")
        for stat_suffix_str in stats_to_plot_suffixes:
            aggregated_col_name = f'{feature_name_original}_{stat_suffix_str}'
            if aggregated_col_name not in aggregated_dataframe.columns: continue

            f_out.write(f"  Aggregated Statistic Type: Values of '{stat_suffix_str}' calculated per sequence\n")
            
            # Plotting
            plt.figure(figsize=(max(9, len(ordered_class_names_list)*1.5), 7))
            sns.boxplot(data=aggregated_dataframe, x='class_name', y=aggregated_col_name, order=ordered_class_names_list)
            plt.title(f'Seq-{stat_suffix_str.capitalize()} of {feature_name_original} by Class', fontsize=10)
            plt.xlabel('Class'); plt.ylabel(f'Seq {stat_suffix_str.capitalize()} of {feature_name_original}')
            plt.xticks(rotation=30, ha="right"); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
            plot_filename = f'EDA_{feature_name_original}_seq_{stat_suffix_str}_boxplot.png'
            plot_path = os.path.join(agg_plots_subdir, plot_filename)
            plt.savefig(plot_path); plt.close()

            # Numerical summary for text file
            class_stats_list = []
            for class_name_str_iter in ordered_class_names_list:
                class_data = aggregated_dataframe[aggregated_dataframe['class_name'] == class_name_str_iter][aggregated_col_name].dropna()
                descriptive_stats = get_descriptive_stats_df(class_data, name=class_name_str_iter)
                class_stats_list.append(descriptive_stats)
            
            summary_df_for_stat = pd.concat(class_stats_list, axis=1)
            f_out.write(f"    Descriptive Statistics for '{aggregated_col_name}':\n")
            f_out.write(summary_df_for_stat.to_string() + "\n")

            # Mann-Whitney U tests vs HC
            if hc_class_name in ordered_class_names_list:
                hc_values = aggregated_dataframe[aggregated_dataframe['class_name'] == hc_class_name][aggregated_col_name].dropna()
                if not hc_values.empty:
                    f_out.write(f"    Mann-Whitney U tests (vs {hc_class_name}):\n")
                    for other_class_name in ordered_class_names_list:
                        if other_class_name == hc_class_name: continue
                        other_values = aggregated_dataframe[aggregated_dataframe['class_name'] == other_class_name][aggregated_col_name].dropna()
                        if not other_values.empty and (len(np.unique(hc_values)) > 1 or len(np.unique(other_values)) > 1 or not np.array_equal(np.unique(hc_values), np.unique(other_values))):
                            try:
                                u_stat, p_value = mannwhitneyu(hc_values, other_values, alternative='two-sided')
                                significance = "(Significant at p < 0.05)" if p_value < 0.05 else "(Not Significant)"
                                f_out.write(f"      {hc_class_name} vs {other_class_name}: p-value = {p_value:.4f} {significance}\n")
                                if p_value < 0.05 and "&" not in aggregated_col_name: # Avoid printing for plots
                                     print(f"    Significant diff (p={p_value:.4f}) in median of '{aggregated_col_name}' between '{hc_class_name}' and '{other_class_name}'.")
                            except ValueError: 
                                f_out.write(f"      {hc_class_name} vs {other_class_name}: Could not compute (e.g., all values identical).\n")
                        elif other_values.empty:
                             f_out.write(f"      {hc_class_name} vs {other_class_name}: No data for {other_class_name}.\n")
                        else: # Data was constant or identical
                             f_out.write(f"      {hc_class_name} vs {other_class_name}: Data identical or constant, test not applicable.\n")

            f_out.write("\n") # separator between stat types for a feature
        f_out.write("\n") # separator between features
    print(f"  Saved aggregated feature statistics boxplots to: {agg_plots_subdir}")
    print(f"  Note: For overall multi-class comparison, consider Kruskal-Wallis test followed by post-hoc tests if significant.")
    f_out.write("Note: For overall multi-class comparison, consider Kruskal-Wallis test followed by post-hoc tests (e.g., Dunn's test with Bonferroni correction) if an overall significance is found.\n")
    f_out.write("-" * 70 + "\n\n")


def analyze_pooled_feature_distributions(engineered_items_list_of_dicts, list_of_feature_names, ordered_class_names_list, results_dir_path, f_out):
    f_out.write("="*70 + "\n")
    f_out.write("IV. Pooled Raw Feature Value Distributions (Summary of all time steps combined per class)\n")
    f_out.write("="*70 + "\n")
    print("\n" + "="*50)
    print("Analyzing Pooled Raw Feature Value Distributions (all time steps, comparing classes)")
    print("="*50)

    if not engineered_items_list_of_dicts:
        no_data_msg = "  No engineered items. Skipping pooled feature distribution analysis.\n"
        print(no_data_msg.strip())
        f_out.write(no_data_msg)
        f_out.write("-" * 70 + "\n\n")
        return
        
    pooled_plots_subdir = os.path.join(results_dir_path, 'EDA_pooled_feature_dist_plots')
    create_results_directory(pooled_plots_subdir)

    for feature_idx, feature_name_str in enumerate(tqdm(list_of_feature_names, desc="  Generating plots & stats for pooled distributions")):
        f_out.write(f"--- Feature: {feature_name_str} ---\n")
        plt.figure(figsize=(max(10,len(ordered_class_names_list)*1.5), 7)); any_data_plotted = False # Dynamic width
        
        all_class_stats_pooled = []
        for class_name_val_str in ordered_class_names_list:
            pooled_values_for_class_list = [item['data'][:, feature_idx] for item in engineered_items_list_of_dicts if item['class_name'] == class_name_val_str]
            
            if not pooled_values_for_class_list: 
                descriptive_stats = get_descriptive_stats_df(pd.Series([], dtype=float), name=class_name_val_str)
                all_class_stats_pooled.append(descriptive_stats)
                continue

            pooled_values_flat = np.concatenate(pooled_values_for_class_list)
            if pooled_values_flat.size == 0: 
                descriptive_stats = get_descriptive_stats_df(pd.Series([], dtype=float), name=class_name_val_str)
                all_class_stats_pooled.append(descriptive_stats)
                continue

            sns.kdeplot(pooled_values_flat, label=f'{class_name_val_str} (N_seq={len(pooled_values_for_class_list)})', fill=True, alpha=0.3)
            any_data_plotted = True
            descriptive_stats = get_descriptive_stats_df(pd.Series(pooled_values_flat), name=class_name_val_str)
            all_class_stats_pooled.append(descriptive_stats)
        
        if not any_data_plotted: 
            plt.close()
            if not all_class_stats_pooled: # if no classes had any data at all
                 f_out.write("  No data found for any class for this feature.\n\n")
                 continue # skip to next feature
        else: # Save plot if data was plotted
            plt.title(f'Pooled Distribution of Raw Values for Feature: {feature_name_str}')
            plt.xlabel(f'Value of {feature_name_str}'); plt.ylabel('Density')
            plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout(rect=[0,0,0.85,1]) # Adjust for legend
            plot_path = os.path.join(pooled_plots_subdir, f'EDA_{feature_name_str}_pooled_distribution.png')
            plt.savefig(plot_path); plt.close()
        
        # Write numerical summary to file
        if all_class_stats_pooled:
            summary_df_pooled = pd.concat(all_class_stats_pooled, axis=1)
            f_out.write("  Descriptive Statistics of Pooled Raw Values:\n")
            f_out.write(summary_df_pooled.to_string() + "\n\n")
        else: # Should not happen if any_data_plotted was true, but as a fallback
            f_out.write("  No pooled data to summarize for this feature.\n\n")

    print(f"  Saved pooled feature distribution KDE plots to: {pooled_plots_subdir}")
    f_out.write("-" * 70 + "\n\n")


def plot_average_time_series(processed_data_transposed_list, processed_labels_list, list_of_feature_names, common_seq_len, inv_class_mapping_dict, ordered_class_names_list, results_dir_path, f_out):
    f_out.write("="*70 + "\n")
    f_out.write(f"V. Average Time Series (Mean values after padding/truncation to {common_seq_len} steps)\n")
    f_out.write("="*70 + "\n")
    print("\n" + "="*50)
    print("Plotting Average Time Series (after padding/truncation)")
    print("="*50)

    if not processed_data_transposed_list or common_seq_len == 0:
        no_data_msg = "  No processed data or zero seq length. Skipping average time series plots.\n"
        print(no_data_msg.strip())
        f_out.write(no_data_msg)
        f_out.write("-" * 70 + "\n\n")
        return
    if processed_data_transposed_list[0].shape[0] != len(list_of_feature_names):
        err_msg = f"  CRITICAL ERROR: Channel count mismatch. Cannot plot/report average time series.\n"
        print(err_msg.strip())
        f_out.write(err_msg)
        f_out.write("-" * 70 + "\n\n")
        return

    avg_ts_plots_subdir = os.path.join(results_dir_path, 'EDA_average_time_series_plots')
    create_results_directory(avg_ts_plots_subdir)
    time_axis = np.arange(common_seq_len)

    for channel_idx, feature_name_str in enumerate(tqdm(list_of_feature_names, desc="  Plotting & Reporting average time series per feature")):
        f_out.write(f"--- Feature: {feature_name_str} ---\n")
        plt.figure(figsize=(max(12, len(ordered_class_names_list)*1.2), 8)); any_data_plotted = False # Dynamic width
        
        for class_name_val_str in ordered_class_names_list:
            label_val = CLASS_MAPPING[class_name_val_str]
            sequences_for_class = [processed_data_transposed_list[k][channel_idx, :] 
                                   for k, s_label in enumerate(processed_labels_list) if s_label == label_val]
            
            f_out.write(f"  Class '{class_name_val_str}':\n")
            if not sequences_for_class: 
                f_out.write("    No data available for this class.\n")
                continue

            class_seq_np = np.array(sequences_for_class)
            mean_ts = np.mean(class_seq_np, axis=0)
            sem_ts = np.std(class_seq_np, axis=0) / np.sqrt(class_seq_np.shape[0]) if class_seq_np.shape[0] > 0 else np.zeros_like(mean_ts)
            
            plt.plot(time_axis, mean_ts, label=f'{class_name_val_str} (N={class_seq_np.shape[0]})')
            plt.fill_between(time_axis, mean_ts - sem_ts, mean_ts + sem_ts, alpha=0.2)
            any_data_plotted = True

            # Write mean and SEM series to file
            f_out.write(f"    Mean Time Series (length {common_seq_len}):\n    {np.array2string(mean_ts, precision=4, separator=', ', max_line_width=120)}\n")
            f_out.write(f"    SEM Time Series (length {common_seq_len}):\n    {np.array2string(sem_ts, precision=4, separator=', ', max_line_width=120)}\n\n")
        
        if not any_data_plotted: 
            plt.close()
            f_out.write("  No data plotted for any class for this feature.\n")
        else:
            plt.title(f'Avg Time Series: {feature_name_str} (Mean ± SEM)\n(Padded/Truncated to {common_seq_len} steps)')
            plt.xlabel('Time Step'); plt.ylabel(f'Value of {feature_name_str}')
            plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout(rect=[0,0,0.85,1]) # Adjust for legend
            plot_path = os.path.join(avg_ts_plots_subdir, f'EDA_{feature_name_str}_avg_timeseries.png')
            plt.savefig(plot_path); plt.close()
        f_out.write("\n") # Separator between features
        
    print(f"  Saved average time series plots to: {avg_ts_plots_subdir}")
    f_out.write("-" * 70 + "\n\n")


def analyze_feature_correlations_aggregated(aggregated_dataframe, list_of_feature_names, ordered_class_names_list, results_dir_path, f_out):
    f_out.write("="*70 + "\n")
    f_out.write("VI. Correlations of Aggregated Feature Means (per class)\n")
    f_out.write("="*70 + "\n")
    f_out.write("Correlation matrices are computed from sequence-mean feature values for each class.\n\n")
    print("\n" + "="*50)
    print("Analyzing Correlations of Aggregated Feature Means (per sequence, by class)")
    print("="*50)

    if aggregated_dataframe.empty:
        no_data_msg = "  Aggregated features DataFrame empty. Skipping correlation analysis.\n"
        print(no_data_msg.strip())
        f_out.write(no_data_msg)
        f_out.write("-" * 70 + "\n\n")
        return

    mean_feature_cols = [f'{fname}_mean' for fname in list_of_feature_names if f'{fname}_mean' in aggregated_dataframe.columns]
    if not mean_feature_cols:
        no_cols_msg = "  No mean aggregated feature columns found. Skipping correlation analysis.\n"
        print(no_cols_msg.strip())
        f_out.write(no_cols_msg)
        f_out.write("-" * 70 + "\n\n")
        return

    corr_plots_subdir = os.path.join(results_dir_path, 'EDA_correlation_plots')
    create_results_directory(corr_plots_subdir)

    for class_name_str in ordered_class_names_list:
        f_out.write(f"--- Class: {class_name_str} ---\n")
        class_agg_df = aggregated_dataframe[aggregated_dataframe['class_name'] == class_name_str][mean_feature_cols]
        cleaned_cols = [col.replace('_mean', '') for col in class_agg_df.columns]
        class_agg_df.columns = cleaned_cols

        if class_agg_df.shape[0] < 2 or class_agg_df.shape[1] < 2:
            info_msg = f"  INFO: Not enough data (samples: {class_agg_df.shape[0]}, features: {class_agg_df.shape[1]}) for class '{class_name_str}' for correlation matrix. Skipping.\n"
            print(info_msg.strip())
            f_out.write(info_msg + "\n")
            continue
            
        corr_matrix = class_agg_df.corr()
        f_out.write("Correlation Matrix (of sequence-mean feature values):\n")
        f_out.write(corr_matrix.to_string(float_format="%.3f") + "\n\n") # Format floats

        plt.figure(figsize=(max(12, len(cleaned_cols)*0.7), max(10, len(cleaned_cols)*0.6)))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, annot_kws={"size": 7 if len(cleaned_cols) < 20 else 5})
        plt.title(f'Correlation Matrix of Sequence-Mean Feature Values\nClass: {class_name_str}')
        plt.xticks(rotation=45, ha="right", fontsize=8); plt.yticks(rotation=0, fontsize=8); plt.tight_layout()
        plot_path = os.path.join(corr_plots_subdir, f'EDA_correlation_matrix_agg_means_{class_name_str}.png')
        plt.savefig(plot_path); plt.close()
        print(f"  Saved correlation heatmap for class '{class_name_str}' to: {plot_path}")
    f_out.write("-" * 70 + "\n\n")

# --- Main Script Execution for EDA ---
if __name__ == '__main__':
    print("="*70)
    print("Starting End-to-End Data Inspection and Statistics Gathering (Multi-Class)...")
    # ... (other print statements from previous version) ...
    print("="*70)

    create_results_directory(RESULTS_DIR)
    numerical_summary_filepath = os.path.join(RESULTS_DIR, NUMERICAL_SUMMARY_FILENAME)

    with open(numerical_summary_filepath, 'w', encoding='utf-8') as f_report:
        f_report.write("======================================================================\n")
        f_report.write("Exploratory Data Analysis - Numerical Summary\n")
        f_report.write("======================================================================\n")
        f_report.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_report.write(f"Base Directory: {BASE_DIR}\n")
        f_report.write(f"Class Definitions Used: {CLASS_DEFINITIONS}\n")
        f_report.write(f"Minimum Sequence Length Threshold: {MIN_SEQ_LEN_THRESHOLD}\n")
        f_report.write(f"Target Percentile for Avg. Time Series Length: {TARGET_SEQ_LEN_PERCENTILE}th\n")
        f_report.write("======================================================================\n\n")

        raw_items_list = load_raw_sequences_and_labels(
            BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS, CSV_ENCODING, CSV_SEPARATOR, MIN_SEQ_LEN_THRESHOLD, f_report
        )
        if not raw_items_list: print("\nCRITICAL: No data loaded. EDA cannot proceed."); exit()

        engineered_items_list, num_engineered_channels, list_engineered_feature_names = engineer_features_from_raw_data(
            raw_items_list, FEATURE_COLUMNS, f_report
        )
        if not engineered_items_list or not list_engineered_feature_names: 
            print("\nCRITICAL: No data after feature engineering. EDA cannot proceed."); exit()
        
        items_master_df = analyze_general_statistics(engineered_items_list, ORDERED_CLASS_NAMES, f_report)
        if items_master_df.empty: print("\nWARNING: DataFrame from engineered items is empty.")
        
        analyze_sequence_lengths(items_master_df, RESULTS_DIR, ORDERED_CLASS_NAMES, f_report)
        
        aggregated_features_master_df = build_aggregated_features_df(engineered_items_list, list_engineered_feature_names, f_report)
        if aggregated_features_master_df.empty: print("\nWARNING: Aggregated features DataFrame is empty.")
        
        analyze_aggregated_feature_stats(aggregated_features_master_df, list_engineered_feature_names, ORDERED_CLASS_NAMES, RESULTS_DIR, f_report)
        
        analyze_pooled_feature_distributions(engineered_items_list, list_engineered_feature_names, ORDERED_CLASS_NAMES, RESULTS_DIR, f_report)
        
        processed_data_for_avg_ts, processed_labels_for_avg_ts, common_length_for_avg_ts = preprocess_data_for_avg_ts(
            engineered_items_list, TARGET_SEQ_LEN_PERCENTILE, num_engineered_channels, f_report
        )
        if not processed_data_for_avg_ts: print("\nWARNING: No data for average time series plots.")
        else:
            plot_average_time_series(processed_data_for_avg_ts, processed_labels_for_avg_ts, list_engineered_feature_names, common_length_for_avg_ts, INV_CLASS_MAPPING, ORDERED_CLASS_NAMES, RESULTS_DIR, f_report)

        analyze_feature_correlations_aggregated(aggregated_features_master_df, list_engineered_feature_names, ORDERED_CLASS_NAMES, RESULTS_DIR, f_report)
        
        f_report.write("======================================================================\n")
        f_report.write("End of Numerical Summary Report\n")
        f_report.write("======================================================================\n")

    print(f"\nNumerical EDA summary saved to: {numerical_summary_filepath}")
    print("\n" + "="*70)
    print("Exploratory Data Analysis and Statistics Generation Finished.")
    print(f"All plots and the numerical summary have been saved in: {RESULTS_DIR}")
    print("="*70)
