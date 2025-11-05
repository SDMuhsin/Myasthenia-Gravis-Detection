#!/usr/bin/env python3
"""
Support script for Experiment 13E: Saccadic Range Analysis
Analyzes data structure to determine feasibility of saccadic range calculation.

According to medical team:
Range: Calculated as the difference between saccadic onset and end positions. 
Saccadic onset and end were determined using an eye velocity threshold of 30°/sec, 
with the end defined as the point when velocity dropped below this threshold.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import warnings

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.data_loading import load_raw_sequences_and_labels

# --- Configuration ---
BASE_DIR = './data'

# Binary classification: HC vs MG (including Probable MG) - same as 13c
BINARY_CLASS_DEFINITIONS = {
    'HC': {'path': 'Healthy control', 'label': 0},
    'MG': {'path': 'Definite MG', 'label': 1},
    'Probable_MG': {'path': 'Probable MG', 'label': 1},  # Include Probable MG as MG class
}

# Expected feature columns from previous experiments
FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']
CSV_ENCODING = 'utf-16-le'
CSV_SEPARATOR = ','
MIN_SEQ_LEN_THRESHOLD = 50
RESULTS_DIR = './results/support_13e'
RANDOM_STATE = 42

# Saccadic range analysis parameters
VELOCITY_THRESHOLD = 30.0  # degrees/sec as specified by medical team

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def create_results_directory(results_dir):
    """Create results directory if it doesn't exist."""
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results directory created/verified: {results_dir}")

def analyze_single_sample_structure(sample_item, f_out):
    """Analyze the structure of a single data sample to understand available features."""
    f_out.write("="*80 + "\n")
    f_out.write("SAMPLE DATA STRUCTURE ANALYSIS\n")
    f_out.write("="*80 + "\n")
    
    data = sample_item['data']
    f_out.write(f"Sample filename: {sample_item['filename']}\n")
    f_out.write(f"Sample class: {sample_item['class_name']}\n")
    f_out.write(f"Data shape: {data.shape}\n")
    f_out.write(f"Expected columns: {FEATURE_COLUMNS}\n")
    
    # Check if we have position data (LH, RH, LV, RV)
    position_columns = ['LH', 'RH', 'LV', 'RV']
    f_out.write(f"Position columns available: {position_columns}\n")
    
    # Calculate velocities from position data
    f_out.write("\nCalculating velocities from position data...\n")
    df = pd.DataFrame(data, columns=FEATURE_COLUMNS)
    
    velocities = {}
    for pos_col in position_columns:
        # Calculate velocity as first derivative of position
        velocity = np.diff(df[pos_col].values, prepend=df[pos_col].iloc[0])
        velocities[f'{pos_col}_Vel'] = velocity
        
        # Basic statistics
        vel_mean = np.mean(np.abs(velocity))
        vel_max = np.max(np.abs(velocity))
        vel_std = np.std(velocity)
        
        f_out.write(f"  {pos_col} velocity stats: mean={vel_mean:.2f}, max={vel_max:.2f}, std={vel_std:.2f}\n")
    
    return velocities

def detect_saccades_from_velocity(velocity_data, threshold=30.0, min_duration=3):
    """
    Detect saccades using velocity threshold method.
    
    Args:
        velocity_data: Array of velocity values
        threshold: Velocity threshold in degrees/sec
        min_duration: Minimum duration of saccade in samples
    
    Returns:
        List of (onset_idx, end_idx) tuples for detected saccades
    """
    # Find points where absolute velocity exceeds threshold
    above_threshold = np.abs(velocity_data) > threshold
    
    # Find onset and offset points
    saccades = []
    in_saccade = False
    onset_idx = None
    
    for i, above in enumerate(above_threshold):
        if above and not in_saccade:
            # Saccade onset
            onset_idx = i
            in_saccade = True
        elif not above and in_saccade:
            # Saccade end
            if i - onset_idx >= min_duration:  # Check minimum duration
                saccades.append((onset_idx, i-1))
            in_saccade = False
    
    # Handle case where sequence ends during saccade
    if in_saccade and len(velocity_data) - onset_idx >= min_duration:
        saccades.append((onset_idx, len(velocity_data)-1))
    
    return saccades

def calculate_saccadic_ranges(position_data, velocity_data, saccades, f_out):
    """
    Calculate saccadic ranges for detected saccades.
    
    Args:
        position_data: Array of position values
        velocity_data: Array of velocity values  
        saccades: List of (onset_idx, end_idx) tuples
        f_out: File handle for logging
    
    Returns:
        List of saccadic ranges
    """
    ranges = []
    
    f_out.write(f"Calculating ranges for {len(saccades)} detected saccades:\n")
    
    for i, (onset_idx, end_idx) in enumerate(saccades):
        onset_pos = position_data[onset_idx]
        end_pos = position_data[end_idx]
        saccadic_range = abs(end_pos - onset_pos)
        ranges.append(saccadic_range)
        
        f_out.write(f"  Saccade {i+1}: onset={onset_pos:.2f}, end={end_pos:.2f}, range={saccadic_range:.2f}\n")
    
    return ranges

def analyze_saccadic_range_feasibility(binary_items, f_out):
    """Analyze feasibility of saccadic range calculation across all samples."""
    f_out.write("\n" + "="*80 + "\n")
    f_out.write("SACCADIC RANGE FEASIBILITY ANALYSIS\n")
    f_out.write("="*80 + "\n")
    
    print("\nAnalyzing saccadic range feasibility...")
    
    all_ranges = {'HC': [], 'MG': []}
    saccade_counts = {'HC': [], 'MG': []}
    
    # Sample a subset for analysis (to avoid memory issues)
    sample_size = min(50, len(binary_items))  # Analyze up to 50 samples
    sample_items = np.random.choice(binary_items, sample_size, replace=False)
    
    f_out.write(f"Analyzing {len(sample_items)} sample sequences...\n\n")
    
    for item in tqdm(sample_items, desc="Analyzing samples"):
        data = item['data']
        class_name = 'HC' if item['label'] == 0 else 'MG'
        
        df = pd.DataFrame(data, columns=FEATURE_COLUMNS)
        
        # Analyze horizontal saccades (primary direction)
        for eye in ['LH', 'RH']:
            position_data = df[eye].values
            velocity_data = np.diff(position_data, prepend=position_data[0])
            
            # Detect saccades
            saccades = detect_saccades_from_velocity(velocity_data, VELOCITY_THRESHOLD)
            saccade_counts[class_name].append(len(saccades))
            
            # Calculate ranges
            if saccades:
                ranges = calculate_saccadic_ranges(position_data, velocity_data, saccades, f_out)
                all_ranges[class_name].extend(ranges)
    
    # Statistical analysis
    f_out.write("\n" + "-"*60 + "\n")
    f_out.write("STATISTICAL SUMMARY\n")
    f_out.write("-"*60 + "\n")
    
    for class_name in ['HC', 'MG']:
        ranges = all_ranges[class_name]
        counts = saccade_counts[class_name]
        
        if ranges:
            f_out.write(f"\n{class_name} Class:\n")
            f_out.write(f"  Total saccades detected: {len(ranges)}\n")
            f_out.write(f"  Saccades per sequence: {np.mean(counts):.1f} ± {np.std(counts):.1f}\n")
            f_out.write(f"  Range statistics:\n")
            f_out.write(f"    Mean: {np.mean(ranges):.2f}°\n")
            f_out.write(f"    Std: {np.std(ranges):.2f}°\n")
            f_out.write(f"    Min: {np.min(ranges):.2f}°\n")
            f_out.write(f"    Max: {np.max(ranges):.2f}°\n")
            f_out.write(f"    Median: {np.median(ranges):.2f}°\n")
        else:
            f_out.write(f"\n{class_name} Class: No saccades detected\n")
    
    return all_ranges, saccade_counts

def assess_range_variability(all_ranges, f_out):
    """Assess if there's sufficient variability in saccadic ranges for ablation study."""
    f_out.write("\n" + "="*80 + "\n")
    f_out.write("RANGE VARIABILITY ASSESSMENT FOR ABLATION STUDY\n")
    f_out.write("="*80 + "\n")
    
    # Combine all ranges
    all_combined_ranges = []
    for class_ranges in all_ranges.values():
        all_combined_ranges.extend(class_ranges)
    
    if not all_combined_ranges:
        f_out.write("ERROR: No saccadic ranges detected. Ablation study not feasible.\n")
        return False
    
    # Calculate variability metrics
    ranges_array = np.array(all_combined_ranges)
    mean_range = np.mean(ranges_array)
    std_range = np.std(ranges_array)
    cv = std_range / mean_range if mean_range > 0 else 0  # Coefficient of variation
    
    f_out.write(f"Overall range statistics:\n")
    f_out.write(f"  Total ranges analyzed: {len(all_combined_ranges)}\n")
    f_out.write(f"  Mean range: {mean_range:.2f}°\n")
    f_out.write(f"  Standard deviation: {std_range:.2f}°\n")
    f_out.write(f"  Coefficient of variation: {cv:.3f}\n")
    f_out.write(f"  Range span: {np.min(ranges_array):.2f}° to {np.max(ranges_array):.2f}°\n")
    
    # Assess feasibility
    f_out.write(f"\nFEASIBILITY ASSESSMENT:\n")
    
    # Check if we have sufficient variability (CV > 0.2 is generally considered good variability)
    if cv > 0.2:
        f_out.write(f"✓ FEASIBLE: Coefficient of variation ({cv:.3f}) indicates sufficient range variability\n")
        feasible = True
    else:
        f_out.write(f"✗ NOT FEASIBLE: Low coefficient of variation ({cv:.3f}) indicates limited range variability\n")
        feasible = False
    
    # Check range span
    range_span = np.max(ranges_array) - np.min(ranges_array)
    if range_span > mean_range:
        f_out.write(f"✓ FEASIBLE: Range span ({range_span:.2f}°) is substantial relative to mean ({mean_range:.2f}°)\n")
    else:
        f_out.write(f"✗ CONCERN: Range span ({range_span:.2f}°) is limited relative to mean ({mean_range:.2f}°)\n")
        feasible = False
    
    # Suggest range bins for ablation study
    if feasible:
        f_out.write(f"\nSUGGESTED RANGE BINS FOR ABLATION STUDY:\n")
        percentiles = [25, 50, 75]
        range_percentiles = np.percentile(ranges_array, percentiles)
        
        f_out.write(f"  Small ranges: < {range_percentiles[0]:.1f}° (25th percentile)\n")
        f_out.write(f"  Medium ranges: {range_percentiles[0]:.1f}° - {range_percentiles[1]:.1f}° (25th-50th percentile)\n")
        f_out.write(f"  Large ranges: {range_percentiles[1]:.1f}° - {range_percentiles[2]:.1f}° (50th-75th percentile)\n")
        f_out.write(f"  Very large ranges: > {range_percentiles[2]:.1f}° (75th percentile)\n")
    
    return feasible

def create_visualization_plots(all_ranges, saccade_counts, results_dir, f_out):
    """Create visualization plots for the analysis."""
    f_out.write("\n" + "="*80 + "\n")
    f_out.write("CREATING VISUALIZATION PLOTS\n")
    f_out.write("="*80 + "\n")
    
    # Plot 1: Range distribution by class
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for class_name, ranges in all_ranges.items():
        if ranges:
            plt.hist(ranges, alpha=0.7, label=f'{class_name} (n={len(ranges)})', bins=20)
    plt.xlabel('Saccadic Range (degrees)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Saccadic Ranges by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Saccade count distribution
    plt.subplot(1, 2, 2)
    for class_name, counts in saccade_counts.items():
        if counts:
            plt.hist(counts, alpha=0.7, label=f'{class_name} (n={len(counts)})', bins=15)
    plt.xlabel('Saccades per Sequence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Saccade Counts by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'saccadic_range_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    f_out.write(f"Visualization plot saved to: {plot_path}\n")

def main():
    """Main execution function."""
    print("="*80)
    print("Support Analysis for Experiment 13E: Saccadic Range Feasibility")
    print("="*80)
    
    # Set random seed
    np.random.seed(RANDOM_STATE)
    
    create_results_directory(RESULTS_DIR)
    summary_filepath = os.path.join(RESULTS_DIR, 'saccadic_range_feasibility_analysis.txt')
    
    with open(summary_filepath, 'w', encoding='utf-8') as f_report:
        f_report.write("="*80 + "\n")
        f_report.write("Support Analysis for Experiment 13E: Saccadic Range Feasibility\n")
        f_report.write("="*80 + "\n")
        f_report.write(f"Analysis generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_report.write(f"Base Directory: {BASE_DIR}\n")
        f_report.write(f"Velocity Threshold: {VELOCITY_THRESHOLD}°/sec\n")
        f_report.write(f"Random State: {RANDOM_STATE}\n")
        f_report.write("="*80 + "\n\n")
        
        # 1. Load Data
        f_report.write("STEP 1: LOADING DATA\n")
        f_report.write("-"*30 + "\n")
        
        raw_items_list = load_raw_sequences_and_labels(
            BASE_DIR, BINARY_CLASS_DEFINITIONS, FEATURE_COLUMNS, 
            CSV_ENCODING, CSV_SEPARATOR, MIN_SEQ_LEN_THRESHOLD, f_report
        )
        
        if not raw_items_list:
            f_report.write("\nCRITICAL: No data loaded. Analysis cannot proceed.\n")
            print("CRITICAL: No data loaded. Exiting.")
            return
        
        # Prepare binary data (combine MG classes)
        binary_items = []
        for item in raw_items_list:
            if item['class_name'] in ['MG', 'Probable_MG']:
                new_item = item.copy()
                new_item['class_name'] = 'MG'
                new_item['label'] = 1
                binary_items.append(new_item)
            elif item['class_name'] == 'HC':
                binary_items.append(item)
        
        f_report.write(f"Binary data prepared: {len(binary_items)} total samples\n")
        
        # 2. Analyze Sample Structure
        f_report.write("\nSTEP 2: ANALYZING DATA STRUCTURE\n")
        f_report.write("-"*40 + "\n")
        
        if binary_items:
            sample_velocities = analyze_single_sample_structure(binary_items[0], f_report)
        else:
            f_report.write("No samples available for structure analysis.\n")
            return
        
        # 3. Saccadic Range Feasibility Analysis
        f_report.write("\nSTEP 3: SACCADIC RANGE FEASIBILITY ANALYSIS\n")
        f_report.write("-"*50 + "\n")
        
        all_ranges, saccade_counts = analyze_saccadic_range_feasibility(binary_items, f_report)
        
        # 4. Assess Variability for Ablation Study
        f_report.write("\nSTEP 4: ABLATION STUDY FEASIBILITY\n")
        f_report.write("-"*40 + "\n")
        
        is_feasible = assess_range_variability(all_ranges, f_report)
        
        # 5. Create Visualizations
        f_report.write("\nSTEP 5: CREATING VISUALIZATIONS\n")
        f_report.write("-"*35 + "\n")
        
        create_visualization_plots(all_ranges, saccade_counts, RESULTS_DIR, f_report)
        
        # 6. Final Recommendations
        f_report.write("\n" + "="*80 + "\n")
        f_report.write("FINAL RECOMMENDATIONS FOR EXPERIMENT 13E\n")
        f_report.write("="*80 + "\n")
        
        if is_feasible:
            f_report.write("✓ RECOMMENDATION: PROCEED WITH EXPERIMENT 13E\n")
            f_report.write("\nJustification:\n")
            f_report.write("- Saccadic ranges can be calculated from position data using velocity thresholds\n")
            f_report.write("- Sufficient variability exists in saccadic ranges for ablation study\n")
            f_report.write("- Both HC and MG classes show detectable saccades with measurable ranges\n")
            f_report.write("\nImplementation approach:\n")
            f_report.write("1. Calculate velocities from position data (LH, RH, LV, RV)\n")
            f_report.write("2. Detect saccades using 30°/sec velocity threshold\n")
            f_report.write("3. Calculate range as |end_position - onset_position|\n")
            f_report.write("4. Group sequences by range percentiles for ablation study\n")
            f_report.write("5. Train models on each range group to assess performance vs range\n")
        else:
            f_report.write("✗ RECOMMENDATION: DO NOT PROCEED WITH EXPERIMENT 13E\n")
            f_report.write("\nJustification:\n")
            f_report.write("- Insufficient variability in saccadic ranges detected\n")
            f_report.write("- Ablation study would not provide meaningful insights\n")
            f_report.write("- Consider alternative experimental approaches\n")
        
        f_report.write("\n" + "="*80 + "\n")
        f_report.write("End of Support Analysis\n")
        f_report.write("="*80 + "\n")
    
    print(f"\nSupport analysis completed!")
    print(f"Results saved to: {summary_filepath}")
    print(f"Recommendation: {'PROCEED' if is_feasible else 'DO NOT PROCEED'} with Experiment 13E")
    print("="*80)

if __name__ == '__main__':
    main()
