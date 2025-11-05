#!/usr/bin/env python3
"""
Support script for Experiment 13F: Saccadic Duration Analysis
Analyzes data structure to determine feasibility of saccadic duration calculation.

According to medical team:
Duration: Defined as the time from saccade onset to the eye's arrival at the target position.

This analysis will determine if we can infer saccadic duration from the available data channels.
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

# Binary classification: HC vs MG (including Probable MG) - same as 13e
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
RESULTS_DIR = './results/support_13f'
RANDOM_STATE = 42

# Saccadic duration analysis parameters
VELOCITY_THRESHOLD = 30.0  # degrees/sec for saccade detection
TARGET_ARRIVAL_THRESHOLD = 2.0  # degrees - threshold for considering eye "arrived" at target
MIN_SACCADE_DURATION = 3  # minimum samples for valid saccade
SAMPLING_RATE_HZ = 1000  # Assumed sampling rate (need to verify from data)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def create_results_directory(results_dir):
    """Create results directory if it doesn't exist."""
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results directory created/verified: {results_dir}")

def analyze_data_structure_for_duration(sample_item, f_out):
    """Analyze the data structure to understand what's available for duration calculation."""
    f_out.write("="*80 + "\n")
    f_out.write("DATA STRUCTURE ANALYSIS FOR SACCADIC DURATION\n")
    f_out.write("="*80 + "\n")
    
    data = sample_item['data']
    f_out.write(f"Sample filename: {sample_item['filename']}\n")
    f_out.write(f"Sample class: {sample_item['class_name']}\n")
    f_out.write(f"Data shape: {data.shape}\n")
    f_out.write(f"Available columns: {FEATURE_COLUMNS}\n")
    
    df = pd.DataFrame(data, columns=FEATURE_COLUMNS)
    
    f_out.write("\nColumn analysis:\n")
    f_out.write("Position channels: LH, RH, LV, RV (eye positions)\n")
    f_out.write("Target channels: TargetH, TargetV (target positions)\n")
    
    # Check if we have the necessary data for duration calculation
    f_out.write("\nRequired for saccadic duration calculation:\n")
    f_out.write("1. Eye position data: ✓ Available (LH, RH, LV, RV)\n")
    f_out.write("2. Target position data: ✓ Available (TargetH, TargetV)\n")
    f_out.write("3. Temporal information: ? Need to infer from sequence\n")
    
    # Analyze temporal structure
    f_out.write(f"\nTemporal analysis:\n")
    f_out.write(f"Sequence length: {len(df)} samples\n")
    f_out.write(f"Assumed sampling rate: {SAMPLING_RATE_HZ} Hz\n")
    f_out.write(f"Total recording duration: {len(df)/SAMPLING_RATE_HZ:.2f} seconds\n")
    
    return df

def detect_saccades_with_duration(position_data, velocity_data, target_data, f_out):
    """
    Detect saccades and calculate their durations.
    
    Duration is defined as time from saccade onset to eye arrival at target position.
    """
    # Find points where absolute velocity exceeds threshold (saccade detection)
    above_threshold = np.abs(velocity_data) > VELOCITY_THRESHOLD
    
    saccades_with_duration = []
    in_saccade = False
    onset_idx = None
    
    for i, above in enumerate(above_threshold):
        if above and not in_saccade:
            # Saccade onset
            onset_idx = i
            in_saccade = True
        elif not above and in_saccade:
            # Potential saccade end - but we need to check target arrival
            if i - onset_idx >= MIN_SACCADE_DURATION:
                # Find when eye arrives at target position
                target_arrival_idx = find_target_arrival(
                    position_data, target_data, onset_idx, i, f_out
                )
                
                if target_arrival_idx is not None:
                    duration_samples = target_arrival_idx - onset_idx
                    duration_ms = (duration_samples / SAMPLING_RATE_HZ) * 1000  # Convert to milliseconds
                    
                    saccades_with_duration.append({
                        'onset_idx': onset_idx,
                        'end_idx': i-1,
                        'target_arrival_idx': target_arrival_idx,
                        'duration_samples': duration_samples,
                        'duration_ms': duration_ms
                    })
            in_saccade = False
    
    # Handle case where sequence ends during saccade
    if in_saccade and len(velocity_data) - onset_idx >= MIN_SACCADE_DURATION:
        target_arrival_idx = find_target_arrival(
            position_data, target_data, onset_idx, len(velocity_data)-1, f_out
        )
        if target_arrival_idx is not None:
            duration_samples = target_arrival_idx - onset_idx
            duration_ms = (duration_samples / SAMPLING_RATE_HZ) * 1000
            
            saccades_with_duration.append({
                'onset_idx': onset_idx,
                'end_idx': len(velocity_data)-1,
                'target_arrival_idx': target_arrival_idx,
                'duration_samples': duration_samples,
                'duration_ms': duration_ms
            })
    
    return saccades_with_duration

def find_target_arrival(position_data, target_data, onset_idx, saccade_end_idx, f_out):
    """
    Find when the eye arrives at the target position after saccade onset.
    
    Target arrival is defined as when eye position is within TARGET_ARRIVAL_THRESHOLD
    of the target position.
    """
    # Search from saccade onset to some time after saccade end
    search_end = min(len(position_data), saccade_end_idx + 50)  # Search up to 50 samples after saccade end
    
    for i in range(onset_idx, search_end):
        eye_pos = position_data[i]
        target_pos = target_data[i] if i < len(target_data) else target_data[-1]
        
        # Check if eye is within threshold of target
        distance_to_target = abs(eye_pos - target_pos)
        if distance_to_target <= TARGET_ARRIVAL_THRESHOLD:
            return i
    
    # If no clear target arrival found, return None
    return None

def analyze_saccadic_duration_feasibility(binary_items, f_out):
    """Analyze feasibility of saccadic duration calculation across all samples."""
    f_out.write("\n" + "="*80 + "\n")
    f_out.write("SACCADIC DURATION FEASIBILITY ANALYSIS\n")
    f_out.write("="*80 + "\n")
    
    print("\nAnalyzing saccadic duration feasibility...")
    
    all_durations = {'HC': [], 'MG': []}
    saccade_counts = {'HC': [], 'MG': []}
    successful_duration_calculations = {'HC': 0, 'MG': 0}
    total_saccades_detected = {'HC': 0, 'MG': 0}
    
    # Sample a subset for analysis (to avoid memory issues)
    sample_size = min(50, len(binary_items))
    sample_items = np.random.choice(binary_items, sample_size, replace=False)
    
    f_out.write(f"Analyzing {len(sample_items)} sample sequences...\n\n")
    
    for item in tqdm(sample_items, desc="Analyzing samples"):
        data = item['data']
        class_name = 'HC' if item['label'] == 0 else 'MG'
        
        df = pd.DataFrame(data, columns=FEATURE_COLUMNS)
        
        # Analyze horizontal saccades (primary direction)
        for eye, target in [('LH', 'TargetH'), ('RH', 'TargetH')]:
            position_data = df[eye].values
            target_data = df[target].values
            velocity_data = np.diff(position_data, prepend=position_data[0])
            
            # Detect saccades with duration calculation
            saccades_with_duration = detect_saccades_with_duration(
                position_data, velocity_data, target_data, f_out
            )
            
            total_saccades_detected[class_name] += len(saccades_with_duration)
            
            # Extract durations for successful calculations
            durations = [s['duration_ms'] for s in saccades_with_duration if s['duration_ms'] > 0]
            successful_duration_calculations[class_name] += len(durations)
            all_durations[class_name].extend(durations)
        
        # Count saccades per sequence
        sequence_saccade_count = sum(len(detect_saccades_with_duration(
            df[eye].values, 
            np.diff(df[eye].values, prepend=df[eye].values[0]), 
            df[target].values, 
            f_out
        )) for eye, target in [('LH', 'TargetH'), ('RH', 'TargetH')])
        
        saccade_counts[class_name].append(sequence_saccade_count)
    
    # Statistical analysis
    f_out.write("\n" + "-"*60 + "\n")
    f_out.write("STATISTICAL SUMMARY\n")
    f_out.write("-"*60 + "\n")
    
    for class_name in ['HC', 'MG']:
        durations = all_durations[class_name]
        counts = saccade_counts[class_name]
        
        f_out.write(f"\n{class_name} Class:\n")
        f_out.write(f"  Total saccades detected: {total_saccades_detected[class_name]}\n")
        f_out.write(f"  Successful duration calculations: {successful_duration_calculations[class_name]}\n")
        f_out.write(f"  Success rate: {successful_duration_calculations[class_name]/max(1, total_saccades_detected[class_name])*100:.1f}%\n")
        f_out.write(f"  Saccades per sequence: {np.mean(counts):.1f} ± {np.std(counts):.1f}\n")
        
        if durations:
            f_out.write(f"  Duration statistics:\n")
            f_out.write(f"    Mean: {np.mean(durations):.1f} ms\n")
            f_out.write(f"    Std: {np.std(durations):.1f} ms\n")
            f_out.write(f"    Min: {np.min(durations):.1f} ms\n")
            f_out.write(f"    Max: {np.max(durations):.1f} ms\n")
            f_out.write(f"    Median: {np.median(durations):.1f} ms\n")
        else:
            f_out.write(f"    No successful duration calculations\n")
    
    return all_durations, saccade_counts, successful_duration_calculations, total_saccades_detected

def assess_duration_variability(all_durations, f_out):
    """Assess if there's sufficient variability in saccadic durations for ablation study."""
    f_out.write("\n" + "="*80 + "\n")
    f_out.write("DURATION VARIABILITY ASSESSMENT FOR ABLATION STUDY\n")
    f_out.write("="*80 + "\n")
    
    # Combine all durations
    all_combined_durations = []
    for class_durations in all_durations.values():
        all_combined_durations.extend(class_durations)
    
    if not all_combined_durations:
        f_out.write("ERROR: No saccadic durations calculated. Ablation study not feasible.\n")
        return False
    
    # Calculate variability metrics
    durations_array = np.array(all_combined_durations)
    mean_duration = np.mean(durations_array)
    std_duration = np.std(durations_array)
    cv = std_duration / mean_duration if mean_duration > 0 else 0  # Coefficient of variation
    
    f_out.write(f"Overall duration statistics:\n")
    f_out.write(f"  Total durations analyzed: {len(all_combined_durations)}\n")
    f_out.write(f"  Mean duration: {mean_duration:.1f} ms\n")
    f_out.write(f"  Standard deviation: {std_duration:.1f} ms\n")
    f_out.write(f"  Coefficient of variation: {cv:.3f}\n")
    f_out.write(f"  Duration range: {np.min(durations_array):.1f} ms to {np.max(durations_array):.1f} ms\n")
    
    # Assess feasibility
    f_out.write(f"\nFEASIBILITY ASSESSMENT:\n")
    
    # Check if we have sufficient variability (CV > 0.2 is generally considered good variability)
    if cv > 0.2:
        f_out.write(f"✓ FEASIBLE: Coefficient of variation ({cv:.3f}) indicates sufficient duration variability\n")
        feasible = True
    else:
        f_out.write(f"✗ NOT FEASIBLE: Low coefficient of variation ({cv:.3f}) indicates limited duration variability\n")
        feasible = False
    
    # Check duration span
    duration_span = np.max(durations_array) - np.min(durations_array)
    if duration_span > mean_duration * 0.5:  # Duration span should be at least 50% of mean
        f_out.write(f"✓ FEASIBLE: Duration span ({duration_span:.1f} ms) is substantial relative to mean ({mean_duration:.1f} ms)\n")
    else:
        f_out.write(f"✗ CONCERN: Duration span ({duration_span:.1f} ms) is limited relative to mean ({mean_duration:.1f} ms)\n")
        feasible = False
    
    # Check if we have reasonable duration values (typical saccades are 20-100ms)
    if 20 <= mean_duration <= 200:
        f_out.write(f"✓ FEASIBLE: Mean duration ({mean_duration:.1f} ms) is within expected physiological range (20-200 ms)\n")
    else:
        f_out.write(f"✗ CONCERN: Mean duration ({mean_duration:.1f} ms) is outside expected physiological range\n")
        feasible = False
    
    # Suggest duration bins for ablation study
    if feasible:
        f_out.write(f"\nSUGGESTED DURATION BINS FOR ABLATION STUDY:\n")
        percentiles = [25, 50, 75]
        duration_percentiles = np.percentile(durations_array, percentiles)
        
        f_out.write(f"  Short durations: < {duration_percentiles[0]:.1f} ms (25th percentile)\n")
        f_out.write(f"  Medium durations: {duration_percentiles[0]:.1f} - {duration_percentiles[1]:.1f} ms (25th-50th percentile)\n")
        f_out.write(f"  Long durations: {duration_percentiles[1]:.1f} - {duration_percentiles[2]:.1f} ms (50th-75th percentile)\n")
        f_out.write(f"  Very long durations: > {duration_percentiles[2]:.1f} ms (75th percentile)\n")
    
    return feasible

def create_visualization_plots(all_durations, saccade_counts, results_dir, f_out):
    """Create visualization plots for the duration analysis."""
    f_out.write("\n" + "="*80 + "\n")
    f_out.write("CREATING VISUALIZATION PLOTS\n")
    f_out.write("="*80 + "\n")
    
    # Plot 1: Duration distribution by class
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    for class_name, durations in all_durations.items():
        if durations:
            plt.hist(durations, alpha=0.7, label=f'{class_name} (n={len(durations)})', bins=20)
    plt.xlabel('Saccadic Duration (ms)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Saccadic Durations by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Saccade count distribution
    plt.subplot(1, 3, 2)
    for class_name, counts in saccade_counts.items():
        if counts:
            plt.hist(counts, alpha=0.7, label=f'{class_name} (n={len(counts)})', bins=15)
    plt.xlabel('Saccades per Sequence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Saccade Counts by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Box plot comparison
    plt.subplot(1, 3, 3)
    duration_data = []
    duration_labels = []
    for class_name, durations in all_durations.items():
        if durations:
            duration_data.append(durations)
            duration_labels.append(class_name)
    
    if duration_data:
        plt.boxplot(duration_data, labels=duration_labels)
        plt.ylabel('Saccadic Duration (ms)')
        plt.title('Duration Distribution Comparison')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'saccadic_duration_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    f_out.write(f"Visualization plot saved to: {plot_path}\n")

def main():
    """Main execution function."""
    print("="*80)
    print("Support Analysis for Experiment 13F: Saccadic Duration Feasibility")
    print("="*80)
    
    # Set random seed
    np.random.seed(RANDOM_STATE)
    
    create_results_directory(RESULTS_DIR)
    summary_filepath = os.path.join(RESULTS_DIR, 'saccadic_duration_feasibility_analysis.txt')
    
    with open(summary_filepath, 'w', encoding='utf-8') as f_report:
        f_report.write("="*80 + "\n")
        f_report.write("Support Analysis for Experiment 13F: Saccadic Duration Feasibility\n")
        f_report.write("="*80 + "\n")
        f_report.write(f"Analysis generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_report.write(f"Base Directory: {BASE_DIR}\n")
        f_report.write(f"Velocity Threshold: {VELOCITY_THRESHOLD}°/sec\n")
        f_report.write(f"Target Arrival Threshold: {TARGET_ARRIVAL_THRESHOLD}°\n")
        f_report.write(f"Assumed Sampling Rate: {SAMPLING_RATE_HZ} Hz\n")
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
        
        # 2. Analyze Data Structure
        f_report.write("\nSTEP 2: ANALYZING DATA STRUCTURE FOR DURATION CALCULATION\n")
        f_report.write("-"*60 + "\n")
        
        if binary_items:
            sample_df = analyze_data_structure_for_duration(binary_items[0], f_report)
        else:
            f_report.write("No samples available for structure analysis.\n")
            return
        
        # 3. Saccadic Duration Feasibility Analysis
        f_report.write("\nSTEP 3: SACCADIC DURATION FEASIBILITY ANALYSIS\n")
        f_report.write("-"*55 + "\n")
        
        all_durations, saccade_counts, successful_calcs, total_saccades = analyze_saccadic_duration_feasibility(
            binary_items, f_report
        )
        
        # 4. Assess Variability for Ablation Study
        f_report.write("\nSTEP 4: ABLATION STUDY FEASIBILITY\n")
        f_report.write("-"*40 + "\n")
        
        is_feasible = assess_duration_variability(all_durations, f_report)
        
        # 5. Create Visualizations
        f_report.write("\nSTEP 5: CREATING VISUALIZATIONS\n")
        f_report.write("-"*35 + "\n")
        
        create_visualization_plots(all_durations, saccade_counts, RESULTS_DIR, f_report)
        
        # 6. Final Recommendations
        f_report.write("\n" + "="*80 + "\n")
        f_report.write("FINAL RECOMMENDATIONS FOR EXPERIMENT 13F\n")
        f_report.write("="*80 + "\n")
        
        # Calculate overall success rate
        total_success = sum(successful_calcs.values())
        total_detected = sum(total_saccades.values())
        overall_success_rate = (total_success / max(1, total_detected)) * 100
        
        f_report.write(f"Overall duration calculation success rate: {overall_success_rate:.1f}%\n\n")
        
        if is_feasible and overall_success_rate > 50:
            f_report.write("✓ RECOMMENDATION: PROCEED WITH EXPERIMENT 13F\n")
            f_report.write("\nJustification:\n")
            f_report.write("- Saccadic durations can be calculated from available position and target data\n")
            f_report.write("- Sufficient variability exists in saccadic durations for ablation study\n")
            f_report.write("- Duration calculation success rate is acceptable\n")
            f_report.write("- Duration values are within physiologically reasonable ranges\n")
            f_report.write("\nImplementation approach:\n")
            f_report.write("1. Detect saccade onset using velocity threshold (30°/sec)\n")
            f_report.write("2. Find target arrival as when eye position is within 2° of target\n")
            f_report.write("3. Calculate duration as time from onset to target arrival\n")
            f_report.write("4. Group sequences by duration percentiles for ablation study\n")
            f_report.write("5. Train models on each duration group to assess performance vs duration\n")
        else:
            f_report.write("✗ RECOMMENDATION: DO NOT PROCEED WITH EXPERIMENT 13F\n")
            f_report.write("\nJustification:\n")
            if overall_success_rate <= 50:
                f_report.write("- Low success rate in duration calculation (< 50%)\n")
            if not is_feasible:
                f_report.write("- Insufficient variability in saccadic durations detected\n")
            f_report.write("- Ablation study would not provide meaningful insights\n")
            f_report.write("- Consider alternative experimental approaches\n")
        
        f_report.write("\n" + "="*80 + "\n")
        f_report.write("End of Support Analysis\n")
        f_report.write("="*80 + "\n")
    
    print(f"\nSupport analysis completed!")
    print(f"Results saved to: {summary_filepath}")
    print(f"Recommendation: {'PROCEED' if is_feasible and overall_success_rate > 50 else 'DO NOT PROCEED'} with Experiment 13F")
    print("="*80)

if __name__ == '__main__':
    main()
