#!/usr/bin/env python3
"""
Experiment 15: Eye Difference Analysis Testbench

This testbench allows rapid validation of equations that quantify
the difference between left and right eyes in saccade data.

Goal: Find equations that show significant eye differences in MG patients
but not in healthy controls.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_timeseries_data, merge_mg_classes
from equations import (time_to_target_baseline, h2_mad_variability,
                       h11_combined_static_dynamic, h24_upward_vertical_saccades,
                       apply_equation_to_sequence, compute_eye_difference)
from validation import validate_equation, plot_validation_results


# Configuration
BASE_DIR = './data'
RESULTS_DIR = './results/exp_15_eyediff'

CLASS_DEFINITIONS = {
    'HC': {'path': 'Healthy control', 'label': 0},
    'MG': {'path': 'Definite MG', 'label': 1},
    'Probable_MG': {'path': 'Probable MG', 'label': 1},
}

FEATURE_COLUMNS = ['LH', 'RH', 'LV', 'RV', 'TargetH', 'TargetV']
CSV_ENCODING = 'utf-16-le'
CSV_SEPARATOR = ','
MIN_SEQ_LEN = 50


def run_testbench(equation_func, equation_name, metric_key='mean_time_to_target',
                 direction='horizontal', equation_kwargs=None):
    """
    Run the complete testbench for a given equation.

    Args:
        equation_func: Function that computes metrics for a single eye
        equation_name: Name of the equation for logging
        metric_key: Which metric to use for comparison (e.g., 'mean_time_to_target')
        direction: 'horizontal' or 'vertical' saccades
        equation_kwargs: Additional kwargs to pass to equation_func

    Returns:
        Dictionary with validation results
    """
    if equation_kwargs is None:
        equation_kwargs = {}

    print("\n" + "="*80)
    print(f"TESTBENCH RUN: {equation_name}")
    print(f"Direction: {direction}")
    print(f"Metric: {metric_key}")
    print("="*80)

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{equation_name.replace(' ', '_')}_{direction}_{timestamp}.log"
    log_path = os.path.join(RESULTS_DIR, log_filename)

    with open(log_path, 'w') as log_file:
        log_file.write(f"Experiment 15: Eye Difference Analysis\n")
        log_file.write(f"Equation: {equation_name}\n")
        log_file.write(f"Direction: {direction}\n")
        log_file.write(f"Metric: {metric_key}\n")
        log_file.write(f"Timestamp: {timestamp}\n")
        log_file.write("="*80 + "\n\n")

        # Step 1: Load data
        print("\nStep 1: Loading data...")
        log_file.write("Step 1: Loading Data\n")
        log_file.write("-"*80 + "\n")

        raw_sequences = load_timeseries_data(
            BASE_DIR, CLASS_DEFINITIONS, FEATURE_COLUMNS,
            CSV_ENCODING, CSV_SEPARATOR, MIN_SEQ_LEN
        )

        # Merge MG classes
        sequences = merge_mg_classes(raw_sequences)

        log_file.write(f"Total sequences: {len(sequences)}\n")
        log_file.write(f"HC: {sum(1 for s in sequences if s['class_name']=='HC')}\n")
        log_file.write(f"MG: {sum(1 for s in sequences if s['class_name']=='MG')}\n\n")

        # Step 2: Apply equation to all sequences
        print("\nStep 2: Applying equation to all sequences...")
        log_file.write("Step 2: Applying Equation\n")
        log_file.write("-"*80 + "\n")

        hc_differences = []
        mg_differences = []

        failed_sequences = 0

        for seq in sequences:
            try:
                # Apply equation
                results = apply_equation_to_sequence(
                    seq['data'], equation_func, FEATURE_COLUMNS,
                    **equation_kwargs
                )

                # Extract left and right metrics based on direction
                if direction == 'horizontal':
                    left_metrics = results['left_horizontal']
                    right_metrics = results['right_horizontal']
                elif direction == 'vertical':
                    left_metrics = results['left_vertical']
                    right_metrics = results['right_vertical']
                else:
                    raise ValueError(f"Invalid direction: {direction}")

                # Compute eye difference
                eye_diff = compute_eye_difference(left_metrics, right_metrics, metric_key)

                # Store by class
                if seq['class_name'] == 'HC':
                    hc_differences.append(eye_diff)
                elif seq['class_name'] == 'MG':
                    mg_differences.append(eye_diff)

            except Exception as e:
                failed_sequences += 1
                log_file.write(f"Failed on sequence {seq['filename']}: {str(e)}\n")

        log_file.write(f"\nProcessed sequences:\n")
        log_file.write(f"  HC: {len(hc_differences)} differences computed\n")
        log_file.write(f"  MG: {len(mg_differences)} differences computed\n")
        log_file.write(f"  Failed: {failed_sequences}\n\n")

        # Remove NaN values for reporting
        hc_valid = [d for d in hc_differences if not np.isnan(d)]
        mg_valid = [d for d in mg_differences if not np.isnan(d)]

        log_file.write(f"Valid (non-NaN) differences:\n")
        log_file.write(f"  HC: {len(hc_valid)}/{len(hc_differences)}\n")
        log_file.write(f"  MG: {len(mg_valid)}/{len(mg_differences)}\n\n")

        # Step 3: Validate equation
        print("\nStep 3: Validating equation...")
        log_file.write("Step 3: Validation\n")
        log_file.write("-"*80 + "\n")

        validation_results = validate_equation(
            hc_differences, mg_differences,
            equation_name=equation_name,
            results_dir=RESULTS_DIR
        )

        if validation_results is not None:
            # Write validation results to log
            log_file.write(f"\nValidation Results:\n")
            log_file.write(f"  HC mean: {validation_results['hc_mean']:.4f}\n")
            log_file.write(f"  HC median: {validation_results['hc_median']:.4f}\n")
            log_file.write(f"  MG mean: {validation_results['mg_mean']:.4f}\n")
            log_file.write(f"  MG median: {validation_results['mg_median']:.4f}\n")
            log_file.write(f"  Mann-Whitney p-value: {validation_results['mann_whitney_p']:.6f}\n")
            log_file.write(f"  Cohen's d: {validation_results['cohens_d']:.4f}\n")
            log_file.write(f"  Validation score: {validation_results['validation_score']}/3\n\n")

            # Step 4: Generate plots
            print("\nStep 4: Generating plots...")
            log_file.write("Step 4: Generating Plots\n")
            log_file.write("-"*80 + "\n")

            plot_validation_results(validation_results, RESULTS_DIR, equation_name)

            log_file.write(f"Plots saved to: {RESULTS_DIR}\n\n")
        else:
            log_file.write("Validation failed due to insufficient data.\n\n")

        log_file.write("="*80 + "\n")
        log_file.write("Testbench run complete.\n")

    print(f"\nLog file saved to: {log_path}")
    print("="*80 + "\n")

    return validation_results


def main():
    """Main entry point for testbench."""
    print("\n" + "="*80)
    print("EXPERIMENT 15: EYE DIFFERENCE ANALYSIS TESTBENCH")
    print("="*80)
    print("\nObjective: Find equations that discriminate affected eye in MG patients")
    print("Validation: MG should show significant L-R differences, HC should not\n")

    # Run baseline equation: Time to Target
    print("\n" + "="*80)
    print("BASELINE EQUATION: Time to Target")
    print("="*80)

    results_horizontal = run_testbench(
        equation_func=time_to_target_baseline,
        equation_name="Time_to_Target_Baseline",
        metric_key='mean_time_to_target',
        direction='horizontal',
        equation_kwargs={'threshold_degrees': 2.0, 'sampling_rate': 1000}
    )

    # Also run for vertical saccades
    results_vertical = run_testbench(
        equation_func=time_to_target_baseline,
        equation_name="Time_to_Target_Baseline_Vertical",
        metric_key='mean_time_to_target',
        direction='vertical',
        equation_kwargs={'threshold_degrees': 2.0, 'sampling_rate': 1000}
    )

    # Run H2: MAD Variability
    print("\n" + "="*80)
    print("H2: MAD VARIABILITY (Vertical)")
    print("="*80)

    results_h2_vertical = run_testbench(
        equation_func=h2_mad_variability,
        equation_name="H2_MAD_Variability_Vertical",
        metric_key='mad_position',
        direction='vertical',
        equation_kwargs={}
    )

    # Run H11: Combined Static-Dynamic
    print("\n" + "="*80)
    print("H11: COMBINED STATIC-DYNAMIC (Vertical)")
    print("="*80)

    results_h11_vertical = run_testbench(
        equation_func=h11_combined_static_dynamic,
        equation_name="H11_Combined_Static_Dynamic_Vertical",
        metric_key='combined_score',
        direction='vertical',
        equation_kwargs={}
    )

    # Run H24: Upward Vertical Saccades (BREAKTHROUGH)
    print("\n" + "="*80)
    print("H24: UPWARD VERTICAL SACCADES (BREAKTHROUGH)")
    print("="*80)

    results_h24_vertical = run_testbench(
        equation_func=h24_upward_vertical_saccades,
        equation_name="H24_Upward_Vertical_Saccades",
        metric_key='combined_score',
        direction='vertical',
        equation_kwargs={'sample_rate_hz': 120}
    )

    # Summary
    print("\n" + "="*80)
    print("TESTBENCH SUMMARY")
    print("="*80)

    print("\n" + "-"*80)
    print("BASELINE: Time to Target")
    print("-"*80)

    if results_horizontal is not None:
        print("\nHorizontal Saccades:")
        print(f"  Validation Score: {results_horizontal['validation_score']}/3")
        print(f"  Mann-Whitney p-value: {results_horizontal['mann_whitney_p']:.6f}")
        print(f"  Cohen's d: {results_horizontal['cohens_d']:.4f}")

    if results_vertical is not None:
        print("\nVertical Saccades:")
        print(f"  Validation Score: {results_vertical['validation_score']}/3")
        print(f"  Mann-Whitney p-value: {results_vertical['mann_whitney_p']:.6f}")
        print(f"  Cohen's d: {results_vertical['cohens_d']:.4f}")

    print("\n" + "-"*80)
    print("H2: MAD Variability (Vertical)")
    print("-"*80)

    if results_h2_vertical is not None:
        print("\nVertical Saccades:")
        print(f"  Validation Score: {results_h2_vertical['validation_score']}/3")
        print(f"  Mann-Whitney p-value: {results_h2_vertical['mann_whitney_p']:.6f}")
        print(f"  Cohen's d: {results_h2_vertical['cohens_d']:.4f}")

    print("\n" + "-"*80)
    print("H11: Combined Static-Dynamic (Vertical)")
    print("-"*80)

    if results_h11_vertical is not None:
        print("\nVertical Saccades:")
        print(f"  Validation Score: {results_h11_vertical['validation_score']}/3")
        print(f"  Mann-Whitney p-value: {results_h11_vertical['mann_whitney_p']:.6f}")
        print(f"  Cohen's d: {results_h11_vertical['cohens_d']:.4f}")

    print("\n" + "-"*80)
    print("H24: Upward Vertical Saccades (BREAKTHROUGH)")
    print("-"*80)

    if results_h24_vertical is not None:
        print("\nUpward Vertical Saccades:")
        print(f"  Validation Score: {results_h24_vertical['validation_score']}/3")
        print(f"  Mann-Whitney p-value: {results_h24_vertical['mann_whitney_p']:.6f}")
        print(f"  Cohen's d: {results_h24_vertical['cohens_d']:.4f}")

        if results_h24_vertical['cohens_d'] >= 0.5:
            print("\n  *** BREAKTHROUGH: Cohen's d >= 0.5 ACHIEVED! ***")
            print("  *** Publication-worthy effect size! ***")

    print("\n" + "="*80)
    print(f"All results saved to: {RESULTS_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
