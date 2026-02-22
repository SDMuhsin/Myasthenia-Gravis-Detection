# Experiment 15: Eye Difference Analysis - File Index

## Overview
This experiment develops equations to identify which eye is more affected in MG patients.

## Directory Structure

```
src/experiment_15_eyediff/
├── INDEX.md                  # This file
├── README.md                 # User guide and instructions
├── data_loader.py            # Time-series data loading utilities
├── equations.py              # Eye difference equation implementations
├── validation.py             # Statistical validation framework
└── testbench.py              # Main testbench runner

results/exp_15_eyediff/
├── EXPERIMENT_15_SUMMARY.md  # Comprehensive results summary
├── *.log                     # Detailed execution logs
└── *.png                     # Validation visualization plots
```

## Quick Start

```bash
# Activate virtual environment
source env/bin/activate

# Run testbench with baseline equation
python3 src/experiment_15_eyediff/testbench.py

# View results
cat results/exp_15_eyediff/EXPERIMENT_15_SUMMARY.md
```

## File Descriptions

### Source Files

**data_loader.py**
- `load_timeseries_data()`: Loads raw saccade sequences without aggregation
- `merge_mg_classes()`: Combines Definite MG and Probable MG into single class
- Handles UTF-16-LE encoding, missing data, and validation

**equations.py**
- `time_to_target_baseline()`: Baseline equation measuring time to reach target
- `apply_equation_to_sequence()`: Wrapper to apply any equation to a sequence
- `compute_eye_difference()`: Calculates absolute difference between L/R eyes
- Template for adding new equations

**validation.py**
- `validate_equation()`: Runs full statistical validation (Mann-Whitney, Wilcoxon tests)
- `plot_validation_results()`: Generates 4-panel visualization
  - Box plots (HC vs MG)
  - Histograms (distribution comparison)
  - Violin plots (shape analysis)
  - Summary statistics table
- Computes Cohen's d effect size
- 3-point validation scoring system

**testbench.py**
- Main entry point for running experiments
- Orchestrates data loading → equation application → validation → plotting
- Generates timestamped logs
- Runs both horizontal and vertical saccade analyses

### Result Files

**EXPERIMENT_15_SUMMARY.md**
- Complete analysis of baseline equation results
- Statistical interpretation
- Recommendations for future equations
- Performance metrics and limitations

**Log Files (*.log)**
- Timestamped execution records
- Sample sizes, statistics, p-values
- Failure tracking and debugging info

**Visualization Plots (*.png)**
- 4-panel validation figures
- Box plots, histograms, violin plots
- Statistical summary tables
- Saved at 150 DPI for publication quality

## Adding New Equations

### Step 1: Implement Equation
Edit `equations.py` and add:

```python
def my_new_equation(eye_position, target_position, **kwargs):
    """
    Compute custom metric for a single eye.

    Args:
        eye_position: 1D array of eye position (degrees)
        target_position: 1D array of target position (degrees)
        **kwargs: Additional parameters

    Returns:
        Dictionary with metrics, e.g.:
        {
            'my_metric': value,
            'additional_info': other_value
        }
    """
    # Your implementation here
    result = compute_something(eye_position, target_position)

    return {
        'my_metric': result,
        'other_metric': other_result
    }
```

### Step 2: Run Testbench
Edit `testbench.py` main() function:

```python
from equations import my_new_equation

results = run_testbench(
    equation_func=my_new_equation,
    equation_name="My_New_Equation",
    metric_key='my_metric',
    direction='horizontal',
    equation_kwargs={'param1': value1}
)
```

### Step 3: Analyze Results
- Check log files in `results/exp_15_eyediff/`
- Review validation plots
- Compare validation score (0-3) to baseline (2/3 for horizontal)

## Validation Criteria

**Ideal Equation Should:**
1. Mann-Whitney p < 0.05 (MG > HC) ✓
2. Wilcoxon HC p ≥ 0.05 (HC ≈ 0) ✓
3. Wilcoxon MG p < 0.05 (MG > 0) ✓
4. Cohen's d > 0.5 (medium to large effect)

**Score = 3/3 with large effect size = Publication-worthy candidate**

## Current Status

**Baseline Equation: Time to Target**
- Horizontal: 2/3 score, Cohen's d = 0.08 (negligible)
- Vertical: 1/3 score, Cohen's d = -0.01 (negligible)
- **Conclusion**: Weak but functional baseline, needs improvement

## Research Directions

See EXPERIMENT_15_SUMMARY.md for:
- Detailed statistical analysis
- Interpretation of baseline results
- 6 recommended equation categories to explore
- Methodological improvement suggestions

## Contact & Contribution

When developing new equations:
1. Maintain docstring standards
2. Handle NaN/inf values robustly
3. Log intermediate results for debugging
4. Run full testbench before committing
5. Update this INDEX.md with new findings

---

**Last Updated**: November 22, 2025
**Status**: Testbench operational and validated
**Next**: Implement novel discriminatory equations
