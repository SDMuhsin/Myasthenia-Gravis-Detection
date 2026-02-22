# Experiment 15: Eye Difference Analysis

## Objective
Develop publication-worthy equations to identify which eye is more affected in MG patients using saccade data.

## Approach
Since we don't have labels for which eye is affected, we use an unsupervised validation approach:
- **MG patients**: Should show significant differences between left and right eyes
- **Healthy controls**: Should show minimal/no differences between left and right eyes

## Structure

### Files
- `data_loader.py`: Loads time-series saccade data (no aggregation)
- `equations.py`: Contains equations for computing eye metrics
- `validation.py`: Statistical validation framework
- `testbench.py`: Main testbench script for rapid equation testing

### Usage

Run from repository root:
```bash
source env/bin/activate
python src/experiment_15_eyediff/testbench.py
```

### Adding New Equations

1. Add your equation function to `equations.py`:
```python
def my_new_equation(eye_position, target_position, **kwargs):
    """
    Compute a metric for a single eye.

    Returns:
        Dictionary with metrics (e.g., {'my_metric': value, ...})
    """
    # Your implementation here
    return {'my_metric': computed_value}
```

2. Run it through the testbench:
```python
from testbench import run_testbench
from equations import my_new_equation

results = run_testbench(
    equation_func=my_new_equation,
    equation_name="My_New_Equation",
    metric_key='my_metric',
    direction='horizontal'
)
```

### Validation Metrics

The testbench computes:
1. **Mann-Whitney U Test**: Tests if MG differences > HC differences
2. **Wilcoxon Test (HC)**: Tests if HC differences ≈ 0
3. **Wilcoxon Test (MG)**: Tests if MG differences > 0
4. **Cohen's d**: Effect size
5. **Validation Score**: 0-3 points based on passing above tests

### Baseline Equation

**Time to Target**: Measures how long it takes for the eye to reach and stabilize at the target position.
- Uses 2° threshold for "target reached"
- Requires sustained 20ms within threshold
- Computed separately for left and right eyes
- Difference = |left_time - right_time|

## Results

All results saved to `./results/exp_15_eyediff/`:
- Log files with detailed statistics
- Validation plots (box plots, histograms, violin plots)
- Summary tables
