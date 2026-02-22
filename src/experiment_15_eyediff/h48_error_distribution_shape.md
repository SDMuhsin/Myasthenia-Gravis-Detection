# CYCLE 48: Error Distribution Shape Analysis

## PHASE 2: HYPOTHESIS FORMULATION

### Outside-the-Box Insight #2

**ALL previous cycles** used summary statistics (mean, median, std, MAD) of error distributions. These capture LOCATION and SCALE but ignore SHAPE.

**What we haven't tested**: Do MG eyes show different ERROR DISTRIBUTION SHAPES?

### The Distribution Shape Hypothesis

**Observation**: We compute tracking error for each eye. Standard approach:
- `mean(|error|)` - central tendency
- `std(error)` or `MAD(error)` - dispersion

But distributions have additional properties:
- **Skewness**: Asymmetry of distribution (long tail on one side)
- **Kurtosis**: "Tailedness" (outlier frequency)
- **Bimodality**: Multiple modes (two performance states)

**Hypothesis**:
- **MG affected eye**: May show BIMODAL distribution (good saccades vs fatigued saccades)
  - OR high positive skewness (occasional very poor saccades)
  - OR high kurtosis (frequent outliers)
- **MG unaffected eye / HC**: More GAUSSIAN (normal distribution)

**Why this could work**:
- Intermittent fatigue: MG doesn't affect every saccade equally
- Some saccades executed well (muscle not fatigued), others poorly (post-fatigue)
- Creates mixed distribution, detectable via shape metrics
- Standard summary stats (mean/MAD) AVERAGE over both modes, losing information

**Clinical precedent**: MG symptoms fluctuate - patients report "good moments" and "bad moments" even within short periods. This should manifest as distribution shape changes.

### APPROACH H48

```python
For each eye:
    # Extract all per-saccade errors (60+ saccades)
    errors = [error_1, error_2, ..., error_60]

    # Distribution shape metrics
    skewness = scipy.stats.skew(errors)  # 0=symmetric, >0=right tail
    kurtosis = scipy.stats.kurtosis(errors)  # 0=normal, >0=heavy tails
    bimodality_coeff = (skew^2 + 1) / (kurtosis + 3)  # >0.555 suggests bimodal

    # Degradation of shape over time
    early_skew = skew(errors[:n/3])
    late_skew = skew(errors[-n/3:])
    skew_deg = late_skew - early_skew

# Asymmetry
skew_asym = |skewness_L - skewness_R|
kurt_asym = |kurtosis_L - kurtosis_R|
bimod_asym = |bimodality_L - bimodality_R|
```

**Expected pattern**:
- MG affected eye: High skewness/kurtosis/bimodality (mixed performance)
- MG unaffected eye: Low (more Gaussian)
- Asymmetry captures differential distribution shapes

---

## PHASE 3: Adversarial Review

**Challenge 1**: "You need large sample sizes (n>100) for reliable skewness/kurtosis estimation - 60 saccades not enough"

**Response**: Valid statistical concern. Mitigation:
- Use robust estimators (quantile-based skewness: (Q3-Q2)-(Q2-Q1))
- Accept higher variance, look for LARGE effect sizes (d>0.6)
- Bootstrap confidence intervals

**Challenge 2**: "Skewness/kurtosis are sensitive to outliers - measurement noise will dominate"

**Response**: TRUE for raw kurtosis. Use:
- Robust alternatives: Median-based quartile coefficient of kurtosis
- Outlier filtering (remove errors >3 MAD from median)
- Focus on skewness (less outlier-sensitive than kurtosis)

**Challenge 3**: "Why would distribution shape differ if mean degradation already captures fatigue?"

**Response**: Mean degradation captures AVERAGE worsening. Shape captures PATTERN:
- Scenario A: Gradual linear decline (normal distribution throughout)
- Scenario B: Intermittent failures (bimodal: cluster of good + cluster of bad)
- Same mean degradation, different shapes

**Challenge 4**: "Insight #34 already tested per-saccade features - all showed d<0.26"

**Response**: That tested MAGNITUDE of individual saccade metrics (duration, amplitude, velocity). This tests DISTRIBUTION PROPERTIES of error across saccades. Different:
- Insight #34: max(saccade velocities) across all saccades
- H48: skewness(saccade errors) - shape of the error distribution

---

## PHASE 4: Empirical Pre-Implementation Analysis

**Required**:
1. Extract per-saccade errors (all 60+ saccades per eye)
2. Compute skewness, kurtosis, bimodality coefficient
3. Test if MG shows different distribution shapes than HC
4. Measure asymmetry discrimination
5. Compare to degradation-based metrics

**GO Criteria**:
- Shape asymmetry d≥0.45
- Low correlation with degradation (r<0.7)
- Can improve combined metric

**NO-GO Criteria**:
- d<0.30 (too weak)
- High variance (unreliable estimation with n=60)
- No difference in shape between MG and HC

---

**Rationale**: This is GENUINELY different - we've never analyzed distribution SHAPE, only location/scale. If MG shows intermittent fatigue (bimodal/skewed), standard summary stats miss this. Worth one empirical test.
