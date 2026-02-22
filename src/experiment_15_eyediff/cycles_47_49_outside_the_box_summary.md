# CYCLES 47-49: OUTSIDE-THE-BOX RESEARCH SUMMARY

## Overview

After 10 consecutive cycles (38-46) showing NO improvement over the established ceiling (gap~17), three fundamentally creative approaches were tested to break through the performance barrier.

**Key Finding**: Only 1 of 3 approaches (H47: Recovery Dynamics) showed ANY promise, and even that is MARGINAL. The performance ceiling at gap~17 appears to be genuine.

---

## CYCLE 47: Recovery Dynamics [MARGINAL SUCCESS]

### Hypothesis
Measure fatigue CLEARANCE (recovery between saccades) not just ACCUMULATION (degradation over session).

**Rationale**: MG pathophysiology involves BOTH impaired muscle contraction AND impaired recovery at the neuromuscular junction. All previous cycles measured degradation; recovery dynamics may be orthogonal.

### Approach
```python
# For consecutive saccades i and i+1:
error_improvement = saccade_errors[i] - saccade_errors[i+1]
rest_duration = saccade_times[i+1] - saccade_times[i]
recovery_rate = error_improvement / rest_duration  # degrees/second

# Measure decline in recovery capacity over session:
recovery_degradation = mean(late_recovery) - mean(early_recovery)
```

### Results

**Recovery Degradation Characteristics**:
```
  HC Left:   0.0355 ± 0.0436 °/s
  HC Right:  0.0336 ± 0.0454 °/s
  MG Left:   0.0542 ± 0.0668 °/s
  MG Right:  0.0435 ± 0.0537 °/s
```

**Asymmetry Discrimination**:
```
Recovery Degradation Asymmetry:
  HC: 0.0355 ± 0.0436 °/s
  MG: 0.0542 ± 0.0668 °/s
  Cohen's d = 0.317 (MARGINAL)

Position Degradation Asymmetry:
  HC: 0.6829 ± 0.6445°
  MG: 1.2870 ± 1.3040°
  Cohen's d = 0.549 (baseline)
```

**Orthogonality Test**: ✓ PASS
```
Correlation between position and recovery degradation:
  HC: r=0.076 (p=0.134)
  MG: r=0.065 (p=0.189)

|r|<0.7 → Recovery is ORTHOGONAL to position degradation
```

**Combined Performance**:
```
Metric                           Cohen_d    vs PosOnly
Position only (baseline)         0.549      baseline
Recovery only                    0.317      -42.3%
Equal (50%/50%)                  0.540      -1.7%
Pos-heavy (70%/30%)              0.573      +4.3%  ← BEST
Rec-heavy (30%/70%)              0.465      -15.3%
```

### GO/NO-GO Decision: **GO** (2/3 criteria passed)

**Criteria**:
1. Orthogonality (|r|<0.7): ✓ PASS (r=0.076 HC, r=0.065 MG)
2. Recovery discrimination (d≥0.45): ✗ FAIL (d=0.317, marginal)
3. Combined improvement: ✓ PASS (best d=0.573, +4.3% improvement)

**Verdict**: Recovery dynamics are GENUINELY ORTHOGONAL to position degradation and provide small improvement when combined. This is the FIRST new direction in 10 cycles (38-46) to show ANY improvement.

**Clinical Interpretation**: MG affects both muscle contraction (position degradation) AND recovery capacity (recovery degradation). Measuring both captures complementary aspects of neuromuscular junction pathology.

---

## CYCLE 48: Distribution Shape [CATASTROPHIC FAIL]

### Hypothesis
Measure error distribution SHAPE (skewness, kurtosis, bimodality) to detect intermittent fatigue patterns creating bimodal/skewed distributions.

**Rationale**: Standard metrics (mean, MAD, degradation) measure location and scale. If MG shows bimodal behavior (good saccades when not fatigued + bad saccades during fatigue), shape metrics may detect this mixture.

### Approach
```python
# For per-saccade errors:
errors = [error_1, error_2, ..., error_60]

# Statistical shape metrics:
skewness = scipy.stats.skew(errors)  # >0 = right tail
kurtosis = scipy.stats.kurtosis(errors)  # >0 = heavy tails
bimodality_coeff = (skewness**2 + 1) / (kurtosis + 3)  # >0.555 suggests bimodal

# Asymmetry:
skew_asym = |skewness_L - skewness_R|
kurt_asym = |kurtosis_L - kurtosis_R|
```

### Results

**Distribution Shape Characteristics**:
```
Skewness (>0 = right tail):
  HC Left:   1.171 ± 1.026
  HC Right:  1.154 ± 1.030
  MG Left:   1.231 ± 0.902
  MG Right:  1.159 ± 0.947

Kurtosis (>0 = heavy tails):
  HC Left:   2.253 ± 5.131
  HC Right:  2.205 ± 5.226
  MG Left:   2.556 ± 4.344
  MG Right:  2.212 ± 4.434
```

**Asymmetry Discrimination**:
```
Skewness Asymmetry:
  HC: 0.2512 ± 0.4712
  MG: 0.2486 ± 0.3041
  Cohen's d = -0.007 (ZERO discrimination)

Kurtosis Asymmetry:
  HC: 0.9717 ± 3.3539
  MG: 0.6443 ± 1.3201
  Cohen's d = -0.141 (WRONG DIRECTION)

Best shape metric: Kurtosis with d=-0.141
  vs Degradation: -125.7% (WORSE)
```

### GO/NO-GO Decision: **NO-GO** (0/3 criteria passed)

**Criteria**:
1. Shape discrimination (d≥0.45): ✗ FAIL (d=-0.141)
2. Improvement over degradation: ✗ FAIL (-125.7%)

**Verdict**: Distribution shape provides ZERO discriminative signal. Both MG and HC show similar skewed distributions with heavy tails. MG does NOT manifest as bimodal or differently-shaped error distributions.

**Interpretation**: Intermittent fatigue does NOT create detectable bimodality in saccade error distributions. Standard summary statistics (mean, degradation) already capture the relevant signal. Shape analysis adds no value.

---

## CYCLE 49: Consistency Clustering [CATASTROPHIC FAIL]

### Hypothesis
Use k-means clustering to detect whether saccades form distinct quality clusters. MG may show bimodal performance (good saccades + bad saccades), while HC shows unimodal (all similar).

**Rationale**: Standard metrics average over distributions, losing information about cluster structure. If MG shows mixed populations (good/bad saccades), clustering metrics like silhouette score may detect this.

### Approach
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Per-saccade errors for each eye:
errors = [error_1, error_2, ..., error_60].reshape(-1, 1)

# K-means clustering (k=2):
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(errors)

# Silhouette score: How well-separated are clusters?
# +1 = perfect clustering (tight groups, well-separated)
# 0 = overlapping clusters
# -1 = wrong clustering
silhouette = silhouette_score(errors, labels)

# Asymmetry:
silh_asym = |silhouette_L - silhouette_R|
```

### Results

**Clustering Characteristics**:
```
Silhouette Score (+1=perfect clustering, 0=overlapping):
  HC Left:   0.648 ± 0.087
  HC Right:  0.647 ± 0.082
  MG Left:   0.643 ± 0.082
  MG Right:  0.645 ± 0.086
```

**Interpretation**: ALL groups show similar high silhouette scores (~0.64-0.65). Both HC and MG show natural bimodal clustering (likely early vs late saccades due to general fatigue), but there's NO DIFFERENCE between groups.

**Asymmetry Discrimination**:
```
Silhouette Asymmetry:
  HC: 0.0312 ± 0.0436
  MG: 0.0314 ± 0.0350
  Cohen's d = 0.007 (ZERO discrimination)

Degradation Asymmetry (baseline):
  HC: 0.6829 ± 0.6445°
  MG: 1.2870 ± 1.3040°
  Cohen's d = 0.549

Comparison: -98.8% (WORSE)
```

**Orthogonality Test**: ✓ PASS
```
Correlation between degradation and silhouette:
  HC: r=-0.077 (p=0.0843)
  MG: r=-0.005 (p=0.8974)
```

### GO/NO-GO Decision: **NO-GO** (1/3 criteria passed)

**Criteria**:
1. Orthogonality (|r|<0.7): ✓ PASS
2. Silhouette discrimination (d≥0.40): ✗ FAIL (d=0.007)
3. Improvement over degradation: ✗ FAIL (-98.8%)

**Verdict**: Saccade clustering provides ZERO discrimination. MG does NOT show different cluster separation patterns compared to HC.

**Interpretation**: While both MG and HC show bimodal saccade patterns (silhouette ~0.64), this is likely due to general fatigue effects (early vs late saccades) affecting EVERYONE, not MG-specific intermittent fatigue. Cluster structure contains no MG-specific information.

---

## COMPREHENSIVE ANALYSIS

### Summary Table

| Cycle | Hypothesis | Approach | Cohen's d | Orthogonal | Decision | Improvement |
|-------|-----------|----------|-----------|------------|----------|-------------|
| 47 | Recovery Dynamics | Fatigue clearance rate | 0.317 (marginal) | ✓ YES (r=0.076) | GO | +4.3% combined |
| 48 | Distribution Shape | Skewness, kurtosis | -0.007 (zero) | N/A | NO-GO | -125.7% |
| 49 | Clustering | Silhouette score | 0.007 (zero) | ✓ YES (r=-0.077) | NO-GO | -98.8% |

### Key Insights

**1. Performance Ceiling is REAL**

After 13 consecutive cycles (38-49) attempting to break through the gap~17 barrier:
- 10 cycles (38-46): TTT/FAT variations, all failed (d<0.26)
- 3 cycles (47-49): Outside-the-box approaches, 2/3 catastrophic failures
- ONLY H47 (recovery dynamics) showed marginal improvement (+4.3%)

**Conclusion**: The gap~17 ceiling appears to be genuine, representing the discriminative limit of saccade asymmetry features for this dataset.

**2. Orthogonality Does NOT Guarantee Utility**

Two approaches (H47, H49) showed excellent orthogonality to degradation:
- H47 recovery: r=0.076 → marginal discrimination (d=0.317)
- H49 clustering: r=-0.077 → ZERO discrimination (d=0.007)

**Lesson**: Measuring a genuinely different aspect of the data does NOT guarantee that aspect is discriminative for MG.

**3. MG Does NOT Show Expected Statistical Signatures**

Tested clinical hypotheses:
- ✗ Bimodal error distributions (H48 shape analysis)
- ✗ Distinct saccade quality clusters (H49 clustering)
- ~ Declining recovery capacity (H47 recovery - marginal only)

**Conclusion**: MG-related fatigue in this dataset does NOT manifest as easily-detectable statistical patterns beyond simple degradation metrics.

**4. Recovery Dynamics: Only Marginal Success**

H47 is the ONLY outside-the-box approach to show ANY improvement:
- d=0.317 alone (too weak)
- Combined 70% position + 30% recovery → d=0.573 (+4.3%)
- Genuinely orthogonal (r=0.076)

**Clinical significance**: MG affects BOTH contraction AND recovery, but recovery effects are subtle and provide only marginal additional information.

---

## FINAL ASSESSMENT

### What Worked
- **H47 Recovery Dynamics**: Marginal success, +4.3% improvement when combined
  - Genuinely orthogonal to existing metrics
  - Captures complementary aspect of neuromuscular pathology
  - Worth including in final feature set

### What Failed
- **H48 Distribution Shape**: Catastrophic failure (d=-0.007)
  - No bimodal or skewed patterns specific to MG
  - Shape metrics provide zero information
  - Reject permanently

- **H49 Consistency Clustering**: Catastrophic failure (d=0.007)
  - Both MG and HC show similar cluster structure
  - Bimodality likely reflects general fatigue, not MG-specific
  - Reject permanently

### Research Strategy Assessment

**13 consecutive cycles** (38-49) attempted to break through gap~17:
- Traditional variations (38-46): All failed
- Outside-the-box creative approaches (47-49): 2/3 catastrophic failures, 1/3 marginal

**Conclusion**: The performance ceiling at gap~17 is ROBUST against:
1. Feature engineering variations
2. Temporal window adjustments
3. Statistical transformation approaches
4. Completely different measurement angles

This strongly suggests gap~17 represents the **true discriminative limit** of saccade asymmetry features for this MG detection task with this dataset.

### Recommendation

**STOP pure feature engineering research** on saccade asymmetry metrics. After 49 cycles:
- Best single metric: H24 TTT-based (gap=17.5)
- Best combination: H24 + marginal improvements (gap~17)
- Ceiling: gap~17 appears genuine

**Next directions**:
1. Accept gap~17 as realistic ceiling for this feature class
2. Explore fundamentally different data modalities (if available)
3. Focus on robust implementation of H24 + H47 recovery
4. OR investigate alternative classification approaches (ensemble methods, etc.)

---

## Technical Implementation Notes

### H47 Recovery Dynamics - Implementation Details

```python
def compute_recovery_degradation(eye_pos, target_pos, sample_rate_hz=120):
    """
    Extract recovery dynamics alongside position degradation.

    Returns:
        pos_deg: Position degradation (late_error - early_error)
        recovery_deg: Recovery degradation (late_recovery - early_recovery)
    """
    # Detect saccades
    target_diff = np.diff(target_pos)
    up_indices = np.where(target_diff > 5.0)[0] + 1

    # Extract per-saccade errors and times
    saccade_errors = []
    saccade_times = []
    for idx in up_indices:
        start = idx + 20
        end = min(idx + 50, len(eye_pos))
        error = np.mean(np.abs(eye_pos[start:end] - target_pos[start:end]))
        time_sec = idx / sample_rate_hz
        saccade_errors.append(error)
        saccade_times.append(time_sec)

    # Recovery dynamics: error improvement per second
    recovery_rates = []
    for i in range(len(saccade_errors) - 1):
        error_improvement = saccade_errors[i] - saccade_errors[i+1]
        rest_duration = saccade_times[i+1] - saccade_times[i]
        if 0.1 < rest_duration < 5.0:  # Filter unreasonable intervals
            recovery_rate = error_improvement / rest_duration
            recovery_rates.append(recovery_rate)

    # Degradation metrics
    n = len(recovery_rates)
    third = max(2, n // 3)
    early_recovery = np.mean(recovery_rates[:third])
    late_recovery = np.mean(recovery_rates[-third:])
    recovery_deg = late_recovery - early_recovery

    n_err = len(saccade_errors)
    third_err = max(2, n_err // 3)
    early_err = np.mean(saccade_errors[:third_err])
    late_err = np.mean(saccade_errors[-third_err:])
    pos_deg = late_err - early_err

    return pos_deg, recovery_deg
```

### Recommended Feature Combination

```python
# Normalize
pos_asym_norm = np.abs(pos_deg_L - pos_deg_R) / std_pos
rec_asym_norm = np.abs(rec_deg_L - rec_deg_R) / std_rec

# Optimal combination (70% position, 30% recovery)
combined_score = 0.7 * pos_asym_norm + 0.3 * rec_asym_norm
```

**Expected performance**: Cohen's d = 0.573 (vs 0.549 position-only, +4.3%)

---

**Date**: 2025-11-23
**Cycles**: 47-49
**Status**: Research complete - recommend ending feature engineering phase
