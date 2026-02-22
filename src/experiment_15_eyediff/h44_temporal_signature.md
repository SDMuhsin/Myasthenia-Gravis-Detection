# CYCLE 44: Multi-Temporal Fatigue Signature

## PHASE 2: HYPOTHESIS FORMULATION

### Context from Previous Cycles

**Cycle 42 TTT/FAT Benchmark - Key Findings:**

| Metric | Description | Gap | Cohen's d |
|--------|-------------|-----|-----------|
| FAT1 | Error degradation (late 1/3 - early 1/3) | 17.0 | 0.540 |
| FAT3 | Error slope (linear regression) | 17.8 | 0.558 |
| H38b | Baseline (70% Deg + 30% MAD + 50% Lat) | 17.8 | 0.574 |

**Observation**: Both FAT1 and FAT3 measure fatigue but use DIFFERENT temporal aggregations:
- FAT1: Difference of means (late - early)
- FAT3: Linear regression slope

Both achieve gap≈17-18, nearly identical to H38b. This suggests:
1. Fatigue/degradation IS the key discriminator (confirmed)
2. The TEMPORAL PATTERN of fatigue may contain additional information
3. Current metrics use single temporal summaries (difference OR slope), missing richer dynamics

**Cycle 43 Result:**
- Nonlinear interactions (MAD×Deg, Deg², etc.) all failed
- Linear combinations of COMPONENT features are optimal
- But we haven't tried combining MULTIPLE TEMPORAL MEASUREMENTS of the same component

### Hypothesis: Multi-Temporal Fatigue Signature

**CORE IDEA**: Fatigue has multiple temporal characteristics:
1. **Magnitude**: How much performance degrades (late - early)
2. **Rate**: How fast it degrades (slope)
3. **Consistency**: How linear the degradation is (R² of slope fit)
4. **Variability**: How much performance fluctuates over time (std across windows)

**HYPOTHESIS**: Combining multiple temporal measurements of degradation captures a richer "fatigue signature" that:
- Magnitude alone (FAT1) may miss cases where degradation is small but VERY consistent
- Slope alone (FAT3) may miss cases where degradation is non-linear (accelerating fatigue)
- Combining them captures both absolute change AND temporal dynamics

**APPROACH H44: Temporal Signature Score**

For each eye:
1. Divide saccades into 5 equal windows (quintiles)
2. Compute mean error per window: [e1, e2, e3, e4, e5]
3. Extract temporal features:
   - `deg = e5 - e1` (magnitude of degradation)
   - `slope = linear_regression_slope([e1, e2, e3, e4, e5])` (rate of degradation)
   - `rsquared = R² of slope fit` (linearity of degradation)

4. Temporal signature score per eye:
   ```
   score_L = 0.50 * deg_L + 0.30 * slope_L + 0.20 * rsquared_L
   score_R = 0.50 * deg_R + 0.30 * slope_R + 0.20 * rsquared_R
   ```

5. Asymmetry:
   ```
   asymmetry = |score_L - score_R|
   ```

**Rationale for weights**:
- 50% deg: Magnitude is strongest discriminator (FAT1 d=0.540)
- 30% slope: Rate adds temporal dynamics (FAT3 d=0.558)
- 20% R²: Consistency differentiates true fatigue (linear) from noise (erratic)

### PHASE 3: Adversarial Review

**Challenge 1**: "This is just repackaging FAT1 and FAT3 - you already know they perform similarly to H38b"
- **Response**: True, but:
  - FAT1 and FAT3 are CORRELATED measurements (both capture degradation)
  - Combining them may provide ORTHOGONAL information only when they DIVERGE
  - Cases where deg is high but slope is low (non-linear fatigue) or vice versa
- **Counter**: Insight #135 says nonlinear features don't help. Is this just another repackaging?
- **Distinction**: This is NOT nonlinear (no MAD×Deg products), it's MULTI-TEMPORAL (multiple summaries of same degradation signal)

**Challenge 2**: "R² will be noisy with only 5 windows"
- **Response**: Valid concern. With n=5 points, R² may be unstable.
- **Mitigation**: Could use correlation coefficient r instead of R² (less sensitive to outliers)
- **Alternative**: Drop R² component and only combine deg + slope (simpler)

**Challenge 3**: "You're adding complexity without theoretical justification - Pitfall #5"
- **Response**: MG fatigue physiology suggests:
  - Some patients: rapid onset fatigue (high slope, moderate deg)
  - Others: gradual accumulation (high deg, moderate slope)
  - Others: intermittent (low R², erratic pattern)
- **Counter**: Do we have EVIDENCE these subtypes exist in the data?
- **Resolution**: Phase 4 analysis MUST show deg and slope have LOW correlation AND that combining them improves discrimination

**Challenge 4**: "H38b already combines degradation with MAD and latency - why is this better?"
- **Response**: H38b combines DIFFERENT COMPONENTS (variability + fatigue + speed). H44 combines DIFFERENT TEMPORAL VIEWS of the SAME component (degradation magnitude + rate + linearity).
- **Advantage**: If degradation is the strongest signal (confirmed by FAT1/FAT3), enriching the degradation measurement may yield more improvement than adding weaker components.

**Challenge 5**: "Insight #129 says temporal approaches failed - H40 MAX gave gap=17.1, no improvement"
- **Response**: H40 used MAX (extremal statistic). H44 uses COMBINATION of central tendency measures (mean difference + slope). Different aggregation approach.
- **Risk**: But still temporal windowing, which may suffer from same issues

### PHASE 4: Empirical Pre-Implementation Analysis Required

**Analysis 1 - Feature Correlation**:
```python
# Compute degradation magnitude and slope for all sequences
deg = late_error - early_error  # FAT1
slope = linregress(window_errors).slope  # FAT3

# Check correlation
r_hc = correlation(deg_HC, slope_HC)
r_mg = correlation(deg_MG, slope_MG)

# GO criteria: r < 0.7 (orthogonal)
# NO-GO: r > 0.9 (redundant)
```

**Analysis 2 - Individual Component Discrimination**:
```python
# Asymmetry per component
deg_asym = |deg_L - deg_R|
slope_asym = |slope_L - slope_R|

# Cohen's d for each
d_deg = cohens_d(deg_asym_MG, deg_asym_HC)
d_slope = cohens_d(slope_asym_MG, slope_asym_HC)

# GO criteria: d_slope ≥ 0.45 (comparable to deg)
# NO-GO: d_slope < 0.35 (too weak to contribute)
```

**Analysis 3 - Combined Performance**:
```python
# Simple combination (equal weight)
combined = 0.5 * deg_asym + 0.5 * slope_asym
d_combined = cohens_d(combined_MG, combined_HC)

# GO criteria: d_combined ≥ 0.55 (improvement over FAT1 d=0.540)
# NO-GO: d_combined < 0.50 (worse than components)
```

### PHASE 5: Go/No-Go Decision

**GO if:**
1. deg and slope correlation r < 0.7 (orthogonal measurements)
2. slope_asym achieves d ≥ 0.45 (meaningful discriminator)
3. Combined asymmetry d ≥ 0.55 (improvement over single-temporal metrics)
4. Gap ≤ 17.0 (matches or beats FAT1)

**NO-GO if:**
1. deg and slope correlation r > 0.9 (redundant)
2. Combined performance d < 0.50 (worse than components alone)
3. Gap > 20 (substantially worse than H38b)

**If NO-GO**: Accept that gap~17 is realistic ceiling, document findings, and shift focus to:
- Clinical interpretation of results
- Validation on external datasets
- Or fundamentally different approach (smooth pursuit, fixation stability, pupillometry)

---

**Next Step**: Implement Phase 4 empirical pre-implementation analysis.
