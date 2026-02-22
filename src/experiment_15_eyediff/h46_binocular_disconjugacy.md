# CYCLE 46: Binocular Disconjugacy Analysis

## PHASE 2: HYPOTHESIS FORMULATION

### Context: Ceiling Confirmed (Cycles 38-45)

**8 consecutive cycles FAILED** to improve beyond gap~17:
- Cycle 38: H38b gap=16.8 (BEST)
- Cycles 39-41: Various approaches, gap=17.1-48.3 (worse)
- Cycle 42: FAT1 gap=17.0 (matches H38b)
- Cycle 43: Nonlinear features d=0.399 vs baseline d=0.487 (-18%)
- Cycle 44: Multi-temporal d=0.520 vs FAT1 d=0.540 (-3.8%)
- Cycle 45: Velocity d=-0.006 (ZERO discrimination)

**Root cause**: ~47% MG patients have asymmetry <0.45° overlapping HC natural variation.

### Final Unexplored Direction: Binocular Disconjugacy

**ALL previous metrics** measured PER-EYE performance separately, then computed asymmetry:
- score_L vs score_R
- Position error, velocity, latency - all measured independently per eye

**UNEXPLORED**: How well do the eyes COORDINATE with each other during the SAME saccade?

**Hypothesis**: MG-affected eye may show:
1. **Timing mismatch**: Saccade onset delayed relative to unaffected eye
2. **Trajectory mismatch**: Different amplitude/direction during same target jump
3. **Vergence error**: Horizontal misalignment (eyes don't converge on same point)

**Rationale**:
- Normal saccades are highly conjugate (both eyes move together within ~5ms)
- MG weakness in one eye may cause DISCONJUGACY:
  - Affected eye: slower onset, reduced amplitude, different trajectory
  - Unaffected eye: normal movement
  - Result: temporary binocular misalignment during saccade

**Clinical precedent**:
- Diplopia (double vision) is common MG symptom
- Caused by eye misalignment due to differential muscle weakness
- May manifest transiently during saccades even if static alignment normal

### APPROACH H46: Binocular Disconjugacy Index

For each upward saccade, measure per-saccade disconjugacy:

```python
# Saccade detection
target_jumps = where(diff(target) > 5.0)

for each saccade at idx:
    # Extract 30-sample window (0-250ms post-jump)
    L_trace = LV[idx:idx+30]
    R_trace = RV[idx:idx+30]

    # Onset timing difference
    L_onset = first sample where |diff(L_trace)| > threshold
    R_onset = first sample where |diff(R_trace)| > threshold
    onset_diff = |L_onset - R_onset|  # ms

    # Amplitude difference
    L_amplitude = max(L_trace) - L_trace[0]
    R_amplitude = max(R_trace) - R_trace[0]
    amp_diff = |L_amplitude - R_amplitude|  # degrees

    # Trajectory correlation (how similarly do they move?)
    trajectory_corr = correlation(L_trace, R_trace)
    disconjugacy = 1 - trajectory_corr  # 0=perfect conjugacy, 1=completely independent

# Aggregation
early_disconj = mean(disconjugacy[:n/3])
late_disconj = mean(disconjugacy[-n/3:])
disconj_deg = late_disconj - early_disconj  # Fatigue effect
```

Asymmetry:
```
H46_score = disconj_deg  # Already captures inter-eye coordination, no L vs R needed
```

**Expected**:
- MG: Disconjugacy increases over session (fatigue → worse coordination)
- HC: Disconjugacy remains low and stable (both eyes healthy)
- This is NOT an asymmetry metric - it's a COORDINATION metric

**Critical difference**:
- Previous metrics: |performance_L - performance_R|
- H46: How well L and R coordinate during simultaneous movements
- Measures INTERACTION, not individual performances

### PHASE 3: Adversarial Review

**Challenge 1**: "You're not measuring asymmetry anymore - this violates the core objective"
- **Response**: TRUE - this is a different class of metric. But the objective is "identify which eye is affected." Disconjugacy degradation may indicate ONE eye is weakening relative to the other without needing to separate L vs R scores.
- **Counterpoint**: But how do you determine WHICH eye is affected if disconjugacy is a single value?
- **Resolution**: You CAN'T with this approach alone. This would need to be combined with H38b (use H38b for direction, use H46 for magnitude).

**Challenge 2**: "Correlation between eye traces will be very high (~0.99) for conjugate saccades - measuring disconjugacy (1-r) will be noisy"
- **Response**: Valid concern. 1 - 0.99 = 0.01, small signal buried in noise.
- **Mitigation**: Use alternative disconjugacy measures:
  - Onset timing difference (ms): More robust than correlation
  - Amplitude difference (degrees): Directly meaningful
- **Alternative**: Combine multiple disconjugacy features

**Challenge 3**: "This is fundamentally different from asymmetry detection - it's a BILATERAL metric"
- **Response**: CORRECT. Disconjugacy measures total binocular dysfunction, not asymmetry.
- **Problem**: If MG affects BOTH eyes (bilateral), disconjugacy may be high but doesn't tell you which eye is worse.
- **Conclusion**: This approach may work for DETECTING MG (bilateral metric) but NOT for identifying which eye is affected (asymmetry).

**Challenge 4**: "Why would disconjugacy degradation work when velocity degradation failed (Cycle 45 d=-0.006)?"
- **Response**: Velocity measured PER-EYE independently. Disconjugacy measures INTER-EYE coordination.
  - Velocity degradation: Does each eye slow down?
  - Disconjugacy degradation: Do eyes become less synchronized?
- **Hypothesis**: MG may preserve individual eye velocity but disrupt binocular coordination.
- **Counterpoint**: But Insight #30 says "MG shows slightly BETTER correlation than HC (0.59 vs 0.56)" - correlation was tested in Cycle 7!
- **CRITICAL**: Cycle 7 tested POSITION correlation (how similarly eyes track target). This would test SACCADE correlation (how similarly eyes execute movements).

**Challenge 5**: "You already tested inter-eye correlation in Cycle 7 (H7) - it FAILED (d=-0.06, wrong direction)"
- **Full context**: Insight #29-30:
  - "Inter-eye correlation ~0.56 for both HC and MG (not near 1.0)"
  - "MG shows slightly BETTER correlation than HC (0.59 vs 0.56), opposite of hypothesis"
- **Response**: That was TRACKING correlation (continuous position over time). This would be SACCADE correlation (discrete movement events).
- **Counterpoint**: But why would saccade correlation differ from tracking correlation?
- **Resolution**: Need empirical test, but precedent (Cycle 7) suggests this will likely fail.

### PHASE 4: Empirical Pre-Implementation Analysis

**Required tests**:
1. Compute per-saccade trajectory correlation between L and R
2. Check if MG shows lower correlation than HC
3. Measure disconjugacy degradation (early vs late)
4. Compare to H38b performance

**GO Criteria**:
- MG disconjugacy significantly higher than HC (d≥0.40)
- Disconjugacy degradation shows fatigue effect (MG > HC)
- Can be combined with H38b to improve directional detection

**NO-GO Criteria**:
- MG and HC show similar disconjugacy (precedent: Cycle 7 correlation)
- d<0.30 (too weak)
- Technical failure: Saccade detection unreliable

### PHASE 5: Decision

**LIKELY NO-GO** based on:
1. **Precedent**: Cycle 7 already tested inter-eye correlation - FAILED (d=-0.06)
2. **Objective mismatch**: Disconjugacy doesn't identify WHICH eye is affected
3. **8 consecutive failures**: Strong pattern that ceiling is real

**If this cycle also fails**:
→ **ACCEPT CEILING** (gap~17, H38b final metric)
→ Shift to external validation and clinical deployment
→ Document comprehensive negative results for publication

---

**Recommendation**: Proceed with Phase 4 analysis, but expect NO-GO based on Cycle 7 precedent and consistent failure pattern of Cycles 38-45.
