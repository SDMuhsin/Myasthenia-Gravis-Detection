# CYCLE 47: Recovery Dynamics (Reverse Fatigue)

## PHASE 2: HYPOTHESIS FORMULATION

### Outside-the-Box Insight

**ALL previous cycles** measured DEGRADATION (worsening over time). But MG is characterized by:
- Fatigue with sustained activity
- **RECOVERY with rest**

**What we haven't tested**: Do eyes show DIFFERENTIAL RECOVERY rates during brief rest periods between saccades?

### The Recovery Hypothesis

**Observation**: Saccade task has ~121 target jumps over ~25 seconds. Between consecutive upward saccades, there are brief rest periods (1-3 seconds) when target is stationary.

**Hypothesis**:
- **MG affected eye**: Slow recovery during rest → next saccade starts from MORE fatigued state
- **MG unaffected eye**: Normal recovery → each saccade starts fresh
- **HC both eyes**: Similar fast recovery

**Metric**: Recovery slope between consecutive saccades
```python
# For consecutive upward saccades i and i+1
error_saccade_i = settling_error[saccade_i]
error_saccade_i_plus_1 = settling_error[saccade_i+1]
rest_duration = time_between_saccades

# Recovery rate (negative = worsening, positive = improving)
recovery_rate = (error_i - error_i+1) / rest_duration

# Aggregate
early_recovery = mean(recovery_rate[:n/3])
late_recovery = mean(recovery_rate[-n/3:])

# MG hypothesis: affected eye shows DECLINING recovery capacity
recovery_deg = late_recovery - early_recovery  # Should be NEGATIVE for affected eye
```

**Asymmetry**: `|recovery_deg_L - recovery_deg_R|`

**Expected pattern**:
- MG affected eye: Recovery becomes LESS effective over session (more negative slope)
- MG unaffected eye: Maintains recovery capacity
- HC: Both eyes maintain recovery

**Why this is different**:
- Degradation measures ACCUMULATION of fatigue (error increase)
- Recovery measures CLEARANCE of fatigue (error improvement between saccades)
- These are OPPOSITE processes, may be dissociable

**Clinical precedent**: MG patients report needing longer rest periods to recover from fatigue. Brief rest may be sufficient for HC but inadequate for MG.

---

## PHASE 3: Adversarial Review

**Challenge 1**: "This is just degradation measured differently - if error increases over time, recovery must be decreasing"

**Response**: NOT necessarily. Two scenarios with same degradation, different recovery:
- Scenario A: Constant fatigue accumulation, no recovery (linear degradation)
- Scenario B: Intermittent fatigue spikes with partial recovery (non-linear, fluctuating)
- Recovery dynamics capture HOW fatigue progresses, not just endpoint change

**Challenge 2**: "Rest periods are too short (1-3s) for meaningful MG recovery"

**Response**: True for full recovery, but we're measuring DIFFERENTIAL recovery rates:
- Even if both eyes don't fully recover, affected eye may show SLOWER partial recovery
- Cumulative effect over 60 saccades may reveal pattern

**Challenge 3**: "Inter-saccade intervals are variable - confounds recovery rate"

**Response**: Valid. Mitigation:
- Normalize recovery by rest duration: `(error_i - error_i+1) / time_delta`
- Filter to similar-duration intervals (e.g., 1-2s rest periods only)

---

## PHASE 4: Empirical Pre-Implementation Analysis

**Required**:
1. Compute per-saccade error for consecutive upward saccades
2. Measure inter-saccade intervals
3. Compute recovery rates (normalized by rest duration)
4. Test if late recovery differs from early recovery
5. Compare asymmetry discrimination vs degradation

**GO Criteria**:
- Recovery degradation shows d≥0.45
- Low correlation with position degradation (r<0.7) - measures different aspect
- Can improve combined metric

**NO-GO Criteria**:
- d<0.30 (too weak)
- High correlation with degradation (r>0.9) - redundant
- Technical: Can't reliably measure consecutive saccades

---

**Rationale for trying**: This is genuinely DIFFERENT - measures fatigue clearance, not accumulation. MG pathophysiology involves BOTH impaired contraction AND impaired recovery at neuromuscular junction (acetylcholine receptor dysfunction affects BOTH).
