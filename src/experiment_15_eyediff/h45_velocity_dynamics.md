# CYCLE 45: Saccadic Velocity Dynamics

## PHASE 2: HYPOTHESIS FORMULATION

### Evidence from Previous Cycles

**Cycles 38-44 Summary:**
- H38b achieves gap=16.8 (MG 30.9% neither, HC 55.9% neither)
- Cycles 39-44: 6 different approaches all FAIL to improve:
  - Z-score normalization (gap=48.3, worse)
  - Temporal slope/max (gap=17.1-21.3, no improvement)
  - Ensemble voting (gap=21.3, worse)
  - Nonlinear features (d=0.399 vs baseline d=0.487, -18%)
  - Multi-temporal views (d=0.520 vs FAT1 d=0.540, -3.8%)

**Ceiling Analysis (Insight #133)**:
- ~47% of MG patients have asymmetry <0.45° (overlaps HC natural variation)
- Target MG 10-20% neither appears UNACHIEVABLE with current approach
- Root cause: Subtle/bilateral MG presentations indistinguishable from HC

**Critical Gap in Current Approach**:
ALL tested metrics (H30-H44, TTT1-5, FAT1-5) measure **STATIC PERFORMANCE**:
- Positional error (where eye settles)
- Latency (when eye reaches target)
- Degradation (how error changes over time)

But MG is a neuromuscular disorder affecting MUSCLE CONTRACTION DYNAMICS. None of our metrics measure the MOVEMENT ITSELF - the saccade velocity profile.

### Hypothesis: Saccadic Velocity Degradation Asymmetry

**RATIONALE**:
1. **Saccadic velocity reflects muscle strength**: Main sequence relationship (peak velocity ∝ amplitude) is fundamental to eye movement physiology
2. **MG affects muscle power**: Weak muscles generate slower, more fatigable contractions
3. **Velocity degradation may be more sensitive**: Even if eye eventually reaches target (normal settling error), it may get there SLOWER in affected eye
4. **Orthogonal to position metrics**: Velocity dynamics are kinematic, position metrics are static endpoint measures

**APPROACH H45**:

For each upward saccade, detect peak velocity:
```
velocity = |diff(eye_position)|  # Instantaneous velocity
peak_vel = max(velocity) during saccade
```

Extract per-eye metrics:
1. **Early peak velocity**: Mean of fastest saccades in first 1/3
2. **Late peak velocity**: Mean of fastest saccades in last 1/3
3. **Velocity degradation**: `VelDeg = late_peak_vel - early_peak_vel`

Asymmetry score:
```
vel_deg_asym = |VelDeg_L - VelDeg_R|
```

**Expected outcome**:
- MG affected eye: Peak velocity DECREASES over session (muscle fatigue)
- MG unaffected eye: Stable or minimal decrease
- HC both eyes: Similar minimal decrease (no pathological fatigue)
- Asymmetry: MG shows larger differential velocity degradation

### PHASE 3: Adversarial Review

**Challenge 1**: "You tested TTT4 (peak velocity latency) - it FAILED with gap=27.8, d=0.081"
- **Response**: TTT4 measured LATENCY to peak velocity (time to reach max speed). H45 measures MAGNITUDE of peak velocity and its degradation. Different phenomenon:
  - TTT4: When does peak velocity occur? (timing)
  - H45: How fast is peak velocity? (magnitude) and does it decline? (fatigue)
- **Distinction**: Timing vs magnitude of velocity are orthogonal

**Challenge 2**: "Insight #34 says individual saccade features show d<0.26"
- **Full insight**: "Per-saccade dynamics analysis (28k+ saccades) reveals ALL individual saccade features (duration, amplitude, velocity, settling, overshoot) show negligible discrimination (d<0.26)"
- **Response**: That analysis measured AVERAGE saccade velocity across entire session. H45 measures DEGRADATION (early vs late) in velocity, not static average.
- **Counterpoint**: But if saccade velocity shows d<0.26, why would velocity degradation be better?
- **Resolution**: Need Phase 4 analysis to check if velocity degradation asymmetry is stronger than velocity asymmetry

**Challenge 3**: "This is just applying the degradation concept to a new component - not fundamentally different"
- **Response**: TRUE, but that's actually the LESSON from Cycles 43-44:
  - Cycle 43: Nonlinear combinations of existing components DON'T help
  - Cycle 44: Multi-temporal views of same component DON'T help
  - But FAT1/FAT3/H38b all show degradation IS the key discriminator
- **Approach**: Instead of trying to engineer better combinations, find NEW COMPONENTS where degradation metric can be applied
- **Hypothesis**: Velocity degradation may capture muscle fatigue that position degradation misses (kinematic vs static)

**Challenge 4**: "Velocity is noisy - sampling rate 120Hz, saccades are 30-60ms → only 4-7 samples per saccade"
- **Response**: Valid concern. Peak velocity estimation may be unstable with sparse sampling.
- **Mitigation**: Use median of top 3 fastest saccades instead of single fastest (robust estimation)
- **Alternative**: Smooth velocity with 3-point moving average before finding peak

**Challenge 5**: "Why would velocity degrade if position error doesn't show strong enough signal?"
- **Response**: Position error is endpoint metric - eye may reach target but use less forceful movement. Velocity captures process, not outcome.
- **Clinical precedent**: MG patients can eventually complete movements (normal ROM) but with reduced force/speed
- **Counterpoint**: But we're measuring ASYMMETRY - if both eyes slow down equally (bilateral), asymmetry won't increase
- **Resolution**: Need empirical test - do MG patients show differential velocity decline between eyes?

### PHASE 4: Empirical Pre-Implementation Analysis Required

**Analysis 1**: Compute per-saccade peak velocity for all sequences
- Extract early vs late peak velocity per eye
- Measure velocity degradation: VelDeg = late - early

**Analysis 2**: Check correlation with existing degradation metric
```python
position_deg = late_position_error - early_position_error
velocity_deg = late_peak_vel - early_peak_vel
r = correlation(position_deg, velocity_deg)

# GO if r<0.7 (orthogonal)
# NO-GO if r>0.9 (redundant with existing metrics)
```

**Analysis 3**: Individual discrimination
```python
vel_deg_asym = |VelDeg_L - VelDeg_R|
d = cohens_d(vel_deg_asym_MG, vel_deg_asym_HC)

# GO if d ≥ 0.50
# NO-GO if d < 0.40
```

### PHASE 5: Go/No-Go Decision

**GO Criteria** (need ≥2/3):
1. Velocity degradation uncorrelated with position degradation (r<0.7)
2. Velocity degradation asymmetry shows d≥0.50
3. Can reliably detect saccades and measure peak velocity (≥70% success rate)

**NO-GO Criteria**:
1. Velocity degradation highly correlated with position degradation (r>0.9) - redundant
2. Effect size d<0.40 - too weak to contribute
3. Technical failure: Can't robustly detect velocity peaks due to noise

**If NO-GO**:
Accept that gap~17 is realistic ceiling. Options:
1. Document findings: "Analytical metrics achieve gap=17.8 (H38b), further improvement requires supervised ML or additional diagnostic modalities"
2. Shift to validation: Test H38b on external dataset, compute clinical metrics (sensitivity/specificity at various thresholds)
3. Abandon eye difference detection, explore alternative approaches (smooth pursuit, pupillometry, fixation stability)

---

**Next Step**: Implement Phase 4 empirical analysis
