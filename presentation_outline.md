# Eye-Based Disease Differentiation: Saccade Time Series Analysis
## Presentation for Medical Expert Consultation
### Sayed Muhsin, PhD Student, University of Saskatchewan

---

## Slide 1: Introduction & Objectives
- **Goal**: Differentiate eye-based diseases (HC vs MG vs CNPs) using saccade time series data
- **Diseases studied**: 
  - Healthy Controls (HC): 510 sequences, 85 patients
  - Myasthenia Gravis (MG): 431 sequences, 72 patients  
  - Cranial Nerve Palsies: CNP3 (245 seq), CNP4 (186 seq), CNP6 (293 seq)
  - Thyroid-Associated Orbitopathy (TAO): 12 sequences (limited data)
- **Data**: 1,677 saccade sequences from 273 patients
- **Features**: 14 channels including eye positions, velocities, targets, and error signals

---

## Slide 2: Key Feature Analysis Findings - Eye Position Differences
### Significant Horizontal Eye Position Variations
- **Mean horizontal positions differ significantly** between HC and pathological groups
- **HC vs MG**: p < 0.0001 (highly significant)
- **HC vs CNP4**: p = 0.0016 (significant) 
- **HC vs CNP6**: p = 0.0010 (significant)
- **Clinical Implication**: Pathological groups show systematic horizontal gaze biases

![Feature Importance Analysis](results/EXP_7_FEATURE_ANALYSIS_HC_vs_MG_feature_importance.png)

---

## Slide 3: Key Feature Analysis Findings - Movement Variability
### Reduced Eye Movement Velocity Variability in Disease
- **Most pathological groups show reduced velocity variability** compared to HC
- **Left Horizontal Velocity (LH_Vel_std)**:
  - HC vs CNP3: p < 0.0001
  - HC vs CNP4: p = 0.0008  
  - HC vs CNP6: p < 0.0001
- **Right Horizontal Velocity (RH_Vel_std)**:
  - HC vs CNP3: p < 0.0001
  - HC vs CNP4: p = 0.0008
  - HC vs CNP6: p = 0.0025

**Clinical Significance**: Disease states may involve more constrained or less variable saccadic movements

![HC vs CNP Feature Analysis](results/EXP_7_FEATURE_ANALYSIS_HC_vs_CNP_feature_importance.png)

---

## Slide 4: Key Feature Analysis Findings - Gaze Error Patterns  
### Higher Gaze Error Variability in Pathological Groups
- **Error signal variability within trials is elevated** in MG and CNP groups
- **Horizontal Error Variability (ErrorH_L_iqr & ErrorH_R_iqr)**:
  - HC vs MG: p < 0.0001
  - HC vs CNP3: p < 0.0001
  - HC vs CNP4: p < 0.0001
  - HC vs CNP6: p < 0.0001
- **Vertical Error Variability** shows similar patterns

**Clinical Interpretation**: Pathological conditions lead to less precise gaze control

---

## Slide 5: Key Feature Analysis Findings - Eye Coordination
### Potential Dysconjugacy in Disease Groups
- **Reduced correlation between left and right eye movements** suggests dysconjugacy
- **Most pronounced in TAO patients** (though sample size is limited: n=12)
- **MG patients also show coordination deficits**
- **CNP groups show variable patterns** depending on affected nerve

**Medical Question**: How does dysconjugacy manifest clinically in your experience with these conditions?

---

## Slide 6: Experiment 1 - Statistical Models Baseline
### Establishing Performance Baseline with Traditional Features
- **Approach**: Statistical aggregation of saccade features (mean, std, median, IQR)
- **Model**: Linear Discriminant Analysis (LDA) with 103 aggregated features
- **Results**: **45% accuracy** in 5-class classification
- **Key Finding**: Statistical summaries capture meaningful disease signatures
- **Limitation**: Insufficient for clinical deployment but establishes strong baseline

**Clinical Relevance**: Traditional statistical measures of eye movements contain diagnostic information

---

## Slide 7: Experiments 2-3 - Deep Learning Approaches
### Neural Networks on Raw Time Series Data
**Experiment 2 - RNN Model**:
- **Approach**: Bi-GRU with attention on raw 14-channel time series
- **Results**: **34% accuracy** (worse than statistical baseline)
- **Finding**: Raw temporal patterns less informative than statistical summaries

**Experiment 3 - Hybrid CNN-RNN**:
- **Approach**: Combined convolutional and recurrent architecture
- **Results**: **38% accuracy** (slight improvement but still below baseline)
- **Conclusion**: Complex deep learning models did not outperform simpler statistical approach

![RNN Results](results/EXP_2/EXP_2_aggregated_confusion_matrix.png)

---

## Slide 8: Experiment 4 - Spectral Analysis
### Frequency Domain Features Investigation
- **Approach**: FFT-based spectral features to capture oscillatory patterns
- **Hypothesis**: Disease-specific frequency signatures in saccadic movements
- **Results**: **34% accuracy**
- **Conclusion**: **Oscillatory patterns are not key differentiators** for these conditions
- **Clinical Insight**: Saccadic abnormalities appear more related to amplitude/timing than frequency content

![Spectral Analysis Results](results/EXP_4_Spectral_LGBM/EXP_4_Spectral_LGBM_aggregated_confusion_matrix.png)

**Medical Question**: Do you observe rhythmic or oscillatory abnormalities in saccades clinically?

---

## Slide 9: Experiment 5 - Saccade Segmentation
### Event-Based Analysis Approach
- **Approach**: Segment individual saccade events, analyze with deep learning
- **Results**: **38% accuracy** with significant overfitting
- **Challenge**: Limited saccade events per sequence led to data scarcity
- **Finding**: Event-level analysis did not improve upon sequence-level statistics

**Clinical Question**: How important is individual saccade analysis versus overall movement patterns in your diagnostic process?

---

## Slide 10: Experiment 6 - Binary Classification Breakthrough
### Specialized Models Show Dramatic Improvement
**Key Finding**: **Binary classifications achieve ~72% accuracy** vs 45% multiclass

**Binary Classification Results**:
- **HC vs MG**: 72% accuracy (precision: 0.71-0.73)
- **HC vs CNP (pooled)**: 72% accuracy (precision: 0.66-0.77)
- **CNP subgroups vs HC**: 55% accuracy

![Binary Classification: HC vs MG](results/EXP_6_ADDITIONAL_INVESTIGATIONS_LDA_MG_vs_HC_confusion_matrix.png)

**Strategic Insight**: **Specialized binary classifiers outperform single multiclass model**

**Clinical Implication**: Diagnostic workflow should use disease-specific models rather than universal classifier

---

## Slide 11: Experiment 7 - Feature Optimization & Multi-Stage Classification
### Distinct Feature Sets for Different Comparisons
**Feature Importance Analysis**:
- **HC vs MG**: Eye movement variability (LV_std, LH_std) and error signals most important
- **HC vs CNP**: Horizontal movement variability, error signals critical  
- **MG vs CNP**: Mixed feature importance patterns

**Important Note**: Target features (TargetH_std, TargetV_std) appearing as top discriminators likely indicate **experimental protocol differences** rather than biological differences - these are reference signals that should be consistent across groups.

![MG vs CNP Feature Importance](results/EXP_7_FEATURE_ANALYSIS_MG_vs_CNP_feature_importance.png)

**Multi-Stage Classifier**:
- **Approach**: Sequential binary decisions (HC vs Disease → MG vs CNP → CNP subtypes)
- **Results**: **34% accuracy** due to error compounding
- **Conclusion**: Separate specialized models preferred over staged approach

![Multi-Stage Results](results/EXP_7_FEATURE_ANALYSIS_Multi_Stage_System_confusion_matrix.png)

---

## Slide 12: Experiment 8 - Overfitting Mitigation Attempts
### Comprehensive Regularization Strategy
**Techniques Applied**:
- Aggressive dropout (0.7)
- Weight decay regularization
- Data augmentation (noise injection, time warping)
- Simplified CNN architecture
- Early stopping

**Results**: **37% accuracy** - no improvement
**Conclusion**: **Overfitting was not the primary limitation**; fundamental signal-to-noise ratio challenges exist

---

## Slide 13: Experiment 9 - Saccade Dynamics & Downsampling
### Temporal Resolution and Clinical Features Investigation
**Downsampling Analysis**:
- **Finding**: Neural networks perform better with **downsampled data** (optimal at 1:600 ratio)
- **Best accuracy**: **43% at reduced temporal resolution**

![Downsampling Analysis](results/EXP_9/EXP_9_Model_Accuracy_vs._Downsampling_Rate.png)

**Clinical Feature Analysis**:
- **More Affected Eye (MAE) features**: 38% accuracy
- **Fatigue-related parameters**: Did not exceed 45% baseline
- **Eye asymmetry features**: Showed promise but limited improvement

![MAE Analysis](results/EXP_9/EXP_9_LDA_MAE_Only_confusion_matrix.png)

**Clinical Question**: How do you assess fatigue and asymmetry in clinical practice?

---

## Slide 14: Current Challenges & Limitations
### Key Technical and Clinical Obstacles
**Technical Challenges**:
- **Class imbalance**: TAO severely underrepresented (n=12)
- **Patient variability**: High within-class variance
- **Feature complexity**: 103 features may include noise
- **Temporal dynamics**: Optimal time resolution unclear

**Clinical Challenges**:
- **Disease heterogeneity**: CNP subtypes show overlapping patterns
- **Severity staging**: No incorporation of disease severity
- **Comorbidities**: Potential confounding factors not addressed

---

## Slide 15: Questions for Medical Expert Team
### Critical Clinical Input Needed

1. **Disease Progression**: How do saccadic abnormalities change with disease severity/duration?

2. **Fatigue Assessment**: What clinical markers of fatigue should we incorporate into our analysis?

3. **Dysconjugacy Patterns**: How does eye coordination breakdown manifest differently across conditions?

4. **Recording Protocols**: What standardized saccade tasks would be most diagnostically valuable?

5. **Sample Size Planning**: What minimum sample sizes per condition would you recommend for robust clinical validation?

6. **Clinical Workflow**: How would you envision integrating automated saccade analysis into diagnostic practice?

---

## Slide 16: Future Data Collection Recommendations
### Optimizing Next Phase Data Acquisition

**Sample Size Targets** (based on current findings):
- **Minimum 100 patients per condition** for robust binary classifiers
- **Prioritize TAO recruitment** (currently n=2 patients)
- **Balance CNP subtypes** for subgroup analysis

**Protocol Enhancements**:
- **Standardize recording frequency** (0.5-1.0 Hz optimal range identified)
- **Include fatigue assessment** before/after testing
- **Document disease severity** and duration
- **Control for medications** affecting eye movements

**Feature Collection**:
- **Affected vs unaffected eye designation**
- **Clinical severity scores**
- **Comorbidity documentation**

---

## Slide 17: Summary & Next Steps
### Key Findings and Path Forward

**Major Discoveries**:
1. **Statistical features outperform deep learning** (45% vs 34-38%)
2. **Binary classification achieves 72% accuracy** - clinically promising
3. **Disease-specific feature patterns** identified
4. **Specialized models superior** to universal classifier

**Immediate Next Steps**:
1. **Expand dataset** with medical team guidance
2. **Implement binary classifier pipeline** for clinical testing
3. **Validate findings** on independent cohort
4. **Develop clinical decision support tool**

**Long-term Vision**: **Automated saccade-based diagnostic assistant** to support clinical decision-making in neuromuscular and ocular motor disorders

---

## Slide 18: Technical Appendix - Model Performance Summary
### Comprehensive Results Overview

| Experiment | Approach | Accuracy | Key Finding |
|------------|----------|----------|-------------|
| 1 | Statistical LDA | **45%** | Strong baseline |
| 2 | RNN | 34% | Raw data insufficient |
| 3 | CNN-RNN | 38% | Complex models don't help |
| 4 | Spectral | 34% | Frequency not discriminative |
| 5 | Segmentation | 38% | Event-level analysis limited |
| 6 | Binary LDA | **72%** | Specialized models excel |
| 7 | Multi-stage | 34% | Error compounding issue |
| 8 | Regularized | 37% | Overfitting not main issue |
| 9 | Downsampled | 43% | Temporal resolution matters |

**Best Performance**: Binary HC vs MG and HC vs CNP classifications at 72% accuracy
