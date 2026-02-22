# CYCLE 49: Saccade Consistency Clustering

## PHASE 2: HYPOTHESIS FORMULATION

### Outside-the-Box Insight #3

**ALL previous cycles** measured individual components or aggregates. What if MG shows a **PATTERN** of saccade quality rather than just degraded performance?

**Completely different angle**: Instead of measuring degradation, measure CONSISTENCY - do saccades cluster into "good" and "bad" groups?

### The Clustering Hypothesis

**Observation**: We have 60+ saccades per eye. Standard approach:
- Compute summary stat (mean error, degradation, MAD)
- These assume ONE underlying distribution

**Alternative view**: MG may show BIMODAL BEHAVIOR:
- "Good" saccades: When muscle not fatigued → tight cluster, low error
- "Bad" saccades: During fatigue episodes → dispersed, high error
- Creates TWO clusters instead of one shifted distribution

**Hypothesis - Silhouette Score**:
Use k-means (k=2) to cluster saccade errors into "good" and "bad" groups:
```python
errors = [error_1, error_2, ..., error_60]  # Per-saccade errors

# K-means with k=2
clusters = kmeans(errors, k=2)  # Find 2 groups

# Silhouette score: How well-separated are clusters?
# Score ranges [-1, 1]:
#   +1 = perfect clustering (tight groups, well-separated)
#   0 = overlapping clusters
#   -1 = wrong clustering

silhouette = silhouette_score(errors, clusters)
```

**Expected pattern**:
- **MG affected eye**: High silhouette (saccades cluster into good/bad groups)
  - E.g., first 30 saccades good (cluster 1), last 30 bad (cluster 2)
- **HC both eyes**: Low silhouette (all saccades similar, no clear clustering)

**Asymmetry**:
```
asym = |silhouette_L - silhouette_R|
```

**Why this is radically different**:
- NOT measuring magnitude (mean, MAD)
- NOT measuring degradation (late - early)
- NOT measuring shape (skewness, kurtosis)
- Measuring **MIXTURE**: Does error distribution contain distinct subpopulations?

**Clinical interpretation**:
- MG: "Sometimes my eyes work, sometimes they don't" → bimodal
- HC: "My eyes work consistently" → unimodal

This captures **intermittent fatigue** rather than progressive degradation.

---

## PHASE 3: Adversarial Review

**Challenge 1**: "Silhouette score requires choosing k=2 arbitrarily - what if true k is different?"

**Response**: Use silhouette WITHOUT pre-specifying k:
- Compute optimal k via silhouette maximization (k=1 to 5)
- If MG truly bimodal, optimal k=2
- If HC unimodal, optimal k=1
- Asymmetry in OPTIMAL K itself could be discriminative

**Challenge 2**: "With only 60 points, clustering will be unstable/unreliable"

**Response**: Valid. But:
- We're not doing inference on cluster centers, just measuring separability
- Silhouette is relatively robust with n≥50
- Can use alternative: Calinski-Harabasz index (variance ratio)

**Challenge 3**: "This assumes temporal bimodality (early vs late) - what if fatigue is intermittent/random?"

**Response**: K-means doesn't assume temporal structure - it finds natural groups regardless of order. If MG shows intermittent fatigue (random timing), clusters will still form.

**Challenge 4**: "Seems like you're just measuring VARIANCE in a complicated way"

**Response**: Not quite:
- High variance: Errors spread across wide range (could be unimodal with long tail)
- High silhouette: Errors form DISTINCT groups with gap between them
- Example: Errors [1,1,1,10,10,10] have same variance as [1,4,4,7,7,10] but different clustering

---

## PHASE 4: Empirical Pre-Implementation Analysis

**Required**:
1. Extract per-saccade errors for each eye (60+ saccades)
2. Apply k-means clustering (k=2)
3. Compute silhouette score
4. Test if MG shows higher silhouette than HC
5. Measure silhouette asymmetry discrimination

**GO Criteria**:
- Silhouette asymmetry d≥0.40
- Orthogonal to degradation (r<0.7)
- MG shows significantly higher silhouette (indicates bimodal pattern)

**NO-GO Criteria**:
- d<0.30
- High correlation with MAD/degradation (redundant)
- Silhouette scores near zero for both MG and HC (no clustering)

---

**Rationale**: This is the MOST outside-the-box approach - measuring whether saccades form distinct quality groups rather than just shifted/wider distributions. Captures intermittent fatigue pattern if it exists.
