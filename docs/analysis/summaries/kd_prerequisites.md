# Prerequisites for Effective Knowledge Distillation

A systematic segmentation of factors that determine when KD helps, hurts, or is neutral.

---

## Task Segmentation

| Task | Train Size | Classes | Balance | Metric | Difficulty | KD Effect |
|------|------------|---------|---------|--------|------------|-----------|
| MNLI | 393k | 3 | Balanced | Accuracy | Medium | **Helps** (+1%) |
| QQP | 364k | 2 | 63/37 | F1 | Medium | **Catastrophic** (-34%) or Helps (+0.2%) |
| SST-2 | 67k | 2 | Balanced | Accuracy | Easy | **Neutral** (±0.6%) |
| CoLA | 8.5k | 2 | 70/30 | MCC | Hard | **Hurts** (-5%) |

---

## 1. Data Quantity

### Hypothesis
KD requires sufficient data to:
1. Train the teacher effectively
2. Learn the teacher-student mapping
3. Avoid overfitting to teacher's biases

### Evidence

| Task | Train Size | KD Benefit |
|------|------------|------------|
| MNLI | 393k | ✓ Co-train helps |
| QQP | 364k | ✓ Co-train helps |
| SST-2 | 67k | ~ Neutral |
| CoLA | **8.5k** | ✗ KD hurts |

### Threshold Estimate
- **>100k examples**: KD likely beneficial
- **10k-100k examples**: Task-dependent
- **<10k examples**: KD may hurt (CoLA)

### Why Small Data Hurts KD
1. Teacher overfits to training set
2. Teacher's errors are systematic, not random
3. Student inherits teacher's overfitting
4. Not enough examples to average out teacher noise

---

## 2. Class Distribution

### Hypothesis
Imbalanced classes amplify frozen teacher failures because:
1. Teacher learns majority class bias
2. Confident wrong predictions on minority class
3. KL loss fights CE on minority examples

### Evidence

| Task | Distribution | Frozen SFT Effect |
|------|--------------|-------------------|
| MNLI | 33/33/33 (balanced) | -1.8% |
| SST-2 | 50/50 (balanced) | 0% |
| QQP | 63/37 (imbalanced) | **-34%** |
| CoLA | 70/30 (imbalanced) | -3% |

### The QQP Catastrophe Explained
- 63% of QQP is "not duplicate" (class 0)
- Teacher likely overconfident on majority class
- When teacher wrong on minority class (duplicates):
  - CE says: "this IS a duplicate"
  - KL says: "teacher says NOT duplicate with 95% confidence"
  - Gradient conflict → student stuck

### Imbalance Risk Score
```
Risk = (majority_class_% - 50%) × num_classes
```
- MNLI: (33% - 50%) × 3 = -51% (low risk, balanced)
- QQP: (63% - 50%) × 2 = 26% (high risk)
- CoLA: (70% - 50%) × 2 = 40% (high risk)

---

## 3. Number of Classes

### Hypothesis
Binary classification is higher risk for KD failure because:
1. Wrong prediction = maximally wrong (0% vs 100%)
2. No "partial credit" for nearby classes
3. Gradient conflict is absolute

### Evidence

| Task | Classes | Wrong Prediction Severity | Frozen SFT Effect |
|------|---------|---------------------------|-------------------|
| MNLI | 3 | Partial (e.g., entail→neutral) | -1.8% |
| QQP | 2 | **Maximal** (dup→not dup) | **-34%** |
| SST-2 | 2 | Maximal (pos→neg) | 0% |
| CoLA | 2 | Maximal (accept→reject) | -3% |

### Why SST-2 Escapes
- Task is easy → teacher rarely wrong
- When wrong, sentiment is genuinely ambiguous
- Less "confident wrong" predictions

---

## 4. Task Difficulty

### Hypothesis
Task difficulty interacts with KD in complex ways:
- **Easy tasks**: Student already good → KD unnecessary
- **Medium tasks**: KD can help bridge gap
- **Hard tasks**: Teacher may not be reliably better → KD adds noise

### Difficulty Proxies

| Task | Student CE-only | Teacher Accuracy | Gap | Difficulty |
|------|-----------------|------------------|-----|------------|
| SST-2 | 90.9% | ~93% | 2% | Easy |
| MNLI | 81.6% | ~84% | 2.4% | Medium |
| QQP | 90.4% (F1) | ~91% | 0.6% | Medium |
| CoLA | 54.6% (MCC) | ~60% | 5.4% | Hard |

### The Difficulty Paradox
- CoLA has the **largest** teacher-student gap (5.4%)
- Yet KD **hurts** on CoLA
- Why? Teacher's 60% MCC means 40% systematic errors
- Student inherits these errors via KL

### Sweet Spot
```
KD works when:
- Teacher is significantly better (gap > 2%)
- Teacher is reliable (accuracy > 80%)
- Task isn't saturated (student < 90%)
```

---

## 5. Teacher Quality & Calibration

### Hypothesis
Teacher must be:
1. **Accurate**: Actually knows the right answers
2. **Calibrated**: Confidence matches correctness
3. **Generalizable**: Patterns transfer to student

### Teacher Failure Modes

| Failure Mode | Symptom | Affected Task |
|--------------|---------|---------------|
| Overconfident-wrong | High confidence, wrong answer | QQP |
| Underconfident-right | Low confidence, right answer | (less harmful) |
| Systematic bias | Always wrong on subgroup | CoLA (grammar edge cases) |
| Distribution shift | Good on train, bad on eval | All |

### Calibration Check (Proposed)
Before distillation, verify:
```python
# Expected Calibration Error (ECE)
ece = mean(|confidence - accuracy|) per confidence bin

# If ECE > 0.1, teacher is miscalibrated
# Consider: higher temperature, lower α, or skip KD
```

---

## 6. KD Scope Framework

### When KD Helps

| Condition | Why |
|-----------|-----|
| Large dataset (>100k) | Enough data to learn teacher mapping |
| Balanced classes | No majority class bias amplification |
| Medium difficulty | Room for improvement, teacher reliable |
| Well-calibrated teacher | Confidence matches correctness |
| Multi-class (3+) | Partial credit for wrong predictions |

### When KD Hurts

| Condition | Why |
|-----------|-----|
| Small dataset (<10k) | Teacher overfits, student inherits |
| Imbalanced classes | Majority bias amplified |
| Very hard task | Teacher unreliable |
| Binary classification | Maximally wrong predictions |
| Frozen overconfident teacher | Gradient conflict |

### When KD is Neutral

| Condition | Why |
|-----------|-----|
| Easy/saturated task | Student already at ceiling |
| Tiny teacher-student gap | Nothing to transfer |
| Very high temperature | Soft labels become uniform |

---

## 7. Decision Framework

```
                    ┌─────────────────┐
                    │ Dataset Size?   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
           <10k          10-100k         >100k
              │              │              │
              ▼              ▼              ▼
         Skip KD      Check other      Likely OK
         (CoLA)         factors
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
              Balanced?          Imbalanced?
                    │                 │
                    ▼                 ▼
               Check            Use Co-train
              difficulty         (not frozen)
                    │                 │
         ┌─────────┴─────────┐       │
         ▼                   ▼       │
      Easy?              Medium?     │
         │                   │       │
         ▼                   ▼       ▼
    Skip KD              Use KD    Co-train
    (SST-2)             (MNLI)    mandatory
                                   (QQP)
```

---

## 8. Recommendations by Task Profile

### Profile A: Large + Balanced + Medium
**Examples:** MNLI
**Recommendation:** KD beneficial, co-train preferred
**Expected gain:** +1-3%

### Profile B: Large + Imbalanced + Medium
**Examples:** QQP
**Recommendation:** Co-train ONLY, never frozen SFT
**Risk:** Catastrophic failure with frozen teacher
**Expected gain:** +0.2% (co-train) or -34% (frozen SFT)

### Profile C: Medium + Balanced + Easy
**Examples:** SST-2
**Recommendation:** Skip KD, diminishing returns
**Expected gain:** ±0.5%

### Profile D: Small + Imbalanced + Hard
**Examples:** CoLA
**Recommendation:** Skip KD entirely
**Expected gain:** Negative (-3% to -5%)

---

## 9. Metrics for KD Suitability Score

Proposed composite score:

```python
def kd_suitability(dataset_size, class_balance, task_difficulty, teacher_accuracy):
    """
    Returns score 0-100. Higher = KD more likely to help.
    """
    # Size factor (0-25)
    if dataset_size > 100_000:
        size_score = 25
    elif dataset_size > 10_000:
        size_score = 15
    else:
        size_score = 0

    # Balance factor (0-25)
    # class_balance = min_class_% / max_class_%
    balance_score = class_balance * 25

    # Difficulty factor (0-25)
    # Sweet spot is 70-85% student accuracy
    if 0.70 <= task_difficulty <= 0.85:
        difficulty_score = 25
    elif 0.60 <= task_difficulty <= 0.90:
        difficulty_score = 15
    else:
        difficulty_score = 5

    # Teacher factor (0-25)
    if teacher_accuracy > 0.85:
        teacher_score = 25
    elif teacher_accuracy > 0.75:
        teacher_score = 15
    else:
        teacher_score = 5

    return size_score + balance_score + difficulty_score + teacher_score
```

### Applying to Our Tasks

| Task | Size | Balance | Difficulty | Teacher | Score | Actual |
|------|------|---------|------------|---------|-------|--------|
| MNLI | 25 | 25 | 25 | 20 | **95** | ✓ Helps |
| QQP | 25 | 15 | 15 | 20 | **75** | ⚠️ Risky |
| SST-2 | 15 | 25 | 5 | 25 | **70** | ~ Neutral |
| CoLA | 0 | 15 | 15 | 15 | **45** | ✗ Hurts |

---

## 10. Summary

### The Case for KD
KD is beneficial when:
1. **Data is plentiful** (>100k examples)
2. **Classes are balanced** (or co-training is used)
3. **Task is medium difficulty** (60-85% baseline)
4. **Teacher is reliable** (>80% accuracy)

### The Case Against KD
Skip KD when:
1. **Data is scarce** (<10k examples)
2. **Task is easy** (>90% baseline)
3. **Task is very hard** (teacher unreliable)
4. **Frozen teacher on imbalanced binary task**

### The Co-Training Imperative
When using KD:
1. **Always validate with CE-only baseline**
2. **Prefer co-training over frozen** (safety)
3. **Never freeze on imbalanced binary tasks**
4. **Monitor for training instability**
