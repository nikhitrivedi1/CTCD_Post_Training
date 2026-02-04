# CoLA Ablation Analysis

**Date:** 2024-02-04
**Task:** Corpus of Linguistic Acceptability (Grammar Judgment)
**Metric:** Matthews Correlation Coefficient (MCC)
**Model Pair:** BERT-base → DistilBERT

---

## Results

| Rank | Config | MCC | Loss | Δ vs Baseline |
|------|--------|-----|------|---------------|
| 1 | 00_ce-only | **54.56%** | 0.558 | baseline |
| 2 | 01_frozen_base | 53.39% | 0.482 | -1.17% |
| 3 | 03_cotrain_base | 51.73% | 0.560 | -2.83% |
| 4 | 02_frozen_sft | 51.45% | 0.516 | -3.11% |
| 5 | 04_cotrain_sft | 49.67% | 0.643 | **-4.89%** |

---

## Key Observations

### 1. CE-only wins - distillation hurts!
- Best performance with NO teacher
- All distillation methods underperform
- Complete reversal of MNLI/QQP pattern

### 2. Co-training makes it worse
- 03_cotrain_base: -2.83%
- 04_cotrain_sft: -4.89% (worst)
- Opposite of MNLI where co-training helps

### 3. Frozen base is best distillation
- 01_frozen_base: -1.17% (smallest gap)
- Still worse than CE-only

---

## Why is CoLA Different?

### Small Dataset
- Only 8,551 training examples
- Distillation may need more data to be effective
- Teacher signal adds noise on small data

### Hard Task
- Linguistic acceptability is subtle
- MCC is a stricter metric than accuracy
- Teacher may not be reliably better

### MCC Sensitivity
- MCC penalizes false positives AND false negatives
- Small prediction shifts can hurt MCC significantly
- KL distortion may cause miscalibration

### Teacher Quality
- BERT isn't dramatically better than DistilBERT on CoLA
- Teacher guidance may be more harmful than helpful
- Less headroom for distillation gains

---

## Gradient Conflict Hypothesis

On CoLA, the distillation gradient may:
1. **Add noise:** Teacher uncertain on grammar edge cases
2. **Fight CE:** Teacher disagrees with ground truth on ambiguous examples
3. **Hurt calibration:** KL term pushes toward teacher's uncertainty

---

## Implications

1. **For Paper 1:** CoLA is a counterexample - needs discussion
2. **For methodology:** Distillation isn't universally beneficial
3. **Conditions for benefit:**
   - Sufficient training data (CoLA too small)
   - Teacher significantly better than student
   - Task allows clear knowledge transfer

---

## Recommendations

1. **Don't include CoLA in main findings** - it muddles the story
2. **Mention as limitation/future work** - "distillation doesn't help on all tasks"
3. **Investigate further:** Why does co-training hurt on CoLA?
