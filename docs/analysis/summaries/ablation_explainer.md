# Classification Ablation Results Explained

**Date:** 2024-02-04

---

## The 5 Configs Tested

| Config | What it means |
|--------|---------------|
| **00_ce-only** | No teacher. Student learns from ground truth only (baseline) |
| **01_frozen_base** | Teacher = pretrained BERT, frozen during training |
| **02_frozen_sft** | Teacher = fine-tuned BERT, frozen during training |
| **03_cotrain_base** | Teacher = pretrained BERT, trains alongside student |
| **04_cotrain_sft** | Teacher = fine-tuned BERT, trains alongside student |

---

## Why Each Metric?

| Task | Metric | Why |
|------|--------|-----|
| **MNLI** | Accuracy | Balanced 3-class (entailment/neutral/contradiction). Accuracy works. |
| **QQP** | F1 | Imbalanced binary (63% non-duplicate). F1 handles class imbalance. |
| **SST-2** | Accuracy | Balanced binary sentiment. Accuracy is fine. |
| **CoLA** | MCC | Highly imbalanced (70% acceptable). MCC is robust to imbalance and measures correlation between predictions and labels. |

**MCC (Matthews Correlation Coefficient)** ranges from -1 to +1:
- +1 = perfect
- 0 = random guessing
- -1 = perfect inverse

---

## The Story

### MNLI & QQP: Frozen SFT Teachers Hurt

```
                MNLI (Acc)    QQP (F1)
CE-only         81.6%         90.4%
Frozen SFT      79.8% ↓       55.9% ↓↓↓  ← CATASTROPHIC
Co-train SFT    82.7% ↑       90.5% ↑
```

**Why?** The fine-tuned teacher is overconfident in its predictions. When it's wrong AND confident, the KL loss pushes the student toward wrong answers. This fights against the CE loss (ground truth).

**QQP is worse because:** Binary classification means wrong predictions are maximally wrong (0% vs 100% confidence). In 3-class MNLI, a wrong prediction might still give some probability to the correct class.

---

### SST-2: Task is Too Easy

```
All methods: 90.9% - 91.5%  (only 0.6% spread)
```

**Why?** DistilBERT is already very good at sentiment. There's no room for distillation to help or hurt significantly.

---

### CoLA: Distillation Hurts (Reversal!)

```
CE-only         54.6%  ← BEST
Frozen base     53.4%
Co-train SFT    49.7%  ← WORST
```

**Why the reversal?**
1. **Small dataset** (8.5k examples) - not enough data for distillation to help
2. **Hard task** - grammar acceptability is subtle, teacher isn't reliably better
3. **MCC is strict** - small calibration errors hurt the metric

---

## Bottom Line

| Task | Finding | Evidence Strength |
|------|---------|-------------------|
| MNLI | Frozen SFT hurts, co-train helps | Strong |
| QQP | Frozen SFT is catastrophic | **Very strong** |
| SST-2 | No clear pattern | Weak (saturated) |
| CoLA | Distillation hurts | Counter-example |

**For Paper 1:** Lead with QQP (dramatic), support with MNLI, mention CoLA as limitation.
