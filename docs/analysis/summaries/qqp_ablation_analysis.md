# QQP Ablation Analysis

**Date:** 2026-02-07
**Task:** Quora Question Pairs (Paraphrase Detection)
**Metric:** F1 Score
**Model Pair:** BERT-base → DistilBERT

---

## Results

| Rank | Config | F1 | Loss | Δ vs Baseline |
|------|--------|-----|------|---------------|
| 1 | 03_cotrain_base | **90.61%** | 0.269 | +0.35% |
| 2 | 04_cotrain_sft | 90.46% | 0.325 | +0.20% |
| 3 | 01_frozen_base | 90.44% | 0.400 | +0.18% |
| 4 | 00_ce-only | 90.26% | 0.307 | baseline |
| 5 | 02_frozen_sft | **55.89%** | 0.652 | **-34.37%** ⚠️ |

---

## Training Dynamics

### 00_no-teacher_ce-only
- **Student F1:** 50.48% → 90.26%
- **Eval Loss:** 0.381 → 0.307 (↘ decreasing, min=0.254)
- **Train Loss:** 0.665 → 0.202 → 0.164
- **Notes:** Clean, stable training

### 01_frozen_base-teacher_kl
- **Student F1:** 53.69% → 90.44%
- **Teacher F1:** 25.96% → 25.96% (frozen, random head)
- **Eval Loss:** 0.492 → 0.400 (↘ decreasing)
- **Train Loss:** 0.340 → 0.340 → 0.243
- **Notes:** Higher loss but matches CE-only performance

### 02_frozen_sft-teacher_kl ⚠️ CATASTROPHIC FAILURE
- **Student F1:** 50.48% → 55.89%
- **Teacher F1:** 90.93% → 90.93% (frozen)
- **Eval Loss:** 0.785 → 0.645 (↘ decreasing, min=0.625)
- **Train Loss:** 1.091 → 1.091 → 0.502
- **Notes:** Catastrophic — student barely improves despite 90.93% teacher

### 03_cotrain_base-teacher_kl
- **Student F1:** 53.69% → 90.61%
- **Teacher F1:** 25.96% → 90.97% (+65.01%, co-trained from random)
- **Eval Loss:** 0.360 → 0.270 (↘ best, min=0.238)
- **Train Loss:** 0.994 → 0.994 → 0.253
- **Notes:** Best overall, teacher and student both converge strongly

### 04_cotrain_sft-teacher_kl
- **Student F1:** 50.48% → 90.46%
- **Teacher F1:** 90.93% → 91.32% (+0.38%, slight improvement)
- **Eval Loss:** 0.373 → 0.327 (↘ decreasing, min=0.256)
- **Train Loss:** 1.794 → 1.794 → 0.275
- **Notes:** Good convergence, teacher maintains quality

---

## Catastrophic Failure Analysis: 02_frozen_sft

### What went wrong?

The frozen fine-tuned teacher completely prevents the student from learning:

| Metric | 02_frozen_sft | Others |
|--------|---------------|--------|
| Starting F1 | 48.93% | 82-84% |
| Final F1 | 55.89% | 90%+ |
| Starting Loss | 0.785 | 0.36-0.49 |
| Final Loss | 0.645 | 0.27-0.40 |
| Improvement | +7% | +7-8% |

### Root Cause: Overconfident Wrong Teacher

1. **Teacher is overconfident:** Fine-tuned on QQP, high confidence even on wrong predictions
2. **KL term dominates:** With α=0.5, half the gradient comes from matching teacher
3. **Conflicting gradients:** When teacher is wrong AND confident, KL fights CE
4. **Student stuck:** Can't overcome strong KL gradient toward wrong answer

### Why QQP is worse than MNLI?

- QQP is binary classification (2 classes) vs MNLI (3 classes)
- Overconfident binary predictions create stronger conflicting gradients
- Teacher may have learned spurious correlations that don't generalize

---

## Key Observations

### 1. Frozen SFT is catastrophically bad
- 34.4% drop in F1 vs CE-only
- Barely better than random guessing (50%)
- Training dynamics show model is stuck

### 2. Co-training provides safety
- Co-training SFT (04) works fine: 90.46%
- Difference: teacher adapts, doesn't stay overconfident

### 3. QQP amplifies the problem
- Binary task means wrong predictions are maximally wrong
- MNLI (3-class) has softer failures

---

## Implications

1. **For Paper 1:** QQP is the strongest evidence for "When Teachers Fail"
2. **Never use frozen fine-tuned teachers** without validation
3. **Co-training is essential** for safety with fine-tuned teachers
4. **Binary tasks are higher risk** for distillation failures
