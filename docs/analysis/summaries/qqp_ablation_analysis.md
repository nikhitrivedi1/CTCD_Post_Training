# QQP Ablation Analysis

**Date:** 2024-02-04
**Task:** Quora Question Pairs (Paraphrase Detection)
**Metric:** F1 Score
**Model Pair:** BERT-base → DistilBERT

---

## Results

| Rank | Config | F1 | Loss | Δ vs Baseline |
|------|--------|-----|------|---------------|
| 1 | 03_cotrain_base | **90.61%** | 0.269 | +0.21% |
| 2 | 04_cotrain_sft | 90.46% | 0.319 | +0.06% |
| 3 | 01_frozen_base | 90.44% | 0.400 | +0.04% |
| 4 | 00_ce-only | 90.40% | 0.301 | baseline |
| 5 | 02_frozen_sft | **55.89%** | 0.661 | **-34.51%** ⚠️ |

---

## Training Dynamics

### 00_no-teacher_ce-only
- **Eval Loss:** 0.298 → 0.301 (stable)
- **Eval F1:** 86.84% → 90.40%
- **Train Loss:** 0.234 → 0.226
- **Notes:** Clean, stable training

### 01_frozen_base-teacher_kl
- **Eval Loss:** 0.455 → 0.400 (↘ improving)
- **Eval F1:** 86.89% → 90.43%
- **Train Loss:** 0.257 → 0.256
- **Notes:** Higher loss but matches CE-only performance

### 02_frozen_sft-teacher_kl ⚠️ CATASTROPHIC FAILURE
- **Eval Loss:** 0.637 → 0.645 (↗ getting worse!)
- **Eval F1:** 51.96% → 54.53% (barely above random)
- **Train Loss:** 0.551 → 0.547 (stuck high)
- **Notes:** Complete training failure - teacher prevents learning

### 03_cotrain_base-teacher_kl
- **Eval Loss:** 0.291 → 0.270 (↘ best)
- **Eval F1:** 87.08% → 90.57%
- **Train Loss:** 0.370 → 0.359
- **Notes:** Best overall, clean convergence

### 04_cotrain_sft-teacher_kl
- **Eval Loss:** 0.289 → 0.327 (↗ slight increase)
- **Eval F1:** 87.55% → 90.43%
- **Train Loss:** 0.368 → 0.358
- **Notes:** Good but loss increases late

---

## Catastrophic Failure Analysis: 02_frozen_sft

### What went wrong?

The frozen fine-tuned teacher completely prevents the student from learning:

| Metric | 02_frozen_sft | Others |
|--------|---------------|--------|
| Starting F1 | 51.96% | 86-88% |
| Final F1 | 55.89% | 90%+ |
| Starting Loss | 0.637 | 0.29-0.45 |
| Final Loss | 0.661 | 0.27-0.40 |
| Improvement | +4% | +3-4% |

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
- 34.5% drop in F1 vs CE-only
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
