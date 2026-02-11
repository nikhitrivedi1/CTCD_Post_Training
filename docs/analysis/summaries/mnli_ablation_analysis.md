# MNLI Ablation Analysis

**Date:** 2026-02-07
**Task:** MultiNLI (Natural Language Inference)
**Metric:** Accuracy
**Model Pair:** BERT-base → DistilBERT

---

## Results

| Rank | Config | Accuracy | Loss | Δ vs Baseline |
|------|--------|----------|------|---------------|
| 1 | 04_cotrain_sft | **82.69%** | 0.588 | +0.50% |
| 2 | 03_cotrain_base | 82.62% | 0.522 | +0.43% |
| 3 | 00_ce-only | 82.19% | 0.547 | baseline |
| 4 | 01_frozen_base | 81.58% | 0.649 | -0.61% |
| 5 | 02_frozen_sft | 79.82% | 0.647 | **-2.37%** |

---

## Training Dynamics

### 00_no-teacher_ce-only
- **Student Accuracy:** 33.39% → 82.19%
- **Eval Loss:** 0.689 → 0.547 (↘ decreasing, min=0.488)
- **Train Loss:** 1.097 → 0.450 → 0.332
- **Notes:** Stable training, strong improvement

### 01_frozen_base-teacher_kl
- **Student Accuracy:** 34.69% → 81.58%
- **Teacher Accuracy:** 31.32% → 31.32% (frozen, random head)
- **Eval Loss:** 0.839 → 0.649 (↘ decreasing)
- **Train Loss:** 0.560 → 0.560 → 0.381
- **Notes:** KL from random teacher adds noise, -0.61% vs CE-only

### 02_frozen_sft-teacher_kl
- **Student Accuracy:** 33.39% → 79.82%
- **Teacher Accuracy:** 84.57% → 84.57% (frozen)
- **Eval Loss:** 0.810 → 0.647 (↘ decreasing)
- **Train Loss:** 0.751 → 0.751 → 0.413
- **Notes:** ⚠️ Frozen SFT teacher hurts student by -2.37%

### 03_cotrain_base-teacher_kl
- **Student Accuracy:** 34.69% → 82.62%
- **Teacher Accuracy:** 31.32% → 73.25% (+41.93%, co-trained from random)
- **Eval Loss:** 0.714 → 0.522 (↘ decreasing, min=0.477)
- **Train Loss:** 1.667 → 1.667 → 0.493
- **Notes:** Best loss trajectory, teacher learns alongside student

### 04_cotrain_sft-teacher_kl
- **Student Accuracy:** 33.39% → 82.69%
- **Teacher Accuracy:** 84.57% → 81.66% (-2.91%, co-training destabilizes)
- **Eval Loss:** 0.721 → 0.588 (↘ decreasing, min=0.502)
- **Train Loss:** 1.685 → 1.685 → 0.502
- **Notes:** Highest student accuracy despite teacher degradation

---

## Key Observations

### 1. Co-training wins
Both co-training configs outperform frozen and CE-only:
- 04_cotrain_sft: 82.69%
- 03_cotrain_base: 82.62%
- Gap of ~0.5% over CE-only baseline

### 2. Frozen SFT hurts
02_frozen_sft underperforms CE-only by 2.37%:
- Frozen fine-tuned teacher provides overconfident wrong guidance
- Student can't overcome strong KL gradients toward incorrect predictions

### 3. Co-training SFT teacher degrades
04_cotrain_sft achieves highest student accuracy despite teacher degradation:
- Teacher: 84.57% → 81.66% (-2.91%)
- Student: 82.69% (best overall)
- Shared learning rate slightly destabilizes already-converged teacher

---

## Implications

1. **For Paper 1:** Clear evidence that frozen SFT teachers hurt on MNLI
2. **For methodology:** Co-training should be default for distillation
3. **For future work:** Investigate teacher degradation in 04 - separate learning rates?
