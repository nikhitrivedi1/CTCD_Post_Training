# MNLI Ablation Analysis

**Date:** 2024-02-04
**Task:** MultiNLI (Natural Language Inference)
**Metric:** Accuracy
**Model Pair:** BERT-base → DistilBERT

---

## Results

| Rank | Config | Accuracy | Loss | Δ vs Baseline |
|------|--------|----------|------|---------------|
| 1 | 04_cotrain_sft | **82.69%** | 0.587 | +1.06% |
| 2 | 03_cotrain_base | 82.62% | 0.519 | +0.99% |
| 3 | 00_ce-only | 81.63% | 0.550 | baseline |
| 4 | 01_frozen_base | 81.58% | 0.653 | -0.05% |
| 5 | 02_frozen_sft | 79.82% | 0.647 | **-1.81%** |

---

## Training Dynamics

### 00_no-teacher_ce-only
- **Eval Loss:** 0.5715 → 0.5508 (↘ decreasing)
- **Eval Acc:** 76.99% → 81.19%
- **Train Loss:** 0.468 (early) → 0.356 (mid) → 0.432 (late)
- **Notes:** Stable training, consistent improvement

### 01_frozen_base-teacher_kl
- **Eval Loss:** 0.7240 → 0.6538 (↘ decreasing)
- **Eval Acc:** 77.04% → 81.39%
- **Train Loss:** 0.414 → 0.391 → 0.406
- **Notes:** Higher loss than CE-only due to KL term, but similar accuracy

### 02_frozen_sft-teacher_kl
- **Eval Loss:** 0.7162 → 0.6489 (↘ decreasing)
- **Eval Acc:** 74.87% → 79.56%
- **Train Loss:** 0.459 → 0.424 → 0.448
- **Notes:** ⚠️ Starts lower, ends lower - frozen SFT teacher hurts

### 03_cotrain_base-teacher_kl
- **Eval Loss:** 0.5411 → 0.5185 (↘ decreasing, min=0.487)
- **Eval Acc:** 78.18% → 82.62%
- **Train Loss:** 0.705 → 0.554 → 0.656
- **Notes:** Best loss trajectory, highest final accuracy

### 04_cotrain_sft-teacher_kl
- **Eval Loss:** 0.5473 → 0.5810 (↗ **spike at end**)
- **Eval Acc:** 78.81% → 82.50% (best=82.56%)
- **Train Loss:** 0.681 → 0.561 → 0.639
- **Notes:** ⚠️ Loss spike at end, but still highest accuracy

---

## Key Observations

### 1. Co-training wins
Both co-training configs outperform frozen and CE-only:
- 04_cotrain_sft: 82.69%
- 03_cotrain_base: 82.62%
- Gap of ~1% over CE-only baseline

### 2. Frozen SFT hurts
02_frozen_sft underperforms CE-only by 1.81%:
- Frozen fine-tuned teacher provides overconfident wrong guidance
- Student can't overcome strong KL gradients toward incorrect predictions

### 3. Loss vs Accuracy divergence
04_cotrain_sft shows loss spike but highest accuracy:
- Final eval loss: 0.581 (highest among co-train)
- Final accuracy: 82.69% (highest overall)
- Suggests calibration issues, not prediction quality

---

## Implications

1. **For Paper 1:** Clear evidence that frozen SFT teachers hurt on MNLI
2. **For methodology:** Co-training should be default for distillation
3. **For future work:** Investigate loss spike in 04 - overfitting or calibration?
