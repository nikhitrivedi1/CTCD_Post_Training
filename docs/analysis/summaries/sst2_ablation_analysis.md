# SST-2 Ablation Analysis

**Date:** 2026-02-07
**Task:** Stanford Sentiment Treebank (Binary Sentiment)
**Metric:** Accuracy
**Model Pair:** BERT-base → DistilBERT

---

## Results

| Rank | Config | Accuracy | Loss | Δ vs Baseline |
|------|--------|----------|------|---------------|
| 1 | 01_frozen_base | **91.51%** | 0.393 | +0.57% |
| 1 | 03_cotrain_base | 91.51% | 0.240 | +0.57% |
| 3 | 04_cotrain_sft | 91.28% | 0.268 | +0.34% |
| 4 | 00_ce-only | 90.94% | 0.300 | baseline |
| 4 | 02_frozen_sft | 90.94% | 0.368 | 0% |

---

## Key Observations

### 1. Task is saturated
- All methods within 0.57% of each other
- 90.9% - 91.5% range is very narrow
- No method significantly outperforms

### 2. Frozen SFT doesn't hurt (unlike MNLI/QQP)
- 02_frozen_sft matches CE-only exactly
- No catastrophic failure like QQP
- Possible reason: simpler binary sentiment task

### 3. Frozen base ties with co-train
- 01_frozen_base = 03_cotrain_base = 91.51%
- Co-training overhead not justified for this task

---

## Why is SST-2 Different?

### Task Simplicity
- Binary sentiment classification
- Clear positive/negative signals
- Less room for teacher to be confidently wrong

### Student Capacity
- DistilBERT already very capable at sentiment
- ~91% may be near student ceiling
- Distillation can't push beyond capacity

### Teacher Quality
- BERT is excellent at sentiment
- Few overconfident wrong predictions
- KL guidance is mostly helpful

---

## Implications

1. **For Paper 1:** SST-2 doesn't support "frozen teachers fail" narrative
2. **For methodology:** Simple tasks may not need distillation
3. **Recommendation:** Focus ablation stories on MNLI and QQP
