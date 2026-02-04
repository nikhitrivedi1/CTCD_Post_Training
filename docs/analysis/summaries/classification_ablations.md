# Classification Ablation Analysis

**Date:** 2024-02-04
**Tasks:** MNLI, QQP, SST-2, CoLA
**Model Pair:** BERT-base → DistilBERT

---

## Summary Results

### MNLI (Accuracy)

| Rank | Config | Accuracy | Δ vs CE-only |
|------|--------|----------|--------------|
| 1 | 04_cotrain_sft | **82.69%** | +1.06% |
| 2 | 03_cotrain_base | 82.62% | +0.99% |
| 3 | 00_ce-only | 81.63% | baseline |
| 4 | 01_frozen_base | 81.58% | -0.05% |
| 5 | 02_frozen_sft | 79.82% | **-1.81%** |

### QQP (F1)

| Rank | Config | F1 | Δ vs CE-only |
|------|--------|-----|--------------|
| 1 | 03_cotrain_base | **90.61%** | +0.21% |
| 2 | 04_cotrain_sft | 90.46% | +0.06% |
| 3 | 01_frozen_base | 90.44% | +0.04% |
| 4 | 00_ce-only | 90.40% | baseline |
| 5 | 02_frozen_sft | 55.89% | **-34.51%** ⚠️ |

### SST-2 (Accuracy)

| Rank | Config | Accuracy | Δ vs CE-only |
|------|--------|----------|--------------|
| 1 | 01_frozen_base | **91.51%** | +0.57% |
| 1 | 03_cotrain_base | 91.51% | +0.57% |
| 3 | 04_cotrain_sft | 91.28% | +0.34% |
| 4 | 00_ce-only | 90.94% | baseline |
| 4 | 02_frozen_sft | 90.94% | 0% |

### CoLA (MCC)

| Rank | Config | MCC | Δ vs CE-only |
|------|--------|-----|--------------|
| 1 | 00_ce-only | **54.56%** | baseline |
| 2 | 01_frozen_base | 53.39% | -1.17% |
| 3 | 03_cotrain_base | 51.73% | -2.83% |
| 4 | 02_frozen_sft | 51.45% | -3.11% |
| 5 | 04_cotrain_sft | 49.67% | **-4.89%** |

---

## Training Dynamics

### MNLI Trends

| Config | Eval Loss Trend | Train Loss | Notes |
|--------|-----------------|------------|-------|
| 00_ce-only | 0.57 → 0.55 ↘ | 0.47 → 0.43 | Stable |
| 01_frozen_base | 0.72 → 0.65 ↘ | 0.41 → 0.41 | Stable |
| 02_frozen_sft | 0.72 → 0.65 ↘ | 0.46 → 0.45 | Stable but low acc |
| 03_cotrain_base | 0.54 → 0.52 ↘ | 0.70 → 0.66 | Best overall |
| 04_cotrain_sft | 0.55 → 0.58 ↗ | 0.68 → 0.64 | ⚠️ Loss spike at end |

### QQP Trends

| Config | Eval Loss | Eval F1 | Train Loss | Notes |
|--------|-----------|---------|------------|-------|
| 00_ce-only | 0.30 → 0.30 | 0.87 → 0.90 | 0.23 → 0.23 | Stable |
| 01_frozen_base | 0.45 → 0.40 | 0.87 → 0.90 | 0.26 → 0.26 | Stable |
| 02_frozen_sft | 0.64 → 0.65 | 0.52 → 0.55 | 0.55 → 0.55 | ⚠️ CATASTROPHIC |
| 03_cotrain_base | 0.29 → 0.27 | 0.87 → 0.91 | 0.37 → 0.36 | Best |
| 04_cotrain_sft | 0.29 → 0.33 | 0.88 → 0.90 | 0.37 → 0.36 | Slight loss increase |

---

## Key Findings

### 1. Frozen SFT Teacher is Dangerous

**Evidence:**
- MNLI: -1.81% vs CE-only
- QQP: -34.51% vs CE-only (catastrophic failure)
- CoLA: -3.11% vs CE-only

**Hypothesis:** Overconfident wrong predictions from fine-tuned teacher create strong gradients that fight against ground truth learning.

### 2. Co-training Generally Helps

**Evidence:**
- MNLI: +1.06% (cotrain_sft), +0.99% (cotrain_base)
- QQP: +0.21% (cotrain_base)
- SST-2: +0.57% (cotrain_base)

**Hypothesis:** Teacher that adapts alongside student provides more useful guidance than frozen expert.

### 3. CoLA is Different

**Evidence:**
- CE-only wins (54.56%) over all distillation methods
- Co-training hurts more than frozen (-2.83% to -4.89%)

**Hypothesis:**
- Small dataset (8.5k examples) may not benefit from distillation
- MCC metric may be more sensitive to minor prediction shifts
- Harder task where teacher isn't reliably better

### 4. SST-2 is Saturated

**Evidence:**
- All methods within 0.6% of each other
- All achieve 90.9-91.5% accuracy

**Hypothesis:** Task is too easy for model capacity; ceiling effect.

---

## Config Reference

| Config | α | Freeze | Teacher Type |
|--------|---|--------|--------------|
| 00_no-teacher_ce-only | 0.0 | - | None |
| 01_frozen_base-teacher_kl | 0.5 | True | bert-base-uncased |
| 02_frozen_sft-teacher_kl | 0.5 | True | bert-base fine-tuned |
| 03_cotrain_base-teacher_kl | 0.5 | False | bert-base-uncased |
| 04_cotrain_sft-teacher_kl | 0.5 | False | bert-base fine-tuned |

---

## Implications for Paper 1

1. **Title validated:** "When Teachers Fail" - frozen SFT teachers DO fail
2. **Main finding confirmed:** Frozen SFT < CE-only on MNLI and catastrophically on QQP
3. **Co-training story:** Generally beneficial, provides robustness
4. **Nuance needed:** CoLA shows opposite pattern - may need discussion of task difficulty
