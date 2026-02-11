# Classification Ablation Analysis

**Date:** 2026-02-07
**Tasks:** MNLI, QQP, SST-2, CoLA
**Model Pair:** BERT-base → DistilBERT

---

## Summary Results

### MNLI (Accuracy)

| Rank | Config | Accuracy | Δ vs CE-only |
|------|--------|----------|--------------|
| 1 | 04_cotrain_sft | **82.69%** | +0.50% |
| 2 | 03_cotrain_base | 82.62% | +0.43% |
| 3 | 00_ce-only | 82.19% | baseline |
| 4 | 01_frozen_base | 81.58% | -0.61% |
| 5 | 02_frozen_sft | 79.82% | **-2.37%** |

### QQP (F1)

| Rank | Config | F1 | Δ vs CE-only |
|------|--------|-----|--------------|
| 1 | 03_cotrain_base | **90.61%** | +0.35% |
| 2 | 04_cotrain_sft | 90.46% | +0.20% |
| 3 | 01_frozen_base | 90.44% | +0.18% |
| 4 | 00_ce-only | 90.26% | baseline |
| 5 | 02_frozen_sft | 55.89% | **-34.37%** ⚠️ |

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
| 1 | 00_ce-only | **53.80%** | baseline |
| 2 | 01_frozen_base | 53.39% | -0.41% |
| 3 | 03_cotrain_base | 51.73% | -2.07% |
| 4 | 02_frozen_sft | 51.45% | -2.35% |
| 5 | 04_cotrain_sft | 49.67% | **-4.13%** |

---

## Training Dynamics

### MNLI Trends (Student/Pre → Student/Post Accuracy)

| Config | Student Acc | Teacher Acc | Eval Loss | Notes |
|--------|-----------|-------------|-----------|-------|
| 00_ce-only | 33.39% → 82.19% | — | 0.547 | Baseline |
| 01_frozen_base | 34.69% → 81.58% | 31.32% → 31.32% | 0.649 | Frozen random teacher |
| 02_frozen_sft | 33.39% → 79.82% | 84.57% → 84.57% | 0.647 | ⚠️ Frozen SFT hurts |
| 03_cotrain_base | 34.69% → 82.62% | 31.32% → 73.25% | 0.522 | Teacher learns too |
| 04_cotrain_sft | 33.39% → 82.69% | 84.57% → 81.66% | 0.588 | Teacher degrades slightly |

### QQP Trends (Student/Pre → Student/Post F1)

| Config | Student F1 | Teacher F1 | Eval Loss | Notes |
|--------|-----------|------------|-----------|-------|
| 00_ce-only | 50.48% → 90.26% | — | 0.307 | Baseline |
| 01_frozen_base | 53.69% → 90.44% | 25.96% → 25.96% | 0.400 | Frozen random teacher |
| 02_frozen_sft | 50.48% → 55.89% | 90.93% → 90.93% | 0.652 | ⚠️ CATASTROPHIC |
| 03_cotrain_base | 53.69% → 90.61% | 25.96% → 90.97% | 0.269 | Teacher converges too |
| 04_cotrain_sft | 50.48% → 90.46% | 90.93% → 91.32% | 0.325 | Teacher improves slightly |

### SST-2 Trends (Student/Pre → Student/Post Accuracy)

| Config | Student Acc | Teacher Acc | Eval Loss | Notes |
|--------|-----------|-------------|-----------|-------|
| 00_ce-only | 43.46% → 90.94% | — | 0.392 | Baseline |
| 01_frozen_base | 45.76% → 91.51% | 50.92% → 50.92% | 0.399 | Frozen random teacher |
| 02_frozen_sft | 43.46% → 90.94% | 92.43% → 92.43% | 0.368 | Frozen SFT neutral |
| 03_cotrain_base | 45.76% → 91.51% | 50.92% → 93.58% | 0.311 | Teacher surpasses SFT |
| 04_cotrain_sft | 43.46% → 91.28% | 92.43% → 92.32% | 0.376 | Teacher essentially stable |

### CoLA Trends (Student/Pre → Student/Post MCC)

| Config | Student MCC | Teacher MCC | Eval Loss | Notes |
|--------|-----------|-------------|-----------|-------|
| 00_ce-only | 5.07% → 53.80% | — | 0.556 | Baseline (best) |
| 01_frozen_base | 5.04% → 53.39% | 10.40% → 10.40% | 0.482 | Frozen random, nearly ties |
| 02_frozen_sft | 5.07% → 51.45% | 53.39% → 53.39% | 0.516 | Teacher ~= student final |
| 03_cotrain_base | 5.04% → 51.73% | 10.40% → 58.08% | 0.560 | Teacher learns, student doesn't benefit |
| 04_cotrain_sft | 5.07% → 49.67% | 53.39% → 56.29% | 0.643 | ⚠️ Worst student |

---

## Key Findings

### 1. Frozen SFT Teacher is Dangerous

**Evidence:**
- MNLI: -2.37% vs CE-only
- QQP: -34.37% vs CE-only (catastrophic failure)
- CoLA: -2.35% vs CE-only

**Hypothesis:** Overconfident wrong predictions from fine-tuned teacher create strong gradients that fight against ground truth learning.

### 2. Co-training Generally Helps

**Evidence:**
- MNLI: +0.50% (cotrain_sft), +0.43% (cotrain_base)
- QQP: +0.35% (cotrain_base)
- SST-2: +0.57% (cotrain_base)

**Hypothesis:** Teacher that adapts alongside student provides more useful guidance than frozen expert.

### 3. CoLA is Different

**Evidence:**
- CE-only wins (53.80%) over all distillation methods
- Co-training hurts more than frozen (-2.07% to -4.13%)

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

## FAR Matrix (MNLI only)

The FAR study tests all combinations of Feature (F), Attention (A), and Response (R) distillation signals.

| Config | Components | Accuracy | Δ vs CE-only |
|--------|------------|----------|--------------|
| 06_ar | A + R | **81.73%** | +1.41% |
| 03_r-only | R only | 81.70% | +1.38% |
| 05_fr | F + R | 81.45% | +1.13% |
| 07_far | F + A + R | 81.43% | +1.11% |
| 01_f-only | F only | 80.82% | +0.50% |
| 04_fa | F + A | 80.58% | +0.26% |
| 00_ce-only | None | 80.32% | baseline |
| 02_a-only | A only | 80.19% | -0.13% |

### FAR Key Findings

1. **R (Response/KL) is most valuable** - R-only achieves 81.70%, nearly as good as any combination
2. **A (Attention) alone hurts** - A-only is the only config worse than CE-only (-0.13%)
3. **Adding F or A to R provides diminishing returns** - AR (81.73%) barely beats R (81.70%)
4. **Complexity doesn't help** - FAR (81.43%) < R-only (81.70%)

### Component Contribution Analysis

| Component Added | Base | Result | Delta |
|-----------------|------|--------|-------|
| +R to CE-only | 80.32% | 81.70% | **+1.38%** |
| +F to CE-only | 80.32% | 80.82% | +0.50% |
| +A to CE-only | 80.32% | 80.19% | -0.13% |
| +A to R | 81.70% | 81.73% | +0.03% |
| +F to R | 81.70% | 81.45% | -0.25% |
| +A to F | 80.82% | 80.58% | -0.24% |

---

## Staged Distillation (MNLI only)

Tests progressive training: start with one signal, transition to another.

| Config | Stages | Accuracy | Δ vs R-only |
|--------|--------|----------|-------------|
| 00_r | R only | **81.68%** | baseline |
| 01_f-r | F → R | 81.52% | -0.16% |
| 02_a-r | A → R | 80.77% | -0.91% |
| 03_f-a-r | F → A → R | 80.18% | **-1.50%** |

### Staged Key Findings

1. **More stages = worse results** - Simple R-only beats all staged approaches
2. **A stage is harmful** - Anything with A stage performs worse
3. **Staging adds complexity without benefit** - No evidence that curriculum helps

---

## Implications for Paper 1

1. **Title validated:** "When Teachers Fail" - frozen SFT teachers DO fail
2. **Main finding confirmed:** Frozen SFT < CE-only on MNLI and catastrophically on QQP
3. **Co-training story:** Generally beneficial, provides robustness
4. **Nuance needed:** CoLA shows opposite pattern - may need discussion of task difficulty
5. **FAR insight:** Response (KL) is the only signal that matters; F and A add noise
6. **Staging fails:** Curriculum-style distillation doesn't help on MNLI
