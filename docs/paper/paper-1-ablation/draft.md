# When Teachers Fail: The Overconfident-Wrong Teacher Paradox in Knowledge Distillation

**Status:** Data complete for MNLI, QQP, SST-2, CoLA

---

## Abstract

Knowledge distillation transfers knowledge from large teacher models to smaller students. Standard practice freezes a fine-tuned teacher and trains the student to match soft labels. We present a systematic ablation study across four GLUE tasks revealing that **frozen fine-tuned teachers can catastrophically harm student performance**—dropping F1 by 34% on QQP. The culprit: overconfident wrong predictions create gradient conflict with ground truth. Co-training provides a solution: teachers that adapt alongside students self-correct their errors. Our key insight: *a teacher's willingness to update matters more than its starting accuracy*.

---

## 1. Introduction

### Research Questions
1. Does distillation actually help over training without a teacher?
2. Does teacher quality always correlate with student quality?
3. When should teachers be frozen vs co-trained?

### Contributions
1. Systematic 5-config ablation study across 4 GLUE tasks
2. Discovery of **catastrophic failure** with frozen fine-tuned teachers (QQP: -34% F1)
3. Evidence that co-training provides robustness to teacher miscalibration
4. Identification of task-dependent effects (CoLA reversal)

---

## 2. Experimental Setup

### Models

| Role | Model | Parameters |
|------|-------|------------|
| Teacher | bert-base-uncased | 110M |
| Student | distilbert-base-uncased | 66M |

### Ablation Configs

| Config | Teacher | Frozen | α | Description |
|--------|---------|--------|---|-------------|
| 00 | None | - | 0.0 | CE-only baseline |
| 01 | Base | ✓ | 0.5 | Frozen pretrained |
| 02 | SFT | ✓ | 0.5 | Frozen fine-tuned |
| 03 | Base | ✗ | 0.5 | Co-train pretrained |
| 04 | SFT | ✗ | 0.5 | Co-train fine-tuned |

### Tasks & Metrics

| Task | Examples | Classes | Metric | Why This Metric |
|------|----------|---------|--------|-----------------|
| MNLI | 393k | 3 | Accuracy | Balanced classes |
| QQP | 364k | 2 | F1 | Imbalanced (63% negative) |
| SST-2 | 67k | 2 | Accuracy | Balanced sentiment |
| CoLA | 8.5k | 2 | MCC | Highly imbalanced, correlation-based |

---

## 3. Results

### MNLI (Accuracy)

| Rank | Config | Accuracy | Δ vs CE-only |
|------|--------|----------|--------------|
| 1 | 04_cotrain_sft | **82.69%** | +1.06% |
| 2 | 03_cotrain_base | 82.62% | +0.99% |
| 3 | 00_ce-only | 81.63% | baseline |
| 4 | 01_frozen_base | 81.58% | -0.05% |
| 5 | 02_frozen_sft | 79.82% | **-1.81%** |

**Finding:** Frozen SFT underperforms CE-only by 1.81%. Co-training recovers +2.87% over frozen SFT.

### QQP (F1) ⚠️ CATASTROPHIC FAILURE

| Rank | Config | F1 | Δ vs CE-only |
|------|--------|-----|--------------|
| 1 | 03_cotrain_base | **90.61%** | +0.21% |
| 2 | 04_cotrain_sft | 90.46% | +0.06% |
| 3 | 01_frozen_base | 90.44% | +0.04% |
| 4 | 00_ce-only | 90.40% | baseline |
| 5 | 02_frozen_sft | **55.89%** | **-34.51%** |

**Finding:** Frozen SFT is catastrophic—barely above random (50%). Student cannot learn when teacher provides confident wrong guidance on this binary task.

### SST-2 (Accuracy) — Saturated

| Rank | Config | Accuracy | Δ vs CE-only |
|------|--------|----------|--------------|
| 1 | 01_frozen_base | 91.51% | +0.57% |
| 1 | 03_cotrain_base | 91.51% | +0.57% |
| 3 | 04_cotrain_sft | 91.28% | +0.34% |
| 4 | 00_ce-only | 90.94% | baseline |
| 4 | 02_frozen_sft | 90.94% | 0% |

**Finding:** All methods within 0.6%. Task is saturated—DistilBERT already near ceiling.

### CoLA (MCC) — Reversal

| Rank | Config | MCC | Δ vs CE-only |
|------|--------|-----|--------------|
| 1 | 00_ce-only | **54.56%** | baseline |
| 2 | 01_frozen_base | 53.39% | -1.17% |
| 3 | 03_cotrain_base | 51.73% | -2.83% |
| 4 | 02_frozen_sft | 51.45% | -3.11% |
| 5 | 04_cotrain_sft | 49.67% | **-4.89%** |

**Finding:** CE-only wins! Distillation hurts on this small, difficult task. Co-training makes it worse.

---

## 4. Cross-Task Summary

| Task | Best Config | Worst Config | Frozen SFT Effect |
|------|-------------|--------------|-------------------|
| MNLI | cotrain_sft | frozen_sft | -1.81% (hurts) |
| QQP | cotrain_base | frozen_sft | **-34.51%** (catastrophic) |
| SST-2 | frozen_base / cotrain_base | - | 0% (neutral) |
| CoLA | ce-only | cotrain_sft | -3.11% (hurts) |

### Pattern Consistency

| Finding | MNLI | QQP | SST-2 | CoLA |
|---------|------|-----|-------|------|
| Frozen SFT hurts | ✓ | ✓✓✓ | - | ✓ |
| Co-train helps | ✓ | ✓ | - | ✗ |
| CE-only competitive | ✓ | ✓ | ✓ | **Best** |

---

## 5. Analysis

### The Overconfident-Wrong Teacher Problem

When a frozen fine-tuned teacher makes a confident wrong prediction:

| Gradient Source | Direction | Effect |
|-----------------|-----------|--------|
| CE Loss | → ground truth | Correct signal |
| KL Loss | → teacher's prediction | **Wrong signal** |

With α=0.5, half the gradient pushes toward wrong answer.

**Why QQP is catastrophic:**
- Binary classification = maximally wrong predictions (0% vs 100%)
- In 3-class MNLI, wrong predictions still assign some probability to correct class
- QQP's confident binary errors create irreconcilable gradient conflict

### Why Co-Training Helps

Self-correction mechanism:
1. Teacher makes confident wrong prediction
2. CE loss signals error to both models
3. Teacher updates, reducing confidence in wrong answer
4. Student receives progressively better guidance

### Why CoLA is Different

1. **Small dataset** (8.5k) — not enough data for distillation benefit
2. **Hard task** — linguistic acceptability is subtle
3. **MCC metric** — penalizes calibration errors harshly
4. **Teacher uncertainty** — BERT isn't dramatically better than DistilBERT on grammar

---

## 6. Practical Guidelines

### When to Use Frozen Teachers
- Teacher is well-calibrated on target distribution
- Task is similar to teacher's training data
- Compute budget is limited

### When to Use Co-Training
- Teacher may have distribution mismatch
- Teacher calibration is uncertain
- Binary classification tasks (higher risk)

### When to Skip Distillation
- Small datasets (< 10k examples)
- Task where teacher isn't significantly better
- Student already near ceiling

---

## 7. Conclusion

1. **Frozen fine-tuned teachers can catastrophically fail** — QQP shows -34% F1
2. **Co-training provides safety** — adaptive teachers self-correct errors
3. **Binary tasks are higher risk** — maximally wrong confident predictions
4. **Not all tasks benefit** — CoLA shows distillation can hurt
5. **Baselines matter** — CE-only is often competitive

Key insight: *A teacher's willingness to update matters more than its starting accuracy.*

---

## Appendix: Training Dynamics

### MNLI Loss Trends

| Config | Eval Loss Trend | Notes |
|--------|-----------------|-------|
| 00_ce-only | 0.57 → 0.55 ↘ | Stable |
| 01_frozen_base | 0.72 → 0.65 ↘ | Higher due to KL |
| 02_frozen_sft | 0.72 → 0.65 ↘ | Stable but low accuracy |
| 03_cotrain_base | 0.54 → 0.52 ↘ | Best trajectory |
| 04_cotrain_sft | 0.55 → 0.58 ↗ | Loss spike at end |

### QQP: Frozen SFT Failure Mode

| Metric | 02_frozen_sft | Others |
|--------|---------------|--------|
| Starting F1 | 51.96% | 86-88% |
| Final F1 | 55.89% | 90%+ |
| Starting Loss | 0.637 | 0.29-0.45 |
| Training Loss | Stuck at 0.55 | Decreases |

The student cannot escape the local minimum created by confident wrong teacher guidance.

---

## Data Sources

- W&B Project: `karthik-ravi-1537-personal/Knowledge Distillation`
- Raw data: [classification_ablations_2024-02-04.json](../../analysis/data/classification_ablations_2024-02-04.json)
- Analysis: [classification_ablations.md](../../analysis/summaries/classification_ablations.md)
