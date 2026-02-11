# Training Dynamics Analysis

**Date:** 2026-02-07
**Data Source:** W&B ablation runs (20 runs across MNLI, QQP, SST-2, CoLA)

---

## Executive Summary

| Finding | Evidence | Implication |
|---------|----------|-------------|
| Teacher 82% confident when wrong | 42/500 wrong, avg 82% confidence | Gradient conflict inevitable |
| QQP frozen SFT stuck from start | eval/loss never improves (0.64) | Early detection possible |
| CoLA co-training amplifies overfitting | eval/loss +13% for cotrain_sft | Small data = skip KD |
| eval/loss misleading | Low loss ≠ best metric on CoLA | Always check primary metric |

---

## Key Findings

### 1. 🚨 QQP Frozen SFT: The Catastrophic Case

The frozen SFT teacher on QQP shows **training failure mode** - the model gets stuck:

| Metric | Start | End | Change |
|--------|-------|-----|--------|
| train/loss | 0.688 | 0.508 | -26% (barely learns) |
| eval/loss | 0.637 | 0.645 | +1% (never improves!) |

Compare to QQP CE-only baseline:
| Metric | Start | End | Change |
|--------|-------|-----|--------|
| train/loss | 0.429 | 0.120 | -72% |
| eval/loss | 0.298 | 0.301 | ~stable |

**The smoking gun:** Eval loss is **stuck at 0.64** for frozen SFT while CE-only achieves 0.30.

**Why?** The confident-wrong teacher creates a floor the student can't escape. The KL loss component keeps pulling the student toward wrong predictions, preventing train loss from decreasing as fast.

#### Root Cause: Teacher Confidence Analysis

Analyzed teacher predictions on 500 QQP validation examples:

| Metric | Value |
|--------|-------|
| Overall accuracy | 91.6% ✓ |
| Wrong predictions | 42 (8.4%) |
| **Avg confidence when WRONG** | **82.1%** ⚠️ |
| >90% confident but wrong | 20 cases |
| >80% confident but wrong | 26 cases |

**Key insight:** The teacher is accurate but **dangerously overconfident when wrong**. Those 42 predictions with 82% average confidence create irreconcilable gradient conflict:

```
For each confident-wrong example:
  CE gradient:  → correct answer (weight: 0.5)
  KL gradient:  → wrong answer at 82% confidence (weight: 0.5)
  Net effect:   → stuck in the middle
```

This explains why eval/loss is stuck from the start—the student can never escape the teacher's confident mistakes.

---

### 2. ⚠️ CoLA: The Small Data Problem

CoLA shows a **split pattern** - co-training hurts while frozen helps with loss:

| Config | eval/loss Start | eval/loss End | Trend | MCC |
|--------|----------------|---------------|-------|-----|
| 00_ce-only | 0.524 | 0.556 | ↗️ +6% | **53.8%** ✓ |
| 01_frozen_base | 0.523 | 0.482 | ↘️ -8% | 53.4% |
| 02_frozen_sft | 0.579 | 0.516 | ↘️ -11% | 51.5% |
| 03_cotrain_base | 0.510 | 0.560 | ↗️ +10% | 51.7% |
| 04_cotrain_sft | 0.571 | 0.643 | ↗️ +13% | 49.7% ✗ |

**The paradox:** Frozen configs have LOWER eval/loss but WORSE MCC than CE-only!

**Why?**
1. **eval/loss includes KL term** for distillation configs → inflated loss doesn't mean worse performance
2. **Co-training amplifies overfitting** on small data → teacher and student reinforce each other's mistakes
3. **Frozen teachers regularize** but with noise → prevents overfitting but adds teacher's biases
4. **CE-only wins** because 8.5k examples isn't enough for meaningful knowledge transfer

---

### 3. 📊 Loss Component Breakdown

Co-training configs log `train/teacher_ce_loss` showing teacher learning:

#### MNLI Co-train SFT
| Metric | Start | End |
|--------|-------|-----|
| train/loss (student total) | 1.102 | 0.528 |
| train/teacher_ce_loss | 1.169 | 0.106 |

Teacher CE loss drops from 1.17 → 0.11 = **teacher is learning** alongside student.

#### QQP Co-train SFT
| Metric | Start | End |
|--------|-------|-----|
| train/loss (student) | 0.648 | 0.264 |
| train/teacher_ce_loss | 2.037 | 0.004 |

Teacher starts with HIGH loss (2.0!) but drops to near-zero. This self-correction saves the student from catastrophe.

**Note:** The high starting teacher_ce_loss (2.0) indicates the SFT teacher has room to improve. Cross-entropy of 2.0 on binary = ~13.5% accuracy, which seems wrong for a "fine-tuned" model. This might indicate:
1. Teacher evaluating on different data distribution than it was trained on
2. Batching effects in first logged step
3. 🔬 **Worth investigating:** What's the teacher's actual accuracy at step 0?

---

### 4. 📈 eval/loss vs Performance

**eval/loss doesn't always predict performance:**

| Task | Config | eval/loss | Primary Metric | Notes |
|------|--------|-----------|----------------|-------|
| MNLI | cotrain_base | 0.522 ✓ | 82.6% acc ✓ | Lowest loss = best |
| MNLI | cotrain_sft | 0.588 ⚠️ | **82.7%** acc ✓ | Higher loss, same perf |
| QQP | frozen_sft | **0.652** ✗ | 55.9% F1 ✗ | High loss = terrible |
| CoLA | frozen_base | 0.482 ✓ | 53.4% MCC ⚠️ | Lowest loss ≠ best |
| CoLA | ce-only | 0.556 ⚠️ | **53.8%** MCC ✓ | Higher loss = best! |

**Insight:** eval/loss = CE + α×KL, so distillation configs have inflated loss from the KL term even when performing well. CE-only configs have lower loss by definition (no KL term). **Always compare primary metrics, not loss.**

---

### 5. 🔄 Training Trajectories by Task

#### MNLI (Large, Balanced, 3-class)
- All configs converge smoothly
- Frozen SFT slightly worse but not catastrophic
- Co-training achieves lowest train/loss

#### QQP (Large, Imbalanced, Binary)
- Frozen SFT gets stuck immediately (loss plateau)
- Co-training rescues performance
- eval/loss gap between frozen_sft and others: 0.64 vs 0.30

#### SST-2 (Medium, Balanced, Binary, Easy)
- Very fast convergence (67k examples, easy task)
- All configs reach similar performance
- train/loss near zero for frozen_sft (student memorizes teacher)

#### CoLA (Small, Imbalanced, Binary, Hard)
- Rapid overfitting (eval/loss increases mid-training)
- Co-training amplifies overfitting
- Best to stop early or skip KD entirely

---

## Warning Signs to Watch

### 🚨 Catastrophic Failure (QQP frozen_sft pattern)
1. eval/loss doesn't decrease from start
2. train/loss decreases slowly (stuck above 0.5)
3. Large gap between train and eval loss
4. Final F1/accuracy much worse than CE-only baseline

### ⚠️ Overfitting (CoLA pattern)
1. eval/loss starts decreasing then increases
2. train/loss keeps decreasing (to near zero)
3. Co-training makes it worse, not better
4. Final metric worse than early checkpoint

### ✓ Healthy Training (MNLI cotrain pattern)
1. Both train and eval loss decrease
2. eval/loss stabilizes or slowly decreases
3. train/teacher_ce_loss decreases (teacher learning)
4. Final metric beats CE-only baseline

---

## Recommendations

### Before Training
1. **Check dataset size**: <10k = skip KD or use early stopping
2. **Check class balance**: Imbalanced binary = co-train ONLY
3. **Set up monitoring**: Track eval/loss AND primary metric

### During Training
1. **Watch eval/loss trend**: Should decrease or stabilize
2. **Watch train/loss vs eval/loss gap**: Large gap = overfitting
3. **For co-training**: Check teacher_ce_loss decreases

### After Training
1. **Compare to CE-only baseline**: KD should beat it
2. **Check if early stopping helps**: Overfitting may be present
3. **Verify metric matches expectation**: Low loss ≠ good metric always

---

## Next Investigations

### 1. Pre/Post Metric Logging

Add evaluation at start and end of training for both models:

| Metric Set | Purpose |
|------------|---------|
| Teacher (Pre) | Baseline quality, confidence distribution |
| Student (Pre) | Starting point before any training |
| Teacher (Post) | Did co-training improve/hurt teacher? |
| Student (Post) | Final performance (current logging) |

**Key metric:** Knowledge transfer efficiency = `(student_post - student_pre) / (teacher_pre - student_pre)`

### 2. Confidence-Weighted KL Loss

**Problem:** High confidence ≠ correctness. Teacher with 95% confidence might be wrong.

**Hypothesis:** Down-weight KL loss when teacher is very confident (precisely when it's most dangerous).

```python
# Current: KL weighted equally for all examples
kl_loss = alpha * KL(student, teacher)

# Proposed: Reduce weight when teacher is overconfident
confidence = teacher_probs.max(dim=-1)
weight = 1.0 - (confidence - 0.9).clamp(0, 0.1) * 10  # Reduce weight above 90%
kl_loss = alpha * weight * KL(student, teacher)
```

| Teacher Confidence | Weight | Effect |
|-------------------|--------|--------|
| <90% | 1.0 | Full KL signal (normal) |
| 95% | 0.5 | Half KL signal (reduced) |
| 100% | 0.0 | No KL signal (CE-only) |

**Intuition:** Trust teacher more when uncertain (soft labels valuable), trust less when overconfident (may be confidently wrong).

### 3. Early Stopping on eval/loss Trend

Current: Stop on best eval metric.
Proposed: Also stop if eval/loss hasn't improved for N steps from the START.

This would catch the QQP frozen SFT pattern early (eval/loss stuck at 0.64 from step 1).

---

## Raw Data

Training histories saved to: `../data/ablation_history.json`

### Key Metric Columns
- `train/loss`: Student's total loss (CE + α×KL)
- `train/loss_total`: Same as above (naming inconsistency)
- `train/teacher_ce_loss`: Teacher's CE loss (co-train only)
- `eval/loss`: Evaluation loss on validation set
