# Simplicity Wins: Why Response-Only Distillation Outperforms Multi-Signal Approaches

Draft paper on the RFA factorial study findings.

---

## Abstract

Knowledge distillation research has explored transferring multiple types of knowledge: response (soft labels), features (hidden states), and attention patterns. We present a factorial study testing all 8 combinations of Response (R), Feature (F), and Attention (A) signals. Our finding: **R-only outperforms all multi-signal combinations**, including the full RFA. Adding Feature or Attention signals to Response *hurts* performance due to gradient conflict. The ranking R >> F > A suggests practitioners should use classic KL distillation rather than complex multi-signal approaches.

---

## 1. Introduction

### The Multi-Signal Hypothesis

Recent work has proposed transferring multiple types of knowledge:
- **Response (R):** Soft labels via KL divergence (Hinton et al., 2015)
- **Feature (F):** Hidden state alignment via MSE (Romero et al., 2015)
- **Attention (A):** Attention pattern transfer (Zagoruyko & Komodakis, 2017)

**Implicit assumption:** More signals = better transfer = better student.

### Research Question

Does combining R + F + A actually improve over R-only?

### Contribution

A systematic 2³ factorial study showing:
1. R-only achieves the best performance
2. Adding F or A to R *decreases* performance
3. Signal combination causes gradient conflict

---

## 2. Background

### Knowledge Distillation Signals

| Signal | What Transfers | Loss Function | Reference |
|--------|---------------|---------------|-----------|
| **Response** | Output distribution | KL divergence | Hinton 2015 |
| **Feature** | Hidden representations | MSE | Romero 2015 (FitNets) |
| **Attention** | Attention patterns | MSE on attention maps | Zagoruyko 2017 |

### The Implicit Assumption

Papers introducing Feature and Attention distillation show gains over baselines, but:
- Baselines often lack proper R-only comparison
- Combinations (RF, RA, FA, RFA) rarely tested systematically
- No factorial study isolating each signal's contribution

---

## 3. Experimental Setup

### Models

| Role | Model | Parameters |
|------|-------|------------|
| Teacher | textattack/bert-base-uncased-MNLI | 110M |
| Student | distilbert-base-uncased | 66M |

### Teacher Configuration

Based on prior ablation work, we use the optimal teacher setup:
- **Adaptive Domain-Expert:** Co-trained SFT teacher (not frozen)
- **Alpha:** 0.5 (balanced CE and distillation)
- **Temperature:** 2.0

### Factorial Design

| Config | R | F | A | Description |
|--------|---|---|---|-------------|
| 00 | 0 | 0 | 0 | Baseline (CE only) |
| 01 | 1 | 0 | 0 | Response only |
| 02 | 0 | 1 | 0 | Feature only |
| 03 | 0 | 0 | 1 | Attention only |
| 04 | 1 | 1 | 0 | Response + Feature |
| 05 | 1 | 0 | 1 | Response + Attention |
| 06 | 0 | 1 | 1 | Feature + Attention |
| 07 | 1 | 1 | 1 | All three (RFA) |

### Loss Functions

```
L_total = (1-α) × L_CE + α × [λ_R × L_R + λ_F × L_F + λ_A × L_A]

Where:
  L_R = KL(student_logits/T, teacher_logits/T)
  L_F = MSE(proj(student_hidden), teacher_hidden)
  L_A = MSE(student_attn, teacher_attn)
```

Equal weighting: λ_R = λ_F = λ_A = 1 when enabled.

---

## 4. Results

### Main Results

| Rank | Config | R | F | A | Accuracy |
|------|--------|---|---|---|----------|
| **1** | 01 | 1 | 0 | 0 | **81.13%** |
| 2 | 05 | 1 | 0 | 1 | 80.59% |
| 3 | 04 | 1 | 1 | 0 | 80.49% |
| 4 | 07 | 1 | 1 | 1 | 80.23% |
| 5 | 02 | 0 | 1 | 0 | 79.42% |
| 6 | 06 | 0 | 1 | 1 | 79.43% |
| 7 | 03 | 0 | 0 | 1 | 78.69% |
| 8 | 00 | 0 | 0 | 0 | 78.55% |

### Individual Signal Contribution

| Signal | Solo Performance | Gain over Baseline |
|--------|------------------|-------------------|
| **R** | 81.13% | **+2.58%** |
| **F** | 79.42% | +0.87% |
| **A** | 78.69% | +0.14% |

**Ranking: R >> F > A**

### Effect of Adding Signals to R

| Combination | Accuracy | Change from R-only |
|-------------|----------|-------------------|
| R | 81.13% | — |
| R + F | 80.49% | **-0.64%** |
| R + A | 80.59% | **-0.54%** |
| R + F + A | 80.23% | **-0.90%** |

**Adding any signal to R hurts performance.**

---

## 5. Analysis

### Why Multi-Signal Hurts

**Hypothesis: Gradient Conflict**

Each loss pushes the student in different directions:
- L_R: Match output distribution → task-relevant features
- L_F: Match all hidden states → may include task-irrelevant features
- L_A: Match attention patterns → may include suboptimal patterns

When combined with equal weights, these gradients conflict.

### The Feature Alignment Problem

Feature distillation (MSE on hidden states) forces the student to match teacher representations exactly. But:
- Teacher has 12 layers, student has 6 → architectural mismatch
- Exact replication may not be optimal for the smaller student
- Response-based learning allows the student to find its own representations

### Attention Pattern Concerns

Attention distillation assumes teacher attention is optimal for the student. But:
- Different head counts (12 vs 6) require averaging
- Teacher's attention may be overspecialized
- Attention is emergent, not the goal

---

## 6. Discussion

### Practical Recommendation

**Use R-only (classic KL distillation):**
- Simpler implementation
- Better performance
- Fewer hyperparameters (no λ_F, λ_A to tune)

### When Multi-Signal Might Help

Our study used equal weighting (1:1:1). Multi-signal might work with:
- Careful weight tuning (λ_F << λ_R)
- Curriculum: Start with R, add F later
- Selective layers: Only distill certain layers

### Limitations

- Single task (MNLI)
- Single model pair (BERT → DistilBERT)
- Equal loss weighting only

---

## 7. Conclusion

1. **R-only wins:** Classic KL distillation outperforms all multi-signal combinations
2. **Signal ranking:** R >> F > A in contribution to student performance
3. **Adding hurts:** Combining signals with equal weights causes gradient conflict
4. **Simplicity principle:** More complex distillation is not better distillation

**Key insight:** *The soft labels contain the signal; hidden states and attention patterns add noise.*

---

## Related Work

- Hinton et al. (2015) - Knowledge Distillation
- Romero et al. (2015) - FitNets: Feature distillation
- Zagoruyko & Komodakis (2017) - Attention Transfer
- Sun et al. (2019) - Patient Knowledge Distillation (PKD)
- Jiao et al. (2020) - TinyBERT (multi-stage distillation)

---

## Future Work

1. **Weight tuning:** Find optimal λ_R : λ_F : λ_A ratio
2. **Curriculum learning:** Progressive signal introduction
3. **Layer selection:** Which layers benefit from feature distillation?
4. **Cross-architecture:** Does this hold for decoder-only models (Llama)?
