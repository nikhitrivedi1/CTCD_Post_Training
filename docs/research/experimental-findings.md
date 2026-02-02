# Experimental Findings

Observations and insights discovered during knowledge distillation experiments.

---

## Base Model Accuracies

### MNLI (validation_matched, n=9815)

| Model | Role | Accuracy | Notes |
|-------|------|----------|-------|
| distilbert-base-uncased | Student | **32.11%** | Random classifier init |
| bert-base-uncased | Teacher (base) | **36.07%** | Random classifier init |
| textattack/bert-base-uncased-MNLI | Teacher (SFT) | **84.57%** | Fine-tuned on MNLI |

### QQP (validation, n=40430)

| Model | Role | Accuracy | Notes |
|-------|------|----------|-------|
| distilbert-base-uncased | Student | **36.82%** | Random classifier init |
| bert-base-uncased | Teacher (base) | **37.68%** | Random classifier init |
| textattack/bert-base-uncased-QQP | Teacher (SFT) | **90.91%** | Fine-tuned on QQP |

---

## Final Ablation Results

### MNLI

| Rank | Config | Val Accuracy | Description |
|------|--------|--------------|-------------|
| 1 | 04 | **82.62%** | Co-train SFT teacher |
| 2 | 03 | 82.06% | Co-train base teacher |
| 3 | 01 | 81.25% | Frozen base teacher |
| 4 | 00 | 80.94% | No teacher (CE only) |
| 5 | 02 | 79.26% | Frozen SFT teacher |

### QQP

| Rank | Config | Val Accuracy | Description |
|------|--------|--------------|-------------|
| 1 | 03 | **90.37%** | Co-train base teacher |
| 2 | 04 | 90.28% | Co-train SFT teacher |
| 3 | 01 | 90.19% | Frozen base teacher |
| 4 | 00 | 89.71% | No teacher (CE only) |
| 5 | 02 | 65.50% | Frozen SFT teacher |

---

## The Overconfident-Wrong Teacher Paradox

### The Paradox

**Why does an 84.5% accurate teacher produce worse results than no teacher?**

### Explanation: Accuracy ≠ Distillation Quality

A fine-tuned teacher (SFT) is typically:
- **Highly confident when correct** (84.5% of samples) → helpful
- **Highly confident when wrong** (15.5% of samples) → harmful

### Untrained vs Trained-but-Wrong

```
Untrained Teacher (Config 01):
  When wrong (67% of cases):
    CE:  "Move toward label A" (confident)
    KL:  "Move... somewhere" (low confidence)
    Net: Mostly correct direction, slight noise

Trained Teacher (Config 02):
  When wrong (15.5% of cases):
    CE:  "Move toward label A" (confident)
    KL:  "Move toward label B!" (high confidence)
    Net: Gradients fight → confused learning
```

**Key insight:** Random noise is less harmful than confident misinformation.

---

## Label Mapping Verification

### QQP: Labels ALIGNED ✓

No `teacher_label_map` needed.

### MNLI: Labels MISALIGNED ✗

| Index | TextAttack | HuggingFace |
|-------|------------|-------------|
| 0 | contradiction | entailment |
| 1 | entailment | neutral |
| 2 | neutral | contradiction |

**Required mapping:** `teacher_label_map: [1, 2, 0]`

---

## Cross-Task Consistency

Both MNLI and QQP show the same pattern:

| Observation | MNLI | QQP |
|-------------|------|-----|
| Best config | Co-train (03/04) | Co-train (03/04) |
| Worst config | Frozen SFT (02) | Frozen SFT (02) |
| CE-only baseline | Competitive | Competitive |
| Frozen base > Frozen SFT | ✓ | ✓ |

**Universal findings:**
1. Co-training > Frozen for distillation
2. Untrained teacher > Trained teacher when frozen
3. CE-only baseline is hard to beat with frozen distillation
4. Overconfident wrong predictions hurt more than uncertain noise

---

## Summary: Ablation Ranking Explained

| Rank | Config | Why This Position |
|------|--------|-------------------|
| 1st | 04 | Best of both: task knowledge + adaptation |
| 2nd | 03 | Teacher learns alongside student; no prior baggage |
| 3rd | 01 | Noise doesn't hurt; slight regularization benefit |
| 4th | 00 | Clean baseline; no interference |
| 5th | 02 | Confident wrong predictions interfere with learning |

### Key Takeaways

1. **Co-training > Frozen** for robustness
2. **Untrained > Trained-misaligned** for frozen distillation
3. **Balanced α > Extreme α** for stable learning
4. **Ground truth is essential** when teacher quality is uncertain

---

## RFA Factorial Study

Testing Response (R), Feature (F), Attention (A) combinations on MNLI with Adaptive Domain-Expert.

### Setup

| Parameter | Value |
|-----------|-------|
| Teacher | textattack/bert-base-uncased-MNLI (SFT) |
| Freeze Teacher | No (Adaptive Domain-Expert) |
| Alpha | 0.5 |
| Temperature | 2.0 |

### Results

| Config | R | F | A | Accuracy | Rank |
|--------|---|---|---|----------|------|
| rfa-01 | 1 | 0 | 0 | **81.13%** | **1** |
| rfa-05 | 1 | 0 | 1 | 80.59% | 2 |
| rfa-04 | 1 | 1 | 0 | 80.49% | 3 |
| rfa-07 | 1 | 1 | 1 | 80.23% | 4 |
| rfa-02 | 0 | 1 | 0 | 79.42% | 5 |
| rfa-06 | 0 | 1 | 1 | 79.43% | 6 |
| rfa-03 | 0 | 0 | 1 | 78.69% | 7 |
| rfa-00 | 0 | 0 | 0 | 78.55% | 8 |

### Key Finding: R-Only Wins

**Classic KL distillation (R-only) outperforms all multi-signal combinations.**

Individual signal contribution (vs baseline):
- R: +2.58%
- F: +0.87%
- A: +0.14%

**Ranking: R >> F > A**

Adding F or A to R **hurts** performance due to gradient conflict:
- R → RF: -0.64%
- R → RA: -0.54%
- R → RFA: -0.90%

### Conclusion

Even with the best teacher setup (Adaptive Domain-Expert):
- **Use R-only (KL divergence)** - simpler and best performing
- Feature and Attention signals cause gradient conflict when combined
- Multi-signal distillation needs sophisticated weighting, not equal 1:1:1
