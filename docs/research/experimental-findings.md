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
