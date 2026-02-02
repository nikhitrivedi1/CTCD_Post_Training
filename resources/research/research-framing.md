# Knowledge Distillation Research Framing

> When Does Distillation Actually Help? A Systematic Ablation Study

---

## Problem Space

### The Assumed Wisdom

| Assumption | What Papers Claim |
|------------|-------------------|
| **A1** | Better teacher → Better student |
| **A2** | Soft labels provide "dark knowledge" |
| **A3** | Distillation always beats CE-only training |
| **A4** | Frozen expert teachers are optimal |

### The Questions Nobody Asked

| Question | Why It Matters |
|----------|----------------|
| Does distillation actually help when student is pretrained? | Most papers use fresh students |
| Does a more accurate teacher always produce better students? | Seems obvious, but is it? |
| When does an expert teacher hurt rather than help? | Counterintuitive but possible |

---

## Our Experimental Design

### Ablation Configurations

| Config | Teacher | Freeze | Alpha | Description |
|--------|---------|--------|-------|-------------|
| **00** | None | N/A | 0.0 | CE only (no distillation) |
| **01** | bert-base-uncased | Yes | 0.5 | Frozen untrained teacher |
| **02** | textattack/bert-*-TASK | Yes | 0.5 | Frozen SFT teacher |
| **03** | bert-base-uncased | No | 0.5 | Co-train untrained teacher |
| **04** | textattack/bert-*-TASK | No | 0.5 | Co-train SFT teacher |

---

## Key Findings

### MNLI Results

| Rank | Config | Val Accuracy | Description |
|------|--------|--------------|-------------|
| 1 | 04 | **82.62%** | Co-train SFT teacher |
| 2 | 03 | 82.06% | Co-train base teacher |
| 3 | 01 | 81.25% | Frozen base teacher |
| 4 | 00 | 80.94% | No teacher (CE only) |
| 5 | 02 | 79.26% | Frozen SFT teacher |

### QQP Results

| Rank | Config | Val Accuracy | Description |
|------|--------|--------------|-------------|
| 1 | 03 | **90.37%** | Co-train base teacher |
| 2 | 04 | 90.28% | Co-train SFT teacher |
| 3 | 01 | 90.19% | Frozen base teacher |
| 4 | 00 | 89.71% | No teacher (CE only) |
| 5 | 02 | 65.50% | Frozen SFT teacher |

### The Surprising Pattern

```
Both tasks show:
  Co-train (03/04) > Frozen base (01) > CE-only (00) > Frozen SFT (02)

Counter-intuitive findings:
  1. Frozen SFT teacher HURTS performance
  2. Untrained teacher beats trained teacher (when frozen)
  3. CE-only baseline is competitive
  4. Distillation gains are marginal (~1-2%)
```

---

## The Core Discovery: Overconfident Wrong Teachers

### Why Frozen SFT Teachers Fail

| Teacher State | When Correct | When Wrong | Net Effect |
|---------------|--------------|------------|------------|
| **Untrained** | Low confidence | Low confidence | Noise averages out |
| **SFT Frozen** | High confidence ✓ | High confidence ✗ | Confident mistakes hurt |
| **SFT Co-train** | High confidence ✓ | Adapts over time | Self-correcting |

### The Gradient Conflict

```python
# With frozen SFT teacher, when teacher is WRONG:
CE_gradient:  "Move toward correct label A"  (confident)
KL_gradient:  "Move toward wrong label B!"   (also confident!)
Net effect:   Gradients fight → confused learning

# With untrained teacher, when teacher is wrong:
CE_gradient:  "Move toward correct label A"  (confident)
KL_gradient:  "Move... somewhere"            (low confidence)
Net effect:   CE dominates → mostly correct direction
```

---

## Practical Recommendations

### For Practitioners

| Scenario | Recommendation |
|----------|----------------|
| Have pretrained teacher + student | Try CE-only first |
| Must use distillation | Use co-training, not frozen |
| Have frozen expert teacher | Increase temperature or reduce α |

### For Researchers

| Practice | Recommendation |
|----------|----------------|
| Always include | α=0 (CE-only) baseline |
| Compare | Frozen vs co-training |
| Check | Label alignment between teacher and dataset |
| Report | Absolute numbers, not just "improvement" |

---

## Paper Potential

### Strongest Angle

**Title:** "When Expert Teachers Hurt: Rethinking Frozen Knowledge Distillation"

**Key claim:** Frozen SFT teachers can hurt student performance due to overconfident wrong predictions. Co-training provides robustness through self-correction.

### Contribution Summary

| Contribution | Type | Novelty |
|--------------|------|---------|
| Frozen SFT teachers hurt | Finding | Counterintuitive, valuable |
| Untrained > trained when frozen | Finding | Challenges assumptions |
| Co-training > frozen | Finding | Practical guidance |
| Systematic ablation framework | Method | Reproducible |
