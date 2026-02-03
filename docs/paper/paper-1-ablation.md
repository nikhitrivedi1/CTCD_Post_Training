# Revisiting Knowledge Distillation: When Teachers Fail and Why Co-Training Wins

Draft paper structure for the knowledge distillation research findings.

---

## Abstract

Knowledge distillation (KD) transfers knowledge from large teacher models to smaller students. Standard practice freezes the teacher and trains the student to match soft labels. We present a systematic ablation study revealing that **a Stubborn Domain-Expert teacher (84.5% accuracy) produces worse students than an Adaptive Teacher starting at 33%**. The key is adaptability: co-training enables self-correction. Following the principle of "Strong Opinions, Loosely Held," adaptive teachers outperform stubborn experts. Teacher accuracy measures correctness frequency, not distillation quality—overconfident wrong predictions harm learning more than uncertain guidance.

---

## 1. Introduction

### Research Questions
1. Does distillation actually help over training without a teacher?
2. Does teacher quality always correlate with student quality?
3. When should teachers be frozen vs co-trained?

### Contributions
1. Systematic 6-config ablation study isolating distillation components
2. Discovery of the "overconfident-wrong teacher paradox"
3. Evidence that co-training provides robustness to teacher miscalibration
4. Practical guidelines for distillation setup selection

---

## 2. Background

**Knowledge Distillation (Hinton et al., 2015)**
- Loss: `L = α × L_KL + (1-α) × L_CE`
- Soft labels via temperature-scaled softmax
- Traditional: freeze teacher, train student only

**Gap in Literature:**
- Common belief: better teacher → better student
- No systematic test of this assumption

---

## 3. Experimental Setup

### Models

| Role | Model | Parameters | Base MNLI Accuracy |
|------|-------|------------|-------------------|
| Teacher (base) | bert-base-uncased | 110M | 36.07% (random) |
| Teacher (SFT) | textattack/bert-base-uncased-MNLI | 110M | 84.57% |
| Student | distilbert-base-uncased | 66M | 32.11% (random) |

### Ablation Matrix

| Config | Teacher | Frozen | α | Description |
|--------|---------|--------|---|-------------|
| 00 | None | - | 0.0 | No distillation baseline |
| 01 | Base | ✓ | 0.5 | Frozen untrained |
| 02 | SFT | ✓ | 0.5 | Frozen fine-tuned |
| 03 | Base | ✗ | 0.5 | Co-train untrained |
| 04 | SFT | ✗ | 0.5 | Co-train fine-tuned |
| 05 | Base | ✓ | 1.0 | KL only (no CE) |

---

## 4. Results

### Main Results

| Rank | Config | Val Accuracy | Description |
|------|--------|--------------|-------------|
| 1 | 04 | **82.62%** | Co-train SFT |
| 2 | 03 | 82.06% | Co-train base |
| 3 | 01 | 81.25% | Frozen base |
| 4 | 00 | 80.94% | No teacher |
| 5 | 02 | 79.26% | Frozen SFT |
| 6 | 05 | 31.61% | KL only |

### The Paradox

**A Stubborn Domain-Expert produces worse students than an Adaptive Teacher with no prior knowledge:**

| Comparison | Result | Insight |
|------------|--------|---------|
| Stubborn Domain-Expert vs Adaptive Teacher | 79.26% < 82.06% | Adaptability beats expertise |
| Stubborn Domain-Expert vs No Teacher | 79.26% < 80.94% | Stubbornness hurts |
| Adaptive vs Stubborn Domain-Expert | 82.62% > 79.26% | Same knowledge, different mindset |

The critical variable is not initial accuracy—it's whether the teacher can update.

---

## 5. Analysis

### Why Frozen SFT Underperforms

**The Overconfident-Wrong Teacher Problem:**

| Teacher State | CE Gradient | KL Gradient | Result |
|---------------|-------------|-------------|--------|
| Correct + confident | → correct | → correct | Aligned |
| Wrong + confident | → correct | → **wrong** | Conflict |

- CE pushes toward ground truth
- KL pushes toward teacher's prediction
- Confident wrong predictions create gradient conflict

**Untrained teacher (high entropy):** Noise doesn't fight CE

**Trained teacher (low entropy):** Confident errors fight CE

### Why Co-Training Wins: Strong Opinions, Loosely Held

The principle of "Strong Opinions, Loosely Held" explains our results. We identify four teacher tiers:

| Tier | Config | Accuracy | Adaptive | Outcome |
|------|--------|----------|----------|---------|
| **Adaptive Domain-Expert** | 04 (co-train SFT) | 84.5% | Yes | **82.62%** |
| **Adaptive Teacher** | 03 (co-train base) | 33% | Yes | 82.06% |
| **No Teacher** | 00 (CE only) | - | - | 80.94% |
| **Stubborn Domain-Expert** | 02 (frozen SFT) | 84.5% | No | 79.26% |

**Stubborn Domain-Expert (02):** Worst outcome. Combines confidence with inflexibility—cannot update when wrong.

**Adaptive Teacher (03):** Starts uncertain but learns alongside the student. Initial 33% accuracy doesn't matter—adaptability does.

**Adaptive Domain-Expert (04):** Best outcome. Strong prior knowledge + willingness to update.

Co-training self-correction mechanism:
1. Teacher makes confident wrong prediction
2. CE loss signals error to both teacher and student
3. Teacher updates, soft labels improve
4. Student receives progressively better guidance

---

## 6. Discussion

### When to Use Frozen Distillation
- Teacher is well-calibrated
- Teacher distribution matches target data
- Compute budget is limited

### When to Use Co-Training
- Teacher may have distribution mismatch
- Teacher calibration is uncertain
- Robustness is prioritized over efficiency

### Mitigations for Frozen Teachers
- Higher temperature (T=4, 8) to soften overconfidence
- Lower α (0.3) to trust CE more
- Post-hoc calibration before distillation

---

## 7. Conclusion

1. **Adaptability > Accuracy:** A Stubborn Domain-Expert (84.5%) underperforms an Adaptive Teacher (33%)
2. **Strong Opinions, Loosely Held:** Adaptive teachers outperform stubborn experts
3. **Overconfidence Harms:** Confident wrong predictions fight ground truth gradients
4. **Same Knowledge, Different Outcome:** The same SFT teacher improves +3.4% when made adaptive
5. **Baselines Matter:** Without a No Teacher baseline, distillation's contribution is unmeasured

Key insight: *A teacher's willingness to learn matters more than its starting knowledge.*

---

## Cross-Task Validation: QQP

### QQP Results

| Rank | Config | Val Accuracy | Description |
|------|--------|--------------|-------------|
| 1 | 03 | **90.37%** | Co-train base |
| 2 | 04 | 90.28% | Co-train SFT |
| 3 | 01 | 90.19% | Frozen base |
| 4 | 00 | 89.71% | No teacher |
| 5 | 02 | 65.50% | Frozen SFT |

### Cross-Task Consistency

| Finding | MNLI | QQP | Consistent? |
|---------|------|-----|-------------|
| Co-train wins | ✓ | ✓ | **Yes** |
| Frozen SFT worst | ✓ | ✓ | **Yes** |
| CE-only competitive | ✓ | ✓ | **Yes** |
| Frozen base > Frozen SFT | ✓ | ✓ | **Yes** |

The pattern holds across both tasks, strengthening the generality of our findings.

---

## Experiment Status

- [x] MNLI ablations (00-04) - Complete
- [x] QQP ablations (00-04) - Complete
- [ ] Hyperparameter sensitivity (α, T sweeps)
- [ ] Additional GLUE tasks (SST-2, CoLA)

---

## Future Work

1. **More tasks:** SST-2, CoLA, RTE from GLUE
2. **Generation tasks:** Does this transfer to translation? (See Paper 3)
3. **Hyperparameter sensitivity:** How robust are findings to α and T?
4. **Mitigation strategies:** Can higher T or lower α rescue frozen SFT?

---

## Related Papers

- [Paper 2: RFA Study](paper-2-rfa.md) - Multi-signal distillation (R vs F vs A)
- [Paper 3: Translation](paper-3-translation.md) - Cross-task transfer to generation
