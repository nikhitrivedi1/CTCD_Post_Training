# Revisiting Knowledge Distillation: When Teachers Fail and Why Co-Training Wins

Draft paper structure for the knowledge distillation research findings.

---

## Abstract

Knowledge distillation (KD) transfers knowledge from large teacher models to smaller students. Standard practice freezes the teacher and trains the student to match soft labels. We present a systematic ablation study revealing that **frozen fine-tuned teachers can underperform no teacher at all**, while **co-training enables self-correction**. Key insight: teacher accuracy measures correctness frequency, not distillation quality—overconfident wrong predictions harm learning more than random noise.

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

**The 84% accurate teacher produces worse students than a 31% random teacher:**

| Comparison | Result |
|------------|--------|
| Frozen SFT (02) vs No teacher (00) | 79.26% < 80.94% |
| Frozen SFT (02) vs Frozen base (01) | 79.26% < 81.25% |
| Co-train SFT (04) vs Frozen SFT (02) | 82.62% > 79.26% |

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

### Why Co-Training Wins

"Strong beliefs, loosely held"

| Config | Strong Beliefs | Loosely Held | Outcome |
|--------|---------------|--------------|---------|
| 02 (frozen SFT) | ✓ | ✗ | Dogmatic |
| 04 (co-train SFT) | ✓ | ✓ | Adaptive |

Co-training teachers self-correct:
1. Teacher makes confident wrong prediction
2. CE loss signals error to teacher
3. Teacher updates, soft labels improve
4. Student receives better guidance

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

1. **Accuracy ≠ Distillation Quality:** An 84.5% teacher underperforms a 33% random one
2. **Overconfidence Harms:** Confident wrong predictions fight ground truth gradients
3. **Co-Training Self-Corrects:** Adaptive teachers outperform frozen ones
4. **Baselines Matter:** Without α=0 baseline, distillation's contribution is unmeasured

Key insight: *Random noise is less harmful than confident misinformation.*

---

## Experiment Progress

### Level 1: Component Ablations
- [x] **00-05** Complete

### Level 2: Hyperparameter Tuning
- [ ] Learning rate sweep
- [ ] Temperature sweep
- [ ] Alpha sweep

### Level 3: Loss Function Choices
- [ ] Forward/Reverse KL
- [ ] MSE on logits
- [ ] Cosine similarity

### Level 4: Advanced Distillation
- [ ] Temperature annealing
- [ ] Intermediate layer distillation
- [ ] Attention transfer

---

## Future Work

1. **Multi-task validation:** QQP, SST-2, other GLUE tasks
2. **Architecture diversity:** Different teacher/student pairs
3. **Loss function exploration:** Reverse KL, MSE, cosine similarity
4. **Temperature annealing:** Dynamic temperature scheduling
