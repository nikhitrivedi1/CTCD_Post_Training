# Revisiting Knowledge Distillation: When Teachers Fail and Why Co-Training Wins

**Draft Paper Structure**

---

## Progress Tracker

### Level 1: Component Ablations
- [x] **00** No teacher baseline (α=0)
- [x] **01** Frozen untrained teacher
- [x] **02** Frozen SFT teacher
- [x] **03** Co-training untrained teacher
- [x] **04** Co-training SFT teacher
- [x] **05** KL-only (α=1.0)

**Status:** COMPLETE

**Key Finding:** `04 > 03 > 01 > 00 > 02 >>> 05`

### Level 2: Hyperparameter Tuning
- [ ] Learning rate sweep (1e-6 to 1e-4)
- [ ] Temperature sweep (T=1, 2, 4, 8)
- [ ] Alpha sweep (0.3, 0.5, 0.7)
- [ ] Batch size scaling

### Level 3: Loss Function Choices
- [ ] Forward KL (standard)
- [ ] Reverse KL (mode-seeking)
- [ ] MSE on logits
- [ ] Cosine similarity
- [ ] JS Divergence

### Level 4: Advanced Distillation
- [ ] Temperature annealing
- [ ] Alpha scheduling
- [ ] Intermediate layer distillation
- [ ] Attention transfer

---

## Abstract

Knowledge distillation (KD) transfers knowledge from large teacher models to smaller students. Standard practice freezes the teacher and trains the student to match soft labels. We present a systematic ablation study revealing that **frozen fine-tuned teachers can underperform no teacher at all**, while **co-training enables self-correction**. Our key insight: teacher accuracy measures correctness frequency, not distillation quality—overconfident wrong predictions harm learning more than random noise.

---

## 1. Introduction

### 1.1 Motivation
- Edge deployment requires smaller models
- Knowledge distillation promises "best of both worlds"
- But: prior work lacks rigorous ablation

### 1.2 Research Questions
1. Does distillation actually help over training without a teacher?
2. Does teacher quality always correlate with student quality?
3. When should teachers be frozen vs co-trained?

### 1.3 Contributions
1. Systematic 6-config ablation study isolating distillation components
2. Discovery of the "overconfident-wrong teacher paradox"
3. Evidence that co-training provides robustness to teacher miscalibration
4. Practical guidelines for distillation setup selection

---

## 2. Background

### 2.1 Knowledge Distillation
- Hinton et al. (2015) formulation
- Soft labels and temperature
- Loss function: `L = α × L_KL + (1-α) × L_CE`

### 2.2 Frozen vs Co-Training
- Traditional: freeze teacher, train student only
- Co-training: both update on same loss
- Prior work assumptions

### 2.3 Teacher Quality Assumptions
- Common belief: better teacher → better student
- Accuracy as proxy for teaching quality
- Gap in literature: no systematic test

---

## 3. Experimental Setup

### 3.1 Task and Data
- MNLI (3-class NLI)
- 392K training, 9.8K validation samples
- Standard HuggingFace splits

### 3.2 Models

| Role | Model | Parameters | Base MNLI Accuracy |
|------|-------|------------|-------------------|
| Teacher (base) | bert-base-uncased | 110M | 36.07% (random) |
| Teacher (SFT) | textattack/bert-base-uncased-MNLI | 110M | 84.57% |
| Student | distilbert-base-uncased | 66M | 32.11% (random) |

*Base accuracy = zero-shot evaluation before any training on MNLI.*

### 3.3 Training Configuration
- Epochs: 3
- Batch size: 16
- Learning rate: 2e-5
- Warmup: 200 steps
- LR schedule: linear decay
- FP16: enabled

### 3.4 Ablation Matrix

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

### 4.1 Main Results

| Rank | Config | Val Accuracy | Δ from Baseline | Description |
|------|--------|--------------|-----------------|-------------|
| 1 | 04 | **82.62%** | +50.51% | Co-train SFT |
| 2 | 03 | 82.06% | +49.95% | Co-train base |
| 3 | 01 | 81.25% | +49.14% | Frozen base |
| 4 | 00 | 80.94% | +48.83% | No teacher |
| 5 | 02 | 79.26% | +47.15% | Frozen SFT |
| 6 | 05 | 31.61% | -0.50% | KL only |

*Baseline = student base accuracy (32.11%). All trained for 3 epochs.*

### 4.2 The Paradox

**The 84% accurate teacher produces worse students than a 31% random teacher:**

| Comparison | Result |
|------------|--------|
| Frozen SFT (02) vs No teacher (00) | 79.26% < 80.94% (-1.68%) |
| Frozen SFT (02) vs Frozen base (01) | 79.26% < 81.25% (-1.99%) |
| Co-train SFT (04) vs Frozen SFT (02) | 82.62% > 79.26% (+3.36%) |

### 4.3 Reproducibility
- Deterministic training (seed=42, cudnn.deterministic=True)
- Identical results across runs for 00, 01, 03

---

## 5. Analysis

### 5.1 Why Frozen SFT Underperforms

**The Overconfident-Wrong Teacher Problem:**

| Teacher State | CE Gradient | KL Gradient | Result |
|---------------|-------------|-------------|--------|
| Correct + confident | → correct | → correct | Aligned |
| Wrong + confident | → correct | → **wrong** | Conflict |

When α=0.5:
- CE pushes toward ground truth
- KL pushes toward teacher's prediction
- Confident wrong predictions create gradient conflict

**Untrained teacher (high entropy):** Noise doesn't fight CE
**Trained teacher (low entropy):** Confident errors fight CE

### 5.2 Why Co-Training Wins

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

### 5.3 Why KL-Only Fails

Config 05 (α=1.0) with untrained teacher:
- No ground truth signal
- Student mimics random distributions
- Result: random-level accuracy

---

## 6. Discussion

### 6.1 Implications for Practitioners

**When to use frozen distillation:**
- Teacher is well-calibrated
- Teacher distribution matches target data
- Compute budget is limited

**When to use co-training:**
- Teacher may have distribution mismatch
- Teacher calibration is uncertain
- Robustness is prioritized over efficiency

### 6.2 Mitigations for Frozen Teachers

If frozen teacher is required:
- Higher temperature (T=4, 8) to soften overconfidence
- Lower α (0.3) to trust CE more
- Post-hoc calibration before distillation

### 6.3 Limitations
- Single task (MNLI)
- Single teacher/student architecture
- Three epochs only

---

## 7. Related Work

### 7.1 Knowledge Distillation
- Hinton et al. (2015) - Original formulation
- Sanh et al. (2019) - DistilBERT
- Jiao et al. (2019) - TinyBERT

### 7.2 Teacher Quality Studies
- [Papers examining teacher selection]
- [Papers on teacher calibration]

### 7.3 Co-Training in Distillation
- [Papers using joint optimization]
- [Online distillation approaches]

---

## 8. Conclusion

Standard knowledge distillation assumes frozen, accurate teachers produce better students. Our ablation study challenges this:

1. **Accuracy ≠ Distillation Quality:** An 84.5% teacher underperforms a 33% random one
2. **Overconfidence Harms:** Confident wrong predictions fight ground truth gradients
3. **Co-Training Self-Corrects:** Adaptive teachers outperform frozen ones
4. **Baselines Matter:** Without α=0 baseline, distillation's contribution is unmeasured

Key insight: *Random noise is less harmful than confident misinformation.*

---

## Appendix

### A. Training Curves
[W&B charts here]

### B. Hyperparameter Sensitivity
[Level 2 experiments]

### C. Loss Function Comparison
[Level 3 experiments]

### D. Reproducibility Details
- Seed: 42
- cudnn.deterministic: True
- GPU: NVIDIA A100 (40GB)
- Training time: ~X hours per config

---

## Future Work

1. **Multi-task validation:** QQP, SST-2, other GLUE tasks
2. **Architecture diversity:** Different teacher/student pairs
3. **Hyperparameter optimization:** Optuna sweep for optimal settings
4. **Loss function exploration:** Reverse KL, MSE, cosine similarity
5. **Temperature annealing:** Dynamic temperature scheduling
