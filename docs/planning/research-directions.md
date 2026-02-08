# Research Directions: Knowledge Distillation

A systematic exploration of the knowledge distillation design space, combining mastery path with future research directions.

---

## Part 1: Mastery Path

### Competency Matrix

| Level | Knowledge | Skills | Output |
|-------|-----------|--------|--------|
| **Awareness** | "KD makes models smaller" | Copy-paste from tutorials | None |
| **Basic** | Soft labels, temperature | Run existing code | Blog post |
| **Applied** | Loss functions, architectures | Implement from paper | Internal tool |
| **Production** | Trade-offs, failure modes | Debug and optimize | Deployed system |
| **Expert** | Why things work/fail | Design new approaches | Published paper |
| **Top 1%** | Frontier + intuition | Advance the field | Influential work |

### Current Progress

| Phase | Status | Key Insight |
|-------|--------|-------------|
| **1: Foundation** | ✓ Complete | Frozen SFT teachers can catastrophically fail (82% confident when wrong) |
| **2: Breadth** | In Progress | R-only beats FAR combinations; gradient conflict hurts |
| **3: Depth** | Planned | Deep dive into failure modes, theoretical understanding |
| **4: Frontier** | Future | Novel contributions to the field |

---

## Part 2: Multi-Axis Framework

### X-Axis: Tasks

| Task | Models | Status | Notes |
|------|--------|--------|-------|
| **Classification** | BERT → DistilBERT | ✓ Complete | Encoder-only, logits over classes |
| **Translation** | Llama 3B → 1B | In Progress | Decoder-only, token-level KL |
| **Summarization** | BART → DistilBART | Planned | Encoder-decoder, sequence-level |
| **Code Generation** | CodeLlama → smaller | Future | Same as translation but code domain |
| **Diffusion (Steps)** | DDPM-1000 → DDPM-4 | In Progress | Same size, fewer steps (Progressive Distillation) |
| **Diffusion (Params)** | Large UNet → Small UNet | Planned | Smaller model, R/F/A on UNet outputs |
| **Voice (StyleTTS2)** | StyleTTS2 → compressed | Planned | Capstone: text encoder + diffusion + decoder KD |

### Y-Axis: Depth (What to Distill)

| Depth | What Transfers | Loss Type | Findings |
|-------|----------------|-----------|----------|
| **Response (R)** | Output logits | KL | ✓ **Best alone** - simpler is better |
| **Feature (F)** | Hidden states | Cosine | ✗ Gradient conflict with R (re-running with fixes) |
| **Attention (A)** | Attention weights | KL | ✗ Gradient conflict with R (re-running with fixes) |
| **Relational** | Inter-sample relationships | Graph matching | Unexplored |
| **Progressive** | Multi-step → fewer steps | Trajectory MSE | Diffusion-specific (in progress) |

### Z-Axis: Teacher Strategy

| Strategy | Description | Findings |
|----------|-------------|----------|
| **Frozen SFT** | Fine-tuned teacher, frozen | ⚠️ Catastrophic on imbalanced binary (QQP) |
| **Frozen Base** | Pretrained teacher, frozen | Safer but limited knowledge |
| **Co-train SFT** | Fine-tuned teacher, learning | ✓ Self-corrects confident mistakes |
| **Co-train Base** | Pretrained teacher, learning | ✓ Best overall for most tasks |

---

## Part 3: Research Directions

### Completed Studies

| Study | Finding | Paper |
|-------|---------|-------|
| **Teacher Ablation** | Co-training beats frozen; SFT dangerous on binary | Paper 1 |
| **FAR Factorial** | R-only > RF > FA > FAR; gradient conflict | Paper 2 |
| **Training Dynamics** | 82% confidence when wrong causes stuck learning | Analysis |

### Active Research Questions

#### 1. When Does Co-training Beat Frozen?

**Question:** Can we predict when co-training is necessary vs when frozen suffices?

**Hypothesis:** Co-training wins when:
- Task is binary classification
- Dataset is imbalanced
- Teacher is overconfident (ECE > 0.1)

**Experiment:** Measure Expected Calibration Error (ECE) before distillation, correlate with co-train vs frozen performance gap.

#### 2. Which Signals Matter for Which Tasks?

**Question:** Does the optimal distillation signal (R, F, A) depend on task type?

**Hypothesis:**
- Classification: R dominates (class relationships in logits)
- Translation: A might matter more (alignment information)
- Summarization: F might help (content selection in hidden states)

**Experiment:** Run FAR factorial on translation, compare signal importance to classification.

#### 3. Knowledge Transfer Efficiency

**Question:** What predicts high transfer efficiency?

**Metrics:**
- `teacher_advantage = teacher_pre - student_pre`
- `student_gain = student_post - student_pre`
- `efficiency = student_gain / teacher_advantage`

**Hypothesis:** Efficiency correlates with:
- Dataset size (larger → higher efficiency)
- Teacher calibration (better calibrated → higher efficiency)
- Class balance (more balanced → higher efficiency)

---

## Part 4: Natural Extensions

### Sequence-Level Distillation

**Current:** Token-level KL divergence (each token independently)

**Extension:** Sequence-level distillation
- Distill on full sequence likelihood, not token-by-token
- Better for generation quality, harder to optimize
- Related: SeqKD (Kim & Rush, 2016)

```python
# Token-level (current)
kl_loss = KL(student_logits[t], teacher_logits[t])  # per token

# Sequence-level (extension)
seq_loss = -log P_student(teacher_generated_sequence)
```

**Tasks:** Translation, summarization, code generation

### Progressive Distillation

**Idea:** Chain of compressions: Large → Medium → Small

```
Llama 70B → Llama 8B → Llama 3B → Llama 1B
```

**Why it might work:**
- Smaller capacity gap at each step
- Each teacher is closer to student capability
- Related: Teacher Assistant (Mirzadeh et al., 2020)

**Experiment:** Compare direct 3B→1B vs 3B→2B→1B

### Self-Distillation

**Idea:** Train model on its own soft labels

```python
# Generate soft labels from current model
with torch.no_grad():
    soft_labels = model(x)

# Train on soft labels
loss = KL(model(x), soft_labels.detach())
```

**Why it works:** Born-Again Networks (Furlanello et al., 2018) show this improves even without compression.

**Experiment:** Add self-distillation phase after standard training

### Multi-Teacher Distillation

**Idea:** Ensemble of teachers, distill combined knowledge

```python
# Average teacher predictions (or learn weights)
teacher_ensemble = (teacher1(x) + teacher2(x) + teacher3(x)) / 3
loss = KL(student(x), teacher_ensemble)
```

**Variants:**
- Equal weighting
- Learned weighting per example
- Learned weighting per class

**Experiment:** Distill from 3 different BERT checkpoints → DistilBERT

---

## Part 5: Stretch Goals

### Diffusion Distillation (In Progress)

Two distinct problems:
1. **Step reduction:** Same model, fewer denoising steps (Progressive Distillation)
2. **Parameter reduction:** Smaller model, same steps (Standard KD)

**Key insight:** These require different approaches. Step reduction needs trajectory matching; parameter reduction uses standard feature/response KD.

**Approach:** Progressive Distillation on CIFAR-10 using pre-trained `google/ddpm-cifar10-32` (~35M params).

| Aspect | Text KD (current) | Diffusion KD |
|--------|-------------------|--------------|
| Signal | Output logits (KL) | Denoised samples (MSE) |
| Compression | Parameters (smaller model) | Steps (fewer iterations) |
| Training loop | HF Trainer subclass | Custom loop (not HF Trainer) |

**Key approaches:**
- **Progressive Distillation** (Salimans & Ho, 2022): Iteratively halve steps. Student matches teacher's 2-step output in 1 step. Requires v-prediction.
- **Consistency Models** (Song et al., 2023): Map any point on ODE trajectory to x_0 directly. Single training phase. 1-step generation by design.

**Estimated GPU time:** 8-16h total on A100 (including FID evaluation).

See [docs/planning/diffusion-distillation.md](diffusion-distillation.md) for full implementation plan.

### Adaptive Distillation

**Idea:** Adjust α (distillation weight) based on teacher confidence

```python
confidence = teacher_probs.max(dim=-1)
# Trust teacher less when overconfident
adaptive_alpha = alpha * (1 - (confidence - 0.9).clamp(0, 0.1) * 10)
loss = (1 - adaptive_alpha) * CE + adaptive_alpha * KL
```

**Motivation:** Our finding that 82% confidence when wrong causes problems

### Calibration-Aware Distillation

**Idea:** Pre-filter or reweight examples based on teacher calibration

```python
# Skip distillation on examples where teacher is miscalibrated
if teacher_confidence > 0.9 and teacher_wrong_historically:
    loss = CE_only  # Don't trust teacher here
else:
    loss = CE + alpha * KL
```

---

## Part 6: Priority Queue

### High Priority (Next Steps)

| # | Task | Rationale |
|---|------|-----------|
| 1 | Translation FAR factorial | Validate if R-only finding transfers |
| 2 | Pre/post metrics analysis | Understand transfer efficiency patterns |
| 3 | Co-train vs frozen predictor | Practical decision rule |

### Medium Priority

| # | Task | Rationale |
|---|------|-----------|
| 4 | Sequence-level distillation | Natural extension for translation |
| 5 | Self-distillation experiment | Low cost, potentially high impact |
| 6 | Temperature ablation | Understand softening effect |

### Low Priority (Stretch)

| # | Task | Rationale |
|---|------|-----------|
| 7 | Progressive distillation (chained) | Requires multiple model sizes |
| 8 | Multi-teacher ensemble | Requires training multiple teachers |
| 9 | Adaptive distillation | Confidence-based alpha scheduling |

### In Progress

| # | Task | Rationale |
|---|------|-----------|
| 10 | Diffusion distillation (CIFAR-10) | Third modality: step reduction via trajectory matching |

---

## Part 7: Reading List

### Foundation (Complete)
- Hinton et al., 2015 - "Distilling the Knowledge in a Neural Network"
- Sanh et al., 2019 - "DistilBERT"

### Breadth (Current)
- Romero et al., 2015 - "FitNets: Hints for Thin Deep Nets"
- Zagoruyko and Komodakis, 2017 - "Attention Transfer"
- Kim and Rush, 2016 - "Sequence-Level Knowledge Distillation"

### Depth (Next)
- Cho and Hariharan, 2019 - "On the Efficacy of Knowledge Distillation"
- Mirzadeh et al., 2020 - "Improved Knowledge Distillation via Teacher Assistant"
- Furlanello et al., 2018 - "Born Again Neural Networks"

### Frontier (Active)
- Ho et al., 2020 - "Denoising Diffusion Probabilistic Models" (DDPM background)
- Salimans and Ho, 2022 - "Progressive Distillation for Fast Sampling" (core method)
- Song et al., 2023 - "Consistency Models" (alternative approach)
- Zhou and Chen et al., 2024 - "Simple and Fast Distillation of Diffusion Models" (practical, 0.64h on A100)

---

## Success Metrics

| Phase | Technical | Knowledge | Output |
|-------|-----------|-----------|--------|
| **Current** | Run translation FAR | Explain signal importance | Course report |
| **Next** | Implement seq-level KD | Predict co-train vs frozen | Workshop paper |
| **Future** | Novel adaptive method | Theoretical grounding | Conference paper |
