# Factors That Dictate Knowledge Distillation

A comprehensive analysis of the design space for knowledge distillation systems.

---

## Overview

Knowledge distillation is not a single technique but a **design space** with multiple interacting factors. Success depends on understanding how these factors interact.

```
                    ┌─────────────────────────────────────────┐
                    │         DISTILLATION OUTCOME            │
                    └─────────────────────────────────────────┘
                                       ▲
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
    ┌─────┴─────┐                ┌─────┴─────┐                ┌─────┴─────┐
    │   DATA    │                │   MODELS  │                │  TRAINING │
    └───────────┘                └───────────┘                └───────────┘
    - Task type                  - Teacher                    - L0: Teacher type
    - Sequence length            - Student                    - L1: Signal depth
    - Dataset size               - Architecture gap           - L2: Hyperparameters
    - Domain                     - Capacity ratio
```

---

## 1. Data Factors

### 1.1 Task Type

The task fundamentally changes what "knowledge" means and how it transfers.

| Task Type | Output Space | KL Complexity | Signal Density |
|-----------|--------------|---------------|----------------|
| **Classification** | Fixed (num_classes) | Low | High |
| **Translation** | Variable (SeqLen × Vocab) | Very High | Low |
| **Summarization** | Variable (SeqLen × Vocab) | Very High | Low |
| **Diffusion** | Continuous (Image dims) | Different | Medium |

**Implications:**
- Classification: Easy to match distributions (3-way softmax)
- Generation: Must match distributions at every token position
- The more output dimensions, the noisier the distillation signal

### 1.2 Sequence Length

Sequence length has **asymmetric effects** across tasks:

| Task | Input SeqLen Effect | Output SeqLen Effect | KL Scales With |
|------|---------------------|----------------------|----------------|
| Classification | Attention cost O(n²) | None (fixed output) | Nothing |
| Translation | Attention cost O(n²) | More KL terms | Output length |

**Key Insight:** For generation tasks, longer outputs mean more positions to align, diluting the distillation signal.

### 1.3 Dataset Size

| Size | Effect on Teacher | Effect on Distillation |
|------|-------------------|------------------------|
| Small (<10K) | May overfit | Student may memorize |
| Medium (10K-100K) | Good generalization | Sweet spot |
| Large (>100K) | Diminishing returns | Co-training shines |

**Our Setup:**
- MNLI: ~393K examples
- QQP: ~364K examples
- Europarl: 45,715 examples (subset)

### 1.4 Domain Complexity

| Domain | Characteristics | Distillation Challenge |
|--------|-----------------|------------------------|
| Formal text (Europarl) | Consistent style, long sentences | High redundancy |
| Social media | Noisy, short, varied | Hard to learn patterns |
| Technical | Domain-specific vocab | Requires specialization |

---

## 2. Teacher Model Factors

### 2.1 Teacher Expertise (SFT vs Base)

| Teacher Type | Task Knowledge | Soft Label Quality | Starting Accuracy |
|--------------|----------------|--------------------|--------------------|
| **Base (LM-pretrained)** | None | Random for task | ~33% (MNLI) |
| **SFT (Task fine-tuned)** | High | Informative | ~84% (MNLI) |

**Key Finding:** Starting expertise matters less than adaptability (see L0).

### 2.2 Teacher Adaptability (Frozen vs Co-training)

| Mode | Teacher Learns | Soft Labels | Gradient Flow |
|------|----------------|-------------|---------------|
| **Frozen** | No | Static | Student only |
| **Co-training** | Yes | Evolve with data | Both models |

**Key Finding:** "Strong Opinions, Loosely Held" - an adaptive teacher that refines its beliefs outperforms a stubborn expert.

### 2.3 The Four-Tier Teacher Taxonomy

| Tier | Expertise | Adaptability | Performance |
|------|-----------|--------------|-------------|
| **Adaptive Domain-Expert** | SFT | Co-train | Best |
| **Adaptive Teacher** | Base | Co-train | Good |
| **No Teacher** | N/A | N/A | Baseline |
| **Stubborn Domain-Expert** | SFT | Frozen | Worst |

**Ranking (MNLI):** Adaptive Domain-Expert > Adaptive Teacher > No Teacher >> Stubborn Domain-Expert

---

## 3. Student Model Factors

### 3.1 Student Initialization

| Init | Starting Point | Learning Dynamic |
|------|----------------|------------------|
| **Fresh (random)** | No knowledge | Must learn everything |
| **Pretrained** | Language understanding | Fine-tune for task |

**Our Choice:** Pretrained (matches successful MNLI setup)

### 3.2 Capacity Gap

The ratio of teacher-to-student capacity affects knowledge transfer:

| Gap | Example | Challenge |
|-----|---------|-----------|
| Small (1.5x) | BERT → DistilBERT (110M → 66M) | Easy transfer |
| Medium (3x) | Llama 3B → 1B | Requires compression |
| Large (10x+) | GPT-4 → Small model | May need intermediate |

**Architecture Mismatches:**

| Attribute | BERT→DistilBERT | Llama 3B→1B |
|-----------|-----------------|-------------|
| Layers | 12 → 6 (2:1) | 28 → 16 (1.75:1) |
| Hidden dim | 768 → 768 (same) | 3072 → 2048 (1.5:1) |
| Attention heads | 12 → 12 (same) | 24 → 8 (3:1) |

### 3.3 Parameter Efficiency (LoRA)

For large models, full fine-tuning is impractical:

| Method | Trainable Params | Memory | Quality |
|--------|------------------|--------|---------|
| Full fine-tune | 100% | High | Best |
| LoRA (r=16) | ~0.1% | Low | Near-best |
| Adapters | ~1% | Medium | Good |

**Our Setup:** LoRA for Llama (r=16, alpha=16, targets: down_proj, up_proj, gate_proj)

---

## 4. Training Factors: The L0/L1/L2 Hierarchy

### L0: Teacher Taxonomy (Structural)

**Question:** What kind of teacher-student relationship works best?

| Config | Teacher | Frozen | Tests |
|--------|---------|--------|-------|
| 00 | None | N/A | Baseline (CE only) |
| 01 | Base | Yes | Untrained frozen teacher |
| 02 | SFT | Yes | Expert frozen teacher |
| 03 | Base | No | Untrained co-training |
| 04 | SFT | No | Expert co-training |

**Finding:** Adaptability > Expertise. Config 04 wins.

**Ranking:** 04 (Adaptive Domain-Expert) > 00 (No Teacher) > 02 (Stubborn Domain-Expert)

**Critical insight:** A frozen expert (02) performs *worse* than no teacher at all (00). The expert's overconfident mistakes actively mislead the student.

**Cross-task validation:** This pattern replicates across both MNLI (3-class) and QQP (2-class). Since QQP has no label mapping complexity, this eliminates implementation bugs as an explanation. The "stubborn expert hurts" finding is robust.

**Contrast with DistilBERT:** The DistilBERT paper reports success with frozen teacher distillation during fine-tuning (SQuAD). However, they used a freshly fine-tuned teacher, not a pre-existing checkpoint. Our finding suggests teacher quality and calibration matter - a frozen teacher can hurt if its errors are systematic.

### L1: Signal Depth (What to Transfer)

**Question:** Which layers/signals should we distill?

| Signal | What Transfers | Loss | Complexity |
|--------|----------------|------|------------|
| **Response (R)** | Output logits | KL divergence | Low |
| **Feature (F)** | Hidden states | MSE | Medium |
| **Attention (A)** | Attention patterns | KL/MSE | Medium |

**RFA Factorial:**

| Config | R | F | A | Hypothesis |
|--------|---|---|---|------------|
| 00 | - | - | - | Baseline |
| 01 | ✓ | - | - | Classic KD |
| 02 | - | ✓ | - | Intermediate knowledge |
| 03 | - | - | ✓ | Attention transfer |
| 04 | ✓ | ✓ | - | R + F synergy? |
| 05 | ✓ | - | ✓ | R + A synergy? |
| 06 | - | ✓ | ✓ | F + A synergy? |
| 07 | ✓ | ✓ | ✓ | All signals |

**Finding (MNLI):** R-only wins. Adding F/A causes gradient conflict.

### L1.1: Pre-training vs Fine-tuning Distillation

DistilBERT (Sanh et al., 2019) uses a **triple loss** during pre-training:

| Loss | Signal Type | Purpose |
|------|-------------|---------|
| L_mlm | Task (MLM) | Masked language modeling |
| L_ce | Response (R) | KL divergence on logits |
| L_cos | Feature (F) | Cosine distance on hidden states |

**DistilBERT Ablation Findings:**
- Removing L_cos hurts performance
- Removing L_mlm has little impact
- "Student learns more from teacher than from training data"

**Contrast with Our Fine-tuning Results:**

| Aspect | DistilBERT (Pre-training) | Our Work (Fine-tuning) |
|--------|---------------------------|------------------------|
| Best signal combo | R + F | R only |
| Feature loss effect | Helpful | Harmful (gradient conflict) |
| Task loss effect | Minor | Important (CE anchors learning) |

**Why the difference?**

1. **Pre-training is general**: Hidden states encode broad linguistic knowledge worth transferring
2. **Fine-tuning is task-specific**: Hidden states may encode task-irrelevant info that conflicts with task objective
3. **Output space**: Pre-training has large vocab (30K+), fine-tuning has small labels (3 classes) - R signal is denser in fine-tuning

**Implication:** Signal depth (L1) findings may not transfer across distillation phases.

### L1.2: Staged Signal Distillation (Future Work)

DistilBERT's success with F during pre-training suggests **temporal staging** of signals:

| Stage | Signal | Rationale |
|-------|--------|-----------|
| Early | F (features) | Transfer structural/representational knowledge |
| Mid | A (attention) | Transfer reasoning patterns |
| Late | R (response) | Transfer task-specific decisions |

**Hypothesis:** The conflict we observe (R-only > R+F+A) may be because signals compete when applied simultaneously. Sequential staging could allow each signal to contribute without interference.

**Potential schedules:**
- **F → R**: Features early, response late (mirrors pre-train → fine-tune)
- **F → A → R**: Full progression from structure to decisions
- **Curriculum**: Weighted blend that shifts over training

**Why this might work:**
1. Early training: Student needs structural guidance (F/A)
2. Late training: Student needs decision refinement (R)
3. Simultaneous signals create gradient conflict; staging avoids this

**Status:** Not yet tested. Priority after L2 hyperparameter tuning.

### L2: Hyperparameters (How to Transfer)

**Question:** How do we optimize the winning configuration?

| Hyperparameter | Role | Values to Test |
|----------------|------|----------------|
| **Temperature** | Softness of distributions | 2, 5, 11, 20 |
| **Alpha** | CE vs KL balance | 0.25, 0.5, 0.75 |
| **Learning rate** | Optimization speed | 1e-5, 2e-5, 5e-5 |
| **Loss weights** | Multi-signal balance | Grid search |

**When to Tune:** Only after L0 and L1 establish WHAT works.

---

## 5. Interaction Effects

### 5.1 Task × Teacher Type

| Task | Best Teacher Type | Why |
|------|-------------------|-----|
| Classification | Adaptive Domain-Expert | Fixed output, clear signal |
| Translation | TBD (testing) | Variable output, noisy signal |

### 5.2 SeqLen × Signal Depth

| SeqLen | Response (R) | Feature (F) | Attention (A) |
|--------|--------------|-------------|---------------|
| Short | Clean signal | Manageable | Manageable |
| Long | Diluted | Memory intensive | O(n²) per layer |

### 5.3 Capacity Gap × Signal Depth

| Gap | Response | Feature | Attention |
|-----|----------|---------|-----------|
| Small | Easy | Easy | Easy |
| Large | Still works | Needs projection | Needs head mapping |

---

## 6. Our Experimental Design

### Phase 1: Classification (Complete)

```
L0: Teacher Taxonomy → Adaptive Domain-Expert wins
L1: RFA Factorial → R-only wins, F/A hurt
L2: (future) → Tune temperature, alpha
```

### Phase 2: Translation (In Progress)

```
L0: Teacher Taxonomy → Testing if Adaptive > Stubborn transfers
L1: RFA Factorial → Testing if R-only finding transfers
L2: (future) → Task-specific tuning
```

### Key Questions

1. **Does L0 finding transfer?** Adaptive > Stubborn for translation?
2. **Does L1 finding transfer?** R-only > R+F+A for translation?
3. **Are there task-specific effects?** Does SeqLen change optimal depth?

---

## 7. Summary: The Distillation Design Space

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DISTILLATION DESIGN SPACE                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  DATA                    MODELS                    TRAINING             │
│  ────                    ──────                    ────────             │
│  • Task type             • Teacher expertise       • L0: Teacher type   │
│  • Sequence length       • Teacher adaptability    • L1: Signal depth   │
│  • Dataset size          • Student init            • L2: Hyperparams    │
│  • Domain                • Capacity gap                                 │
│                          • Architecture match                           │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  KEY FINDINGS                                                           │
│  ────────────                                                           │
│  1. Adaptability > Expertise (Adaptive Domain-Expert wins)              │
│  2. Response-only > Multi-signal (F/A cause gradient conflict)          │
│  3. Pretrained student > Fresh init                                     │
│  4. Task affects optimal configuration (SeqLen matters for generation)  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## References

- Hinton et al. (2015): "Distilling the Knowledge in a Neural Network"
- Sanh et al. (2019): "DistilBERT, a distilled version of BERT"
- Jiao et al. (2019): "TinyBERT: Distilling BERT for NLU"
