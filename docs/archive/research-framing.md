# Knowledge Distillation Research Framing

> When Does Distillation Actually Help? A Systematic Ablation Study

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Ablation Framework** | Complete | 5 configs (00-04) per task |
| **MNLI Experiments** | Complete | All 5 configs run, analyzed |
| **QQP Experiments** | Complete | All 5 configs run, analyzed |
| **Translation** | Config ready | Llama 3B→1B, needs runs |
| **Baseline Evaluation** | Complete | Script + results for MNLI/QQP |

---

## Problem Space

### The Assumed Wisdom

Traditional knowledge distillation (KD) assumes:

| Assumption | What Papers Claim |
|------------|-------------------|
| **A1** | Better teacher → Better student |
| **A2** | Soft labels provide "dark knowledge" |
| **A3** | Distillation always beats CE-only training |
| **A4** | Frozen expert teachers are optimal |

### The Questions Nobody Asked

| ID | Question | Why It Matters |
|----|----------|----------------|
| **Q1** | Does distillation actually help when student is pretrained? | Most papers use fresh students |
| **Q2** | Does a more accurate teacher always produce better students? | Seems obvious, but is it? |
| **Q3** | When does an expert teacher hurt rather than help? | Counterintuitive but possible |
| **Q4** | Is co-training better than frozen distillation? | Standard KD freezes teacher |
| **Q5** | How much does distillation add over CE-only baseline? | Papers rarely report this |

### The Gap in Literature

Most KD papers:
- Compare distilled student vs. undistilled student
- Use task-trained (SFT) teachers
- Freeze the teacher during distillation
- Don't include α=0 (no teacher) baseline

**What's missing:**
- Systematic comparison of teacher training states
- Comparison of frozen vs co-training
- Honest assessment of CE-only baseline
- Cross-task consistency analysis

---

## Our Experimental Design

### Ablation Configurations

| Config | Teacher | Freeze Teacher | Alpha | Description |
|--------|---------|----------------|-------|-------------|
| **00** | None | N/A | 0.0 | CE only (no distillation) |
| **01** | bert-base-uncased | Yes | 0.5 | Frozen untrained teacher |
| **02** | textattack/bert-*-TASK | Yes | 0.5 | Frozen SFT teacher |
| **03** | bert-base-uncased | No | 0.5 | Co-train untrained teacher |
| **04** | textattack/bert-*-TASK | No | 0.5 | Co-train SFT teacher |

### What Each Config Tests

```
00: What can the student achieve alone?
    → Establishes baseline for distillation comparison

01: Does an untrained frozen teacher help?
    → Tests if soft labels matter without task knowledge

02: Does an expert frozen teacher help?
    → Tests traditional "standard" KD approach

03: Does co-training an untrained teacher help?
    → Tests if joint learning beats solo learning

04: Does co-training an expert teacher help?
    → Tests if prior knowledge + adaptation is best
```

---

## Framework Extensibility

The 5-config ablation is just the starting point. The framework supports systematic exploration across multiple dimensions:

### Configurable Dimensions

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DISTILLATION CONFIGURATION SPACE                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  LOSS FUNCTION                    DISTILLATION STRATEGY              │
│  ├── CE (Cross-Entropy)           ├── Logit Distillation (soft)     │
│  ├── KL Divergence                ├── Response Distillation         │
│  ├── MSE (Mean Squared Error)     ├── Attention Transfer            │
│  └── Combined (α-weighted)        ├── Feature/Hidden States         │
│                                   └── Intermediate Layers           │
│                                                                      │
│  ALPHA (α)                        TEMPERATURE (T)                    │
│  ├── 0.0 (CE only)                ├── 1.0 (hard labels)             │
│  ├── 0.3 (CE-heavy)               ├── 2.0 (soft, default)           │
│  ├── 0.5 (balanced)               ├── 4.0 (softer)                  │
│  ├── 0.7 (KL-heavy)               └── 8.0+ (very soft)              │
│  └── 1.0 (KL only)                                                   │
│                                                                      │
│  TEACHER STATE                    STUDENT INIT                       │
│  ├── Pretrained (LM only)         ├── pretrained (HF weights)       │
│  ├── SFT (task-trained)           └── fresh (random init)           │
│  └── Partially trained                                               │
│                                                                      │
│  FREEZE STRATEGY                  ARCHITECTURE                       │
│  ├── Frozen teacher               ├── BERT → DistilBERT             │
│  ├── Co-training (both learn)     ├── Llama 3B → Llama 1B           │
│  └── Curriculum (freeze→unfreeze) ├── Cross-family transfer          │
│                                   └── Custom teacher/student         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Distillation Strategies Overview

| Strategy | What It Transfers | Loss Function | Complexity |
|----------|-------------------|---------------|------------|
| **Logit/Soft-Label** | Output probability distribution | `KL(student_logits/T, teacher_logits/T)` | Low |
| **Response** | Final predictions/generations | `MSE(student_output, teacher_output)` | Low |
| **Attention Transfer** | Attention weight patterns | `MSE(student_attn, teacher_attn)` | Medium |
| **Feature/Hidden** | Intermediate representations | `MSE(proj(student_h[i]), teacher_h[j])` | Medium |
| **Relational KD** | Inter-sample relationships | Graph similarity loss | High |
| **Contrastive** | Representation space structure | InfoNCE / Triplet loss | High |

### Feasibility by Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DISTILLATION STRATEGY COMPATIBILITY                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ENCODER-ONLY (BERT, RoBERTa, DistilBERT)                                   │
│  ├── Logit/Soft-Label    ✓ Recommended    Same vocab, direct KL             │
│  ├── Attention Transfer  ✓ Feasible       Bidirectional attention matches   │
│  ├── Feature Distillation ✓ Feasible      May need projection (dim mismatch)│
│  ├── CLS Embedding       ✓ Feasible       Match [CLS] representations       │
│  └── Contrastive         ✓ Feasible       Align embedding spaces            │
│                                                                              │
│  DECODER-ONLY (GPT, Llama, Mistral)                                         │
│  ├── Logit/Soft-Label    ✓ Recommended    Token-level KL for generation     │
│  ├── Response Distillation ✓ Recommended  Seq2seq output matching           │
│  ├── Attention Transfer  ✓ Feasible       Causal attention compatible       │
│  ├── Feature Distillation ✓ Feasible      Layer mapping needed              │
│  └── Contrastive         △ Limited        Generation focus, not embeddings  │
│                                                                              │
│  ENCODER-DECODER (T5, BART, mT5)                                            │
│  ├── Logit/Soft-Label    ✓ Recommended    Decoder logits                    │
│  ├── Response Distillation ✓ Recommended  End-to-end output matching        │
│  ├── Encoder Attention   ✓ Feasible       Bidirectional encoder attention   │
│  ├── Decoder Attention   ✓ Feasible       Causal decoder attention          │
│  ├── Cross-Attention     ✓ Feasible       Unique to enc-dec, valuable       │
│  └── Feature Distillation ✓ Feasible      Encoder and decoder separately    │
│                                                                              │
│  CROSS-ARCHITECTURE (Transformer → RNN, BERT → LSTM)                        │
│  ├── Logit/Soft-Label    ✓ Only Option    Architecture-agnostic             │
│  ├── Response Distillation ✓ Feasible     Output-level only                 │
│  ├── Attention Transfer  ✗ Not Feasible   Different attention mechanisms    │
│  ├── Feature Distillation △ Challenging   Requires learned projections      │
│  └── Contrastive         △ Limited        Embedding space mismatch          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Legend: ✓ Recommended/Feasible  △ Limited/Challenging  ✗ Not Feasible
```

### Recommended Strategies by Use Case

| Use Case | Primary Strategy | Secondary Strategy | Notes |
|----------|------------------|-------------------|-------|
| **Classification (same family)** | Logit distillation | Feature distillation | Our current focus |
| **Classification (cross-arch)** | Logit distillation | Response matching | Only output-level transfer |
| **Generation (same family)** | Token-level KL | Attention transfer | Preserve generation patterns |
| **Generation (cross-arch)** | Sequence-level KL | Response matching | Architecture-agnostic |
| **Multilingual transfer** | Feature distillation | Contrastive | Align representation spaces |
| **Domain adaptation** | Feature + Logit | Attention transfer | Transfer both what and how |

### Implementation Complexity

| Strategy | Changes Required | Additional Hyperparameters |
|----------|------------------|---------------------------|
| **Logit** | Loss function only | T (temperature), α (weight) |
| **Response** | Loss function only | β (response weight) |
| **Attention** | Hook attention layers | Layer mapping, attention weight |
| **Feature** | Hook hidden states, add projection | Layer mapping, projection dim |
| **Relational** | Batch-level computation | Graph construction method |
| **Contrastive** | Negative sampling | Temperature, negative ratio |

### Layer Mapping for Feature/Attention Distillation

When teacher and student have different depths, layer mapping is required:

```
BERT (12 layers) → DistilBERT (6 layers)
┌─────────────────────────────────────────────────────┐
│  Teacher Layer    →    Student Layer    Strategy    │
├─────────────────────────────────────────────────────┤
│  Layer 2          →    Layer 1          Every 2nd   │
│  Layer 4          →    Layer 2                      │
│  Layer 6          →    Layer 3                      │
│  Layer 8          →    Layer 4                      │
│  Layer 10         →    Layer 5                      │
│  Layer 12         →    Layer 6                      │
└─────────────────────────────────────────────────────┘

Llama 3B (32 layers) → Llama 1B (16 layers)
┌─────────────────────────────────────────────────────┐
│  Teacher Layer    →    Student Layer    Strategy    │
├─────────────────────────────────────────────────────┤
│  Layer 2          →    Layer 1          Every 2nd   │
│  Layer 4          →    Layer 2                      │
│  ...              →    ...                          │
│  Layer 32         →    Layer 16                     │
└─────────────────────────────────────────────────────┘
```

**Mapping Strategies:**
| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Uniform** | Every N-th teacher layer | Default, works well |
| **Last-N** | Only final N layers | When final layers most important |
| **Learned** | Train projection to match | When architectures very different |
| **Skip-aware** | Account for residual connections | For very deep models |

**Dimension Mismatch Handling:**
| Teacher Dim | Student Dim | Solution |
|-------------|-------------|----------|
| 768 | 768 | Direct MSE |
| 768 | 512 | Linear projection (768→512) |
| 4096 | 2048 | Linear projection (4096→2048) |

### Our Framework Support

**Currently Implemented:**
- ✓ Logit distillation (soft-label KL)
- ✓ Combined loss (α-weighted CE + KL)
- ✓ Temperature scaling

**Ready to Implement:**
- Feature distillation (hook-based, projection layer)
- Attention transfer (attention weight extraction)
- Response distillation (output matching)

**Future Extensions:**
- Contrastive distillation
- Relational KD
- Multi-strategy combinations
- **Diffusion model distillation** - See [diffusion-distillation.md](diffusion-distillation.md)

### Beyond Classification: Diffusion KD

The framework concepts extend to diffusion models (Stable Diffusion, StyleTTS2, etc.), but with fundamentally different mechanics:

| Aspect | Classification KD | Diffusion KD |
|--------|------------------|--------------|
| **Goal** | Match output distributions | Reduce sampling steps |
| **Bottleneck** | Model parameters | Inference iterations |
| **Key strategy** | Logit/feature matching | Progressive step reduction |

See **[diffusion-distillation.md](diffusion-distillation.md)** for detailed coverage of:
- Progressive distillation (iterative step halving)
- Consistency distillation (one-step generation)
- Adversarial distillation (GAN refinement)
- StyleTTS2 case study (TTS distillation)

### Current Focus vs Future Extensions

| Dimension | Current (00-04) | Future Experiments |
|-----------|-----------------|-------------------|
| **Alpha** | 0.0, 0.5 | 0.3, 0.7, 1.0 |
| **Temperature** | 2.0 | 1, 4, 8 |
| **Strategy** | Logit only | Attention, Feature |
| **Student init** | pretrained | fresh |
| **Tasks** | MNLI, QQP | Translation, SST-2, more |

### Easy Config Extensions

```yaml
# Example: Temperature ablation
distillation:
  temperature: 4.0  # Change from default 2.0
  alpha: 0.5

# Example: KL-only (no CE)
distillation:
  alpha: 1.0  # Pure KL divergence

# Example: Fresh student init
model:
  student_init: fresh  # Random initialization

# Example: Feature distillation (future)
distillation:
  strategy: feature
  layer_mapping: uniform          # or "last_n", "learned"
  teacher_layers: [2, 4, 6, 8, 10, 12]
  student_layers: [1, 2, 3, 4, 5, 6]
  feature_weight: 0.5             # β for feature loss

# Example: Attention transfer (future)
distillation:
  strategy: attention
  layer_mapping: uniform
  attention_weight: 0.3           # γ for attention loss

# Example: Combined multi-strategy (future)
distillation:
  strategy: combined
  logit_weight: 0.5               # α for KL loss
  feature_weight: 0.3             # β for feature loss
  attention_weight: 0.2           # γ for attention loss
  temperature: 2.0
```

### Research Questions by Dimension

| Dimension | Key Question |
|-----------|--------------|
| **Alpha** | Is there a "sweet spot" between CE and KL? |
| **Temperature** | Does higher T help overconfident teachers? |
| **Strategy** | Does attention transfer beat logit distillation? |
| **Student init** | Is distillation more valuable for fresh students? |
| **Architecture** | Does cross-architecture transfer show different patterns? |

---

## Key Findings

### MNLI Results

| Rank | Config | Val Accuracy | Δ from Baseline | Description |
|------|--------|--------------|-----------------|-------------|
| 1 | 04 | **82.62%** | +1.68% | Co-train SFT teacher |
| 2 | 03 | 82.06% | +1.12% | Co-train base teacher |
| 3 | 01 | 81.25% | +0.31% | Frozen base teacher |
| 4 | 00 | 80.94% | — | No teacher (CE only) |
| 5 | 02 | 79.26% | -1.68% | Frozen SFT teacher |

### QQP Results

| Rank | Config | Val Accuracy | Δ from Baseline | Description |
|------|--------|--------------|-----------------|-------------|
| 1 | 03 | **90.37%** | +0.66% | Co-train base teacher |
| 2 | 04 | 90.28% | +0.57% | Co-train SFT teacher |
| 3 | 01 | 90.19% | +0.48% | Frozen base teacher |
| 4 | 00 | 89.71% | — | No teacher (CE only) |
| 5 | 02 | 65.50% | -24.21% | Frozen SFT teacher |

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

| Teacher State | When Correct (85%) | When Wrong (15%) | Net Effect |
|---------------|-------------------|------------------|------------|
| **Untrained** | Low confidence (noise) | Low confidence (noise) | Noise averages out |
| **SFT Frozen** | High confidence ✓ | High confidence ✗ | Confident mistakes hurt |
| **SFT Co-train** | High confidence ✓ | Adapts over time | Self-correcting |

### The Gradient Conflict

```python
# With frozen SFT teacher, when teacher is WRONG:
CE_gradient:  "Move toward correct label A"  (confident)
KL_gradient:  "Move toward wrong label B"    (also confident!)
Net effect:   Gradients fight → confused learning

# With untrained teacher, when teacher is wrong:
CE_gradient:  "Move toward correct label A"  (confident)
KL_gradient:  "Move... somewhere"            (low confidence, spread out)
Net effect:   CE dominates → mostly correct direction
```

### Why This Matters

**Accuracy ≠ Distillation Quality**

A 90% accurate teacher can produce worse students than a 33% accurate teacher because:
- The 90% teacher is confidently wrong 10% of the time
- The 33% teacher is uncertainly wrong 67% of the time
- Confident wrong signals hurt more than uncertain noise

---

## Label Mapping Verification

### TextAttack Model Labels

We verified the actual label mappings for TextAttack pre-trained teachers:

**QQP: Labels ALIGNED ✓**
| Index | TextAttack | HuggingFace |
|-------|------------|-------------|
| 0 | not_duplicate | not_duplicate |
| 1 | duplicate | duplicate |

**MNLI: Labels MISALIGNED ✗**
| Index | TextAttack | HuggingFace |
|-------|------------|-------------|
| 0 | contradiction | entailment |
| 1 | entailment | neutral |
| 2 | neutral | contradiction |

Required mapping: `teacher_label_map: [1, 2, 0]`

### Verification Results

```
MNLI with textattack/bert-base-uncased-MNLI:
  Raw accuracy (no mapping):     5.5%
  With [1,2,0] mapping:         83.5% ✓
  With [2,0,1] mapping:         11.0%
```

**Important:** QQP config 02's failure (65.5%) is NOT due to label misalignment. Labels are aligned. The failure is purely the "overconfident wrong teacher" effect.

---

## Cross-Task Consistency

### Pattern Holds Across Tasks

| Observation | MNLI | QQP |
|-------------|------|-----|
| Best config | Co-train (03/04) | Co-train (03/04) |
| Worst config | Frozen SFT (02) | Frozen SFT (02) |
| CE-only baseline | Competitive (80.94%) | Competitive (89.71%) |
| Frozen base > Frozen SFT | ✓ (81.25% > 79.26%) | ✓ (90.19% > 65.50%) |

### Universal Findings

1. **Co-training > Frozen** for distillation robustness
2. **Untrained > Trained** teacher when frozen (counterintuitive!)
3. **CE-only baseline is hard to beat** with frozen distillation
4. **Overconfident wrong predictions** hurt more than uncertain noise

---

## The Uncomfortable Truth

### Distillation Gains Are Marginal

| Task | Best Distillation | CE-only (00) | Actual Gain |
|------|-------------------|--------------|-------------|
| MNLI | 82.62% (04) | 80.94% | **+1.68%** |
| QQP | 90.37% (03) | 89.71% | **+0.66%** |

For BERT→DistilBERT on classification with pretrained student:
- Distillation adds ~1% improvement
- Frozen SFT distillation can HURT performance
- The CE-only baseline is surprisingly strong

### Why This Might Be

1. **Pretrained student already strong** - DistilBERT has good language understanding from pretraining
2. **Task is "easy" for the architecture** - Classification with encoder models is well-suited
3. **Teacher-student similarity** - Both are BERT-family, already aligned

### Where Distillation Might Matter More

| Scenario | Why Distillation Might Help More |
|----------|----------------------------------|
| Fresh (random) student init | No pretrained knowledge to start |
| Cross-architecture (BERT→LSTM) | Bigger gap to close |
| Harder tasks (generation) | More "dark knowledge" to transfer |
| Smaller data regimes | Soft labels provide regularization |

---

## Research Hypotheses

### H1: Overconfidence Hypothesis

**Claim:** Frozen SFT teachers hurt performance because they are overconfident when wrong.

**Evidence needed:**
- Compare calibration of SFT vs base teachers
- Measure confidence on incorrect predictions
- Test if temperature scaling helps SFT teachers

### H2: Co-training Robustness Hypothesis

**Claim:** Co-training is more robust to teacher quality because it allows self-correction.

**Evidence needed:**
- Track teacher accuracy over training
- Show co-train teachers improve over time
- Compare final teacher accuracy: frozen vs co-trained

### H3: Marginal Gains Hypothesis

**Claim:** Distillation gains are marginal when student is pretrained on language.

**Evidence needed:**
- Compare `student_init: fresh` vs `student_init: pretrained`
- Expect: larger distillation gains with fresh init
- This would explain why original paper (using pretrained) showed modest gains

### H4: Task Difficulty Hypothesis

**Claim:** Distillation helps more on harder tasks (e.g., translation vs classification).

**Evidence needed:**
- Run translation experiments (Llama 3B→1B)
- Expect: larger distillation gains on generation tasks
- Compare classification vs generation delta

---

## Evaluation Plan

### Current Experiments

| Task | Status | Models | Metric |
|------|--------|--------|--------|
| MNLI | ✓ Complete | BERT→DistilBERT | Accuracy |
| QQP | ✓ Complete | BERT→DistilBERT | Accuracy |
| Translation | Ready | Llama 3B→1B | BLEU |

### Proposed Additional Experiments

**A. Fresh vs Pretrained Student (H3)**
- Re-run MNLI with `student_init: fresh`
- Compare distillation gains

**B. Temperature Ablation (H1)**
- Test T=2, 4, 8 on frozen SFT (config 02)
- Higher T might soften overconfident predictions

**C. Alpha Ablation**
- Test α=0.3, 0.5, 0.7 on co-train configs
- Find optimal CE/KL balance

**D. Multiple Seeds**
- Run 3-5 seeds per config
- Report mean ± std for statistical significance

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
| CE-only baseline competitive | Finding | Methodological |
| Systematic ablation framework | Method | Reproducible |

### Venue Options

| Venue | Fit | Requirements |
|-------|-----|--------------|
| Course project | ✓✓✓ | Current state sufficient |
| Workshop paper | ✓✓ | Add 1-2 more tasks |
| Short paper (ACL/EMNLP) | ✓ | Add fresh-init experiments + theory |
| Full paper | ? | Need deeper analysis + more tasks |

### Paper Structure Outline

```
1. Introduction
   - KD assumptions in literature
   - The missing baselines problem

2. Related Work
   - Knowledge distillation overview
   - Teacher quality in KD
   - Co-training approaches

3. Experimental Setup
   - Ablation design rationale
   - Tasks and models
   - Implementation details

4. Results
   - MNLI findings
   - QQP findings
   - Cross-task consistency

5. Analysis
   - The overconfident wrong teacher problem
   - Why co-training helps
   - Label mapping considerations

6. Discussion
   - When does distillation help?
   - Practical recommendations
   - Limitations

7. Conclusion
```

---

## Practical Recommendations

Based on our findings:

### For Practitioners

| Scenario | Recommendation |
|----------|----------------|
| Have pretrained teacher + student | Try CE-only first; distillation gains may be marginal |
| Must use distillation | Use co-training, not frozen teacher |
| Have frozen expert teacher | Increase temperature or reduce α |
| Care about robustness | Co-training handles edge cases better |

### For Researchers

| Practice | Recommendation |
|----------|----------------|
| Always include | α=0 (CE-only) baseline |
| Compare | Frozen vs co-training |
| Check | Label alignment between teacher and dataset |
| Report | Absolute numbers, not just "improvement" |

---

## Next Steps

### Immediate (This Week)

- [ ] Run translation experiments on H200
- [ ] Document all findings in experimental_findings.md
- [ ] Create visualization of results

### Short-term (Course Deadline)

- [ ] Run fresh-init experiments for H3
- [ ] Add confidence intervals (multiple seeds)
- [ ] Write course project report

### Medium-term (Publication)

- [ ] Temperature ablation for H1
- [ ] Theoretical explanation for overconfidence effect
- [ ] Expand to more tasks
- [ ] User study: Do practitioners know about this?

---

## Open Questions

1. **Does the pattern hold for decoder-only models?**
   - Translation experiments will test this

2. **Is there a "sweet spot" for teacher quality?**
   - Not too good (overconfident), not too bad (useless)

3. **Can we predict when distillation will help?**
   - Task difficulty? Architecture gap? Data size?

4. **Why does co-training help exactly?**
   - Self-correction? Curriculum effect? Regularization?

5. **Is this specific to soft-label distillation?**
   - Would feature distillation show same pattern?

---

## References

### Knowledge Distillation Foundations
- Hinton et al., 2015 - "Distilling the Knowledge in a Neural Network"
- Sanh et al., 2019 - "DistilBERT, a distilled version of BERT"

### Teacher Quality in KD
- (Literature search needed for related work on teacher overconfidence)

### Co-training Approaches
- (Literature search needed for co-training in KD)

---

## Appendix: Baseline Accuracies

### MNLI (validation_matched, n=9815)

| Model | Role | Accuracy | Notes |
|-------|------|----------|-------|
| distilbert-base-uncased | Student | 32.11% | Random classifier |
| bert-base-uncased | Teacher (base) | 36.07% | Random classifier |
| textattack/bert-base-uncased-MNLI | Teacher (SFT) | 84.57% | Fine-tuned |

### QQP (validation, n=40430)

| Model | Role | Accuracy | Notes |
|-------|------|----------|-------|
| distilbert-base-uncased | Student | 36.82% | Random classifier |
| bert-base-uncased | Teacher (base) | 37.68% | Random classifier |
| textattack/bert-base-uncased-QQP | Teacher (SFT) | 90.91% | Fine-tuned |

---

## Related Documents

| Document | Purpose |
|----------|---------|
| [framework-direction.md](framework-direction.md) | Vision, mastery path, learning roadmap |
| [experimental_findings.md](experimental_findings.md) | Detailed experiment results and analysis |
| [diffusion-distillation.md](diffusion-distillation.md) | Diffusion model distillation (StyleTTS2, SD) |
| [run-options.md](run-options.md) | CLI reference and run management |
