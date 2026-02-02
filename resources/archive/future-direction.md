# Future Direction: Multi-Axis Distillation Framework

A systematic exploration of the knowledge distillation design space across multiple axes.

---

## The Axes

### X-Axis: Tasks

What problem are we solving? Each task has associated model families and datasets.

| Task | Models | Datasets | Notes |
|------|--------|----------|-------|
| **Classification** | BERT → DistilBERT, RoBERTa → MiniLM | MNLI, QQP, SST-2, IMDB | Encoder-only, logits over classes |
| **Translation** | Llama 3B → 1B, mT5 → mT5-small | WMT, Europarl, OPUS | Seq2seq or decoder-only |
| **Summarization** | BART → DistilBART, T5 → T5-small | CNN/DailyMail, XSum | Encoder-decoder |
| **Diffusion (Steps)** | DDPM-1000 → DDPM-4, SD → LCM | ImageNet, LAION | Same size, fewer steps |
| **Diffusion (Params)** | SD-XL → SD-Small, DiT-XL → DiT-S | ImageNet, LAION | Smaller model, same steps |
| **TTS** | StyleTTS2 → smaller variant | LJSpeech, LibriTTS | Diffusion-based speech |

### Y-Axis: Depth

What gets transferred from teacher to student? Deeper = more internal knowledge.

| Depth | What Transfers | Loss Type | Complexity |
|-------|----------------|-----------|------------|
| **Response** | Output logits/probabilities | KL, MSE | Low |
| **Feature** | Hidden state representations | MSE, L1, Cosine | Medium |
| **Attention** | Attention weight patterns | MSE, KL | Medium |
| **Relational** | Inter-sample relationships | Graph matching | High |
| **Progressive** | Multi-step → fewer steps | Trajectory matching | High (Diffusion) |

### Z-Axis: Losses (Future)

How do we measure the transfer? Different losses for different signals.

| Loss | Use Case | Formula |
|------|----------|---------|
| **CE** | Hard labels (ground truth) | `-sum(y * log(p))` |
| **KL** | Soft labels (teacher logits) | `sum(p_t * log(p_t / p_s))` |
| **MSE** | Features, attention weights | `mean((x_t - x_s)^2)` |
| **L1** | Features (sparse) | `mean(abs(x_t - x_s))` |
| **Cosine** | Embedding alignment | `1 - cos(x_t, x_s)` |

---

## Primary Exploration: Task × Depth

Starting with the Task × Depth plane since depth availability depends on task/architecture.

### Compatibility Matrix

```
                           ┌────────────────────────────────────────────────────────────────────────────────────────┐
                           │                                       DEPTH                                            │
                           ├──────────────────┬──────────────────┬───────────────────┬────────────┬─────────────────┤
                           │     Response     │     Feature      │    Attention      │ Relational │   Progressive   │
┌──────────────────────────┼──────────────────┼──────────────────┼───────────────────┼────────────┼─────────────────┤
│ Classification           │       ✓✓✓        │        ✓✓        │        ✓✓         │     ✓      │        -        │
│ (Encoder-only)           │   KL on Class    │  Hidden States   │ Attention Weights │   Batch    │       N/A       │
├──────────────────────────┼──────────────────┼──────────────────┼───────────────────┼────────────┼─────────────────┤
│ Translation              │       ✓✓✓        │        ✓         │        ✓          │     △      │        -        │
│ (Decoder/Enc-Dec)        │    Token KL      │  Layer Mapping   │      Causal       │  Sequence  │       N/A       │
├──────────────────────────┼──────────────────┼──────────────────┼───────────────────┼────────────┼─────────────────┤
│ Summarization            │       ✓✓✓        │        ✓         │        ✓✓         │     △      │        -        │
│ (Encoder-Decoder)        │    Token KL      │ Encoder+Decoder  │  Cross-Attention  │  Document  │       N/A       │
├──────────────────────────┼──────────────────┼──────────────────┼───────────────────┼────────────┼─────────────────┤
│ Diffusion (Steps)        │        ✓         │        △         │        △          │     -      │       ✓✓✓       │
│ Same Size, Lesser Steps  │ Noise Prediction │     Indirect     │     Indirect      │    N/A     │    1000→4→1     │
├──────────────────────────┼──────────────────┼──────────────────┼───────────────────┼────────────┼─────────────────┤
│ Diffusion (Params)       │       ✓✓✓        │        ✓✓        │        ✓✓         │     △      │        -        │
│ Smaller Size, Same Steps │ Noise Prediction │  UNet Features   │  Self-Attention   │   Image    │       N/A       │
├──────────────────────────┼──────────────────┼──────────────────┼───────────────────┼────────────┼─────────────────┤
│ TTS (Diffusion)          │        ✓         │        ✓✓        │        △          │     -      │       ✓✓✓       │
│ (StyleTTS2)              │    Mel/Audio     │     Encoder      │     Duration      │    N/A     │ Step Reduction  │
└──────────────────────────┴──────────────────┴──────────────────┴───────────────────┴────────────┴─────────────────┘

Legend: ✓✓✓ Primary approach  ✓✓ Feasible  ✓ Possible  △ Challenging  - N/A
```

---

### Identified Tasks × Depth

Concrete experiments we plan to run, organized by task.

#### Classification

| Dataset | Teacher | Student | Depths to Explore |
|---------|---------|---------|-------------------|
| **MNLI** | BERT | DistilBERT | Response, Feature, Attention |
| **QQP** | BERT | DistilBERT | Response, Feature, Attention |

#### Translation

| Dataset | Teacher | Student | Depths to Explore |
|---------|---------|---------|-------------------|
| **Europarl** | Llama-3.2-3B | Llama-3.2-1B | Response, Feature |

#### Diffusion (TTS)

| Dataset | Teacher | Student | Depths to Explore |
|---------|---------|---------|-------------------|
| **TBD** | StyleTTS2 | StyleTTS2 (fewer steps) | Progressive, Feature |

---

### Task × Depth: Detailed Breakdown

#### Classification × Depth

| Depth | Implementation | What to Match | Expected Benefit |
|-------|----------------|---------------|------------------|
| **Response** | `KL(student_logits/T, teacher_logits/T)` | Class probabilities | Dark knowledge transfer |
| **Feature** | `MSE(proj(s_hidden[i]), t_hidden[j])` | Layer representations | Richer intermediate knowledge |
| **Attention** | `MSE(s_attn[i], t_attn[j])` | Attention patterns | "How" to attend, not just "what" |
| **Relational** | `MSE(s_gram, t_gram)` | Batch similarity matrix | Inter-sample structure |

**Layer Mapping (BERT-12 → DistilBERT-6):**
```
Teacher:  [1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12]
              ↓       ↓       ↓       ↓        ↓        ↓
Student:     [1]     [2]     [3]     [4]      [5]      [6]
```

#### Translation × Depth

| Depth | Implementation | Challenges |
|-------|----------------|------------|
| **Response** | Token-level KL across vocabulary | Large vocab = memory intensive |
| **Feature** | Layer mapping with projection | Different hidden dims (4096 → 2048) |
| **Attention** | Causal attention matching | Must respect autoregressive masking |

**Llama 3B (32 layers) → Llama 1B (16 layers):**
```
Teacher:  [2] [4] [6] ... [30] [32]
           ↓   ↓   ↓        ↓    ↓
Student:  [1] [2] [3] ... [15] [16]
```

#### Diffusion (Steps) × Depth — Step Reduction

Goal: Same model size, fewer inference steps.

| Depth | Implementation | What's Different |
|-------|----------------|------------------|
| **Response** | Match noise prediction at step N with student at step N/2 | Trajectory matching |
| **Feature** | Indirect—features change with step count | Not primary approach |
| **Progressive** | N-step teacher → N/2-step student (iterative) | **Primary approach** |

**Progressive Distillation:**
```
Teacher: 1000 steps → 500 steps → 250 steps → ... → 4 steps → 1 step
         (original)   (distill)   (distill)        (distill)  (consistency)
```

**Use Cases:** Real-time image generation, interactive applications, on-device inference

---

#### Diffusion (Params) × Depth — Size Reduction

Goal: Smaller model (fewer parameters), same number of steps.

| Depth | Implementation | What's Different |
|-------|----------------|------------------|
| **Response** | Match noise predictions at each step | Like classification but continuous |
| **Feature** | UNet encoder/decoder/skip features | Skip connections are critical |
| **Attention** | Self-attention in UNet blocks | Cross-attention for conditioning |

**Architecture Mapping (SD-XL → SD-Small):**
```
Teacher UNet:  [Down1] [Down2] [Down3] [Mid] [Up3] [Up2] [Up1]
                  ↓       ↓       ↓      ↓      ↓      ↓     ↓
Student UNet:  [Down1] [Down2]   -    [Mid]   -   [Up2] [Up1]
               (fewer channels, fewer blocks)
```

**Use Cases:** Mobile deployment, edge devices, reduced VRAM requirements

---

## Current Progress (Task × Depth)

### Completed

| Task | Depth | Status | Key Finding |
|------|-------|--------|-------------|
| Classification (MNLI) | Response | ✓ Complete | Co-train > Frozen |
| Classification (QQP) | Response | ✓ Complete | Frozen SFT hurts |
| Classification | Feature | ✓ Implemented | Configs ready |
| Classification | Attention | ✓ Implemented | Configs ready |

### In Progress

| Task | Depth | Status | Next Step |
|------|-------|--------|-----------|
| Classification | Feature + Attention | Ready | Run RFA factorial study |
| Translation | Response | Config ready | Run on HPC |

### Future

| Task | Depth | Priority | Rationale |
|------|-------|----------|-----------|
| Translation | Feature | Medium | Test if findings generalize |
| Diffusion (Steps) | Progressive | High | Different paradigm—step reduction |
| Diffusion (Params) | Response + Feature | Medium | Same paradigm as classification |
| Summarization | Response + Attention | Low | Similar to translation |

---

## Research Questions by Cell

### Classification × Response (Done)
- ✓ Does distillation help over CE-only? → Marginally (+1-2%)
- ✓ Frozen vs co-training? → Co-training wins
- ✓ Expert vs untrained teacher? → Untrained better when frozen

### Classification × Feature (Next)
- Does feature matching add value over response-only?
- Which layers matter most for transfer?
- Does feature KD help frozen SFT teachers?

### Classification × Attention (Next)
- Do attention patterns transfer useful knowledge?
- Is attention KD complementary to feature KD?
- Does attention help more than features?

### Translation × Response (Next)
- Do classification findings generalize to generation?
- Is token-level KL effective for seq2seq?
- Does co-training help for decoder-only models?

### Diffusion × Progressive (Future)
- Can we achieve 1-step generation with quality?
- How does progressive distillation compare to consistency training?
- Does this connect to our classification findings at all?

---

## Experimental Design: RFA Factorial Study

Testing Response (R), Feature (F), Attention (A) combinations:

| Config | R | F | A | Description |
|--------|---|---|---|-------------|
| 00 | - | - | - | Baseline (CE only) |
| R | ✓ | - | - | Response only (current) |
| F | - | ✓ | - | Feature only |
| A | - | - | ✓ | Attention only |
| RF | ✓ | ✓ | - | Response + Feature |
| RA | ✓ | - | ✓ | Response + Attention |
| FA | - | ✓ | ✓ | Feature + Attention |
| RFA | ✓ | ✓ | ✓ | All three |

**Hypothesis:** R + F or R + A > R alone, but diminishing returns for RFA.

---

## The Z-Axis (Losses) — Future Work

Once Task × Depth is understood, explore loss variations:

### Per-Depth Loss Options

| Depth | Primary Loss | Alternatives to Test |
|-------|--------------|---------------------|
| Response | KL Divergence | MSE, JS Divergence, Focal |
| Feature | MSE | L1, Cosine, Contrastive |
| Attention | MSE | KL, Wasserstein |

### Research Questions

- Does L1 beat MSE for sparse feature matching?
- Does cosine similarity preserve directional info better?
- Is KL appropriate for attention (probability-like) or MSE (weight-like)?

---

## Visualization: The Design Space

```
                         LOSSES (Z)
                           ↑
                           │   KL
                           │   MSE
                           │   L1
                           │   Cosine
                           │
                           │
        DEPTH (Y)          │
           ↑               │
           │ Progressive ──┼────────────────────────────────→ TASKS (X)
           │ Relational    │    Classification
           │ Attention     │    Translation
           │ Feature       │    Summarization
           │ Response      │    Diffusion (Steps)
           │               │    Diffusion (Params)
           └───────────────┘    TTS

Current exploration: Task × Depth plane (X-Y)
Future exploration: Add Loss dimension (Z)
```

---

## Priority Order

### Phase 1: Classification × Depth (Current)
1. ✓ Response distillation ablations (MNLI, QQP)
2. → Feature distillation experiments
3. → Attention distillation experiments
4. → RFA factorial study

### Phase 2: Translation × Depth
5. Response distillation (Llama 3B → 1B)
6. Feature distillation if response shows promise
7. Cross-task comparison with classification

### Phase 3: Diffusion × Depth (Stretch)
8. **Diffusion (Steps):** Literature review on progressive/consistency distillation
9. **Diffusion (Params):** Apply classification learnings (Response + Feature KD)
10. Compare: Do co-training vs frozen findings transfer to diffusion?

### Phase 4: Loss Variations (Z-axis)
11. Revisit best Task × Depth cells with alternative losses
12. Identify loss-specific patterns

---

## Open Questions

1. **Is there a universal "best depth"?**
   - Or is optimal depth task-dependent?

2. **Do findings transfer across tasks?**
   - If co-training wins for classification, does it win for translation?

3. **Are the two diffusion tasks fundamentally different?**
   - Step reduction (Progressive) seems orthogonal to feature/attention
   - Param reduction might follow classification patterns

4. **What's the loss-depth interaction?**
   - Maybe MSE for features, KL for response, cosine for attention?

5. **Diminishing returns?**
   - At what point does adding depth/loss not help?

---

## Related Documents

| Document | Purpose |
|----------|---------|
| [research-framing.md](research-framing.md) | Current research questions and findings |
| [experimental_findings.md](experimental_findings.md) | Detailed experiment results |
| [diffusion-distillation.md](diffusion-distillation.md) | Diffusion-specific distillation |
| [architecture.md](architecture.md) | Framework implementation details |
