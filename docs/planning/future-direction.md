# Future Direction: Multi-Axis Distillation Framework

A systematic exploration of the knowledge distillation design space across multiple axes.

---

## The Axes

### X-Axis: Tasks

| Task | Models | Datasets | Notes |
|------|--------|----------|-------|
| **Classification** | BERT → DistilBERT | MNLI, QQP, SST-2 | Encoder-only, logits over classes |
| **Translation** | Llama 3B → 1B | Europarl | Seq2seq or decoder-only |
| **Summarization** | BART → DistilBART | CNN/DailyMail, XSum | Encoder-decoder |
| **Diffusion (Steps)** | DDPM-1000 → DDPM-4 | ImageNet | Same size, fewer steps |
| **Diffusion (Params)** | SD-XL → SD-Small | ImageNet | Smaller model, same steps |

### Y-Axis: Depth

| Depth | What Transfers | Loss Type | Complexity |
|-------|----------------|-----------|------------|
| **Response** | Output logits | KL, MSE | Low |
| **Feature** | Hidden states | MSE, L1, Cosine | Medium |
| **Attention** | Attention weights | MSE, KL | Medium |
| **Relational** | Inter-sample relationships | Graph matching | High |
| **Progressive** | Multi-step → fewer steps | Trajectory matching | High (Diffusion) |

### Z-Axis: Losses (Future)

| Loss | Use Case | Formula |
|------|----------|---------|
| **CE** | Hard labels | `-sum(y * log(p))` |
| **KL** | Soft labels | `sum(p_t * log(p_t / p_s))` |
| **MSE** | Features, attention | `mean((x_t - x_s)^2)` |
| **Cosine** | Embedding alignment | `1 - cos(x_t, x_s)` |

---

## Compatibility Matrix

```
                         │     Response     │     Feature      │    Attention      │   Progressive   │
├──────────────────────────┼──────────────────┼──────────────────┼───────────────────┼─────────────────┤
│ Classification           │       ✓✓✓        │        ✓✓        │        ✓✓         │        -        │
│ Translation              │       ✓✓✓        │        ✓         │        ✓          │        -        │
│ Summarization            │       ✓✓✓        │        ✓         │        ✓✓         │        -        │
│ Diffusion (Steps)        │        ✓         │        △         │        △          │       ✓✓✓       │
│ Diffusion (Params)       │       ✓✓✓        │        ✓✓        │        ✓✓         │        -        │

Legend: ✓✓✓ Primary approach  ✓✓ Feasible  ✓ Possible  △ Challenging  - N/A
```

---

## Planned Experiments

### Classification × Depth

| Depth | Implementation | Expected Benefit |
|-------|----------------|------------------|
| **Response** | `KL(student_logits/T, teacher_logits/T)` | Dark knowledge transfer |
| **Feature** | `MSE(proj(s_hidden[i]), t_hidden[j])` | Richer intermediate knowledge |
| **Attention** | `MSE(s_attn[i], t_attn[j])` | "How" to attend |

**Layer Mapping (BERT-12 → DistilBERT-6):**
```
Teacher:  [1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12]
              ↓       ↓       ↓       ↓        ↓        ↓
Student:     [1]     [2]     [3]     [4]      [5]      [6]
```

### Translation × Depth

| Depth | Implementation | Challenges |
|-------|----------------|------------|
| **Response** | Token-level KL across vocabulary | Large vocab = memory intensive |
| **Feature** | Layer mapping with projection | Different hidden dims (4096 → 2048) |

**Llama 3B (32 layers) → Llama 1B (16 layers):**
```
Teacher:  [2] [4] [6] ... [30] [32]
           ↓   ↓   ↓        ↓    ↓
Student:  [1] [2] [3] ... [15] [16]
```

---

## RFA Factorial Design

Testing Response (R), Feature (F), Attention (A) combinations:

| Config | R | F | A | Description |
|--------|---|---|---|-------------|
| 00 | - | - | - | Baseline (CE only) |
| R | ✓ | - | - | Response only |
| F | - | ✓ | - | Feature only |
| A | - | - | ✓ | Attention only |
| RF | ✓ | ✓ | - | Response + Feature |
| RA | ✓ | - | ✓ | Response + Attention |
| FA | - | ✓ | ✓ | Feature + Attention |
| RFA | ✓ | ✓ | ✓ | All three |

**Hypothesis:** R + F or R + A > R alone, but diminishing returns for RFA.

---

## Priority Order

### Phase 1: Classification × Depth (Complete)
1. ✓ Response distillation ablations (MNLI, QQP)
2. ✓ RFA factorial study (MNLI) - R-only wins, F/A cause gradient conflict
3. → Translation experiments

### Phase 2: Translation × Depth
5. Response distillation (Llama 3B → 1B)
6. Feature distillation if response shows promise
7. Cross-task comparison with classification

### Phase 3: Diffusion × Depth (Stretch)
8. Literature review on progressive/consistency distillation
9. Apply classification learnings (Response + Feature KD)
10. Compare: Do co-training vs frozen findings transfer?

---

## Open Questions

1. **Is there a universal "best depth"?** Or is optimal depth task-dependent?
2. **Do findings transfer across tasks?** If co-training wins for classification, does it win for translation?
3. **Are the two diffusion tasks fundamentally different?** Step reduction vs param reduction
4. **Diminishing returns?** At what point does adding depth/loss not help?
