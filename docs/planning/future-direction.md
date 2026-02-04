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
│                          │     Response     │     Feature      │    Attention      │   Progressive   │
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
| **Response** | Token-level KL across vocabulary | Large vocab (128K) × SeqLen = memory intensive |
| **Feature** | Layer mapping with projection | Hidden dim mismatch (3072 → 2048) |
| **Attention** | Head averaging + layer mapping | Head count mismatch (24 → 8) |

**Llama 3.2 3B (28 layers) → Llama 3.2 1B (16 layers):**
```
Teacher:  [2] [4] [5] [7] [9] [10] [12] [14] [16] [17] [19] [21] [23] [24] [26] [28]
           ↓   ↓   ↓   ↓   ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓
Student:  [1] [2] [3] [4] [5]  [6]  [7]  [8]  [9] [10] [11] [12] [13] [14] [15] [16]
```

**Model Architecture Comparison:**
| Attribute | Llama 3.2 3B | Llama 3.2 1B | Mapping |
|-----------|--------------|--------------|---------|
| Layers | 28 | 16 | Uniform (28:16 ≈ 1.75:1) |
| Hidden dim | 3072 | 2048 | Linear projection |
| Attention heads | 24 | 8 | Average groups of 3 |
| Vocab size | 128,256 | 128,256 | Same |

---

## FAR Factorial Design

Testing Feature (F), Attention (A), Response (R) combinations:

| Config | F | A | R | Description |
|--------|---|---|---|-------------|
| 00 | - | - | - | Baseline (CE only) |
| F | ✓ | - | - | Feature only |
| A | - | ✓ | - | Attention only |
| R | - | - | ✓ | Response only |
| FA | ✓ | ✓ | - | Feature + Attention |
| FR | ✓ | - | ✓ | Feature + Response |
| AR | - | ✓ | ✓ | Attention + Response |
| FAR | ✓ | ✓ | ✓ | All three |

**Hypothesis:** F + R or A + R > R alone, but diminishing returns for FAR.

---

## Priority Order

### Phase 1: Classification × Depth (Complete)
1. ✓ Response distillation ablations (MNLI, QQP)
2. ✓ FAR factorial study (MNLI) - R-only wins, F/A cause gradient conflict
3. → Translation experiments

### Phase 2: Translation × Depth (Configs Ready)
5. ✓ L0: Teacher taxonomy ablations (configs/translation/llama3p2-3b_llama3p2-1b/ablations/00-04)
6. ✓ L1: FAR factorial study (configs/translation/llama3p2-3b_llama3p2-1b/far/00-07)
7. → Run experiments, compare with classification findings

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
