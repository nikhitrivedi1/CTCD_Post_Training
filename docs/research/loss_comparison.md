# Distillation Loss Functions: Literature vs Implementation

## Reference Papers

### Hinton et al. (2015) - "Distilling the Knowledge in a Neural Network"

**Core Insight**: Soft targets from teacher contain "dark knowledge" - the relative probabilities of incorrect classes reveal the teacher's generalization patterns.

**Key Formulas**:
```
Soft label:     q_i = exp(z_i/T) / Σ exp(z_j/T)
Combined loss:  L = (1-α)·CE_hard + α·T²·KL(soft_student || soft_teacher)
```

**Critical Details**:
| Parameter | Recommendation | Rationale |
|-----------|---------------|-----------|
| T² scaling | Multiply KL by T² | Gradients scale as 1/T², need compensation |
| α (hard target weight) | "considerably lower" | Soft targets carry more information |
| Temperature | T=2-4 for small models, T>8 for larger | Small models can't capture all knowledge; lower T ignores noisy negative logits |

**Experimental Results**:
- ASR: T=2, α=0.5 (hard target weight) achieved 80%+ of ensemble gains
- MNIST: T=20 for large student, T=2.5-4 for small (30 units/layer)
- 3% of data with soft targets ≈ 100% of data with hard targets

---

### DistilBERT (Sanh et al. 2019)

**Core Insight**: Pre-training distillation with triple loss outperforms task-specific distillation.

**Triple Loss**:
```
L_total = L_ce + L_mlm + L_cos

L_ce   = Σ t_i · log(s_i)           # Soft target cross-entropy
L_mlm  = Masked language modeling    # Task-specific (pre-training)
L_cos  = 1 - cos(h_student, h_teacher)  # Hidden state alignment
```

**Ablation Study Impact on GLUE**:
| Ablation | GLUE Δ |
|----------|--------|
| Remove L_cos only | -1.46 |
| Remove L_mlm only | -0.31 |
| Remove both L_cos + L_mlm | -2.96 |
| Random init (vs teacher init) | -3.69 |

**Key Finding**: Teacher weight initialization is the single most important factor (-3.69 without it).

---

## Our Implementation vs Literature

### Response-Based Distillation (R)

| Aspect | Hinton | DistilBERT | Ours | Status |
|--------|--------|------------|------|--------|
| Loss type | KL with T² | Soft CE | KL with T² | ✓ Matches Hinton |
| Temperature | 2-20 (task-dependent) | Not specified | 15.0 (translation) | ✓ Appropriate for 128k vocab |
| Alpha | ~0.5 for hard targets | Linear combination | 0.2 (translation) | ✓ Lower due to high T |

**Implementation**: [losses.py:69-120](src/training/losses.py#L69-L120)

---

### Feature-Based Distillation (F)

| Aspect | FitNets (Romero 2015) | DistilBERT | Ours | Status |
|--------|----------------------|------------|------|--------|
| Loss type | MSE (hint layers) | **Cosine similarity** | MSE + Cosine | ✓ Both available |
| Normalization | Yes | Implicit in cosine | Optional | ✓ |
| Layer mapping | Guided layers | Every-other | Uniform mapping | ✓ |

**Available strategies**:
- `feature_mse` - FitNets-style MSE on hidden states
- `feature_cosine` - DistilBERT-style cosine similarity

**Implementation**: [losses.py](src/training/losses.py)

---

### Attention-Based Distillation (A)

| Aspect | Literature | Ours | Status |
|--------|-----------|------|--------|
| Loss type | KL divergence | KL divergence | ✓ |
| Head mapping | Various | Average (many→few) | ✓ Reasonable default |
| Temperature | Usually 1.0 | 1.0 | ✓ |

**Implementation**: [losses.py:459-570](src/training/losses.py#L459-L570)

---

## Recommended Loss Configurations by Task

### Classification (Small Vocab, e.g., MNLI with 3 classes)

```yaml
distillation:
  temperature: 4.0      # Hinton: 2-4 for smaller output spaces
  alpha: 0.5            # Classic 50-50 split
  loss_type: response_kl
```

**Rationale**: Small vocab means teacher probabilities are already distinguishable. Lower temperature preserves more information about class relationships.

---

### Translation/Generation (Large Vocab, e.g., 128k tokens)

```yaml
distillation:
  temperature: 15.0     # High T needed to soften 128k-way distribution
  alpha: 0.2            # Lower α since T² scaling already amplifies KL
  loss_type: response_kl
```

**Rationale**:
- 128k classes → most probabilities near zero → need high T to reveal structure
- T² scaling at T=15 means KL gradients 225× larger → lower α compensates
- Alternative: use `response_ce` loss (CE-based) which doesn't have T² scaling

---

### Feature + Response (Best Combined)

```yaml
distillation:
  loss_strategies:
    - type: response_kl
      weight: 1.0
    - type: feature_cosine  # DistilBERT-style (or feature_mse)
      weight: 0.5           # Feature loss as auxiliary signal
```

**Rationale**: DistilBERT showed L_cos (feature alignment) provides +1.46 GLUE improvement. Feature loss helps student learn intermediate representations, not just final outputs.

---

### Staged Distillation (Curriculum)

```yaml
distillation:
  stages:
    - strategy: feature_cosine     # DistilBERT-style (scale-invariant)
      until_progress: 0.3         # First 30%: align representations
    - strategy: response_kl
      until_progress: 1.0         # Remaining 70%: match outputs
```

**Rationale**: Feature alignment helps early training (FitNets), then response matching refines outputs (Hinton). Our MNLI experiments showed potential for gradient conflict when combining all three signals simultaneously.

**Available staged configs**:
- `staged/00_r.yaml` - R only (baseline)
- `staged/01_f-r.yaml` - Feature → Response
- `staged/02_a-r.yaml` - Attention → Response
- `staged/03_f-a-r.yaml` - Feature → Attention → Response

---

## Missing from Our Implementation

### 1. Teacher Weight Initialization

**Impact**: -3.69 GLUE points without it (DistilBERT ablation)

**What it is**: Initialize student by copying every-other layer from teacher.

**Why we don't have it**: Our students (1B) have different architecture than teachers (3B). Would require:
- Layer mapping strategy for different depths
- Dimension projection for different hidden sizes

**Potential implementation**:
```python
def init_student_from_teacher(student, teacher, layer_map):
    for t_idx, s_idx in layer_map:
        student.layers[s_idx].load_state_dict(
            teacher.layers[t_idx].state_dict(),
            strict=False  # Allow dimension mismatch
        )
```

---

### 2. ~~Cosine Feature Loss~~ ✓ IMPLEMENTED

**Impact**: -1.46 GLUE points without L_cos (DistilBERT)

**Status**: Now available as `feature_cosine` in LOSS_REGISTRY

```yaml
distillation:
  loss_strategies:
    - type: feature_cosine  # DistilBERT-style
      weight: 1.0
```

---

### 3. Task-Specific Auxiliary Loss

**DistilBERT**: L_mlm (masked language modeling during pre-training)

**For our translation task**: Could add:
- Translation quality metric as auxiliary signal
- Sequence-level loss (vs token-level)

---

## Summary: Best Practices

| Scenario | Temperature | Alpha | Loss Type | Notes |
|----------|-------------|-------|-----------|-------|
| Classification (few classes) | 2-4 | 0.5 | `response_kl` | Classic Hinton |
| Generation (large vocab) | 10-20 | 0.1-0.3 | `response_kl` or `response_ce` | High T, lower α |
| With feature alignment | - | - | `response_kl` + `feature_cosine` | DistilBERT-style |
| Progressive/curriculum | - | - | Staged F→R or A→R | Feature → Response |
| Very small student | 2-4 | 0.7 | `response_kl` | Lower T helps |

**Loss Naming Convention**: `{signal}_{method}`
- Signals: `response`, `feature`, `attention`
- Methods: `kl`, `mse`, `cosine`, `ce`

**Universal**: Always use T² scaling with KL divergence (already implemented in `response_kl`).
