# Layer-wise Analysis for Distillation

Understanding how well a student model learns representations at each layer helps inform architecture decisions (e.g., optimal number of layers) and diagnose training issues.

## Feature vs Attention: What Are We Measuring?

| Signal | Measures | Question Answered |
|--------|----------|-------------------|
| **Feature Loss** | *What* information is encoded | "Does the student know the same things as the teacher at this layer?" |
| **Attention Loss** | *How* information flows | "Does the student route information the same way as the teacher?" |

These are complementary signals:

- **Feature (hidden states)**: The layer's output representation — what concepts/features it encodes
- **Attention (attention weights)**: The layer's routing mechanism — which tokens inform which

A student could:
- ✓ Feature, ✗ Attention → Knows the right things, but via different reasoning paths
- ✗ Feature, ✓ Attention → Routes info similarly, but encodes different concepts
- ✓ Feature, ✓ Attention → Truly mimicking teacher's internal process

Per-layer tracking for both signals lets you pinpoint *where* and *what kind* of divergence occurs.

## Training-time vs Post-hoc Analysis

These techniques work both during training and on a final model:

| Approach | During Training | Post-hoc (Final Model) |
|----------|-----------------|------------------------|
| **Per-layer loss** | Track convergence over time | Snapshot of final alignment on held-out set |
| **CKA** | Watch alignment evolve | Compare final student vs teacher structure |
| **Linear probing** | Expensive, usually skip | Standard analysis — where do features emerge? |
| **Attention entropy** | Track specialization | Analyze final attention patterns |

**Training-time**: Track *dynamics* — is convergence healthy? Which layers are stuck?

**Post-hoc**: Get a *snapshot* — how good is the final result? Compare trained models.

Post-hoc use cases:
- Compare two trained students (which learned better layer alignment?)
- Decide if a model needs more training (are layers still misaligned?)
- Validate architecture choices before next experiment

## Approaches

### 1. Per-Layer Feature Loss Tracking

**What**: Log feature distillation loss separately for each layer instead of summing.

**Why**: Shows which layers converge vs struggle. If loss plateaus after layer N, the student may have saturated capacity.

#### Summary: What Gets Logged

| Metric | Description |
|--------|-------------|
| `feature_loss/total` | Sum of all layer losses (used for backprop) |
| `feature_loss/layer_0` | Loss for student layer 0 → teacher layer mapping |
| `feature_loss/layer_1` | Loss for student layer 1 → teacher layer mapping |
| ... | ... |
| `feature_loss/layer_N` | Loss for student layer N → teacher layer mapping |

This gives you one curve per layer in W&B, plus the total — letting you see both the aggregate and where issues arise.

#### Layer Mapping

Metrics are logged by *student* layer index. The mapping determines which teacher layer each student layer targets:

```
Student (6L) → Teacher (12L) with uniform mapping:
  student_0 → teacher_0   → logged as feature_loss/layer_0
  student_1 → teacher_2   → logged as feature_loss/layer_1
  student_2 → teacher_4   → logged as feature_loss/layer_2
  ...
```

The metric name is student-centric — the mapping is a training config detail, not a logging concern.

#### Avoiding Dashboard Clutter

W&B auto-groups metrics by prefix, so `feature_loss/*` collapses into one section. For quick overview, also log aggregate stats:

```python
layer_losses["feature_loss/mean"] = np.mean(list(per_layer.values()))
layer_losses["feature_loss/max"] = max(per_layer.values())
```

Typical workflow:
1. Glance at `mean` and `max` for overall health
2. Drill into per-layer only when diagnosing issues
3. Create custom W&B panels for key checkpoint layers (e.g., 0, mid, last)

#### Intuition

In feature distillation, we're asking: "How well does student layer S_i match teacher layer T_j?"

When we sum all layer losses into one number, we lose critical information:
- A total loss of 0.5 could mean all layers at 0.5 (uniform struggle)
- Or layers 0-3 at 0.1 and layers 4-5 at 1.5 (deep layer bottleneck)

These have identical totals but completely different implications.

#### What Each Layer Represents

Transformer layers learn hierarchically:
- **Early layers (0-2)**: Token-level features, local patterns, syntax
- **Middle layers (3-6)**: Phrase-level semantics, entity recognition
- **Late layers (7+)**: Task-specific abstractions, high-level reasoning

A student struggling at layer 4 is failing to learn semantic composition — different from struggling at layer 1 (basic encoding) or layer 8 (task abstraction).

#### Expected Curve Patterns

**Pattern A: Monotonic Decrease**
```
Loss
  │
1.0├──●
   │    ╲
0.5├      ●──●
   │          ╲
0.1├            ●──●──●
   └──────────────────────
     L0  L1  L2  L3  L4  L5
```
*Interpretation*: Healthy learning. Early layers are hardest (most information to compress), later layers easier (more abstract, fewer constraints). Student has sufficient capacity.

**Pattern B: U-Shape**
```
Loss
  │
1.0├                    ●
   │  ●              ●
0.5├    ╲          ╱
   │      ●──●──●
0.1├
   └──────────────────────
     L0  L1  L2  L3  L4  L5
```
*Interpretation*: Deep layers struggling. Student can match teacher's intermediate representations but fails at final abstraction. Possible causes:
- Hidden dimension too small for task complexity
- Layer mapping mismatch (student L5 ≠ teacher L10)
- Missing task-specific capacity

**Pattern C: Plateau**
```
Loss
  │
1.0├
   │
0.5├──●──●──●──●──●──●
   │
0.1├
   └──────────────────────
     L0  L1  L2  L3  L4  L5
```
*Interpretation*: Uniform struggle across all layers. Student architecture fundamentally mismatched. Consider:
- Is hidden dimension too small?
- Is the layer mapping reasonable?
- Is the student even learning (check LR, gradients)?

**Pattern D: Early Plateau, Late Drop**
```
Loss
  │
1.0├
   │
0.5├──●──●──●
   │        ╲
0.1├          ●──●──●
   └──────────────────────
     L0  L1  L2  L3  L4  L5
```
*Interpretation*: Early layers saturated. Student's first layers can't encode teacher's low-level features, but deeper layers compensate. May indicate:
- Student needs more early-layer capacity
- Or: early features don't transfer well (different tokenizers, etc.)

**Pattern E: Specific Layer Spike**
```
Loss
  │
1.0├            ●
   │          ╱   ╲
0.5├        ╱       ╲
   │  ●──●           ●──●
0.1├
   └──────────────────────
     L0  L1  L2  L3  L4  L5
```
*Interpretation*: Bottleneck at specific layer. That layer is trying to learn something the student can't represent. Check:
- What does teacher layer N encode? (Probe it)
- Is the layer mapping at that position sensible?
- Does the student have architectural limitations there?

#### Tracking Over Training Time

Per-layer loss is most valuable when tracked across training:

```
        Step 0    Step 1K   Step 5K   Step 10K
Layer 0:  1.2   →   0.8   →   0.4   →   0.3
Layer 1:  1.1   →   0.7   →   0.3   →   0.2
Layer 2:  1.0   →   0.8   →   0.7   →   0.7   ← Stuck!
Layer 3:  0.9   →   0.6   →   0.4   →   0.3
```

Layer 2 stopped improving at step 5K. Questions to ask:
- Is this layer's mapping wrong?
- Did the learning rate decay too fast?
- Is there a gradient flow issue?

#### Implementation

```python
def compute_feature_loss(student_hiddens, teacher_hiddens, layer_map):
    """
    Compute per-layer feature distillation loss.

    Args:
        student_hiddens: List of [batch, seq, hidden] tensors
        teacher_hiddens: List of [batch, seq, hidden] tensors
        layer_map: Dict mapping student layer idx → teacher layer idx

    Returns:
        total_loss: Scalar loss for backprop
        layer_losses: Dict of per-layer losses for logging
    """
    layer_losses = {}
    total_loss = 0

    for student_idx, teacher_idx in layer_map.items():
        s_hidden = student_hiddens[student_idx]
        t_hidden = teacher_hiddens[teacher_idx]

        # Project if dimensions differ
        if s_hidden.size(-1) != t_hidden.size(-1):
            s_hidden = projection_layer(s_hidden)

        # Compute loss (MSE or cosine)
        loss = F.mse_loss(s_hidden, t_hidden)

        layer_losses[f"feature_loss/layer_{student_idx}"] = loss.item()
        total_loss += loss

    return total_loss, layer_losses

# In training loop
loss, layer_losses = compute_feature_loss(...)
wandb.log(layer_losses, step=global_step)
```

#### Actionable Decisions

| Observation | Action |
|-------------|--------|
| Late layers never converge | Increase hidden dim, or accept shallower student |
| Specific layer stuck | Re-examine layer mapping at that position |
| All layers plateau together | Architecture mismatch — reconsider student design |
| Early layers high, late layers low | Normal — early layers compress more information |
| Loss increases with depth | Student lacks capacity for abstraction — reduce layers or increase width |

#### Combining with Other Signals

Per-layer loss tells you *where* the student struggles. Combine with:
- **CKA**: To understand *what* representations differ
- **Linear probing**: To check if the struggle affects downstream task
- **Gradient norms**: To diagnose optimization issues at specific layers

### 2. CKA (Centered Kernel Alignment)

**What**: Compare student layer representations to teacher layer representations using kernel alignment.

**Why**: Shows which student layers align with which teacher layers. Reveals if the student develops similar internal structure.

**Implementation**:
```python
def cka(X, Y):
    """Compute CKA between two representation matrices."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)

    hsic_xy = (X @ X.T * Y @ Y.T).sum()
    hsic_xx = (X @ X.T * X @ X.T).sum()
    hsic_yy = (Y @ Y.T * Y @ Y.T).sum()

    return hsic_xy / (hsic_xx.sqrt() * hsic_yy.sqrt())

# Build similarity matrix
cka_matrix = torch.zeros(student_layers, teacher_layers)
for i, s_hidden in enumerate(student_hiddens):
    for j, t_hidden in enumerate(teacher_hiddens):
        cka_matrix[i, j] = cka(s_hidden, t_hidden)
```

**Signals**:
- Diagonal dominance → good layer-to-layer alignment
- Off-diagonal peaks → student layer N learns teacher layer M features
- Low overall CKA → student developing different representations

### 3. Linear Probing

**What**: Freeze student, train a linear classifier on each layer's output for the downstream task.

**Why**: Directly measures task-relevant information at each layer.

**Implementation**:
```python
for layer_idx in range(num_layers):
    probe = nn.Linear(hidden_size, num_classes)
    # Train probe on frozen representations from layer_idx
    # Evaluate accuracy
    layer_accuracies[layer_idx] = evaluate(probe)
```

**Signals**:
- Accuracy curve shows where useful features emerge
- Compare student vs teacher curves
- If student accuracy peaks earlier, later layers may be redundant

### 4. Attention Pattern Entropy

**What**: Track per-layer attention entropy (how spread out attention is).

**Why**: Lower entropy = more specialized/focused attention patterns.

**Implementation**:
```python
def attention_entropy(attn_weights):
    """Compute entropy of attention distribution."""
    # attn_weights: [batch, heads, seq, seq]
    entropy = -(attn_weights * attn_weights.log()).sum(-1).mean()
    return entropy

# Per layer
layer_entropies = [attention_entropy(attn) for attn in attention_outputs]
```

**Signals**:
- Compare student vs teacher entropy curves
- Student entropy higher → less specialized attention
- Entropy doesn't decrease with depth → attention heads not specializing

## Recommended Starting Point

**Per-layer feature loss tracking** is the cheapest win — requires minimal code changes and leverages existing feature distillation infrastructure. Provides immediate visibility into layer-wise learning dynamics.

## Use Cases

| Question | Approach |
|----------|----------|
| Should I reduce student layers? | Per-layer loss + linear probing |
| Is the student learning teacher structure? | CKA |
| Are attention heads specializing? | Attention entropy |
| Which layers are bottlenecks? | Per-layer loss |
