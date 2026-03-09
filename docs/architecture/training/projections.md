# Training: Projections

Dimension alignment modules for Feature distillation when teacher and student have different hidden sizes.

**Source:** `src/training/projections.py` (~196 lines)

---

## Overview

When the teacher's hidden dimension differs from the student's (e.g., BERT-large 1024 to DistilBERT 768), a projection module aligns dimensions before computing Feature loss. The projector is a learnable component trained during distillation.

---

## Projection Types

### IdentityProjection

No-op pass-through. Used when teacher and student have matching dimensions.

```python
class IdentityProjection:
    def __call__(self, x):
        return x
```

Not an `nn.Module` - has no parameters, no computation. Provides a consistent interface so Feature loss code doesn't need dimension-match checks.

**Example use case:** BERT-base (768) to DistilBERT (768) - same hidden size, no projection needed.

### LinearProjection (nn.Module)

Single linear layer: `teacher_dim -> student_dim`.

```python
class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.to(self.proj.weight.dtype)  # Handle mixed precision
        return self.proj(x)
```

Learnable, trained alongside the student during distillation. The dtype cast handles fp16 training where the input may be fp16 but the projector weights are fp32.

**Example use case:** BERT-large (1024) to DistilBERT (768) - linear projection down.

### MLPProjection (nn.Module)

Two-layer MLP with hidden layer and activation: `input_dim -> hidden_dim -> output_dim`.

```python
class MLPProjection(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, activation="relu"):
        hidden_dim = hidden_dim or (input_dim + output_dim) // 2
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, output_dim)
        )
```

More expressive than linear projection. The hidden layer lets the projection learn a nonlinear transformation between teacher and student representation spaces.

**Default hidden_dim:** average of input and output dims, e.g., (1024 + 768) / 2 = 896.

**Supported activations:** `relu`, `gelu`, `tanh`.

---

## Registry and Factory

```python
PROJECTION_REGISTRY = {
    "identity": IdentityProjection,
    "linear":   LinearProjection,
    "mlp":      MLPProjection,
}
```

### `get_projection(name, input_dim, output_dim, **kwargs)`

Creates a projection by name. `input_dim` and `output_dim` are required for `linear` and `mlp`.

### `create_projection_if_needed(teacher_dim, student_dim, projection_type, **kwargs)`

Convenience function that automatically selects `IdentityProjection` if dims match, otherwise creates the specified projection type. This is the primary API used by `DistillationTrainer.setup_from_config()`.

```python
# In setup_from_config:
if teacher_dim != student_dim or projection_config.get("force", False):
    projector = create_projection_if_needed(teacher_dim, student_dim, proj_type, ...)
```

The `force` config option creates a projection even when dimensions match (e.g., for experiments testing whether a learnable projection layer helps even with same-dim models).

---

## Device and Dtype Placement

After creation, the projector is moved to the student's device and dtype:

```python
projector = projector.to(device=student_model.device, dtype=student_model.dtype)
```

Both `LinearProjection` and `MLPProjection` handle dtype casting in their `forward()` methods, casting input to match projector weight dtype. This handles the case where model hidden states are fp16 but the projector was initialized in fp32.

---

## Config Example

```yaml
distillation:
  projection:
    type: linear         # identity, linear, or mlp
    force: false         # Create even if dims match
    hidden_dim: 512      # Only for mlp
    activation: gelu     # Only for mlp (relu, gelu, tanh)
```

---

*Source: `src/training/projections.py`*
