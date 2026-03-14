# Training: Layer Mapping

Strategies for mapping teacher layers to model layers in Feature and Attention distillation.

**Source:** `src/training/layer_mapping.py` (~162 lines)

---

## Overview

When teacher and model have different depths (e.g., BERT-12 to DistilBERT-6), Feature and Attention losses need to know which teacher layer corresponds to which model layer. Layer mapping strategies produce a list of `(teacher_idx, model_idx)` pairs.

All indices are 0-based and refer to hidden state / attention output positions (which include the embedding layer as index 0 in HuggingFace outputs).

---

## Strategies

### UniformMappingStrategy

Maps every Nth teacher layer to model layers.

```python
step = teacher_layers // model_layers
mapping = [(i * step + step - 1, i) for i in range(model_layers)]
```

**Example (12 teacher, 6 model):**
```
Teacher: [1, 3, 5, 7, 9, 11] -> Model: [0, 1, 2, 3, 4, 5]
```

Takes every 2nd teacher layer. The `step - 1` offset means we take the last layer in each group (layer 1 not 0, layer 3 not 2, etc.).

### LastNMappingStrategy

Maps the last N teacher layers to model layers.

```python
offset = teacher_layers - model_layers
mapping = [(offset + i, i) for i in range(model_layers)]
```

**Example (12 teacher, 6 model):**
```
Teacher: [6, 7, 8, 9, 10, 11] -> Model: [0, 1, 2, 3, 4, 5]
```

**Rationale:** Later layers capture more task-specific information. Early teacher layers may contain low-level features that the model already learns on its own.

### FirstLastMappingStrategy

Emphasizes first and last layers, interpolates middle.

```python
indices = [0]
step = (teacher_layers - 1) / (model_layers - 1)
for i in range(1, model_layers - 1):
    indices.append(int(i * step))
indices.append(teacher_layers - 1)
```

**Example (12 teacher, 6 model):**
```
Teacher: [0, 2, 4, 6, 8, 11] -> Model: [0, 1, 2, 3, 4, 5]
```

**Rationale:** First layers capture low-level features (tokenization, basic syntax), last layers capture high-level semantics (task-specific representations). Both endpoints are important, with evenly-spaced middle layers filling in.

Handles edge cases:
- `model_layers == 1`: maps to last teacher layer only
- `model_layers == 2`: maps to first and last teacher layers

### CustomMappingStrategy

User-defined explicit mapping via config.

```python
CustomMappingStrategy(
    teacher_indices=[1, 3, 5, 7, 9, 11],
    model_indices=[0, 1, 2, 3, 4, 5]
)
```

Validates that both lists have equal length. The `map()` method ignores its arguments and returns the stored mapping.

Config example:
```yaml
layer_mapping:
  type: custom
  mapping:
    - {teacher: 1, model: 0}
    - {teacher: 3, model: 1}
    - {teacher: 5, model: 2}
```

---

## Validation

All strategies have a `validate(teacher_layers, student_layers)` method that checks:
- `teacher_layers >= model_layers` (teacher must have at least as many layers)
- `model_layers > 0`

`CustomMappingStrategy` additionally checks that all indices are within bounds.

---

## Registry and Factory

```python
LAYER_MAPPING_REGISTRY = {
    "uniform":    UniformMappingStrategy,
    "last_n":     LastNMappingStrategy,
    "first_last": FirstLastMappingStrategy,
    "custom":     CustomMappingStrategy,
}
```

`get_layer_mapping(name, **kwargs)` creates a strategy instance. For `custom`, passes `teacher_indices` and `model_indices` kwargs. For all others, returns a no-arg instance.

---

## Usage in the Training Pipeline

Layer mapping is resolved in `DistillationTrainer.setup_from_config()`:

1. Read `distillation.layer_mapping.type` from config
2. Get layer counts from model configs: `model.config.num_hidden_layers`
3. Create mapping strategy and call `.map(teacher_layers, model_layers)`
4. Pass the resulting list of tuples to `DistillationTrainer` constructor
5. Trainer passes it through to loss strategies via `**loss_kwargs`

---

*Source: `src/training/layer_mapping.py`*
