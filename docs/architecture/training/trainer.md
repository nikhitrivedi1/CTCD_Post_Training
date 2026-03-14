# Training: Trainer

Custom HuggingFace Trainer subclasses for distillation and fine-tuning.

**Source:** `src/training/trainer.py` (~464 lines)

---

## Class Hierarchy

```
transformers.Trainer
  └── FinetuningTrainer        # Pass-through base (establishes hierarchy)
        └── DistillationTrainer  # Teacher-model training with pluggable losses
```

---

## FinetuningTrainer

```python
class FinetuningTrainer(Trainer):
    pass
```

Currently a pass-through that wraps HuggingFace Trainer. Exists to establish the class hierarchy for future extensions (LoRA setup hooks, custom eval logic). Used directly for CE-only SFT training (no teacher).

---

## DistillationTrainer

The core training class. Extends HuggingFace Trainer to maintain a teacher model and compute combined losses using pluggable loss strategies.

### Constructor

```python
DistillationTrainer(
    teacher_model: torch.nn.Module,
    temperature: float = 2.0,
    alpha: float = 0.5,
    freeze_teacher: bool = True,
    loss_strategy: DistillationLossStrategy = None,
    loss_type: str = None,
    teacher_label_map: list[int] = None,
    layer_mapping_type: str = None,
    layer_mapping: list[tuple[int, int]] = None,
    projector: torch.nn.Module = None,
    *args, **kwargs  # Passed to HF Trainer
)
```

**Strategy resolution priority:**
1. `loss_strategy` (pre-built instance) - highest priority
2. `loss_type` (string name) - looked up in `LOSS_REGISTRY`
3. Default: `KLDistillationLoss(temperature, alpha)`

**Teacher modes:**
- `freeze_teacher=True`: Teacher set to `eval()`, all params frozen (`requires_grad=False`). Forward pass wrapped in `torch.no_grad()`.
- `freeze_teacher=False`: Teacher stays in `train()` mode. Gradients flow through teacher for its own CE loss, but intermediates (hidden states, attentions) are detached before passing to distillation losses.

### Key Methods

#### `create_optimizer()`

Extends parent optimizer creation. If co-training (`freeze_teacher=False`), adds teacher parameters as a new param group to the model's optimizer. Both models share the same learning rate and schedule.

#### `_detach_teacher_intermediates(teacher_outputs)`

Creates a shallow copy of teacher outputs with detached hidden states and attentions. This is critical for co-training correctness:

**Problem:** Without detaching, Feature/Attention loss gradients flow back into the teacher, pulling teacher representations toward the (worse) model. The teacher should only learn from its own CE loss.

**Solution:** `dataclasses.replace()` creates a new output object with `hidden_states` and `attentions` detached, while keeping `loss` and `logits` connected to the computation graph (so teacher CE loss still gets gradients).

#### `compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None)`

The main training loop override. Step-by-step:

1. **Model forward pass** with optional `output_hidden_states` / `output_attentions` flags
2. **Early return during eval** - skip distillation loss entirely (just return CE loss). Hidden states/attentions are disabled during eval to prevent OOM.
3. **Teacher forward pass** - with or without gradients depending on `freeze_teacher`
4. **Label remapping** - reorder teacher logits if `teacher_label_map` is set (e.g., TextAttack models use different label ordering than HuggingFace datasets)
5. **Vocab alignment** - truncate logits to min vocab size for cross-tokenizer distillation (e.g., Qwen3-VL has 4 extra special tokens vs Qwen2.5-VL)
6. **Detach teacher** - detach logits for distillation loss, detach intermediates if co-training
7. **Compute distillation loss** via `self.loss_strategy.compute()` - returns `LossOutput` with total loss and component breakdown
8. **Add teacher CE** if co-training (recomputed with remapped logits if label map is set)
9. **Accumulate metrics** - per-step loss components stored in `_metric_accumulator`
10. **Flush metrics** - averaged and logged via `self.log()` at `logging_steps` intervals

**Why metric accumulation matters:** Without it, `self.log()` fires every step, creating noisy W&B charts. The accumulator averages over `logging_steps` steps before logging, matching HF Trainer's behavior for the standard loss metric.

#### `setup_from_config(config, teacher_model, model)` (classmethod)

Factory method that parses the `distillation` config section and returns everything needed to construct a `DistillationTrainer`:

```python
trainer_kwargs, layer_mapping, projector = DistillationTrainer.setup_from_config(
    config, teacher, model
)
trainer = DistillationTrainer(
    teacher_model=teacher,
    model=model,
    **trainer_kwargs,
    layer_mapping=layer_mapping,
    projector=projector,
    ...
)
```

Handles:
- Basic params (temperature, alpha)
- Staged distillation (builds `StagedDistillationLoss` from stage configs)
- Layer mapping (gets layer counts from model configs, creates mapping strategy)
- Projection (auto-selects identity if dims match, creates projection if they differ)
- Teacher label remapping

#### `prediction_step(model, inputs, prediction_loss_only, ignore_keys)`

Overrides HF Trainer's prediction step to temporarily disable `output_hidden_states` and `output_attentions` during evaluation. Without this, hidden states accumulate across the full eval set and cause OOM on large datasets.

Uses a try/finally pattern to restore the original settings after each prediction step.

---

## Design Decisions

### Why not a separate Trainer per loss type?

The Strategy Pattern allows swapping loss functions without modifying the Trainer class. A single `DistillationTrainer` handles KL, MSE, Feature, Attention, Combined, and Staged losses through the pluggable `loss_strategy` interface.

### Why detach teacher logits even when frozen?

Even with `torch.no_grad()`, the teacher's logits tensor is still part of the computation graph if created within a grad-enabled context. Explicit `.detach()` on `teacher_logits_for_distill` ensures no accidental gradient flow.

### Why recompute teacher CE with remapped logits?

When `teacher_label_map` is set, HuggingFace computes `teacher_outputs.loss` using the raw (un-remapped) logits against the dataset labels. This is incorrect because the teacher's label ordering doesn't match the dataset. The fix: recompute CE using the remapped logits.

---

*Source: `src/training/trainer.py`*
