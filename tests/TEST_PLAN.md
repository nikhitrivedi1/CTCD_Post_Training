# Forge Test Plan

> Living document. Last updated: March 2026.
> Current focus: **Core only.**

---

## Principles

1. **Core first.** The modules every training run touches get thorough, property-based tests. Everything else waits.
2. **Parametrize, don't copy-paste.** One test function with 9 strategy inputs beats 9 identical test functions.
3. **Test properties, not examples.** "Loss is finite and differentiable" across all strategies, not "KL with alpha=0.5 and T=2 gives 1.3847".
4. **No network calls.** Models are constructed from configs with random init. Datasets are synthetic. Nothing downloads.
5. **No mocking internals.** Use real tiny modules where possible. Mock only at system boundaries (file I/O, HF Hub, SLURM).
6. **Naming follows context.** Distillation tests say `model` and `teacher`. Pure math tests (KL, MSE) say `model_logits` and `target_logits`. General tests say `model`.

---

## Fixtures

All fixtures live in `tests/conftest.py`. Constructed in code, nothing stored on disk.

### Tensor fixtures
```
model_logits(batch_size, num_classes)        # (4, 3) with requires_grad
target_logits(batch_size, num_classes)       # (4, 3) detached
lm_model_logits(batch_size, seq_len, vocab)  # (4, 16, 128) with requires_grad
lm_target_logits(batch_size, seq_len, vocab) # (4, 16, 128) detached
ce_loss                                      # scalar tensor, requires_grad
```

### Hidden state / attention fixtures
```
hidden_states_model(batch, seq, hidden)      # 7 layers (embedding + 6 transformer)
hidden_states_target(batch, seq, hidden)     # 13 layers (embedding + 12 transformer)
attention_weights_model(batch, seq)           # 6 layers, softmax-normalized
attention_weights_target(batch, seq)          # 12 layers, softmax-normalized
```

### Model fixtures
```
tiny_model       # 2-layer nn.Module, returns (logits, loss, hidden_states, attentions)
tiny_teacher     # 4-layer nn.Module, same interface
```

Lightweight `nn.Module` subclasses for pure logic tests (losses, layer mapping, projections).

```
tiny_bert_model     # BertForSequenceClassification, 2 layers, hidden=64, random init
tiny_bert_teacher   # BertForSequenceClassification, 4 layers, hidden=64, random init
```

HF-compatible models for tests that go through HF `Trainer` or `evaluate_model()` (trainer, evaluation). Constructed from `BertConfig`, no downloads.

---

## Core Tests

### 1. Losses (`src/training/losses.py`)

9 strategies following the Strategy pattern. Most tests parametrized across all strategies.

**Universal properties (parametrized across all strategies):**
- `compute()` returns a `LossOutput` with finite `total_loss` that has gradients
- `total_loss` is a scalar (0-dim tensor)
- `components` dict is non-empty and all values are finite tensors
- `name` property returns a non-empty string

**KL-specific:**
- T=1 degenerates to standard KL divergence (exact math, `atol=1e-5`)
- T^2 scaling matches Hinton formula: `KL(softmax(s/T), softmax(t/T)) * T^2`
- Numerical stability sweep: parametrize T in [0.1, 0.5, 1, 2, 5, 10, 50, 100], assert no NaN/Inf
- Reverse KL produces different gradients than forward KL

**Alpha weighting (parametrized across KL, MSE, Cosine, SoftTarget):**
- alpha=0: distillation component is zero, total equals CE
- alpha=1: CE component is zero
- alpha=0.5: total is between CE-only and distillation-only

**Feature/Attention losses:**
- `requires_hidden_states` / `requires_attentions` returns True
- Different head counts between model and target handled (attention averaging)
- Projection dtype casting works (fp16 hidden states through fp32 projector)

**Combined loss:**
- Weights normalize to sum=1
- Each sub-strategy's component appears in output
- Zero weight on a strategy excludes it from gradient flow

**Staged loss:**
- Returns correct strategy based on training progress
- Transitions at stage boundaries

### 2. Trainer (`src/training/trainer.py`)

Central to correctness. Since `DistillationTrainer` extends HF `Trainer` (which introspects `.config`, forward signature, etc.), these tests use `BertConfig` random-init models from the conftest, not the lightweight `tiny_model` fixtures. Still no downloads - everything constructed in code.

**Distillation trainer:**
- One training step produces a finite scalar loss
- Model parameters change after a step
- Teacher parameters frozen when `freeze_teacher=True`
- Teacher parameters update when `freeze_teacher=False` (co-training)
- In co-training, teacher optimizer group exists in `create_optimizer()`
- `_detach_teacher_intermediates()` detaches hidden_states and attentions but preserves logits

**P0 gradient flow (most valuable tests in the suite):**
- `test_teacher_signal_changes_model_gradients`: Run step with teacher, run step without. Model gradients differ.
- `test_co_training_teacher_gets_ce_only`: In co-training, teacher gradients come from CE loss, not KL.
- `test_feature_loss_not_self_comparison`: Feature loss uses teacher hidden states, not model's own.

**Eval behavior:**
- `prediction_step()` disables hidden_states/attentions output (OOM prevention)

### 3. Config (`src/utils/config.py`)

- Load a single YAML file, get a dict
- Child inherits from parent: child values override, parent values preserved
- Deep merge: nested dicts merge recursively, scalars overwrite
- Multiple inheritance: `inherit: [a.yaml, b.yaml]` merges left-to-right
- Circular inheritance: max depth guard prevents infinite recursion
- Missing file raises

### 4. Registry (`src/tasks/registry.py`)

- Every task has: num_labels, type, primary_metric
- `is_classification_task` is mutually exclusive with other task-type categorizations
- No task appears in multiple type categories
- Factory functions return correct values for known tasks, None for unknown

### 5. Model Loading (`src/models/loader.py`)

Pure routing logic tested with mocks. Forward-pass correctness tested in Trainer (Section 2).

- `needs_attention_outputs()` correctly parses config for attention/feature loss types
- `_detect_attn_implementation()`: eager when requested, sdpa on CUDA, None otherwise
- Task routing: classification -> `_load_classifier`

### 6. Checkpoint (`src/utils/checkpoint.py`)

- Save then load round-trips: completed_epochs, completed_steps preserved
- `is_complete()`: True when target reached, False before
- Corrupt JSON recovers gracefully to fresh state
- `get_resume_checkpoint()`: finds latest numeric checkpoint dir, skips non-numeric (e.g., `checkpoint-best`)
- `mark_checkpoint_complete()`: writes atomically (temp file + os.replace)

### 7. Evaluation (`src/evaluation/evaluate.py`, `src/evaluation/metrics.py`)

Uses `tiny_bert_model` + a synthetic dataset (random input_ids, attention_mask, labels).

- `evaluate_model()` returns a dict with expected metric keys and finite values
- Model restored to original training mode after eval
- `log_transfer_metrics()` computes correct teacher advantage, model gain, transfer efficiency
- `get_compute_metrics()` routes to `compute_classification_metrics`

### 8. Collators (`src/data/collators.py`)

- Default collation: output has `input_ids`, `attention_mask`, `labels` with matching batch dim

### 9. Layer Mapping (`src/training/layer_mapping.py`)

**Parametrized across all strategies (Uniform, LastN, FirstLast, Custom):**
- Output length equals model layer count
- All teacher indices are in bounds [0, teacher_layers)
- All model indices are in bounds [0, model_layers)
- `validate()` returns True for valid inputs, False when teacher < model

**Strategy-specific:**
- Uniform: equal layers map to identity
- LastN: uses final teacher layers
- FirstLast: first pair is (0, 0), last pair is (teacher-1, model-1)
- Custom: preserves user-specified indices exactly

### 10. Projections (`src/training/projections.py`)

**Parametrized across Identity, Linear, MLP:**
- Output shape matches expected dimensions (2D and 3D input)
- Gradient flows through projection (non-identity only)
- `create_projection_if_needed()`: returns Identity when dims match, specified type otherwise

**Factory:**
- `get_projection()` returns correct type for each name

### 11. Environment (`src/utils/environment.py`)

- `detect_environment()`: returns "slurm" when SLURM_JOB_ID set, "colab" when IN_COLAB set, "local" otherwise
- `get_device()`: returns "cuda", "mps", or "cpu"
- Precedence: SLURM > Colab > local

---

## Running Tests

```bash
# Fast iteration (single file, no coverage)
pytest tests/unit/test_losses.py --no-cov

# All core tests
pytest tests/unit/ --no-cov

# Full suite with coverage
pytest tests/unit/

# GPU integration (on cluster)
pytest tests/integration/ -m gpu --no-cov

# By marker
pytest -m "not gpu"        # skip GPU tests
pytest -m "not slow"       # skip slow tests
```

---

## Markers

| Marker   | Meaning                                    |
|----------|---------------------------------------------|
| `gpu`    | Requires accelerator (CUDA or MPS)         |
| `slow`   | Takes > 30s                                |

---

## Coverage

Coverage threshold (`--cov-fail-under`) should be enforced in CI only, not in `pyproject.toml` global addopts. Developers running a single test file should not hit coverage failures.

Target: 80%+ on Core modules.

---

## Future Plans

Not in scope now. Add when needed.

- **Extension tests**: Dataset handlers, callbacks, formatters.
- **Workbench tests**: CKA, probing, attention entropy, disagreement analysis.
- **Compatibility tests** (`tests/compat/`): Validate DistilBERT/BERT against the framework.
- **Performance benchmarks**: Training step throughput, memory usage regression checks.
