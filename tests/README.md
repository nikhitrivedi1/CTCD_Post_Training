# Test Architecture

## Overview

Forge tests are organized into concentric rings. **Core** is tested thoroughly with property-based, parametrized tests. Extensions, compatibility, and benchmarks are added later as needed.

```
┌─────────────────────────────────────────────────────┐
│               Compatibility (future)                │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │             Extensions (future)               │  │
│  │                                               │  │
│  │  ┌─────────────────────────────────────────┐  │  │
│  │  │                 CORE                    │  │  │
│  │  │                                         │  │  │
│  │  │  losses, trainer, config,               │  │  │
│  │  │  registry, loader, checkpoint,          │  │  │
│  │  │  evaluate, metrics, collators,          │  │  │
│  │  │  layer_mapping, projections,            │  │  │
│  │  │  environment                            │  │  │
│  │  └─────────────────────────────────────────┘  │  │
│  │                                               │  │
│  │  handlers, rewards, callbacks,                │  │
│  │  formatters, inference engine,                │  │
│  │  workbench (CKA, probing, etc.)               │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  Llama, Qwen VLM, DistilBERT, LoRA adapters         │
└─────────────────────────────────────────────────────┘
```

### Core

The 12 modules every training run touches. Tested to ~99% coverage.

| Module        | Source                           | What it does                                                                            |
|---------------|----------------------------------|-----------------------------------------------------------------------------------------|
| Losses        | `src/training/losses.py`         | 9 distillation loss strategies (KL, MSE, cosine, feature, attention, combined, staged)  |
| Trainer       | `src/training/trainer.py`        | DistillationTrainer: model + teacher forward, loss computation, co-training             |
| Config        | `src/utils/config.py`            | YAML loading with inheritance and deep merge                                            |
| Registry      | `src/tasks/registry.py`          | Task metadata: num_labels, type, primary_metric                                         |
| Loader        | `src/models/loader.py`           | Model loading, task routing, checkpoint resolution                                      |
| Checkpoint    | `src/utils/checkpoint.py`        | Training state persistence, resume logic                                                |
| Evaluate      | `src/evaluation/evaluate.py`     | Evaluation pipeline, transfer metrics                                                   |
| Metrics       | `src/evaluation/metrics.py`      | Classification, translation, OCR metric computation                                     |
| Collators     | `src/data/collators.py`          | Data collation with prompt masking                                                      |
| Layer Mapping | `src/training/layer_mapping.py`  | Teacher-to-model layer correspondence strategies                                        |
| Projections   | `src/training/projections.py`    | Dimension alignment for feature distillation                                            |
| Environment   | `src/utils/environment.py`       | Device detection, SLURM/Colab/local routing                                             |

### Extensions (future)

Pluggable, task-specific modules. Tested at their boundary with one contract test per extension.

- Dataset handlers (GLUE, Europarl, OCR handlers)
- Reward functions (GRPO/RL)
- Callbacks (SLURM signal, teacher eval, peak metrics)
- Formatters (SFT/RL data formatting)
- Inference engine (deployment pipeline)
- Workbench (CKA, probing, attention analysis, disagreement)

### Compatibility (future)

Validates specific model architectures against the framework. Downloads real models, requires network + GPU.

- Llama, Qwen VLM, DistilBERT
- LoRA adapter loading and merging

---

## Principles

1. **Core first.** Thorough, property-based tests. Everything else waits.
2. **Parametrize, don't copy-paste.** One test function with 9 strategy inputs beats 9 identical functions.
3. **Test properties, not examples.** "Loss is finite and differentiable" across all strategies, not "KL with alpha=0.5 and T=2 gives 1.3847".
4. **No network calls.** Models constructed from configs with random init. Datasets are synthetic.
5. **No mocking internals.** Use real tiny modules where possible. Mock only at system boundaries (HF Hub, filesystem, SLURM).
6. **Naming follows context.** `model`/`teacher` in distillation tests. `model_logits`/`target_logits` in pure math tests.

---

## Fixtures

All in `tests/conftest.py`. Constructed in code, nothing stored on disk.

**Tensor fixtures** for pure math tests (losses, projections, layer mapping):
- `model_logits`, `target_logits` - classification shape `(4, 3)`
- `lm_model_logits`, `lm_target_logits` - language model shape `(4, 16, 128)`
- `hidden_states_model`, `hidden_states_target` - per-layer hidden states
- `attention_weights_model`, `attention_weights_target` - softmax-normalized
- `ce_loss` - scalar cross-entropy loss

**Model fixtures** for tests needing HF Trainer compatibility (trainer, evaluation):
- `tiny_bert_model` - 2-layer `BertForSequenceClassification`, random init
- `tiny_bert_teacher` - 4-layer `BertForSequenceClassification`, random init

No fixtures downloaded. No fixtures stored on disk.

---

## File Layout

```text
tests/
  conftest.py                # shared fixtures
  TEST_PLAN.md               # test plan (core sections + future plans)
  README.md                  # this file
  unit/
    test_losses.py           # 9 loss strategies, parametrized universals + KL math
    test_trainer.py          # trainer basics, P0 gradient flow, issue #35 regression
    test_config.py           # YAML deep merge, inheritance, circular guard
    test_registry.py         # structure, mutual exclusivity, factory functions
    test_loader.py           # routing, attention config, mock model loading
    test_checkpoint.py       # round-trip, is_complete, corrupt recovery, resume
    test_evaluate.py         # evaluate_model, transfer metrics, classification/OCR metrics
    test_collators.py        # VLM prompt masking, padding, label creation
    test_layer_mapping.py    # 4 strategies parametrized, bounds, validation
    test_projections.py      # 3 types parametrized, gradient flow, factories
    test_environment.py      # device detection, SLURM/Colab/local, num_workers
```

---

## Commands

```bash
# ── Development ──────────────────────────────────────────

# All core tests, fast (no coverage measurement)
pytest tests/unit/ --no-cov

# Single file, fast iteration
pytest tests/unit/test_losses.py --no-cov

# Single test class or function
pytest tests/unit/test_trainer.py::TestGradientFlow -v --no-cov

# ── Coverage ─────────────────────────────────────────────

# Core modules with coverage report
pytest tests/unit/

# Coverage for a specific module
pytest tests/unit/test_trainer.py --cov=src/training/trainer.py --cov-report=term-missing

# ── Filtering ────────────────────────────────────────────

# Skip GPU tests
pytest -m "not gpu"

# Skip slow tests
pytest -m "not slow"

# Show only failures
pytest tests/unit/ --no-cov -q --tb=short

# ── Debugging ────────────────────────────────────────────

# Stop on first failure
pytest tests/unit/ --no-cov -x

# Show print output
pytest tests/unit/test_trainer.py --no-cov -s

# Run with full traceback
pytest tests/unit/ --no-cov --tb=long
```

---

## Markers

| Marker  | Meaning                                    |
|---------|--------------------------------------------|
| `gpu`   | Requires accelerator (CUDA or MPS)         |
| `slow`  | Takes > 30s                                |
| `xfail` | Known failure, tracked by issue number     |

---

## Coverage

Coverage is measured automatically via `--cov=src` in `pyproject.toml`. The coverage threshold (`--cov-fail-under`) is enforced in CI only, not in local dev. Use `--no-cov` during development for speed.

### Current Core Coverage

| Module           | Coverage |
|------------------|----------|
| losses.py        | 99%      |
| trainer.py       | 100%     |
| config.py        | 100%     |
| registry.py      | 100%     |
| loader.py        | 98%      |
| checkpoint.py    | 100%     |
| evaluate.py      | 94%      |
| metrics.py       | 99%      |
| collators.py     | 100%     |
| layer_mapping.py | 100%     |
| projections.py   | 100%     |
| environment.py   | 100%     |

Remaining uncovered: abstract method stubs (losses.py), W&B logging branches (evaluate.py), one dead code branch (metrics.py), CUDA-only path (loader.py).
