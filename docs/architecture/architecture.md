# Knowledge Distillation Framework Architecture

Complete reference for the knowledge distillation framework.

---

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              distill.py (Entry Point)                         │
│  - Parses CLI arguments                                                     │
│  - Loads config with inheritance                                            │
│  - Orchestrates training pipeline                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│   src/data/         │   │   src/models/       │   │   src/training/     │
│   datasets.py       │   │   loader.py         │   │   trainer.py        │
│                     │   │                     │   │   losses.py         │
│   Load & tokenize   │   │   Load teacher &    │   │   layer_mapping.py  │
│   datasets          │   │   student models    │   │   projections.py    │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
```

---

## Module Reference

### Data Module (`src/data/datasets.py`)

**Contract:** Every dataset loader returns `(train_dataset, eval_dataset, tokenizer)`

**Built-in Loaders:**

| Loader | Task | Dataset | Key Features |
|--------|------|---------|--------------|
| `load_mnli()` | 3-class NLI | GLUE MNLI | Premise + hypothesis pairs |
| `load_qqp()` | Binary classification | GLUE QQP | Question pairs |
| `load_europarl()` | EN→FR translation | Europarl | Custom end token |

**Adding Custom Dataset:**

```python
def load_my_task(tokenizer, max_length=128):
    dataset = load_dataset("dataset_name")

    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(preprocess, batched=True)
    return tokenized["train"], tokenized["validation"], tokenizer

# Register in load_dataset_for_task()
```

---

### Models Module (`src/models/loader.py`)

**Entry Point:** `load_teacher_student(config, device, tokenizer, num_labels)`

**Student Initialization Options:**

| Option | Description | Use Case |
|--------|-------------|----------|
| `pretrained` | Load pretrained weights | Faster convergence |
| `fresh` | Random initialization | True compression |

---

### Training Module (`src/training/`)

#### Trainer (`trainer.py`)

```python
class DistillationTrainer(Trainer):
    def __init__(
        self,
        teacher_model,
        temperature=2.0,
        alpha=0.5,
        freeze_teacher=True,
        loss_strategy=None,
        teacher_label_map=None,
        layer_mapping=None,
        projector=None,
    ):
```

**Training Modes:**

| Mode | `freeze_teacher` | Description |
|------|------------------|-------------|
| Traditional | `True` | Teacher frozen, fixed soft labels |
| Co-training | `False` | Teacher trains alongside student |

#### Loss Strategies (`losses.py`)

All losses implement `DistillationLossStrategy`:

| Strategy | Formula | Use Case |
|----------|---------|----------|
| `kl_divergence` | `(1-α)·CE + α·T²·KL(s/T ‖ t/T)` | Standard Hinton distillation |
| `mse_logits` | `(1-α)·CE + α·MSE(s, t)` | Direct logit matching |
| `feature_mse` | `(1-α)·CE + α·MSE(hidden)` | FitNets-style |
| `attention_kl` | `(1-α)·CE + α·KL(attn)` | Attention transfer |

**Adding Custom Loss:**

```python
class MyCustomLoss(DistillationLossStrategy):
    @property
    def name(self) -> str:
        return "my_custom_loss"

    def compute(self, student_logits, teacher_logits, ce_loss, **kwargs):
        custom_loss = ...
        return LossOutput(
            total_loss=(1 - self.alpha) * ce_loss + self.alpha * custom_loss,
            components={"ce": ce_loss, "custom": custom_loss}
        )

LOSS_REGISTRY["my_custom_loss"] = MyCustomLoss
```

#### Layer Mapping (`layer_mapping.py`)

Maps teacher layers to student layers for feature/attention distillation.

| Strategy | Example (12→6) | Rationale |
|----------|----------------|-----------|
| `uniform` | [1,3,5,7,9,11] | Every Nth teacher layer |
| `last_n` | [6,7,8,9,10,11] | Later = more task-specific |
| `first_last` | [0,2,4,7,9,11] | Preserve low + high level |

#### Projections (`projections.py`)

Align dimensions when teacher/student have different hidden sizes.

| Strategy | Description | Learnable |
|----------|-------------|-----------|
| `identity` | Pass-through | No |
| `linear` | Single linear layer | Yes |
| `mlp` | Two-layer MLP | Yes |

---

### Evaluation Module (`src/evaluation/metrics.py`)

```python
def get_compute_metrics(task: str, tokenizer=None):
    # Returns appropriate metrics function
    # 'mnli', 'qqp' → accuracy, F1
    # 'translation' → BLEU
```

---

### Utils Module (`src/utils/`)

**Config (`config.py`):** YAML with inheritance via `inherit:` key

**Environment (`environment.py`):** Device detection (CUDA/MPS/CPU)

**Checkpoint (`checkpoint.py`):** Training state persistence, resumption

---

## Configuration System

### Config Schema

```yaml
model:
  teacher: "model-name"
  student: "model-name"
  student_init: pretrained

data:
  task: mnli
  max_length: 128

training:
  num_epochs: 3
  batch_size: 16
  learning_rate: 2e-5
  fp16: true

distillation:
  temperature: 2.0
  alpha: 0.5
  freeze_teacher: true
  loss_type: kl_divergence
  layer_mapping:
    type: uniform
  projection:
    type: linear
```

---

## Extension Points

| Component | How to Extend | Register At |
|-----------|---------------|-------------|
| Dataset | Add `load_xxx()` function | `load_dataset_for_task()` |
| Loss Strategy | Subclass `DistillationLossStrategy` | `LOSS_REGISTRY` |
| Layer Mapping | Implement `map()` method | `LAYER_MAPPING_REGISTRY` |
| Projection | Subclass `nn.Module` | `PROJECTION_REGISTRY` |
| Metrics | Add `compute_metrics_xxx()` | `get_compute_metrics()` |

---

## CLI Interface

```bash
# New run
python distill.py --config configs/mnli/.../ablations/00.yaml

# Named run
python distill.py --run_name my_experiment --config configs/...

# Resume existing
python distill.py --run_name existing_run

# Limit steps
python distill.py --config configs/... --max_steps 100
```

---

## Design Patterns

### Strategy Pattern
Swap losses, layer mapping, projections via config.

### Factory Pattern
Registries + `get_xxx()` functions for instantiation.

### Template Method
`compute_loss()` defines algorithm skeleton, delegates to strategy.

### Configuration Inheritance
YAML `inherit:` for DRY configs.

---

## Quick Reference: Adding a New Task

1. **Dataset** — Add loader in `src/data/datasets.py`
2. **Metrics** — Add compute function in `src/evaluation/metrics.py`
3. **Config** — Create `configs/{task}/...yaml`
4. **Labels** — Set `num_labels` if classification
