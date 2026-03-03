# Forge Architecture

Complete reference for Forge, covering all modules, extension points, and implementation details.

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Module Reference](#module-reference)
   - [Data Module](#1-data-module)
   - [Models Module](#2-models-module)
   - [Training Module](#3-training-module)
   - [Evaluation Module](#4-evaluation-module)
   - [Utils Module](#5-utils-module)
3. [Configuration System](#6-configuration-system)
4. [Data Flow](#data-flow)
5. [Extension Points](#extension-points)
6. [CLI Interface](#cli-interface)
7. [Design Patterns](#design-patterns)
8. [Future Extensions](#future-extensions)

---

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              forge.py (Entry Point)                         │
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
└─────────────────────┘   └─────────────────────┘   │   callbacks.py      │
                                                    └─────────────────────┘
          │                           │                           │
          └───────────────────────────┼───────────────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           src/evaluation/                                    │
│                           metrics.py                                         │
│   Compute accuracy, F1, BLEU, etc.                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              src/utils/                                      │
│   config.py      - YAML loading with inheritance                            │
│   environment.py - Device detection (CUDA/MPS/CPU)                          │
│   checkpoint.py  - Training state persistence                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Reference

### 1. Data Module

**Location:** `src/data/datasets.py`

Handles dataset loading, preprocessing, and tokenization.

#### Contract

Every dataset loader must return:
```python
(train_dataset, eval_dataset, tokenizer)
```

Where datasets contain these columns after processing:
- `input_ids` — Tokenized input
- `attention_mask` — Attention mask
- `labels` — Targets (class IDs, regression values, or token IDs)

#### Entry Point

```python
def load_dataset_for_task(config: dict):
    """
    Load dataset based on config specification.

    1. Reads config["data"]["task"] to determine task type
    2. Loads tokenizer from config["model"]["teacher"]
    3. Dispatches to task-specific loader
    4. Returns (train, eval, tokenizer)
    """
```

#### Built-in Loaders

| Loader | Task | Dataset | Key Features |
|--------|------|---------|--------------|
| `load_mnli()` | 3-class NLI | GLUE MNLI | Premise + hypothesis pairs |
| `load_qqp()` | Binary classification | GLUE QQP | Question pairs, filters -1 labels |
| `load_europarl()` | EN→FR translation | Europarl | Custom end token, prompt template |

#### Adding a Custom Dataset

```python
def load_my_task(tokenizer: PreTrainedTokenizer, max_length: int = 128):
    # Option 1: HuggingFace Hub
    dataset = load_dataset("dataset_name", "config")

    # Option 2: Local files
    dataset = load_dataset("json", data_files={
        "train": "train.json",
        "val": "val.json"
    })

    # Option 3: From Python objects
    from datasets import Dataset
    dataset = Dataset.from_dict({"text": [...], "label": [...]})

    def preprocess(examples):
        return tokenizer(
            examples["text_column"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

    tokenized = dataset.map(preprocess, batched=True,
                            remove_columns=dataset.column_names)
    tokenized = tokenized.rename_column("label", "labels")

    return tokenized["train"], tokenized["validation"], tokenizer

# Register in load_dataset_for_task():
elif task_name == "my_task":
    return load_my_task(tokenizer)
```

#### Data Sources Supported

| Source | Load Command |
|--------|--------------|
| HuggingFace Hub | `load_dataset("dataset_name")` |
| Local JSON | `load_dataset("json", data_files="data.json")` |
| Local CSV | `load_dataset("csv", data_files="data.csv")` |
| Local Parquet | `load_dataset("parquet", data_files="data.parquet")` |
| Python dict | `Dataset.from_dict({"col": [...]})` |
| Pandas DataFrame | `Dataset.from_pandas(df)` |

---

### 2. Models Module

**Location:** `src/models/loader.py`

Handles loading and configuring teacher/student model pairs.

#### Entry Point

```python
def load_teacher_student(config: dict, device: str, tokenizer, num_labels: int = None):
    """
    Load teacher and student models based on config.

    Dispatches to:
    - _load_classification_models() for BERT/DistilBERT
    - _load_translation_models() for Llama (with optional LoRA)

    Returns: (teacher, student)
    """
```

#### Classification Models

```python
def _load_classification_models(config, device, tokenizer, num_labels):
    # Teacher: Always pretrained
    teacher = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["teacher"],
        num_labels=num_labels
    )
    teacher.eval()  # Always frozen initially

    # Student: Pretrained or fresh
    if config["model"]["student_init"] == "fresh":
        # Random initialization - harder but better compression
        student_config = AutoConfig.from_pretrained(config["model"]["student"])
        student = AutoModelForSequenceClassification.from_config(student_config)
    else:
        # Pretrained - easier convergence
        student = AutoModelForSequenceClassification.from_pretrained(...)

    return teacher, student
```

#### Translation Models (Causal LM)

```python
def _load_translation_models(config, device, tokenizer):
    # Handles:
    # - fp16 for CUDA
    # - Embedding resizing for custom tokens
    # - Optional LoRA for parameter-efficient training

    if config["training"]["use_lora"]:
        lora_config = LoraConfig(
            r=config["training"]["lora"]["r"],
            lora_alpha=config["training"]["lora"]["alpha"],
            target_modules=["q_proj", "v_proj"],
            ...
        )
        student = get_peft_model(student, lora_config)
```

#### Student Initialization Options

| Option | Description | Use Case |
|--------|-------------|----------|
| `pretrained` | Load pretrained weights | Faster convergence, easier |
| `fresh` | Random initialization | True compression, harder |

---

### 3. Training Module

**Location:** `src/training/`

The core distillation logic using the Strategy Pattern for maximum extensibility.

#### 3.1 Trainer (`trainer.py`)

Extends HuggingFace `Trainer` with distillation capabilities.

```python
class DistillationTrainer(Trainer):
    """
    Custom Trainer for knowledge distillation.

    Key features:
    1. Maintains teacher model (frozen or co-trained)
    2. Uses pluggable loss strategies (Strategy Pattern)
    3. Logs loss components separately for analysis
    """

    def __init__(
        self,
        teacher_model,           # The teacher
        temperature=2.0,         # Softmax temperature
        alpha=0.5,              # KL vs CE weight
        freeze_teacher=True,     # Traditional vs co-training
        loss_strategy=None,      # Pre-configured strategy
        loss_type=None,          # Strategy name for factory
        teacher_label_map=None,  # Remap teacher logits
        layer_mapping=None,      # For feature/attention distillation
        projector=None,          # For dimension alignment
        ...
    ):
```

**Key Method: `compute_loss()`**

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    # 1. Forward pass through student
    outputs = model(**inputs)
    student_logits = outputs.logits

    # 2. Forward pass through teacher
    if self.freeze_teacher:
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
    else:
        teacher_outputs = self.teacher(**inputs)  # Gradients flow

    # 3. Remap teacher logits if needed
    if self.teacher_label_map:
        teacher_logits = teacher_logits[:, self.teacher_label_map]

    # 4. Compute loss using strategy
    loss_output = self.loss_strategy.compute(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        ce_loss=outputs.loss,
        student_outputs=outputs,
        teacher_outputs=teacher_outputs,
    )

    # 5. Log components
    self.log({"loss_total": loss_output.total_loss, ...})

    return loss_output.total_loss
```

**Training Modes**

| Mode | `freeze_teacher` | Description |
|------|------------------|-------------|
| Traditional | `True` | Teacher frozen, provides fixed soft labels |
| Co-training | `False` | Teacher trains alongside student |

#### 3.2 Loss Strategies (`losses.py`)

All losses implement the `DistillationLossStrategy` interface:

```python
class DistillationLossStrategy(ABC):
    requires_hidden_states: bool = False  # Set True for feature distillation
    requires_attentions: bool = False     # Set True for attention distillation

    @abstractmethod
    def compute(
        self,
        student_logits, teacher_logits, ce_loss,
        labels=None, student_outputs=None, teacher_outputs=None,
        **kwargs
    ) -> LossOutput:
        """Return LossOutput(total_loss, components_dict)"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier"""
```

**Built-in Strategies**

| Strategy | Class | Formula | Use Case |
|----------|-------|---------|----------|
| KL Divergence | `KLDistillationLoss` | `(1-α)·CE + α·T²·KL(s/T ‖ t/T)` | Standard Hinton distillation |
| MSE Logits | `MSEDistillationLoss` | `(1-α)·CE + α·MSE(s, t)` | Direct logit matching |
| Cosine | `CosineDistillationLoss` | `(1-α)·CE + α·(1-cos(s,t))` | Direction matching |
| Soft Target | `SoftTargetDistillationLoss` | `(1-α)·CE_hard + α·CE_soft` | Cross-entropy with soft labels |
| Feature MSE | `FeatureLossStrategy` | `(1-α)·CE + α·MSE(hidden)` | FitNets-style intermediate matching |
| Attention KL | `AttentionLossStrategy` | `(1-α)·CE + α·KL(attn)` | Attention pattern transfer |
| Combined | `CombinedDistillationLoss` | Weighted sum | Mix multiple strategies |

**Loss Registry**

```python
LOSS_REGISTRY = {
    "kl_divergence": KLDistillationLoss,
    "mse_logits": MSEDistillationLoss,
    "cosine_embedding": CosineDistillationLoss,
    "soft_target": SoftTargetDistillationLoss,
    "feature_mse": FeatureLossStrategy,
    "attention_kl": AttentionLossStrategy,
}

# Factory function
loss = get_loss_strategy("kl_divergence", temperature=2.0, alpha=0.5)
```

**Adding a Custom Loss**

```python
class MyCustomLoss(DistillationLossStrategy):
    def __init__(self, alpha=0.5, my_param=1.0):
        self.alpha = alpha
        self.my_param = my_param

    @property
    def name(self) -> str:
        return "my_custom_loss"

    def compute(self, student_logits, teacher_logits, ce_loss, **kwargs):
        # Your loss computation
        custom_loss = ...  # e.g., some novel divergence

        ce_weighted = (1 - self.alpha) * ce_loss
        custom_weighted = self.alpha * custom_loss

        return LossOutput(
            total_loss=ce_weighted + custom_weighted,
            components={
                "ce_weighted": ce_weighted,
                "custom_weighted": custom_weighted,
                "ce_raw": ce_loss,
                "custom_raw": custom_loss,
            }
        )

# Register it
LOSS_REGISTRY["my_custom_loss"] = MyCustomLoss
```

#### 3.3 Layer Mapping (`layer_mapping.py`)

Maps teacher layers to student layers for feature/attention distillation.

```
Teacher (12 layers)    Student (6 layers)
       │                      │
   ┌───┴───┐              ┌───┴───┐
   0  1  2  ...11         0  1  ...5
       │                      │
       └──── Mapping ─────────┘
```

**Built-in Strategies**

| Strategy | Example (12→6) | Rationale |
|----------|----------------|-----------|
| `uniform` | [1,3,5,7,9,11] → [0,1,2,3,4,5] | Every Nth teacher layer |
| `last_n` | [6,7,8,9,10,11] → [0,1,2,3,4,5] | Later layers = more task-specific |
| `first_last` | [0,2,4,7,9,11] → [0,1,2,3,4,5] | Preserve low-level + high-level |
| `custom` | User-defined | Full control |

**Usage**

```python
from src.training.layer_mapping import get_layer_mapping

mapper = get_layer_mapping("uniform")
mapping = mapper.map(teacher_layers=12, student_layers=6)
# [(1, 0), (3, 1), (5, 2), (7, 3), (9, 4), (11, 5)]
```

#### 3.4 Projections (`projections.py`)

Align dimensions when teacher and student have different hidden sizes.

| Strategy | Description | Learnable |
|----------|-------------|-----------|
| `identity` | Pass-through (dims match) | No |
| `linear` | Single linear layer | Yes |
| `mlp` | Two-layer MLP with activation | Yes |

**Usage**

```python
from src.training.projections import create_projection_if_needed

projector = create_projection_if_needed(
    teacher_dim=1024,  # BERT-large
    student_dim=768,   # DistilBERT
    projection_type="linear"
)
# Returns LinearProjection(1024, 768)

projector = create_projection_if_needed(
    teacher_dim=768,
    student_dim=768,
    projection_type="linear"
)
# Returns IdentityProjection (no transform needed)
```

#### 3.5 Callbacks (`callbacks.py`)

**SLURMSignalCallback**

Handles graceful termination on HPC clusters:

```python
class SLURMSignalCallback(TrainerCallback):
    """
    Catches SIGUSR1/SIGTERM signals sent by SLURM before job timeout.
    Saves checkpoint and stops training cleanly.

    Configure SLURM with: --signal=SIGUSR1@120 (2 min warning)
    """
```

---

### 4. Evaluation Module

**Location:** `src/evaluation/metrics.py`

Task-specific metric computation.

```python
def get_compute_metrics(task: str, tokenizer=None):
    """
    Returns appropriate compute_metrics function for task.

    - 'mnli', 'qqp' → compute_metrics_classification (accuracy, F1)
    - 'translation' → compute_metrics_translation (BLEU)
    """
```

**Classification Metrics**

```python
def compute_metrics_classification(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = evaluate.load("accuracy").compute(...)
    f1 = evaluate.load("f1").compute(..., average="weighted")

    return {"accuracy": ..., "f1": ...}
```

**Translation Metrics**

```python
def compute_metrics_translation(tokenizer):
    def compute_metrics(eval_pred):
        # Decode predictions and references
        # Compute BLEU score
        return {"bleu": ...}
    return compute_metrics
```

**Adding Custom Metrics**

```python
def compute_metrics_regression(eval_pred):
    """For regression tasks like STS-B."""
    predictions, labels = eval_pred

    mse = np.mean((predictions - labels) ** 2)
    pearson = np.corrcoef(predictions, labels)[0, 1]
    spearman = scipy.stats.spearmanr(predictions, labels).correlation

    return {"mse": mse, "pearson": pearson, "spearman": spearman}

# Register in get_compute_metrics():
elif task in ["stsb", "regression"]:
    return compute_metrics_regression
```

---

### 5. Utils Module

**Location:** `src/utils/`

#### 5.1 Config (`config.py`)

YAML configuration with inheritance support:

```python
def load_config(config_path: str) -> dict:
    """
    Load config with inheritance.

    If config has 'inherit: base.yaml', loads parent first,
    then deep-merges child values on top.
    """

def _deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries, with override taking precedence.
    Nested dicts are merged recursively.
    """
```

**Example Inheritance**

```yaml
# configs/base.yaml
training:
  batch_size: 16
  learning_rate: 2e-5

# configs/mnli/custom.yaml
inherit: ../base.yaml
training:
  batch_size: 32  # Override
# Result: batch_size=32, learning_rate=2e-5 (inherited)
```

#### 5.2 Environment (`environment.py`)

```python
def detect_environment() -> str:
    """Returns 'colab', 'slurm', or 'local'"""

def get_device() -> str:
    """Returns 'cuda', 'mps', or 'cpu'"""
```

#### 5.3 Checkpoint (`checkpoint.py`)

```python
class TrainingState:
    """
    Tracks training progress for resumption.

    - Persists epoch/step to disk
    - Detects completion
    - Finds checkpoint to resume from
    """

    def save(self, epoch: float, step: int): ...
    def is_complete(self) -> bool: ...
    def get_resume_checkpoint(self, output_dir: str) -> str | None: ...
```

---

## 6. Configuration System

### Directory Structure

```
configs/
├── base.yaml                    # Shared defaults
├── TEMPLATE.yaml               # Documented template
├── mnli/
│   └── bert_distilbert/
│       ├── ablations/          # Experiment configs
│       │   ├── 00_no-teacher_ce-only.yaml
│       │   ├── 01_frozen_base-teacher_kl.yaml
│       │   └── ...
│       └── gpu/                # Hardware-specific
│           ├── a100/
│           ├── h200/
│           └── p100/
├── qqp/
│   └── bert_distilbert/
│       └── ablations/
└── translation/
    └── llama3p2-3b_llama3p2-1b/
```

### Config Schema

```yaml
# Required sections
model:
  teacher: "model-name"         # HuggingFace model or path
  student: "model-name"
  student_init: pretrained      # 'pretrained' or 'fresh'

data:
  task: mnli                    # Task name for loader dispatch
  max_length: 128

training:
  num_epochs: 3
  batch_size: 16
  learning_rate: 2e-5
  warmup_steps: 200
  weight_decay: 0.01
  max_grad_norm: 2.0
  gradient_accumulation_steps: 1
  fp16: true

distillation:
  temperature: 2.0              # Softmax temperature
  alpha: 0.5                    # KL weight (0=CE only, 1=KL only)
  freeze_teacher: true          # false for co-training
  loss_type: kl_divergence      # Loss strategy name

  # Optional: for feature/attention distillation
  layer_mapping:
    type: uniform               # uniform, last_n, first_last, custom
  projection:
    type: linear                # identity, linear, mlp

checkpointing:
  save_steps: 500
  save_total_limit: 3

logging:
  report_to: wandb
  logging_steps: 100

evaluation:
  eval_steps: 500
  eval_strategy: steps
```

---

## Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Config    │────▶│   Loader    │────▶│  Tokenizer  │
│   (YAML)    │     │  (models)   │     │  (teacher)  │
└─────────────┘     └─────────────┘     └─────────────┘
                          │                    │
                          ▼                    ▼
                    ┌───────────┐        ┌───────────┐
                    │  Teacher  │        │  Dataset  │
                    │   Model   │        │ (tokenized)│
                    └───────────┘        └───────────┘
                          │                    │
                          ▼                    ▼
┌─────────────────────────────────────────────────────────┐
│              DistillationTrainer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │   Student   │  │   Loss      │  │   Layer         │  │
│  │   Model     │  │   Strategy  │  │   Mapping       │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │   Metrics / W&B     │
              │   Checkpoints       │
              └─────────────────────┘
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
| Callback | Subclass `TrainerCallback` | `trainer.add_callback()` |

---

## CLI Interface

```bash
# New run (auto-named with timestamp)
python forge.py --config configs/mnli/bert_distilbert/ablations/00.yaml

# Named run
python forge.py --run_name my_experiment --config configs/...

# With timestamp suffix
python forge.py --run_name exp --config configs/... --timestamp
# Creates: exp_2026-01-31_12-00Z

# Resume existing run
python forge.py --run_name existing_run

# Force overwrite existing run
python forge.py --run_name old_run --config configs/... --force

# Limit training steps (for testing)
python forge.py --config configs/... --max_steps 100

# Set random seed
python forge.py --config configs/... --seed 123
```

---

## Design Patterns

### 1. Strategy Pattern

Used for losses, layer mapping, and projections. Swap implementations via config without modifying core code.

```python
# Example: Swap loss strategy via config
distillation:
  loss_type: kl_divergence  # or mse_logits, cosine_embedding, etc.
```

### 2. Factory Pattern

Registries + `get_xxx()` functions for instantiation.

```python
loss = get_loss_strategy("kl_divergence", temperature=2.0)
mapper = get_layer_mapping("uniform")
projector = get_projection("linear", input_dim=1024, output_dim=768)
```

### 3. Template Method

`DistillationTrainer.compute_loss()` defines the algorithm skeleton:
1. Get student predictions
2. Get teacher predictions
3. Remap labels if needed
4. Delegate to loss strategy
5. Log components

### 4. Configuration Inheritance

YAML `inherit:` for DRY configs.

```yaml
# Child config
inherit: base.yaml
training:
  batch_size: 32  # Override parent value
```

---

## Future Extensions

### Multi-GPU Support

```
Current:  Single GPU training
Planned:  HuggingFace Accelerate integration
          - DataParallel for teacher (frozen, no gradients)
          - DDP for student
```

### Seq2Seq Models

```
Current:  Classification and causal LM only
Planned:  T5, BART, mBART support
          - Encoder and decoder distillation
          - Cross-attention distillation
```

### Progressive Distillation

```
Current:  Single-stage distillation
Planned:  Multi-stage compression
          - BERT → 9-layer → 6-layer → 3-layer
          - Curriculum on loss weights
```

### Additional Design Patterns

| Pattern | When to Add | Use Case |
|---------|-------------|----------|
| **Composite** | When presets needed | Group R+F as "distilbert" preset |
| **Builder** | For programmatic/test use | `.with_response().with_feature().build()` |
| **Registry** | For plugin extensibility | Third-party strategies via decorators |
| **Decorator** | For cross-cutting concerns | Logging, gradient clipping wrappers |

---

## Quick Reference: Adding a New Task

1. **Dataset** — Add loader in `src/data/datasets.py`, register in `load_dataset_for_task()`
2. **Metrics** — Add compute function in `src/evaluation/metrics.py`, register in `get_compute_metrics()`
3. **Config** — Create `configs/{task}/...yaml` with task-specific settings
4. **Labels** — Set `num_labels` in `forge.py` main function if classification

---

## References

- Hinton et al. (2015): "Distilling the Knowledge in a Neural Network" - Response-based
- Romero et al. (2015): "FitNets" - Feature-based distillation
- Zagoruyko & Komodakis (2017): "Paying More Attention to Attention" - Attention transfer
- Sanh et al. (2019): "DistilBERT" - Combined approach used in practice
