# Tasks Module

Centralized task registry - single source of truth for task properties.

**Source:** `src/tasks/registry.py`

---

## Overview

The task registry defines all supported tasks and their properties. Other modules query this registry to determine task type, label count, and primary metric. Adding a new task starts here.

---

## TASKS Dictionary

```python
TASK_REGISTRY = {
    "mnli": {
        "num_labels": 3,
        "type": "classification",
        "dataset": "glue",
        "subset": "mnli",
        "primary_metric": "accuracy",
    },
    "qqp": {
        "num_labels": 2,
        "type": "classification",
        "dataset": "glue",
        "subset": "qqp",
        "primary_metric": "f1",
    },
    "sst2": {
        "num_labels": 2,
        "type": "classification",
        "dataset": "glue",
        "subset": "sst2",
        "primary_metric": "accuracy",
    },
    "cola": {
        "num_labels": 2,
        "type": "classification",
        "dataset": "glue",
        "subset": "cola",
        "primary_metric": "mcc",
    },
    "mrpc": {
        "num_labels": 2,
        "type": "classification",
        "dataset": "glue",
        "subset": "mrpc",
        "primary_metric": "f1",
    },
    "qnli": {
        "num_labels": 2,
        "type": "classification",
        "dataset": "glue",
        "subset": "qnli",
        "primary_metric": "accuracy",
    },
    "rte": {
        "num_labels": 2,
        "type": "classification",
        "dataset": "glue",
        "subset": "rte",
        "primary_metric": "accuracy",
    },
    "wnli": {
        "num_labels": 2,
        "type": "classification",
        "dataset": "glue",
        "subset": "wnli",
        "primary_metric": "accuracy",
    },
    "clinc150": {
        "num_labels": 150,
        "type": "classification",
        "dataset": "clinc150",
        "subset": "default",
        "primary_metric": "accuracy",
    },
    "europarl": {
        "num_labels": None,
        "type": "translation",
        "dataset": "europarl",
        "subset": None,
        "primary_metric": "bleu",
    },
    "ocr": {
        "num_labels": None,
        "type": "ocr",
        "dataset": "olmocr-mix",
        "subset": None,
        "primary_metric": "edit_similarity",
    },
}
```

### Task Types

| Type | Architecture | num_labels | Models |
|------|-------------|------------|--------|
| `classification` | Encoder-only | 2-150 | BERT, DistilBERT |
| `translation` | Decoder-only | None | Llama, Mistral |
| `ocr` | Vision-Language | None | Qwen2.5-VL, Qwen3-VL |

### Primary Metrics

The primary metric determines which metric is used for:
- Best model selection during training
- Transfer efficiency computation
- Peak metrics tracking

| Task | Primary Metric | Why |
|------|---------------|-----|
| MNLI | accuracy | Standard for 3-class entailment |
| QQP | f1 | Handles class imbalance (70/30 split) |
| SST-2 | accuracy | Balanced binary classification |
| CoLA | mcc | Matthews Correlation Coefficient handles imbalanced binary with chance correction |
| MRPC | f1 | F1 is standard due to class imbalance in paraphrase detection |
| QNLI | accuracy | Standard GLUE metric for binary QA entailment |
| RTE | accuracy | Standard metric for low-resource binary entailment |
| WNLI | accuracy | Standard GLUE metric for binary coreference/entailment |
| CLINC150 | accuracy | Standard for balanced multi-class intent classification (150 classes, ~100 train examples each) |
| Europarl | bleu | Standard machine translation metric |
| OCR | edit_similarity | Character-level similarity (1 - CER) |

---

## Helper Functions

### Type queries

```python
get_num_labels(task: str) -> int | None
get_task_type(task: str) -> str | None
is_classification_task(task: str) -> bool
is_translation_task(task: str) -> bool
is_ocr_task(task: str) -> bool
get_primary_metric(task: str) -> str | None
```

All return `None` or `False` for unknown tasks (no exceptions).

---

## Usage Across the Codebase

The registry is imported and queried throughout the codebase:

| Module | What it checks | Why |
|--------|---------------|-----|
| `src/data/handlers/__init__.py` | `get_handler()` via `DATASET_REGISTRY` | Route to the correct dataset handler |
| `src/models/loader.py` | `is_translation_task()`, `is_ocr_task()` | Route to correct model loader |
| `src/evaluation/metrics.py` | `is_classification_task()`, etc. | Route to correct metrics |
| `src/evaluation/evaluate.py` | `get_primary_metric()` | Transfer efficiency computation |
| `forge.py` | `get_num_labels()` | Set up classification head |

---

## Adding a New Task

1. Add entry to `TASK_REGISTRY` dict with all required fields (`src/tasks/registry.py`)
2. Add a dataset handler implementing `DatasetHandler` (`prepare()`, `validate()`, `load()`) in `src/data/handlers/`
3. Register the handler under the task's dataset name in `DATASET_REGISTRY` (`src/data/handlers/__init__.py`)
4. Add a `compute_metrics` function in `src/evaluation/metrics.py` (if new type)
5. Create config files in `configs/{task_type}/{task}/`

---

*Source: `src/tasks/registry.py`*
