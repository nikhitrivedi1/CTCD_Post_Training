# Tasks Module

Centralized task registry - single source of truth for task properties.

**Source:** `src/tasks/registry.py` (~89 lines)

---

## Overview

The task registry defines all supported tasks and their properties. Other modules query this registry to determine task type, label count, and primary metric. Adding a new task starts here.

---

## TASKS Dictionary

```python
TASKS = {
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
| `classification` | Encoder-only | 2 or 3 | BERT, DistilBERT |
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
| `src/data/datasets.py` | `is_ocr_task()`, `is_translation_task()` | Route to correct loader |
| `src/models/loader.py` | `is_translation_task()`, `is_ocr_task()` | Route to correct model loader |
| `src/evaluation/metrics.py` | `is_classification_task()`, etc. | Route to correct metrics |
| `src/evaluation/evaluate.py` | `get_primary_metric()` | Transfer efficiency computation |
| `forge.py` | `get_num_labels()` | Set up classification head |

---

## Adding a New Task

1. Add entry to `TASKS` dict with all required fields
2. Add dataset loader in `src/data/datasets.py`
3. Add compute_metrics function in `src/evaluation/metrics.py` (if new type)
4. Update `load_dataset_for_task()` router in `src/data/datasets.py`
5. Create config files in `configs/{task}/`

---

*Source: `src/tasks/registry.py`*
