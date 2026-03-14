# Evaluation Module

Model evaluation orchestration and task-specific metrics.

**Source:** `src/evaluation/` (2 files: `evaluate.py`, `metrics.py`)

---

## evaluate.py - Evaluation Orchestration

### Why Not Use HF Trainer for Evaluation?

Creating a temporary HF Trainer spawns a new `Accelerator`. The `AcceleratorState` is a singleton, and the new instance can overwrite the main Trainer's mixed-precision settings (e.g., switching from fp16 to fp32). This corrupts subsequent training if `evaluate_model()` is called mid-run (e.g., for pre-training baselines).

**Solution:** Direct DataLoader + `torch.no_grad()` loop. No Trainer, no Accelerator, no singleton state corruption.

### evaluate_model()

```python
def evaluate_model(
    model, eval_dataset, tokenizer, compute_metrics,
    data_collator=None, batch_size=32, device=None,
    loss_only=False, remove_unused_columns=True
) -> Dict[str, float]
```

Core evaluation loop:

1. **Column filtering:** removes dataset columns that aren't in model's `forward()` signature (prevents "unexpected keyword argument" errors). Disabled for VLM tasks where the collator processes raw columns like `image`.
2. **DataLoader:** no shuffle, pinned memory on CUDA
3. **Inference loop:** `model.eval()`, `torch.no_grad()`, accumulate logits and labels
4. **Loss aggregation:** weighted by batch size for correct averaging
5. **Metrics:** calls `compute_metrics((logits, labels))` on concatenated results
6. **Restores training mode** after eval (`model.train(was_training)`)

### evaluate_models()

Convenience function that evaluates both model and teacher, logging with phase prefix ("pre" or "post").

Handles teacher label remapping by wrapping `compute_metrics` with `_remap_compute_metrics()` when `teacher_label_map` is set.

### run_evaluation()

Full evaluation phase: loads models from config or checkpoint, evaluates, logs transfer metrics.

Models are function-local and automatically garbage-collected on return. No manual `del model; torch.cuda.empty_cache()` needed.

### log_pre_post_metrics()

Logs metrics to console and W&B with proper prefixes (`model/pre`, `teacher/post`, etc.). Uses `commit=False` for W&B to batch metrics.

### log_transfer_metrics()

Computes and logs derived metrics from evaluation results:

| Metric | Formula | Description |
|--------|---------|-------------|
| Teacher advantage | `teacher[metric] - student[metric]` | Gap between teacher and student |
| Student gain | `current[metric] - previous[metric]` | Improvement from training |
| Teacher gain | same, for co-trained teacher | Whether teacher improved too |
| Transfer efficiency | `student_gain / teacher_advantage_pre` | What fraction of the gap was closed |

### _remap_compute_metrics()

Wraps a compute_metrics function to reorder logits before metric computation. Used when teacher was trained with different label ordering (e.g., TextAttack MNLI uses [contradiction, entailment, neutral] but HuggingFace uses [entailment, neutral, contradiction]).

```python
def wrapped(eval_pred):
    logits, labels = eval_pred
    remapped = logits[:, label_map]  # Reorder columns
    return compute_metrics((remapped, labels))
```

---

## metrics.py - Task-Specific Metrics

### compute_classification_metrics(eval_pred)

Computes all standard classification metrics regardless of task. The task registry specifies which is the primary metric for model selection.

| Metric | Library | Notes |
|--------|---------|-------|
| Accuracy | `evaluate.load("accuracy")` | Exact match rate |
| F1 (weighted) | `evaluate.load("f1")` | Weighted by class frequency |
| MCC | `evaluate.load("matthews_correlation")` | Primary for CoLA |

Takes argmax of logits to get predictions. Returns all three metrics for every classification task.

### compute_translation_metrics(tokenizer)

Returns a closure that computes BLEU for translation tasks.

1. Takes argmax over vocab dimension if logits are 3D
2. Replaces `-100` (padding) with `pad_token_id` before decoding
3. Decodes predictions and labels with `skip_special_tokens=True`
4. Computes BLEU via sacrebleu

COMET is imported but not computed during training (needs source text, which isn't available in `eval_pred`). Computed separately by `TranslationMetricsCallback`.

### compute_ocr_metrics(tokenizer)

Returns a closure that computes edit similarity and exact match for OCR tasks.

| Metric | Formula | Description |
|--------|---------|-------------|
| Edit similarity | `1 - levenshtein_distance / max(len(pred), len(ref))` | Character-level similarity |
| Exact match | `pred == ref` (after strip) | Binary perfect match rate |

Uses a custom `_edit_distance()` implementation (Wagner-Fischer algorithm) rather than an external dependency.

### get_compute_metrics(task, tokenizer)

Factory function that routes to the appropriate compute function:

```python
if is_classification_task(task):  # -> compute_classification_metrics
elif is_translation_task(task):    # -> compute_translation_metrics(tokenizer)
elif is_ocr_task(task):            # -> compute_ocr_metrics(tokenizer)
```

---

*Source: `src/evaluation/`*
