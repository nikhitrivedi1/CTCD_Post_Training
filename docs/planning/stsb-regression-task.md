# Adding STS-B as a Regression Task

## Goal

Add `sentence-transformers/stsb` (semantic textual similarity: sentence pairs scored with a
continuous value in [0,1], evaluated with Pearson/Spearman correlation) as a new task. Unlike
every existing task in `TASK_REGISTRY` (discrete N-class classification, translation, OCR),
STS-B is single-value regression — this is a new task `type`, not another classification entry.

Status: **investigated and planned, not yet implemented.**

## Why `type: "regression"`, not `type: "classification"` with `num_labels: 1`

Functionally these are almost the same thing — `AutoModelForSequenceClassification` already
auto-derives `problem_type="regression"` and switches to `MSELoss` whenever `num_labels == 1`,
regardless of what the task registry calls it. But two places in the codebase dispatch on the
registry's `type` string with a hard allowlist (see below), and classification's default KD loss
(`kl_divergence`, softmax-based) is meaningless for a 1-dim logit — so treating this as its own
type keeps the dispatch explicit and lets the config defaults (KD loss, metrics) diverge from
classification's without special-casing `num_labels == 1` everywhere.

## Dataset facts (verified against the live `sentence-transformers/stsb` HF page)

- Single config, `"default"`.
- Splits: `train` (5,750), `validation` (1,500), `test` (1,380).
- Columns: `sentence1` (string), `sentence2` (string), `score` (float64, range 0-1 — rescaled
  from the original STS-B 1-5 scale).
- No documented sentinel/missing-label rows (unlike CLINC150's `label=None` OOS rows, discovered
  only by loading the real dataset — worth re-verifying `score` has no nulls/NaNs before trusting
  the HF card, same lesson learned from CLINC150).

## What works unchanged (verified by reading the source, not just inferring)

- **`src/models/loader.py`** `_load_classifier()` (lines 40-66) passes `num_labels` straight to
  `AutoModelForSequenceClassification` and never sets `problem_type` — HF's own forward-pass
  auto-detection (`num_labels == 1` -> regression -> `MSELoss`) kicks in for free.
- **`src/data/collators.py`** has no int-casting logic for the `labels` column —
  `DataCollatorWithPadding`/`default_data_collator` infer tensor dtype from the Python value type
  (float -> `float32`), so a float `labels` column survives both the training and eval collation
  paths untouched.
- **`forge.py`**'s `TrainingArguments` branching (best-model-selection metric, `greater_is_better`,
  log verbosity, ~line 885-891) is keyed on `num_labels is None` (a proxy for "generative task"),
  not on task type — since STS-B has `num_labels=1` (not `None`), this already does the right
  thing.
- **`src/data/formatters.py`** `_format_classification_sft()` (lines 365-402) never inspects or
  casts the `labels` column dtype — it works unchanged for a sentence-pair + float-label dataset,
  it just needs to be reachable from a new `"regression"` branch in the dispatcher above it.
- **`get_primary_metric`/transfer-metric logic** (`forge.py`, `src/evaluation/evaluate.py`) has no
  metric-name allowlist — any string works, including `"pearson"`.

## What must change

1. **`src/data/formatters.py:226-233`** (`format_for_sft`) — hard `if/elif/else: raise ValueError`
   over `{"ocr", "translation", "classification"}`. Needs `elif task_type == "regression":` calling
   the existing `_format_classification_sft(...)` (it's dtype-agnostic, so no new formatter function
   needed — just a new branch pointing at the old one).
2. **`src/evaluation/metrics.py:240-251`** (`get_compute_metrics`) — falls through to `return None`
   for unknown task types, which would silently disable all eval metrics except loss for STS-B.
   Needs a new `compute_regression_metrics()` (Pearson + Spearman via `scipy.stats`, already a
   pinned dependency — no new package) wired in via `elif is_regression_task(task): return
   compute_regression_metrics`.
3. **`src/tasks/registry.py`** — new `"stsb"` entry (`num_labels: 1`, `type: "regression"`,
   `dataset: "stsb"`, `primary_metric: "pearson"`) plus an `is_regression_task()` helper mirroring
   `is_classification_task`, exported from `src/tasks/__init__.py`.
4. **`src/data/handlers/stsb.py`** (new) — modeled on `src/data/handlers/clinc150.py`: no-op
   `prepare()`, streaming `validate()`, `load()` using the dataset's own `train`/`validation`
   splits, renaming `score` -> `labels`.
5. **Distillation KD loss must be overridden per-task.** `src/training/losses.py:995` defaults
   `distillation.loss_type` to `"kl_divergence"` (softmax over the class dim) when unset —
   degenerate on a 1-dim logit (softmax of a single value is always 1.0, so KD signal is ~0).
   `"response_mse"` (`MSEDistillationLoss`, already registered in `LOSS_REGISTRY`, already
   z-normalizes + takes elementwise MSE) is the correct choice and must be set explicitly in the
   new `configs/regression/base.yaml`, not inherited from classification's default.
6. **New `configs/regression/` top-level config tree** (parallel to `classification/`/
   `translation/`/`ocr/`, matching the `configs/{task_type}/` convention documented in
   `configs/README.md`): `base.yaml` (sets `distillation.loss_type: response_mse`), then
   `stsb/base.yaml`, `stsb/distilbert/base.yaml`, `stsb/distilbert/00_sft.yaml`.
7. **`tests/unit/test_registry.py`**'s `TestTypePredicates.test_exactly_one_type_predicate_true`
   hardcodes the predicate list `[is_classification_task, is_translation_task, is_ocr_task]` —
   must add `is_regression_task` or `stsb` fails that test (0/4 predicates true).
8. **Docs** — `docs/architecture/tasks.md` (registry entry, new "regression" row in Task Types
   table, STS-B row in Primary Metrics table, `is_regression_task` in helper list) and
   `configs/README.md` (new `regression/` tree, updated counts).

## Known sharp edge, intentionally not guarded

`src/training/trainer.py:408-409` — the co-training (`freeze_teacher=False`) +
`teacher_label_map` path calls `F.cross_entropy` on raw logits/labels, which would break for a
float regression target. STS-B configs simply won't set `teacher_label_map` (it exists only to
reorder discrete class label spaces between differently-ordered teacher/student label sets —
meaningless for a scalar), so this path is never exercised. This is a pre-existing sharp edge
shared by every classification task, not something introduced by regression support, so it's not
worth adding a guard for.

## Verification plan (once implemented)

- `python -m pytest tests/unit/test_registry.py tests/unit/test_stsb_handler.py
  tests/unit/test_regression_metrics.py -v`
- Manual smoke check against the real Hub dataset before trusting the HF card (same pattern that
  caught CLINC150's undocumented OOS `label=None` rows):
  ```python
  from src.data.handlers import get_handler
  train, eval_ = get_handler("stsb").load()
  print(len(train), len(eval_), train.column_names, train[0], type(train[0]["labels"]))
  ```
- `python forge.py --config configs/regression/stsb/distilbert/00_sft.yaml --run_name
  stsb_sft_smoketest --max_steps 20 --debug` — confirm the model builds a 1-unit regression head,
  `outputs.loss` is finite (MSE, not NaN), and the formatter/metrics dispatch changes don't crash.
