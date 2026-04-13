# GLUE stopping criteria audit

Audit date: 2026-04-13. Scope: supervised GLUE classification in Forge (`forge.py` SFT/KD path), Hugging Face `transformers` **5.2.0** (see repo `.venv`), and CTCD experiment defaults under `configs/CTCD_Experiments/`.

## 1. Executive summary

Training can end before `num_epochs` for several independent reasons. The log line *“Training complete (early stopping at …)”* in `forge.py` is **not** specific to Hugging Face `EarlyStoppingCallback`; it fires whenever `TrainingState.is_complete()` is false after a normal (non-SLURM-interrupted) finish—so it includes HF early stopping, `--max_steps`, or any other trainer exit before the configured epoch target.

For **CoLA**, the metric that drives best-checkpoint selection and early stopping is **`eval_mcc`** (student), not `eval_loss` and not any teacher-prefixed series. Early stopping counts **consecutive evaluations** that fail to **strictly beat the best value seen so far** on that metric; a **rising** validation curve that stays **below the historical best** still consumes patience.

## 2. GLUE primary metrics (registry → `TrainingArguments`)

Forge sets `metric_for_best_model` and `greater_is_better` from `src/tasks/registry.py`; YAML `evaluation.metric_for_best_model` is **not** read—only the task name matters.

| Task | `primary_metric` | `metric_for_best_model` (classification) | `greater_is_better` |
|------|------------------|------------------------------------------|---------------------|
| MNLI | `accuracy` | `eval_accuracy` | `true` |
| QQP | `f1` | `eval_f1` | `true` |
| SST-2 | `accuracy` | `eval_accuracy` | `true` |
| **CoLA** | **`mcc`** | **`eval_mcc`** | **`true`** |
| MRPC | `f1` | `eval_f1` | `true` |
| QNLI | `accuracy` | `eval_accuracy` | `true` |
| RTE | `accuracy` | `eval_accuracy` | `true` |
| WNLI | `accuracy` | `eval_accuracy` | `true` |

`compute_classification_metrics()` returns keys `accuracy`, `f1`, `mcc`, `ace`; the trainer logs them with the `eval_` prefix (e.g. **`eval_mcc`**). The Matthews metric from `evaluate.load("matthews_correlation")` is stored under the short name **`mcc`**, not `matthews_correlation`.

Code: `forge.py` (`TrainingArguments`), `src/tasks/registry.py`, `src/evaluation/metrics.py`.

## 3. Mechanisms that stop or cap training

### 3.1 Epoch budget

- **`training.num_epochs`** → `TrainingArguments.num_train_epochs` (e.g. CTCD `_shared.yaml`: **10**).

### 3.2 Step budget (CLI)

- **`--max_steps`** → `TrainingArguments.max_steps`. When `> 0`, training stops at that global step regardless of epochs. `TrainingState` can record `target_steps` and treat the run as incomplete vs epoch targets (see §5).

### 3.3 Hugging Face early stopping

- **Enabled when:** `evaluation.early_stopping_patience` is set, positive, and `should_load_best_model(config)` is true (eval actually runs).
- **`should_load_best_model`:** `true` iff `evaluation.eval_strategy` is **`steps`** or **`epoch`** (`forge.py`). Strategy **`no`** disables both `load_best_model_at_end` and the callback (with a warning if patience is set).
- **Implementation:** `transformers.EarlyStoppingCallback(early_stopping_patience=…)` only; **`early_stopping_threshold` is left at the library default** (**`0.0`** in v5.2.0), so an eval must improve the monitored metric by more than zero vs the running best to reset patience (ties and identical floats do not reset).
- **Monitored metric:** `TrainingArguments.metric_for_best_model` (for GLUE classification: table above). If that key were missing from the metrics dict, the callback logs a warning and **does not** stop training on that basis.
- **Interaction with `TrainerState.best_metric`:** In `Trainer._maybe_log_save_evaluate`, evaluation runs, then **`on_evaluate` callbacks run**, then **`_determine_best_metric` updates `state.best_metric`**. So `EarlyStoppingCallback` compares the **current** eval value to the **previous** best—this matches the intended “no strict improvement over best-so-far” semantics.

### 3.4 SLURM / signals

- **`SLURMSignalCallback`** sets `control.should_training_stop` (and save) on **SIGUSR1** or **SIGTERM** after the current step (used for time limits and sweep scripts). This is not metric-based.

### 3.5 No other Forge callbacks stop GLUE training

- **`PeakMetricsCallback`** only tracks peaks; it does not set `should_training_stop`.
- **`TeacherEvalCallback`** (co-training) adds **`eval_teacher_*`** metrics; it does not change **`eval_mcc`**. Charts that show improving **teacher** MCC do not reset student early-stopping patience.

## 4. CTCD / GLUE checkpoint and eval cadence

Shared defaults: `configs/CTCD_Experiments/_shared.yaml`

| Setting | Value | Notes |
|---------|-------|--------|
| `evaluation.eval_strategy` | `steps` | Required for eval + best model + early stopping |
| `evaluation.early_stopping_patience` | **5** | Five **evaluations** without strict improvement on the primary metric |
| `checkpointing.save_steps` / `evaluation.eval_steps` | **2000** unless task `base.yaml` overrides | Must stay **equal** for reliable `load_best_model_at_end` checkpoint alignment |
| `checkpointing.save_total_limit` | **1** | Disk rotation only; does not change stop rules |
| `training.num_epochs` | **10** | Upper bound if nothing else stops first |

Per-task overrides (same `save_steps` / `eval_steps` in each `base.yaml`):

| Task | Steps per eval/save |
|------|---------------------|
| MNLI, QQP, QNLI | 2000 |
| CoLA | **500** |
| MRPC | 200 |
| RTE | 100 |

**CoLA patience in step space:** With patience **5** and eval every **500** steps, stopping can trigger after **2500 global steps** without a new best **`eval_mcc`**, even if later evals would have beaten an older peak had training continued.

## 5. Misleading “early stopping” log in `forge.py`

After training, if `TrainingState.is_complete()` is false, Forge logs *“Training complete (early stopping at step/epoch …)”*. `is_complete()` compares **`int(completed_epochs)`** to `target_epochs` or steps to `target_steps`—it does **not** distinguish HF early stopping from `--max_steps` or other early exits. Treat this line as **“finished before nominal epoch/step target”**, not as proof that `EarlyStoppingCallback` fired.

## 6. Doc / config drift (FYI)

`configs/CTCD_Experiments/REQUIRED_CONFIGS.md` lists YAML fields such as `evaluation.metric_for_best_model` and `training.load_best_model_at_end` and suggests **`eval_matthews_correlation`** for CoLA. In the current codepath, **`forge.py` does not read those YAML keys**; CoLA’s monitored metric is **`eval_mcc`** from the task registry. Patience is correctly keyed off that in code.

Implemented CTCD configs use **`evaluation.early_stopping_patience`** (see `_shared.yaml`); `forge.py` reads patience only from `config["evaluation"]`, not from `training.*`.

## 7. Interpreting “MCC is improving but run stopped early”

Plausible explanations consistent with this codebase:

1. **Improvement vs last eval, but not vs best-ever:** Patience resets only on a **new strict maximum** of **`eval_mcc`**. A rebound that stays below an earlier spike still increments the counter each eval.
2. **Wrong curve in W&B:** For distillation/co-training, **`eval_teacher_mcc`** can rise while **`eval_mcc`** stalls; only the student metric controls stopping.
3. **Not HF early stopping:** SLURM timeout (`SIGUSR1`), **`--max_steps`**, or user interrupt.
4. **Training ended at nominal epochs:** If neither early stopping nor step cap applies, ten epochs completes “normally” even if MCC still looks noisy (CoLA validation is small and high-variance).

## 8. Quick verification checklist

- Confirm logs show **`Early stopping enabled (patience=…)`** when expected (`forge.py`).
- On each eval, compare **`eval_mcc`** (CoLA) to the **running best** in trainer state / W&B, not only step-to-step deltas.
- Ensure **`save_steps == eval_steps`** for the config you launched (see `docs/guides/glue-checkpoint-policy.md`).
- For cluster jobs, check whether **`SIGUSR1`** or walltime preemption occurred.

## 9. Code references

- Task metrics: `src/tasks/registry.py`
- `TrainingArguments` + callback registration: `forge.py` (training setup and `evaluation.early_stopping_patience` handling)
- Classification metric keys: `src/evaluation/metrics.py` (`compute_classification_metrics`)
- Eval cadence: `configs/CTCD_Experiments/_shared.yaml`, per-task `configs/CTCD_Experiments/<task>/base.yaml`
- HF callback semantics: `transformers.EarlyStoppingCallback` (v5.2.0); `Trainer.evaluate` → `on_evaluate` before return; `_maybe_log_save_evaluate` → `_determine_best_metric` after `_evaluate`
