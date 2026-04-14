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

- **`SLURMSignalCallback`** (Python) and the **batch-shell** trap in `scripts/slurm/slurm_essentials.sh` cooperate so walltime limits can trigger a checkpointed stop. Full flow is documented in **§9**.

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

## 9. SIGUSR1 and SLURM walltime: full logic audit

This section traces **every layer** that participates in graceful timeout handling for typical Forge GPU training (`run_training.sh`, CTCD sweep workers, and the Python trainer). It is separate from Hugging Face early stopping: walltime uses **OS signals** and **shell traps**, not validation metrics.

### 9.1 Why two “flags” exist

| Layer | Name / mechanism | Set when | Purpose |
|-------|-------------------|----------|---------|
| **Shell** | `SIGUSR1_RECEIVED=1` | SLURM’s `B:SIGUSR1` trap runs on the **batch script process** | Record that walltime warning fired; drive **resubmit** / sweep **break** logic |
| **Python** | `SLURMSignalCallback.caught_signal` | `SIGUSR1` or `SIGTERM` delivered to the **Python** process | Defer stop + save until **`on_step_end`**; tell `forge.py` to skip final merge / post-eval |

These are **not** the same variable and are **not** guaranteed to flip in the same instant: the shell sets `SIGUSR1_RECEIVED` when it receives SLURM’s signal; Python sets `caught_signal` only after `kill -SIGUSR1` (or a direct `SIGTERM`) reaches `forge.py`.

### 9.2 SLURM → batch shell only (`B:SIGUSR1`)

Training-related sbatch scripts declare **`#SBATCH --signal=B:SIGUSR1@120`** (e.g. `scripts/slurm/run_training.sh`, `run_ctcd_part1_sweep.sh`, `run_bench.sh`). With the **`B:`** prefix, SLURM sends **`SIGUSR1` to the batch shell only**, not to the job’s child processes. Comments in `slurm_essentials.sh` state this explicitly: the shell **must forward** the signal to Python.

Rough timeline (example `@120`):

- **T − 120s** (walltime): SLURM sends **`SIGUSR1`** to the batch shell.
- **T − 0s**: SLURM **`SIGKILL`** on the allocation if the job is still alive (hard cap).

### 9.3 Shell trap (`scripts/slurm/slurm_essentials.sh`)

Sourcing `slurm_essentials.sh` registers a **`SIGUSR1` trap** that:

1. Sets **`SIGUSR1_RECEIVED=1`** (numeric `1` in the sweep scripts’ `[[ ... ]]` tests).
2. If **`_FORGE_CHILD_PID`** is set: runs **`kill -SIGUSR1 $_FORGE_CHILD_PID`** so Python’s handler / `SLURMSignalCallback` can run.
3. Starts a **background kill timer** (subshell):
   - Sleeps **`_FORGE_KILL_DEADLINE`** seconds (default **`90`** if unset).
   - If the child PID is **still alive**, prints a message and runs **`kill -9`** on that PID (uncatchable **`SIGKILL`**).

Commented timeline in the script: with `@120` and default 90s deadline, the timer is intended to fire around **T − 30s** if Python has not exited—leaving margin before SLURM’s hard kill.

**`slurm_wait_child PID`:**

- Assigns **`_FORGE_CHILD_PID`** and **`wait`s** on the background Python PID.
- If **`wait` returns 138** (`128+10`, i.e. interrupted by **`SIGUSR1`**) **and** `SIGUSR1_RECEIVED=1`, it **loops** and waits again until the child **actually exits** (so a spurious `wait` interrupt does not drop the child).
- On exit, clears **`_FORGE_CHILD_PID`** and **cancels** the kill-timer subshell if still running.
- **Special case:** if exit code is **137** (often `128+9` from SIGKILL) **and** `SIGUSR1_RECEIVED=1`, the script **remaps exit code to 0** (“treat as clean exit for resubmit”) so downstream resubmit logic can still run.

There is **no** trap on **`SIGTERM`** in this file for the generic training path; Python still registers **`SIGTERM`** in `SLURMSignalCallback` (§9.5).

### 9.4 How Python is launched (`run_training.sh` and sweeps)

**`run_training.sh`:**

- Defines **`run_forge()`**: starts **`python forge.py … &`**, then **`slurm_wait_child $!`** so the shell always has a single child PID to forward to.
- **Auto-resubmit** (tail of script): if exit code is **0**, **`slurm_read_incomplete`** finds a **`training`** job, and **`slurm_is_complete`** on the output dir returns **`false`**, the script **`sbatch`s itself again** with **`_FORGE_RESUME_PATH`** set. Env vars like **`SEED`**, **`MAX_STEPS`**, **`EPOCHS`** are forwarded.
- If **`_FORGE_SWEEP_WORKER=1`** (set by CTCD sweep scripts when they invoke `run_training.sh` in the background), **auto-resubmit is skipped**—the **parent sweep** owns resubmission and reads **`SIGUSR1_RECEIVED`** itself.

**`run_ctcd_part1_sweep.sh` (representative sweep pattern):**

- Sources the same **`slurm_essentials.sh`** trap.
- Before each grid TID, if **`SIGUSR1_RECEIVED -eq 1`**, breaks out of the worker loop **before** starting a new training run.
- After each `slurm_wait_child` on a nested `run_training.sh`, if **`SIGUSR1_RECEIVED -eq 1`**, breaks (assumes in-flight run already checkpointed via Python).
- After the loop, if **`SIGUSR1_RECEIVED -eq 1`** and work remains, **`sbatch`s the sweep script** again.

Part 2a / 2b sweep scripts mirror the same **`SIGUSR1_RECEIVED`** break-and-resubmit pattern.

### 9.5 Python: when the callback is registered (`forge.py`)

- **`detect_environment()`** returns **`"slurm"`** iff **`SLURM_JOB_ID`** is set (`src/utils/environment.py`).
- **`SLURMSignalCallback`** is attached **only** for `env == "slurm"` in both **`_train`** (supervised SFT/KD) and **`_train_rl`** (GRPO). Local / Colab runs do **not** get this callback.

### 9.6 Python: `SLURMSignalCallback` (`src/training/callbacks.py`)

- **`__init__`**: registers **`signal.signal(signal.SIGUSR1, self._handler)`** and **`signal.signal(signal.SIGTERM, self._handler)`**.
- **`_handler`**: sets **`self.caught_signal = True`** and logs a warning. It does **not** write checkpoints (avoid unsafe work inside the async signal context).
- **`on_step_end`**: if **`caught_signal`**, sets **`control.should_save = True`** and **`control.should_training_stop = True`**.

So the **authoritative** “save and stop” hook is the **next** `on_step_end` after the signal is delivered to Python—**not** the signal handler itself.

### 9.7 Interaction with the Hugging Face training loop (`transformers` Trainer)

In **`Trainer`**’s inner training loop (v5.2.0 in this repo’s `.venv`), after a successful optimizer step (when not skipped), the order is:

1. **`self.state.global_step += 1`**
2. **`callback_handler.on_step_end(...)`** — here **`SLURMSignalCallback`** may set **`should_save`** and **`should_training_stop`** if the signal arrived **before or during** this step and **`caught_signal`** is already true.
3. **`_maybe_log_save_evaluate(...)`** — may still run **logging**, **evaluation** (if `control.should_evaluate`), and **checkpoint save** (if `control.should_save` or periodic save), depending on HF control flags.

Then the loop checks **`should_training_stop`** and **breaks** out of the epoch / training loops.

**Implications:**

- The checkpoint directory name aligns with **`global_step` after increment** for the step whose **`on_step_end`** observed the flag (i.e. **`checkpoint-{global_step}`** for the step that **just finished** when the callback fired at end of that step).
- If **`should_evaluate`** is true for this boundary, **evaluation may still run** in **`_maybe_log_save_evaluate`** **after** `on_step_end` set **`should_training_stop`**—so shutdown is **not always immediate**; there can be **one scheduled eval** before the loop exits.
- If **`SIGUSR1`** reaches Python **after** `on_step_end` but **during** the evaluation work inside **`_maybe_log_save_evaluate`**, **`SLURMSignalCallback` does not run `on_step_end` again** until the **next** optimizer step completes. In the worst case, training can run **one additional full step** (and finish the in-flight eval) before **`on_step_end`** sets stop and save. This is a **real edge case** for long evals or slow I/O.

### 9.8 After `trainer.train` returns (`forge.py`)

- **`signal_interrupted = (slurm_cb is not None and slurm_cb.caught_signal)`**.
- If **`signal_interrupted`**: skips **`_save_final_model`**, optional co-trained teacher finalization, and **`_remove_hf_checkpoint_dirs_after_success`** (SFT path); GRPO path skips final save and cleanup similarly. Log: *“Training interrupted - skipping final model save (checkpoint already saved)”*.
- **`main()`**: if **`signal_interrupted`**, **post-training evaluation** is skipped; pipeline completion / incomplete-job registry behavior follows the existing interrupted path.

So after a walltime stop, the intended artifact is the **HF checkpoint** written via **`should_save`**, not the normal “final merged” save.

### 9.9 Other codepaths (same signal name, different scripts)

- **`src/evaluation/bench_utils.py`**: registers its own **`SIGUSR1` / `SIGTERM`** handlers for benchmark-style runs (not `SLURMSignalCallback`).
- **`scripts/slurm/run_chain.sh`**: uses a separate **`_CHAIN_SIGUSR1`** trap for orchestrator resubmit logic, not `SIGUSR1_RECEIVED` from `slurm_essentials.sh` (different entrypoint).

### 9.10 Operational checklist

- Ensure jobs **source `slurm_essentials.sh`** and use **`run_forge` / `slurm_wait_child`** (or equivalent **`_FORGE_CHILD_PID` + `wait`**) so **`SIGUSR1`** reaches Python; launching `python forge.py` in the **foreground** without the child PID set means the trap **cannot forward** the signal.
- Tune **`_FORGE_KILL_DEADLINE`** if eval+save routinely needs **more than ~90s** after the first `SIGUSR1`, or risk **`SIGKILL`** and a truncated checkpoint.
- Distinguish **`SIGUSR1_RECEIVED`** (shell resubmit / sweep break) from **`caught_signal`** (Python “skip final save”).

## 10. Code references

- Task metrics: `src/tasks/registry.py`
- `TrainingArguments` + callback registration: `forge.py` (training setup and `evaluation.early_stopping_patience` handling)
- Classification metric keys: `src/evaluation/metrics.py` (`compute_classification_metrics`)
- Eval cadence: `configs/CTCD_Experiments/_shared.yaml`, per-task `configs/CTCD_Experiments/<task>/base.yaml`
- HF callback semantics: `transformers.EarlyStoppingCallback` (v5.2.0); `Trainer.evaluate` → `on_evaluate` before return; `_maybe_log_save_evaluate` → `_determine_best_metric` after `_evaluate`
- SLURM shell: `scripts/slurm/slurm_essentials.sh` (`SIGUSR1` trap, `slurm_wait_child`, kill timer, `SIGUSR1_RECEIVED`)
- SLURM training wrapper / resubmit: `scripts/slurm/run_training.sh` (`run_forge`, `_FORGE_SWEEP_WORKER`, `slurm_read_incomplete` / `slurm_is_complete`)
- Sweep worker example: `scripts/slurm/run_ctcd_part1_sweep.sh`
- Python signal callback: `src/training/callbacks.py` (`SLURMSignalCallback`)
- Environment gate: `src/utils/environment.py` (`detect_environment`)
