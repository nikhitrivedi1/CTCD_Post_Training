# Training: Callbacks

HuggingFace TrainerCallback extensions for SLURM signals, checkpointing, evaluation, and metrics.

**Source:** `src/training/callbacks.py` (~525 lines)

---

## Overview

Forge uses six callbacks that hook into HF Trainer's lifecycle at specific points (`on_log`, `on_step_end`, `on_evaluate`, `on_save`, `on_epoch_end`, `on_train_end`). They are added to the trainer in `forge.py` and execute in registration order.

**Registration order matters:** `CheckpointMarkerCallback` should be added LAST so it runs after `TeacherEvalCallback` saves the teacher model. This ensures the `latest_checkpoint` marker is only written after all checkpoint files (model + teacher) are safely on disk.

---

## StepMetricsCallback

Logs training metrics to Python logger in a grep-friendly format for `output.log`.

### How It Works

**`on_log()`** is called each time HF Trainer logs metrics. It extracts core training metrics and formats them as a single structured log line:

```
step=100  epoch=0.64  loss=0.141914  learning_rate=0.000042  grad_norm=0.003396
```

**Core keys** (always logged if present): `loss`, `learning_rate`, `grad_norm`

Also logs reward metrics (prefixed with `rewards/`) during GRPO training, and any `eval_*` metrics during evaluation steps.

### Why This Exists

HF Trainer logs metrics via W&B and the standard logging system, but the format isn't grep-friendly. This callback writes a single structured line per step to `output.log`, making it easy to extract training curves with `grep "step="` or monitor progress with `tail -f`.

---

## SLURMSignalCallback

Handles graceful job termination when SLURM sends a pre-timeout signal.

### How It Works

1. **Constructor** registers signal handlers for `SIGUSR1` and `SIGTERM`
2. **Signal handler** (`_handler`) sets `self.caught_signal = True` - does NO work in the signal context (risky)
3. **`on_step_end()`** checks the flag after each training step
4. If caught: sets `control.should_save = True` and `control.should_training_stop = True`
5. HF Trainer's normal checkpoint logic saves the model
6. The SLURM shell script detects the incomplete run and auto-resubmits

### Why Not Save in the Signal Handler?

Signal handlers run in an unpredictable state. Saving a checkpoint (writing files, syncing CUDA) in a signal handler can corrupt data. Instead, the handler just sets a boolean flag, and the main training loop checks it at a safe point (between steps).

### SLURM Integration

The Explorer HPC cluster has 8-hour job limits. The shell script sends `SIGUSR1` 120 seconds before the time limit. This gives the callback enough time to:
1. Finish the current step
2. Save a checkpoint
3. Write to `.incomplete_jobs`
4. Exit cleanly

The shell script then reads `.incomplete_jobs` and resubmits with `--resume`.

---

## CheckpointMarkerCallback

Marks checkpoints as safely completed after all files are written.

### How It Works

**`on_save()`** is called after HF Trainer finishes writing a checkpoint. It calls `mark_checkpoint_complete(checkpoint_dir)` which:
1. Writes the checkpoint path to a temp file
2. Atomically renames (`os.replace`) to `latest_checkpoint` in the output dir

### Why This Exists

HF Trainer writes many files during checkpoint save (model weights, optimizer state, scheduler, etc.). If the process is killed mid-save, the checkpoint directory exists but is incomplete/corrupt.

The `latest_checkpoint` file is only written after ALL files are saved. On resume, `TrainingState.get_resume_checkpoint()` reads this file to find the last safe checkpoint. If the file points to checkpoint-500 but checkpoint-600 exists but is incomplete, we correctly resume from 500.

---

## TeacherEvalCallback

Evaluates the co-trained teacher at each eval step and saves it alongside the model.

### `on_evaluate(args, state, control, metrics)`

1. Runs `evaluate_model()` on the teacher using the same eval dataset and metrics function
2. Handles teacher label remapping if `teacher_label_map` is set (wraps `compute_metrics` with `_remap_compute_metrics`)
3. Injects results into the metrics dict with `eval_teacher_` prefix (e.g., `eval_teacher_accuracy`)
4. Calls `self.trainer.log()` to push teacher metrics through the full logging pipeline (W&B, console)

**Why re-invoke log():** HF Trainer calls `self.log()` BEFORE `on_evaluate`. So teacher metrics injected into the metrics dict miss the normal logging path. The explicit re-log pushes them through.

### `on_save(args, state, control)`

If co-training (`freeze_teacher=False`), saves the teacher model to `checkpoint-N/teacher/` alongside the model checkpoint. This is needed for resume - the co-trained teacher's weights change during training, so we can't just reload from the original model ID.

---

## TranslationMetricsCallback

Computes BLEU and COMET scores at the end of each epoch for translation tasks.

### `on_epoch_end(args, state, control, model)`

1. Selects a subset of the eval dataset (`num_samples`, default 200)
2. Reloads the raw Europarl dataset to get original text (the tokenized eval dataset doesn't have it)
3. Generates translations using `InferenceEngine.from_model()` with the training model
4. Computes BLEU via sacrebleu and COMET via the comet library
5. Logs to W&B as `eval/bleu` and `eval/comet`

### `_generate_translations(model, sources)`

Formats source texts with chat templates and generates translations in batches using the `InferenceEngine`. Handles models with and without chat templates.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_samples` | 200 | Number of eval examples to translate |
| `max_new_tokens` | 128 | Max tokens per translation |
| `batch_size` | 8 | Generation batch size |
| `source_lang` | "en" | Source language code |
| `target_lang` | "fr" | Target language code |

---

## PeakMetricsCallback

Tracks the best value achieved for each metric along with step/epoch info.

### `on_evaluate(args, state, control, metrics)`

For each `eval_*` metric (excluding loss, runtime stats):
1. Check if value exceeds current peak
2. If new peak: update `self.peaks[metric_name]` with value, step, epoch
3. Log immediately to W&B summary as `peak/{metric_name}`, `peak/{metric_name}_step`, `peak/{metric_name}_epoch`

Skipped metrics: `runtime`, `samples_per_second`, `steps_per_second`, `model_preparation_time` (these are throughput stats, not quality metrics).

Also handles teacher-prefixed metrics (e.g., `eval_teacher_accuracy`) by stripping the prefix for skip-list lookup but keeping the full name as the peak key.

### `on_train_end(args, state, control)`

Logs a summary table of all peak metrics to the console:
```
==================================================
PEAK METRICS SUMMARY
==================================================
  accuracy: 0.8262 (step 1500, epoch 3.00)
  f1: 0.8245 (step 1500, epoch 3.00)
  mcc: 0.7412 (step 1200, epoch 2.40)
==================================================
```

### `get_peaks() -> dict`

Returns the current peaks dict for programmatic access.

---

## Callback Registration in forge.py

### SFT/KD path (`_train()`):

```python
# Step metrics (always first - logs to output.log)
trainer.add_callback(StepMetricsCallback())

# SLURM signal handler (only on cluster)
if env == "slurm":
    trainer.add_callback(SLURMSignalCallback(trainer))

# Teacher eval (only for co-training)
if isinstance(trainer, DistillationTrainer) and not trainer.freeze_teacher:
    trainer.add_callback(TeacherEvalCallback(trainer, teacher_label_map=trainer.teacher_label_map))

# Peak metrics
trainer.add_callback(PeakMetricsCallback())

# Translation metrics (when configured)
if num_labels is None and config.get("evaluation", {}).get("compute_translation_metrics", False):
    trainer.add_callback(TranslationMetricsCallback(eval_dataset, tokenizer, ...))

# Checkpoint marker (MUST be last - after all on_save hooks)
trainer.add_callback(CheckpointMarkerCallback())
```

### RL path (`_train_rl()`):

```python
trainer.add_callback(StepMetricsCallback())

if env == "slurm":
    trainer.add_callback(SLURMSignalCallback(trainer))

trainer.add_callback(CheckpointMarkerCallback())
```

---

*Source: `src/training/callbacks.py`*
