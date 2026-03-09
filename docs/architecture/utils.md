# Utils Module

Configuration loading, checkpoint management, environment detection, job tracking, and logging.

**Source:** `src/utils/` (4 files) + `utils/logging.py`

---

## config.py - YAML with Inheritance

**Source:** `src/utils/config.py` (~73 lines)

### load_config(config_path, _depth=0)

Loads a YAML config file with recursive inheritance via the `inherit` key.

```yaml
# configs/classification/mnli/distilbert/bert-base/01_kd.yaml
inherit: base.yaml
distillation:
  alpha: 0.5
  loss_type: response_kl
```

**Resolution process:**
1. Load current YAML file
2. If `inherit` key exists, resolve relative path and load parent (recursively)
3. Remove `inherit` key from current config
4. Deep merge: parent is base, current overrides

### _deep_merge(base, override)

Recursive dictionary merge. For nested dicts, merges recursively. For all other types, override wins.

```python
base    = {"training": {"lr": 1e-5, "epochs": 3}, "model": {"name": "bert"}}
override = {"training": {"lr": 2e-5}}
result  = {"training": {"lr": 2e-5, "epochs": 3}, "model": {"name": "bert"}}
```

### Circular Reference Protection

`MAX_INHERITANCE_DEPTH = 10`. If `_depth` exceeds this, raises `RecursionError` with the config path for debugging.

### Example Config Hierarchy

```
configs/classification/mnli/
  base.yaml                          # Task defaults (task: mnli, dataset: glue)
  distilbert/
    base.yaml                        # Student defaults (model: distilbert-base)
    bert-base/
      base.yaml                      # Teacher+student combo defaults
      00_sft.yaml                    # inherit: base.yaml -> CE-only
      ablations/
        01_kd_frozen-base.yaml       # inherit: ../base.yaml -> add distillation
        far/
          05_far.yaml                # inherit: ../../base.yaml -> FAR experiment
```

---

## checkpoint.py - Training State Management

**Source:** `src/utils/checkpoint.py` (~192 lines)

### TrainingState

Manages training state across SLURM job restarts. Reads/writes `training_state.json` in the output directory.

```python
state = TrainingState(output_dir)
state.set_targets(epochs=3)      # Set completion criteria
state.save(epoch=1.5, step=750)  # Record progress
state.is_complete()               # Check if target reached
state.is_pipeline_complete()      # Check if training + post-eval done
state.mark_pipeline_complete()    # Set after post-eval succeeds
```

#### State File Format

```json
{
    "completed_epochs": 2,
    "completed_steps": 1500,
    "target_epochs": 3,
    "target_steps": null,
    "pipeline_complete": false,
    "jobs": [
        {"job_id": "4661666", "timestamp": "2026-03-08T12:00:00", "epoch": 1.0, "step": 500},
        {"job_id": "4664300", "timestamp": "2026-03-08T16:00:00", "epoch": 2.0, "step": 1000}
    ]
}
```

#### get_resume_checkpoint(output_dir)

Finds the latest safely-completed checkpoint:

1. **Primary:** read `latest_checkpoint` file (written atomically by `CheckpointMarkerCallback`)
   - Try path as-is (may be absolute)
   - Try relative to output_dir
2. **Fallback:** glob `checkpoint-*` dirs and take the highest step number

The `latest_checkpoint` file is the safe approach - if the process dies mid-checkpoint-save, this file still points to the previous good checkpoint.

#### is_pipeline_complete() vs is_complete()

- `is_complete()`: training steps/epochs reached target (but post-eval might not have run)
- `is_pipeline_complete()`: training AND post-eval finished successfully

If the process is killed after training but during post-eval, `is_complete()` is True but `is_pipeline_complete()` is False. On resubmit, the job skips training and re-runs post-eval.

### mark_checkpoint_complete(checkpoint_dir)

Standalone function called by `CheckpointMarkerCallback.on_save()`.

Atomic write pattern:
1. Write checkpoint path to temp file (`.tmp.{pid}`)
2. `os.replace(tmp, latest_checkpoint)` - atomic on POSIX

---

## environment.py - Platform Detection

**Source:** `src/utils/environment.py` (~78 lines)

### Auto-loading .env

On import, automatically loads `.env` from project root with `override=False`. This means SLURM/Colab environment variables take precedence over `.env` file values.

### detect_environment()

Returns one of:
- `"slurm"` - `SLURM_JOB_ID` env var present
- `"colab"` - `COLAB_GPU` env var present
- `"local"` - neither

### get_device()

Returns the PyTorch device string:
- `"cuda"` - `torch.cuda.is_available()`
- `"mps"` - Apple Silicon (`torch.backends.mps.is_available()`)
- `"cpu"` - fallback

### get_gpu_info()

Returns GPU info for metadata. On SLURM, reads `SLURM_JOB_GPUS`. Otherwise returns `"unknown"`.

---

## jobs.py - Incomplete Jobs Registry

**Source:** `src/utils/jobs.py` (~46 lines)

File-based job tracking with file locking for concurrent SLURM access.

### locked_incomplete_jobs()

Context manager that provides atomic read-modify-write access to `.incomplete_jobs` JSON file.

```python
with locked_incomplete_jobs() as jobs:
    jobs[job_key] = {
        "type": "training",
        "resume_path": "outputs/.../run_name/timestamp",
    }
# File is atomically saved on context exit
```

**Locking mechanism:**
1. Touch `.incomplete_jobs.lock` (ensures it exists)
2. `fcntl.flock(lock_file, LOCK_EX)` - exclusive lock (blocks if another process holds it)
3. Load JSON from `.incomplete_jobs` (or empty dict if doesn't exist/corrupt)
4. Yield mutable dict
5. Atomic save: write to `.tmp.{pid}`, then `os.replace()`
6. Release lock in `finally`

**Why file locking?** SLURM job arrays can have multiple jobs starting/stopping concurrently. Without locking, two jobs writing `.incomplete_jobs` simultaneously can corrupt it.

### get_job_key()

Returns `SLURM_JOB_ID` if running under SLURM, otherwise `"local_{pid}"`.

---

## utils/logging.py - TeeWriter

**Source:** `utils/logging.py` (~90 lines)

### TeeWriter

Writes to both a file and the original stream (stdout/stderr). Handles carriage returns from tqdm progress bars cleanly.

**The tqdm problem:** Progress bars use `\r` (carriage return) to overwrite the current line in the terminal. If naively written to a file, you get hundreds of lines of partial progress bar updates.

**Solution:** Character-by-character processing:
- `\r` (carriage return): clear the line buffer (don't write anything)
- `\n` (newline): flush line buffer to file
- Other chars: append to buffer

**Result:**
- Terminal gets full interactive experience (cursor jumping, progress bar updates)
- File gets clean final states (only the last state of each progress bar line)

### setup_file_logging(log_path, update_transformers=False)

Replaces `sys.stdout` and `sys.stderr` with `TeeWriter` instances. Also updates existing logging handlers to write through the TeeWriter.

If `update_transformers=True`, also updates transformers library's logging handlers (catches transformers-internal log messages that bypass the standard logging hierarchy).

---

*Source: `src/utils/`, `utils/logging.py`*
