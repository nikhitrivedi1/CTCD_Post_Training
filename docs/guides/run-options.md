# Run Options & CLI Reference

Complete reference for `forge.py` command-line options and run management.

---

## Command Line Arguments

| Flag | Description |
|------|-------------|
| `--config CONFIG` | Path to config file (required for new runs) |
| `--run_name NAME` | Run name. Creates if not exists, resumes if exists |
| `--timestamp` | Append timestamp to --run_name (e.g., `mnli_00` → `mnli_00_20260131-1200Z`) |
| `--force` | Overwrite existing run (deletes and starts fresh) |
| `--max_steps N` | Max training steps (for testing) |
| `--seed N` | Random seed (default: 42) |

---

## Quick Reference

```bash
# New run (auto-named with timestamp)
python forge.py --config configs/classification/mnli/.../00.yaml

# New run (custom name, no timestamp)
python forge.py --run_name mnli_00 --config configs/classification/mnli/.../00.yaml

# New run (custom name + timestamp = unique each time)
python forge.py --run_name mnli_00 --timestamp --config configs/classification/mnli/.../00.yaml

# Resume existing
python forge.py --run_name mnli_00

# Overwrite existing
python forge.py --run_name mnli_00 --config configs/... --force
```

---

## All Scenarios

### 1. Auto-generated name (no --run_name)

```bash
python forge.py --config configs/classification/mnli/bert_distilbert/ablations/00_no-teacher_ce-only.yaml
```

| Field | Value |
|-------|-------|
| **Output Dir** | `./outputs/mnli_ablations_00_no-teacher_ce-only_20260131-1200Z/` |
| **Run Name** | `mnli_ablations_00_no-teacher_ce-only_20260131-1200Z` |
| **W&B Run ID** | `mnli_ablations_00_no-teacher_ce-only_20260131-1200Z` |

**Run Name Format:** `{task}_{experiment}_{config}_{timestamp}`

---

### 2. Manual name (CREATE)

```bash
python forge.py --run_name mnli_00 --config configs/...
```

| Field | Value |
|-------|-------|
| **Output Dir** | `./outputs/mnli_00/` |
| **Run Name** | `mnli_00` |
| **W&B Run ID** | `mnli_00` |

---

### 3. Manual name (RESUME)

```bash
python forge.py --run_name mnli_00
```

| Field | Value |
|-------|-------|
| **Output Dir** | `./outputs/mnli_00/` (existing) |
| **Run Name** | `mnli_00` |
| **W&B Run ID** | Loaded from `run_metadata.json` |

**Console output:**
```
Run 'mnli_00' exists at ./outputs/mnli_00/
Resuming existing run. Use --force to overwrite.
```

---

### 4. Manual name + --force (OVERWRITE)

```bash
python forge.py --run_name mnli_00 --force --config configs/...
```

| Field | Value |
|-------|-------|
| **Output Dir** | `./outputs/mnli_00/` (deleted & recreated) |
| **Run Name** | `mnli_00` |
| **W&B Run ID** | `mnli_00` (fresh) |

---

### 5. Manual name + --timestamp (UNIQUE RUN)

```bash
python forge.py --run_name mnli_00 --timestamp --config configs/...
```

| Field | Value |
|-------|-------|
| **Output Dir** | `./outputs/mnli_00_20260131-1200Z/` |
| **Run Name** | `mnli_00_20260131-1200Z` |
| **W&B Run ID** | `mnli_00_20260131-1200Z` |

Each run is unique (timestamp ensures no collision).

---

### 6. Manual name + --timestamp + --force (UNIQUE + OVERWRITE)

```bash
python forge.py --run_name mnli_00 --timestamp --force --config configs/...
```

| Field | Value |
|-------|-------|
| **Output Dir** | `./outputs/mnli_00_20260131-1200Z/` |
| **Run Name** | `mnli_00_20260131-1200Z` |
| **Behavior** | Deletes if exists (rare: same minute), creates fresh |

Mostly redundant since `--timestamp` makes names unique, but valid for re-runs within the same minute.

---

### 7. Long name > 64 characters (TRUNCATION)

**Auto-generated (preserves timestamp):**
```
Input:  mnli_very_long_experiment_name_with_details_20260131-1200Z (70 chars)
Output: mnli_very_long_experiment_name_with_de_20260131-1200Z (64 chars)
```
Truncation: `{prefix[:49]}_{timestamp}` (timestamp is 14 chars: YYYYMMDD-HHMMZ)

**Manual without timestamp (simple truncation):**
```
Input:  this_is_a_very_long_name_exceeding_64_characters_significantly (65 chars)
Output: this_is_a_very_long_name_exceeding_64_characters_significantl (64 chars)
```

---

## Metadata & Environment

### run_metadata.json Structure

```json
{
  "run_name": "mnli_00",
  "wandb_run_id": "mnli_00",
  "config": "configs/classification/mnli/.../00_no-teacher_ce-only.yaml",
  "config_id": "mnli_ablations_00_no-teacher_ce-only",
  "created": "20260131-1200Z",
  "seed": 42
}
```

### W&B Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `WANDB_PROJECT` | `Forge-Default` | Project name |
| `WANDB_RUN_ID` | `{wandb_run_id}` | Run identifier (≤64 chars) |
| `WANDB_RESUME` | `allow` | Enable run resumption |

### Checkpoint Resume

- Auto-detected via `TrainingState.get_resume_checkpoint(output_dir)`
- Finds latest `checkpoint-*` directory in output_dir
- Trainer resumes from checkpoint automatically

---

## Error Cases

### E1: No run_name, no config
```bash
python forge.py
```
```
Error: No --run_name or --config provided.

Usage:
  New run:    python forge.py --config CONFIG
  Named run:  python forge.py --run_name NAME --config CONFIG
  Resume:     python forge.py --run_name EXISTING_RUN
```

### E2: New run_name, no config
```bash
python forge.py --run_name mnli_00
```
```
Error: Run 'mnli_00' does not exist and --config not provided.

Usage:
  New run:    python forge.py --run_name mnli_00 --config CONFIG
  Resume:     python forge.py --run_name EXISTING_RUN
  Overwrite:  python forge.py --run_name NAME --config CONFIG --force
```

### E3: --force without config
```bash
python forge.py --run_name mnli_00 --force   # when run exists
```
```
Error: --config is required for new runs.

Usage:
  python forge.py --run_name NAME --config CONFIG [--force]
```

### E4: Resume with --config provided
```bash
python forge.py --run_name mnli_00 --config configs/...   # when run exists
```
```
Error: --config cannot be used when resuming.
The stored config at ./outputs/mnli_00/config.yaml will be used.

Usage:
  Resume:     python forge.py --run_name mnli_00
  Overwrite:  python forge.py --run_name mnli_00 --config CONFIG --force
```

### E5: Corrupted run (missing config.yaml)
```bash
python forge.py --run_name mnli_00   # run dir exists but no config.yaml
```
```
Error: No stored config found at ./outputs/mnli_00/config.yaml
The run directory exists but appears corrupted.

Options:
  Delete and restart:  python forge.py --run_name mnli_00 --config CONFIG --force
  Or manually delete:  rm -rf ./outputs/mnli_00
```

---

## Warnings

| Command | Warning |
|---------|---------|
| `python forge.py --force --config ...` | --force without --run_name is redundant |
| `python forge.py --timestamp --config ...` | --timestamp without --run_name is redundant |

---

## Decision Matrix

| --run_name | --timestamp | Exists | --force | --config | Result |
|:----------:|:-----------:|:------:|:-------:|:--------:|--------|
| - | - | - | - | Y | **CREATE** (auto-name) |
| - | - | - | - | - | **ERROR E1** |
| Y | - | - | - | Y | **CREATE** |
| Y | - | - | - | - | **ERROR E2** |
| Y | - | Y | - | - | **RESUME** |
| Y | - | Y | - | Y | **ERROR E4** |
| Y | - | Y | Y | Y | **OVERWRITE** |
| Y | - | Y | Y | - | **ERROR E3** |
| Y | Y | - | - | Y | **CREATE** (unique) |
| Y | Y | Y | - | - | **RESUME** (rare) |
| Y | Y | Y | Y | Y | **OVERWRITE** (rare) |

---

## Output Directory Structure

```
outputs/
└── mnli_00/                      # Run directory
    ├── config.yaml               # Frozen config snapshot
    ├── run_metadata.json         # Run metadata (W&B ID, timestamps)
    ├── training_state.json       # Epoch/step tracking for resume
    ├── checkpoint-500/           # Intermediate checkpoints
    ├── checkpoint-1000/
    └── pytorch_model.bin         # Final model (after training)
```

---

## Invocation Modes & Job Tracking

The `.incomplete_jobs` file tracks running/incomplete jobs for SLURM auto-resubmission.

### Job Key Logic

```python
job_key = os.environ.get("SLURM_JOB_ID") or f"local_{os.getpid()}"
```

| Environment | Job Key | Example |
|-------------|---------|---------|
| SLURM (sbatch/srun) | `SLURM_JOB_ID` | `12345` |
| Local (direct) | `local_{pid}` | `local_98765` |

---

### Mode 1: Direct Local Run

```bash
python forge.py --config configs/.../config.yaml
python forge.py --run_name my_run --config configs/.../config.yaml
python forge.py --run_name my_run --timestamp --config configs/.../config.yaml
```

| Event | Action |
|-------|--------|
| Start | `job_key = local_{pid}`, adds entry |
| Complete | Removes entry |
| Ctrl+C/crash | Entry stays (stale) |

---

### Mode 2: Direct Local Resume

```bash
python forge.py --run_name existing_run_20260201-1200Z
```

| Event | Action |
|-------|--------|
| Start | New `local_{pid}`, adds entry |
| Original entry | Stays stale (no `PREV_JOB_ID`) |

---

### Mode 3: srun (Interactive SLURM)

```bash
srun --gres=gpu:1 python forge.py --run_name my_run --config ...
```

| Event | Action |
|-------|--------|
| Start | `job_key = SLURM_JOB_ID`, adds entry |
| Complete | Removes entry |
| Timeout/cancel | Entry stays, no auto-resubmit |

---

### Mode 4: sbatch (Batch SLURM)

```bash
sbatch scripts/run_training.sh my_run configs/.../config.yaml
TIMESTAMP=1 sbatch scripts/run_training.sh my_run configs/.../config.yaml
```

| Event | Action |
|-------|--------|
| Start | `job_key = 12345`, adds entry with actual run_name |
| Complete | Removes entry |
| Timeout | Entry stays, triggers auto-resubmit |

---

### Mode 5: sbatch Auto-Resubmit

When job 12345 times out:

```bash
# run_training.sh automatically does:
PREV_JOB_ID=12345 sbatch scripts/run_training.sh my_run_20260201-1200Z
```

| Event | Action |
|-------|--------|
| New job 12346 starts | Reads `PREV_JOB_ID=12345` |
| | Removes entry `12345` |
| | Adds entry `12346` |
| Complete | Removes `12346` |

---

### Mode 6: Concurrent SLURM Jobs

```bash
sbatch scripts/run_training.sh far-03 configs/.../03.yaml  # Job 12345
sbatch scripts/run_training.sh far-07 configs/.../07.yaml  # Job 12346
```

Each job gets unique key, no collision:

```json
{
  "12345": {"run_name": "far-03_2026...", "output_dir": "..."},
  "12346": {"run_name": "far-07_2026...", "output_dir": "..."}
}
```

---

### Edge Cases

| Case | Behavior |
|------|----------|
| Crash before cleanup | Entry stays stale, manual cleanup |
| PID reuse (local) | New job overwrites old entry (acceptable) |
| Missing `.incomplete_jobs` | Returns `{}`, self-healing |
| Corrupted JSON | `JSONDecodeError` caught, returns `{}` |

---

### Future Enhancements

- [ ] Cleanup script for stale entries
- [ ] `--prev_job_id` flag for manual handoff
- [ ] Notification on job completion/failure
