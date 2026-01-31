# Run Options & CLI Reference

Complete reference for `train.py` command-line options and run management.

---

## Command Line Arguments

| Flag | Description |
|------|-------------|
| `--config CONFIG` | Path to config file (required for new runs) |
| `--run_name NAME` | Run name. Creates if not exists, resumes if exists |
| `--timestamp` | Append timestamp to --run_name (e.g., `mnli_00` → `mnli_00_2026-01-31_12-00Z`) |
| `--force` | Overwrite existing run (deletes and starts fresh) |
| `--max_steps N` | Max training steps (for testing) |
| `--seed N` | Random seed (default: 42) |

---

## Quick Reference

```bash
# New run (auto-named with timestamp)
python train.py --config configs/mnli/.../00.yaml

# New run (custom name, no timestamp)
python train.py --run_name mnli_00 --config configs/mnli/.../00.yaml

# New run (custom name + timestamp = unique each time)
python train.py --run_name mnli_00 --timestamp --config configs/mnli/.../00.yaml

# Resume existing
python train.py --run_name mnli_00

# Overwrite existing
python train.py --run_name mnli_00 --config configs/... --force
```

---

## All Scenarios

### 1. Auto-generated name (no --run_name)

```bash
python train.py --config configs/mnli/bert_distilbert/ablations/00_no-teacher_ce-only.yaml
```

| Field | Value |
|-------|-------|
| **Output Dir** | `./outputs/mnli_00_no-teacher_ce-only_2026-01-31_12-00Z/` |
| **Run Name** | `mnli_00_no-teacher_ce-only_2026-01-31_12-00Z` |
| **W&B Run ID** | `mnli_00_no-teacher_ce-only_2026-01-31_12-00Z` |

---

### 2. Manual name (CREATE)

```bash
python train.py --run_name mnli_00 --config configs/...
```

| Field | Value |
|-------|-------|
| **Output Dir** | `./outputs/mnli_00/` |
| **Run Name** | `mnli_00` |
| **W&B Run ID** | `mnli_00` |

---

### 3. Manual name (RESUME)

```bash
python train.py --run_name mnli_00
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
python train.py --run_name mnli_00 --force --config configs/...
```

| Field | Value |
|-------|-------|
| **Output Dir** | `./outputs/mnli_00/` (deleted & recreated) |
| **Run Name** | `mnli_00` |
| **W&B Run ID** | `mnli_00` (fresh) |

---

### 5. Manual name + --timestamp (UNIQUE RUN)

```bash
python train.py --run_name mnli_00 --timestamp --config configs/...
```

| Field | Value |
|-------|-------|
| **Output Dir** | `./outputs/mnli_00_2026-01-31_12-00Z/` |
| **Run Name** | `mnli_00_2026-01-31_12-00Z` |
| **W&B Run ID** | `mnli_00_2026-01-31_12-00Z` |

Each run is unique (timestamp ensures no collision).

---

### 6. Manual name + --timestamp + --force (UNIQUE + OVERWRITE)

```bash
python train.py --run_name mnli_00 --timestamp --force --config configs/...
```

| Field | Value |
|-------|-------|
| **Output Dir** | `./outputs/mnli_00_2026-01-31_12-00Z/` |
| **Run Name** | `mnli_00_2026-01-31_12-00Z` |
| **Behavior** | Deletes if exists (rare: same minute), creates fresh |

Mostly redundant since `--timestamp` makes names unique, but valid for re-runs within the same minute.

---

### 7. Long name > 64 characters (TRUNCATION)

**Auto-generated (preserves timestamp):**
```
Input:  mnli_very_long_experiment_name_with_details_2026-01-31_12-00Z (70 chars)
Output: mnli_very_long_experiment_name_with_de_2026-01-31_12-00Z (64 chars)
```
Truncation: `{config_name[:46]}_{timestamp}`

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
  "config": "configs/mnli/.../00_no-teacher_ce-only.yaml",
  "config_id": "mnli_00_no-teacher_ce-only",
  "created": "2026-01-31_12-00Z",
  "seed": 42
}
```

### W&B Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `WANDB_PROJECT` | `Knowledge Distillation` | Project name |
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
python train.py
```
```
Error: No --run_name or --config provided.

Usage:
  New run:    python train.py --config CONFIG
  Named run:  python train.py --run_name NAME --config CONFIG
  Resume:     python train.py --run_name EXISTING_RUN
```

### E2: New run_name, no config
```bash
python train.py --run_name mnli_00
```
```
Error: Run 'mnli_00' does not exist and --config not provided.

Usage:
  New run:    python train.py --run_name mnli_00 --config CONFIG
  Resume:     python train.py --run_name EXISTING_RUN
  Overwrite:  python train.py --run_name NAME --config CONFIG --force
```

### E3: --force without config
```bash
python train.py --run_name mnli_00 --force   # when run exists
```
```
Error: --config is required for new runs.

Usage:
  python train.py --run_name NAME --config CONFIG [--force]
```

### E4: Resume with --config provided
```bash
python train.py --run_name mnli_00 --config configs/...   # when run exists
```
```
Error: --config cannot be used when resuming.
The stored config at ./outputs/mnli_00/config.yaml will be used.

Usage:
  Resume:     python train.py --run_name mnli_00
  Overwrite:  python train.py --run_name mnli_00 --config CONFIG --force
```

### E5: Corrupted run (missing config.yaml)
```bash
python train.py --run_name mnli_00   # run dir exists but no config.yaml
```
```
Error: No stored config found at ./outputs/mnli_00/config.yaml
The run directory exists but appears corrupted.

Options:
  Delete and restart:  python train.py --run_name mnli_00 --config CONFIG --force
  Or manually delete:  rm -rf ./outputs/mnli_00
```

---

## Warnings

| Command | Warning |
|---------|---------|
| `python train.py --force --config ...` | --force without --run_name is redundant |
| `python train.py --timestamp --config ...` | --timestamp without --run_name is redundant |

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

## Related Documents

| Document | Purpose |
|----------|---------|
| [research-framing.md](research-framing.md) | Research context and distillation strategies |
| [framework-direction.md](framework-direction.md) | Vision, mastery path, learning roadmap |
| [experimental_findings.md](experimental_findings.md) | Detailed experiment results and analysis |
| [diffusion-distillation.md](diffusion-distillation.md) | Diffusion model distillation guide |
