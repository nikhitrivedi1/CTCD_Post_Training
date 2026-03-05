# Run Options & CLI Reference

Complete reference for `forge.py` command-line options and run management.

---

## Command Line Arguments

| Flag | Description |
|------|-------------|
| `--config CONFIG` | Path to config file (required for new runs) |
| `--resume PATH` | Path to output dir to resume (e.g. `outputs/classification/mnli/.../20260304-1200Z`) |
| `--run_name NAME` | Run name for flat output dir (create-only) |
| `--timestamp` | Append timestamp to --run_name |
| `--force` | Overwrite existing named run (deletes outputs and W&B run) |
| `--max_steps N` | Max training steps (for testing) |
| `--epochs N` | Override num_epochs from config |
| `--seed N` | Random seed (default: 42) |
| `--debug` | Enable DEBUG logging and use W&B Test project |

---

## Two Modes: Create and Resume

### Create Mode (`--config`)

```bash
# Auto-named (hierarchical output, always includes timestamp):
python forge.py --config configs/classification/mnli/distilbert/00_sft.yaml
# -> outputs/classification/mnli/distilbert/00_sft/20260304-1200Z/

# Named (flat output):
python forge.py --config configs/classification/mnli/distilbert/00_sft.yaml --run_name seed42
# -> outputs/seed42/

# Named + timestamp:
python forge.py --config configs/classification/mnli/distilbert/00_sft.yaml --run_name seed42 --timestamp
# -> outputs/seed42_20260304-1200Z/
```

| `--run_name` | `--timestamp` | Output dir |
|:---:|:---:|---|
| - | - | `outputs/{stem}/{timestamp}/` |
| Y | - | `outputs/{run_name}/` |
| Y | Y | `outputs/{run_name}_{timestamp}/` |

The **stem** is derived internally from the config path by stripping the `configs/` prefix and `.yaml` suffix:
```
configs/classification/mnli/distilbert/00_sft.yaml
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  stem: classification/mnli/distilbert/00_sft
```

### Resume Mode (`--resume`)

```bash
# Resume auto-named run:
python forge.py --resume outputs/classification/mnli/distilbert/00_sft/20260304-1200Z

# Resume named run:
python forge.py --resume outputs/seed42
```

Resume always uses `--resume` with the full tab-completable path. It is never implicit.
Cannot combine `--resume` with `--config`, `--run_name`, `--timestamp`, or `--force`.
CLI overrides like `--max_steps`, `--epochs`, `--seed`, `--debug` work with `--resume`.

---

## Metadata

### run_metadata.json

Auto-named run:
```json
{
  "stem": "classification/mnli/distilbert/00_sft",
  "display_name": "classification/mnli/distilbert/00_sft/20260304-1200Z",
  "wandb_run_id": "classification-mnli-distilbert-00_sft-20260304-1200Z",
  "config": "configs/classification/mnli/distilbert/00_sft.yaml",
  "created": "20260304-1200Z",
  "seed": 42
}
```

Named run:
```json
{
  "stem": "classification/mnli/distilbert/00_sft",
  "display_name": "seed42",
  "wandb_run_id": "seed42-20260304-1200Z",
  "config": "configs/classification/mnli/distilbert/00_sft.yaml",
  "created": "20260304-1200Z",
  "seed": 42
}
```

- `display_name` is the W&B display name.
- `wandb_run_id` always includes timestamp for uniqueness (W&B tombstones deleted IDs).
- `stem` drives W&B group and tags.

### W&B Naming

| Field | Auto-named | Named | Named + timestamp |
|-------|-----------|-------|-------------------|
| **Display name** | `{stem}/{timestamp}` | `{run_name}` | `{run_name}_{timestamp}` |
| **Run ID** | `{stem[:x]}-{timestamp}` | `{run_name}-{timestamp}` | `{run_name}-{timestamp}` |
| **Group** | parent dir of stem | parent dir of stem | parent dir of stem |
| **Tags** | stem segments (except leaf) | stem segments (except leaf) | stem segments (except leaf) |

Run IDs use `-` not `/` as separator (W&B forbids `/` in run IDs).
Run IDs are truncated to 64 characters, preserving the timestamp suffix.

---

## Error Cases

### No arguments
```
Error: No --config or --resume provided.

Usage:
  New run:    python forge.py --config configs/PATH.yaml
  Named run:  python forge.py --config configs/PATH.yaml --run_name NAME
  Resume:     python forge.py --resume outputs/PATH_TO_RUN
```

### --run_name without --config
```
Error: --config is required for new runs.

Usage:
  python forge.py --run_name NAME --config CONFIG
```

### Named run already exists (without --force)
```
Error: Output directory already exists: ./outputs/seed42

Options:
  Resume:     python forge.py --resume outputs/seed42
  Overwrite:  python forge.py --run_name seed42 --config CONFIG --force
```

### --resume with incompatible flags
```
Error: --config, --run_name cannot be used with --resume.

Usage:
  python forge.py --resume outputs/PATH
```

### --resume with missing metadata
```
Error: No run_metadata.json found in ./outputs/PATH.
Run may have crashed before setup completed.

Start fresh with:
  python forge.py --config CONFIG
```

---

## Warnings

| Command | Warning |
|---------|---------|
| `--force --config ...` (no --run_name) | Redundant: auto-named runs are always unique |
| `--timestamp --config ...` (no --run_name) | Redundant: auto-named runs always include timestamp |

---

## Output Directory Structure

### Auto-named (hierarchical)
```
outputs/
└── classification/
    └── mnli/
        └── distilbert/
            └── 00_sft/
                └── 20260304-1200Z/
                    ├── config.yaml
                    ├── run_metadata.json
                    ├── training_state.json
                    ├── output.log
                    ├── checkpoint-500/
                    └── checkpoint-1000/
```

### Named (flat)
```
outputs/
└── seed42/
    ├── config.yaml
    ├── run_metadata.json
    ├── training_state.json
    ├── output.log
    ├── checkpoint-500/
    └── checkpoint-1000/
```

---

## SLURM Integration

### Initial Submission

```bash
# Auto-named:
sbatch scripts/slurm/run_training.sh configs/classification/mnli/distilbert/00_sft.yaml

# Named:
sbatch scripts/slurm/run_training.sh seed42 configs/classification/mnli/distilbert/00_sft.yaml

# With env var overrides:
TIMESTAMP=1 SEED=123 DEBUG=1 sbatch scripts/slurm/run_training.sh seed42 configs/...
```

### Auto-Resubmission

When a job times out with incomplete training:

1. Script reads `resume_path` from `.incomplete_jobs`
2. Resubmits with `_FORGE_RESUME_PATH` env var
3. New job runs `python forge.py --resume outputs/{resume_path}`
4. Forwards `SEED`, `MAX_STEPS`, `EPOCHS`, `DEBUG` (not `TIMESTAMP` or `FORCE`)

### Job Tracking (.incomplete_jobs)

```json
{
  "12345": {
    "type": "training",
    "resume_path": "classification/mnli/distilbert/00_sft/20260304-1200Z",
    "output_dir": "./outputs/classification/mnli/distilbert/00_sft/20260304-1200Z"
  }
}
```

Job key: `SLURM_JOB_ID` on cluster, `local_{pid}` locally.
Entry removed on successful completion. Resubmit uses `PREV_JOB_ID` to clean up the old entry.
