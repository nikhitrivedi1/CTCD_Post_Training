# Entry Point: forge.py

Main training script for fine-tuning, knowledge distillation, and reinforcement learning.

**Source:** `forge.py` (~1440 lines)

---

## Overview

`forge.py` is the single entry point for all training modes. It orchestrates the full pipeline: CLI parsing, config loading, data preparation, model loading, trainer construction, training, and post-training evaluation. The training mode (SFT, KD, or RL) is determined by the config, not by separate scripts.

---

## CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | required | Path to YAML config file |
| `--resume` | str | None | Path to output dir to resume from |
| `--run_name` | str | None | Custom run name (auto-generated if omitted) |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--max_steps` | int | -1 | Override max training steps |
| `--epochs` | int | None | Override number of epochs |
| `--force` | flag | False | Overwrite existing named run (deletes outputs and W&B run) |
| `--timestamp` | flag | False | Append timestamp to `--run_name` (e.g. `seed42` becomes `seed42_20260308-1200Z`) |
| `--adapter` | str | None | Path to adapter to merge before training (SFT, KD, or RL) |
| `--debug` | flag | False | Enable DEBUG logging + W&B Test project |

---

## Pipeline Flow

```
1. Parse CLI arguments
2. setup_logging(debug)         # Configure console logging
3. Resolve output dir            # Resume vs create mode, --run_name, --force
4. set_reproducibility_seed()   # Seed torch, numpy, random, HF
5. load_config()                # Resume: load_config_snapshot(), Create: load_config() with inheritance
6. add_file_logging()           # Redirect stdout/stderr to output.log
7. detect_environment()         # SLURM / Colab / local
8. Determine training mode      # "rl" top-level key -> GRPO path, else KD/SFT path
9. Load tokenizer/processor
10. Load datasets
11. Resolve --adapter            # _resolve_adapter(): merge adapter into base, cache in base/
12. Load teacher + student models
13. Setup distillation (loss strategy, layer mapping, projections)
14. Create trainer (DistillationTrainer, FinetuningTrainer, or GRPOTrainer)
15. Add callbacks (step metrics, SLURM, checkpoint marker, teacher eval, peak metrics)
16. Pre-training evaluation (baselines)
17. Train
18. _sync_wandb()               # Sync config/metrics to W&B
19. Update TrainingState
20. _save_final_model()         # Unified save: merged model + adapter + base ref
21. Post-training evaluation
22. Mark pipeline complete
23. Cleanup incomplete jobs registry
```

---

## Key Functions

### `setup_logging(debug: bool)`

Configures root logger with UTC timestamps. In debug mode, sets all `src.*` loggers to DEBUG and enables transformers debug logging. In production mode, suppresses warnings via `warnings.filterwarnings("ignore")`.

### `add_file_logging(output_dir: Path, debug: bool)`

Redirects stdout/stderr through `TeeWriter` to capture all output (including tqdm progress bars) to `output.log` in the run directory. Also updates existing logging handlers to write through the TeeWriter.

### `set_reproducibility_seed(seed: int)`

Sets seeds across all libraries for reproducibility:
- `random.seed(seed)`
- `np.random.seed(seed)`
- `torch.manual_seed(seed)` + `torch.cuda.manual_seed_all(seed)`
- `transformers.set_seed(seed)`

### `create_loss_strategy_from_config(distill_config, teacher_model, student_model, device)`

Parses the distillation config section and creates the appropriate loss strategy:

1. **Staged distillation** - if `stages` key exists, builds `StagedDistillationLoss`
2. **Composite loss** - if `loss_strategies` key exists, builds `CombinedDistillationLoss` with weighted sub-strategies
3. **Single loss** - otherwise, uses `get_loss_strategy(loss_type)` from the registry

For composite losses, also resolves layer mapping and projection for each component that needs them.

### `resolve_config_path(config_arg: str) -> Path`

Validates that the config path exists. Tries four candidates in order: the path as-is, `configs/{path}`, `{path}.yaml`, and `configs/{path}.yaml`. Calls `sys.exit()` with a helpful message listing all tried paths if none resolve.

### Shared Training Helpers

These three functions are used by both `_train()` (SFT/KD) and `_train_rl()` (GRPO) to eliminate duplication in post-training logic.

#### `_resolve_adapter(config, output_dir, task, processor, tokenizer)`

Resolves `--adapter` CLI flag into a base model for training. If no adapter path is set, returns the model name string. Otherwise:

1. If `base/config.json` exists (resume), skips re-merge and loads from cache
2. Otherwise: loads base model, loads adapter via PeftModel, merges via `merge_and_unload()`, saves merged model to `base/`

In the SFT path, called from `main()` before `load_models()` - saves to `base/`, then `config["model"]["name"]` is repointed to `str(base_dir)` so `load_models()` picks up the merged weights. In the RL path, called from `_train_rl()` directly and returns the model object for GRPOTrainer.

#### `_sync_wandb(config, task, metadata, output_dir)`

Syncs config and metadata to W&B after training completes. Updates `wandb.config` with the full config dict, adds `primary_metric` from the task registry, and tracks W&B run ID changes (W&B may reassign IDs on resume). Saves updated metadata to disk.

#### `_save_final_model(trainer, config, output_dir, processor, tokenizer)`

Unified model save with the structure described in [save-structure.md](save-structure.md):

- **LoRA runs**: saves adapter first (before merge is destructive), saves base ref (unless base/ already has full weights from `--adapter`), merges and saves merged model
- **Full FT runs**: saves full model via `trainer.save_model()`, saves base ref (with same guard)
- Always saves processor/tokenizer

#### `_save_base_ref(output_dir, model_name, revision)`

Saves a `model_ref.json` to `base/` with the HF Hub model path and commit hash revision. For HF Hub models, this is a complete reproducible reference without duplicating weights.

### `main(args)`

The main orchestration function. Key implementation details:

**Output directory structure:**
```
# Auto-named (no --run_name): hierarchical from config stem
outputs/{stem}/{timestamp}/

# Named run (--run_name=seed42): flat
outputs/seed42/

# Named + timestamp (--run_name=seed42 --timestamp): flat with timestamp
outputs/seed42_20260308-1200Z/

# Contents:
    config.yaml           # Frozen copy of resolved config
    run_metadata.json     # Run identity, W&B linking, seed
    training_state.json   # Progress tracking for resume
    output.log            # All stdout/stderr output
    checkpoint-N/         # HF Trainer checkpoints
        teacher/          # Co-trained teacher (if applicable)
```

**Run naming:** The `stem` is extracted from the config path by `extract_stem()`, which derives a hierarchical name from the config's directory structure (e.g., `classification/mnli/distilbert/bert-base/01_kd`). If `--run_name` is provided, it's used as a flat directory name instead.

**Resume logic:**
1. Check `training_state.json` for `pipeline_complete` flag
2. If complete, skip (return immediately)
3. Find latest checkpoint via `latest_checkpoint` file
4. Pass checkpoint path to `Trainer.train(resume_from_checkpoint=...)`

**RL path (GRPO):**
- Dispatched when `"rl"` exists as a top-level key in the config
- Uses `trl.GRPOTrainer` instead of custom trainers
- Loads datasets via `load_rl_dataset()` (raw prompts, not tokenized)
- Reward functions from `src/training/rewards.py`
- VLM models pass the processor directly to GRPOTrainer via `processing_class=processor`
- Post-training: `_sync_wandb()`, `state.save()`, `_save_final_model()` (same as SFT)

**SFT/KD adapter resolution (`--adapter`):**
- If `config["model"]["adapter_path"]` is set and `base/config.json` doesn't exist: calls `_resolve_adapter()` to merge adapter into base model, save to `base/`
- Then sets `config["model"]["name"] = str(base_dir)` so `load_models()` picks up the merged weights
- On resume, the merged model already exists in `base/`, so merge is skipped

**W&B integration:**
- Project name from config or `"Forge-Default"` (debug mode uses `"Forge-Test"`)
- Run name matches the output directory name
- Tags auto-derived from config stem path segments
- `_sync_wandb()` pushes full config + primary metric after training

**Metadata saved (`run_metadata.json`):**
```json
{
    "stem": "01_kd_frozen-base",
    "display_name": "01_kd_frozen-base",
    "wandb_run_id": "01_kd_frozen-base_20260308-1200Z",
    "config": "configs/.../01_kd.yaml",
    "created": "20260308-1200Z",
    "seed": 42
}
```

Optional fields added for multi-stage lineage: `base_model` (source model ID) and `adapter` (LoRA checkpoint path) when `--adapter` is used.

---

## Training Mode Dispatch

The training mode is determined by the config, not separate entry points:

| Config | Trainer Used | Description |
|--------|-------------|-------------|
| No teacher, no RL | `FinetuningTrainer` | Standard SFT with CE loss |
| Teacher present | `DistillationTrainer` | Knowledge distillation (frozen or co-trained) |
| `rl:` top-level key | `GRPOTrainer` (trl) | Reinforcement learning with reward functions |

---

## Incomplete Jobs Integration

After setup, `forge.py` registers the current run in `.incomplete_jobs` (via `locked_incomplete_jobs()`). On successful pipeline completion, it removes the entry. If the process is killed mid-training, the entry persists and the SLURM shell script reads it to auto-resubmit with the correct resume path.

---

*Source: `forge.py`*
