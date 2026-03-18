# Native SLURM Job Chaining

## Context

Forge's SLURM infrastructure auto-resubmits jobs on time limits (8h wall clock) via `.incomplete_jobs` registry. But it can only restart the *same* job. We need to chain *different* jobs sequentially, e.g., `ocr_sft -> fmt_sft -> fmt_grpo -> bench`. This enables the KD pipeline (Phase 1: two-stage LoRA SFT) to run unattended.

## Design

**Standalone `run_chain.sh` orchestrator** that submits and monitors steps. `run_training.sh`, `slurm_essentials.sh`, and `forge.py` are completely untouched. `run_bench.sh` gets a minor addition for env var consistency.

### Chain file format

Reserved keys per step:
- `name` - step label (for chain display/logging, always present)
- `run_name` - the `--run_name` arg to forge.py (**required** for chain steps)
- `config` - config YAML path (for training) or bench type name (for bench)
- `type` - `training` (default) or `bench`
- `depends_on` - string or list of `run_name`(s) this step waits for. If absent, step is a root (starts immediately).
- `env:` - dict of env vars exported to `run_training.sh`/`run_bench.sh`

This is a DAG. Order of steps in the YAML is irrelevant - the dependency graph determines execution order. Roots (no `depends_on`) start immediately, steps become ready when all their deps complete. Chaining only supports named runs - output dir is always `outputs/{run_name}`, so env values reference outputs directly.

```yaml
# chains/formatting_pipeline.yaml
# Sequential: ocr_sft -> fmt_sft -> fmt_grpo -> bench
steps:
  - name: OCR Grounding
    run_name: ocr_sft
    config: configs/ocr/olmocr-mix/qwen3vl-2b/base.yaml

  - name: Formatting SFT
    run_name: fmt_sft
    depends_on: ocr_sft
    config: configs/ocr/formatting/qwen3vl-2b/00_sft.yaml
    env:
      ADAPTER: outputs/ocr_sft

  - name: Formatting GRPO
    run_name: fmt_grpo
    depends_on: fmt_sft
    config: configs/ocr/formatting/qwen3vl-2b/rl/00_grpo.yaml
    env:
      ADAPTER: outputs/fmt_sft

  - name: Formatting bench
    run_name: fmt_bench
    type: bench
    depends_on: fmt_grpo
    config: formatting
    env:
      CHECKPOINT: outputs/fmt_grpo
```

Parallel DAG example (two model sizes trained independently, then KD):

```yaml
# chains/kd_pipeline.yaml
steps:
  - name: OCR 2B
    run_name: ocr_sft_2b
    config: configs/ocr/olmocr-mix/qwen3vl-2b/base.yaml

  - name: OCR 7B
    run_name: ocr_sft_7b
    config: configs/ocr/olmocr-mix/qwen3vl-7b/base.yaml

  - name: Fmt SFT 2B
    run_name: fmt_sft_2b
    depends_on: ocr_sft_2b
    config: configs/ocr/formatting/qwen3vl-2b/00_sft.yaml
    env:
      ADAPTER: outputs/ocr_sft_2b

  - name: Fmt SFT 7B
    run_name: fmt_sft_7b
    depends_on: ocr_sft_7b
    config: configs/ocr/formatting/qwen3vl-7b/00_sft.yaml
    env:
      ADAPTER: outputs/ocr_sft_7b

  - name: KD 7B->2B
    run_name: kd_7b_to_2b
    depends_on: [fmt_sft_2b, fmt_sft_7b]
    config: configs/kd/7b_to_2b.yaml
```

Per-step env overrides:

```yaml
  - name: OCR Grounding (debug)
    run_name: ocr_sft_debug
    config: configs/ocr/olmocr-mix/qwen3vl-2b/base.yaml
    env:
      MAX_STEPS: 10
      DEBUG: 1
      FORCE: 1
```

Usage:
```bash
# Login node (in tmux) - recommended for most cases
scripts/slurm/run_chain.sh chains/formatting_pipeline.yaml

# CPU node (unattended, 48h max with auto-resubmit)
sbatch scripts/slurm/run_chain.sh chains/formatting_pipeline.yaml

# Resume an incomplete chain (either mode)
scripts/slurm/run_chain.sh --resume formatting_pipeline
sbatch scripts/slurm/run_chain.sh --resume formatting_pipeline
```

### Dual-mode execution

The script auto-detects its mode by checking `$SLURM_JOB_ID`:

| | Login node | CPU node (sbatch) |
|---|---|---|
| **How to run** | `scripts/slurm/run_chain.sh chains/pipe.yaml` (in tmux) | `sbatch scripts/slurm/run_chain.sh chains/pipe.yaml` |
| **Wall time** | Unlimited | 48h (`short` partition, auto-resubmits) |
| **SIGUSR1 handling** | Not needed | Trap + auto-resubmit via `--resume` |
| **Gap risk** | None (always running) | Possible (orchestrator pending in queue while child jobs progress) |
| **Best for** | Interactive pipelines, short chains | Long unattended pipelines |

SBATCH headers in the file are ignored when run directly. Same script, two modes.

### How it works

For each step, the orchestrator:

1. **Account check**: picks the best fairshare account via `sshare` (same logic as `sbest()` in aliases.sh) and sets it as default before submitting
2. Reads step from chain YAML
3. Submits the step via `sbatch run_training.sh` (or `run_bench.sh`) with env vars
4. Updates `.incomplete_chains` with current step progress
5. Polls `training_state.json` for `pipeline_complete` (sleep loop, ~60s interval)
6. On completion: logs success, moves to next step
7. On failure: detects via `.incomplete_jobs` (registered job no longer in squeue, not complete), halts chain

```
run_chain.sh (login node tmux or CPU short partition)
  |
  | [roots: ocr_sft_2b, ocr_sft_7b have no deps - submit both]
  +-- sbatch run_training.sh ocr_sft_2b configs/...
  +-- sbatch run_training.sh ocr_sft_7b configs/...
  |     (running in parallel on separate GPUs)
  |
  | [ocr_sft_2b completes -> fmt_sft_2b is ready]
  +-- sbatch run_training.sh fmt_sft_2b configs/...
  |     (ocr_sft_7b still running)
  |
  | [ocr_sft_7b completes -> fmt_sft_7b is ready]
  +-- sbatch run_training.sh fmt_sft_7b configs/...
  |
  | [both fmt_sft_2b and fmt_sft_7b complete -> kd_7b_to_2b is ready]
  +-- sbatch run_training.sh kd_7b_to_2b configs/...
  |
  | All steps complete. Chain done.
```

The orchestrator loop:
```
while unfinished steps exist:
    ready = steps whose deps are all complete, not yet submitted
    for each ready step: pick_best_account(), submit
    wait_for_any(running_steps)  # polls all, returns on first completion/failure
    on failure: halt chain
```

The existing `run_training.sh` resubmit loop works as usual. `run_chain.sh` just watches from the outside.

### Failure detection

The orchestrator tracks each submitted step's SLURM job ID (from `sbatch` output). The poll loop checks activity first, then completion (order matters to avoid false positives on bench steps that lack `training_state.json`):

1. Is the tracked job ID still in `squeue`? -> active, keep waiting
2. If not, scan `.incomplete_jobs` for entry with `resume_path == $run_name` -> if found, update tracked job ID (resubmit happened), keep waiting
3. If neither active nor in `.incomplete_jobs`, check `slurm_is_complete("outputs/$run_name")` -> if true, step succeeded
4. Otherwise, step failed - halt chain

This handles all cases: normal completion, multi-resubmit completion, OOM crashes, config errors.

### Account check before each step

Before submitting each GPU job, the orchestrator picks the best fairshare account (reuses the `sshare` + `sacctmgr` logic from `sbest()` in `scripts/env/aliases.sh`). This ensures every step in a long chain uses the optimal account, since fairshare scores change over time as GPU hours are consumed.

### `.incomplete_chains` registry

Parallel to `.incomplete_jobs`. Tracks in-progress chains for resumption.

```json
{
  "formatting_pipeline": {
    "chain_file": "chains/formatting_pipeline.yaml",
    "completed": ["ocr_sft", "fmt_sft"],
    "submitted": ["fmt_grpo"],
    "started": "2026-03-13T10:00:00"
  }
}
```

- **Key**: chain name, derived from the chain YAML filename (without path/extension)
- `completed`: `run_name`s that finished successfully
- `submitted`: `run_name`s that were submitted but not yet completed
- No job IDs stored - `.incomplete_jobs` is the source of truth for which SLURM job currently owns a run_name (handles resubmits that changed the job ID while the orchestrator was down)
- Written at chain start, updated after each step is submitted or completes
- Removed when the full chain finishes
- Uses file locking (same `fcntl.flock` pattern as `.incomplete_jobs`)

**Resumption** (manual or auto-resubmit): `run_chain.sh --resume <chain_name>` reads `.incomplete_chains`, then reconciles each `submitted` step:
1. `slurm_is_complete`? -> move to `completed`
2. Entry in `.incomplete_jobs` with matching `resume_path`? -> still active (may have resubmitted with new job ID while orchestrator was down), resume monitoring
3. Neither? -> step failed during the gap. Halt chain.

Then continues the DAG from the reconciled state.

### Chain orchestrator auto-resubmit (SLURM mode only)

When sbatched, the orchestrator can hit the 48h wall time. It uses the same SIGUSR1 pattern as training jobs:

```
#SBATCH --signal=B:SIGUSR1@120
```

On SIGUSR1:
1. Update `.incomplete_chains` with current completed/running state
2. Resubmit `run_chain.sh --resume <chain_name>`
3. Exit

Running steps are independent SLURM jobs, unaffected by the orchestrator's resubmit. The resumed orchestrator picks them back up via `.incomplete_chains`.

On the login node, none of this applies - the script just runs until the chain completes or fails. If killed (node reboot, tmux lost), manually re-run with `--resume`.

### Skip-ahead for already-complete steps

Before submitting a step, check if `outputs/{run_name}/training_state.json` already has `pipeline_complete: true`. If so, skip it and move to the next step. This works for both fresh starts and resumes - the chain always converges to the first incomplete step.

## Files to create/modify

### 1. `scripts/slurm/run_chain.sh` (new file)

SLURM script for chain orchestration:

```
#SBATCH --job-name=forge-chain
#SBATCH --partition=short
#SBATCH --time=48:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/chain_%j.out
#SBATCH --error=logs/chain_%j.out
#SBATCH --signal=B:SIGUSR1@120
```

SBATCH headers are ignored when run directly on login node. When sbatched, traps SIGUSR1 to auto-resubmit with `--resume`.

**Mode detection**: `if [ -n "$SLURM_JOB_ID" ]` to gate SIGUSR1 trap setup and auto-resubmit logic.

Source `slurm_essentials.sh` for `slurm_setup_env` and `slurm_is_complete`. Implement:

- **`validate_chain(chain_file)`** - Python snippet. Checks: YAML parses, every step has `name` + `run_name` + `config`, valid `type`, no duplicate `run_name`, all `depends_on` references point to existing `run_name`s, no cycles in the dependency graph. Exits on failure.
- **`read_chain_step(chain_file, step_idx)`** - Python snippet. Sets `_STEP_NAME`, `_STEP_RUN_NAME`, `_STEP_CONFIG`, `_STEP_TYPE`, `_STEP_ENV`. Returns 1 if out of bounds.
- **`chain_step_count(chain_file)`** - prints step count.
- **`pick_best_account()`** - reads `sshare`, picks highest fairshare account, sets as default via `sacctmgr`. Same logic as `sbest()` in `aliases.sh`.
- **`submit_step()`** - calls `pick_best_account()`, exports env vars from `_STEP_ENV`, then runs `sbatch`. For training: passes `run_name config` as positional args to `run_training.sh`. For bench: passes `config` (bench type) as first arg to `run_bench.sh`, auto-adds `OUTPUT=outputs/{run_name}`. Captures job ID from sbatch output. No special-casing of env vars - both run_X scripts read from env uniformly.
- **`wait_for_any(run_names...)`** - poll loop checking all running steps. Returns the `run_name` of the first step to complete (or fail). Uses `slurm_is_complete` and `.incomplete_jobs`/`squeue` for each.
- **`update_incomplete_chain(chain_name, chain_file, completed, submitted)`** - Python snippet. Writes/updates `.incomplete_chains` entry (`completed`/`submitted` lists) with file locking.
- **`remove_incomplete_chain(chain_name)`** - Python snippet. Removes entry from `.incomplete_chains`.
- **Main loop** (DAG executor):
  - Parse args: `chain_file` for new chain, `--resume chain_name` for resume
  - If resume: read `.incomplete_chains` to get chain_file
  - Validate chain (including cycle detection), register in `.incomplete_chains`
  - Mark already-complete steps (skip-ahead)
  - Loop: find ready steps (deps met, not submitted), submit them in parallel, `wait_for_any`, update `.incomplete_chains`, repeat
  - On failure: halt chain, log which step failed
  - On full completion: remove from `.incomplete_chains`

### 2. `chains/formatting_pipeline.yaml` (new file)

Example chain definition (see chain format section above).

### 3. `scripts/slurm/run_bench.sh` (minor change)

Add env var support to match `run_training.sh` pattern. Before the `python $SCRIPT` call, read env vars and append to `BENCH_ARGS`:

```bash
[ -n "$CHECKPOINT" ] && BENCH_ARGS="$BENCH_ARGS --checkpoint $CHECKPOINT"
[ -n "$MODEL" ]      && BENCH_ARGS="$BENCH_ARGS --model $MODEL"
[ -n "$OUTPUT" ]     && BENCH_ARGS="$BENCH_ARGS --output $OUTPUT"
[ -n "$SHARD" ]      && BENCH_ARGS="$BENCH_ARGS --shard $SHARD"
[ -n "$DEBUG" ]      && BENCH_ARGS="$BENCH_ARGS --debug"
```

CLI args still work as before. Env vars just provide a second way to set them (used by the chain orchestrator).

## Not changing

- `forge.py`
- `run_training.sh`
- `slurm_essentials.sh`
- `src/utils/jobs.py`
- `src/utils/checkpoint.py`

## Verification

1. **Validation**: Run with an invalid chain YAML (missing run_name, duplicate names). Verify it fails with clear errors before submitting any jobs.
2. **Local dry-run**: `MAX_STEPS=2` chain with two steps. Verify step 0 completes, step 1 starts automatically.
3. **Skip-ahead**: Run a chain where step 0 is already complete. Verify it skips to step 1.
4. **Failure halt**: Chain where step 1 has an invalid config. Verify chain detects failure and halts with clear log.
5. **Env passthrough**: Verify ADAPTER and other env vars reach `run_training.sh` correctly.
