# Scripts

## SLURM: Graceful Shutdown & Resume

All SLURM scripts use `#SBATCH --signal=SIG@120` to send a signal 120 seconds
before the time limit. This gives Python time to finish the current unit of work,
save state, and exit cleanly.

### Training (`run_slurm.sh`)

- Signal: `SIGUSR1@120`
- Resume: Automatic — script checks `.incomplete_jobs` and resubmits via `sbatch`
- State: HuggingFace Trainer checkpoints + `training_state.json`
- Job tracking: via `.incomplete_jobs` JSON file

### Evaluation (`run_olmocr_bench.sh` + `eval_olmocr_bench.py`)

- Signal: `SIGTERM@120`
- Resume: Manual — script prints the exact `sbatch` command to resubmit
- State: Per-page atomic writes (temp file + rename); skip existing on resume
- Job tracking: `metadata.json` in output dir contains ordered `slurm_jobs` list
- Model validation: `--resume` checks `metadata.json` to prevent model mismatch

### Writing New SLURM Scripts

1. Add `#SBATCH --signal=SIGTERM@120` (or `SIGUSR1` if you need a custom handler)
2. Register a signal handler in Python that sets a shutdown flag
3. Check the flag at safe points in the main loop (between units of work)
4. Save all state before exiting
5. Print or log the exact command needed to resume
6. Use atomic writes for any output files (write to `.tmp`, then `os.rename`)
