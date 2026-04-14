# Scripts

## Directory Structure

```
scripts/
  slurm/                    # SLURM job scripts + shared utilities
    slurm_essentials.sh     # Shared: SIGUSR1 trap, env setup, GPU detection, .incomplete_jobs reader
    run_training.sh         # Training with auto-resubmit
    run_bench.sh            # Benchmark eval (olmocr/omnidoc) with auto-resubmit
    run_score.sh            # Score-only (CPU, no GPU)
    run_data_prep.sh        # Data preparation (olmocr-mix, synthmix, omnidocbench)
  gpu_keep-alive.py         # Prevent idle GPU timeout in OOD sessions
  eval/
    olmocr_bench.py         # olmOCR-Bench scoring
    omnidoc_bench.py        # OmniDocBench formatting scoring
  data/
    prepare.py                  # Unified data prep CLI (all datasets)
    olmocr_mix_pipeline.py      # Internal: olmOCR-mix download + render pipeline
    formatting_generator.py     # Internal: synthetic formatting data generator
  models/
    push_to_hub.py          # Push trained models/checkpoints to HF Hub
    pull_from_hub.py        # Download models from HF Hub
  env/
    init.sh                 # One-time: install conda hooks for PATH/LD_LIBRARY_PATH
    activate.sh             # Per-session: set LD_LIBRARY_PATH, source .env
  run_colab.ipynb           # Google Colab notebook
```

## Environment Setup

### First-time setup

```bash
conda env create -f environment.yml
conda activate forge-ctcd
source scripts/env/init.sh
```

`init.sh` installs conda hooks so that `conda activate forge-ctcd`
automatically sets `PATH`, `LD_LIBRARY_PATH`, and `HF_HOME`. This only needs
to be run once per environment (rerun if you recreate the env from scratch).

### Troubleshooting

If you hit library conflicts (e.g. `CXXABI_1.3.15 not found`) in an
interactive session, source the setup script manually:

```bash
source scripts/env/activate.sh
```

### Why this is needed

On HPC nodes, `import torch` loads CUDA libraries which pull in the system's
older `libstdc++.so`. This "poisons" the process - conda packages like olmocr
(via ICU) need `CXXABI_1.3.15` (only in conda's newer `libstdc++.so`), but the
linker already loaded the old one. Setting `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`
ensures conda's libraries are found first.

The `environment.yml` installs `git` via conda so that `LD_LIBRARY_PATH`
doesn't cause an OpenSSL version mismatch with system git.

## SLURM: Graceful Shutdown & Resume

All SLURM scripts source `scripts/slurm/slurm_essentials.sh` which provides:
- **SIGUSR1 trap**: Sets `SIGUSR1_RECEIVED=1` flag, shell survives the signal
- **GPU detection**: Auto-detects NVIDIA GPU (via `nvidia-smi`), conditionally loads CUDA
- **`slurm_setup_env`**: Load modules, activate conda, source .env
- **`slurm_print_header/footer`**: Job info banners (GPU details when available)
- **`slurm_read_incomplete`**: Read `.incomplete_jobs` entry for current job
- **`slurm_is_complete`**: Check `training_state.json` for completion

### Training (`slurm/run_training.sh`)

- Signal: `SIGUSR1@120`
- Python handles SIGUSR1 internally (SLURMSignalCallback)
- Resume: Automatic via `.incomplete_jobs` JSON + `sbatch`
- State: HuggingFace Trainer checkpoints + `training_state.json`

### Evaluation (`slurm/run_bench.sh`)

- Signal: `SIGUSR1@120`
- Shell kills background Python on SIGUSR1 (C extensions block signal handlers)
- Resume: Automatic via `.incomplete_jobs` JSON + `sbatch`
- State: Per-page atomic writes (temp file + rename), skip existing on resume
- Supports both olmOCR-Bench and OmniDocBench via first arg (`olmocr` or `omnidoc`)

### Writing New SLURM Scripts

1. Source `scripts/slurm/slurm_essentials.sh` at the top
2. Call `slurm_setup_env` and `slurm_print_header`
3. Add `#SBATCH --signal=B:SIGUSR1@120` for time limit handling
4. Register a Python signal handler that sets a shutdown flag
5. Check the flag at safe points in the main loop (between units of work)
6. Write to `.incomplete_jobs` before exiting for auto-resubmit
7. Use atomic writes for output files (write to `.tmp`, then `os.replace`)
