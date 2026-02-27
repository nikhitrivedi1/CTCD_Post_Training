#!/bin/bash
# Environment setup for the distillation project on HPC.
#
# Activates the conda environment and sets all required environment variables.
# Single entry point for SLURM jobs and interactive (OOD/SSH) sessions.
#
# Usage:
#   source scripts/env/activate.sh
#
# What it sets:
#   PATH             — Prepends conda's bin dir (ensures conda executables first)
#   LD_LIBRARY_PATH  — Prepends conda's lib dir so the dynamic linker finds
#                      conda's newer libstdc++ before the system's. Without this,
#                      torch/CUDA loads the system's older libstdc++ which lacks
#                      CXXABI_1.3.15 needed by conda packages (e.g. olmocr's ICU).
#   HF_HOME          — Redirects HuggingFace cache to /scratch (avoids home quota)

# Activate conda environment if not already active
if [ -z "$CONDA_PREFIX" ] || [ "$(basename "$CONDA_PREFIX")" != "distillation" ]; then
    eval "$(conda shell.bash hook)"
    conda activate distillation
fi

# Conda executables before system (git, python, etc.)
export PATH="$CONDA_PREFIX/bin:$PATH"

# Conda libs before system libs (fixes libstdc++ CXXABI conflict)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# HuggingFace cache on scratch (home dir has quota limits on HPC)
if [ -d "/scratch" ]; then
    export HF_HOME="/scratch/$USER/.cache/huggingface"
fi
