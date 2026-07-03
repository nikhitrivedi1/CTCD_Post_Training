#!/bin/bash
# Environment setup for the Forge project on HPC.
#
# Activates the conda environment and sets all required environment variables.
# Single entry point for SLURM jobs and interactive (OOD/SSH) sessions.
#
# Usage:
#   source scripts/env/activate.sh
#
# What it sets:
#   PATH             — Prepends conda's bin dir (ensures conda executables first)
#   LD_PRELOAD       — Preloads conda's libstdc++ so torch/CUDA gets the newer
#                      version with CXXABI_1.3.15 (needed by olmocr's ICU etc.).
#                      Uses LD_PRELOAD instead of LD_LIBRARY_PATH to avoid
#                      overriding other libraries (e.g. OpenSSL for git).
#   HF_HOME          — Redirects HuggingFace cache to /scratch (avoids home quota)

# Activate conda environment if not already active
if [ -z "$CONDA_PREFIX" ] || [ "$(basename "$CONDA_PREFIX")" != "forge" ]; then
    eval "$(conda shell.bash hook)"
    conda activate forge
fi

# Conda executables before system (git, python, etc.)
export PATH="$CONDA_PREFIX/bin:$PATH"

# Preload conda's libstdc++ (fixes CXXABI conflict without affecting other libs)
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

# HuggingFace cache on scratch (home dir has quota limits on HPC)
if [ -d "/scratch" ]; then
    export HF_HOME="/scratch/$USER/.cache/huggingface"
fi

# Source .env for HF_TOKEN, WANDB_API_KEY, etc.
_ACTIVATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_REPO_DIR="$(cd "$_ACTIVATE_DIR/../.." && pwd)"
if [ -f "$_REPO_DIR/.env" ]; then
    set -a
    source "$_REPO_DIR/.env"
    set +a
fi

# Load shell aliases (SLURM shortcuts, etc.)
source "$_ACTIVATE_DIR/aliases.sh"

unset _ACTIVATE_DIR _REPO_DIR