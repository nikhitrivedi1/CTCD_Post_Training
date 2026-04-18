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
#
# Conda env name:
#   FORGE_CONDA_ENV  — If set, only this env is activated (must exist).
#   Otherwise: try forge-ctcd (from environment.yml), then forge for older setups.

# Activate conda environment if not already active
_forge_conda_env_exists() {
    conda run -n "$1" true 2>/dev/null
}

eval "$(conda shell.bash hook)"

if [ -n "${FORGE_CONDA_ENV:-}" ]; then
    if ! _forge_conda_env_exists "$FORGE_CONDA_ENV"; then
        echo "forge: FORGE_CONDA_ENV=$FORGE_CONDA_ENV is not a valid conda env name" >&2
        unset -f _forge_conda_env_exists 2>/dev/null || true
        return 1
    fi
    _FORGE_TARGET="$FORGE_CONDA_ENV"
elif _forge_conda_env_exists forge-ctcd; then
    _FORGE_TARGET=forge-ctcd
elif _forge_conda_env_exists forge; then
    _FORGE_TARGET=forge
else
    echo "forge: no conda env forge-ctcd or forge. Create CTCD stack: conda env create -f environment.yml" >&2
    echo "  Or: conda create -n forge python=3.11 && pip install -e .  then export FORGE_CONDA_ENV=forge" >&2
    unset -f _forge_conda_env_exists 2>/dev/null || true
    return 1
fi

if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "$CONDA_PREFIX")" != "$_FORGE_TARGET" ]; then
    conda activate "$_FORGE_TARGET"
fi

unset -f _forge_conda_env_exists 2>/dev/null || true
unset _FORGE_TARGET

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
