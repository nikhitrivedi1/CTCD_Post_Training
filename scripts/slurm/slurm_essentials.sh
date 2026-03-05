#!/bin/bash
# Shared SLURM job utilities for Forge scripts.
#
# Source this at the top of any SLURM job script:
#   source scripts/slurm/slurm_essentials.sh
#
# Provides:
#   SIGUSR1_RECEIVED  - Flag set to 1 when SIGUSR1 arrives (trap is auto-registered)
#   HAS_GPU           - "1" if NVIDIA GPU detected (via nvidia-smi), "0" otherwise
#   slurm_setup_env   - Load anaconda, activate conda, source .env; loads CUDA if GPU present
#   slurm_print_header - Print job info banner (includes GPU info when available)
#
# NOTE: GPU detection currently assumes NVIDIA (nvidia-smi, CUDA). Update
# the HAS_GPU check and slurm_setup_env if non-NVIDIA GPUs are added.
#   slurm_print_footer - Print end timestamp
#   slurm_read_incomplete - Read .incomplete_jobs entry for current job
#   slurm_is_complete  - Check if training_state.json shows completion

# ---------------------------------------------------------------------------
# SIGUSR1 trap
# ---------------------------------------------------------------------------
# Shell survives SIGUSR1 and sets a flag. Callers decide what to do:
#   - Training: ignore flag (Python handles SIGUSR1 via SLURMSignalCallback)
#   - Bench: use flag to kill background Python (C extensions block signals)
# Uses a handler (not '') so children inherit default disposition, not SIG_IGN.
SIGUSR1_RECEIVED=0
trap 'SIGUSR1_RECEIVED=1; echo "SIGUSR1 received by shell ($(date -u +%H:%M:%S))"' SIGUSR1

# ---------------------------------------------------------------------------
# GPU detection (NVIDIA-specific)
# ---------------------------------------------------------------------------
if nvidia-smi &>/dev/null; then
    HAS_GPU=1
else
    HAS_GPU=0
fi

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
slurm_setup_env() {
    if [ "$HAS_GPU" = "1" ]; then
        module load cuda/12.1
    fi
    module load anaconda3/2024.06
    eval "$(conda shell.bash hook)"
    source scripts/env/activate.sh
    mkdir -p logs
}

# ---------------------------------------------------------------------------
# Job header / footer
# ---------------------------------------------------------------------------
slurm_print_header() {
    echo "========================================="
    echo "SLURM Job: $SLURM_JOB_ID"
    echo "Node: $SLURM_NODELIST"
    if [ "$HAS_GPU" = "1" ]; then
        echo "GPU Device: $CUDA_VISIBLE_DEVICES"
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
        echo "GPU: $GPU_NAME ($GPU_MEM)"
    fi
    echo "Start: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "========================================="
}

slurm_print_footer() {
    echo "========================================="
    echo "End: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "========================================="
}

# ---------------------------------------------------------------------------
# .incomplete_jobs registry
# ---------------------------------------------------------------------------
# Read the .incomplete_jobs entry for the current SLURM_JOB_ID.
# Sets global vars: _JOB_TYPE, _RESUME_PATH, _OUTPUT_DIR, _SHARD
# Returns 0 if entry found, 1 if not.
slurm_read_incomplete() {
    _JOB_TYPE=""
    _RESUME_PATH=""
    _OUTPUT_DIR=""
    _SHARD=""
    [ -f ".incomplete_jobs" ] || return 1

    eval "$(python3 -c "
import json, shlex
try:
    jobs = json.load(open('.incomplete_jobs'))
    info = jobs.get('$SLURM_JOB_ID', {})
    print(f'_JOB_TYPE={shlex.quote(info.get(\"type\", \"\"))}')
    print(f'_RESUME_PATH={shlex.quote(info.get(\"resume_path\", \"\"))}')
    print(f'_OUTPUT_DIR={shlex.quote(info.get(\"output_dir\", \"\"))}')
    print(f'_SHARD={shlex.quote(info.get(\"shard\", \"\"))}')
except:
    pass
")"
    [ -n "$_JOB_TYPE" ]
}

# ---------------------------------------------------------------------------
# Completion check
# ---------------------------------------------------------------------------
# Check if training is complete by reading training_state.json.
# Args: $1 = output directory
# Prints "true" or "false".
slurm_is_complete() {
    local state_file="$1/training_state.json"
    if [ ! -f "$state_file" ]; then
        echo "true"
        return
    fi
    python3 -c "
import json
try:
    with open('$state_file') as f:
        s = json.load(f)
    target = s.get('target_epochs') or s.get('target_steps') or float('inf')
    current = s.get('completed_epochs', 0)
    print('true' if current >= target else 'false')
except:
    print('false')
"
}
