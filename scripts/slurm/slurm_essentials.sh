#!/bin/bash
# Shared SLURM utilities for Forge job scripts (training, bench, score).
#
# Source this at the top of any SLURM job script:
#   source scripts/slurm/slurm_essentials.sh
#
# Variables:
#   SIGUSR1_RECEIVED   - Flag set to "1" when SIGUSR1 arrives (trap auto-registered)
#   HAS_GPU            - "1" if NVIDIA GPU detected, "0" otherwise
#
# Functions:
#   slurm_setup_env       - Load modules, activate conda, source .env
#   slurm_print_header    - Print job info banner (GPU info when available)
#   slurm_print_footer    - Print end timestamp
#   slurm_wait_child PID  - Wait for background Python with SIGUSR1/kill timer handling
#   slurm_read_incomplete - Read .incomplete_jobs entry for current job
#   slurm_is_complete DIR - Check if training_state.json shows completion

# ---------------------------------------------------------------------------
# SIGUSR1 trap
# ---------------------------------------------------------------------------
# With B:SIGUSR1, SLURM sends the signal only to the batch shell, not to
# child processes. The shell must forward it to Python. Callers run Python
# in background, set _FORGE_CHILD_PID, and `wait` for it.
#
# When _FORGE_CHILD_PID is set, the trap forwards SIGUSR1 to Python and
# starts a kill timer (_FORGE_KILL_DEADLINE seconds). Python handles the
# signal gracefully (training: SLURMSignalCallback, bench: bench_utils
# shutdown handler). If Python doesn't exit in time, it gets force-killed.
#
# Timeline with --signal=B:SIGUSR1@120:
#   T-120s: SIGUSR1 -> forward to Python, start kill timer
#   T-30s:  Kill timer fires if Python still running (deadline=90s default)
#   T-0s:   SLURM sends SIGKILL (but shell should be done by now)
SIGUSR1_RECEIVED=0
_FORGE_CHILD_PID=""
_FORGE_KILL_DEADLINE=${_FORGE_KILL_DEADLINE:-90}
_FORGE_KILL_TIMER_PID=""
trap '
    SIGUSR1_RECEIVED=1
    echo "SIGUSR1 received by shell ($(date -u +%H:%M:%S))"
    if [ -n "$_FORGE_CHILD_PID" ]; then
        echo "Forwarding SIGUSR1 to Python (PID $_FORGE_CHILD_PID)"
        kill -SIGUSR1 $_FORGE_CHILD_PID 2>/dev/null
        # Start kill timer: if Python does not exit in time, force-kill it
        (
            sleep $_FORGE_KILL_DEADLINE
            if kill -0 $_FORGE_CHILD_PID 2>/dev/null; then
                echo "Kill deadline reached (${_FORGE_KILL_DEADLINE}s) - force-killing Python"
                kill -9 $_FORGE_CHILD_PID 2>/dev/null
            fi
        ) &
        _FORGE_KILL_TIMER_PID=$!
    fi
' SIGUSR1

# ---------------------------------------------------------------------------
# Wait for Python child with signal handling
# ---------------------------------------------------------------------------
# Waits for a background Python process, handling SIGUSR1 interrupts.
# The trap above forwards SIGUSR1 to the child and starts a kill timer.
# This function re-waits until the child actually exits.
#
# Args: $1 = PID of background Python process
# Sets: EXIT_CODE (global)
slurm_wait_child() {
    _FORGE_CHILD_PID=$1
    while true; do
        wait $_FORGE_CHILD_PID 2>/dev/null
        EXIT_CODE=$?
        # If wait was interrupted by SIGUSR1 (128+10=138), re-wait for child
        if [ "$EXIT_CODE" -eq 138 ] && [ "$SIGUSR1_RECEIVED" = "1" ]; then
            continue
        fi
        break
    done
    _FORGE_CHILD_PID=""
    # Clean up kill timer if it was started
    if [ -n "$_FORGE_KILL_TIMER_PID" ]; then
        kill $_FORGE_KILL_TIMER_PID 2>/dev/null
        wait $_FORGE_KILL_TIMER_PID 2>/dev/null
        _FORGE_KILL_TIMER_PID=""
    fi
    # If force-killed by deadline timer, treat as clean exit for resubmit
    if [ $EXIT_CODE -eq 137 ] && [ "$SIGUSR1_RECEIVED" = "1" ]; then
        echo "Python force-killed by deadline timer, treating as clean exit"
        EXIT_CODE=0
    fi
    return $EXIT_CODE
}

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
    [ -n "${PREV_JOB_ID:-}" ] && echo "Previous Job: $PREV_JOB_ID"
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
    echo "Job: ${SLURM_JOB_ID:-N/A}"
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
    # pipeline_complete is set after training AND post-eval both finish.
    # Fall back to training_complete for runs that predate pipeline_complete.
    if 'pipeline_complete' in s:
        is_complete = bool(s.get('pipeline_complete'))
    else:
        is_complete = bool(s.get('training_complete', False))
    print('true' if is_complete else 'false')
except Exception:
    print('false')
"
}
