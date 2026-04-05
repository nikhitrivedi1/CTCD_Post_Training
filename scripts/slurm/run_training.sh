#!/bin/bash
#SBATCH --job-name=forge-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.out
#SBATCH --signal=B:SIGUSR1@120

# SLURM training script (SFT, KD, RL) with auto-resubmission.
#
# Tracks incomplete jobs via .incomplete_jobs and auto-resubmits on time
# limit. Signal handling (SIGUSR1 trap, kill timer) lives in
# slurm_essentials.sh. Python's SLURMSignalCallback saves a checkpoint
# on SIGUSR1, then the shell resubmits.
#
# Usage:
#   sbatch scripts/slurm/run_training.sh <config>                    # Auto-named run
#   sbatch scripts/slurm/run_training.sh <run_name> <config>         # Named run
#
# Examples:
#   # SFT (auto-named):
#   sbatch scripts/slurm/run_training.sh configs/ocr/formatting/qwen3vl-2b/00_sft.yaml
#   -> outputs/ocr/formatting/qwen3vl-2b/00_sft/20260311-1200Z/
#
#   # RL with named run:
#   sbatch scripts/slurm/run_training.sh fmt_grpo configs/ocr/formatting/qwen3vl-2b/rl/00_grpo.yaml
#   -> outputs/fmt_grpo/
#
#   # RL with pre-merged SFT adapter:
#   ADAPTER=outputs/fmt_sft sbatch scripts/slurm/run_training.sh fmt_grpo configs/.../00_grpo.yaml
#
# Environment variables:
#   SEED=123                    - Custom random seed
#   MAX_STEPS=10                - Limit steps (for testing)
#   EPOCHS=5                    - Override num_epochs from config
#   ADAPTER=path                - SFT adapter to merge before RL training
#   TEACHER_ADAPTER=path        - Teacher checkpoint/adapter for KD (overrides model.teacher.name)
#   TIMESTAMP=1                 - Append timestamp to run_name
#   FORCE=1                     - Overwrite existing run
#   DEBUG=1                     - Enable verbose DEBUG logging
#
# Internal (set by auto-resubmit, not for manual use):
#   _FORGE_RESUME_PATH=name    - Resume path relative to outputs/
#   PREV_JOB_ID=12345          - Previous job ID (cleans up .incomplete_jobs)
#
# First-time setup:
#   ssh login.explorer.northeastern.edu
#   module load anaconda3/2024.06
#   conda env create -f environment.yml
#   conda activate forge
#   source scripts/env/init.sh

source scripts/slurm/slurm_essentials.sh

slurm_setup_env
slurm_print_header

# Run Python in background so slurm_wait_child can handle SIGUSR1.
run_forge() {
    python forge.py "$@" &
    slurm_wait_child $!
    return $?
}

# Build common ARGS from env vars
ARGS=""
[ -n "$SEED" ]      && ARGS="$ARGS --seed $SEED"           && echo "Seed: $SEED"
[ -n "$MAX_STEPS" ] && ARGS="$ARGS --max_steps $MAX_STEPS" && echo "Max steps: $MAX_STEPS"
[ -n "$EPOCHS" ]    && ARGS="$ARGS --epochs $EPOCHS"       && echo "Epochs: $EPOCHS"
[ -n "$ADAPTER" ]           && ARGS="$ARGS --adapter $ADAPTER"                   && echo "Adapter: $ADAPTER"
[ -n "$TEACHER_ADAPTER" ]   && ARGS="$ARGS --teacher_adapter $TEACHER_ADAPTER" && echo "Teacher adapter: $TEACHER_ADAPTER"
[ -n "$DEBUG" ]             && ARGS="$ARGS --debug"                             && echo "Debug mode: verbose logging enabled"

# Check if this is a resubmit (resume via _FORGE_RESUME_PATH env var)
if [ -n "$_FORGE_RESUME_PATH" ]; then
    echo ""
    echo "========================================="
    echo "Resubmit resume mode"
    echo "Resume path: outputs/$_FORGE_RESUME_PATH"
    echo "========================================="
    echo ""

    CMD="python forge.py --resume \"outputs/$_FORGE_RESUME_PATH\" $ARGS"
    echo "CMD: $CMD"
    run_forge --resume "outputs/$_FORGE_RESUME_PATH" $ARGS || EXIT_CODE=$?
else
    # Initial submission: parse positional arguments
    # Detect if $1 is config (ends with .yaml/.yml) or run_name
    if [[ "$1" == *.yaml ]] || [[ "$1" == *.yml ]]; then
        CONFIG="$1"
        RUN_NAME=""
    elif [ -n "$1" ]; then
        RUN_NAME="$1"
        CONFIG="$2"
    else
        echo "Error: No arguments provided."
        echo "Usage:"
        echo "  sbatch scripts/slurm/run_training.sh <config>              # Auto-named run"
        echo "  sbatch scripts/slurm/run_training.sh <run_name> <config>   # Named run"
        exit 1
    fi

    # Add create-only flags
    [ -n "$TIMESTAMP" ] && ARGS="$ARGS --timestamp" && echo "Timestamp mode: appending timestamp to run_name"
    [ -n "$FORCE" ]     && ARGS="$ARGS --force"      && echo "Force mode: will overwrite existing run"

    # Pre-completion check for named runs (prevents wasteful GPU allocation)
    if [ -n "$RUN_NAME" ] && [ -z "$TIMESTAMP" ] && [ -z "$FORCE" ]; then
        OUTPUT_DIR="./outputs/$RUN_NAME"
        COMPLETE=$(slurm_is_complete "$OUTPUT_DIR")
        if [ "$COMPLETE" = "true" ] && [ -f "$OUTPUT_DIR/training_state.json" ]; then
            echo "Training already complete for: $RUN_NAME"
            exit 0
        fi
    fi

    # Run training
    echo ""
    echo "========================================="
    echo "Starting training..."
    echo "Config: $CONFIG"
    if [ -n "$RUN_NAME" ]; then
        echo "Run name: $RUN_NAME"
    fi
    echo "========================================="
    echo ""

    if [ -n "$RUN_NAME" ]; then
        if [ -z "$CONFIG" ]; then
            echo "Error: --config required for new run '$RUN_NAME'"
            echo "Usage: sbatch scripts/slurm/run_training.sh $RUN_NAME <config.yaml>"
            exit 1
        fi
        CMD="python forge.py --run_name \"$RUN_NAME\" --config \"$CONFIG\" $ARGS"
        echo "CMD: $CMD"
        run_forge --run_name "$RUN_NAME" --config "$CONFIG" $ARGS || EXIT_CODE=$?
    else
        CMD="python forge.py --config \"$CONFIG\" $ARGS"
        echo "CMD: $CMD"
        run_forge --config "$CONFIG" $ARGS || EXIT_CODE=$?
    fi
fi

EXIT_CODE=${EXIT_CODE:-0}

# Auto-resubmit if training is incomplete.
# Skipped when called from a sweep worker (_FORGE_SWEEP_WORKER=1) — the sweep
# manages resubmission itself and owns the SLURM job ID namespace.
if [ -n "${_FORGE_SWEEP_WORKER:-}" ]; then
    [ $EXIT_CODE -ne 0 ] && echo "Job failed with exit code: $EXIT_CODE"
elif [ $EXIT_CODE -eq 0 ] && slurm_read_incomplete; then
    if [ "$_JOB_TYPE" = "training" ]; then
        COMPLETE=$(slurm_is_complete "$_OUTPUT_DIR")
        if [ "$COMPLETE" = "false" ]; then
            echo ""
            echo "========================================="
            echo "Training incomplete. Resubmitting job..."
            echo "Resume: outputs/$_RESUME_PATH"
            echo "========================================="

            _FORGE_RESUME_PATH="$_RESUME_PATH" PREV_JOB_ID="$SLURM_JOB_ID" \
                SEED="$SEED" MAX_STEPS="$MAX_STEPS" EPOCHS="$EPOCHS" ADAPTER="$ADAPTER" \
                TEACHER_ADAPTER="$TEACHER_ADAPTER" DEBUG="$DEBUG" \
                sbatch "$0"
        else
            echo ""
            echo "Training complete!"
        fi
    fi
elif [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "========================================="
    echo "Job failed with exit code: $EXIT_CODE"
    echo "========================================="
else
    echo ""
    echo "Training complete!"
fi

slurm_print_footer
