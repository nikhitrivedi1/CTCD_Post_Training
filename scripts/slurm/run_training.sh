#!/bin/bash
#SBATCH --job-name=forge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --signal=B:SIGUSR1@120

# SLURM script for training on Explorer HPC
# Handles job time limits by automatically resubmitting incomplete jobs
#
# Auto-resubmission: When training is incomplete after 8 hours, the script
# automatically resubmits using --resume. The .incomplete_jobs file tracks
# jobs by SLURM_JOB_ID, supporting concurrent job execution.
#
# Usage:
#   sbatch scripts/slurm/run_training.sh <config>                    # Auto-named run
#   sbatch scripts/slurm/run_training.sh <run_name> <config>         # Named run
#
# Examples:
#   # Auto-named (hierarchical output):
#   sbatch scripts/slurm/run_training.sh configs/classification/mnli/distilbert/00_sft.yaml
#   -> outputs/classification/mnli/distilbert/00_sft/20260304-1200Z/
#
#   # Named run (flat output):
#   sbatch scripts/slurm/run_training.sh seed42 configs/classification/mnli/distilbert/00_sft.yaml
#   -> outputs/seed42/
#
#   # Named run with timestamp:
#   TIMESTAMP=1 sbatch scripts/slurm/run_training.sh seed42 configs/...
#   -> outputs/seed42_20260304-1200Z/
#
# Environment variables:
#   SEED=123      - Custom random seed
#   MAX_STEPS=10  - Limit steps (for testing)
#   EPOCHS=5      - Override num_epochs from config
#   TIMESTAMP=1   - Append timestamp to run_name
#   FORCE=1       - Overwrite existing run
#   DEBUG=1       - Enable verbose DEBUG logging
#
# First-time setup:
#   ssh login.explorer.northeastern.edu
#   module load anaconda3/2024.06
#   conda env create -f environment.yml
#   conda activate forge

source scripts/slurm/slurm_essentials.sh
set -e

slurm_setup_env
slurm_print_header

# Build common ARGS from env vars
ARGS=""

if [ -n "$SEED" ]; then
    ARGS="$ARGS --seed $SEED"
    echo "Custom seed: $SEED"
fi

if [ -n "$MAX_STEPS" ]; then
    ARGS="$ARGS --max_steps $MAX_STEPS"
    echo "Testing mode: max_steps=$MAX_STEPS"
fi

if [ -n "$EPOCHS" ]; then
    ARGS="$ARGS --epochs $EPOCHS"
    echo "Epoch override: $EPOCHS"
fi

if [ -n "$DEBUG" ]; then
    ARGS="$ARGS --debug"
    echo "Debug mode: verbose logging enabled"
fi

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
    python forge.py --resume "outputs/$_FORGE_RESUME_PATH" $ARGS || EXIT_CODE=$?
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
    if [ -n "$TIMESTAMP" ]; then
        ARGS="$ARGS --timestamp"
        echo "Timestamp mode: appending timestamp to run_name"
    fi

    if [ -n "$FORCE" ]; then
        ARGS="$ARGS --force"
        echo "Force mode: will overwrite existing run"
    fi

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
        python forge.py --run_name "$RUN_NAME" --config "$CONFIG" $ARGS || EXIT_CODE=$?
    else
        CMD="python forge.py --config \"$CONFIG\" $ARGS"
        echo "CMD: $CMD"
        python forge.py --config "$CONFIG" $ARGS || EXIT_CODE=$?
    fi
fi

EXIT_CODE=${EXIT_CODE:-0}

# Auto-resubmit if training is incomplete
if [ $EXIT_CODE -eq 0 ] && slurm_read_incomplete; then
    if [ "$_JOB_TYPE" = "training" ]; then
        COMPLETE=$(slurm_is_complete "$_OUTPUT_DIR")
        if [ "$COMPLETE" = "false" ]; then
            echo ""
            echo "========================================="
            echo "Training incomplete. Resubmitting job..."
            echo "Resume: outputs/$_RESUME_PATH"
            echo "========================================="

            _FORGE_RESUME_PATH="$_RESUME_PATH" PREV_JOB_ID="$SLURM_JOB_ID" \
                SEED="$SEED" MAX_STEPS="$MAX_STEPS" EPOCHS="$EPOCHS" DEBUG="$DEBUG" \
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
