#!/bin/bash
#SBATCH --job-name=distillation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --signal=SIGUSR1@120

# SLURM script for knowledge distillation on Explorer HPC
# Handles job time limits by automatically resubmitting incomplete jobs
#
# Auto-resubmission: When training is incomplete after 8 hours, the script
# automatically resubmits with the actual run name. This works for ALL run
# types (auto-named, named, timestamped). The .incomplete_jobs file tracks
# jobs by SLURM_JOB_ID, supporting concurrent job execution.
#
# Usage:
#   sbatch scripts/run_slurm.sh <run_name> <config>   # Named run (new or resume)
#   sbatch scripts/run_slurm.sh <run_name>            # Resume existing run
#   sbatch scripts/run_slurm.sh <config>              # Auto-named run
#
# Examples:
#   # Named run (new)
#   sbatch scripts/run_slurm.sh mnli_00 configs/mnli/bert_distilbert/default.yaml
#   -> outputs/mnli_00/
#
#   # Resume existing run
#   sbatch scripts/run_slurm.sh mnli_00
#   -> outputs/mnli_00/ (resumes from checkpoint)
#
#   # Auto-named run (includes timestamp)
#   sbatch scripts/run_slurm.sh configs/mnli/bert_distilbert/default.yaml
#   -> outputs/mnli_default_2026-01-31_03-15Z/
#
#   # Named run with timestamp (unique each submission)
#   TIMESTAMP=1 sbatch scripts/run_slurm.sh mnli_00 configs/...
#   -> outputs/mnli_00_2026-01-31_03-15Z/
#
# Environment variables:
#   SEED=123      - Custom random seed
#   MAX_STEPS=10  - Limit steps (for testing)
#   TIMESTAMP=1   - Append timestamp to run_name
#   FORCE=1       - Overwrite existing run
#   DEBUG=1       - Enable verbose DEBUG logging
#
# First-time setup:
#   ssh login.explorer.northeastern.edu
#   module load anaconda3/2024.06
#   conda env create -f environment.yml
#   conda activate distillation

set -e  # Exit on error

echo "========================================="
echo "SLURM Job: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU Device: $CUDA_VISIBLE_DEVICES"
echo "========================================="

# Load modules and activate environment
module load cuda/12.1

# Show GPU info after CUDA is loaded
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
echo "GPU: $GPU_NAME ($GPU_MEM)"
module load anaconda3/2024.06
eval "$(conda shell.bash hook)"
conda activate distillation

# Parse arguments: detect if $1 is config (ends with .yaml/.yml) or run_name
if [[ "$1" == *.yaml ]] || [[ "$1" == *.yml ]]; then
    # Auto-named run: sbatch script.sh config.yaml
    CONFIG="$1"
    RUN_NAME=""
elif [ -n "$1" ]; then
    # Named run: sbatch script.sh run_name [config.yaml]
    RUN_NAME="$1"
    CONFIG="$2"
else
    # No arguments: error
    echo "Error: No arguments provided."
    echo "Usage:"
    echo "  sbatch scripts/run_slurm.sh <run_name> <config>  # New named run"
    echo "  sbatch scripts/run_slurm.sh <run_name>           # Resume existing"
    echo "  sbatch scripts/run_slurm.sh <config>             # Auto-named run"
    exit 1
fi

# Build command arguments
ARGS=""

# Optional: seed
if [ -n "$SEED" ]; then
    ARGS="$ARGS --seed $SEED"
    echo "Custom seed: $SEED"
fi

# Optional: max_steps for testing
if [ -n "$MAX_STEPS" ]; then
    ARGS="$ARGS --max_steps $MAX_STEPS"
    echo "Testing mode: max_steps=$MAX_STEPS"
fi

# Optional: append timestamp to run_name
if [ -n "$TIMESTAMP" ]; then
    ARGS="$ARGS --timestamp"
    echo "Timestamp mode: appending timestamp to run_name"
fi

# Optional: force overwrite
if [ -n "$FORCE" ]; then
    ARGS="$ARGS --force"
    echo "Force mode: will overwrite existing run"
fi

# Optional: debug logging
if [ -n "$DEBUG" ]; then
    ARGS="$ARGS --debug"
    echo "Debug mode: verbose logging enabled"
fi

# Create logs directory
mkdir -p logs

# Check if training already complete (for named runs without TIMESTAMP or FORCE)
# Skip check if TIMESTAMP (actual name will be different) or FORCE (user wants overwrite)
if [ -n "$RUN_NAME" ] && [ -z "$TIMESTAMP" ] && [ -z "$FORCE" ]; then
    OUTPUT_DIR="./outputs/$RUN_NAME"
    STATE_FILE="$OUTPUT_DIR/training_state.json"

    if [ -f "$STATE_FILE" ]; then
        COMPLETE=$(python -c "
import json
try:
    with open('$STATE_FILE') as f:
        s = json.load(f)
    target = s.get('target_epochs') or s.get('target_steps') or float('inf')
    current = s.get('completed_epochs', 0)
    print('true' if current >= target else 'false')
except:
    print('false')
")
        if [ "$COMPLETE" = "true" ]; then
            echo "Training already complete for: $RUN_NAME"
            exit 0
        else
            echo "Resuming training from previous checkpoint..."
        fi
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

# Build and run the training command
# Use || true to capture exit code despite set -e
if [ -n "$RUN_NAME" ]; then
    # Named run provided as argument (user or auto-resubmit)
    OUTPUT_DIR="./outputs/$RUN_NAME"
    # Resume only if: dir exists AND no FORCE AND no TIMESTAMP
    # (TIMESTAMP creates a new dir, so always needs --config)
    if [ -d "$OUTPUT_DIR" ] && [ -z "$FORCE" ] && [ -z "$TIMESTAMP" ]; then
        # Resume existing run (no --config needed)
        CMD="python distill.py --run_name \"$RUN_NAME\" $ARGS"
        echo "CMD: $CMD"
        python distill.py --run_name "$RUN_NAME" $ARGS || EXIT_CODE=$?
    else
        # Create new run (--config required)
        if [ -z "$CONFIG" ]; then
            echo "Error: --config required for new run '$RUN_NAME'"
            echo "Usage: sbatch scripts/run_slurm.sh $RUN_NAME <config.yaml>"
            exit 1
        fi
        CMD="python distill.py --run_name \"$RUN_NAME\" --config \"$CONFIG\" $ARGS"
        echo "CMD: $CMD"
        python distill.py --run_name "$RUN_NAME" --config "$CONFIG" $ARGS || EXIT_CODE=$?
    fi
else
    # Auto-named run: always start fresh
    CMD="python distill.py --config \"$CONFIG\" $ARGS"
    echo "CMD: $CMD"
    python distill.py --config "$CONFIG" $ARGS || EXIT_CODE=$?
fi

EXIT_CODE=${EXIT_CODE:-0}

# Check if more training needed and resubmit if so
# Read job info from .incomplete_jobs using SLURM_JOB_ID
if [ $EXIT_CODE -eq 0 ] && [ -f ".incomplete_jobs" ]; then
    JOB_INFO=$(python -c "
import json
try:
    jobs = json.load(open('.incomplete_jobs'))
    info = jobs.get('$SLURM_JOB_ID', {})
    print(info.get('run_name', ''))
    print(info.get('output_dir', ''))
except:
    print('')
    print('')
")
    ACTUAL_RUN_NAME=$(echo "$JOB_INFO" | sed -n '1p')
    ACTUAL_OUTPUT_DIR=$(echo "$JOB_INFO" | sed -n '2p')

    if [ -n "$ACTUAL_RUN_NAME" ] && [ -n "$ACTUAL_OUTPUT_DIR" ]; then
        STATE_FILE="$ACTUAL_OUTPUT_DIR/training_state.json"

        if [ -f "$STATE_FILE" ]; then
            COMPLETE=$(python -c "
import json
try:
    with open('$STATE_FILE') as f:
        s = json.load(f)
    target = s.get('target_epochs') or s.get('target_steps') or float('inf')
    current = s.get('completed_epochs', 0)
    print('true' if current >= target else 'false')
except:
    print('false')
")

            if [ "$COMPLETE" = "false" ]; then
                echo ""
                echo "========================================="
                echo "Training incomplete. Resubmitting job..."
                echo "Run: $ACTUAL_RUN_NAME"
                echo "========================================="

                # Resubmit with PREV_JOB_ID for handoff (new job removes old entry)
                PREV_JOB_ID="$SLURM_JOB_ID" SEED="$SEED" MAX_STEPS="$MAX_STEPS" sbatch "$0" "$ACTUAL_RUN_NAME"
            else
                echo ""
                echo "========================================="
                echo "Training complete!"
                echo "SLURM Job: $SLURM_JOB_ID"
                echo "Node: $SLURM_NODELIST"
                echo "========================================="
            fi
        else
            echo ""
            echo "========================================="
            echo "Training complete! (no state file)"
            echo "SLURM Job: $SLURM_JOB_ID"
            echo "Node: $SLURM_NODELIST"
            echo "========================================="
        fi
    else
        echo ""
        echo "========================================="
        echo "Training complete! (no job entry found)"
        echo "SLURM Job: $SLURM_JOB_ID"
        echo "Node: $SLURM_NODELIST"
        echo "========================================="
    fi
else
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "========================================="
        echo "Training complete!"
        echo "SLURM Job: $SLURM_JOB_ID"
        echo "Node: $SLURM_NODELIST"
        echo "========================================="
    else
        echo ""
        echo "========================================="
        echo "Job failed with exit code: $EXIT_CODE"
        echo "SLURM Job: $SLURM_JOB_ID"
        echo "Node: $SLURM_NODELIST"
        echo "========================================="
    fi
fi
