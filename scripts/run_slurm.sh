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
# Handles job limits by automatically resubmitting incomplete jobs
#
# Usage:
#   sbatch scripts/run_slurm.sh <config> [run_name]
#
# Examples:
#   sbatch scripts/run_slurm.sh configs/mnli/bert_distilbert/ablations/02_frozen_sft-teacher_kl.yaml
#   -> auto-generates: outputs/02_frozen_sft-teacher_kl_2026-01-31_03-15Z
#
#   sbatch scripts/run_slurm.sh configs/... my_experiment
#   -> outputs/my_experiment (resumes if exists)
#
# Environment variables:
#   SEED=123 sbatch scripts/run_slurm.sh configs/... test_run_seed_123
#   MAX_STEPS=10 sbatch scripts/run_slurm.sh configs/... test_run
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

# Configuration: <config> [run_name]
CONFIG=${1:-configs/mnli/bert_distilbert/gpu/a100/batch256_lr3p2e-4.yaml}

# Generate run name from config filename + timestamp if not provided
# e.g., configs/mnli/bert_distilbert/ablations/02_frozen_sft-teacher_kl.yaml
#    -> 02_frozen_sft-teacher_kl_2026-01-31_03-15Z
# Full config path is stored in run_metadata for searchability
if [ -z "$2" ]; then
    TIMESTAMP=$(date -u +"%Y-%m-%d_%H-%MZ")
    CONFIG_NAME=$(basename "$CONFIG" .yaml)
    RUN_NAME="${CONFIG_NAME}_${TIMESTAMP}"
else
    RUN_NAME="$2"
fi

OUTPUT_DIR="./outputs/$RUN_NAME"
STATE_FILE="$OUTPUT_DIR/training_state.json"

# Optional: seed (via environment variable)
SEED_ARG=""
if [ -n "$SEED" ]; then
    SEED_ARG="--seed $SEED"
    echo "Custom seed: $SEED"
fi

# Optional: max_steps for testing (via environment variable)
MAX_STEPS_ARG=""
if [ -n "$MAX_STEPS" ]; then
    MAX_STEPS_ARG="--max_steps $MAX_STEPS"
    echo "Testing mode: max_steps=$MAX_STEPS"
fi

# Create logs directory
mkdir -p logs

# Check if training already complete
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
        echo "Training already complete. Exiting."
        exit 0
    else
        echo "Resuming training from previous checkpoint..."
    fi
fi

# Run training
echo ""
echo "========================================="
echo "Starting training..."
echo "Config: $CONFIG"
echo "Run name: $RUN_NAME"
echo "Output: $OUTPUT_DIR"
echo "========================================="
echo ""

# Use --run_name for new runs (--new), resume otherwise
if [ -d "$OUTPUT_DIR" ]; then
    python train.py --run_name "$RUN_NAME" $SEED_ARG $MAX_STEPS_ARG
else
    python train.py --config "$CONFIG" --run_name "$RUN_NAME" --new $SEED_ARG $MAX_STEPS_ARG
fi

EXIT_CODE=$?

# Check if more training needed and resubmit if so
if [ $EXIT_CODE -eq 0 ] && [ -f "$STATE_FILE" ]; then
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
        echo "========================================="

        # Resubmit with same run_name (W&B ID managed via metadata in train.py)
        sbatch "$0" "$CONFIG" "$RUN_NAME"
    else
        echo ""
        echo "========================================="
        echo "Training complete!"
        echo "========================================="
    fi
else
    echo ""
    echo "========================================="
    echo "Job failed with exit code: $EXIT_CODE"
    echo "========================================="
fi
