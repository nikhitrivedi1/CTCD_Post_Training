#!/bin/bash
#SBATCH --job-name=distill
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --signal=SIGUSR1@120

# SLURM script for knowledge distillation on Explorer HPC
# Handles job limits by automatically resubmitting incomplete jobs
#
# Usage:
#   sbatch scripts/run_slurm.sh [config] [output_dir] [seed]
#   sbatch scripts/run_slurm.sh configs/bert_distilbert_mnli_a100.yaml ./outputs/hpc_run 42
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
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

# Load modules and activate environment
module load cuda/12.1
module load anaconda3/2024.06
source activate distillation

# Configuration
CONFIG=${1:-configs/bert_distilbert_mnli_a100.yaml}
OUTPUT_DIR=${2:-./outputs/slurm_run}
SEED=${3:-42}
STATE_FILE="$OUTPUT_DIR/training_state.json"

# Create logs directory
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# Set up W&B run ID for continuation across jobs
# If WANDB_RUN_ID is not set, generate a new one
if [ -z "$WANDB_RUN_ID" ]; then
    export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
    echo "Generated new W&B run ID: $WANDB_RUN_ID"
else
    echo "Resuming W&B run: $WANDB_RUN_ID"
fi

export WANDB_RESUME="allow"

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
echo "Output: $OUTPUT_DIR"
echo "Seed: $SEED"
echo "========================================="
echo ""

python train.py \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED"

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

        # Resubmit with same W&B run ID and seed
        sbatch --export=WANDB_RUN_ID=$WANDB_RUN_ID "$0" "$CONFIG" "$OUTPUT_DIR" "$SEED"
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
