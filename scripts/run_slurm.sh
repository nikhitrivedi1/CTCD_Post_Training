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
#   sbatch scripts/run_slurm.sh [config] [run_name] [seed]
#   sbatch scripts/run_slurm.sh configs/mnli/bert_distilbert/a100/batch256_lr3p2e-4.yaml my_experiment 42
#
# The run_name becomes the output directory (./outputs/run_name) and is used for W&B tracking.
# If run_name exists, training resumes from checkpoint. W&B run ID is managed via metadata.
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
CONFIG=${1:-configs/mnli/bert_distilbert/gpu/a100/batch256_lr3p2e-4.yaml}
RUN_NAME=${2:-slurm_run}
SEED=${3:-42}
OUTPUT_DIR="./outputs/$RUN_NAME"
STATE_FILE="$OUTPUT_DIR/training_state.json"

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
echo "Seed: $SEED"
echo "========================================="
echo ""

# Use --output_dir for compatibility (train.py manages W&B ID via metadata)
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

        # Resubmit with same run_name (W&B ID managed via metadata in train.py)
        sbatch "$0" "$CONFIG" "$RUN_NAME" "$SEED"
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
