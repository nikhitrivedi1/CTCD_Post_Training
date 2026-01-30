#!/bin/bash
#SBATCH --job-name=mnli_bert_distilbert_h200_cotrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load modules
module load cuda/12.4

# Activate conda environment
source ~/.bashrc
conda activate distillation

# Create logs directory if it doesn't exist
mkdir -p logs

# Navigate to project directory
cd /scratch/ravi.ka/Distillation

# Run training (unbuffered output for real-time logging)
python -u train.py --config configs/mnli/bert_distilbert/h200/batch64_lr8e-5_cotrain.yaml
