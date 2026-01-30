#!/bin/bash
#SBATCH --job-name=mnli_bert_distilbert_p100_cotrain
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load modules
module load cuda/12.1

# Activate conda environment
source ~/.bashrc
conda activate distillation

# Create logs directory if it doesn't exist
mkdir -p logs

# Navigate to project directory
cd /scratch/ravi.ka/Distillation

# Run training (unbuffered output for real-time logging)
python -u train.py --config configs/mnli/bert_distilbert/p100_batch16_lr2e-5_cotrain.yaml
