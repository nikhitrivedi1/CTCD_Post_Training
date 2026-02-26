#!/bin/bash
#SBATCH --job-name=ocr-data-prep
#SBATCH --partition=courses
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --output=logs/data_prep_%j.out

# Prepare olmOCR-mix-1025 dataset for training
#
# Usage:
#   sbatch scripts/run_prepare_data.sh                                              # full 268K (all subsets)
#   sbatch scripts/run_prepare_data.sh --subset 00_documents --output data/ocr/docs # single subset
#   sbatch scripts/run_prepare_data.sh --max_samples 5000 --output data/ocr/olmocr-mix-5k

module load anaconda3/2024.06
eval "$(conda shell.bash hook)"
conda activate distillation
export HF_HOME=/scratch/$USER/.cache/huggingface

mkdir -p logs

echo "=== OCR Data Preparation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Args: $@"
echo "Start: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "==========================="

python scripts/prepare_ocr_data.py "$@"

echo "==========================="
echo "End: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
