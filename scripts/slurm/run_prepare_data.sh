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
#   sbatch scripts/slurm/run_prepare_data.sh                                              # full 268K (all subsets)
#   sbatch scripts/slurm/run_prepare_data.sh --subset 00_documents --output data/ocr/docs # single subset
#   sbatch scripts/slurm/run_prepare_data.sh --max_samples 5000 --output data/ocr/olmocr-mix-5k

source scripts/slurm/slurm_essentials.sh

slurm_setup_env
slurm_print_header
echo "Args: $@"

python scripts/data/prepare_ocr_data.py "$@"

slurm_print_footer
