#!/bin/bash
#SBATCH --job-name=forge-data
#SBATCH --partition=short
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/data_prep_%j.out
#SBATCH --error=logs/data_prep_%j.out

# Data preparation on SLURM (CPU-only)
#
# Usage:
#   sbatch scripts/slurm/run_data_prep.sh omnidocbench
#   sbatch scripts/slurm/run_data_prep.sh synthmix
#   sbatch scripts/slurm/run_data_prep.sh formatting --num_pages 200 --augment
#   sbatch scripts/slurm/run_data_prep.sh olmocr-mix --max_samples 5000
#   sbatch scripts/slurm/run_data_prep.sh all

source scripts/slurm/slurm_essentials.sh

slurm_setup_env
slurm_print_header

DATASET="$1"
shift 2>/dev/null || true

echo "Dataset: $DATASET"
echo "Args: $@"

python scripts/data/prepare.py --dataset "$DATASET" "$@"

slurm_print_footer
