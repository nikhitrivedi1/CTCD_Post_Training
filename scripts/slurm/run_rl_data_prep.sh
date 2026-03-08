#!/bin/bash
#SBATCH --job-name=rl-data-prep
#SBATCH --partition=short
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/rl_data_prep_%j.out

# Download and prepare synthmix + OmniDocBench for GRPO training
#
# Usage:
#   sbatch scripts/slurm/run_rl_data_prep.sh              # both datasets
#   sbatch scripts/slurm/run_rl_data_prep.sh synthmix     # synthmix only
#   sbatch scripts/slurm/run_rl_data_prep.sh omnidocbench # OmniDocBench only

source scripts/slurm/slurm_essentials.sh

slurm_setup_env
slurm_print_header

DATASET="${1:-all}"
echo "Dataset: $DATASET"

bash scripts/data/download_rl_datasets.sh "$DATASET"

slurm_print_footer
