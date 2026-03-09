#!/bin/bash
#SBATCH --job-name=forge-data
#SBATCH --partition=short
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/data_prep_%j.out

# Data preparation on SLURM (CPU-only)
#
# Usage:
#   sbatch scripts/slurm/run_data_prep.sh olmocr-mix                              # prepare olmOCR-mix dataset
#   sbatch scripts/slurm/run_data_prep.sh olmocr-mix --max_samples 5000           # with extra args
#   sbatch scripts/slurm/run_data_prep.sh synthmix                                # download + convert synthmix
#   sbatch scripts/slurm/run_data_prep.sh omnidocbench                            # download + convert OmniDocBench
#   sbatch scripts/slurm/run_data_prep.sh rl                                      # both RL datasets

source scripts/slurm/slurm_essentials.sh

slurm_setup_env
slurm_print_header

DATASET="$1"
shift 2>/dev/null || true

echo "Dataset: $DATASET"
echo "Args: $@"

case "$DATASET" in
    olmocr-mix)
        python scripts/data/prepare_ocr_data.py "$@"
        ;;
    synthmix)
        bash scripts/data/download_rl_datasets.sh synthmix
        ;;
    omnidocbench)
        bash scripts/data/download_rl_datasets.sh omnidocbench
        ;;
    rl)
        bash scripts/data/download_rl_datasets.sh all
        ;;
    *)
        echo "Error: unknown dataset '$DATASET'"
        echo "Usage: sbatch $0 {olmocr-mix|synthmix|omnidocbench|rl} [args...]"
        exit 1
        ;;
esac

slurm_print_footer
