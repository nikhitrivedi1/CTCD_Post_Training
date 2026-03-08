#!/bin/bash
#SBATCH --job-name=olmocr-score
#SBATCH --partition=short
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/score_%j.out

# Score olmOCR-Bench results (CPU-only, parallel)
#
# Usage:
#   sbatch scripts/slurm/run_olmocr_score.sh --output_dir outputs/bench/my-run

source scripts/slurm/slurm_essentials.sh

slurm_setup_env
slurm_print_header
echo "Args: $@"

python scripts/eval/olmocr_bench.py --score_only "$@"

slurm_print_footer
