#!/bin/bash
#SBATCH --job-name=forge-score
#SBATCH --partition=short
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/score_%j.out

# Score benchmark results (CPU-only, parallel)
#
# Usage:
#   sbatch scripts/slurm/run_score.sh olmocr --output_dir outputs/bench/olmocr/my-run
#   sbatch scripts/slurm/run_score.sh omnidoc --output_dir outputs/bench/omnidoc/my-run

source scripts/slurm/slurm_essentials.sh

slurm_setup_env
slurm_print_header

BENCH="$1"
shift

case "$BENCH" in
    olmocr)
        SCRIPT="scripts/eval/olmocr_bench.py"
        ;;
    omnidoc)
        SCRIPT="scripts/eval/omnidoc_bench.py"
        ;;
    *)
        echo "Error: first argument must be 'olmocr' or 'omnidoc'"
        echo "Usage: sbatch $0 {olmocr|omnidoc} --output_dir <path>"
        exit 1
        ;;
esac

echo "Benchmark: $BENCH"
echo "Args: $@"

python $SCRIPT --score_only "$@"

slurm_print_footer
