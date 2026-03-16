#!/bin/bash
#SBATCH --job-name=forge-score
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/score_%j.out
#SBATCH --error=logs/score_%j.out

# SLURM scoring script (CPU-only). Re-scores existing benchmark results
# without needing a GPU.
#
# Usage:
#   sbatch scripts/slurm/run_score.sh {olmocr|omnidoc|formatting} --output <path>
#
# Examples:
#   sbatch scripts/slurm/run_score.sh olmocr --output outputs/bench/olmocr/fmt_sft
#   sbatch scripts/slurm/run_score.sh omnidoc --output outputs/bench/omnidoc/fmt_sft
#   sbatch scripts/slurm/run_score.sh formatting --output outputs/bench/formatting/fmt_sft

source scripts/slurm/slurm_essentials.sh

slurm_setup_env
slurm_print_header

BENCH="$1"
shift

case "$BENCH" in
    olmocr)
        SCRIPT="scripts/benchmarks/olmocr_bench.py"
        ;;
    omnidoc)
        SCRIPT="scripts/benchmarks/omnidoc_bench.py"
        ;;
    formatting)
        SCRIPT="scripts/benchmarks/formatting_bench.py"
        ;;
    *)
        echo "Error: first argument must be 'olmocr', 'omnidoc', or 'formatting'"
        echo "Usage: sbatch $0 {olmocr|omnidoc|formatting} --output_dir <path>"
        exit 1
        ;;
esac

echo "Benchmark: $BENCH"

# Build args from env vars (used by run_chain.sh)
SCORE_ARGS="$@"
[ -n "$OUTPUT" ] && SCORE_ARGS="$SCORE_ARGS --output $OUTPUT" && echo "Output: $OUTPUT"
[ -n "$DEBUG" ]  && SCORE_ARGS="$SCORE_ARGS --debug"          && echo "Debug mode: verbose logging enabled"

echo "Args: $SCORE_ARGS"

python $SCRIPT --score_only $SCORE_ARGS

slurm_print_footer
