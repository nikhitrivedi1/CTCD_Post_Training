#!/bin/bash
#SBATCH --job-name=forge-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.out
#SBATCH --signal=B:SIGUSR1@120

# SLURM benchmark script (olmOCR-Bench, OmniDocBench, Formatting) with
# auto-resubmission. Signal handling lives in slurm_essentials.sh.
#
# Usage:
#   sbatch scripts/slurm/run_bench.sh {olmocr|omnidoc|formatting} [args...]
#
# Examples:
#   # HuggingFace model:
#   sbatch scripts/slurm/run_bench.sh olmocr --model allenai/olmOCR-2-7B-1025-FP8
#   sbatch scripts/slurm/run_bench.sh formatting --model Qwen/Qwen3-VL-2B-Instruct
#
#   # Local checkpoint:
#   sbatch scripts/slurm/run_bench.sh olmocr --checkpoint outputs/fmt_sft
#   sbatch scripts/slurm/run_bench.sh omnidoc --checkpoint outputs/fmt_sft
#
#   # Named output dir:
#   sbatch scripts/slurm/run_bench.sh omnidoc --checkpoint outputs/fmt_sft \
#     --output outputs/bench/omnidoc/fmt_sft
#
#   # Sharded (4 GPUs):
#   for i in 1 2 3 4; do
#     sbatch scripts/slurm/run_bench.sh omnidoc --checkpoint outputs/fmt_sft \
#       --output outputs/bench/omnidoc/fmt_sft --shard $i/4
#   done
#
#   # Resume an interrupted run:
#   sbatch scripts/slurm/run_bench.sh omnidoc --output outputs/bench/omnidoc/fmt_sft
#
# Internal (set by auto-resubmit, not for manual use):
#   _BENCH_TYPE=omnidoc         - Benchmark type for resubmit
#   PREV_JOB_ID=12345          - Previous job ID (cleans up .incomplete_jobs)

source scripts/slurm/slurm_essentials.sh

slurm_setup_env
slurm_print_header

# Determine benchmark type: first arg or _BENCH_TYPE env var (from resubmit)
if [ -n "$_BENCH_TYPE" ]; then
    BENCH="$_BENCH_TYPE"
    BENCH_ARGS="$@"
elif [ "$1" = "olmocr" ] || [ "$1" = "omnidoc" ] || [ "$1" = "formatting" ]; then
    BENCH="$1"
    shift
    BENCH_ARGS="$@"
else
    echo "Error: first argument must be 'olmocr', 'omnidoc', or 'formatting'"
    echo "Usage: sbatch $0 {olmocr|omnidoc|formatting} [bench args...]"
    exit 1
fi

case "$BENCH" in
    olmocr)
        SCRIPT="scripts/benchmarks/olmocr_bench.py"
        # Download benchmark data if not present
        BENCH_DIR="data/olmocr-bench"
        if [ ! -d "$BENCH_DIR/bench_data" ]; then
            echo "Downloading olmOCR-Bench data..."
            rm -rf "$BENCH_DIR"
            git lfs install 2>/dev/null
            git clone https://huggingface.co/datasets/allenai/olmOCR-bench "$BENCH_DIR"
        fi
        ;;
    omnidoc)
        SCRIPT="scripts/benchmarks/omnidoc_bench.py"
        ;;
    formatting)
        SCRIPT="scripts/benchmarks/formatting_bench.py"
        ;;
    *)
        echo "Error: unknown benchmark '$BENCH'. Use 'olmocr', 'omnidoc', or 'formatting'."
        exit 1
        ;;
esac

echo "Benchmark: $BENCH"
echo "Script: $SCRIPT"

# Build args from env vars (mirrors run_training.sh pattern, used by run_chain.sh)
[ -n "$CHECKPOINT" ] && BENCH_ARGS="$BENCH_ARGS --checkpoint $CHECKPOINT" && echo "Checkpoint: $CHECKPOINT"
[ -n "$MODEL" ]      && BENCH_ARGS="$BENCH_ARGS --model $MODEL"           && echo "Model: $MODEL"
[ -n "$OUTPUT" ]     && BENCH_ARGS="$BENCH_ARGS --output $OUTPUT"         && echo "Output: $OUTPUT"
[ -n "$SHARD" ]      && BENCH_ARGS="$BENCH_ARGS --shard $SHARD"           && echo "Shard: $SHARD"
[ -n "$DEBUG" ]      && BENCH_ARGS="$BENCH_ARGS --debug"                  && echo "Debug mode: verbose logging enabled"

echo "Args: $BENCH_ARGS"

# --force: remove existing output dir so the bench starts fresh
for arg in $BENCH_ARGS; do
    if [ "$prev_arg" = "--output" ]; then
        OUTPUT_DIR="$arg"
    fi
    prev_arg="$arg"
done
if echo "$BENCH_ARGS" | grep -q -- "--force"; then
    BENCH_ARGS=$(echo "$BENCH_ARGS" | sed 's/--force//')
    if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
        echo "Force: removing existing $OUTPUT_DIR"
        rm -rf "$OUTPUT_DIR"
    fi
fi

# Run Python in background so slurm_wait_child can handle SIGUSR1.
python $SCRIPT $BENCH_ARGS &
slurm_wait_child $!
EXIT_CODE=$?

# Auto-resubmit if job was interrupted
if [ $EXIT_CODE -eq 0 ] && slurm_read_incomplete; then
    if { [ "$_JOB_TYPE" = "omnidoc" ] || [ "$_JOB_TYPE" = "olmocr" ] || [ "$_JOB_TYPE" = "formatting" ]; } && [ -n "$_OUTPUT_DIR" ]; then
        echo ""
        echo "=== Auto-resubmitting ==="
        echo "Benchmark: $BENCH"
        echo "Output dir: $_OUTPUT_DIR"
        RESUBMIT_ARGS="--output $_OUTPUT_DIR"
        if [ -n "$_SHARD" ]; then
            RESUBMIT_ARGS="$RESUBMIT_ARGS --shard $_SHARD"
            echo "Shard: $_SHARD"
        fi
        _BENCH_TYPE="$BENCH" PREV_JOB_ID="$SLURM_JOB_ID" sbatch "$0" $RESUBMIT_ARGS
        echo "========================="
    fi
fi

slurm_print_footer
