#!/bin/bash
#SBATCH --job-name=olmocr-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/bench_%j.out
#SBATCH --signal=B:SIGUSR1@120

# olmOCR-Bench evaluation on SLURM
#
# Usage:
#   sbatch scripts/slurm/run_olmocr_bench.sh --model allenai/olmOCR-2-7B-1025-FP8
#   sbatch scripts/slurm/run_olmocr_bench.sh --model X --output_dir D --shard 1/4
#   sbatch scripts/slurm/run_olmocr_bench.sh --output_dir D                          # resume
#   sbatch scripts/slurm/run_olmocr_bench.sh --score_only --output_dir D
#
# Auto-resubmit: on SIGUSR1 (time limit), the job registers in .incomplete_jobs
# and the shell auto-resubmits with just --output_dir [--shard].
#
# SIGUSR1 handling: Python's signal handlers can't interrupt model.generate()
# (C extension, blocks for ~25 min/page). Instead, the shell uses the
# SIGUSR1_RECEIVED flag from slurm_essentials.sh to kill Python and auto-resubmit.

source scripts/slurm/slurm_essentials.sh

slurm_setup_env
slurm_print_header
echo "Args: $@"

# Download benchmark data if not present (uses git-lfs, avoids HF rate limits)
BENCH_DIR="data/olmocr-bench"
if [ ! -d "$BENCH_DIR/bench_data" ]; then
    echo "Downloading olmOCR-Bench data..."
    rm -rf "$BENCH_DIR"
    git lfs install 2>/dev/null
    git clone https://huggingface.co/datasets/allenai/olmOCR-bench "$BENCH_DIR"
fi

# Run Python in background so `wait` can be interrupted by SIGUSR1
python scripts/eval/olmocr_bench.py "$@" &
PYTHON_PID=$!
wait $PYTHON_PID
EXIT_CODE=$?

# If SIGUSR1 was received, kill Python and treat as clean exit for resubmit
if [ "$SIGUSR1_RECEIVED" = "1" ]; then
    echo ""
    echo "SIGUSR1 received - stopping Python for auto-resubmit"
    kill $PYTHON_PID 2>/dev/null
    # Force kill after 60s if Python doesn't exit (model.generate() is a C
    # extension that blocks for minutes and can't respond to SIGTERM)
    ( sleep 60 && kill -9 $PYTHON_PID 2>/dev/null ) &
    TIMER_PID=$!
    wait $PYTHON_PID 2>/dev/null
    kill $TIMER_PID 2>/dev/null
    wait $TIMER_PID 2>/dev/null
    EXIT_CODE=0
fi

# Auto-resubmit if job was interrupted
if [ $EXIT_CODE -eq 0 ] && slurm_read_incomplete; then
    if [ "$_JOB_TYPE" = "bench" ] && [ -n "$_OUTPUT_DIR" ]; then
        echo ""
        echo "=== Auto-resubmitting ==="
        echo "Output dir: $_OUTPUT_DIR"
        RESUBMIT_ARGS="--output_dir $_OUTPUT_DIR"
        if [ -n "$_SHARD" ]; then
            RESUBMIT_ARGS="$RESUBMIT_ARGS --shard $_SHARD"
            echo "Shard: $_SHARD"
        fi
        PREV_JOB_ID="$SLURM_JOB_ID" sbatch "$0" $RESUBMIT_ARGS
        echo "========================="
    fi
fi

slurm_print_footer
