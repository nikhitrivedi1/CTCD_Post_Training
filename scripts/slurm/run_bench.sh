#!/bin/bash
#SBATCH --job-name=forge-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/bench_%j.out
#SBATCH --signal=B:SIGUSR1@120

# Benchmark evaluation on SLURM (olmOCR-Bench or OmniDocBench)
#
# Usage:
#   # olmOCR-Bench
#   sbatch scripts/slurm/run_bench.sh olmocr --model allenai/olmOCR-2-7B-1025-FP8
#   sbatch scripts/slurm/run_bench.sh olmocr --model X --output_dir D --shard 1/4
#
#   # OmniDocBench
#   sbatch scripts/slurm/run_bench.sh omnidoc --checkpoint outputs/ablations_00/
#   sbatch scripts/slurm/run_bench.sh omnidoc --model X --output_dir D --shard 1/4
#
#   # Sharded (4 GPUs):
#   for i in 1 2 3 4; do
#     sbatch scripts/slurm/run_bench.sh omnidoc --checkpoint outputs/ablations_00/ \
#       --output_dir outputs/bench/omnidoc/ablations_00 --shard $i/4
#   done
#   sbatch scripts/slurm/run_bench.sh omnidoc --score_only \
#     --output_dir outputs/bench/omnidoc/ablations_00
#
#   # Resume (auto-resubmit passes _BENCH_TYPE env var)
#   sbatch scripts/slurm/run_bench.sh omnidoc --output_dir D
#
# Auto-resubmit: on SIGUSR1 (time limit), the job registers in .incomplete_jobs
# and the shell auto-resubmits with --output_dir [--shard].

source scripts/slurm/slurm_essentials.sh

slurm_setup_env
slurm_print_header

# Determine benchmark type: first arg or _BENCH_TYPE env var (from resubmit)
if [ -n "$_BENCH_TYPE" ]; then
    BENCH="$_BENCH_TYPE"
    BENCH_ARGS="$@"
elif [ "$1" = "olmocr" ] || [ "$1" = "omnidoc" ]; then
    BENCH="$1"
    shift
    BENCH_ARGS="$@"
else
    echo "Error: first argument must be 'olmocr' or 'omnidoc'"
    echo "Usage: sbatch $0 {olmocr|omnidoc} [bench args...]"
    exit 1
fi

case "$BENCH" in
    olmocr)
        SCRIPT="scripts/eval/olmocr_bench.py"
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
        SCRIPT="scripts/eval/omnidoc_bench.py"
        ;;
    *)
        echo "Error: unknown benchmark '$BENCH'. Use 'olmocr' or 'omnidoc'."
        exit 1
        ;;
esac

echo "Benchmark: $BENCH"
echo "Script: $SCRIPT"
echo "Args: $BENCH_ARGS"

# Run Python in background so `wait` can be interrupted by SIGUSR1.
python $SCRIPT $BENCH_ARGS &
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
    if { [ "$_JOB_TYPE" = "bench" ] || [ "$_JOB_TYPE" = "omnidocbench" ]; } && [ -n "$_OUTPUT_DIR" ]; then
        echo ""
        echo "=== Auto-resubmitting ==="
        echo "Benchmark: $BENCH"
        echo "Output dir: $_OUTPUT_DIR"
        RESUBMIT_ARGS="--output_dir $_OUTPUT_DIR"
        if [ -n "$_SHARD" ]; then
            RESUBMIT_ARGS="$RESUBMIT_ARGS --shard $_SHARD"
            echo "Shard: $_SHARD"
        fi
        _BENCH_TYPE="$BENCH" PREV_JOB_ID="$SLURM_JOB_ID" sbatch "$0" $RESUBMIT_ARGS
        echo "========================="
    fi
fi

slurm_print_footer
