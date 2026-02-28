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
#   sbatch scripts/run_olmocr_bench.sh --model allenai/olmOCR-2-7B-1025-FP8
#   sbatch scripts/run_olmocr_bench.sh --model X --output_dir D --shard 1/4
#   sbatch scripts/run_olmocr_bench.sh --output_dir D                          # resume
#   sbatch scripts/run_olmocr_bench.sh --score_only --output_dir D
#
# Auto-resubmit: on SIGUSR1 (time limit), the job registers in .incomplete_jobs
# and the shell auto-resubmits with just --output_dir [--shard].

# SIGUSR1 handling: Python's signal handlers can't interrupt model.generate()
# (C extension, blocks for ~25 min/page). Instead, the shell handles SIGUSR1:
# run Python in background, trap the signal, kill Python, then auto-resubmit.
SIGUSR1_RECEIVED=0
trap 'SIGUSR1_RECEIVED=1' SIGUSR1

module load cuda/12.1
module load anaconda3/2024.06
eval "$(conda shell.bash hook)"
source scripts/env/activate.sh

mkdir -p logs

# Download benchmark data if not present (uses git-lfs, avoids HF rate limits)
BENCH_DIR="data/olmocr-bench"
if [ ! -d "$BENCH_DIR/bench_data" ]; then
    echo "Downloading olmOCR-Bench data..."
    rm -rf "$BENCH_DIR"
    git lfs install 2>/dev/null
    git clone https://huggingface.co/datasets/allenai/olmOCR-bench "$BENCH_DIR"
fi

echo "=== olmOCR-Bench Evaluation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Args: $@"
echo "Start: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=============================="

# Run Python in background so `wait` can be interrupted by SIGUSR1
python scripts/eval_olmocr_bench.py "$@" &
PYTHON_PID=$!
wait $PYTHON_PID
EXIT_CODE=$?

# If SIGUSR1 was received, kill Python and treat as clean exit for resubmit
if [ "$SIGUSR1_RECEIVED" = "1" ]; then
    echo ""
    echo "SIGUSR1 received — stopping Python for auto-resubmit"
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

# Auto-resubmit if job was interrupted (entry still in .incomplete_jobs registry)
if [ $EXIT_CODE -eq 0 ] && [ -f ".incomplete_jobs" ]; then
    JOB_INFO=$(python3 -c "
import json
try:
    jobs = json.load(open('.incomplete_jobs'))
    info = jobs.get('$SLURM_JOB_ID', {})
    if info.get('type') == 'bench':
        print(info.get('output_dir', ''))
        print(info.get('shard', ''))
    else:
        print('')
        print('')
except:
    print('')
    print('')
")
    OUTPUT_DIR=$(echo "$JOB_INFO" | sed -n '1p')
    SHARD=$(echo "$JOB_INFO" | sed -n '2p')

    if [ -n "$OUTPUT_DIR" ]; then
        echo ""
        echo "=== Auto-resubmitting ==="
        echo "Output dir: $OUTPUT_DIR"
        RESUBMIT_ARGS="--output_dir $OUTPUT_DIR"
        if [ -n "$SHARD" ]; then
            RESUBMIT_ARGS="$RESUBMIT_ARGS --shard $SHARD"
            echo "Shard: $SHARD"
        fi
        PREV_JOB_ID="$SLURM_JOB_ID" sbatch "$0" $RESUBMIT_ARGS
        echo "========================="
    fi
fi

echo "=============================="
echo "End: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
