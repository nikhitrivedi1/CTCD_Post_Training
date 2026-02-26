#!/bin/bash
#SBATCH --job-name=olmocr-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/bench_%j.out
#SBATCH --signal=SIGTERM@300

# olmOCR-Bench evaluation on SLURM
#
# Usage:
#   sbatch scripts/run_olmocr_bench.sh --model allenai/olmOCR-2-7B-1025-FP8
#   sbatch scripts/run_olmocr_bench.sh --model Qwen/Qwen3-VL-2B-Instruct
#   sbatch scripts/run_olmocr_bench.sh --checkpoint outputs/.../checkpoint-400

module load cuda/12.1
module load anaconda3/2024.06
eval "$(conda shell.bash hook)"
conda activate distillation
export HF_HOME=/scratch/$USER/.cache/huggingface

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

python scripts/eval_olmocr_bench.py "$@"

# Auto-resubmit if generation was interrupted (SIGTERM from time limit)
RESUME_FILE=".bench_resume_$SLURM_JOB_ID"
if [ -f "$RESUME_FILE" ]; then
    OUTPUT_DIR=$(cat "$RESUME_FILE")
    rm -f "$RESUME_FILE"

    # Rebuild args: replace --resume or add it, pointing to the output dir
    RESUBMIT_ARGS=""
    SKIP_NEXT=false
    for arg in "$@"; do
        if $SKIP_NEXT; then
            SKIP_NEXT=false
            continue
        fi
        if [ "$arg" = "--resume" ]; then
            SKIP_NEXT=true
            continue
        fi
        RESUBMIT_ARGS="$RESUBMIT_ARGS $arg"
    done
    RESUBMIT_ARGS="$RESUBMIT_ARGS --resume $OUTPUT_DIR"

    echo ""
    echo "=== Auto-resubmitting ==="
    echo "Output dir: $OUTPUT_DIR"
    echo "Args:$RESUBMIT_ARGS"
    sbatch "$0" $RESUBMIT_ARGS
    echo "========================="
fi

echo "=============================="
echo "End: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
