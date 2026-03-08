#!/bin/bash
# Re-score all bench models in parallel.
#
# Usage:
#   bash scripts/eval/rescore_all.sh [max_jobs]
#
# Launches one --score_only process per model directory under outputs/bench/.
# Default max_jobs=4 to avoid overloading the node.

set -euo pipefail

BENCH_DIR="outputs/bench"
MAX_JOBS="${1:-4}"
LOG_DIR="$BENCH_DIR/.rescore_logs"
mkdir -p "$LOG_DIR"

if [ ! -d "$BENCH_DIR" ]; then
    echo "ERROR: $BENCH_DIR not found" >&2
    exit 1
fi

# Collect model directories (those containing a markdowns/ subdirectory)
models=()
for d in "$BENCH_DIR"/*/; do
    if [ -d "$d/markdowns" ]; then
        models+=("$d")
    fi
done

if [ ${#models[@]} -eq 0 ]; then
    echo "No model directories with markdowns/ found in $BENCH_DIR" >&2
    exit 1
fi

echo "Re-scoring ${#models[@]} models (max $MAX_JOBS parallel):"
printf "  %s\n" "${models[@]}"
echo ""

# Launch jobs with a simple semaphore
running=0
pids=()
names=()

for model_dir in "${models[@]}"; do
    model_name=$(basename "$model_dir")
    log_file="$LOG_DIR/$model_name.log"

    echo "Starting: $model_name -> $log_file"
    python scripts/eval/olmocr_bench.py --score_only --output_dir "$model_dir" \
        > "$log_file" 2>&1 &
    pids+=($!)
    names+=("$model_name")
    running=$((running + 1))

    # Wait for a slot if at max capacity
    if [ "$running" -ge "$MAX_JOBS" ]; then
        wait -n 2>/dev/null || true
        running=$((running - 1))
    fi
done

# Wait for remaining jobs
echo ""
echo "Waiting for remaining jobs..."
failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "  DONE: ${names[$i]}"
    else
        echo "  FAIL: ${names[$i]} (see $LOG_DIR/${names[$i]}.log)"
        failed=$((failed + 1))
    fi
done

echo ""
echo "Complete: $((${#models[@]} - failed))/${#models[@]} succeeded"
if [ "$failed" -gt 0 ]; then
    echo "Check logs in $LOG_DIR/ for failures"
    exit 1
fi
