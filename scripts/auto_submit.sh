#!/bin/bash
# Auto-submit jobs when slots available
# Usage: nohup ./scripts/auto_submit.sh configs_list.txt &

set -e

CONFIGS_FILE="${1:-configs_to_run.txt}"
MAX_OPEN=8
CHECK_INTERVAL=300  # 5 minutes

if [ ! -f "$CONFIGS_FILE" ]; then
    echo "Error: Config list file not found: $CONFIGS_FILE"
    echo "Create a file with one config path per line"
    exit 1
fi

# Track submitted configs
SUBMITTED_FILE=".submitted_configs"
touch "$SUBMITTED_FILE"

get_open_jobs() {
    squeue -u $USER -h | wc -l
}

get_next_config() {
    while read -r config; do
        # Skip empty lines and comments
        [[ -z "$config" || "$config" =~ ^# ]] && continue
        # Skip already submitted
        grep -qxF "$config" "$SUBMITTED_FILE" && continue
        echo "$config"
        return 0
    done < "$CONFIGS_FILE"
    return 1
}

submit_job() {
    local config="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Submitting: $config"
    TIMESTAMP=1 sbatch --gres=gpu:h200:1 scripts/run_slurm.sh "$config"
    echo "$config" >> "$SUBMITTED_FILE"
}

echo "========================================="
echo "Auto-submit started"
echo "Config file: $CONFIGS_FILE"
echo "Max open jobs: $MAX_OPEN"
echo "Check interval: ${CHECK_INTERVAL}s"
echo "========================================="

while true; do
    open_jobs=$(get_open_jobs)
    slots=$((MAX_OPEN - open_jobs))

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Open jobs: $open_jobs, Available slots: $slots"

    # Submit jobs up to available slots
    submitted=0
    while [ $slots -gt 0 ]; do
        config=$(get_next_config) || break
        submit_job "$config"
        slots=$((slots - 1))
        submitted=$((submitted + 1))
    done

    # Check if all done
    remaining=$(grep -cvxFf "$SUBMITTED_FILE" "$CONFIGS_FILE" 2>/dev/null | grep -v '^#' || echo 0)
    if [ "$remaining" -eq 0 ] && [ "$open_jobs" -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] All jobs submitted and completed!"
        break
    fi

    if [ $submitted -eq 0 ] && [ $slots -eq 0 ]; then
        echo "Waiting for slots..."
    fi

    sleep $CHECK_INTERVAL
done
