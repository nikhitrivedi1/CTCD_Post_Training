#!/bin/bash
# Auto-submit jobs from local machine via SSH
#
# Setup:
#   chmod +x scripts/auto_submit_local.sh
#
# If resuming (some jobs already submitted):
#   grep -v '^#\|^$' configs_to_run.txt | head -N > .submitted_configs
#   (where N = number of configs already submitted)
#
# Run:
#   ./scripts/auto_submit_local.sh configs_to_run.txt
#
# Can Ctrl+C and resume anytime - tracks submitted configs in .submitted_configs

set -e

CLUSTER="explorer"
REMOTE_DIR="/scratch/ravi.ka/Distillation"
CONFIGS_FILE="${1:-configs_to_run.txt}"
MAX_OPEN=8
CHECK_INTERVAL=300  # 5 minutes

if [ ! -f "$CONFIGS_FILE" ]; then
    echo "Error: Config list file not found: $CONFIGS_FILE"
    exit 1
fi

SUBMITTED_FILE=".submitted_configs"
touch "$SUBMITTED_FILE"

get_open_jobs() {
    ssh "$CLUSTER" "squeue -u \$USER -h | wc -l"
}

get_next_config() {
    while read -r config; do
        [[ -z "$config" || "$config" =~ ^# ]] && continue
        grep -qF "$config" "$SUBMITTED_FILE" && continue
        echo "$config"
        return 0
    done < "$CONFIGS_FILE"
    return 1
}

submit_job() {
    local config="$1"
    echo "[$(date '+%H:%M:%S')] Submitting: $config"
    result=$(ssh "$CLUSTER" "cd $REMOTE_DIR && sbatch --gres=gpu:h200:1 scripts/run_slurm.sh $config")
    job_id=$(echo "$result" | grep -oE '[0-9]+$')
    echo "$result"
    echo "$job_id $config" >> "$SUBMITTED_FILE"
}

echo "================================"
echo "Auto-submit (local → cluster)"
echo "Cluster: $CLUSTER"
echo "Max open: $MAX_OPEN"
echo "Interval: ${CHECK_INTERVAL}s"
echo "================================"

while true; do
    open_jobs=$(get_open_jobs)
    slots=$((MAX_OPEN - open_jobs))

    echo "[$(date '+%H:%M:%S')] Jobs: $open_jobs | Slots: $slots"

    while [ $slots -gt 0 ]; do
        config=$(get_next_config) || break
        submit_job "$config"
        slots=$((slots - 1))
        sleep 1
    done

    # Check if done
    total=$(grep -cv '^#\|^$' "$CONFIGS_FILE" || echo 0)
    submitted=$(wc -l < "$SUBMITTED_FILE" | tr -d ' ')

    if [ "$submitted" -ge "$total" ] && [ "$open_jobs" -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] All done!"
        break
    fi

    sleep $CHECK_INTERVAL
done
