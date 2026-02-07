#!/bin/bash
# Auto-submit jobs from local machine via SSH
#
# Run from auto-submit/:
#   ./submit-local.sh configs_to_run.txt
#   ./submit-local.sh configs_to_run.txt --force       # Overwrite existing runs
#   ./submit-local.sh configs_to_run.txt --gpu=a100    # Override GPU (default: h200)
#
# Can Ctrl+C and resume - tracks progress in .submitted_configs

set -e

CLUSTER="explorer"
REMOTE_DIR="/scratch/ravi.ka/Distillation"
MAX_OPEN=8
CHECK_INTERVAL=300  # 5 minutes

# Parse flags
FORCE_FLAG=""
GPU_FLAG=""
CONFIGS_FILE=""
for arg in "$@"; do
    case "$arg" in
        --force) FORCE_FLAG="FORCE=1" ;;
        --gpu=*) GPU_FLAG="${arg#--gpu=}" ;;
        *) CONFIGS_FILE="$arg" ;;
    esac
done
CONFIGS_FILE="${CONFIGS_FILE:-configs_to_run.txt}"

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

generate_run_name() {
    # Generate run_name from config path: {task}_{experiment}_{config}_baseline
    # Path: configs/{type}/{task}/{model_pair}/{experiment}/{config}.yaml
    local config="$1"
    IFS="/" read -ra parts <<< "$config"
    local config_name=$(basename "$config" .yaml)
    local task="${parts[2]}"       # mnli, sst2, cola, qqp, europarl
    local experiment="${parts[4]}" # ablations, far, staged

    echo "${task}_${experiment}_${config_name}_baseline"
}

submit_job() {
    local config="$1"
    local run_name=$(generate_run_name "$config")
    echo "[$(date '+%H:%M:%S')] Submitting: $run_name"
    local cmd="cd $REMOTE_DIR && ${FORCE_FLAG:+$FORCE_FLAG }sbatch ${GPU_FLAG:+--gres=gpu:$GPU_FLAG:1 }scripts/run_slurm.sh $run_name $config"
    echo "  cmd: ssh $CLUSTER \"$cmd\""
    result=$(ssh "$CLUSTER" "$cmd")
    job_id=$(echo "$result" | grep -oE '[0-9]+$')
    echo "$result"
    echo "$job_id $config $run_name" >> "$SUBMITTED_FILE"
}

echo "================================"
echo "Auto-submit (local → cluster)"
echo "Cluster: $CLUSTER"
echo "Max open: $MAX_OPEN"
echo "Interval: ${CHECK_INTERVAL}s"
[ -n "$FORCE_FLAG" ] && echo "Force:    YES (overwrite existing runs)"
[ -n "$GPU_FLAG" ] && echo "GPU:      $GPU_FLAG (override)"
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
