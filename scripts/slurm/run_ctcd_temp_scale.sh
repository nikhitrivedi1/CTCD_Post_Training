#!/bin/bash
# CTCD symmetric distillation.temperature sweep on Part-1 best configs for ablations
# 01 and 02 only (frozen teacher). Merges temperature + checkpointing.no_weights on each
# configs/CTCD_Experiments/<task>/best/<task>_0{1,2}_best.yaml leaf.
#
# Grid: 12 best configs × 4 temperatures × |CTCD_SEEDS| (default 1 seed → 48 runs).
# Temperatures: 2, 3, 4, 5.
#
# Worker-loop pattern: 4 SLURM array ranks (0–3) iterate TIDs R, R+4, R+8, …
# On SIGUSR1: checkpoint, break, resubmit; .done markers skip completed TIDs.
#
# Usage (from repo root):
#   sbatch scripts/slurm/run_ctcd_temp_scale.sh
#
# Optional env:
#   CTCD_COMPLETION_DIR   default: completed_runs/ctcd_temp_scale
#   CTCD_SEEDS            space-separated seeds (default: 42). Set multiple for a larger grid.
#   SEED is set per run from CTCD_SEEDS; MAX_STEPS EPOCHS DEBUG FORCE TIMESTAMP
#                         forwarded by run_training.sh
#
# Requires bash 4+ (typical Slurm compute images).

#SBATCH --job-name=ctcd_ts
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3
#SBATCH --output=logs/ctcd_ts_%A_%a.out
#SBATCH --error=logs/ctcd_ts_%A_%a.out
#SBATCH --signal=B:SIGUSR1@120

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT" || exit 1

mkdir -p logs

source scripts/slurm/slurm_essentials.sh

slurm_setup_env
slurm_print_header

CONFIG_ROOT="configs/CTCD_Experiments"
TASKS=(cola mrpc rte qqp)

if [[ -n "${CTCD_SEEDS:-}" ]]; then
    read -r -a SEEDS <<<"$CTCD_SEEDS"
else
    SEEDS=(42)
fi
NUM_SEEDS=${#SEEDS[@]}
if [[ $NUM_SEEDS -lt 1 ]]; then
    echo "CTCD_SEEDS must provide at least one integer seed"
    exit 1
fi

TEMPS=(2 3 4 5)
NUM_TEMPS=${#TEMPS[@]}

CONFIG_FILES=()
for TASK in "${TASKS[@]}"; do
    for suffix in 01 02; do
        f="${CONFIG_ROOT}/${TASK}/best/${TASK}_${suffix}_best.yaml"
        if [[ -f "$f" ]]; then
            CONFIG_FILES+=("$f")
        else
            echo "Warning: missing $f — skipping"
        fi
    done
done

NUM_CFG=${#CONFIG_FILES[@]}
if [[ $NUM_CFG -eq 0 ]]; then
    echo "No ablation 01/02 best configs found under $CONFIG_ROOT — exiting"
    exit 1
fi

PER_SLICE=$((NUM_TEMPS * NUM_SEEDS))
TOTAL=$((NUM_CFG * PER_SLICE))

WORKERS=4
RANK=${SLURM_ARRAY_TASK_ID:-0}

COMPLETION_DIR="${CTCD_COMPLETION_DIR:-completed_runs/ctcd_temp_scale}"
mkdir -p "$COMPLETION_DIR"

echo "========================================="
echo "CTCD temperature sweep  rank=$RANK / $((WORKERS - 1))"
echo "Configs: $NUM_CFG  Temps: ${TEMPS[*]}  Seeds: ${SEEDS[*]}  Total runs: $TOTAL  Workers: $WORKERS"
echo "Completion markers: $COMPLETION_DIR/"
echo "========================================="

decode_tid() {
    local tid=$1

    local cfg_idx=$((tid / PER_SLICE))
    local rem=$((tid % PER_SLICE))
    local temp_idx=$((rem / NUM_SEEDS))
    local seed_idx=$((rem % NUM_SEEDS))

    BASE_CFG="${CONFIG_FILES[$cfg_idx]}"
    TEMP_VAL="${TEMPS[$temp_idx]}"
    SEED_VAL="${SEEDS[$seed_idx]}"

    local stem
    stem=$(basename "$BASE_CFG" .yaml)
    if [[ $NUM_SEEDS -eq 1 ]]; then
        RUN_NAME="ts_${stem}_T${TEMP_VAL}"
    else
        RUN_NAME="ts_${stem}_T${TEMP_VAL}_seed${SEED_VAL}"
    fi

    DONE="${COMPLETION_DIR}/${RUN_NAME}.done"
    OUTPUT_DIR="./outputs/${RUN_NAME}"
}

TASKS_COMPLETED=0
TASKS_FAILED=0

for ((TID = RANK; TID < TOTAL; TID += WORKERS)); do

    if [[ $SIGUSR1_RECEIVED -eq 1 ]]; then
        echo "Timeout signal received — stopping before TID $TID"
        break
    fi

    decode_tid "$TID"

    if [[ -f "$DONE" ]]; then
        echo "Skip (marker): $RUN_NAME"
        continue
    fi
    if [[ -d "$OUTPUT_DIR" ]] && [[ -f "$OUTPUT_DIR/training_state.json" ]]; then
        if [[ "$(slurm_is_complete "$OUTPUT_DIR")" == "true" ]]; then
            echo "Skip (output complete, heal marker): $RUN_NAME"
            touch "$DONE"
            continue
        fi
    fi

    if [[ ! -f "$BASE_CFG" ]]; then
        echo "Missing config: $BASE_CFG — skipping TID $TID"
        continue
    fi

    TMP_OVR=$(mktemp "${TMPDIR:-/tmp}/ctcd_ts_ovr_${SLURM_JOB_ID:-local}_${TID}.XXXXXX.yaml")
    TMP_MERGED=$(mktemp "${TMPDIR:-/tmp}/ctcd_ts_merged_${SLURM_JOB_ID:-local}_${TID}.XXXXXX.yaml")
    trap 'rm -f "$TMP_OVR" "$TMP_MERGED"' EXIT

    cat >"$TMP_OVR" <<EOF
distillation:
  temperature: ${TEMP_VAL}
checkpointing:
  no_weights: true
EOF

    python scripts/slurm/merge_yaml_overrides.py "$BASE_CFG" "$TMP_OVR" "$TMP_MERGED"

    echo "========================================="
    echo "CTCD temp sweep  rank=$RANK  tid=$TID  run=$RUN_NAME"
    echo "Base: $BASE_CFG"
    echo "distillation.temperature=$TEMP_VAL  seed=$SEED_VAL"
    echo "Merged: $TMP_MERGED"
    echo "========================================="

    if [[ -d "$OUTPUT_DIR" ]]; then
        SEED="$SEED_VAL" _FORGE_SWEEP_WORKER=1 _FORGE_RESUME_PATH="$RUN_NAME" bash scripts/slurm/run_training.sh "$RUN_NAME" "$TMP_MERGED" &
    else
        SEED="$SEED_VAL" _FORGE_SWEEP_WORKER=1 bash scripts/slurm/run_training.sh "$RUN_NAME" "$TMP_MERGED" &
    fi
    slurm_wait_child $!
    TRAIN_EXIT=$EXIT_CODE

    rm -f "$TMP_OVR" "$TMP_MERGED"
    trap - EXIT

    if [[ $SIGUSR1_RECEIVED -eq 1 ]]; then
        echo "Run $RUN_NAME interrupted — checkpoint saved"
        break
    fi

    if [[ $TRAIN_EXIT -eq 0 ]]; then
        if [[ -f "$OUTPUT_DIR/training_state.json" ]] &&
            [[ "$(slurm_is_complete "$OUTPUT_DIR")" == "true" ]]; then
            touch "$DONE"
            echo "Marked complete: $DONE"
            TASKS_COMPLETED=$((TASKS_COMPLETED + 1))
        else
            echo "Warning: $RUN_NAME exited 0 but not pipeline-complete — skipping mark"
            TASKS_FAILED=$((TASKS_FAILED + 1))
        fi
    else
        echo "Training failed for $RUN_NAME (exit $TRAIN_EXIT) — continuing to next run"
        TASKS_FAILED=$((TASKS_FAILED + 1))
    fi

done

echo "========================================="
echo "Rank $RANK summary: completed=$TASKS_COMPLETED  failed/skipped=$TASKS_FAILED"
echo "========================================="

if [[ $SIGUSR1_RECEIVED -eq 1 ]]; then
    REMAINING=0
    for ((TID = RANK; TID < TOTAL; TID += WORKERS)); do
        decode_tid "$TID"
        [[ -f "$DONE" ]] || REMAINING=$((REMAINING + 1))
    done

    if [[ $REMAINING -gt 0 ]]; then
        SWEEP_SCRIPT="${SLURM_SUBMIT_DIR}/scripts/slurm/run_ctcd_temp_scale.sh"
        NEW_JOB=$(sbatch --parsable "$SWEEP_SCRIPT")
        echo "Resubmitted sweep as job $NEW_JOB ($REMAINING runs remaining for rank $RANK)"
    else
        echo "Rank $RANK has no remaining runs — no resubmit needed"
    fi
fi

slurm_print_footer
