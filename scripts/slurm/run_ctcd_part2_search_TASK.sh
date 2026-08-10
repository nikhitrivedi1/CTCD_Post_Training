#!/bin/bash
# TASK clone of run_ctcd_part2_search.sh — scoped to ablation 05 only (full CTCD:
# sweep all α × β preset combinations), with run names suffixed so outputs never
# collide with the original part2 search's runs.
#
# Base configs: configs/CTCD_Experiments/<task>/best/<task>_05_best.yaml
#
# Coefficient presets (hard, soft): (1,1), (1,2), (1,4), (4,4) — swept for both α and β.
#
# Tasks (edit TASKS):
#   clinc150: 05 config exists
#   GLUE later: append tasks; use 05 when best YAML exists
#
# Grid search only — no seed dimension. Grid per task: 4α × 4β = 16 runs.
#
# Jobs are built as an explicit list, then workers loop TIDs.
# On timeout (SIGUSR1): checkpoint, break, sweep resubmits; .done markers skip finished TIDs.
#
# Usage (from repo root):
#   sbatch scripts/slurm/run_ctcd_part2_search_TASK.sh
#
# Optional env:
#   CTCD_TASKS            comma-separated tasks, e.g. mnli,qqp (default: clinc150)
#   CTCD_RUN_SUFFIX       appended to every run name (default: _TASK)
#   CTCD_COMPLETION_DIR   default: completed_runs/ctcd_part2_search_task
#   MAX_STEPS EPOCHS DEBUG FORCE TIMESTAMP  forwarded by run_training.sh
#
# --- Smoke tests (repo root, conda env on) ---
# 1) File exists:
#    test -f "configs/CTCD_Experiments/clinc150/best/clinc150_05_best.yaml" || echo MISSING
# 2) One rank: MAX_STEPS=20 DEBUG=1 CTCD_COMPLETION_DIR=completed_runs/ctcd_p2s_task_smoke sbatch --array=0 "$0"

#SBATCH --job-name=ctcd_p2s_task
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3
#SBATCH --output=logs/ctcd_p2s_task_%A_%a.out
#SBATCH --error=logs/ctcd_p2s_task_%A_%a.out
#SBATCH --signal=B:SIGUSR1@120

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT" || exit 1

mkdir -p logs

source scripts/slurm/slurm_essentials.sh

slurm_setup_env
slurm_print_header

CONFIG_ROOT="configs/CTCD_Experiments"

if [[ -n "${CTCD_TASKS:-}" ]]; then
    IFS=',' read -ra TASKS <<<"${CTCD_TASKS}"
else
    TASKS=(clinc150)
fi

ABL="05"

# (hard, soft) presets for α and β — a05 sweeps the full α × β matrix
PRESET_H=(1 1 1 4)
PRESET_S=(1 2 4 4)
NUM_PRESETS=${#PRESET_H[@]}

RUN_SUFFIX="${CTCD_RUN_SUFFIX:-_TASK}"

# Build explicit job list: each entry is "task|ah|as|bh|bs"
JOBS=()
for TASK in "${TASKS[@]}"; do
    for ((ai = 0; ai < NUM_PRESETS; ai++)); do
        AH="${PRESET_H[$ai]}"
        AS="${PRESET_S[$ai]}"
        for ((bi = 0; bi < NUM_PRESETS; bi++)); do
            BH="${PRESET_H[$bi]}"
            BS="${PRESET_S[$bi]}"
            JOBS+=("${TASK}|${AH}|${AS}|${BH}|${BS}")
        done
    done
done

TOTAL=${#JOBS[@]}
if [[ $TOTAL -lt 1 ]]; then
    echo "Error: job list is empty."
    exit 1
fi

WORKERS=4
RANK=${SLURM_ARRAY_TASK_ID:-0}

COMPLETION_DIR="${CTCD_COMPLETION_DIR:-completed_runs/ctcd_part2_search_task}"
mkdir -p "$COMPLETION_DIR"

echo "========================================="
echo "CTCD Part 2 search (TASK, ablation 05) worker rank=$RANK / $((WORKERS - 1))"
echo "Tasks: ${TASKS[*]}"
echo "Total jobs: $TOTAL  Workers: $WORKERS  Run suffix: $RUN_SUFFIX"
echo "This rank handles TIDs: $RANK, $((RANK + WORKERS)), $((RANK + WORKERS * 2)), ..."
echo "Completion markers: $COMPLETION_DIR/"
echo "========================================="

decode_tid() {
    local tid=$1
    local spec="${JOBS[$tid]}"
    IFS='|' read -r TASK AH AS BH BS <<<"$spec"

    BASE_CFG="${CONFIG_ROOT}/${TASK}/best/${TASK}_${ABL}_best.yaml"
    RUN_NAME="p2s_${TASK}_a${ABL}_ah${AH}as${AS}_bh${BH}bs${BS}${RUN_SUFFIX}"
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

    TMP_OVR=$(mktemp "${TMPDIR:-/tmp}/ctcd_p2s_task_ovr_${SLURM_JOB_ID:-local}_${TID}.XXXXXX.yaml")
    TMP_MERGED=$(mktemp "${TMPDIR:-/tmp}/ctcd_p2s_task_merged_${SLURM_JOB_ID:-local}_${TID}.XXXXXX.yaml")
    trap 'rm -f "$TMP_OVR" "$TMP_MERGED"' EXIT

    cat >"$TMP_OVR" <<EOF
distillation:
  alpha_h: ${AH}
  alpha_s: ${AS}
  beta_h: ${BH}
  beta_s: ${BS}
checkpointing:
  no_weights: true
EOF

    python scripts/slurm/merge_yaml_overrides.py "$BASE_CFG" "$TMP_OVR" "$TMP_MERGED"

    echo "========================================="
    echo "CTCD Part 2 search (TASK)  rank=$RANK  tid=$TID  run=$RUN_NAME"
    echo "Base: $BASE_CFG"
    echo "alpha_h=$AH alpha_s=$AS  beta_h=$BH beta_s=$BS"
    echo "Merged: $TMP_MERGED"
    echo "========================================="

    if [[ -d "$OUTPUT_DIR" ]]; then
        _FORGE_SWEEP_WORKER=1 _FORGE_RESUME_PATH="$RUN_NAME" bash scripts/slurm/run_training.sh "$RUN_NAME" "$TMP_MERGED" &
    else
        _FORGE_SWEEP_WORKER=1 bash scripts/slurm/run_training.sh "$RUN_NAME" "$TMP_MERGED" &
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
        SWEEP_SCRIPT="${SLURM_SUBMIT_DIR}/scripts/slurm/run_ctcd_part2_search_TASK.sh"
        NEW_JOB=$(sbatch --parsable "$SWEEP_SCRIPT")
        echo "Resubmitted sweep as job $NEW_JOB ($REMAINING runs remaining for rank $RANK)"
    else
        echo "Rank $RANK has no remaining runs — no resubmit needed"
    fi
fi

slurm_print_footer
