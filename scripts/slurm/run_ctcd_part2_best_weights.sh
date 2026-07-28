#!/bin/bash
# Retrain Part-2 best α×β cells (every best-score matrix cell) with weights kept.
#
# Base configs: same as run_ctcd_part2_search.sh
#   configs/CTCD_Experiments/<task>/best/<task>_NN_best.yaml
#
# BEST_CELLS come from part_2_vis.ipynb —
# "All best α×β configs (every matrix cell)":
#   1) winning α×β by 3-seed mean per (task, ablation)
#   2) best_seed = argmax eval score within that cell
# Format: task|abl|ah|as|bh|bs|seed
#
# Run names use p2sw_ so existing no-weight p2s_* dirs are left alone:
#   p2sw_mnli_a05_ah1as4_bh1bs2_s127
#
# Worker-loop pattern: 4 SLURM array tasks (ranks 0–3) loop TIDs R, R+4, …
# On timeout (SIGUSR1): checkpoint, break, sweep resubmits; .done markers skip finished TIDs.
#
# Usage (from repo root):
#   sbatch scripts/slurm/run_ctcd_part2_best_weights.sh
#
# Optional env:
#   CTCD_COMPLETION_DIR   default: completed_runs/ctcd_part2_best_weights
#   MAX_STEPS EPOCHS DEBUG FORCE TIMESTAMP  forwarded by run_training.sh
#
# --- Smoke tests (repo root, conda env on) ---
# 1) Base YAMLs exist for every BEST_CELLS task/ablation
# 2) One rank:
#    DEBUG=1 MAX_STEPS=10 CTCD_COMPLETION_DIR=completed_runs/ctcd_p2sw_smoke \
#      sbatch --array=0 scripts/slurm/run_ctcd_part2_best_weights.sh

#SBATCH --job-name=ctcd_p2sw
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3
#SBATCH --output=logs/ctcd_p2sw_%A_%a.out
#SBATCH --error=logs/ctcd_p2sw_%A_%a.out
#SBATCH --signal=B:SIGUSR1@120

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT" || exit 1

mkdir -p logs

source scripts/slurm/slurm_essentials.sh

slurm_setup_env
slurm_print_header

export WANDB_INIT_TIMEOUT=300

CONFIG_ROOT="configs/CTCD_Experiments"

# From part_2_vis "All best α×β configs (every matrix cell)" (best_seed annotated).
# task|abl|ah|as|bh|bs|seed
BEST_CELLS=(
  "clinc150|01|1|2|0|0|41"
  "clinc150|03|1|4|1|4|127"
  "clinc150|05|1|4|1|4|127"
  "cola|01|1|2|0|0|256"
  "cola|02|4|4|0|0|256"
  "cola|03|1|1|1|0|41"
  "cola|04|1|2|1|0|256"
  "cola|05|4|4|4|4|256"
  "mnli|01|1|1|0|0|41"
  "mnli|02|1|1|0|0|256"
  "mnli|03|1|4|1|0|41"
  "mnli|04|1|4|1|0|256"
  "mnli|05|1|4|1|2|127"
  "mrpc|01|1|2|0|0|127"
  "mrpc|02|1|1|0|0|256"
  "mrpc|03|1|4|1|0|41"
  "mrpc|04|4|4|1|0|127"
  "mrpc|05|1|2|1|1|127"
  "qnli|01|4|4|0|0|127"
  "qnli|02|4|4|0|0|256"
  "qnli|03|1|4|1|0|127"
  "qnli|04|1|4|1|0|127"
  "qnli|05|1|4|1|2|127"
  "qqp|01|1|1|0|0|256"
  "qqp|02|4|4|0|0|127"
  "qqp|03|1|4|1|0|256"
  "qqp|04|1|4|1|0|256"
  "qqp|05|1|4|1|4|256"
  "rte|01|4|4|0|0|41"
  "rte|02|4|4|0|0|127"
  "rte|03|4|4|1|0|41"
  "rte|04|4|4|1|0|41"
  "rte|05|4|4|1|1|41"
  "sst2|01|1|2|0|0|127"
  "sst2|02|1|4|0|0|127"
  "sst2|03|1|2|1|0|256"
  "sst2|04|1|2|1|0|127"
  "sst2|05|1|2|1|2|41"
)

JOBS=()
for cell in "${BEST_CELLS[@]}"; do
    IFS='|' read -r TASK abl AH AS BH BS SEED <<<"$cell"
    if [[ -z "${SEED:-}" ]] || ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
        echo "Error: invalid seed in BEST_CELLS entry: $cell" >&2
        exit 1
    fi
    abl_padded=$(printf '%02d' "$((10#$abl))")
    JOBS+=("${TASK}|${abl_padded}|${AH}|${AS}|${BH}|${BS}|${SEED}")
done

TOTAL=${#JOBS[@]}
if [[ $TOTAL -lt 1 ]]; then
    echo "Error: BEST_CELLS / JOBS is empty."
    exit 1
fi

WORKERS=4
RANK=${SLURM_ARRAY_TASK_ID:-0}

COMPLETION_DIR="${CTCD_COMPLETION_DIR:-completed_runs/ctcd_part2_best_weights}"
mkdir -p "$COMPLETION_DIR"

echo "========================================="
echo "CTCD Part-2 best-weights worker rank=$RANK / $((WORKERS - 1))"
echo "Jobs: $TOTAL  Workers: $WORKERS"
echo "This rank handles TIDs: $RANK, $((RANK + WORKERS)), $((RANK + WORKERS * 2)), ..."
echo "Completion markers: $COMPLETION_DIR/"
echo "========================================="

decode_tid() {
    local tid=$1
    local spec="${JOBS[$tid]}"
    IFS='|' read -r TASK abl_padded AH AS BH BS SEED <<<"$spec"

    BASE_CFG="${CONFIG_ROOT}/${TASK}/best/${TASK}_${abl_padded}_best.yaml"
    # Distinct prefix from search runs (p2s_*) which used no_weights:true
    RUN_NAME="p2sw_${TASK}_a${abl_padded}_ah${AH}as${AS}_bh${BH}bs${BS}_s${SEED}"
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
        TASKS_FAILED=$((TASKS_FAILED + 1))
        continue
    fi

    TMP_OVR=$(mktemp "${TMPDIR:-/tmp}/ctcd_p2sw_ovr_${SLURM_JOB_ID:-local}_${TID}.XXXXXX.yaml")
    TMP_MERGED=$(mktemp "${TMPDIR:-/tmp}/ctcd_p2sw_merged_${SLURM_JOB_ID:-local}_${TID}.XXXXXX.yaml")
    trap 'rm -f "$TMP_OVR" "$TMP_MERGED"' EXIT

    # Same merge shape as part2_search, but keep final weights for reference / Hub upload
    cat >"$TMP_OVR" <<EOF
distillation:
  alpha_h: ${AH}
  alpha_s: ${AS}
  beta_h: ${BH}
  beta_s: ${BS}
checkpointing:
  no_weights: false
EOF

    python scripts/slurm/merge_yaml_overrides.py "$BASE_CFG" "$TMP_OVR" "$TMP_MERGED"

    export SEED
    echo "========================================="
    echo "CTCD Part-2 best-weights  rank=$RANK  tid=$TID  run=$RUN_NAME"
    echo "Base: $BASE_CFG"
    echo "alpha_h=$AH alpha_s=$AS  beta_h=$BH beta_s=$BS  SEED=$SEED"
    echo "Merged: $TMP_MERGED"
    echo "========================================="

    if [[ -d "$OUTPUT_DIR" ]]; then
        _FORGE_SWEEP_WORKER=1 _FORGE_RESUME_PATH="$RUN_NAME" bash scripts/slurm/run_training.sh "$RUN_NAME" "$TMP_MERGED" &
    else
        _FORGE_SWEEP_WORKER=1 bash scripts/slurm/run_training.sh "$RUN_NAME" "$TMP_MERGED" &
    fi
    slurm_wait_child $!
    TRAIN_EXIT=$EXIT_CODE

    unset SEED
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
        SWEEP_SCRIPT="${SLURM_SUBMIT_DIR}/scripts/slurm/run_ctcd_part2_best_weights.sh"
        NEW_JOB=$(sbatch --parsable "$SWEEP_SCRIPT")
        echo "Resubmitted sweep as job $NEW_JOB ($REMAINING runs remaining for rank $RANK)"
    else
        echo "Rank $RANK has no remaining runs — no resubmit needed"
    fi
fi

slurm_print_footer
