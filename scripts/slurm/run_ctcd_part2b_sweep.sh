#!/bin/bash
# Part 2b CTCD: fix teacher (βh, βs) = (1, 1) in YAML; sweep student (αh, αs) on ablations 01–05.
# Base configs: configs/CTCD_Experiments/<task>/best/<task>_NN_best.yaml (Part 1 best LR/batch/clip).
# Presets (hard, soft) for (αh, αs): (1,1), (1,2), (1,4), (4,4).
# Grid: 6 tasks × 5 ablations × 4 presets = 120 runs.
#
# For frozen-teacher ablations 01/02, resolve_distillation_coefficients may pin β to (1,0) at runtime;
# the experiment knob is still student α.
#
# Worker-loop pattern: 4 SLURM array tasks (ranks 0–3) loop TIDs R, R+4, R+8, …
#
# Decode order (slow → fast): task, ablation (01..05), loss-weight preset.
#
# On timeout (SIGUSR1): checkpoint, break, sweep resubmits; .done markers skip finished TIDs.
#
# Usage (from repo root):
#   sbatch scripts/slurm/run_ctcd_part2b_sweep.sh
#
# Optional env:
#   CTCD_COMPLETION_DIR   default: completed_runs/ctcd_part2b
#   SEED MAX_STEPS EPOCHS DEBUG FORCE TIMESTAMP  forwarded by run_training.sh
#
# --- Smoke tests (repo root, conda env on) ---
# 1) Files exist:
#    TASKS=(mnli qqp qnli cola mrpc rte); for t in "${TASKS[@]}"; do for n in 01 02 03 04 05; do
#      test -f "configs/CTCD_Experiments/${t}/best/${t}_${n}_best.yaml" || echo MISSING "$t" "$n"; done; done
# 2) Merge + coeffs: merge_yaml_overrides + resolve_distillation_coefficients (check frozen 01 → β 1,0).
# 3) Short train: DEBUG=1 MAX_STEPS=10 FORCE=1 bash scripts/slurm/run_training.sh p2_smoke <merged.yaml>
# 4) One rank: MAX_STEPS=20 DEBUG=1 CTCD_COMPLETION_DIR=completed_runs/ctcd_part2b_smoke sbatch --array=0 "$0"

#SBATCH --job-name=ctcd_p2b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3
#SBATCH --output=logs/ctcd_p2b_%A_%a.out
#SBATCH --error=logs/ctcd_p2b_%A_%a.out
#SBATCH --signal=B:SIGUSR1@120

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT" || exit 1

mkdir -p logs

source scripts/slurm/slurm_essentials.sh

slurm_setup_env
slurm_print_header

CONFIG_ROOT="configs/CTCD_Experiments"
TASKS=(mnli qqp qnli cola mrpc rte)
NUM_TASKS=${#TASKS[@]}

# (hard, soft) → (αh, αs)
PRESET_H=(1 1 1 4)
PRESET_S=(1 2 4 4)
NUM_PRESETS=${#PRESET_H[@]}
NUM_ABLATIONS=5

WORKERS=4
RANK=${SLURM_ARRAY_TASK_ID:-0}
TOTAL=$((NUM_TASKS * NUM_ABLATIONS * NUM_PRESETS))

COMPLETION_DIR="${CTCD_COMPLETION_DIR:-completed_runs/ctcd_part2b}"
mkdir -p "$COMPLETION_DIR"

echo "========================================="
echo "CTCD Part 2b worker rank=$RANK / $((WORKERS - 1))"
echo "Total grid size: $TOTAL  Workers: $WORKERS"
echo "This rank handles TIDs: $RANK, $((RANK + WORKERS)), $((RANK + WORKERS * 2)), ..."
echo "Completion markers: $COMPLETION_DIR/"
echo "========================================="

decode_tid() {
    local tid=$1
    local per_abl=$((NUM_ABLATIONS * NUM_PRESETS))
    local task_idx=$((tid / per_abl))
    local rem=$((tid % per_abl))
    local abl_slot=$((rem / NUM_PRESETS))
    local preset_idx=$((rem % NUM_PRESETS))

    TASK="${TASKS[$task_idx]}"
    # abl_slot 0..4 → ablation indices 01..05
    local abl_num=$((1 + abl_slot))
    BASE_CFG="${CONFIG_ROOT}/${TASK}/best/${TASK}_$(printf '%02d' "$abl_num")_best.yaml"

    AH="${PRESET_H[$preset_idx]}"
    AS="${PRESET_S[$preset_idx]}"

    RUN_NAME="p2b_${TASK}_a$(printf '%02d' "$abl_num")_ah${AH}as${AS}"

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

    TMP_OVR=$(mktemp "${TMPDIR:-/tmp}/ctcd_p2b_ovr_${SLURM_JOB_ID:-local}_${TID}.XXXXXX.yaml")
    TMP_MERGED=$(mktemp "${TMPDIR:-/tmp}/ctcd_p2b_merged_${SLURM_JOB_ID:-local}_${TID}.XXXXXX.yaml")
    trap 'rm -f "$TMP_OVR" "$TMP_MERGED"' EXIT

    cat >"$TMP_OVR" <<EOF
distillation:
  alpha_h: ${AH}
  alpha_s: ${AS}
  beta_h: 1.0
  beta_s: 1.0
checkpointing:
  no_weights: true
EOF

    python scripts/slurm/merge_yaml_overrides.py "$BASE_CFG" "$TMP_OVR" "$TMP_MERGED"

    echo "========================================="
    echo "CTCD Part 2b  rank=$RANK  tid=$TID  run=$RUN_NAME"
    echo "Base: $BASE_CFG"
    echo "alpha_h=$AH alpha_s=$AS  beta_h=1 beta_s=1"
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
        SWEEP_SCRIPT="${SLURM_SUBMIT_DIR}/scripts/slurm/run_ctcd_part2b_sweep.sh"
        NEW_JOB=$(sbatch --parsable "$SWEEP_SCRIPT")
        echo "Resubmitted sweep as job $NEW_JOB ($REMAINING runs remaining for rank $RANK)"
    else
        echo "Rank $RANK has no remaining runs — no resubmit needed"
    fi
fi

slurm_print_footer
