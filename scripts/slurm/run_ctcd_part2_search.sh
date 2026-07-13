#!/bin/bash
# Part 2 CTCD joint α×β search on clinc150 only.
# Base configs: configs/CTCD_Experiments/clinc150/best/clinc150_NN_best.yaml
# Ablations: 01, 03, 05 (skip 02 and 04).
# Presets (hard, soft) for (αh, αs) and (βh, βs): (1,1), (1,2), (1,4), (4,4).
# Seeds: 3 (default 41, 127, 256; override with CTCD_SEEDS).
# Grid: 1 task × 3 ablations × 4 α × 4 β × 3 seeds = 144 runs.
#
# For frozen-teacher ablation 01, resolve_distillation_coefficients may pin β at runtime;
# the YAML still records the requested (βh, βs).
#
# Worker-loop pattern: 4 SLURM array tasks (ranks 0–3) loop TIDs R, R+4, R+8, …
#
# Decode order (slow → fast): ablation, α preset, β preset, seed.
#
# On timeout (SIGUSR1): checkpoint, break, sweep resubmits; .done markers skip finished TIDs.
#
# Usage (from repo root):
#   sbatch scripts/slurm/run_ctcd_part2_search.sh
#
# Optional env:
#   CTCD_COMPLETION_DIR   default: completed_runs/ctcd_part2_search
#   CTCD_SEEDS            comma-separated seeds (default: 41,127,256)
#   MAX_STEPS EPOCHS DEBUG FORCE TIMESTAMP  forwarded by run_training.sh
#
# --- Smoke tests (repo root, conda env on) ---
# 1) Files exist:
#    for n in 01 03 05; do
#      test -f "configs/CTCD_Experiments/clinc150/best/clinc150_${n}_best.yaml" || echo MISSING "$n"; done
# 2) Merge + coeffs: merge_yaml_overrides.py + resolve_distillation_coefficients
# 3) Short train: DEBUG=1 MAX_STEPS=10 FORCE=1 bash scripts/slurm/run_training.sh p2s_smoke <merged.yaml>
# 4) One rank: MAX_STEPS=20 DEBUG=1 CTCD_COMPLETION_DIR=completed_runs/ctcd_p2s_smoke sbatch --array=0 "$0"

#SBATCH --job-name=ctcd_p2s
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3
#SBATCH --output=logs/ctcd_p2s_%A_%a.out
#SBATCH --error=logs/ctcd_p2s_%A_%a.out
#SBATCH --signal=B:SIGUSR1@120

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT" || exit 1

mkdir -p logs

source scripts/slurm/slurm_essentials.sh

slurm_setup_env
slurm_print_header

CONFIG_ROOT="configs/CTCD_Experiments"
TASKS=(clinc150)
NUM_TASKS=${#TASKS[@]}

# Skip ablations 02 and 04
ABLATIONS=(01 03 05)
NUM_ABLATIONS=${#ABLATIONS[@]}

# (hard, soft) presets shared by α and β
PRESET_H=(1 1 1 4)
PRESET_S=(1 2 4 4)
NUM_PRESETS=${#PRESET_H[@]}

if [[ -n "${CTCD_SEEDS:-}" ]]; then
    IFS=',' read -ra SEEDS <<<"${CTCD_SEEDS}"
else
    SEEDS=(41 127 256)
fi
NUM_SEEDS=${#SEEDS[@]}
if [[ $NUM_SEEDS -lt 1 ]]; then
    echo "Error: no seeds configured (set CTCD_SEEDS or edit default SEEDS array)."
    exit 1
fi

WORKERS=4
RANK=${SLURM_ARRAY_TASK_ID:-0}
TOTAL=$((NUM_TASKS * NUM_ABLATIONS * NUM_PRESETS * NUM_PRESETS * NUM_SEEDS))

COMPLETION_DIR="${CTCD_COMPLETION_DIR:-completed_runs/ctcd_part2_search}"
mkdir -p "$COMPLETION_DIR"

echo "========================================="
echo "CTCD Part 2 search worker rank=$RANK / $((WORKERS - 1))"
echo "Total grid size: $TOTAL  Workers: $WORKERS  Seeds: $NUM_SEEDS"
echo "This rank handles TIDs: $RANK, $((RANK + WORKERS)), $((RANK + WORKERS * 2)), ..."
echo "Completion markers: $COMPLETION_DIR/"
echo "========================================="

decode_tid() {
    local tid=$1
    local per_abl=$((NUM_PRESETS * NUM_PRESETS * NUM_SEEDS))
    local per_alpha=$((NUM_PRESETS * NUM_SEEDS))
    local task_idx=$((tid / (NUM_ABLATIONS * per_abl)))
    local rem=$((tid % (NUM_ABLATIONS * per_abl)))
    local abl_slot=$((rem / per_abl))
    rem=$((rem % per_abl))
    local alpha_idx=$((rem / per_alpha))
    rem=$((rem % per_alpha))
    local beta_idx=$((rem / NUM_SEEDS))
    local seed_idx=$((rem % NUM_SEEDS))

    TASK="${TASKS[$task_idx]}"
    local abl_padded="${ABLATIONS[$abl_slot]}"
    BASE_CFG="${CONFIG_ROOT}/${TASK}/best/${TASK}_${abl_padded}_best.yaml"

    AH="${PRESET_H[$alpha_idx]}"
    AS="${PRESET_S[$alpha_idx]}"
    BH="${PRESET_H[$beta_idx]}"
    BS="${PRESET_S[$beta_idx]}"
    SEED="${SEEDS[$seed_idx]}"

    RUN_NAME="p2s_${TASK}_a${abl_padded}_ah${AH}as${AS}_bh${BH}bs${BS}_s${SEED}"

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

    TMP_OVR=$(mktemp "${TMPDIR:-/tmp}/ctcd_p2s_ovr_${SLURM_JOB_ID:-local}_${TID}.XXXXXX.yaml")
    TMP_MERGED=$(mktemp "${TMPDIR:-/tmp}/ctcd_p2s_merged_${SLURM_JOB_ID:-local}_${TID}.XXXXXX.yaml")
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

    export SEED
    echo "========================================="
    echo "CTCD Part 2 search  rank=$RANK  tid=$TID  run=$RUN_NAME"
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
        SWEEP_SCRIPT="${SLURM_SUBMIT_DIR}/scripts/slurm/run_ctcd_part2_search.sh"
        NEW_JOB=$(sbatch --parsable "$SWEEP_SCRIPT")
        echo "Resubmitted sweep as job $NEW_JOB ($REMAINING runs remaining for rank $RANK)"
    else
        echo "Rank $RANK has no remaining runs — no resubmit needed"
    fi
fi

slurm_print_footer
