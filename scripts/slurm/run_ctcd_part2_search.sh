#!/bin/bash
# Part 2 CTCD coefficient search with ablation-specific α/β constraints.
# Base configs: configs/CTCD_Experiments/<task>/best/<task>_NN_best.yaml
#
# Coefficient presets (hard, soft): (1,1), (1,2), (1,4), (4,4).
# Ablation constraints:
#   a01 / a02  — frozen teacher: sweep α only; fix β = (0, 0)
#   a03 / a04  — co-train, no CD: sweep α only; fix β = (1, 0)  (βs = 0)
#   a05        — full CTCD: sweep all α × β preset combinations
#
# Tasks / ablations (edit TASKS + task_ablations):
#   clinc150: 01, 03, 05  (no 02/04 configs)
#   GLUE later: append tasks; use 01–05 when best YAMLs exist
#
# Seeds: 3 (default 41, 127, 256; override with CTCD_SEEDS).
# clinc150 grid (defaults): a01/a03 each 4α×1β×3seeds (=12) + a05 4α×4β×3seeds (=48)
#   = 12 + 12 + 48 = 72 runs.
#
# Jobs are built as an explicit list (non-rectangular), then workers loop TIDs.
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

# Current launch set. Append GLUE tasks when Part-1 best configs are ready, e.g.:
#   TASKS=(clinc150 mnli qqp qnli cola mrpc rte sst2)
TASKS=(clinc150)

# Per-task ablation lists (space-separated). clinc150 has no 02/04; GLUE typically 01–05.
task_ablations() {
    case "$1" in
        clinc150) echo "01 03 05" ;;
        # GLUE / others: full family when configs exist
        mnli | qqp | qnli | cola | mrpc | rte | sst2 | wnli) echo "01 02 03 04 05" ;;
        *)
            echo "Error: no ablation list for task '$1' (add a branch in task_ablations)." >&2
            return 1
            ;;
    esac
}

# (hard, soft) presets for α (and for β on a05)
PRESET_H=(1 1 1 4)
PRESET_S=(1 2 4 4)
NUM_PRESETS=${#PRESET_H[@]}

# Fixed β by ablation family (written into YAML + run name)
# a01/a02 frozen: β unused at runtime but recorded as (0,0)
FROZEN_BH=0
FROZEN_BS=0
# a03/a04 co-train without CD: βs must stay 0
COTRAIN_BH=1
COTRAIN_BS=0

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

# Build explicit job list: each entry is "task|abl|ah|as|bh|bs|seed"
# Avoids a rectangular decode that would re-introduce invalid α×β cells.
JOBS=()
for TASK in "${TASKS[@]}"; do
    abl_str=$(task_ablations "$TASK") || exit 1
    # shellcheck disable=SC2206
    ABL_LIST=($abl_str)
    for abl in "${ABL_LIST[@]}"; do
        for ((ai = 0; ai < NUM_PRESETS; ai++)); do
            AH="${PRESET_H[$ai]}"
            AS="${PRESET_S[$ai]}"
            case "$abl" in
                01 | 02)
                    BETA_PAIRS=("${FROZEN_BH}:${FROZEN_BS}")
                    ;;
                03 | 04)
                    BETA_PAIRS=("${COTRAIN_BH}:${COTRAIN_BS}")
                    ;;
                05)
                    BETA_PAIRS=()
                    for ((bi = 0; bi < NUM_PRESETS; bi++)); do
                        BETA_PAIRS+=("${PRESET_H[$bi]}:${PRESET_S[$bi]}")
                    done
                    ;;
                *)
                    echo "Error: unknown ablation '$abl' for task $TASK" >&2
                    exit 1
                    ;;
            esac
            for pair in "${BETA_PAIRS[@]}"; do
                BH="${pair%%:*}"
                BS="${pair##*:}"
                for SEED in "${SEEDS[@]}"; do
                    JOBS+=("${TASK}|${abl}|${AH}|${AS}|${BH}|${BS}|${SEED}")
                done
            done
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

COMPLETION_DIR="${CTCD_COMPLETION_DIR:-completed_runs/ctcd_part2_search}"
mkdir -p "$COMPLETION_DIR"

echo "========================================="
echo "CTCD Part 2 search worker rank=$RANK / $((WORKERS - 1))"
echo "Tasks: ${TASKS[*]}"
echo "Total jobs: $TOTAL  Workers: $WORKERS  Seeds: $NUM_SEEDS"
echo "Constraints: a01/a02 β=(${FROZEN_BH},${FROZEN_BS}); a03/a04 β=(${COTRAIN_BH},${COTRAIN_BS}); a05 full α×β"
echo "This rank handles TIDs: $RANK, $((RANK + WORKERS)), $((RANK + WORKERS * 2)), ..."
echo "Completion markers: $COMPLETION_DIR/"
echo "========================================="

decode_tid() {
    local tid=$1
    local spec="${JOBS[$tid]}"
    IFS='|' read -r TASK abl_padded AH AS BH BS SEED <<<"$spec"

    BASE_CFG="${CONFIG_ROOT}/${TASK}/best/${TASK}_${abl_padded}_best.yaml"
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
