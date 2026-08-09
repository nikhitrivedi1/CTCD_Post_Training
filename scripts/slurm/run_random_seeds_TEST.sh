#!/bin/bash
# TEST clone of run_random_seeds.sh — scoped reruns of the CTCD best-config random-seed
# sweep without touching production output dirs or completion markers.
#
# By default this still walks the full task x ablation x seed grid, but CTCD_TASKS /
# CTCD_ABLATIONS can restrict it to a subset (e.g. just mnli ablation 05), and every
# run name gets a suffix (default _TEST) so it never collides with a prior real run.
#
# Worker-loop pattern: 4 SLURM array tasks (ranks 0-3) loop TIDs R, R+4, R+8, ...
#
# On timeout (SIGUSR1): checkpoint, break, sweep resubmits; .done markers skip finished TIDs.
#
# Usage (from repo root):
#   sbatch scripts/slurm/run_random_seeds_TEST.sh
#
# Optional env:
#   CTCD_TASKS            comma-separated tasks, e.g. mnli (default: all 6 tasks)
#   CTCD_ABLATIONS        comma-separated ablation indices, e.g. 05 (default: 00,01,02,03,04,05)
#   CTCD_SEEDS            comma-separated seeds, e.g. 41,127 (default: 10 built-in seeds)
#   CTCD_RUN_SUFFIX        appended to every run name (default: _TEST)
#   CTCD_COMPLETION_DIR   default: completed_runs/ctcd_random_seeds_test
#   MAX_STEPS EPOCHS DEBUG FORCE TIMESTAMP  forwarded by run_training.sh (same as other CTCD sweeps)
#
# --- Smoke test (repo root, conda env on) ---
#   MAX_STEPS=20 DEBUG=1 CTCD_TASKS=mnli CTCD_ABLATIONS=05 CTCD_SEEDS=41 sbatch --array=0 "$0"

#SBATCH --job-name=ctcd_rs_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3
#SBATCH --output=logs/ctcd_rs_test_%A_%a.out
#SBATCH --error=logs/ctcd_rs_test_%A_%a.out
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
    TASKS=(qqp qnli cola mrpc rte mnli)
fi
NUM_TASKS=${#TASKS[@]}

if [[ -n "${CTCD_ABLATIONS:-}" ]]; then
    IFS=',' read -ra ABLATIONS <<<"${CTCD_ABLATIONS}"
else
    ABLATIONS=(00 01 02 03 04 05)
fi
NUM_ABLATIONS=${#ABLATIONS[@]}

if [[ -n "${CTCD_SEEDS:-}" ]]; then
    IFS=',' read -ra SEEDS <<<"${CTCD_SEEDS}"
else
    # Ten fixed seeds (reproducible); replace via CTCD_SEEDS if you want a different set.
    SEEDS=(41 127 256 512 733 991 1237 2468 3691 4242)
fi
NUM_SEEDS=${#SEEDS[@]}
if [[ $NUM_SEEDS -lt 1 ]]; then
    echo "Error: no seeds configured (set CTCD_SEEDS or edit default SEEDS array)."
    exit 1
fi

RUN_SUFFIX="${CTCD_RUN_SUFFIX:-_TEST}"

WORKERS=4
RANK=${SLURM_ARRAY_TASK_ID:-0}
TOTAL=$((NUM_TASKS * NUM_ABLATIONS * NUM_SEEDS))

COMPLETION_DIR="${CTCD_COMPLETION_DIR:-completed_runs/ctcd_random_seeds_test}"
mkdir -p "$COMPLETION_DIR"

echo "========================================="
echo "CTCD random-seed sweep (TEST) worker rank=$RANK / $((WORKERS - 1))"
echo "Tasks: ${TASKS[*]}  Ablations: ${ABLATIONS[*]}  Seeds: ${SEEDS[*]}"
echo "Total grid size: $TOTAL  Workers: $WORKERS  Run suffix: $RUN_SUFFIX"
echo "This rank handles TIDs: $RANK, $((RANK + WORKERS)), $((RANK + WORKERS * 2)), ..."
echo "Completion markers: $COMPLETION_DIR/"
echo "========================================="

decode_tid() {
    local tid=$1
    local per_task=$((NUM_ABLATIONS * NUM_SEEDS))
    local task_idx=$((tid / per_task))
    local rem=$((tid % per_task))
    local abl_idx=$((rem / NUM_SEEDS))
    local seed_idx=$((rem % NUM_SEEDS))

    TASK="${TASKS[$task_idx]}"
    local abl_padded
    abl_padded=$(printf '%02d' "$((10#${ABLATIONS[$abl_idx]}))")
    BASE_CFG="${CONFIG_ROOT}/${TASK}/best/${TASK}_${abl_padded}_best.yaml"

    SEED="${SEEDS[$seed_idx]}"
    RUN_NAME="rs_${TASK}_a${abl_padded}_s${SEED}${RUN_SUFFIX}"

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

    export SEED
    echo "========================================="
    echo "CTCD random seeds (TEST)  rank=$RANK  tid=$TID  run=$RUN_NAME"
    echo "Base: $BASE_CFG"
    echo "SEED=$SEED"
    echo "========================================="

    if [[ -d "$OUTPUT_DIR" ]]; then
        _FORGE_SWEEP_WORKER=1 _FORGE_RESUME_PATH="$RUN_NAME" bash scripts/slurm/run_training.sh "$RUN_NAME" "$BASE_CFG" &
    else
        _FORGE_SWEEP_WORKER=1 bash scripts/slurm/run_training.sh "$RUN_NAME" "$BASE_CFG" &
    fi
    slurm_wait_child $!
    TRAIN_EXIT=$EXIT_CODE

    unset SEED

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
        SWEEP_SCRIPT="${SLURM_SUBMIT_DIR}/scripts/slurm/run_random_seeds_TEST.sh"
        NEW_JOB=$(sbatch --parsable "$SWEEP_SCRIPT")
        echo "Resubmitted sweep as job $NEW_JOB ($REMAINING runs remaining for rank $RANK)"
    else
        echo "Rank $RANK has no remaining runs — no resubmit needed"
    fi
fi

slurm_print_footer
