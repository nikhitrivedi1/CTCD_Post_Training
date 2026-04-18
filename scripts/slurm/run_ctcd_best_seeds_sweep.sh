#!/bin/bash
# CTCD: train every Part-1 "best" YAML under configs/CTCD_Experiments/<task>/best/
# with four distinct random seeds (reproducible defaults; override via CTCD_SEEDS).
#
# Grid: N configs × 4 seeds (N = sum over tasks of *_best.yaml files; currently 36×4).
#
# Worker-loop pattern: 4 SLURM array tasks (ranks 0–3) loop TIDs R, R+4, R+8, …
# Decode order (slow → fast): config index (task order, then sorted ablation files), seed slot.
#
# On timeout (SIGUSR1): checkpoint, break, sweep resubmits; .done markers skip finished TIDs.
#
# Usage (from repo root):
#   sbatch scripts/slurm/run_ctcd_best_seeds_sweep.sh
#
# Optional env:
#   CTCD_COMPLETION_DIR   default: completed_runs/ctcd_best_seeds
#   CTCD_SEEDS            space-separated seeds (default: 42 1337 9001 4242)
#   SEED is set per run from CTCD_SEEDS; MAX_STEPS EPOCHS DEBUG FORCE TIMESTAMP
#                         forwarded by run_training.sh
#
# Requires bash 4+ (typical Slurm compute images).

#SBATCH --job-name=ctcd_rs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3
#SBATCH --output=logs/ctcd_best_seeds_%A_%a.out
#SBATCH --error=logs/ctcd_best_seeds_%A_%a.out
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

if [[ -n "${CTCD_SEEDS:-}" ]]; then
    read -r -a SEEDS <<<"$CTCD_SEEDS"
else
    SEEDS=(42 1337 9001 4242)
fi
NUM_SEEDS=${#SEEDS[@]}
if [[ $NUM_SEEDS -lt 1 ]]; then
    echo "CTCD_SEEDS must provide at least one integer seed"
    exit 1
fi

shopt -s nullglob
CONFIG_FILES=()
for TASK in "${TASKS[@]}"; do
    best_dir="${CONFIG_ROOT}/${TASK}/best"
    if [[ ! -d "$best_dir" ]]; then
        echo "Warning: missing directory $best_dir — skipping task $TASK"
        continue
    fi
    matches=("${best_dir}/${TASK}"_*_best.yaml)
    if [[ ${#matches[@]} -eq 0 ]] || [[ ! -f "${matches[0]}" ]]; then
        echo "Warning: no ${TASK}_*_best.yaml under $best_dir — skipping"
        continue
    fi
    mapfile -t sorted < <(printf '%s\n' "${matches[@]}" | LC_ALL=C sort)
    CONFIG_FILES+=("${sorted[@]}")
done
shopt -u nullglob

NUM_CFG=${#CONFIG_FILES[@]}
if [[ $NUM_CFG -eq 0 ]]; then
    echo "No best configs discovered under $CONFIG_ROOT — exiting"
    exit 1
fi

WORKERS=4
RANK=${SLURM_ARRAY_TASK_ID:-0}
TOTAL=$((NUM_CFG * NUM_SEEDS))

COMPLETION_DIR="${CTCD_COMPLETION_DIR:-completed_runs/ctcd_best_seeds}"
mkdir -p "$COMPLETION_DIR"

echo "========================================="
echo "CTCD best×seeds worker rank=$RANK / $((WORKERS - 1))"
echo "Configs: $NUM_CFG  Seeds per config: $NUM_SEEDS  Total runs: $TOTAL  Workers: $WORKERS"
echo "Seeds: ${SEEDS[*]}"
echo "Completion markers: $COMPLETION_DIR/"
echo "========================================="

decode_tid() {
    local tid=$1
    local cfg_idx=$((tid / NUM_SEEDS))
    local seed_idx=$((tid % NUM_SEEDS))

    BASE_CFG="${CONFIG_FILES[$cfg_idx]}"
    SEED_VAL="${SEEDS[$seed_idx]}"

    local stem
    stem=$(basename "$BASE_CFG" .yaml)
    RUN_NAME="rs_${stem}_seed${SEED_VAL}"

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

    TMP_OVR=$(mktemp "${TMPDIR:-/tmp}/ctcd_rs_ovr_${SLURM_JOB_ID:-local}_${TID}.XXXXXX.yaml")
    TMP_MERGED=$(mktemp "${TMPDIR:-/tmp}/ctcd_rs_merged_${SLURM_JOB_ID:-local}_${TID}.XXXXXX.yaml")
    trap 'rm -f "$TMP_OVR" "$TMP_MERGED"' EXIT

    cat >"$TMP_OVR" <<EOF
checkpointing:
  no_weights: true
EOF

    python scripts/slurm/merge_yaml_overrides.py "$BASE_CFG" "$TMP_OVR" "$TMP_MERGED"

    echo "========================================="
    echo "CTCD best×seeds  rank=$RANK  tid=$TID  run=$RUN_NAME"
    echo "Base: $BASE_CFG"
    echo "seed=$SEED_VAL"
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
        SWEEP_SCRIPT="${SLURM_SUBMIT_DIR}/scripts/slurm/run_ctcd_best_seeds_sweep.sh"
        NEW_JOB=$(sbatch --parsable "$SWEEP_SCRIPT")
        echo "Resubmitted sweep as job $NEW_JOB ($REMAINING runs remaining for rank $RANK)"
    else
        echo "Rank $RANK has no remaining runs — no resubmit needed"
    fi
fi

slurm_print_footer
