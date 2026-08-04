#!/bin/bash
# Part 1 CTCD grid under configs/CTCD_Experiments/: N tasks × 1 ablation (02, hardcoded
# below) × 4 LRs × 2 batch sizes × 2 grad-clip modes. N = len(TASKS).
# To restore the full 6-ablation sweep: set NUM_ABLATIONS=6 and revert decode_tid()'s
# abl_idx to the commented-out computed version below.
#
# Worker-loop pattern: 4 SLURM array tasks (ranks 0–3) each loop through their
# assigned slice of the grid sequentially (rank R handles TIDs R, R+4, R+8, ...).
# This keeps total submitted jobs at 4, within the QOS per-user limit.
#
# Decode order within each TID (slow → fast):
#   task, ablation, learning_rate, batch_size, max_grad_norm (clip).
#
# On timeout (SIGUSR1): current training run checkpoints cleanly, loop breaks,
# sweep resubmits itself. On next run, .done markers skip completed TIDs and
# forge.py resumes from the saved checkpoint.
#
# Usage (from repo root):
#   sbatch scripts/slurm/run_ctcd_part1_sweep.sh
#
# Optional env:
#   CTCD_COMPLETION_DIR   default: completed_runs/ctcd_part1
#   SEED MAX_STEPS EPOCHS DEBUG FORCE TIMESTAMP  forwarded by run_training.sh

#SBATCH --job-name=ctcd_p1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3
#SBATCH --output=logs/ctcd_p1_%A_%a.out
#SBATCH --error=logs/ctcd_p1_%A_%a.out
#SBATCH --signal=B:SIGUSR1@120

set -uo pipefail

# SLURM copies the script to a spool dir before execution, so BASH_SOURCE[0] resolves
# to the spool path rather than the repo. SLURM_SUBMIT_DIR is always the directory from
# which sbatch was invoked — use it as the authoritative repo root.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT" || exit 1

mkdir -p logs

source scripts/slurm/slurm_essentials.sh

slurm_setup_env
slurm_print_header

export WANDB_INIT_TIMEOUT=300

CONFIG_ROOT="configs/CTCD_Experiments"
TASKS=(sst2 clinc150)
NUM_TASKS=${#TASKS[@]}
# shellcheck disable=SC2034
LRS=(1e-5 2e-5 3e-5 5e-5)
BATCHES=(16 32)
# 0.0 => no clipping (TrainingArguments only clips when max_grad_norm > 0)
CLIPS=(0.0 1.0)

ablation_yaml() {
    case "$1" in
        0) echo "00_sft.yaml" ;;
        1) echo "01_kd_frozen_base.yaml" ;;
        2) echo "02_kd_frozen_sft.yaml" ;;
        3) echo "03_cotrain_base.yaml" ;;
        4) echo "04_cotrain_sft.yaml" ;;
        5) echo "05_ctcd.yaml" ;;
        *) echo ""
            return 1
            ;;
    esac
}

WORKERS=4
RANK=${SLURM_ARRAY_TASK_ID:-0}
PER_GRID=16
NUM_ABLATIONS=1  # restricted to ablation 02 only; see decode_tid()
# NUM_TASKS × NUM_ABLATIONS × PER_GRID (e.g. 6×1×16 = 96)
TOTAL=$((NUM_TASKS * NUM_ABLATIONS * PER_GRID))

COMPLETION_DIR="${CTCD_COMPLETION_DIR:-completed_runs/ctcd_part1}"
mkdir -p "$COMPLETION_DIR"

echo "========================================="
echo "CTCD Part 1 worker rank=$RANK / $((WORKERS-1))"
echo "Total grid size: $TOTAL  Workers: $WORKERS"
echo "This rank handles TIDs: $RANK, $((RANK+WORKERS)), $((RANK+WORKERS*2)), ..."
echo "Completion markers: $COMPLETION_DIR/"
echo "========================================="

# ---------------------------------------------------------------------------
# Helper: decode a TID and run one training job.
# Sets DONE and OUTPUT_DIR in the outer scope via global assignments.
# ---------------------------------------------------------------------------
decode_tid() {
    local tid=$1

    local task_idx=$(( tid / PER_GRID ))
    local grid_idx=$(( tid % PER_GRID ))
    local abl_idx=2  # hardcoded to 02_kd_frozen_sft.yaml
    # Full 6-ablation version (revert alongside NUM_ABLATIONS=6 above):
    #   local task_idx=$(( tid / (NUM_ABLATIONS * PER_GRID) ))
    #   local rem=$(( tid % (NUM_ABLATIONS * PER_GRID) ))
    #   local abl_idx=$(( rem / PER_GRID ))
    #   local grid_idx=$(( rem % PER_GRID ))

    local lr_idx=$(( grid_idx / 4 ))
    local rem4=$(( grid_idx % 4 ))
    local batch_idx=$(( rem4 / 2 ))
    local clip_idx=$(( rem4 % 2 ))

    TASK="${TASKS[$task_idx]}"
    ABL_FILE=$(ablation_yaml "$abl_idx") || { echo "Bad ablation index $abl_idx"; return 1; }
    LR="${LRS[$lr_idx]}"
    BS="${BATCHES[$batch_idx]}"
    CLIP="${CLIPS[$clip_idx]}"

    BASE_CFG="${CONFIG_ROOT}/${TASK}/${ABL_FILE}"

    local lr_tag=${LR//./p}
    lr_tag=${lr_tag//-/m}
    local clip_tag=${CLIP//./p}
    RUN_NAME="p1_${TASK}_a$(printf '%02d' "$abl_idx")_lr${lr_tag}_b${BS}_c${clip_tag}"

    DONE="${COMPLETION_DIR}/${RUN_NAME}.done"
    OUTPUT_DIR="./outputs/${RUN_NAME}"
}

# ---------------------------------------------------------------------------
# Main worker loop
# ---------------------------------------------------------------------------
TASKS_COMPLETED=0
TASKS_FAILED=0

for (( TID=RANK; TID<TOTAL; TID+=WORKERS )); do

    # Check for timeout before starting a new run
    if [[ $SIGUSR1_RECEIVED -eq 1 ]]; then
        echo "Timeout signal received — stopping before TID $TID"
        break
    fi

    decode_tid "$TID" || continue

    # Skip if already done
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

    TMP_OVR=$(mktemp "${TMPDIR:-/tmp}/ctcd_ovr_${SLURM_JOB_ID:-local}_${TID}.XXXXXX.yaml")
    TMP_MERGED=$(mktemp "${TMPDIR:-/tmp}/ctcd_merged_${SLURM_JOB_ID:-local}_${TID}.XXXXXX.yaml")
    # Clean up temp files when this iteration is done or on unexpected exit
    trap 'rm -f "$TMP_OVR" "$TMP_MERGED"' EXIT

    cat >"$TMP_OVR" <<EOF
training:
  learning_rate: ${LR}
  batch_size: ${BS}
  max_grad_norm: ${CLIP}
checkpointing:
  no_weights: true
EOF

    python scripts/slurm/merge_yaml_overrides.py "$BASE_CFG" "$TMP_OVR" "$TMP_MERGED"

    echo "========================================="
    echo "CTCD Part 1  rank=$RANK  tid=$TID  run=$RUN_NAME"
    echo "Base: $BASE_CFG"
    echo "lr=$LR  batch_size=$BS  max_grad_norm=$CLIP"
    echo "Merged: $TMP_MERGED"
    echo "========================================="

    # Resume if output dir exists (incomplete run), otherwise start fresh
    if [[ -d "$OUTPUT_DIR" ]]; then
        _FORGE_SWEEP_WORKER=1 _FORGE_RESUME_PATH="$RUN_NAME" bash scripts/slurm/run_training.sh "$RUN_NAME" "$TMP_MERGED" &
    else
        _FORGE_SWEEP_WORKER=1 bash scripts/slurm/run_training.sh "$RUN_NAME" "$TMP_MERGED" &
    fi
    slurm_wait_child $!
    TRAIN_EXIT=$EXIT_CODE

    rm -f "$TMP_OVR" "$TMP_MERGED"
    trap - EXIT  # reset trap; next iteration will re-register

    # If SIGUSR1 arrived during training, checkpoint was saved — break and resubmit
    if [[ $SIGUSR1_RECEIVED -eq 1 ]]; then
        echo "Run $RUN_NAME interrupted — checkpoint saved"
        break
    fi

    if [[ $TRAIN_EXIT -eq 0 ]]; then
        if [[ -f "$OUTPUT_DIR/training_state.json" ]] && \
           [[ "$(slurm_is_complete "$OUTPUT_DIR")" == "true" ]]; then
            touch "$DONE"
            echo "Marked complete: $DONE"
            TASKS_COMPLETED=$(( TASKS_COMPLETED + 1 ))
        else
            echo "Warning: $RUN_NAME exited 0 but not pipeline-complete — skipping mark"
            TASKS_FAILED=$(( TASKS_FAILED + 1 ))
        fi
    else
        echo "Training failed for $RUN_NAME (exit $TRAIN_EXIT) — continuing to next run"
        TASKS_FAILED=$(( TASKS_FAILED + 1 ))
    fi

done

# ---------------------------------------------------------------------------
# Summary and resubmit if interrupted
# ---------------------------------------------------------------------------
echo "========================================="
echo "Rank $RANK summary: completed=$TASKS_COMPLETED  failed/skipped=$TASKS_FAILED"
echo "========================================="

if [[ $SIGUSR1_RECEIVED -eq 1 ]]; then
    # Count remaining work for this rank
    REMAINING=0
    for (( TID=RANK; TID<TOTAL; TID+=WORKERS )); do
        decode_tid "$TID" || continue
        [[ -f "$DONE" ]] || REMAINING=$(( REMAINING + 1 ))
    done

    if [[ $REMAINING -gt 0 ]]; then
        SWEEP_SCRIPT="${SLURM_SUBMIT_DIR}/scripts/slurm/run_ctcd_part1_sweep.sh"
        NEW_JOB=$(sbatch --parsable "$SWEEP_SCRIPT")
        echo "Resubmitted sweep as job $NEW_JOB ($REMAINING runs remaining for rank $RANK)"
    else
        echo "Rank $RANK has no remaining runs — no resubmit needed"
    fi
fi

slurm_print_footer
