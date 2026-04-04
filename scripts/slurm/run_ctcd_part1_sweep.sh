#!/bin/bash
# Part 1 CTCD grid under configs/CTCD_Experiments/: 6 tasks × 6 ablations × 4 LRs ×
# 2 batch sizes × 2 grad-clip modes = 576 runs.
#
# Each array task handles exactly one grid index (tid). Decode order (slow → fast):
#   task, ablation, learning_rate, batch_size, max_grad_norm (clip).
#
# Ephemeral merged YAMLs live in mktemp paths; a trap removes them on exit. HF
# checkpoint-* cleanup after successful training is handled in forge.py.
#
# Usage (from anywhere; script cds to repo root):
#   sbatch scripts/slurm/run_ctcd_part1_sweep.sh
#
# Optional env:
#   CTCD_COMPLETION_DIR   default: completed_runs/ctcd_part1
#   SEED MAX_STEPS EPOCHS DEBUG FORCE TIMESTAMP  forwarded by run_training.sh
#
# Array: 576 tasks (indices 0–575). The "%N" suffix is SLURM's concurrent cap for
# this array only N tasks run at once; the rest stay pending until a slot frees.
# Match N to your partition / account limit (e.g. %4 if you may only run 4 GPUs).

#SBATCH --job-name=ctcd_p1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-575%4
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

CONFIG_ROOT="configs/CTCD_Experiments"
TASKS=(mnli qqp qnli cola mrpc rte)
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

TID=${SLURM_ARRAY_TASK_ID:-0}
PER_GRID=16
TOTAL=$((6 * 6 * PER_GRID))

if [[ "$TID" -lt 0 ]] || [[ "$TID" -ge "$TOTAL" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$TID out of range [0, $((TOTAL - 1))]"
    exit 1
fi

TASK_IDX=$((TID / (6 * PER_GRID)))
REM=$((TID % (6 * PER_GRID)))
ABL_IDX=$((REM / PER_GRID))
GRID_IDX=$((REM % PER_GRID))

LR_IDX=$((GRID_IDX / 4))
REM4=$((GRID_IDX % 4))
BATCH_IDX=$((REM4 / 2))
CLIP_IDX=$((REM4 % 2))

TASK="${TASKS[$TASK_IDX]}"
ABL_FILE=$(ablation_yaml "$ABL_IDX") || {
    echo "Bad ablation index"
    exit 1
}
LR="${LRS[$LR_IDX]}"
BS="${BATCHES[$BATCH_IDX]}"
CLIP="${CLIPS[$CLIP_IDX]}"

BASE_CFG="${CONFIG_ROOT}/${TASK}/${ABL_FILE}"
# Filesystem-friendly run name (avoid raw scientific notation / dots in tags)
lr_tag=${LR//./p}
lr_tag=${lr_tag//-/m}
clip_tag=${CLIP//./p}
RUN_NAME="p1_${TASK}_a$(printf '%02d' "$ABL_IDX")_lr${lr_tag}_b${BS}_c${clip_tag}"

TMP_OVR=$(mktemp "${TMPDIR:-/tmp}/ctcd_ovr_${SLURM_JOB_ID:-local}_${TID}.XXXXXX.yaml")
TMP_MERGED=$(mktemp "${TMPDIR:-/tmp}/ctcd_merged_${SLURM_JOB_ID:-local}_${TID}.XXXXXX.yaml")
trap 'rm -f "$TMP_OVR" "$TMP_MERGED"' EXIT

if [[ ! -f "$BASE_CFG" ]]; then
    echo "Missing config: $BASE_CFG"
    exit 1
fi

cat >"$TMP_OVR" <<EOF
training:
  learning_rate: ${LR}
  batch_size: ${BS}
  max_grad_norm: ${CLIP}
EOF

python scripts/slurm/merge_yaml_overrides.py "$BASE_CFG" "$TMP_OVR" "$TMP_MERGED"

COMPLETION_DIR="${CTCD_COMPLETION_DIR:-completed_runs/ctcd_part1}"
mkdir -p "$COMPLETION_DIR"
DONE="${COMPLETION_DIR}/${RUN_NAME}.done"
OUTPUT_DIR="./outputs/${RUN_NAME}"

if [[ -f "$DONE" ]]; then
    echo "Skip (marker): $RUN_NAME"
    slurm_print_footer
    exit 0
fi
if [[ -d "$OUTPUT_DIR" ]] && [[ -f "$OUTPUT_DIR/training_state.json" ]]; then
    if [[ "$(slurm_is_complete "$OUTPUT_DIR")" == "true" ]]; then
        echo "Skip (output complete, heal marker): $RUN_NAME"
        touch "$DONE"
        slurm_print_footer
        exit 0
    fi
fi

echo "========================================="
echo "CTCD Part 1  tid=$TID  run=$RUN_NAME"
echo "Base: $BASE_CFG"
echo "lr=$LR  batch_size=$BS  max_grad_norm=$CLIP"
echo "Merged: $TMP_MERGED"
echo "========================================="

bash scripts/slurm/run_training.sh "$RUN_NAME" "$TMP_MERGED" &
slurm_wait_child $!
TRAIN_EXIT=$EXIT_CODE

if [[ $TRAIN_EXIT -eq 0 ]]; then
    if [[ -f "$OUTPUT_DIR/training_state.json" ]] && [[ "$(slurm_is_complete "$OUTPUT_DIR")" == "true" ]]; then
        touch "$DONE"
        echo "Marked complete: $DONE"
    else
        # Training was interrupted; run_training.sh auto-resubmits a standalone continuation job.
        # Exit 1 so this array task is not falsely marked done. On re-run of the sweep array,
        # slurm_is_complete will detect the checkpoint and skip cleanly.
        echo "Training exited 0 but output not pipeline-complete; continuation job submitted."
        slurm_print_footer
        exit 1
    fi
else
    echo "Training failed for $RUN_NAME (exit $TRAIN_EXIT)"
    slurm_print_footer
    exit 1
fi

slurm_print_footer
