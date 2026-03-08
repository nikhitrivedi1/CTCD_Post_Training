#!/bin/bash
# Live end-to-end test of SIGUSR1 -> checkpoint -> resubmit -> resume.
#
# Runs real DistilBERT/MNLI training with a 3-minute SLURM time limit.
# SIGUSR1 fires at T-30s (~2.5 min of training), forcing a checkpoint save.
# The script then resubmits itself in resume mode, which picks up from
# the latest checkpoint and continues training until completion.
#
# What this validates:
#   1. SLURMSignalCallback catches SIGUSR1 and saves a checkpoint
#   2. forge.py writes .incomplete_jobs with the correct resume path
#   3. Shell detects incomplete training and resubmits
#   4. Resumed job finds checkpoint-N and resumes from that step
#   5. HF Trainer restores optimizer/scheduler/step state correctly
#   6. W&B run ID is preserved across job boundaries
#   7. Chain terminates when training completes
#
# Usage:
#   sbatch tests/slurm/test_resubmit_live.sh
#
# Monitor:
#   tail -f logs/<job_id>.out
#   cat outputs/resubmit_live_test/training_state.json
#   cat .incomplete_jobs
#
# Clean up:
#   rm -rf outputs/resubmit_live_test logs/resubmit_live_*.out
#
#SBATCH --job-name=resubmit-live
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:03:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/resubmit_live_%j.out
#SBATCH --error=logs/resubmit_live_%j.err
#SBATCH --signal=B:SIGUSR1@30

source scripts/slurm/slurm_essentials.sh
set -e

# 30s signal window (@30), kill after 15s, leaves 15s for resubmit
_FORGE_KILL_DEADLINE=15

slurm_setup_env
slurm_print_header

# Run Python in background so shell can forward SIGUSR1 to it.
# Wait loop: re-wait if interrupted by SIGUSR1 trap, otherwise capture exit code.
run_forge() {
    python forge.py "$@" &
    _FORGE_CHILD_PID=$!
    set +e
    while true; do
        wait $_FORGE_CHILD_PID 2>/dev/null
        EXIT_CODE=$?
        if [ "$EXIT_CODE" -eq 138 ] && [ "$SIGUSR1_RECEIVED" = "1" ]; then
            continue
        fi
        break
    done
    set -e
    _FORGE_CHILD_PID=""
    if [ -n "$_FORGE_KILL_TIMER_PID" ]; then
        kill $_FORGE_KILL_TIMER_PID 2>/dev/null
        wait $_FORGE_KILL_TIMER_PID 2>/dev/null
        _FORGE_KILL_TIMER_PID=""
    fi
    # If killed by our own kill timer (exit 137), treat as clean for resubmit
    if [ $EXIT_CODE -eq 137 ] && [ "$SIGUSR1_RECEIVED" = "1" ]; then
        echo "Python was force-killed by deadline timer, treating as clean exit"
        EXIT_CODE=0
    fi
    return $EXIT_CODE
}

CONFIG="configs/classification/mnli/distilbert/00_sft.yaml"
RUN_NAME="resubmit_live_test"
ITERATION=${_FORGE_TEST_ITERATION:-0}

echo ""
echo "========================================="
echo "LIVE RESUBMIT TEST"
echo "Iteration: $ITERATION"
echo "Config: $CONFIG"
echo "Run: $RUN_NAME"
echo "Time limit: 3 min, SIGUSR1 at T-30s"
echo "========================================="
echo ""

# Common args: debug logging
# MNLI has ~393k examples, ~24k steps/epoch at batch_size=16, so 3 epochs
# takes ~20+ min on V100, ensuring SIGUSR1 interrupts mid-training
ARGS="--debug"

if [ -n "$_FORGE_RESUME_PATH" ]; then
    # ---- Resumed job ----
    echo "RESUMING from: outputs/$_FORGE_RESUME_PATH"
    echo ""
    run_forge --resume "outputs/$_FORGE_RESUME_PATH" $ARGS || EXIT_CODE=$?
else
    # ---- First job ----
    # Check if a previous test run exists and is complete
    OUTPUT_DIR="./outputs/$RUN_NAME"
    COMPLETE=$(slurm_is_complete "$OUTPUT_DIR")
    if [ "$COMPLETE" = "true" ] && [ -f "$OUTPUT_DIR/training_state.json" ]; then
        echo "Previous test run already complete. Delete outputs/$RUN_NAME to re-test."
        exit 0
    fi

    # Force overwrite if partial previous run exists (test cleanup)
    if [ -d "$OUTPUT_DIR" ] && [ "$ITERATION" -eq 0 ]; then
        ARGS="$ARGS --force"
    fi

    run_forge --run_name "$RUN_NAME" --config "$CONFIG" $ARGS || EXIT_CODE=$?
fi

EXIT_CODE=${EXIT_CODE:-0}

# ---- Resubmit logic ----
if [ $EXIT_CODE -eq 0 ] && slurm_read_incomplete; then
    if [ "$_JOB_TYPE" = "training" ]; then
        COMPLETE=$(slurm_is_complete "$_OUTPUT_DIR")
        if [ "$COMPLETE" = "false" ]; then
            NEXT=$((ITERATION + 1))
            echo ""
            echo "========================================="
            echo "Training incomplete after iteration $ITERATION"
            echo "Resubmitting (iteration $NEXT)..."
            echo "Resume path: outputs/$_RESUME_PATH"
            echo "========================================="

            _FORGE_RESUME_PATH="$_RESUME_PATH" \
                _FORGE_TEST_ITERATION=$NEXT \
                PREV_JOB_ID="$SLURM_JOB_ID" \
                sbatch "$0"
        else
            echo ""
            echo "========================================="
            echo "Training COMPLETE after $((ITERATION + 1)) job(s)!"
            echo "========================================="
            echo ""
            echo "Verify:"
            echo "  cat outputs/$RUN_NAME/training_state.json"
            echo "  ls outputs/$RUN_NAME/checkpoint-*"
        fi
    fi
elif [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "========================================="
    echo "Job FAILED with exit code: $EXIT_CODE"
    echo "========================================="
fi

slurm_print_footer
