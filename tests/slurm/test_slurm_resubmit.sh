#!/bin/bash
# End-to-end test of SIGUSR1 -> .incomplete_jobs -> resubmit chain.
#
# Simulates 3 SLURM job iterations (0, 1, 2). Each iteration:
#   - Starts mock Python training
#   - Receives SIGUSR1 (self-sent in interactive mode, SLURM-sent in sbatch mode)
#   - Python writes .incomplete_jobs and exits cleanly
#   - Shell reads .incomplete_jobs and "resubmits" the next iteration
#   - Iteration 3: training is "complete", chain stops
#
# Works in two modes:
#   1. Interactive (OOD/local): Sends SIGUSR1 to itself after a delay
#   2. SLURM: Submit with sbatch, SLURM sends SIGUSR1 via --signal
#
# Usage:
#   bash tests/slurm/test_slurm_resubmit.sh        # interactive (OOD/local)
#   sbatch tests/slurm/test_slurm_resubmit.sh     # real SLURM job
#
# Clean up:
#   rm -f .incomplete_jobs .incomplete_jobs_test logs/resubmit_test_*.out
#
#SBATCH --job-name=resubmit-test
#SBATCH --partition=courses
#SBATCH --cpus-per-task=1
#SBATCH --time=00:00:15
#SBATCH --mem=1G
#SBATCH --output=logs/resubmit_test_%j.out
#SBATCH --signal=B:SIGUSR1@5

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

source scripts/slurm/slurm_essentials.sh

# Detect if this was submitted via sbatch (not just running on an OOD node).
# OOD sessions have SLURM_JOB_ID set but aren't sbatch-submitted scripts.
# Check SLURM_JOB_NAME matches our #SBATCH --job-name to distinguish.
# _FORGE_TEST_INTERACTIVE is set by inline resubmit to force interactive mode.
if [ "$SLURM_JOB_NAME" = "resubmit-test" ] && [ -z "$_FORGE_TEST_INTERACTIVE" ]; then
    MODE="slurm"
else
    MODE="interactive"
    export SLURM_JOB_ID="local_$$"
fi

mkdir -p logs
ITERATION=${_FORGE_TEST_ITERATION:-0}
MAX_ITERATION=3
TEST_INCOMPLETE="$REPO_DIR/.incomplete_jobs_test"

echo "========================================="
echo "Resubmit chain test"
echo "Mode: $MODE"
echo "Job ID: $SLURM_JOB_ID"
echo "Iteration: $ITERATION / $MAX_ITERATION"
echo "HAS_GPU: $HAS_GPU"
echo "========================================="
echo ""

# Final iteration: training "complete", stop the chain
if [ "$ITERATION" -ge "$MAX_ITERATION" ]; then
    echo "Iteration $ITERATION: training complete - stopping chain"
    echo ""
    echo "TEST PASSED: Full resubmit chain ($MAX_ITERATION iterations)"
    rm -f "$TEST_INCOMPLETE" .incomplete_jobs
    exit 0
fi

# Create mock Python training
MOCK_SCRIPT=$(mktemp /tmp/mock_training_XXXX.py)
cat > "$MOCK_SCRIPT" << PYEOF
"""Mock training that handles SIGUSR1."""
import json
import os
import signal
import sys
import time

INCOMPLETE_FILE = "$TEST_INCOMPLETE"
JOB_ID = os.environ.get("SLURM_JOB_ID", "unknown")
ITERATION = $ITERATION

caught = False

def handler(signum, frame):
    global caught
    caught = True
    print(f"Python: SIGUSR1 caught (iteration {ITERATION})")
    jobs = {}
    if os.path.exists(INCOMPLETE_FILE):
        with open(INCOMPLETE_FILE) as f:
            jobs = json.load(f)
    jobs[JOB_ID] = {
        "type": "training",
        "resume_path": f"test/resubmit/iteration_{ITERATION}",
        "output_dir": f"outputs/test/resubmit/iteration_{ITERATION}",
    }
    with open(INCOMPLETE_FILE, "w") as f:
        json.dump(jobs, f, indent=2)
    print("Python: wrote .incomplete_jobs")
    sys.stdout.flush()

signal.signal(signal.SIGUSR1, handler)

print(f"Python: training started (iteration {ITERATION}, job {JOB_ID})")
sys.stdout.flush()

# Runs up to 30s. In interactive mode, SIGUSR1 arrives at ~1s.
# In SLURM mode (15s limit, SIGUSR1@5), signal arrives at ~10s.
for step in range(60):
    time.sleep(0.5)
    if step % 20 == 0:
        print(f"Python: step {step}")
        sys.stdout.flush()
    if caught:
        print(f"Python: stopping at step {step}")
        sys.stdout.flush()
        break

print("Python: exiting gracefully")
sys.stdout.flush()
PYEOF

# Run mock training in background (mirrors run_training.sh pattern)
python3 "$MOCK_SCRIPT" &
PYTHON_PID=$!

# In interactive mode, send SIGUSR1 after 3 seconds
if [ "$MODE" = "interactive" ]; then
    (
        sleep 1
        # Send to Python first (so it can handle + write .incomplete_jobs)
        kill -SIGUSR1 $PYTHON_PID 2>/dev/null
        sleep 0.2
        # Send to shell (so wait is interrupted and trap fires)
        kill -SIGUSR1 $$ 2>/dev/null
    ) &
    TIMER_PID=$!
fi

# wait may be interrupted by SIGUSR1 - that's expected, wait again for Python
wait $PYTHON_PID 2>/dev/null || true
wait $PYTHON_PID 2>/dev/null
EXIT_CODE=$?
rm -f "$MOCK_SCRIPT"

# Kill timer if still running
[ -n "$TIMER_PID" ] && kill $TIMER_PID 2>/dev/null; wait $TIMER_PID 2>/dev/null

echo ""
echo "Shell: Python exited with code $EXIT_CODE"
echo "Shell: SIGUSR1_RECEIVED=$SIGUSR1_RECEIVED"

# Copy test incomplete file to where slurm_read_incomplete expects it
if [ -f "$TEST_INCOMPLETE" ]; then
    cp "$TEST_INCOMPLETE" .incomplete_jobs
fi

# Resubmit logic
NEXT_ITERATION=$((ITERATION + 1))
if [ $EXIT_CODE -eq 0 ] && slurm_read_incomplete; then
    if [ "$_JOB_TYPE" = "training" ] && [ -n "$_RESUME_PATH" ]; then
        echo ""
        echo "========================================="
        echo "Resubmit triggered!"
        echo "  Job type: $_JOB_TYPE"
        echo "  Resume path: $_RESUME_PATH"
        echo "  Next iteration: $NEXT_ITERATION / $MAX_ITERATION"

        if [ "$MODE" = "slurm" ]; then
            echo "  Action: sbatch $0"
            echo "========================================="
            _FORGE_TEST_ITERATION=$NEXT_ITERATION \
                PREV_JOB_ID="$SLURM_JOB_ID" \
                sbatch "$0"
        else
            echo "  Action: simulating sbatch (inline)"
            echo "========================================="
            echo ""
            # Clean up before next iteration
            rm -f .incomplete_jobs "$TEST_INCOMPLETE"
            # Run next iteration inline (force interactive mode with fresh job ID)
            _FORGE_TEST_ITERATION=$NEXT_ITERATION _FORGE_TEST_INTERACTIVE=1 \
                bash "$0"
        fi
    fi
else
    echo "Shell: No resubmit needed (exit=$EXIT_CODE)"
fi

# Clean up (only top-level call)
if [ "$ITERATION" -eq 0 ]; then
    rm -f "$TEST_INCOMPLETE" .incomplete_jobs
fi
