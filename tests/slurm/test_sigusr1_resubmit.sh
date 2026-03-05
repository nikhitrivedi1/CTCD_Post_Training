#!/bin/bash
# Test SIGUSR1 handling and auto-resubmit flow
#
# Simulates the SLURM SIGUSR1 -> checkpoint -> resubmit cycle without
# needing SLURM, GPU, or real training. Tests that:
#   1. Shell trap survives SIGUSR1 (slurm_essentials.sh trap)
#   2. Python handles SIGUSR1 gracefully (writes .incomplete_jobs, exits 0)
#   3. Shell reads .incomplete_jobs via slurm_read_incomplete and would resubmit
#
# Usage:
#   bash tests/slurm/test_sigusr1_resubmit.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

TEST_DIR=$(mktemp -d)
MOCK_INCOMPLETE="$TEST_DIR/.incomplete_jobs"
LOG="$TEST_DIR/test.log"
PASS=0
FAIL=0

echo "Test dir: $TEST_DIR"
echo ""

# =========================================================================
# Test 1: Shell trap survives SIGUSR1
# =========================================================================
echo "=== Test 1: Shell trap survives SIGUSR1 ==="

cat > "$TEST_DIR/test_trap.sh" << 'SCRIPT'
#!/bin/bash
set -e
SIGUSR1_RECEIVED=0
trap 'SIGUSR1_RECEIVED=1; echo "TRAP: SIGUSR1 caught"' SIGUSR1

sleep 5 &
CHILD_PID=$!
wait $CHILD_PID 2>/dev/null || true

if [ "$SIGUSR1_RECEIVED" = "1" ]; then
    echo "RESULT: Shell survived SIGUSR1"
    exit 0
else
    echo "RESULT: SIGUSR1 never received"
    exit 1
fi
SCRIPT
chmod +x "$TEST_DIR/test_trap.sh"

bash "$TEST_DIR/test_trap.sh" > "$LOG" 2>&1 &
TEST_PID=$!
sleep 1
kill -SIGUSR1 $TEST_PID 2>/dev/null || true
wait $TEST_PID 2>/dev/null

if grep -q "Shell survived SIGUSR1" "$LOG"; then
    echo "PASS"; PASS=$((PASS + 1))
else
    echo "FAIL: Shell died or trap didn't fire"; FAIL=$((FAIL + 1))
    cat "$LOG"
fi
echo ""

# =========================================================================
# Test 2: Full SIGUSR1 -> checkpoint -> resubmit flow
# =========================================================================
echo "=== Test 2: Python SIGUSR1 handler + slurm_read_incomplete ==="

cat > "$TEST_DIR/mock_forge.py" << PYEOF
"""Mock forge.py that simulates SIGUSR1 handling."""
import json
import os
import signal
import sys
import time

INCOMPLETE_FILE = "$MOCK_INCOMPLETE"
FAKE_JOB_ID = os.environ.get("SLURM_JOB_ID", "local_mock")

caught = False

def handler(signum, frame):
    global caught
    caught = True
    print(f"Python: caught signal {signum}, saving state...")
    jobs = {}
    if os.path.exists(INCOMPLETE_FILE):
        with open(INCOMPLETE_FILE) as f:
            jobs = json.load(f)
    jobs[FAKE_JOB_ID] = {
        "type": "training",
        "resume_path": "classification/mnli/distilbert/00_sft/20260304-1200Z",
        "output_dir": "outputs/classification/mnli/distilbert/00_sft/20260304-1200Z",
    }
    with open(INCOMPLETE_FILE, "w") as f:
        json.dump(jobs, f, indent=2)
    print("Python: wrote .incomplete_jobs")

signal.signal(signal.SIGUSR1, handler)

print("Python: training started...")
for i in range(30):
    time.sleep(0.5)
    if caught:
        print(f"Python: stopping at step {i}")
        break

print("Python: exiting gracefully")
sys.exit(0)
PYEOF

# Shell wrapper that mirrors run_training.sh structure
cat > "$TEST_DIR/test_resubmit.sh" << WRAPEOF
#!/bin/bash
set -e

SIGUSR1_RECEIVED=0
trap 'SIGUSR1_RECEIVED=1; echo "SHELL: SIGUSR1 received"' SIGUSR1

export SLURM_JOB_ID="test_12345"

# Mock slurm_read_incomplete (same logic as slurm_essentials.sh)
slurm_read_incomplete() {
    _JOB_TYPE=""
    _RESUME_PATH=""
    _OUTPUT_DIR=""
    _SHARD=""
    [ -f "$MOCK_INCOMPLETE" ] || return 1
    eval "\$(python3 -c "
import json, shlex
try:
    jobs = json.load(open('$MOCK_INCOMPLETE'))
    info = jobs.get('\$SLURM_JOB_ID', {})
    print(f'_JOB_TYPE={shlex.quote(info.get(\"type\", \"\"))}')
    print(f'_RESUME_PATH={shlex.quote(info.get(\"resume_path\", \"\"))}')
    print(f'_OUTPUT_DIR={shlex.quote(info.get(\"output_dir\", \"\"))}')
    print(f'_SHARD={shlex.quote(info.get(\"shard\", \"\"))}')
except:
    pass
")"
    [ -n "\$_JOB_TYPE" ]
}

MOCK_INCOMPLETE="$MOCK_INCOMPLETE"

python3 "$TEST_DIR/mock_forge.py" || EXIT_CODE=\$?
EXIT_CODE=\${EXIT_CODE:-0}

echo "SHELL: Python exited with code \$EXIT_CODE"

if [ \$EXIT_CODE -eq 0 ] && slurm_read_incomplete; then
    if [ "\$_JOB_TYPE" = "training" ] && [ -n "\$_RESUME_PATH" ]; then
        echo "SHELL: Would resubmit with _FORGE_RESUME_PATH=\$_RESUME_PATH"
        echo "RESUBMIT_OK"
    else
        echo "SHELL: Wrong job type or missing resume_path"
        echo "RESUBMIT_FAIL"
    fi
else
    echo "SHELL: No resubmit needed"
fi
WRAPEOF
chmod +x "$TEST_DIR/test_resubmit.sh"

bash "$TEST_DIR/test_resubmit.sh" > "$LOG" 2>&1 &
WRAPPER_PID=$!
sleep 2
kill -SIGUSR1 $WRAPPER_PID 2>/dev/null || true
PYTHON_PID=$(pgrep -P $WRAPPER_PID 2>/dev/null || true)
if [ -n "$PYTHON_PID" ]; then
    kill -SIGUSR1 $PYTHON_PID 2>/dev/null || true
fi
wait $WRAPPER_PID 2>/dev/null

echo "--- Output ---"
cat "$LOG"
echo "--- End ---"
echo ""

if grep -q "RESUBMIT_OK" "$LOG"; then
    echo "PASS"; PASS=$((PASS + 1))
else
    echo "FAIL: Resubmit section did not execute"; FAIL=$((FAIL + 1))
fi
echo ""

# =========================================================================
# Test 3: .incomplete_jobs content
# =========================================================================
echo "=== Test 3: .incomplete_jobs content ==="
if [ -f "$MOCK_INCOMPLETE" ]; then
    if python3 -c "
import json, sys
jobs = json.load(open('$MOCK_INCOMPLETE'))
info = jobs.get('test_12345', {})
assert info.get('type') == 'training', f'wrong type: {info.get(\"type\")}'
assert info.get('resume_path') == 'classification/mnli/distilbert/00_sft/20260304-1200Z'
print('Content verified')
"; then
        echo "PASS"; PASS=$((PASS + 1))
    else
        echo "FAIL: content incorrect"; FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: .incomplete_jobs not created"; FAIL=$((FAIL + 1))
fi
echo ""

# =========================================================================
# Cleanup + summary
# =========================================================================
rm -rf "$TEST_DIR"

echo "=== Results: $PASS passed, $FAIL failed ==="
[ $FAIL -eq 0 ] || exit 1
