#!/usr/bin/env bash
# End-to-end tests for run_chain.sh chain orchestrator.
#
# Tests the Python helper functions (validate, read, save) and argument
# parsing without actually submitting SLURM jobs. Uses a temp directory
# for all artifacts.
#
# Usage:
#   bash tests/slurm/test_chain.sh
#
# Requires: bash 4+ (associative arrays), python3 with PyYAML

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# Test framework
# ---------------------------------------------------------------------------
PASS=0
FAIL=0
TESTS=()

assert_eq() {
    local desc="$1" expected="$2" actual="$3"
    if [ "$expected" = "$actual" ]; then
        PASS=$((PASS + 1))
        echo "  PASS: $desc"
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL: $desc"
        echo "    expected: $expected"
        echo "    actual:   $actual"
    fi
}

assert_ok() {
    local desc="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        PASS=$((PASS + 1))
        echo "  PASS: $desc"
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL: $desc (exit code $?)"
    fi
}

assert_fail() {
    local desc="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        FAIL=$((FAIL + 1))
        echo "  FAIL: $desc (expected failure, got success)"
    else
        PASS=$((PASS + 1))
        echo "  PASS: $desc"
    fi
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if ! python3 -c "import yaml" 2>/dev/null; then
    echo "SKIP: python3 PyYAML not available (activate conda env)"
    exit 0
fi

# Check bash version (need 4+ for associative arrays)
if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
    echo "SKIP: bash ${BASH_VERSION} too old (need 4+ for declare -A)"
    exit 0
fi

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

# Source chain functions (but override CHAIN_REGISTRY to use temp dir)
source scripts/slurm/slurm_essentials.sh 2>/dev/null || true
CHAIN_REGISTRY="$TMPDIR/.incomplete_chains"

# Extract function definitions from run_chain.sh.
# We can't source the whole file (it runs main logic).
eval "$(sed -n '/^validate_chain/,/^}/p' scripts/slurm/run_chain.sh)"
eval "$(sed -n '/^chain_step_count/,/^}/p' scripts/slurm/run_chain.sh)"
eval "$(sed -n '/^read_chain_step/,/^}/p' scripts/slurm/run_chain.sh)"
eval "$(sed -n '/^chain_run_names/,/^}/p' scripts/slurm/run_chain.sh)"
eval "$(sed -n '/^update_incomplete_chain/,/^}/p' scripts/slurm/run_chain.sh)"
eval "$(sed -n '/^remove_incomplete_chain/,/^}/p' scripts/slurm/run_chain.sh)"
eval "$(sed -n '/^read_incomplete_chain/,/^}/p' scripts/slurm/run_chain.sh)"

# Create test chain files
cat > "$TMPDIR/valid_chain.yaml" << 'EOF'
steps:
  - name: Step A
    run_name: step_a
    config: configs/a.yaml

  - name: Step B
    run_name: step_b
    depends_on: step_a
    config: configs/b.yaml
    env:
      ADAPTER: outputs/step_a

  - name: Step C bench
    run_name: step_c
    type: bench
    depends_on: step_b
    config: formatting
    env:
      CHECKPOINT: outputs/step_b
      OUTPUT: outputs/bench/formatting/step_b

  - name: Step D score
    run_name: step_d
    type: score
    depends_on: step_c
    config: formatting
    env:
      OUTPUT: outputs/bench/formatting/step_b
EOF

cat > "$TMPDIR/missing_run_name.yaml" << 'EOF'
steps:
  - name: Bad step
    config: configs/a.yaml
EOF

cat > "$TMPDIR/duplicate_names.yaml" << 'EOF'
steps:
  - name: Step A
    run_name: step_a
    config: configs/a.yaml
  - name: Step A copy
    run_name: step_a
    config: configs/b.yaml
EOF

cat > "$TMPDIR/bad_dep.yaml" << 'EOF'
steps:
  - name: Step A
    run_name: step_a
    depends_on: nonexistent
    config: configs/a.yaml
EOF

cat > "$TMPDIR/cycle.yaml" << 'EOF'
steps:
  - name: Step A
    run_name: step_a
    depends_on: step_b
    config: configs/a.yaml
  - name: Step B
    run_name: step_b
    depends_on: step_a
    config: configs/b.yaml
EOF

cat > "$TMPDIR/bad_type.yaml" << 'EOF'
steps:
  - name: Step A
    run_name: step_a
    type: invalid
    config: configs/a.yaml
EOF

cat > "$TMPDIR/sharded.yaml" << 'EOF'
steps:
  - name: Train
    run_name: train_a
    config: configs/a.yaml

  - name: Bench shard 0
    run_name: bench_s0
    type: bench
    depends_on: train_a
    config: omnidoc
    env:
      CHECKPOINT: outputs/train_a
      OUTPUT: outputs/bench/omnidoc/train_a
      SHARD: 0/3

  - name: Bench shard 1
    run_name: bench_s1
    type: bench
    depends_on: train_a
    config: omnidoc
    env:
      CHECKPOINT: outputs/train_a
      OUTPUT: outputs/bench/omnidoc/train_a
      SHARD: 1/3

  - name: Score
    run_name: bench_score
    type: score
    depends_on:
      - bench_s0
      - bench_s1
    config: omnidoc
    env:
      OUTPUT: outputs/bench/omnidoc/train_a
EOF

# =========================================================================
echo ""
echo "==========================================="
echo "Chain Orchestrator Tests"
echo "==========================================="

# ---------------------------------------------------------------------------
echo ""
echo "--- validate_chain ---"
# ---------------------------------------------------------------------------
assert_ok "valid chain passes validation" validate_chain "$TMPDIR/valid_chain.yaml"
assert_fail "missing run_name fails" validate_chain "$TMPDIR/missing_run_name.yaml"
assert_fail "duplicate run_names fails" validate_chain "$TMPDIR/duplicate_names.yaml"
assert_fail "bad dependency reference fails" validate_chain "$TMPDIR/bad_dep.yaml"
assert_fail "dependency cycle fails" validate_chain "$TMPDIR/cycle.yaml"
assert_fail "invalid type fails" validate_chain "$TMPDIR/bad_type.yaml"
assert_ok "sharded chain passes validation" validate_chain "$TMPDIR/sharded.yaml"
assert_fail "nonexistent file fails" validate_chain "$TMPDIR/nonexistent.yaml"

# ---------------------------------------------------------------------------
echo ""
echo "--- chain_step_count ---"
# ---------------------------------------------------------------------------
count=$(chain_step_count "$TMPDIR/valid_chain.yaml")
assert_eq "valid chain has 4 steps" "4" "$count"

count=$(chain_step_count "$TMPDIR/sharded.yaml")
assert_eq "sharded chain has 4 steps" "4" "$count"

# ---------------------------------------------------------------------------
echo ""
echo "--- read_chain_step ---"
# ---------------------------------------------------------------------------
read_chain_step "$TMPDIR/valid_chain.yaml" 0
assert_eq "step 0 name" "Step A" "$_STEP_NAME"
assert_eq "step 0 run_name" "step_a" "$_STEP_RUN_NAME"
assert_eq "step 0 config" "configs/a.yaml" "$_STEP_CONFIG"
assert_eq "step 0 type defaults to training" "training" "$_STEP_TYPE"
assert_eq "step 0 no deps" "[]" "$_STEP_DEPS"
assert_eq "step 0 no env" "{}" "$_STEP_ENV"

read_chain_step "$TMPDIR/valid_chain.yaml" 1
assert_eq "step 1 name" "Step B" "$_STEP_NAME"
assert_eq "step 1 depends_on" '["step_a"]' "$_STEP_DEPS"
env_adapter=$(echo "$_STEP_ENV" | python3 -c "import json,sys; print(json.load(sys.stdin)['ADAPTER'])")
assert_eq "step 1 env ADAPTER" "outputs/step_a" "$env_adapter"

read_chain_step "$TMPDIR/valid_chain.yaml" 2
assert_eq "step 2 type is bench" "bench" "$_STEP_TYPE"

read_chain_step "$TMPDIR/valid_chain.yaml" 3
assert_eq "step 3 type is score" "score" "$_STEP_TYPE"

assert_fail "out-of-bounds index fails" read_chain_step "$TMPDIR/valid_chain.yaml" 99

# Sharded step env
read_chain_step "$TMPDIR/sharded.yaml" 1
shard_val=$(echo "$_STEP_ENV" | python3 -c "import json,sys; print(json.load(sys.stdin)['SHARD'])")
assert_eq "shard env value preserved" "0/3" "$shard_val"
output_val=$(echo "$_STEP_ENV" | python3 -c "import json,sys; print(json.load(sys.stdin)['OUTPUT'])")
assert_eq "shard output shared across shards" "outputs/bench/omnidoc/train_a" "$output_val"

# ---------------------------------------------------------------------------
echo ""
echo "--- chain_run_names ---"
# ---------------------------------------------------------------------------
names=$(chain_run_names "$TMPDIR/valid_chain.yaml")
assert_eq "run_names list" "step_a
step_b
step_c
step_d" "$names"

# ---------------------------------------------------------------------------
echo ""
echo "--- update_incomplete_chain / read_incomplete_chain ---"
# ---------------------------------------------------------------------------

# Write a chain entry
update_incomplete_chain "test_chain" "chains/test.yaml" '["step_a"]' '["step_b"]' '{}'
assert_ok "registry file exists" test -f "$CHAIN_REGISTRY"

# Read it back
read_incomplete_chain "test_chain"
assert_eq "read chain_file" "chains/test.yaml" "$_CHAIN_FILE"
read_completed=$(echo "$_CHAIN_COMPLETED" | python3 -c "import json,sys; print(json.load(sys.stdin))")
assert_eq "read completed" "['step_a']" "$read_completed"
read_submitted=$(echo "$_CHAIN_SUBMITTED" | python3 -c "import json,sys; print(json.load(sys.stdin))")
assert_eq "read submitted" "['step_b']" "$read_submitted"

# Read nonexistent chain
assert_fail "read nonexistent chain fails" read_incomplete_chain "no_such_chain"

# Update existing entry (should merge, not duplicate)
update_incomplete_chain "test_chain" "chains/test.yaml" '["step_a", "step_b"]' '["step_c"]' '{"step_a": 1}'
read_incomplete_chain "test_chain"
read_completed=$(echo "$_CHAIN_COMPLETED" | python3 -c "import json,sys; print(len(json.load(sys.stdin)))")
assert_eq "updated completed count" "2" "$read_completed"
read_retries=$(echo "$_CHAIN_RETRIES" | python3 -c "import json,sys; print(json.load(sys.stdin).get('step_a', 0))")
assert_eq "retries preserved" "1" "$read_retries"

# Remove chain
remove_incomplete_chain "test_chain"
assert_fail "removed chain not found" read_incomplete_chain "test_chain"

# ---------------------------------------------------------------------------
echo ""
echo "--- read_incomplete_chain with bad input ---"
# ---------------------------------------------------------------------------

# Empty registry
rm -f "$CHAIN_REGISTRY"
assert_fail "read from missing registry fails" read_incomplete_chain "anything"

# Corrupt registry
echo "not json" > "$CHAIN_REGISTRY"
assert_fail "read from corrupt registry fails" read_incomplete_chain "anything"

# Valid registry, missing key
echo '{"other_chain": {"chain_file": "x.yaml", "completed": [], "submitted": []}}' > "$CHAIN_REGISTRY"
assert_fail "read missing chain name fails" read_incomplete_chain "test_chain"

# ---------------------------------------------------------------------------
echo ""
echo "--- save_chain_state with empty arrays ---"
# ---------------------------------------------------------------------------

# Simulate empty STEP_STATE (no steps loaded) - this was the bug
rm -f "$CHAIN_REGISTRY"
_CHAIN_NAME="empty_test"
_CHAIN_FILE="chains/test.yaml"
declare -A STEP_STATE
declare -A STEP_RETRIES

# This mimics save_chain_state logic
completed_json=$(python3 -c "
import json
states = dict()
$(for rn in "${!STEP_STATE[@]}"; do echo "states['$rn'] = '${STEP_STATE[$rn]}'"; done)
print(json.dumps([k for k, v in states.items() if v == 'complete']))
")
assert_eq "empty states produce valid json" "[]" "$completed_json"

submitted_json=$(python3 -c "
import json
states = dict()
$(for rn in "${!STEP_STATE[@]}"; do echo "states['$rn'] = '${STEP_STATE[$rn]}'"; done)
print(json.dumps([k for k, v in states.items() if v == 'submitted']))
")
assert_eq "empty submitted produce valid json" "[]" "$submitted_json"

# Now test update with these values
update_incomplete_chain "empty_test" "chains/test.yaml" "$completed_json" "$submitted_json" '{}'
assert_ok "update with empty arrays succeeds" read_incomplete_chain "empty_test"

# ---------------------------------------------------------------------------
echo ""
echo "--- chain name derivation ---"
# ---------------------------------------------------------------------------

# Test basename logic used in fresh mode
test_derive() {
    local input="$1"
    local name=$(basename "$input" .yaml)
    name=$(basename "$name" .yml)
    echo "$name"
}

assert_eq "simple yaml" "pipeline" "$(test_derive 'chains/pipeline.yaml')"
assert_eq "nested path" "qwen35-27b_ocr_pipeline" "$(test_derive 'chains/qwen35-27b_ocr_pipeline.yaml')"
assert_eq "yml extension" "pipeline" "$(test_derive 'chains/pipeline.yml')"
assert_eq "no path" "pipeline" "$(test_derive 'pipeline.yaml')"

# ---------------------------------------------------------------------------
echo ""
echo "--- argument parsing edge cases ---"
# ---------------------------------------------------------------------------

# Simulate what happens when user passes file path to --resume
# This was the bug: _CHAIN_NAME gets set to full path
bad_chain_name="chains/qwen35-27b_ocr_pipeline.yaml"
echo '{"qwen35-27b_ocr_pipeline": {"chain_file": "chains/test.yaml", "completed": [], "submitted": []}}' > "$CHAIN_REGISTRY"
if read_incomplete_chain "$bad_chain_name" 2>/dev/null; then
    FAIL=$((FAIL + 1))
    echo "  FAIL: resume with file path should fail (key mismatch)"
else
    PASS=$((PASS + 1))
    echo "  PASS: resume with file path correctly fails (key mismatch)"
fi

# ---------------------------------------------------------------------------
echo ""
echo "--- real pipeline validation ---"
# ---------------------------------------------------------------------------
if [ -f "chains/qwen35-27b_ocr_pipeline.yaml" ]; then
    assert_ok "production pipeline validates" validate_chain "chains/qwen35-27b_ocr_pipeline.yaml"
    count=$(chain_step_count "chains/qwen35-27b_ocr_pipeline.yaml")
    assert_eq "production pipeline has 20 steps" "20" "$count"
fi

if [ -f "chains/formatting_pipeline.yaml" ]; then
    assert_ok "formatting pipeline validates" validate_chain "chains/formatting_pipeline.yaml"
fi

# =========================================================================
echo ""
echo "==========================================="
echo "Results: $PASS passed, $FAIL failed"
echo "==========================================="

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
