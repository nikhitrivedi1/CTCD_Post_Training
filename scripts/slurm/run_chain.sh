#!/bin/bash
#SBATCH --job-name=forge-chain
#SBATCH --partition=short
#SBATCH --time=48:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/chain_%j.out
#SBATCH --error=logs/chain_%j.out
#SBATCH --signal=B:SIGUSR1@120

# DAG-based SLURM job chain orchestrator.
#
# Submits and monitors a sequence of training/bench steps defined in a chain
# YAML file. Each step can depend on other steps (DAG). Steps with no
# dependencies start immediately; steps become ready when all deps complete.
#
# Runs in two modes (auto-detected):
#   Login node:  Run directly in tmux. No wall time limit.
#   CPU node:    sbatch to short partition. 48h max, auto-resubmits via SIGUSR1.
#
# Usage:
#   # Login node (in tmux):
#   scripts/slurm/run_chain.sh chains/formatting_pipeline.yaml
#
#   # CPU node:
#   sbatch scripts/slurm/run_chain.sh chains/formatting_pipeline.yaml
#
#   # Resume an incomplete chain (either mode):
#   scripts/slurm/run_chain.sh --resume formatting_pipeline
#
# See docs/planning/chain.md for full design details.

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
# Source essentials for slurm_is_complete and slurm_setup_env.
# When run on login node, SBATCH headers are ignored.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/slurm_essentials.sh"

slurm_setup_env

POLL_INTERVAL=${POLL_INTERVAL:-60}
MAX_RETRIES=${MAX_RETRIES:-3}
STALE_THRESHOLD=2
CHAIN_REGISTRY=".incomplete_chains"

# ---------------------------------------------------------------------------
# SIGUSR1 handling (SLURM mode only)
# ---------------------------------------------------------------------------
_CHAIN_NAME=""
_CHAIN_FILE=""
_CHAIN_SIGUSR1=0

if [ -n "${SLURM_JOB_ID:-}" ]; then
    slurm_print_header
    trap '
        _CHAIN_SIGUSR1=1
        echo ""
        echo "SIGUSR1 received by chain orchestrator ($(date -u +%H:%M:%S))"
        echo "Will save state and resubmit..."
    ' SIGUSR1
fi

# ---------------------------------------------------------------------------
# Python helpers (inline snippets)
# ---------------------------------------------------------------------------

# Validate chain YAML: structure, required fields, dependency references, cycles.
# Exits with error message on failure.
validate_chain() {
    python3 -c "
import sys, yaml

with open('$1') as f:
    chain = yaml.safe_load(f)

if not isinstance(chain, dict) or 'steps' not in chain:
    print('Error: chain YAML must have a top-level \"steps\" list', file=sys.stderr)
    sys.exit(1)

steps = chain['steps']
if not isinstance(steps, list) or len(steps) == 0:
    print('Error: \"steps\" must be a non-empty list', file=sys.stderr)
    sys.exit(1)

run_names = set()
valid_types = {'training', 'bench', 'score'}
errors = []

for i, step in enumerate(steps):
    prefix = f'Step {i}'
    if not isinstance(step, dict):
        errors.append(f'{prefix}: must be a dict')
        continue

    name = step.get('name', f'(unnamed step {i})')
    prefix = f'Step {i} ({name})'

    if 'run_name' not in step:
        errors.append(f'{prefix}: missing required field \"run_name\"')
    else:
        import re
        rn = step['run_name']
        if not re.match(r'^[a-zA-Z0-9_][a-zA-Z0-9_./-]*$', rn):
            errors.append(f'{prefix}: run_name \"{rn}\" contains invalid characters (use alphanumeric, _, -, ., /)')
        elif rn in run_names:
            errors.append(f'{prefix}: duplicate run_name \"{rn}\"')
        else:
            run_names.add(rn)

    if 'config' not in step:
        errors.append(f'{prefix}: missing required field \"config\"')

    step_type = step.get('type', 'training')
    if step_type not in valid_types:
        errors.append(f'{prefix}: invalid type \"{step_type}\" (must be one of {valid_types})')

    deps = step.get('depends_on', [])
    if isinstance(deps, str):
        deps = [deps]
    if not isinstance(deps, list):
        errors.append(f'{prefix}: depends_on must be a string or list')

# Check dependency references
all_run_names = {s['run_name'] for s in steps if 'run_name' in s}
for step in steps:
    deps = step.get('depends_on', [])
    if isinstance(deps, str):
        deps = [deps]
    for dep in deps:
        if dep not in all_run_names:
            errors.append(f'Step \"{step.get(\"name\", \"?\")}\": depends_on \"{dep}\" not found in chain')

# Cycle detection (topological sort)
adj = {}
in_degree = {}
for step in steps:
    rn = step.get('run_name', '')
    if not rn:
        continue
    adj.setdefault(rn, [])
    in_degree.setdefault(rn, 0)
    deps = step.get('depends_on', [])
    if isinstance(deps, str):
        deps = [deps]
    for dep in deps:
        adj.setdefault(dep, [])
        adj[dep].append(rn)
        in_degree[rn] = in_degree.get(rn, 0) + 1

queue = [n for n, d in in_degree.items() if d == 0]
visited = 0
while queue:
    node = queue.pop(0)
    visited += 1
    for child in adj.get(node, []):
        in_degree[child] -= 1
        if in_degree[child] == 0:
            queue.append(child)

if visited != len(in_degree):
    errors.append('Dependency cycle detected in chain')

if errors:
    print('Chain validation failed:', file=sys.stderr)
    for e in errors:
        print(f'  - {e}', file=sys.stderr)
    sys.exit(1)

print(f'Chain validated: {len(steps)} steps, {len([s for s in steps if not s.get(\"depends_on\")])} root(s)')
"
}

# Get total step count
chain_step_count() {
    python3 -c "
import yaml
with open('$1') as f:
    chain = yaml.safe_load(f)
print(len(chain['steps']))
"
}

# Read a chain step by index. Sets shell vars via eval.
# Returns 1 if index out of bounds.
read_chain_step() {
    local chain_file="$1" step_idx="$2"
    local _py_out
    _py_out=$(python3 -c "
import yaml, shlex, json

with open('$chain_file') as f:
    chain = yaml.safe_load(f)

steps = chain['steps']
idx = $step_idx
if idx < 0 or idx >= len(steps):
    exit(1)

step = steps[idx]
deps = step.get('depends_on', [])
if isinstance(deps, str):
    deps = [deps]
env = step.get('env', {})

print(f'_STEP_NAME={shlex.quote(step.get(\"name\", \"\"))}')
print(f'_STEP_RUN_NAME={shlex.quote(step.get(\"run_name\", \"\"))}')
print(f'_STEP_CONFIG={shlex.quote(step.get(\"config\", \"\"))}')
print(f'_STEP_TYPE={shlex.quote(step.get(\"type\", \"training\"))}')
print(f'_STEP_DEPS={shlex.quote(json.dumps(deps))}')
print(f'_STEP_ENV={shlex.quote(json.dumps(env))}')
") || return 1
    eval "$_py_out"
}

# Read all step run_names as newline-separated list
chain_run_names() {
    python3 -c "
import yaml
with open('$1') as f:
    chain = yaml.safe_load(f)
for step in chain['steps']:
    print(step['run_name'])
"
}

# ---------------------------------------------------------------------------
# .incomplete_chains registry
# ---------------------------------------------------------------------------

update_incomplete_chain() {
    local chain_name="$1" chain_file="$2" completed_json="$3" submitted_json="$4"
    local retries_json="$5"
    [ -z "$retries_json" ] && retries_json='{}'
    python3 -c "
import json, os, fcntl
from pathlib import Path

registry = Path('$CHAIN_REGISTRY')
lock = registry.with_suffix('.lock')
lock.touch(exist_ok=True)

with open(lock, 'r') as lf:
    fcntl.flock(lf, fcntl.LOCK_EX)
    try:
        if registry.exists():
            try:
                data = json.loads(registry.read_text())
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

        entry = {
            'chain_file': '$chain_file',
            'completed': json.loads('$completed_json'),
            'submitted': json.loads('$submitted_json'),
            'started': data.get('$chain_name', {}).get('started', '$(date -u +%Y-%m-%dT%H:%M:%S)'),
        }
        retries = json.loads('$retries_json')
        if retries:
            entry['retries'] = retries
        data['$chain_name'] = entry

        tmp = registry.with_suffix(f'.tmp.{os.getpid()}')
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, registry)
    finally:
        fcntl.flock(lf, fcntl.LOCK_UN)
"
}

remove_incomplete_chain() {
    local chain_name="$1"
    python3 -c "
import json, os, fcntl
from pathlib import Path

registry = Path('$CHAIN_REGISTRY')
lock = registry.with_suffix('.lock')
lock.touch(exist_ok=True)

with open(lock, 'r') as lf:
    fcntl.flock(lf, fcntl.LOCK_EX)
    try:
        if registry.exists():
            try:
                data = json.loads(registry.read_text())
            except json.JSONDecodeError:
                data = {}
            data.pop('$chain_name', None)
            if data:
                tmp = registry.with_suffix(f'.tmp.{os.getpid()}')
                tmp.write_text(json.dumps(data, indent=2))
                os.replace(tmp, registry)
            else:
                registry.unlink(missing_ok=True)
    finally:
        fcntl.flock(lf, fcntl.LOCK_UN)
"
}

read_incomplete_chain() {
    local chain_name="$1"
    local _py_out
    _py_out=$(python3 -c "
import json, shlex
from pathlib import Path

registry = Path('$CHAIN_REGISTRY')
if not registry.exists():
    exit(1)

try:
    data = json.loads(registry.read_text())
except json.JSONDecodeError:
    exit(1)

entry = data.get('$chain_name')
if not entry:
    exit(1)

print(f'_CHAIN_FILE={shlex.quote(entry[\"chain_file\"])}')
print(f'_CHAIN_COMPLETED={shlex.quote(json.dumps(entry.get(\"completed\", [])))}')
print(f'_CHAIN_SUBMITTED={shlex.quote(json.dumps(entry.get(\"submitted\", [])))}')
print(f'_CHAIN_RETRIES={shlex.quote(json.dumps(entry.get(\"retries\", {})))}')
") || return 1
    eval "$_py_out"
}

# ---------------------------------------------------------------------------
# Account selection (pick best fairshare)
# ---------------------------------------------------------------------------

pick_best_account() {
    local best_acct share_data

    share_data=$(sshare -u "$USER" -U -p --noheader 2>/dev/null) || true

    if [ -z "$share_data" ]; then
        echo "  Could not read fairshare data, skipping account switch"
        return 0
    fi

    best_acct=$(echo "$share_data" \
        | awk -F'|' 'NF>=7 && $7+0 > max { max=$7+0; acct=$1 } END { print acct }')

    if [ -z "$best_acct" ]; then
        echo "  Could not determine best account, skipping"
        return 0
    fi

    local current
    current=$(sacctmgr show user "$USER" withassoc format=DefaultAccount%30 \
              --parsable2 --noheader 2>/dev/null | head -1 | xargs) || true

    if [ "$best_acct" = "$current" ]; then
        echo "  Account: $best_acct (already best)"
    else
        echo "  Switching account: $current -> $best_acct"
        sacctmgr -i modify user "$USER" set DefaultAccount="$best_acct" 2>/dev/null || true
    fi
}

# ---------------------------------------------------------------------------
# Step submission
# ---------------------------------------------------------------------------

# Submit a step. Uses _STEP_* vars set by read_chain_step.
# Sets SUBMITTED_JOB_ID on success.
submit_step() {
    local run_name="$_STEP_RUN_NAME"
    local config="$_STEP_CONFIG"
    local step_type="$_STEP_TYPE"
    local env_json="$_STEP_ENV"

    # Check for existing job with this run_name (prevents duplicate submissions)
    local existing_job
    existing_job=$(find_incomplete_job "$run_name" 2>/dev/null) || true
    if [ -n "$existing_job" ] && job_in_squeue "$existing_job"; then
        echo ""
        echo "  Found existing job $existing_job for $run_name, adopting"
        SUBMITTED_JOB_ID="$existing_job"
        return 0
    fi

    echo ""
    echo "  Picking best account..."
    pick_best_account

    # Build --export args for sbatch from step env dict
    local export_args
    export_args=$(python3 -c "
import json
env = json.loads('$env_json')
if env:
    pairs = [f'{k}={v}' for k, v in env.items()]
    print(','.join(pairs))
" 2>/dev/null) || export_args=""

    local sbatch_args=""
    if [ -n "$export_args" ]; then
        sbatch_args="--export=ALL,$export_args"
    fi

    local job_output sbatch_cmd
    if [ "$step_type" = "score" ]; then
        sbatch_cmd="sbatch $sbatch_args scripts/slurm/run_score.sh $config"
    elif [ "$step_type" = "bench" ]; then
        sbatch_cmd="sbatch $sbatch_args scripts/slurm/run_bench.sh $config"
    else
        sbatch_cmd="sbatch $sbatch_args scripts/slurm/run_training.sh $run_name $config"
    fi

    # Log the sbatch command with env vars expanded inline
    local display_cmd="sbatch"
    if [ "$step_type" = "score" ]; then
        display_cmd="$display_cmd scripts/slurm/run_score.sh $config"
    elif [ "$step_type" = "bench" ]; then
        display_cmd="$display_cmd scripts/slurm/run_bench.sh $config"
    else
        display_cmd="$display_cmd scripts/slurm/run_training.sh $run_name $config"
    fi
    if [ -n "$export_args" ]; then
        echo "  CMD: $display_cmd"
        echo "  ENV: $export_args"
    else
        echo "  CMD: $display_cmd"
    fi
    STEP_CMD[$_STEP_RUN_NAME]="$display_cmd"
    STEP_ENV[$_STEP_RUN_NAME]="$export_args"
    job_output=$(eval "$sbatch_cmd" 2>&1)
    SUBMITTED_JOB_ID=$(echo "$job_output" | grep -oE '[0-9]+$' || echo "")

    if [ -z "$SUBMITTED_JOB_ID" ]; then
        echo "  ERROR: Failed to submit job"
        echo "  Output: $job_output"
        return 1
    fi

    echo "  Submitted job $SUBMITTED_JOB_ID for $run_name"
}

# ---------------------------------------------------------------------------
# Completion and failure detection
# ---------------------------------------------------------------------------

# Check if a run_name's .incomplete_jobs entry exists. Prints the job ID if found.
# Matches training jobs by resume_path and bench jobs by output_dir.
find_incomplete_job() {
    local run_name="$1"
    python3 -c "
import json
from pathlib import Path

ij = Path('.incomplete_jobs')
if not ij.exists():
    exit(1)

try:
    jobs = json.loads(ij.read_text())
except json.JSONDecodeError:
    exit(1)

for job_id, info in jobs.items():
    # Training jobs: match by resume_path
    rp = info.get('resume_path', '').rstrip('/')
    if rp == '$run_name':
        print(job_id)
        exit(0)
    # Bench jobs: match by output_dir (ends with run_name)
    od = info.get('output_dir', '').rstrip('/')
    if od.endswith('/$run_name') or od == 'outputs/$run_name':
        print(job_id)
        exit(0)

exit(1)
" 2>/dev/null
}

# Check if a SLURM job ID is still in the queue (running or pending)
job_in_squeue() {
    squeue -j "$1" --noheader 2>/dev/null | grep -q .
}

# Check step status. Returns via _STEP_STATUS: "active", "complete", "failed"
# _STEP_STALE_COUNT tracks consecutive polls where job is in .incomplete_jobs
# but not in squeue (dead job with stale registry entry).
declare -A _STEP_STALE_COUNT

check_step_status() {
    local run_name="$1"
    local tracked_job_id="$2"

    # 1. Is the tracked job still in squeue?
    if [ -n "$tracked_job_id" ] && job_in_squeue "$tracked_job_id"; then
        _STEP_STATUS="active"
        _STEP_TRACKED_JOB="$tracked_job_id"
        _STEP_STALE_COUNT[$run_name]=0
        return 0
    fi

    # 2. Check .incomplete_jobs for a resubmitted job
    local ij_job
    ij_job=$(find_incomplete_job "$run_name" 2>/dev/null) || true
    if [ -n "$ij_job" ]; then
        if job_in_squeue "$ij_job"; then
            _STEP_STATUS="active"
            _STEP_TRACKED_JOB="$ij_job"
            _STEP_STALE_COUNT[$run_name]=0
            return 0
        fi
        # Job in .incomplete_jobs but not in squeue. Could be:
        # a) Between resubmits (brief window) - give it a grace period
        # b) Crashed and left a stale entry - detect after STALE_THRESHOLD polls
        local stale=${_STEP_STALE_COUNT[$run_name]:-0}
        stale=$((stale + 1))
        _STEP_STALE_COUNT[$run_name]=$stale
        if [ "$stale" -lt "$STALE_THRESHOLD" ]; then
            _STEP_STATUS="active"
            _STEP_TRACKED_JOB="$ij_job"
            return 0
        fi
        # Stale for 2+ polls (~2 min). Job is dead.
        echo "  $run_name: stale .incomplete_jobs entry (job $ij_job gone from squeue for ${stale} polls)"
    fi

    # 3. No active job - check completion
    local complete
    complete=$(slurm_is_complete "./outputs/$run_name")
    if [ "$complete" = "true" ]; then
        _STEP_STATUS="complete"
        return 0
    fi

    # 4. Not active and not complete = failed
    _STEP_STATUS="failed"
    return 0
}

# Handle a step failure. Retries up to MAX_RETRIES, then halts.
# Returns 0 if retrying (step reset to pending), 1 if exhausted (halt chain).
handle_step_failure() {
    local rn="$1"
    local reason="$2"
    local retries=${STEP_RETRIES[$rn]:-0}
    retries=$((retries + 1))
    STEP_RETRIES[$rn]=$retries

    if [ "$retries" -le "$MAX_RETRIES" ]; then
        echo "  $rn: $reason (retry $retries/$MAX_RETRIES)"
        STEP_STATE[$rn]="pending"
        _STEP_STALE_COUNT[$rn]=0
        save_chain_state
        return 0
    else
        echo "  $rn: $reason (retries exhausted: $retries/$MAX_RETRIES)"
        STEP_STATE[$rn]="failed"
        echo ""
        echo "========================================="
        echo "CHAIN HALTED: step '$rn' failed after $MAX_RETRIES retries"
        echo "========================================="
        save_chain_state
        return 1
    fi
}

save_chain_state() {
    local completed_json submitted_json retries_json
    completed_json=$(python3 -c "
import json
states = dict()
$(for rn in "${!STEP_STATE[@]}"; do echo "states['$rn'] = '${STEP_STATE[$rn]}'"; done)
print(json.dumps([k for k, v in states.items() if v == 'complete']))
")
    submitted_json=$(python3 -c "
import json
states = dict()
$(for rn in "${!STEP_STATE[@]}"; do echo "states['$rn'] = '${STEP_STATE[$rn]}'"; done)
cmds = dict()
$(for rn in "${!STEP_CMD[@]}"; do echo "cmds['$rn'] = '${STEP_CMD[$rn]}'"; done)
envs = dict()
$(for rn in "${!STEP_ENV[@]}"; do echo "envs['$rn'] = '${STEP_ENV[$rn]}'"; done)
jobs = dict()
$(for rn in "${!STEP_JOB_ID[@]}"; do echo "jobs['$rn'] = '${STEP_JOB_ID[$rn]}'"; done)
submitted = {}
for k, v in states.items():
    if v == 'submitted':
        entry = {}
        if jobs.get(k): entry['job_id'] = jobs[k]
        if cmds.get(k): entry['cmd'] = cmds[k]
        if envs.get(k): entry['env'] = envs[k]
        submitted[k] = entry
print(json.dumps(submitted))
")
    retries_json=$(python3 -c "
import json
retries = dict()
$(for rn in "${!STEP_RETRIES[@]}"; do echo "retries['$rn'] = ${STEP_RETRIES[$rn]}"; done)
print(json.dumps({k: v for k, v in retries.items() if v > 0}))
")
    update_incomplete_chain "$_CHAIN_NAME" "$_CHAIN_FILE" "$completed_json" "$submitted_json" "$retries_json"
}

# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

# Parse arguments
RESUME_MODE=0
if [ "${1:-}" = "--resume" ]; then
    RESUME_MODE=1
    _CHAIN_NAME="${2:-}"
    if [ -z "$_CHAIN_NAME" ]; then
        echo "Error: --resume requires a chain name"
        echo "Usage: $0 --resume <chain_name>"
        exit 1
    fi
else
    _CHAIN_FILE="${1:-}"
    if [ -z "$_CHAIN_FILE" ]; then
        echo "Error: no chain file provided"
        echo "Usage: $0 <chain.yaml>        # new or auto-resume"
        echo "       $0 --resume <name>     # explicit resume"
        exit 1
    fi
    # Derive chain name from filename (without path/extension)
    _CHAIN_NAME=$(basename "$_CHAIN_FILE" .yaml)
    _CHAIN_NAME=$(basename "$_CHAIN_NAME" .yml)
    # Auto-resume: if .incomplete_chains has an entry for this chain, resume it
    # Use FORCE=1 to ignore existing state and start fresh
    if [ -z "${FORCE:-}" ] && read_incomplete_chain "$_CHAIN_NAME" 2>/dev/null; then
        RESUME_MODE=1
        echo "Found existing chain state for '$_CHAIN_NAME', auto-resuming (use FORCE=1 to start fresh)"
    fi
fi

echo "========================================="
echo "Chain Orchestrator: $_CHAIN_NAME"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "Mode: SLURM (job $SLURM_JOB_ID)"
else
    echo "Mode: Login node"
fi
echo "Start: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================="

# Resume: load state from .incomplete_chains
if [ $RESUME_MODE -eq 1 ]; then
    echo ""
    echo "Resuming chain: $_CHAIN_NAME"
    if ! read_incomplete_chain "$_CHAIN_NAME"; then
        echo "Error: no incomplete chain entry for '$_CHAIN_NAME'"
        exit 1
    fi
    echo "Chain file: $_CHAIN_FILE"
    echo "Previously completed: $_CHAIN_COMPLETED"
    echo "Previously submitted: $_CHAIN_SUBMITTED"
fi

# Validate chain
echo ""
echo "Validating chain..."
validate_chain "$_CHAIN_FILE"

# Load step count and all run_names
TOTAL_STEPS=$(chain_step_count "$_CHAIN_FILE")

# Initialize tracking arrays (bash associative arrays)
declare -A STEP_STATE      # run_name -> "pending"|"submitted"|"complete"|"failed"
declare -A STEP_JOB_ID     # run_name -> SLURM job ID
declare -A STEP_INDEX      # run_name -> index in chain YAML
declare -A STEP_RETRIES    # run_name -> number of retries so far
declare -A STEP_CMD        # run_name -> display command (script + args)
declare -A STEP_ENV        # run_name -> env vars string

# Build step index and initialize states
for ((i=0; i<TOTAL_STEPS; i++)); do
    read_chain_step "$_CHAIN_FILE" "$i"
    STEP_INDEX[$_STEP_RUN_NAME]=$i
    STEP_STATE[$_STEP_RUN_NAME]="pending"
    STEP_RETRIES[$_STEP_RUN_NAME]=0
done

# Restore state from resume
if [ $RESUME_MODE -eq 1 ]; then
    # Restore retry counts
    eval "$(echo "$_CHAIN_RETRIES" | python3 -c "
import json, sys
retries = json.load(sys.stdin)
for rn, count in retries.items():
    print(f'STEP_RETRIES[{rn}]={count}')
")"

    # Mark completed steps
    for rn in $(echo "$_CHAIN_COMPLETED" | python3 -c "import json,sys; [print(x) for x in json.load(sys.stdin)]"); do
        if [ -n "${STEP_STATE[$rn]+x}" ]; then
            STEP_STATE[$rn]="complete"
        fi
    done

    # Restore cmd/env from submitted entries (dict format from newer registry)
    eval "$(echo "$_CHAIN_SUBMITTED" | python3 -c "
import json, sys, shlex
data = json.load(sys.stdin)
if isinstance(data, dict):
    for rn, info in data.items():
        if isinstance(info, dict):
            cmd = info.get('cmd', '')
            env = info.get('env', '')
            print(f'STEP_CMD[{rn}]={shlex.quote(cmd)}')
            print(f'STEP_ENV[{rn}]={shlex.quote(env)}')
")"

    # Reconcile submitted steps (handles both list and dict formats)
    for rn in $(echo "$_CHAIN_SUBMITTED" | python3 -c "
import json, sys
data = json.load(sys.stdin)
keys = data.keys() if isinstance(data, dict) else data
for x in keys: print(x)
"); do
        if [ -z "${STEP_STATE[$rn]+x}" ]; then
            continue
        fi
        check_step_status "$rn" ""
        case "$_STEP_STATUS" in
            complete)
                echo "  $rn: completed while orchestrator was down"
                STEP_STATE[$rn]="complete"
                ;;
            active)
                echo "  $rn: still active (job $_STEP_TRACKED_JOB)"
                STEP_STATE[$rn]="submitted"
                STEP_JOB_ID[$rn]="$_STEP_TRACKED_JOB"
                ;;
            failed)
                if ! handle_step_failure "$rn" "failed during gap"; then
                    exit 1
                fi
                ;;
        esac
    done
fi

# Skip-ahead: mark already-complete steps
echo ""
echo "Checking for already-complete steps..."
for rn in "${!STEP_STATE[@]}"; do
    if [ "${STEP_STATE[$rn]}" = "pending" ]; then
        complete=$(slurm_is_complete "./outputs/$rn")
        if [ "$complete" = "true" ] && [ -f "./outputs/$rn/training_state.json" ]; then
            echo "  $rn: already complete, skipping"
            STEP_STATE[$rn]="complete"
        fi
    fi
done

# ---------------------------------------------------------------------------
# DAG execution loop
# ---------------------------------------------------------------------------

# Get deps for a run_name
get_step_deps() {
    local idx="${STEP_INDEX[$1]}"
    read_chain_step "$_CHAIN_FILE" "$idx"
    echo "$_STEP_DEPS"
}

# Check if all deps are complete
deps_met() {
    local run_name="$1"
    local deps_json
    deps_json=$(get_step_deps "$run_name")
    python3 -c "
import json
deps = json.loads('$deps_json')
states = dict()
$(for rn in "${!STEP_STATE[@]}"; do echo "states['$rn'] = '${STEP_STATE[$rn]}'"; done)
for dep in deps:
    if states.get(dep) != 'complete':
        exit(1)
exit(0)
"
}

# Count steps in each state
count_state() {
    local target="$1" count=0
    for rn in "${!STEP_STATE[@]}"; do
        [ "${STEP_STATE[$rn]}" = "$target" ] && ((count++)) || true
    done
    echo "$count"
}

echo ""
echo "========================================="
echo "Starting DAG execution ($TOTAL_STEPS steps)"
echo "========================================="

# Save initial state
save_chain_state

while true; do
    # Check for SIGUSR1 (SLURM mode resubmit)
    if [ "$_CHAIN_SIGUSR1" = "1" ]; then
        echo ""
        echo "========================================="
        echo "Saving state and resubmitting chain..."
        echo "========================================="
        save_chain_state
        sbatch "$0" --resume "$_CHAIN_NAME"
        echo "Resubmitted. Exiting."
        exit 0
    fi

    # Find and submit ready steps
    for rn in "${!STEP_STATE[@]}"; do
        if [ "${STEP_STATE[$rn]}" = "pending" ] && deps_met "$rn"; then
            local_idx="${STEP_INDEX[$rn]}"
            read_chain_step "$_CHAIN_FILE" "$local_idx"

            echo ""
            echo "-----------------------------------------"
            echo "Submitting: $_STEP_NAME ($rn)"
            echo "-----------------------------------------"

            if submit_step; then
                STEP_STATE[$rn]="submitted"
                STEP_JOB_ID[$rn]="$SUBMITTED_JOB_ID"
                save_chain_state
            else
                if ! handle_step_failure "$rn" "submission failed"; then
                    exit 1
                fi
            fi
        fi
    done

    # Check if we're done
    n_complete=$(count_state "complete")
    n_submitted=$(count_state "submitted")
    n_pending=$(count_state "pending")
    n_failed=$(count_state "failed")

    if [ "$n_complete" -eq "$TOTAL_STEPS" ]; then
        echo ""
        echo "========================================="
        echo "CHAIN COMPLETE: all $TOTAL_STEPS steps finished"
        echo "End: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        echo "========================================="
        remove_incomplete_chain "$_CHAIN_NAME"
        exit 0
    fi

    if [ "$n_submitted" -eq 0 ] && [ "$n_pending" -eq 0 ]; then
        echo ""
        echo "========================================="
        echo "CHAIN HALTED: no steps to submit or wait for"
        echo "Complete: $n_complete, Failed: $n_failed, Pending: $n_pending"
        echo "========================================="
        exit 1
    fi

    # Poll running steps
    echo ""
    echo "--- Polling ($n_complete/$TOTAL_STEPS complete, $n_submitted running, $n_pending pending) ---"

    sleep "$POLL_INTERVAL"

    # Check status of all submitted steps
    for rn in "${!STEP_STATE[@]}"; do
        if [ "${STEP_STATE[$rn]}" = "submitted" ]; then
            check_step_status "$rn" "${STEP_JOB_ID[$rn]:-}"
            case "$_STEP_STATUS" in
                complete)
                    echo "  $rn: COMPLETE"
                    STEP_STATE[$rn]="complete"
                    save_chain_state
                    ;;
                active)
                    STEP_JOB_ID[$rn]="$_STEP_TRACKED_JOB"
                    ;;
                failed)
                    if ! handle_step_failure "$rn" "job failed"; then
                        exit 1
                    fi
                    ;;
            esac
        fi
    done
done
