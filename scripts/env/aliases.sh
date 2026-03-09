#!/bin/bash
# SLURM shorthand aliases for Forge.
#
# Sourced automatically by activate.sh when on HPC (squeue in PATH).
# Wraps verbose squeue/sinfo/sacct/scancel invocations into short s* commands.

# Guard: only load on SLURM clusters
command -v squeue &>/dev/null || return 0

# ---- Resolve repo root (works from any cwd) ----
_FORGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"

# =========================================================================
# Monitoring
# =========================================================================

# My jobs with GRES info
alias sq='squeue -u $USER -o "%10i %20a %20j %10T %15M %5D %25R %25b"'

# Watch my jobs (refresh every 5s)
sqw() { watch -n 5 "squeue -u $USER -o '%10i %20a %20j %10T %15M %5D %25R %25b'"; }

# All GPU jobs with user, state, runtime, node, GRES
alias sqall='squeue -p gpu -o "%10i %20u %10T %15M %5D %25R %25b"'

# GPU node availability
alias sgpu='sinfo -p gpu -N -o "%20N %25G %10t %15C %10m"'

# H200 usage per user
alias sh200='squeue -p gpu -t RUNNING -o "%u %b" | grep h200 | awk "{print \$1}" | sort | uniq -c | sort -rn'

# =========================================================================
# Logs
# =========================================================================
# Log filenames vary by job type:
#   training:  logs/<jobid>.out / .err
#   bench:     logs/bench_<jobid>.out
#   data prep: logs/data_prep_<jobid>.out
# Functions glob for the job ID to handle all patterns.

# Tail job stdout (no arg = most recent log)
#   stail 12345   -> tail -f logs/*12345*.out
#   stail         -> tail -f logs/<latest>.out
stail() {
    local logfile
    if [ -n "$1" ]; then
        logfile="$(ls -t "$_FORGE_ROOT"/logs/*"${1}"*.out 2>/dev/null | head -1)"
    else
        logfile="$(ls -t "$_FORGE_ROOT"/logs/*.out 2>/dev/null | head -1)"
    fi
    if [ -z "$logfile" ] || [ ! -f "$logfile" ]; then
        echo "No log found${1:+: logs/*${1}*.out}" >&2
        return 1
    fi
    tail -f "$logfile"
}

# View job stdout
slog() {
    if [ -z "$1" ]; then echo "Usage: slog <job_id>" >&2; return 1; fi
    local logfile="$(ls -t "$_FORGE_ROOT"/logs/*"${1}"*.out 2>/dev/null | head -1)"
    if [ -z "$logfile" ]; then
        echo "No log found: logs/*${1}*.out" >&2
        return 1
    fi
    cat "$logfile"
}

# View job stderr (training jobs only, bench/data combine stderr into .out)
serr() {
    if [ -z "$1" ]; then echo "Usage: serr <job_id>" >&2; return 1; fi
    local logfile="$(ls -t "$_FORGE_ROOT"/logs/*"${1}"*.err 2>/dev/null | head -1)"
    if [ -z "$logfile" ]; then
        echo "No .err file for job $1 (stderr may be in .out)" >&2
        return 1
    fi
    cat "$logfile"
}

# =========================================================================
# Job Management
# =========================================================================

# Cancel a job
sc() {
    if [ -z "$1" ]; then echo "Usage: sc <job_id>" >&2; return 1; fi
    scancel "$1"
}

# Cancel all my jobs
alias scall='scancel -u $USER'

# Cancel only pending jobs
alias scpend='scancel -u $USER --state=PENDING'

# Job details (exit code, runtime, memory)
sjob() {
    if [ -z "$1" ]; then echo "Usage: sjob <job_id>" >&2; return 1; fi
    sacct -j "$1" --format=JobID%15,JobName%20,State%15,ExitCode,Elapsed,MaxRSS%10,NodeList%20
}

# =========================================================================
# Accounts & Fairshare
# =========================================================================

# List my accounts and default account
alias saccts='sacctmgr show user $USER withassoc format=Account%30,DefaultAccount%30'

# Fairshare scores (higher = better priority)
alias sfair='sshare -u $USER -U'

# Switch account for a pending job
#   sswitch 12345 cs6140.202630
sswitch() {
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "Usage: sswitch <job_id> <account>" >&2
        return 1
    fi
    scontrol update JobId="$1" Account="$2"
}

# Set default account for all future submissions
#   sdefault cs6140.202630
sdefault() {
    if [ -z "$1" ]; then echo "Usage: sdefault <account>" >&2; return 1; fi
    sacctmgr modify user "$USER" set DefaultAccount="$1"
}

# SSH into the node running a job
snode() {
    local node
    if [ -z "$1" ]; then echo "Usage: snode <job_id>" >&2; return 1; fi
    node=$(squeue -j "$1" -h -o "%N" 2>/dev/null)
    if [ -z "$node" ]; then
        echo "Job $1 not running or not found" >&2
        return 1
    fi
    echo "Connecting to $node (job $1)..."
    ssh "$node"
}
