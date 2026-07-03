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

# My jobs with GRES info (sorted by job ID ascending)
alias sq='squeue -u $USER --sort=i -o "%10i %20a %20j %10T %15M %5D %25R %25b"'

# Watch my jobs (refresh every 5s)
sqw() { watch -n 5 "squeue -u $USER --sort=i -o '%10i %20a %20j %10T %15M %5D %25R %25b'"; }

# All GPU jobs with user, state, runtime, node, GRES (sorted by job ID ascending)
alias sqall='squeue -p gpu --sort=i -o "%10i %20u %10T %15M %5D %25R %25b"'

# GPU node availability
alias sgpu='sinfo -p gpu -N -o "%20N %25G %10t %15C %10m"'

# Estimated start times for pending jobs, converted to UTC.
# Timezone offset is derived from the SLURM controller's config timestamp.
#   seta          -> all my pending jobs
#   seta 12345    -> single job detail
seta() {
    # Derive cluster timezone from SLURM config. Parse the same timestamp
    # as-if-UTC and as-local: the difference reveals the offset exactly.
    local _cfg_ts _as_utc _as_local _off_s _abs _sign _hh _mm _tz_label
    _cfg_ts=$(scontrol show config 2>/dev/null | head -1 | grep -oP '\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}')
    _as_utc=$(TZ=UTC date -d "$_cfg_ts" +%s 2>/dev/null || echo 0)
    _as_local=$(date -d "$_cfg_ts" +%s 2>/dev/null || echo 0)
    _off_s=$(( _as_utc - _as_local ))
    if [ $_off_s -ge 0 ]; then _sign="+"; _abs=$_off_s
    else _sign="-"; _abs=$(( -_off_s )); fi
    _hh=$(( _abs / 3600 )); _mm=$(( (_abs % 3600) / 60 ))
    printf -v _tz_label "UTC%s%d:%02d" "$_sign" "$_hh" "$_mm"

    # Convert a SLURM local timestamp to UTC
    _to_utc() {
        local ts="$1"
        if [ -n "$ts" ] && [ "$ts" != "Unknown" ] && [ "$ts" != "N/A" ]; then
            date -u -d "$ts" '+%Y-%m-%d %H:%M UTC' 2>/dev/null || echo "$ts"
        else
            echo "Unknown"
        fi
    }

    if [ -n "$1" ]; then
        local _info _start _state _reason
        _info=$(scontrol show job "$1" 2>&1)
        _state=$(echo "$_info" | grep -oP 'JobState=\K\S+')
        _start=$(echo "$_info" | grep -oP 'StartTime=\K\S+')
        _reason=$(echo "$_info" | grep -oP 'Reason=\K\S+')
        echo "Job $1: $_state  Start: $(_to_utc "$_start")  Reason: $_reason"
    else
        echo "Cluster: $_tz_label"
        printf "%-10s %-20s %-22s %s\n" "JOBID" "NAME" "EST. START (UTC)" "REASON"
        printf "%-10s %-20s %-22s %s\n" "-----" "----" "----------------" "------"
        squeue -u "$USER" -t PENDING --sort=i -h -o "%i|%j|%S|%r" \
            | while IFS='|' read -r jid name start reason; do
                printf "%-10s %-20s %-22s %s\n" "$jid" "$name" "$(_to_utc "$start")" "$reason"
            done
    fi

    unset -f _to_utc
}

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

# Resolve log file by job ID (or most recent if no ID given)
_slog_file() {
    if [ -n "$1" ]; then
        ls -t "$_FORGE_ROOT"/logs/*"${1}"*.out 2>/dev/null | head -1
    else
        ls -t "$_FORGE_ROOT"/logs/*.out 2>/dev/null | head -1
    fi
}

# Tail job log (last N lines, default 50)
#   stail           -> last 50 lines of most recent log
#   stail 12345     -> last 50 lines of job 12345
#   stail 100       -> last 100 lines of most recent log (if no matching job log)
#   stail 12345 100 -> last 100 lines of job 12345
#   stail f         -> tail -f most recent log
#   stail 12345 f   -> tail -f job 12345
stail() {
    local jobid="" lines="50" follow=false
    # Parse args: stail [jobid] [lines|f]
    if [ $# -eq 2 ]; then
        jobid="$1"
        if [ "$2" = "f" ]; then follow=true; else lines="$2"; fi
    elif [ $# -eq 1 ]; then
        if [ "$1" = "f" ]; then
            follow=true
        elif [[ "$1" =~ ^[0-9]+$ ]] && [ ! -f "$(_slog_file "$1")" ]; then
            # Numeric arg but no matching job log: treat as line count
            lines="$1"
        else
            jobid="$1"
        fi
    fi
    local logfile="$(_slog_file "$jobid")"
    if [ -z "$logfile" ] || [ ! -f "$logfile" ]; then
        echo "No log found${jobid:+: logs/*${jobid}*.out}" >&2
        return 1
    fi
    if $follow; then
        tail -f "$logfile"
    else
        tail -n "$lines" "$logfile"
    fi
}

# Head of job log (first N lines, default 50)
#   shead           -> first 50 lines of most recent log
#   shead 12345     -> first 50 lines of job 12345
#   shead 100       -> first 100 lines of most recent log
#   shead 12345 100 -> first 100 lines of job 12345
shead() {
    local jobid="" lines="50"
    if [ $# -eq 2 ]; then
        jobid="$1"; lines="$2"
    elif [ $# -eq 1 ]; then
        if [[ "$1" =~ ^[0-9]+$ ]] && [ ! -f "$(_slog_file "$1")" ]; then
            lines="$1"
        else
            jobid="$1"
        fi
    fi
    local logfile="$(_slog_file "$jobid")"
    if [ -z "$logfile" ] || [ ! -f "$logfile" ]; then
        echo "No log found${jobid:+: logs/*${jobid}*.out}" >&2
        return 1
    fi
    head -n "$lines" "$logfile"
}

# View full job stdout
slog() {
    if [ -z "$1" ]; then echo "Usage: slog <job_id>" >&2; return 1; fi
    local logfile="$(_slog_file "$1")"
    if [ -z "$logfile" ] || [ ! -f "$logfile" ]; then
        echo "No log found: logs/*${1}*.out" >&2
        return 1
    fi
    cat "$logfile"
}

# View job stderr (training jobs only, bench/data combine stderr into .out)
serr() {
    if [ -z "$1" ]; then echo "Usage: serr <job_id>" >&2; return 1; fi
    local logfile="$(ls -t "$_FORGE_ROOT"/logs/*"${1}"*.err 2>/dev/null | head -1)"
    if [ -z "$logfile" ] || [ ! -f "$logfile" ]; then
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

# Auto-pick the account with the best fairshare and set it as default
sbest() {
    local current best_acct best_score

    current=$(sacctmgr show user "$USER" withassoc format=DefaultAccount%30 \
              --parsable2 --noheader | head -1 | xargs)

    local share_data
    share_data=$(sshare -u "$USER" -U -p --noheader)

    if [ -z "$share_data" ]; then
        echo "Could not read fairshare data" >&2
        return 1
    fi

    # Pick account with highest FairShare
    best_acct=$(echo "$share_data" \
        | awk -F'|' 'NF>=7 && $7+0 > max { max=$7+0; acct=$1 } END { print acct }')

    if [ -z "$best_acct" ]; then
        echo "Could not determine best account" >&2
        return 1
    fi

    # Show ranking
    echo "Account ranking by FairShare:"
    echo "$share_data" \
        | awk -F'|' 'NF>=7 { printf "  %-30s  %.4f\n", $1, $7 }' \
        | sort -t' ' -k2 -rn

    local best_score
    best_score=$(echo "$share_data" \
        | awk -F'|' -v acct="$best_acct" '$1==acct { printf "%.4f", $7 }')

    echo ""
    if [ "$best_acct" = "$current" ]; then
        echo "Already on best account: $best_acct ($best_score)"
        return 0
    fi

    echo "Switching: $current -> $best_acct ($best_score)"
    sacctmgr -i modify user "$USER" set DefaultAccount="$best_acct"
}

# Switch one or more jobs to the best fairshare account
#   sbj 12345 12346
sbj() {
    if [ $# -eq 0 ]; then echo "Usage: sbj <job_id> [job_id ...]" >&2; return 1; fi

    local best_acct
    best_acct=$(sshare -u "$USER" -U -p --noheader \
        | awk -F'|' 'NF>=7 && $7+0 > max { max=$7+0; acct=$1 } END { print acct }')

    if [ -z "$best_acct" ]; then
        echo "Could not determine best account" >&2
        return 1
    fi

    echo "Best account: $best_acct"
    for jid in "$@"; do
        scontrol update JobId="$jid" Account="$best_acct" 2>&1 \
            | sed "s/^/  $jid: /" || true
        echo "  $jid: -> $best_acct"
    done
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
