"""Incomplete jobs registry with file locking for concurrent SLURM access."""

import contextlib
import json
import logging
import os
import subprocess
from pathlib import Path

import fcntl

logger = logging.getLogger(__name__)

_INCOMPLETE_JOBS = Path(".incomplete_jobs")


@contextlib.contextmanager
def locked_incomplete_jobs():
    """Context manager that locks, loads, and atomically saves .incomplete_jobs.

    Yields a mutable dict. Changes made to it are written back on exit.
    Uses fcntl.flock to serialize concurrent access across SLURM shards.

    Usage:
        with locked_incomplete_jobs() as jobs:
            jobs[key] = {"type": "bench", ...}
    """
    lock = _INCOMPLETE_JOBS.with_suffix(".lock")
    lock.touch(exist_ok=True)
    with open(lock, "r") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if _INCOMPLETE_JOBS.exists():
                try:
                    jobs = json.loads(_INCOMPLETE_JOBS.read_text())
                except json.JSONDecodeError:
                    jobs = {}
            else:
                jobs = {}
            yield jobs
            tmp = _INCOMPLETE_JOBS.with_suffix(f".tmp.{os.getpid()}")
            tmp.write_text(json.dumps(jobs, indent=2))
            os.replace(tmp, _INCOMPLETE_JOBS)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def get_job_key() -> str:
    """Get unique key for current job (SLURM_JOB_ID or local_{pid})."""
    return os.environ.get("SLURM_JOB_ID") or f"local_{os.getpid()}"


def _job_is_active(job_id: str) -> bool:
    """Check if a SLURM job is still in the queue (running or pending)."""
    try:
        result = subprocess.run(
            ["squeue", "-j", job_id, "--noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def guard_duplicate_job(jobs: dict, field: str, value: str, current_key: str):
    """Raise if another active SLURM job is already registered for the same run.

    Call this inside a locked_incomplete_jobs() context before writing a new entry.

    Args:
        jobs: Current .incomplete_jobs dict.
        field: Field to match on ("resume_path" or "output_dir").
        value: Value to match.
        current_key: Current job key (to skip self-match).

    Raises:
        RuntimeError: If a duplicate active job is found.
    """
    prev_job_id = os.environ.get("PREV_JOB_ID")
    for job_id, info in jobs.items():
        if job_id == current_key or job_id == prev_job_id:
            continue
        if info.get(field) == value and _job_is_active(job_id):
            raise RuntimeError(
                f"Duplicate job detected: job {job_id} is already running "
                f"for {field}={value}. Cancel it first or use --force."
            )
