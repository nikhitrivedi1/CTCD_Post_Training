"""Incomplete jobs registry with file locking for concurrent SLURM access."""

import contextlib
import fcntl
import json
import os
from pathlib import Path

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
