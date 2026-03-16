"""Tests for chain orchestrator job completion detection.

Verifies that the chain correctly distinguishes between:
- Jobs that completed successfully (sacct 0:0, not in IJ)
- Jobs that resubmitted via SIGUSR1 (sacct 0:0, still in IJ with new job ID)

The bug: bench/score jobs have chain run_names (e.g., fmt_lora_qwen35-9b_bench-olmocr_s0)
that don't match the output_dir in .incomplete_jobs (e.g., outputs/bench/olmocr/fmt_lora_qwen35-9b).
find_incomplete_job fails to match, so the chain falls through to sacct which sees 0:0
(clean SIGUSR1 exit) and falsely marks the step complete.

Fix: find_incomplete_job should also match by guard_key (output_dir:shard) constructed
from the chain step's env vars.
"""

import json
import os
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Helpers that mirror the chain's inline Python logic
# ---------------------------------------------------------------------------


def find_incomplete_job(ij_path, run_name, guard_key=""):
    """FIXED version of find_incomplete_job() in run_chain.sh.

    Args:
        ij_path: Path to .incomplete_jobs JSON file.
        run_name: Chain step's run_name.
        guard_key: Optional guard_key (output_dir or output_dir:shard) for
            bench/score jobs whose run_name doesn't match IJ entries.

    Returns:
        Job ID string if found, None otherwise.
    """
    if not os.path.exists(ij_path):
        return None

    with open(ij_path) as f:
        try:
            jobs = json.load(f)
        except json.JSONDecodeError:
            return None

    for job_id, info in jobs.items():
        # Training jobs: match by resume_path
        rp = info.get("resume_path", "").rstrip("/")
        if rp == run_name:
            return job_id
        # Bench jobs: match by output_dir (ends with run_name)
        od = info.get("output_dir", "").rstrip("/")
        if od.endswith(f"/{run_name}") or od == f"outputs/{run_name}":
            return job_id
        # Guard key match for bench/score jobs
        if guard_key:
            gk = info.get("guard_key", "").rstrip("/")
            if gk == guard_key:
                return job_id

    return None


def build_guard_key(step_env):
    """Construct a guard_key from a chain step's env vars.

    Args:
        step_env: Dict of env vars from chain step (OUTPUT, SHARD, etc.)

    Returns:
        Guard key string, or empty string if OUTPUT not set.
    """
    output = step_env.get("OUTPUT", "")
    if not output:
        return ""
    shard = step_env.get("SHARD", "")
    if shard:
        return f"{output}:{shard}"
    return output


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ij_dir():
    """Temp directory for .incomplete_jobs files."""
    with tempfile.TemporaryDirectory() as d:
        yield d


def write_ij(ij_dir, jobs):
    """Write a .incomplete_jobs file."""
    path = os.path.join(ij_dir, ".incomplete_jobs")
    with open(path, "w") as f:
        json.dump(jobs, f)
    return path


# ---------------------------------------------------------------------------
# Tests: find_incomplete_job matching
# ---------------------------------------------------------------------------

class TestFindIncompleteJobTraining:
    """Training jobs match by resume_path."""

    def test_matches_training_job_by_resume_path(self, ij_dir):
        path = write_ij(ij_dir, {
            "12345": {
                "resume_path": "fmt_grpo_qwen35-9b",
                "output_dir": "outputs/fmt_grpo_qwen35-9b",
            }
        })
        assert find_incomplete_job(path, "fmt_grpo_qwen35-9b") == "12345"

    def test_no_match_for_different_run_name(self, ij_dir):
        path = write_ij(ij_dir, {
            "12345": {
                "resume_path": "fmt_grpo_qwen35-9b",
                "output_dir": "outputs/fmt_grpo_qwen35-9b",
            }
        })
        assert find_incomplete_job(path, "ocr_lora_qwen35-9b") is None


class TestFindIncompleteJobBenchByRunName:
    """Bench jobs where run_name happens to match output_dir suffix."""

    def test_matches_bench_by_output_dir_suffix(self, ij_dir):
        path = write_ij(ij_dir, {
            "12345": {
                "type": "formatting",
                "output_dir": "outputs/bench/formatting/fmt_lora_qwen35-9b",
            }
        })
        # This works when run_name matches the last path component
        assert find_incomplete_job(path, "fmt_lora_qwen35-9b") == "12345"


class TestFindIncompleteJobBenchByGuardKey:
    """Bench/score jobs where run_name doesn't match output_dir.

    This is the bug scenario: chain run_name is fmt_lora_qwen35-9b_bench-olmocr_s0
    but output_dir in IJ is outputs/bench/olmocr/fmt_lora_qwen35-9b.
    """

    def test_no_match_without_guard_key(self, ij_dir):
        """Without guard_key, bench step with mismatched run_name is not found."""
        path = write_ij(ij_dir, {
            "5084578": {
                "type": "olmocr",
                "output_dir": "outputs/bench/olmocr/fmt_lora_qwen35-9b",
                "shard": "1/3",
                "guard_key": "outputs/bench/olmocr/fmt_lora_qwen35-9b:1/3",
            }
        })
        # run_name doesn't match output_dir suffix
        assert find_incomplete_job(
            path, "fmt_lora_qwen35-9b_bench-olmocr_s0"
        ) is None

    def test_matches_with_guard_key(self, ij_dir):
        """With guard_key from step env, the resubmitted job is found."""
        path = write_ij(ij_dir, {
            "5084578": {
                "type": "olmocr",
                "output_dir": "outputs/bench/olmocr/fmt_lora_qwen35-9b",
                "shard": "1/3",
                "guard_key": "outputs/bench/olmocr/fmt_lora_qwen35-9b:1/3",
            }
        })
        assert find_incomplete_job(
            path,
            "fmt_lora_qwen35-9b_bench-olmocr_s0",
            guard_key="outputs/bench/olmocr/fmt_lora_qwen35-9b:1/3",
        ) == "5084578"

    def test_guard_key_no_shard(self, ij_dir):
        """Formatting bench has no shard, guard_key is just output_dir."""
        path = write_ij(ij_dir, {
            "5084416": {
                "type": "formatting",
                "output_dir": "outputs/bench/formatting/fmt_grpo_qwen35-9b",
                "shard": "",
                "guard_key": "outputs/bench/formatting/fmt_grpo_qwen35-9b",
            }
        })
        assert find_incomplete_job(
            path,
            "fmt_grpo_qwen35-9b_bench-formatting",
            guard_key="outputs/bench/formatting/fmt_grpo_qwen35-9b",
        ) == "5084416"

    def test_guard_key_wrong_shard_no_match(self, ij_dir):
        """Guard key with different shard should not match."""
        path = write_ij(ij_dir, {
            "5084578": {
                "type": "olmocr",
                "output_dir": "outputs/bench/olmocr/fmt_lora_qwen35-9b",
                "shard": "1/3",
                "guard_key": "outputs/bench/olmocr/fmt_lora_qwen35-9b:1/3",
            }
        })
        # Shard 2/3 should not match shard 1/3
        assert find_incomplete_job(
            path,
            "fmt_lora_qwen35-9b_bench-olmocr_s1",
            guard_key="outputs/bench/olmocr/fmt_lora_qwen35-9b:2/3",
        ) is None

    def test_multiple_shards_correct_match(self, ij_dir):
        """With multiple shards in IJ, guard_key picks the right one."""
        path = write_ij(ij_dir, {
            "100": {
                "type": "olmocr",
                "output_dir": "outputs/bench/olmocr/fmt_lora",
                "shard": "1/3",
                "guard_key": "outputs/bench/olmocr/fmt_lora:1/3",
            },
            "200": {
                "type": "olmocr",
                "output_dir": "outputs/bench/olmocr/fmt_lora",
                "shard": "2/3",
                "guard_key": "outputs/bench/olmocr/fmt_lora:2/3",
            },
            "300": {
                "type": "olmocr",
                "output_dir": "outputs/bench/olmocr/fmt_lora",
                "shard": "3/3",
                "guard_key": "outputs/bench/olmocr/fmt_lora:3/3",
            },
        })
        assert find_incomplete_job(
            path, "x_bench_s1", guard_key="outputs/bench/olmocr/fmt_lora:2/3"
        ) == "200"


class TestFindIncompleteJobEdgeCases:
    """Edge cases: missing file, empty file, corrupt JSON."""

    def test_missing_file(self, ij_dir):
        path = os.path.join(ij_dir, ".incomplete_jobs")
        assert find_incomplete_job(path, "anything") is None

    def test_empty_file(self, ij_dir):
        path = os.path.join(ij_dir, ".incomplete_jobs")
        with open(path, "w") as f:
            f.write("")
        assert find_incomplete_job(path, "anything") is None

    def test_empty_dict(self, ij_dir):
        path = write_ij(ij_dir, {})
        assert find_incomplete_job(path, "anything") is None


# ---------------------------------------------------------------------------
# Tests: build_guard_key
# ---------------------------------------------------------------------------

class TestBuildGuardKey:
    """Guard key construction from chain step env vars."""

    def test_output_with_shard(self):
        env = {"CHECKPOINT": "outputs/foo", "OUTPUT": "outputs/bench/olmocr/foo", "SHARD": "1/3"}
        assert build_guard_key(env) == "outputs/bench/olmocr/foo:1/3"

    def test_output_without_shard(self):
        env = {"CHECKPOINT": "outputs/foo", "OUTPUT": "outputs/bench/formatting/foo"}
        assert build_guard_key(env) == "outputs/bench/formatting/foo"

    def test_no_output(self):
        env = {"ADAPTER": "outputs/foo"}
        assert build_guard_key(env) == ""

    def test_empty_shard_ignored(self):
        env = {"OUTPUT": "outputs/bench/formatting/foo", "SHARD": ""}
        assert build_guard_key(env) == "outputs/bench/formatting/foo"


# ---------------------------------------------------------------------------
# Tests: the actual bug scenario end-to-end
# ---------------------------------------------------------------------------

class TestResubmitDetection:
    """End-to-end: chain should detect resubmitted bench jobs, not mark complete.

    Scenario: olmOCR bench shard 1/3 hits 8h time limit on job 5082079.
    run_bench.sh resubmits as job 5084578 and updates .incomplete_jobs.
    Chain polls: old job gone from squeue, new job in IJ.
    Without fix: find_incomplete_job misses it, sacct 0:0 -> falsely complete.
    With fix: find_incomplete_job matches by guard_key -> correctly active.
    """

    def test_resubmitted_bench_found_with_guard_key(self, ij_dir):
        """The fix: guard_key lets us find the resubmitted job."""
        path = write_ij(ij_dir, {
            "5084578": {
                "type": "olmocr",
                "output_dir": "outputs/bench/olmocr/fmt_lora_qwen35-9b",
                "shard": "1/3",
                "guard_key": "outputs/bench/olmocr/fmt_lora_qwen35-9b:1/3",
            }
        })
        step_env = {
            "CHECKPOINT": "outputs/fmt_lora_qwen35-9b",
            "OUTPUT": "outputs/bench/olmocr/fmt_lora_qwen35-9b",
            "SHARD": "1/3",
        }
        guard_key = build_guard_key(step_env)
        job_id = find_incomplete_job(
            path,
            "fmt_lora_qwen35-9b_bench-olmocr_s0",
            guard_key=guard_key,
        )
        assert job_id == "5084578"

    def test_completed_bench_not_in_ij(self, ij_dir):
        """Truly completed bench job: not in IJ, guard_key finds nothing."""
        path = write_ij(ij_dir, {})
        step_env = {
            "CHECKPOINT": "outputs/fmt_lora_qwen35-9b",
            "OUTPUT": "outputs/bench/olmocr/fmt_lora_qwen35-9b",
            "SHARD": "1/3",
        }
        guard_key = build_guard_key(step_env)
        job_id = find_incomplete_job(
            path,
            "fmt_lora_qwen35-9b_bench-olmocr_s0",
            guard_key=guard_key,
        )
        assert job_id is None

    def test_score_job_resubmit_detected(self, ij_dir):
        """Score jobs also use guard_key (output_dir only, no shard)."""
        path = write_ij(ij_dir, {
            "9999": {
                "type": "olmocr",
                "output_dir": "outputs/bench/olmocr/fmt_lora_qwen35-9b",
                "shard": "",
                "guard_key": "outputs/bench/olmocr/fmt_lora_qwen35-9b",
            }
        })
        step_env = {"OUTPUT": "outputs/bench/olmocr/fmt_lora_qwen35-9b"}
        guard_key = build_guard_key(step_env)
        job_id = find_incomplete_job(
            path,
            "fmt_lora_qwen35-9b_bench-olmocr_score",
            guard_key=guard_key,
        )
        assert job_id == "9999"


# ---------------------------------------------------------------------------
# Tests: chain decision flow simulation
# ---------------------------------------------------------------------------

# The chain's check_step_status logic:
#   1. Job in squeue?          -> ACTIVE
#   2. Job in IJ (find_incomplete_job)? -> RESUBMITTED (update tracked ID)
#   3. slurm_is_complete?      -> COMPLETE
#   4. sacct exit 0:0?         -> COMPLETE (fallback for bench without training_state)
#   5. else                    -> FAILED
#
# The bug: step 2 misses bench jobs, so SIGUSR1 exits (sacct 0:0) fall through
# to step 4 and falsely report COMPLETE.

ACTIVE = "active"
RESUBMITTED = "resubmitted"
COMPLETE = "complete"
FAILED = "failed"


def check_step_status(ij_path, run_name, tracked_job_id, guard_key="",
                      in_squeue=False, sacct_exit_0=False, slurm_complete=False):
    """Simulate the chain's check_step_status decision flow.

    Args:
        ij_path: Path to .incomplete_jobs file.
        run_name: Chain step's run_name.
        tracked_job_id: The job ID the chain is tracking.
        guard_key: Guard key for bench/score matching.
        in_squeue: Whether tracked_job_id is in squeue.
        sacct_exit_0: Whether sacct shows 0:0 for tracked_job_id.
        slurm_complete: Whether slurm_is_complete returns true.

    Returns:
        Tuple of (status, new_job_id).
    """
    if in_squeue:
        return ACTIVE, tracked_job_id

    ij_job = find_incomplete_job(ij_path, run_name, guard_key=guard_key)
    if ij_job:
        return RESUBMITTED, ij_job

    if slurm_complete:
        return COMPLETE, tracked_job_id

    if sacct_exit_0:
        return COMPLETE, tracked_job_id

    return FAILED, tracked_job_id


class TestChainDecisionFlow:
    """Simulate the full chain polling scenario for bench resubmits.

    Scenario: olmOCR shard 1/3 hits 8h wall time on job 5082079.
    run_bench.sh catches SIGUSR1, resubmits as job 5084578, updates IJ.
    Old job 5082079 exits cleanly (sacct 0:0).
    Chain polls: 5082079 not in squeue, sacct shows 0:0.
    """

    IJ_AFTER_RESUBMIT = {
        "5084578": {
            "type": "olmocr",
            "output_dir": "outputs/bench/olmocr/fmt_lora_qwen35-9b",
            "shard": "1/3",
            "guard_key": "outputs/bench/olmocr/fmt_lora_qwen35-9b:1/3",
        }
    }

    STEP_ENV = {
        "CHECKPOINT": "outputs/fmt_lora_qwen35-9b",
        "OUTPUT": "outputs/bench/olmocr/fmt_lora_qwen35-9b",
        "SHARD": "1/3",
    }

    def test_without_guard_key_falsely_completes(self, ij_dir):
        """Without guard_key, chain sees sacct 0:0 and falsely marks complete."""
        path = write_ij(ij_dir, self.IJ_AFTER_RESUBMIT)
        status, job_id = check_step_status(
            path,
            "fmt_lora_qwen35-9b_bench-olmocr_s0",
            tracked_job_id="5082079",
            guard_key="",  # no guard_key - old behavior
            in_squeue=False,  # old job left squeue
            sacct_exit_0=True,  # SIGUSR1 -> clean exit
        )
        # BUG: chain thinks the step completed, but it actually resubmitted
        assert status == COMPLETE
        assert job_id == "5082079"  # still tracking the dead job

    def test_with_guard_key_detects_resubmit(self, ij_dir):
        """With guard_key, chain finds the new job and keeps monitoring."""
        path = write_ij(ij_dir, self.IJ_AFTER_RESUBMIT)
        guard_key = build_guard_key(self.STEP_ENV)
        status, job_id = check_step_status(
            path,
            "fmt_lora_qwen35-9b_bench-olmocr_s0",
            tracked_job_id="5082079",
            guard_key=guard_key,
            in_squeue=False,
            sacct_exit_0=True,
        )
        # FIX: chain finds the resubmitted job via guard_key
        assert status == RESUBMITTED
        assert job_id == "5084578"  # tracking the new job now

    def test_truly_complete_bench_detected(self, ij_dir):
        """After bench finishes, IJ entry is removed. Chain correctly marks complete."""
        path = write_ij(ij_dir, {})  # bench cleaned up its IJ entry
        guard_key = build_guard_key(self.STEP_ENV)
        status, _ = check_step_status(
            path,
            "fmt_lora_qwen35-9b_bench-olmocr_s0",
            tracked_job_id="5084578",
            guard_key=guard_key,
            in_squeue=False,
            sacct_exit_0=True,
            slurm_complete=True,
        )
        assert status == COMPLETE

    def test_crashed_bench_detected(self, ij_dir):
        """Bench OOMs: not in squeue, not in IJ, sacct non-zero. Chain marks failed."""
        path = write_ij(ij_dir, {})
        guard_key = build_guard_key(self.STEP_ENV)
        status, _ = check_step_status(
            path,
            "fmt_lora_qwen35-9b_bench-olmocr_s0",
            tracked_job_id="5082079",
            guard_key=guard_key,
            in_squeue=False,
            sacct_exit_0=False,
            slurm_complete=False,
        )
        assert status == FAILED

    def test_still_running_in_squeue(self, ij_dir):
        """Job still in squeue - active regardless of IJ state."""
        path = write_ij(ij_dir, self.IJ_AFTER_RESUBMIT)
        guard_key = build_guard_key(self.STEP_ENV)
        status, job_id = check_step_status(
            path,
            "fmt_lora_qwen35-9b_bench-olmocr_s0",
            tracked_job_id="5084578",
            guard_key=guard_key,
            in_squeue=True,
        )
        assert status == ACTIVE
        assert job_id == "5084578"