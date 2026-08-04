"""Tests for src/utils/checkpoint.py

Covers TrainingState round-trip, is_complete(), corrupt JSON recovery,
get_resume_checkpoint(), mark_checkpoint_complete(), is_pipeline_complete(),
and mark_pipeline_complete().

Run:
    python -m pytest tests/unit/test_checkpoint.py -v
"""

import json

import pytest

from src.utils.checkpoint import TrainingState, mark_checkpoint_complete

# ===========================================================================
# 1. Save/load round-trip
# ===========================================================================


class TestRoundTrip:
    """Save then load preserves completed_epochs and completed_steps."""

    def test_completed_epochs_preserved(self, tmp_path):
        ts1 = TrainingState(str(tmp_path))
        ts1.save(epoch=5, step=2500)
        ts2 = TrainingState(str(tmp_path))
        assert ts2.state["completed_epochs"] == 5

    def test_completed_steps_preserved(self, tmp_path):
        ts1 = TrainingState(str(tmp_path))
        ts1.save(epoch=2, step=1200)
        ts2 = TrainingState(str(tmp_path))
        assert ts2.state["completed_steps"] == 1200

    def test_targets_preserved(self, tmp_path):
        ts1 = TrainingState(str(tmp_path))
        ts1.set_targets(epochs=10, steps=5000)
        ts2 = TrainingState(str(tmp_path))
        assert ts2.state["target_epochs"] == 10
        assert ts2.state["target_steps"] == 5000

    def test_jobs_list_preserved(self, tmp_path):
        ts1 = TrainingState(str(tmp_path))
        ts1.save(epoch=1, step=100)
        ts1.save(epoch=2, step=200)
        ts2 = TrainingState(str(tmp_path))
        assert len(ts2.state["jobs"]) == 2

    def test_fresh_state_defaults(self, tmp_path):
        ts = TrainingState(str(tmp_path))
        assert ts.state["completed_epochs"] == 0
        assert ts.state["completed_steps"] == 0
        assert ts.state["target_epochs"] is None
        assert ts.state["jobs"] == []


# ===========================================================================
# 2. is_complete()
# ===========================================================================


class TestIsComplete:
    """True when target reached, False before."""

    @pytest.mark.parametrize(
        "current,target,expected",
        [
            (2, 3, False),
            (3, 3, True),
            (5, 3, True),
        ],
        ids=["before", "at-target", "beyond"],
    )
    def test_epoch_target(self, tmp_path, current, target, expected):
        ts = TrainingState(str(tmp_path))
        ts.set_targets(epochs=target)
        ts.save(epoch=current, step=0)
        assert ts.is_complete() is expected

    @pytest.mark.parametrize(
        "current,target,expected",
        [
            (500, 1000, False),
            (1000, 1000, True),
            (1500, 1000, True),
        ],
        ids=["before", "at-target", "beyond"],
    )
    def test_step_target(self, tmp_path, current, target, expected):
        ts = TrainingState(str(tmp_path))
        ts.set_targets(steps=target)
        ts.save(epoch=0, step=current)
        assert ts.is_complete() is expected

    def test_no_target_returns_false(self, tmp_path):
        ts = TrainingState(str(tmp_path))
        ts.save(epoch=999, step=999999)
        assert ts.is_complete() is False

    def test_epoch_target_checked_before_step(self, tmp_path):
        """When both targets set, epoch target is checked first."""
        ts = TrainingState(str(tmp_path))
        ts.set_targets(epochs=3, steps=10000)
        ts.save(epoch=3, step=100)  # epoch met, steps not
        assert ts.is_complete() is True


# ===========================================================================
# 3. Corrupt JSON recovery
# ===========================================================================


class TestCorruptRecovery:
    """Corrupt JSON recovers gracefully to fresh state."""

    @pytest.mark.parametrize(
        "content",
        [
            "{truncated",
            "",
            "not json at all",
        ],
        ids=["truncated", "empty", "garbage"],
    )
    def test_corrupt_falls_back_to_defaults(self, tmp_path, content):
        (tmp_path / "training_state.json").write_text(content)
        ts = TrainingState(str(tmp_path))
        assert ts.state["completed_epochs"] == 0
        assert ts.state["completed_steps"] == 0


# ===========================================================================
# 4. get_resume_checkpoint()
# ===========================================================================


class TestGetResumeCheckpoint:
    """Finds latest numeric checkpoint dir, skips non-numeric."""

    def test_no_checkpoints_returns_none(self, tmp_path):
        ts = TrainingState(str(tmp_path))
        assert ts.get_resume_checkpoint() is None

    def test_finds_latest_numeric_checkpoint(self, tmp_path):
        for step in [100, 500, 200]:
            (tmp_path / f"checkpoint-{step}").mkdir()
        ts = TrainingState(str(tmp_path))
        assert "checkpoint-500" in ts.get_resume_checkpoint()

    def test_skips_non_numeric_suffix(self, tmp_path):
        (tmp_path / "checkpoint-100").mkdir()
        (tmp_path / "checkpoint-best").mkdir()
        (tmp_path / "final_model").mkdir()
        ts = TrainingState(str(tmp_path))
        result = ts.get_resume_checkpoint()
        assert "checkpoint-100" in result

    def test_skips_files_named_checkpoint(self, tmp_path):
        """A file (not dir) named checkpoint-500 must be skipped."""
        (tmp_path / "checkpoint-200").mkdir()
        (tmp_path / "checkpoint-500").write_text("not a directory")
        ts = TrainingState(str(tmp_path))
        assert "checkpoint-200" in ts.get_resume_checkpoint()

    def test_latest_checkpoint_file_preferred(self, tmp_path):
        """latest_checkpoint file takes precedence over dir globbing."""
        (tmp_path / "checkpoint-100").mkdir()
        (tmp_path / "checkpoint-500").mkdir()
        (tmp_path / "latest_checkpoint").write_text(str(tmp_path / "checkpoint-100"))
        ts = TrainingState(str(tmp_path))
        assert "checkpoint-100" in ts.get_resume_checkpoint()

    def test_stale_latest_file_falls_back_to_glob(self, tmp_path):
        (tmp_path / "checkpoint-300").mkdir()
        (tmp_path / "latest_checkpoint").write_text(str(tmp_path / "checkpoint-999"))
        ts = TrainingState(str(tmp_path))
        assert "checkpoint-300" in ts.get_resume_checkpoint()

    def test_output_dir_parameter(self, tmp_path):
        """get_resume_checkpoint accepts an explicit output_dir parameter."""
        alt_dir = tmp_path / "alt_output"
        alt_dir.mkdir()
        (alt_dir / "checkpoint-42").mkdir()
        ts = TrainingState(str(tmp_path))
        result = ts.get_resume_checkpoint(output_dir=str(alt_dir))
        assert "checkpoint-42" in result

    def test_relative_path_in_latest_checkpoint(self, tmp_path):
        """latest_checkpoint with relative path resolves against output_dir."""
        (tmp_path / "checkpoint-77").mkdir()
        (tmp_path / "latest_checkpoint").write_text("checkpoint-77")
        ts = TrainingState(str(tmp_path))
        result = ts.get_resume_checkpoint()
        assert "checkpoint-77" in result


# ===========================================================================
# 5. mark_checkpoint_complete()
# ===========================================================================


class TestMarkCheckpointComplete:
    """Writes atomically, no temp file leftovers."""

    def test_creates_latest_checkpoint_file(self, tmp_path):
        ckpt = tmp_path / "checkpoint-500"
        ckpt.mkdir()
        mark_checkpoint_complete(str(ckpt))
        latest = tmp_path / "latest_checkpoint"
        assert latest.exists()
        assert str(ckpt) in latest.read_text()

    def test_overwrites_previous(self, tmp_path):
        for step in [100, 200]:
            (tmp_path / f"checkpoint-{step}").mkdir()
            mark_checkpoint_complete(str(tmp_path / f"checkpoint-{step}"))
        assert "checkpoint-200" in (tmp_path / "latest_checkpoint").read_text()

    def test_no_temp_file_leftover(self, tmp_path):
        ckpt = tmp_path / "checkpoint-500"
        ckpt.mkdir()
        mark_checkpoint_complete(str(ckpt))
        assert list(tmp_path.glob("latest_checkpoint.tmp.*")) == []


# ===========================================================================
# 6. is_pipeline_complete() and mark_pipeline_complete()
# ===========================================================================


class TestPipelineComplete:
    """Pipeline completion tracking."""

    def test_initially_not_complete(self, tmp_path):
        ts = TrainingState(str(tmp_path))
        assert ts.is_pipeline_complete() is False

    def test_mark_pipeline_complete_sets_flag(self, tmp_path):
        ts = TrainingState(str(tmp_path))
        ts.mark_pipeline_complete()
        assert ts.is_pipeline_complete() is True

    def test_pipeline_complete_persists_across_reload(self, tmp_path):
        ts1 = TrainingState(str(tmp_path))
        ts1.mark_pipeline_complete()
        ts2 = TrainingState(str(tmp_path))
        assert ts2.is_pipeline_complete() is True

    def test_pipeline_complete_written_to_file(self, tmp_path):
        ts = TrainingState(str(tmp_path))
        ts.mark_pipeline_complete()
        with open(tmp_path / "training_state.json") as f:
            data = json.load(f)
        assert data["pipeline_complete"] is True
