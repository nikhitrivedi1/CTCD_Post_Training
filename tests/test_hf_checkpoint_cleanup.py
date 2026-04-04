"""Tests for post-success removal of Hugging Face checkpoint-* directories."""

from pathlib import Path

from src.utils.checkpoint import LATEST_CHECKPOINT_FILE

from forge import _remove_hf_checkpoint_dirs_after_success


def test_remove_hf_checkpoint_dirs_keeps_run_root(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    ckpt = out / "checkpoint-42"
    ckpt.mkdir()
    (ckpt / "dummy.txt").write_text("x")
    (out / "config.yaml").write_text("k: v\n")
    teacher = out / "teacher"
    teacher.mkdir()
    (out / LATEST_CHECKPOINT_FILE).write_text(str(ckpt.resolve()))

    _remove_hf_checkpoint_dirs_after_success(out)

    assert not ckpt.exists()
    assert (out / "config.yaml").exists()
    assert teacher.exists()
    assert not (out / LATEST_CHECKPOINT_FILE).exists()


def test_remove_skips_non_numeric_checkpoint_name(tmp_path: Path) -> None:
    out = tmp_path / "run2"
    out.mkdir()
    weird = out / "checkpoint-foo"
    weird.mkdir()
    _remove_hf_checkpoint_dirs_after_success(out)
    assert weird.is_dir()
