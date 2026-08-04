"""Tests for src/utils/environment.py

Covers detect_environment(), get_device(), load_env(), get_num_workers(),
and get_gpu_info() with monkeypatched env vars and mocked torch backends.

Run:
    python -m pytest tests/unit/test_environment.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from src.utils.environment import (
    detect_environment,
    get_device,
    get_gpu_info,
    get_num_workers,
    load_env,
)

# ===========================================================================
# 1. detect_environment()
# ===========================================================================


class TestDetectEnvironment:
    """Returns 'slurm' when SLURM_JOB_ID set, 'colab' when COLAB_GPU set, 'local' otherwise."""

    @pytest.mark.parametrize(
        "slurm,colab,expected",
        [
            ("12345", None, "slurm"),
            (None, "1", "colab"),
            (None, None, "local"),
            ("99", "1", "slurm"),  # SLURM > Colab precedence
        ],
        ids=["slurm", "colab", "local", "slurm-over-colab"],
    )
    def test_detection(self, monkeypatch, slurm, colab, expected):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("COLAB_GPU", raising=False)
        if slurm is not None:
            monkeypatch.setenv("SLURM_JOB_ID", slurm)
        if colab is not None:
            monkeypatch.setenv("COLAB_GPU", colab)
        assert detect_environment() == expected


# ===========================================================================
# 2. get_device()
# ===========================================================================


class TestGetDevice:
    """Returns 'cuda', 'mps', or 'cpu' based on availability."""

    @pytest.mark.parametrize(
        "cuda,mps,expected",
        [
            (True, False, "cuda"),
            (False, True, "mps"),
            (False, False, "cpu"),
            (True, True, "cuda"),  # CUDA > MPS precedence
        ],
        ids=["cuda", "mps", "cpu", "cuda-over-mps"],
    )
    def test_device_selection(self, cuda, mps, expected):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = cuda
        mock_torch.backends.mps.is_available.return_value = mps
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert get_device() == expected

    def test_returns_valid_string(self):
        result = get_device()
        assert result in {"cuda", "mps", "cpu"}


# ===========================================================================
# 3. load_env()
# ===========================================================================


class TestLoadEnv:
    """load_env() returns True when .env exists, False otherwise."""

    def test_returns_true_when_env_exists(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("SOME_VAR=value\n")
        with patch("src.utils.environment.load_dotenv") as mock_load:
            with patch("src.utils.environment.Path") as mock_path:
                mock_instance = MagicMock()
                mock_instance.parent.parent.parent.__truediv__.return_value = env_file
                mock_path.return_value = mock_instance
                result = load_env()
                assert result is True
                mock_load.assert_called_once()

    def test_returns_false_when_no_env(self, tmp_path):
        with patch("src.utils.environment.load_dotenv") as mock_load:
            with patch("src.utils.environment.Path") as mock_path:
                mock_instance = MagicMock()
                nonexistent = tmp_path / ".env"
                mock_instance.parent.parent.parent.__truediv__.return_value = nonexistent
                mock_path.return_value = mock_instance
                result = load_env()
                assert result is False
                mock_load.assert_not_called()


# ===========================================================================
# 4. get_num_workers()
# ===========================================================================


class TestGetNumWorkers:
    """Worker count respects SLURM CPU allocation."""

    def test_with_slurm_cpus(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
        assert get_num_workers() == 7  # 8 - 1

    def test_without_slurm_falls_back_to_cpu_count(self, monkeypatch):
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        result = get_num_workers()
        assert result >= 1

    def test_minimum_one_worker(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
        assert get_num_workers() == 1  # max(1, 1-1) = 1


# ===========================================================================
# 5. get_gpu_info()
# ===========================================================================


class TestGetGpuInfo:
    """GPU info detection for different environments."""

    def test_slurm_returns_gpu_id(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("SLURM_JOB_GPUS", "0,1")
        monkeypatch.delenv("COLAB_GPU", raising=False)
        assert get_gpu_info() == "0,1"

    def test_slurm_no_gpus_returns_unknown(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
        monkeypatch.delenv("COLAB_GPU", raising=False)
        assert get_gpu_info() == "unknown"

    def test_local_returns_unknown(self, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("COLAB_GPU", raising=False)
        assert get_gpu_info() == "unknown"
