"""Environment detection for local/Colab/SLURM platforms."""

import os
from pathlib import Path

from dotenv import load_dotenv


def load_env():
    """
    Load environment variables from .env file if it exists.

    Uses override=False so existing env vars (from Colab userdata,
    SLURM modules, or shell) take precedence over .env file.

    Returns:
        bool: True if .env file was loaded, False otherwise
    """
    # Look for .env in project root (parent of src/)
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / ".env"

    if env_file.exists():
        load_dotenv(env_file, override=False)
        return True
    return False


# Auto-load .env when module is imported
_env_loaded = load_env()


def detect_environment():
    """
    Detect the current execution environment.

    Returns:
        str: One of 'slurm', 'colab', or 'local'
    """
    if os.environ.get("SLURM_JOB_ID"):
        return "slurm"
    elif os.environ.get("COLAB_GPU"):
        return "colab"
    else:
        return "local"


def get_device():
    """
    Get the appropriate device for PyTorch.

    Returns:
        str: One of 'cuda', 'mps', or 'cpu'
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_gpu_info():
    """
    Get GPU information based on environment.

    Returns:
        str: GPU information or 'unknown'
    """
    env = detect_environment()

    if env == "slurm":
        return os.environ.get("SLURM_JOB_GPUS", "unknown")

    # Could add nvidia-smi parsing here for more detailed info
    return "unknown"
