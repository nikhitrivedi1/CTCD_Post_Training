#!/usr/bin/env python
"""
Test script to verify the development environment is set up correctly.

Run this after installation to ensure all dependencies are working:
    python test_setup.py
"""

import sys
from typing import Tuple


def check_python_version() -> Tuple[bool, str]:
    """Check Python version is 3.10+"""
    version = sys.version_info
    if version >= (3, 10):
        return True, f"✓ Python {version.major}.{version.minor}.{version.micro}"
    return False, f"✗ Python {version.major}.{version.minor} (need 3.10+)"


def check_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Test if a module can be imported"""
    display_name = package_name or module_name
    try:
        __import__(module_name)
        return True, f"✓ {display_name}"
    except ImportError as e:
        return False, f"✗ {display_name}: {str(e)}"


def check_pytorch_gpu() -> Tuple[bool, str]:
    """Check if PyTorch can detect GPU"""
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            return True, f"✓ CUDA available: {device}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True, "✓ MPS (Apple Silicon) available"
        else:
            return True, "⚠ CPU only (GPU not available)"
    except Exception as e:
        return False, f"✗ PyTorch GPU check failed: {str(e)}"


def check_qwen35_support() -> Tuple[bool, str]:
    """Check if transformers supports Qwen3.5 architecture natively"""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=False)
        return True, f"✓ Qwen3.5 (model_type={cfg.model_type})"
    except KeyError:
        return False, "✗ Qwen3.5: model_type not recognized (need transformers>=5.2.0)"
    except Exception as e:
        return False, f"✗ Qwen3.5: {str(e)}"


def check_framework_imports() -> Tuple[bool, str]:
    """Test custom framework imports"""
    try:
        from src.utils import detect_environment, load_config
        from src.data import get_handler, format_for_sft
        from src.models import load_model
        from src.training import FinetuningTrainer, DistillationTrainer

        return True, "✓ Framework modules"
    except Exception as e:
        return False, f"✗ Framework modules: {str(e)}"


def main():
    """Run all tests and report results"""
    print("=" * 60)
    print("Forge - Environment Test")
    print("=" * 60)
    print()

    # Core dependencies to check
    tests = [
        ("Python Version", check_python_version()),
        ("PyTorch", check_import("torch")),
        ("Transformers", check_import("transformers")),
        ("Datasets", check_import("datasets")),
        ("Evaluate", check_import("evaluate")),
        ("PEFT", check_import("peft")),
        ("Accelerate", check_import("accelerate")),
        ("WandB", check_import("wandb")),
        ("PyYAML", check_import("yaml", "PyYAML")),
        ("NumPy", check_import("numpy")),
        ("olmocr", check_import("olmocr")),
        ("SacreBLEU", check_import("sacrebleu")),
        ("Scikit-learn", check_import("sklearn", "scikit-learn")),
        ("PyTorch GPU", check_pytorch_gpu()),
        ("Qwen3.5 Support", check_qwen35_support()),
        ("Framework Modules", check_framework_imports()),
    ]

    # Print results
    all_passed = True
    for name, (passed, message) in tests:
        print(f"{name:.<30} {message}")
        if not passed:
            all_passed = False

    print()
    print("=" * 60)

    if all_passed:
        print("✓ All checks passed! Environment is ready.")
        print()
        print("Next steps:")
        print("  1. Configure W&B: wandb login")
        print("  2. Test training: python forge.py --config configs/classification/mnli/distilbert/bert-base/00_sft.yaml --max_steps 10")
        return 0
    else:
        print("✗ Some checks failed. Please install missing dependencies:")
        print("  conda env update --file environment.yml")
        return 1


if __name__ == "__main__":
    sys.exit(main())
