"""
Knowledge Distillation Framework

A universal, config-driven framework for compressing neural networks
through knowledge distillation.

Supports:
- Any HuggingFace model pair (BERT, GPT, Llama, T5, etc.)
- Classification, generation, and translation tasks
- SLURM integration with automatic job chaining
- Cross-platform: Local, Google Colab, HPC clusters
"""

__version__ = "1.0.0"

__all__ = [
    "data",
    "models",
    "training",
    "evaluation",
    "utils",
]
