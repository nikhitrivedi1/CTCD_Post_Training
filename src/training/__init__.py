"""Training modules for knowledge distillation."""

from .callbacks import SLURMSignalCallback
from .losses import distillation_loss, kl_divergence_loss
from .trainer import DistillationTrainer

__all__ = [
    "DistillationTrainer",
    "distillation_loss",
    "kl_divergence_loss",
    "SLURMSignalCallback",
]
