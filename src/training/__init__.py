"""Training modules for knowledge distillation."""

from .callbacks import SLURMSignalCallback
from .losses import (
    DistillationLossStrategy,
    KLDistillationLoss,
    MSEDistillationLoss,
    CosineDistillationLoss,
    SoftTargetDistillationLoss,
    CombinedDistillationLoss,
    LossOutput,
    get_loss_strategy,
    LOSS_REGISTRY,
)
from .trainer import DistillationTrainer

__all__ = [
    # Trainer
    "DistillationTrainer",
    # Loss strategies
    "DistillationLossStrategy",
    "KLDistillationLoss",
    "MSEDistillationLoss",
    "CosineDistillationLoss",
    "SoftTargetDistillationLoss",
    "CombinedDistillationLoss",
    "LossOutput",
    "get_loss_strategy",
    "LOSS_REGISTRY",
    # Callbacks
    "SLURMSignalCallback",
]
