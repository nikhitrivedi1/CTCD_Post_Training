"""Training modules for knowledge distillation."""

from .callbacks import SLURMSignalCallback
from .losses import (
    DistillationLossStrategy,
    KLDivergenceLoss,
    MSELogitsLoss,
    CosineEmbeddingLoss,
    SoftTargetLoss,
    CombinedLoss,
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
    "KLDivergenceLoss",
    "MSELogitsLoss",
    "CosineEmbeddingLoss",
    "SoftTargetLoss",
    "CombinedLoss",
    "LossOutput",
    "get_loss_strategy",
    "LOSS_REGISTRY",
    # Callbacks
    "SLURMSignalCallback",
]
