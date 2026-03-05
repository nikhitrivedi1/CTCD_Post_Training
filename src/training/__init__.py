"""Training modules for knowledge distillation."""

from .callbacks import SLURMSignalCallback
from .losses import (
    DistillationLossStrategy,
    KLDistillationLoss,
    MSEDistillationLoss,
    CosineDistillationLoss,
    SoftTargetDistillationLoss,
    CombinedDistillationLoss,
    FeatureLossStrategy,
    AttentionLossStrategy,
    LossOutput,
    get_loss_strategy,
    LOSS_REGISTRY,
)
from .layer_mapping import (
    UniformMappingStrategy,
    LastNMappingStrategy,
    FirstLastMappingStrategy,
    CustomMappingStrategy,
    get_layer_mapping,
    LAYER_MAPPING_REGISTRY,
)
from .projections import (
    IdentityProjection,
    LinearProjection,
    MLPProjection,
    get_projection,
    create_projection_if_needed,
    PROJECTION_REGISTRY,
)
from .trainer import FinetuningTrainer, DistillationTrainer

__all__ = [
    # Trainers
    "FinetuningTrainer",
    "DistillationTrainer",
    # Loss strategies
    "DistillationLossStrategy",
    "KLDistillationLoss",
    "MSEDistillationLoss",
    "CosineDistillationLoss",
    "SoftTargetDistillationLoss",
    "CombinedDistillationLoss",
    "FeatureLossStrategy",
    "AttentionLossStrategy",
    "LossOutput",
    "get_loss_strategy",
    "LOSS_REGISTRY",
    # Layer mapping
    "UniformMappingStrategy",
    "LastNMappingStrategy",
    "FirstLastMappingStrategy",
    "CustomMappingStrategy",
    "get_layer_mapping",
    "LAYER_MAPPING_REGISTRY",
    # Projections
    "IdentityProjection",
    "LinearProjection",
    "MLPProjection",
    "get_projection",
    "create_projection_if_needed",
    "PROJECTION_REGISTRY",
    # Callbacks
    "SLURMSignalCallback",
]
