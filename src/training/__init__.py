"""Training modules for knowledge distillation."""

from .callbacks import SLURMSignalCallback, StepMetricsCallback
from .layer_mapping import (
    LAYER_MAPPING_REGISTRY,
    CustomMappingStrategy,
    FirstLastMappingStrategy,
    LastNMappingStrategy,
    UniformMappingStrategy,
    get_layer_mapping,
)
from .losses import (
    LOSS_REGISTRY,
    AttentionLossStrategy,
    CombinedDistillationLoss,
    CosineDistillationLoss,
    DistillationLossStrategy,
    FeatureLossStrategy,
    KLDistillationLoss,
    LossOutput,
    MSEDistillationLoss,
    SoftTargetDistillationLoss,
    get_loss_strategy,
)
from .projections import (
    PROJECTION_REGISTRY,
    IdentityProjection,
    LinearProjection,
    MLPProjection,
    create_projection_if_needed,
    get_projection,
)
from .trainer import DistillationTrainer, FinetuningTrainer

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
    "StepMetricsCallback",
]
