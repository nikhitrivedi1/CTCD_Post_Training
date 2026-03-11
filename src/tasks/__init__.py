"""Task registry module."""

from .registry import (
    TASK_REGISTRY,
    get_num_labels,
    get_task_type,
    is_classification_task,
    is_translation_task,
    is_ocr_task,
    get_primary_metric,
    get_default_reward,
)

__all__ = [
    "TASK_REGISTRY",
    "get_num_labels",
    "get_task_type",
    "is_classification_task",
    "is_translation_task",
    "is_ocr_task",
    "get_primary_metric",
    "get_default_reward",
]
