"""Task registry module."""

from .registry import (
    TASKS,
    get_num_labels,
    get_task_type,
    is_classification_task,
    is_translation_task,
    is_ocr_task,
    get_primary_metric,
)

__all__ = [
    "TASKS",
    "get_num_labels",
    "get_task_type",
    "is_classification_task",
    "is_translation_task",
    "is_ocr_task",
    "get_primary_metric",
]
