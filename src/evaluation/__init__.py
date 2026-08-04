"""Evaluation metrics and orchestration for classification, translation, and OCR tasks."""

from .evaluate import log_transfer_metrics, run_evaluation
from .metrics import (
    compute_classification_metrics,
    compute_ocr_metrics,
    compute_translation_metrics,
    get_compute_metrics,
)

__all__ = [
    "get_compute_metrics",
    "compute_classification_metrics",
    "compute_translation_metrics",
    "compute_ocr_metrics",
    "run_evaluation",
    "log_transfer_metrics",
]
