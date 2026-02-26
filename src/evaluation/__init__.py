"""Evaluation metrics and orchestration for classification, translation, and OCR tasks."""

from .metrics import get_compute_metrics, compute_classification_metrics, compute_translation_metrics, compute_ocr_metrics
from .evaluate import run_evaluation, log_transfer_metrics

__all__ = [
    "get_compute_metrics",
    "compute_classification_metrics",
    "compute_translation_metrics",
    "compute_ocr_metrics",
    "run_evaluation",
    "log_transfer_metrics",
]
