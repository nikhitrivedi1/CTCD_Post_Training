"""Evaluation metrics for classification and translation tasks."""

from .metrics import get_compute_metrics, compute_metrics_classification, compute_metrics_translation

__all__ = [
    "get_compute_metrics",
    "compute_metrics_classification",
    "compute_metrics_translation",
]
