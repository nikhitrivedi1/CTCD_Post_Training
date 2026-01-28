"""Data loading modules for GLUE tasks and Europarl translation."""

from .datasets import load_dataset_for_task, load_mnli, load_qqp, load_europarl

__all__ = [
    "load_dataset_for_task",
    "load_mnli",
    "load_qqp",
    "load_europarl",
]
