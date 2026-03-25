"""Dataset handler registry.

Usage:
    from src.data.handlers import get_handler
    handler = get_handler("omnidocbench")
    handler.prepare("data/ocr/omnidocbench")
    train, eval = handler.load("data/ocr/omnidocbench")
"""

from .base import DatasetHandler
from .europarl import EuroparlHandler
from .formatting import FormattingHandler
from .glue import GLUEHandler
from .olmocr_mix import OlmOCRMixHandler
from .omnidocbench import OmniDocBenchHandler
from .synthmix import SynthmixHandler

DATASET_REGISTRY: dict[str, DatasetHandler] = {
    # OCR datasets
    "omnidocbench": OmniDocBenchHandler(),
    "synthmix": SynthmixHandler(),
    "formatting": FormattingHandler(),
    "olmocr-mix": OlmOCRMixHandler(),
    # Translation
    "europarl": EuroparlHandler(),
    # Classification (GLUE)
    "mnli": GLUEHandler(task="mnli"),
    "qqp": GLUEHandler(task="qqp"),
    "sst2": GLUEHandler(task="sst2"),
    "cola": GLUEHandler(task="cola"),
    "mrpc": GLUEHandler(task="mrpc"),
    "qnli": GLUEHandler(task="qnli"),
    "rte": GLUEHandler(task="rte"),
    "wnli": GLUEHandler(task="wnli"),
}


def get_handler(dataset_name: str) -> DatasetHandler:
    """Look up a dataset handler by name.

    Args:
        dataset_name: Key in DATASET_REGISTRY.

    Returns:
        DatasetHandler instance.

    Raises:
        ValueError: If dataset_name is not registered.
    """
    if dataset_name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset: '{dataset_name}'. Available: {available}")
    return DATASET_REGISTRY[dataset_name]


__all__ = [
    "DatasetHandler",
    "DATASET_REGISTRY",
    "get_handler",
    "EuroparlHandler",
    "FormattingHandler",
    "GLUEHandler",
    "OlmOCRMixHandler",
    "OmniDocBenchHandler",
    "SynthmixHandler",
]
