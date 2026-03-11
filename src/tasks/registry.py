"""Centralized task registry for all supported tasks.

Add new tasks here - single source of truth for task properties.
"""

TASK_REGISTRY = {
    # Classification tasks (GLUE)
    "mnli": {
        "num_labels": 3,
        "type": "classification",
        "dataset": "glue",
        "subset": "mnli",
        "primary_metric": "accuracy",
    },
    "qqp": {
        "num_labels": 2,
        "type": "classification",
        "dataset": "glue",
        "subset": "qqp",
        "primary_metric": "f1",
    },
    "sst2": {
        "num_labels": 2,
        "type": "classification",
        "dataset": "glue",
        "subset": "sst2",
        "primary_metric": "accuracy",
    },
    "cola": {
        "num_labels": 2,
        "type": "classification",
        "dataset": "glue",
        "subset": "cola",
        "primary_metric": "mcc",
    },
    # Translation tasks
    "europarl": {
        "num_labels": None,
        "type": "translation",
        "dataset": "europarl",
        "subset": None,
        "primary_metric": "bleu",
        "default_reward": "bleu",
    },
    # OCR tasks
    "ocr": {
        "num_labels": None,
        "type": "ocr",
        "dataset": "olmocr-mix",
        "subset": None,
        "primary_metric": "edit_similarity",
        "default_reward": "edit_similarity",
    },
}


def get_num_labels(task: str) -> int | None:
    """Get number of labels for a task."""
    if task not in TASK_REGISTRY:
        return None
    return TASK_REGISTRY[task]["num_labels"]


def get_task_type(task: str) -> str | None:
    """Get task type ('classification' or 'translation')."""
    if task not in TASK_REGISTRY:
        return None
    return TASK_REGISTRY[task]["type"]


def is_classification_task(task: str) -> bool:
    """Check if task is a classification task."""
    return get_task_type(task) == "classification"


def is_translation_task(task: str) -> bool:
    """Check if task is a translation task."""
    return get_task_type(task) == "translation"


def is_ocr_task(task: str) -> bool:
    """Check if task is an OCR task."""
    return get_task_type(task) == "ocr"


def get_primary_metric(task: str) -> str | None:
    """Get the primary metric for a task (used for best model selection)."""
    if task not in TASK_REGISTRY:
        return None
    return TASK_REGISTRY[task].get("primary_metric")


def get_default_reward(task: str) -> str | None:
    """Get the default verifiable reward for a task.

    Args:
        task: Task name.

    Returns:
        Reward name string, or None if task has no default reward.
    """
    if task not in TASK_REGISTRY:
        return None
    return TASK_REGISTRY[task].get("default_reward")
