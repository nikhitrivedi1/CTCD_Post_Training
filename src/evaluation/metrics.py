"""Evaluation metrics for different tasks."""

import logging
from typing import Dict, Any, Optional, Callable

import evaluate
import numpy as np

from src.tasks import is_classification_task, is_translation_task

logger = logging.getLogger(__name__)


def compute_classification_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute metrics for classification tasks.

    Computes all standard metrics so they're available for any task.
    The task registry specifies which is the primary metric for each task.

    Args:
        eval_pred: Tuple of (predictions, labels)

    Returns:
        Dictionary with accuracy, F1, and MCC scores
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Handle empty predictions
    if len(predictions) == 0:
        logger.warning("Empty predictions array, returning zero metrics")
        return {"accuracy": 0.0, "f1": 0.0, "mcc": 0.0}

    # Load metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    mcc_metric = evaluate.load("matthews_correlation")

    # Compute all metrics
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    mcc = mcc_metric.compute(predictions=predictions, references=labels)

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "mcc": mcc["matthews_correlation"],
    }


def compute_translation_metrics(tokenizer):
    """
    Create a compute_metrics function for translation tasks.

    Args:
        tokenizer: Tokenizer for decoding predictions

    Returns:
        Function that computes BLEU and COMET scores
    """

    def compute_metrics(eval_pred) -> Dict[str, float]:
        """
        Compute metrics for translation tasks (Europarl).

        Args:
            eval_pred: Tuple of (predictions, labels)

        Returns:
            Dictionary with BLEU and COMET scores
        """
        logits, labels = eval_pred

        # Handle different logit shapes (sequence classification vs generation)
        if len(logits.shape) == 3:
            # Take argmax over vocabulary dimension
            predictions = np.argmax(logits, axis=-1)
        else:
            predictions = logits

        # Replace -100 in labels (used for padding)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Clean up whitespace
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # BLEU expects list of references for each prediction
        references = [[label] for label in decoded_labels]

        # Load metrics
        bleu_metric = evaluate.load("sacrebleu")

        # Compute BLEU
        bleu_result = bleu_metric.compute(predictions=decoded_preds, references=references)

        metrics = {
            "bleu": bleu_result["score"],
        }

        # Try to compute COMET (may fail if not available or too slow)
        try:
            comet_metric = evaluate.load("comet")
            # COMET needs source text, but we don't have it in eval_pred
            # For now, we'll skip COMET during training and compute it separately
            # comet_result = comet_metric.compute(predictions=decoded_preds, references=decoded_labels)
            # metrics["comet"] = comet_result["mean_score"]
        except (ImportError, FileNotFoundError) as e:
            logger.warning(f"Skipping COMET metric: {e}")

        return metrics

    return compute_metrics


def get_compute_metrics(task: str, tokenizer: Optional[Any] = None) -> Optional[Callable]:
    """
    Get the appropriate compute_metrics function for a task.

    Args:
        task: Task name (see src/tasks/registry.py for supported tasks)
        tokenizer: Tokenizer (required for generation tasks)

    Returns:
        Compute metrics function or None
    """
    if is_classification_task(task):
        return compute_classification_metrics
    elif is_translation_task(task):
        if tokenizer is None:
            raise ValueError("Tokenizer required for translation metrics")
        return compute_translation_metrics(tokenizer)
    else:
        return None
