"""Evaluation metrics for different tasks."""

import logging
from typing import Dict, Any, Optional, Callable

import numpy as np

import evaluate
from calibration_toolbox import ACE
from src.tasks import is_classification_task, is_translation_task, is_ocr_task

logger = logging.getLogger(__name__)


def _ace_num_bins(num_samples: int) -> int:
    """Adaptive number of ACE ranges R = min(100, floor(sqrt(N))) (Nixon et al.; see plan)."""
    if num_samples <= 0:
        return 1
    return max(1, min(100, int(np.floor(np.sqrt(num_samples)))))


def compute_classification_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute metrics for classification tasks.

    Computes all standard metrics so they're available for any task.
    The task registry specifies which is the primary metric for each task.

    Adaptive Calibration Error (ACE) uses calibration-toolbox (Nixon et al., 2020,
    "Measuring Calibration in Deep Learning," CVPR Workshops; arXiv:1904.01685) with
    R = min(100, floor(sqrt(N))) adaptive bins per class.

    Args:
        eval_pred: Tuple of (predictions, labels)

    Returns:
        Dictionary with accuracy, F1, MCC, and ACE scores
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Handle empty predictions
    if len(predictions) == 0:
        logger.warning("Empty predictions array, returning zero metrics")
        return {"accuracy": 0.0, "f1": 0.0, "mcc": 0.0, "ace": 0.0}

    # Load metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    mcc_metric = evaluate.load("matthews_correlation")

    # Compute all metrics
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    mcc = mcc_metric.compute(predictions=predictions, references=labels)

    n_samples = int(logits.shape[0])
    n_bins = _ace_num_bins(n_samples)
    try:
        ace_val = float(
            ACE(
                np.asarray(logits, dtype=np.float64),
                np.asarray(labels).reshape(-1),
                n_bins=n_bins,
                logits=True,
            )
        )
    except Exception as e:
        logger.warning("ACE computation failed: %s", e)
        ace_val = 0.0

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "mcc": mcc["matthews_correlation"],
        "ace": ace_val,
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


def _edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def compute_ocr_metrics(tokenizer):
    """
    Create a compute_metrics function for OCR tasks.

    Computes edit similarity (1 - normalized edit distance) and exact match rate.

    Args:
        tokenizer: Tokenizer (or processor.tokenizer) for decoding predictions

    Returns:
        Function that computes edit_similarity and exact_match
    """

    def compute_metrics(eval_pred) -> Dict[str, float]:
        """Compute OCR metrics: edit similarity and exact match.

        Args:
            eval_pred: Tuple of (predictions, labels)

        Returns:
            Dictionary with edit_similarity and exact_match scores
        """
        logits, labels = eval_pred

        if len(logits.shape) == 3:
            predictions = np.argmax(logits, axis=-1)
        else:
            predictions = logits

        # Replace -100 padding in labels
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        labels = np.where(labels != -100, labels, pad_token_id)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        edit_sims = []
        exact_matches = 0

        for pred, label in zip(decoded_preds, decoded_labels):
            pred, label = pred.strip(), label.strip()
            if pred == label:
                exact_matches += 1
                edit_sims.append(1.0)
            else:
                max_len = max(len(pred), len(label))
                if max_len == 0:
                    edit_sims.append(1.0)
                else:
                    dist = _edit_distance(pred, label)
                    edit_sims.append(1.0 - dist / max_len)

        return {
            "edit_similarity": np.mean(edit_sims) if edit_sims else 0.0,
            "exact_match": exact_matches / len(decoded_preds) if decoded_preds else 0.0,
        }

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
    elif is_ocr_task(task):
        if tokenizer is None:
            raise ValueError("Tokenizer required for OCR metrics")
        return compute_ocr_metrics(tokenizer)
    else:
        return None
