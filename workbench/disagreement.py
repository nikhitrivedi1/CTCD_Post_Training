"""
Prediction disagreement and per-class accuracy analysis.

Analyzes where two models' predictions differ, and breaks down
accuracy by class to understand model behavior on different categories.
"""

import inspect
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

from workbench.utils import load_task_data, TASK_CONFIG, get_label_map, CLASS_LABELS

logger = logging.getLogger(__name__)


def load_classification_model(model_path: str, num_labels: int, device: str = "cuda"):
    """
    Load a model with classification head for prediction.

    Args:
        model_path: HuggingFace model name or local checkpoint path
        num_labels: Number of output classes
        device: Device to load on

    Returns:
        Tuple of (model, tokenizer, config)
    """
    checkpoint_path = Path(model_path)

    if checkpoint_path.exists():
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, config=config
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Loaded {model_path}: {param_count:.1f}M params")

    return model, tokenizer, config


def get_predictions(
    model,
    tokenizer,
    texts: List,
    batch_size: int = 32,
    max_length: int = 128,
    device: str = "cuda",
    show_progress: bool = True,
    label_map: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get model predictions and confidence scores.

    Args:
        model: Classification model
        tokenizer: Tokenizer
        texts: List of text inputs
        batch_size: Batch size
        max_length: Max sequence length
        device: Device
        show_progress: Show progress bar
        label_map: Optional list to reorder logits to GLUE standard order

    Returns:
        Tuple of (predictions [N], confidences [N])
    """
    all_preds = []
    all_confs = []

    # Process in batches, tokenizing per-batch (not upfront) to avoid padding issues
    num_batches = (len(texts) + batch_size - 1) // batch_size
    iterator = tqdm(range(num_batches), desc="Predicting") if show_progress else range(num_batches)

    is_pair = isinstance(texts[0], tuple)

    with torch.no_grad():
        for batch_idx in iterator:
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]

            # Tokenize this batch only
            if is_pair:
                inputs = tokenizer(
                    [t[0] for t in batch_texts],
                    [t[1] for t in batch_texts],
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                inputs = tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_tensors="pt",
                )

            # Move inputs to device, filtering to only what model accepts
            # (DistilBERT doesn't use token_type_ids, BERT does)
            forward_params = set(inspect.signature(model.forward).parameters.keys())
            inputs = {k: v.to(device) for k, v in inputs.items() if k in forward_params}

            outputs = model(**inputs)
            logits = outputs.logits

            # Apply label mapping if needed (reorder logits to GLUE standard)
            if label_map is not None:
                logits = logits[:, label_map]

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            confs = probs.max(dim=-1).values

            all_preds.append(preds.cpu().numpy())
            all_confs.append(confs.cpu().numpy())

    predictions = np.concatenate(all_preds)
    confidences = np.concatenate(all_confs)

    return predictions, confidences


def compute_disagreement_analysis(
    model_preds: np.ndarray,
    model_confs: np.ndarray,
    ref_preds: np.ndarray,
    ref_confs: np.ndarray,
    labels: np.ndarray,
    task: str,
) -> Dict:
    """
    Analyze prediction disagreements between two models.

    Args:
        model_preds: Model predictions
        model_confs: Model confidence scores
        ref_preds: Reference model predictions
        ref_confs: Reference model confidence scores
        labels: Ground truth labels
        task: Task name for class labels

    Returns:
        Dictionary with disagreement statistics
    """
    n_samples = len(labels)
    class_labels = CLASS_LABELS.get(task, {})

    # Basic disagreement
    agree_mask = model_preds == ref_preds
    disagree_mask = ~agree_mask

    # Correctness masks
    model_correct = model_preds == labels
    ref_correct = ref_preds == labels

    # Disagreement categories
    both_correct = model_correct & ref_correct
    both_wrong = ~model_correct & ~ref_correct
    model_only_correct = model_correct & ~ref_correct
    ref_only_correct = ~model_correct & ref_correct

    # Agreement on correct/wrong
    agree_correct = agree_mask & model_correct
    agree_wrong = agree_mask & ~model_correct

    results = {
        "total_samples": n_samples,
        "agreement": {
            "agree": int(agree_mask.sum()),
            "agree_pct": float(agree_mask.mean()),
            "agree_and_correct": int(agree_correct.sum()),
            "agree_and_correct_pct": float(agree_correct.mean()),
            "agree_and_wrong": int(agree_wrong.sum()),
            "agree_and_wrong_pct": float(agree_wrong.mean()),
        },
        "disagreement": {
            "disagree": int(disagree_mask.sum()),
            "disagree_pct": float(disagree_mask.mean()),
            "model_only_correct": int(model_only_correct.sum()),
            "model_only_correct_pct": float(model_only_correct.mean()),
            "reference_only_correct": int(ref_only_correct.sum()),
            "reference_only_correct_pct": float(ref_only_correct.mean()),
            "both_wrong_disagree": int((disagree_mask & both_wrong).sum()),
        },
        "correctness": {
            "both_correct": int(both_correct.sum()),
            "both_correct_pct": float(both_correct.mean()),
            "both_wrong": int(both_wrong.sum()),
            "both_wrong_pct": float(both_wrong.mean()),
            "model_only_correct": int(model_only_correct.sum()),
            "reference_only_correct": int(ref_only_correct.sum()),
        },
        "accuracy": {
            "model": float(model_correct.mean()),
            "reference": float(ref_correct.mean()),
        },
        "confidence": {
            "model_mean": float(model_confs.mean()),
            "reference_mean": float(ref_confs.mean()),
            "model_when_wrong": float(model_confs[~model_correct].mean()) if (~model_correct).any() else 0.0,
            "reference_when_wrong": float(ref_confs[~ref_correct].mean()) if (~ref_correct).any() else 0.0,
        },
    }

    # Per-class disagreement breakdown
    class_breakdown = {}
    for class_idx, class_name in class_labels.items():
        class_mask = labels == class_idx
        if not class_mask.any():
            continue

        class_disagree = disagree_mask & class_mask
        class_model_correct = model_correct & class_mask
        class_ref_correct = ref_correct & class_mask

        class_breakdown[class_name] = {
            "count": int(class_mask.sum()),
            "disagree_count": int(class_disagree.sum()),
            "disagree_pct": float(class_disagree.sum() / class_mask.sum()),
            "model_acc": float(class_model_correct.sum() / class_mask.sum()),
            "reference_acc": float(class_ref_correct.sum() / class_mask.sum()),
        }

    results["per_class_disagreement"] = class_breakdown

    return results


def compute_per_class_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    task: str,
    model_name: str = "model",
) -> Dict:
    """
    Compute per-class accuracy breakdown.

    Args:
        predictions: Model predictions
        labels: Ground truth labels
        task: Task name
        model_name: Name for logging

    Returns:
        Dictionary with per-class statistics
    """
    class_labels = CLASS_LABELS.get(task, {})
    correct = predictions == labels

    results = {
        "model": model_name,
        "overall_accuracy": float(correct.mean()),
        "total_samples": len(labels),
        "per_class": {},
    }

    # Confusion matrix components
    num_classes = len(class_labels)
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            confusion[true_class, pred_class] = int(
                ((labels == true_class) & (predictions == pred_class)).sum()
            )

    results["confusion_matrix"] = confusion.tolist()

    for class_idx, class_name in class_labels.items():
        class_mask = labels == class_idx
        if not class_mask.any():
            continue

        class_correct = correct & class_mask
        class_preds = predictions[class_mask]

        # Class-specific metrics
        true_positives = int(((predictions == class_idx) & (labels == class_idx)).sum())
        false_positives = int(((predictions == class_idx) & (labels != class_idx)).sum())
        false_negatives = int(((predictions != class_idx) & (labels == class_idx)).sum())

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Prediction distribution for this true class
        pred_distribution = {}
        for pred_class, pred_name in class_labels.items():
            pred_distribution[pred_name] = int((class_preds == pred_class).sum())

        results["per_class"][class_name] = {
            "count": int(class_mask.sum()),
            "correct": int(class_correct.sum()),
            "accuracy": float(class_correct.sum() / class_mask.sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "prediction_distribution": pred_distribution,
        }

    return results


def run_disagreement_analysis(
    model_name: str,
    reference_model: str,
    task: str,
    device: str = "cuda",
    max_samples: int = None,
) -> Dict:
    """
    Run full prediction disagreement analysis.

    Args:
        model_name: Model path
        reference_model: Reference model path
        task: Task name
        device: Device
        max_samples: Limit samples

    Returns:
        Dictionary with analysis results
    """
    task_cfg = TASK_CONFIG[task]
    num_labels = task_cfg["num_labels"]

    # Load data
    texts, labels, _ = load_task_data(task, max_samples)
    labels = np.array(labels)

    # Load models
    logger.info("Loading model...")
    model, model_tok, model_config = load_classification_model(model_name, num_labels, device)

    logger.info("Loading reference model...")
    ref, ref_tok, ref_config = load_classification_model(reference_model, num_labels, device)

    # Get label mappings for models with non-standard label orderings
    model_label_map = get_label_map(model_name, task, model_config)
    ref_label_map = get_label_map(reference_model, task, ref_config)

    if model_label_map:
        logger.info(f"Model label map: {model_label_map}")
    if ref_label_map:
        logger.info(f"Reference label map: {ref_label_map}")

    # Get predictions
    logger.info("Getting model predictions...")
    model_preds, model_confs = get_predictions(
        model, model_tok, texts, device=device, label_map=model_label_map
    )

    logger.info("Getting reference predictions...")
    ref_preds, ref_confs = get_predictions(
        ref, ref_tok, texts, device=device, label_map=ref_label_map
    )

    # Compute analyses
    logger.info("Computing disagreement analysis...")
    disagreement = compute_disagreement_analysis(
        model_preds, model_confs,
        ref_preds, ref_confs,
        labels, task
    )

    logger.info("Computing per-class breakdown...")
    model_class = compute_per_class_accuracy(model_preds, labels, task, "model")
    ref_class = compute_per_class_accuracy(ref_preds, labels, task, "reference")

    return {
        "model": {
            "model": model_name,
            "per_class": model_class,
            "label_map": model_label_map,
        },
        "reference": {
            "model": reference_model,
            "per_class": ref_class,
            "label_map": ref_label_map,
        },
        "disagreement": disagreement,
        "task": task,
        "num_samples": len(labels),
    }
