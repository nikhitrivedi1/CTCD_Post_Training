"""
Prediction disagreement and per-class accuracy analysis.

Analyzes where teacher and student predictions differ, and breaks down
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
    teacher_preds: np.ndarray,
    teacher_confs: np.ndarray,
    student_preds: np.ndarray,
    student_confs: np.ndarray,
    labels: np.ndarray,
    task: str,
) -> Dict:
    """
    Analyze prediction disagreements between teacher and student.

    Args:
        teacher_preds: Teacher predictions
        teacher_confs: Teacher confidence scores
        student_preds: Student predictions
        student_confs: Student confidence scores
        labels: Ground truth labels
        task: Task name for class labels

    Returns:
        Dictionary with disagreement statistics
    """
    n_samples = len(labels)
    class_labels = CLASS_LABELS.get(task, {})

    # Basic disagreement
    agree_mask = teacher_preds == student_preds
    disagree_mask = ~agree_mask

    # Correctness masks
    teacher_correct = teacher_preds == labels
    student_correct = student_preds == labels

    # Disagreement categories
    both_correct = teacher_correct & student_correct
    both_wrong = ~teacher_correct & ~student_correct
    teacher_only_correct = teacher_correct & ~student_correct
    student_only_correct = ~teacher_correct & student_correct

    # Agreement on correct/wrong
    agree_correct = agree_mask & teacher_correct
    agree_wrong = agree_mask & ~teacher_correct

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
            "teacher_only_correct": int(teacher_only_correct.sum()),
            "teacher_only_correct_pct": float(teacher_only_correct.mean()),
            "student_only_correct": int(student_only_correct.sum()),
            "student_only_correct_pct": float(student_only_correct.mean()),
            "both_wrong_disagree": int((disagree_mask & both_wrong).sum()),
        },
        "correctness": {
            "both_correct": int(both_correct.sum()),
            "both_correct_pct": float(both_correct.mean()),
            "both_wrong": int(both_wrong.sum()),
            "both_wrong_pct": float(both_wrong.mean()),
            "teacher_only_correct": int(teacher_only_correct.sum()),
            "student_only_correct": int(student_only_correct.sum()),
        },
        "accuracy": {
            "teacher": float(teacher_correct.mean()),
            "student": float(student_correct.mean()),
        },
        "confidence": {
            "teacher_mean": float(teacher_confs.mean()),
            "student_mean": float(student_confs.mean()),
            "teacher_when_wrong": float(teacher_confs[~teacher_correct].mean()) if (~teacher_correct).any() else 0.0,
            "student_when_wrong": float(student_confs[~student_correct].mean()) if (~student_correct).any() else 0.0,
        },
    }

    # Per-class disagreement breakdown
    class_breakdown = {}
    for class_idx, class_name in class_labels.items():
        class_mask = labels == class_idx
        if not class_mask.any():
            continue

        class_disagree = disagree_mask & class_mask
        class_teacher_correct = teacher_correct & class_mask
        class_student_correct = student_correct & class_mask

        class_breakdown[class_name] = {
            "count": int(class_mask.sum()),
            "disagree_count": int(class_disagree.sum()),
            "disagree_pct": float(class_disagree.sum() / class_mask.sum()),
            "teacher_acc": float(class_teacher_correct.sum() / class_mask.sum()),
            "student_acc": float(class_student_correct.sum() / class_mask.sum()),
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
    teacher_model: str,
    student_model: str,
    task: str,
    device: str = "cuda",
    max_samples: int = None,
) -> Dict:
    """
    Run full prediction disagreement analysis.

    Args:
        teacher_model: Teacher model path
        student_model: Student model path
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
    logger.info("Loading teacher model...")
    teacher, teacher_tok, teacher_config = load_classification_model(teacher_model, num_labels, device)

    logger.info("Loading student model...")
    student, student_tok, student_config = load_classification_model(student_model, num_labels, device)

    # Get label mappings for models with non-standard label orderings
    teacher_label_map = get_label_map(teacher_model, task, teacher_config)
    student_label_map = get_label_map(student_model, task, student_config)

    if teacher_label_map:
        logger.info(f"Teacher label map: {teacher_label_map}")
    if student_label_map:
        logger.info(f"Student label map: {student_label_map}")

    # Get predictions
    logger.info("Getting teacher predictions...")
    teacher_preds, teacher_confs = get_predictions(
        teacher, teacher_tok, texts, device=device, label_map=teacher_label_map
    )

    logger.info("Getting student predictions...")
    student_preds, student_confs = get_predictions(
        student, student_tok, texts, device=device, label_map=student_label_map
    )

    # Compute analyses
    logger.info("Computing disagreement analysis...")
    disagreement = compute_disagreement_analysis(
        teacher_preds, teacher_confs,
        student_preds, student_confs,
        labels, task
    )

    logger.info("Computing per-class breakdown...")
    teacher_class = compute_per_class_accuracy(teacher_preds, labels, task, "teacher")
    student_class = compute_per_class_accuracy(student_preds, labels, task, "student")

    return {
        "teacher": {
            "model": teacher_model,
            "per_class": teacher_class,
            "label_map": teacher_label_map,
        },
        "student": {
            "model": student_model,
            "per_class": student_class,
            "label_map": student_label_map,
        },
        "disagreement": disagreement,
        "task": task,
        "num_samples": len(labels),
    }
