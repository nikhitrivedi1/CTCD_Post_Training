#!/usr/bin/env python3
"""Evaluate baseline model performance on GLUE tasks.

Evaluates teacher/student models to establish baselines for distillation comparison.
Supports: MNLI, SST2, CoLA, QQP with their primary metrics.

Usage:
    python scripts/eval_baselines.py --task mnli
    python scripts/eval_baselines.py --task sst2 --models teacher-sft student
    python scripts/eval_baselines.py --all
"""

import argparse
import sys
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tasks import get_num_labels, get_primary_metric

# Task configurations
TASK_CONFIG = {
    "mnli": {
        "dataset": ("glue", "mnli"),
        "eval_split": "validation_matched",
        "text_fields": ("premise", "hypothesis"),
        "sft_model": "textattack/bert-base-uncased-MNLI",
        "label_map": {  # TextAttack uses different label order
            "teacher-sft": [2, 0, 1],  # [C,E,N] -> [E,N,C]
        },
    },
    "sst2": {
        "dataset": ("glue", "sst2"),
        "eval_split": "validation",
        "text_fields": ("sentence",),
        "sft_model": "textattack/bert-base-uncased-SST-2",
        "label_map": {},
    },
    "cola": {
        "dataset": ("glue", "cola"),
        "eval_split": "validation",
        "text_fields": ("sentence",),
        "sft_model": "textattack/bert-base-uncased-CoLA",
        "label_map": {},
    },
    "qqp": {
        "dataset": ("glue", "qqp"),
        "eval_split": "validation",
        "text_fields": ("question1", "question2"),
        "sft_model": "textattack/bert-base-uncased-QQP",
        "label_map": {},
    },
}

# Model roles
MODEL_ROLES = {
    "student": "distilbert-base-uncased",
    "teacher-base": "bert-base-uncased",
    "teacher-sft": None,  # Task-specific, filled from TASK_CONFIG
}


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict:
    """Compute all classification metrics."""
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    mcc_metric = evaluate.load("matthews_correlation")

    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"],
        "mcc": mcc_metric.compute(predictions=predictions, references=labels)["matthews_correlation"],
    }


def evaluate_model(
    model_name: str,
    task: str,
    role: str,
    batch_size: int = 32,
    device: torch.device = None,
) -> dict:
    """
    Evaluate a model on a GLUE task.

    Args:
        model_name: HuggingFace model identifier
        task: Task name (mnli, sst2, cola, qqp)
        role: Model role for label mapping (student, teacher-base, teacher-sft)
        batch_size: Evaluation batch size
        device: Torch device

    Returns:
        Dictionary with all metrics
    """
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    config = TASK_CONFIG[task]
    num_labels = get_num_labels(task)

    print(f"\n  Loading {model_name}...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,  # For base models without classifier
    )
    model.to(device)
    model.eval()

    # Load dataset
    dataset_args = config["dataset"]
    dataset = load_dataset(*dataset_args)
    eval_data = dataset[config["eval_split"]]

    # Filter invalid labels
    eval_data = eval_data.filter(lambda x: x["label"] != -1)

    text_fields = config["text_fields"]
    label_map = config["label_map"].get(role)

    all_preds = []
    all_labels = []

    # Evaluate in batches
    for i in tqdm(range(0, len(eval_data), batch_size), desc="  Evaluating", leave=False):
        batch = eval_data[i : i + batch_size]

        # Tokenize based on number of text fields
        if len(text_fields) == 1:
            inputs = tokenizer(
                batch[text_fields[0]],
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = tokenizer(
                batch[text_fields[0]],
                batch[text_fields[1]],
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt",
            )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)

        # Apply label mapping if needed
        if label_map:
            label_map_tensor = torch.tensor(label_map, device=device)
            preds = label_map_tensor[preds]

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["label"])

    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = compute_metrics(all_preds, all_labels)
    metrics["samples"] = len(all_labels)

    return metrics


def evaluate_task(task: str, models: list[str], batch_size: int = 32) -> dict:
    """
    Evaluate multiple models on a single task.

    Args:
        task: Task name
        models: List of model roles to evaluate
        batch_size: Evaluation batch size

    Returns:
        Dictionary mapping role -> metrics
    """
    config = TASK_CONFIG[task]
    primary_metric = get_primary_metric(task)

    print(f"\n{'='*60}")
    print(f"Task: {task.upper()} (primary metric: {primary_metric})")
    print(f"{'='*60}")

    results = {}

    for role in models:
        if role == "teacher-sft":
            model_name = config["sft_model"]
        else:
            model_name = MODEL_ROLES[role]

        metrics = evaluate_model(model_name, task, role, batch_size)
        results[role] = {
            "model": model_name,
            "metrics": metrics,
        }

        # Print result
        pm_value = metrics[primary_metric]
        print(f"  {role:<15} {primary_metric}={pm_value:.4f} (acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, mcc={metrics['mcc']:.4f})")

    return results


def print_summary(all_results: dict):
    """Print summary table of all results."""
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    # Header
    tasks = list(all_results.keys())
    print(f"{'Model':<20}", end="")
    for task in tasks:
        pm = get_primary_metric(task)
        print(f" | {task.upper():^12}", end="")
    print()

    print(f"{'':<20}", end="")
    for task in tasks:
        pm = get_primary_metric(task)
        print(f" | {pm:^12}", end="")
    print()

    print("-" * 80)

    # Collect all roles
    all_roles = set()
    for task_results in all_results.values():
        all_roles.update(task_results.keys())

    # Print each role
    for role in sorted(all_roles):
        print(f"{role:<20}", end="")
        for task in tasks:
            if role in all_results[task]:
                pm = get_primary_metric(task)
                value = all_results[task][role]["metrics"][pm]
                print(f" | {value*100:>10.2f}%", end="")
            else:
                print(f" | {'--':^12}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline models on GLUE tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/eval_baselines.py --task mnli
    python scripts/eval_baselines.py --task sst2 --models teacher-sft student
    python scripts/eval_baselines.py --all --models teacher-sft
        """,
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["mnli", "sst2", "cola", "qqp"],
        help="Single task to evaluate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all tasks",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["student", "teacher-base", "teacher-sft"],
        choices=["student", "teacher-base", "teacher-sft"],
        help="Models to evaluate (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    args = parser.parse_args()

    if not args.task and not args.all:
        parser.error("Either --task or --all must be specified")

    tasks = ["mnli", "sst2", "cola", "qqp"] if args.all else [args.task]

    all_results = {}
    for task in tasks:
        all_results[task] = evaluate_task(task, args.models, args.batch_size)

    if len(tasks) > 1:
        print_summary(all_results)


if __name__ == "__main__":
    main()
