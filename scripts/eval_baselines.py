#!/usr/bin/env python3
"""Evaluate baseline model performance on supported tasks.

Evaluates preset models to establish baselines for distillation comparison.
Currently supports:
  - Classification: MNLI, SST2, CoLA, QQP (accuracy/f1/mcc)
  - Translation: Europarl EN-FR (BLEU)

Usage:
    python scripts/eval_baselines.py                    # All tasks, all models
    python scripts/eval_baselines.py --task mnli        # Single task
    python scripts/eval_baselines.py --task mnli sst2   # Multiple tasks
    python scripts/eval_baselines.py --models bert-sft  # Specific model(s)
"""

import argparse
import sys
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import compute_classification_metrics
from src.tasks import get_num_labels, get_primary_metric, is_translation_task

# =============================================================================
# Classification models
# =============================================================================

# Base models (task-agnostic, untrained classifiers)
CLASSIFICATION_BASE_MODELS = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
}

# Task-specific SFT models (textattack fine-tuned)
CLASSIFICATION_SFT_MODELS = {
    "mnli": {"bert-sft": "textattack/bert-base-uncased-MNLI"},
    "sst2": {"bert-sft": "textattack/bert-base-uncased-SST-2"},
    "cola": {"bert-sft": "textattack/bert-base-uncased-CoLA"},
    "qqp": {"bert-sft": "textattack/bert-base-uncased-QQP"},
}

# Label mapping for textattack MNLI (uses different label order)
# Direct indexing: new_logits = logits[:, LABEL_MAPS[model]]
# TextAttack MNLI: [C,E,N] at indices [0,1,2]
# Dataset MNLI:    [E,N,C] at indices [0,1,2]
# So: new[0]=old[1], new[1]=old[2], new[2]=old[0] → [1,2,0]
LABEL_MAPS = {
    "textattack/bert-base-uncased-MNLI": [1, 2, 0],
}

# =============================================================================
# Translation models
# =============================================================================

TRANSLATION_BASE_MODELS = {
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
}

# =============================================================================
# Task configurations
# =============================================================================

TASK_CONFIG = {
    # Classification tasks
    "mnli": {
        "type": "classification",
        "dataset": ("glue", "mnli"),
        "eval_split": "validation_matched",
        "text_fields": ("premise", "hypothesis"),
    },
    "sst2": {
        "type": "classification",
        "dataset": ("glue", "sst2"),
        "eval_split": "validation",
        "text_fields": ("sentence",),
    },
    "cola": {
        "type": "classification",
        "dataset": ("glue", "cola"),
        "eval_split": "validation",
        "text_fields": ("sentence",),
    },
    "qqp": {
        "type": "classification",
        "dataset": ("glue", "qqp"),
        "eval_split": "validation",
        "text_fields": ("question1", "question2"),
    },
    # Translation tasks
    "europarl": {
        "type": "translation",
        "dataset": ("europarl_bilingual", "en-fr"),
        "eval_split": "train",  # Europarl only has train split
        "eval_samples": 1000,  # Subsample for evaluation
        "source_lang": "en",
        "target_lang": "fr",
        "max_new_tokens": 128,
    },
}


def get_models_for_task(task: str) -> dict:
    """Get all available models for a task."""
    config = TASK_CONFIG[task]
    if config["type"] == "classification":
        models = CLASSIFICATION_BASE_MODELS.copy()
        models.update(CLASSIFICATION_SFT_MODELS.get(task, {}))
    else:
        models = TRANSLATION_BASE_MODELS.copy()
    return models


def evaluate_classification_model(
    model_name: str,
    model_id: str,
    task: str,
    batch_size: int = 32,
    device: torch.device = None,
) -> dict:
    """
    Evaluate a model on a classification task.

    Args:
        model_name: Short name for display (e.g., "bert-sft")
        model_id: HuggingFace model ID
        task: Task name (e.g., mnli, sst2, cola, qqp)
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

    print(f"\n  Loading {model_id}...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    model.eval()

    # Load dataset
    dataset = load_dataset(*config["dataset"])
    eval_data = dataset[config["eval_split"]]
    eval_data = eval_data.filter(lambda x: x["label"] != -1)

    text_fields = config["text_fields"]
    label_map = LABEL_MAPS.get(model_id)

    all_logits = []
    all_labels = []

    for i in tqdm(range(0, len(eval_data), batch_size), desc="  Evaluating", leave=False):
        batch = eval_data[i : i + batch_size]

        # Tokenize
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

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Apply label mapping if needed (direct indexing)
        if label_map:
            logits = logits[:, label_map]

        all_logits.append(logits.cpu().numpy())
        all_labels.extend(batch["label"])

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.array(all_labels)
    metrics = compute_classification_metrics((all_logits, all_labels))
    metrics["samples"] = len(all_labels)

    return metrics


def evaluate_translation_model(
    model_name: str,
    model_id: str,
    task: str,
    batch_size: int = 8,
    device: torch.device = None,
) -> dict:
    """
    Evaluate a model on a translation task.

    Args:
        model_name: Short name for display (e.g., "llama-3b")
        model_id: HuggingFace model ID
        task: Task name (e.g., europarl)
        batch_size: Evaluation batch size
        device: Torch device

    Returns:
        Dictionary with BLEU score
    """
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    config = TASK_CONFIG[task]

    print(f"\n  Loading {model_id}...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # Load dataset
    dataset = load_dataset(*config["dataset"], trust_remote_code=True)
    eval_data = dataset[config["eval_split"]]

    # Subsample for faster evaluation
    eval_samples = config.get("eval_samples", len(eval_data))
    if eval_samples < len(eval_data):
        eval_data = eval_data.shuffle(seed=42).select(range(eval_samples))

    source_lang = config["source_lang"]
    target_lang = config["target_lang"]
    max_new_tokens = config.get("max_new_tokens", 128)

    all_predictions = []
    all_references = []

    for i in tqdm(range(0, len(eval_data), batch_size), desc="  Evaluating", leave=False):
        batch = eval_data[i : i + batch_size]

        # Format prompts for translation
        prompts = [
            f"Translate the following English text to French:\n\n{item['translation'][source_lang]}\n\nFrench translation:"
            for item in batch
        ]

        inputs = tokenizer(
            prompts,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated part
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            generated = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            all_predictions.append(generated.strip())
            all_references.append(batch[j]["translation"][target_lang])

    # Compute BLEU
    bleu_metric = evaluate.load("sacrebleu")
    references = [[ref] for ref in all_references]
    bleu_result = bleu_metric.compute(predictions=all_predictions, references=references)

    return {
        "bleu": bleu_result["score"] / 100.0,  # Normalize to 0-1 range
        "samples": len(all_predictions),
    }


def evaluate_model(
    model_name: str,
    model_id: str,
    task: str,
    batch_size: int = 32,
    device: torch.device = None,
) -> dict:
    """
    Evaluate a model on a task.

    Args:
        model_name: Short name for display
        model_id: HuggingFace model ID
        task: Task name
        batch_size: Evaluation batch size
        device: Torch device

    Returns:
        Dictionary with metrics
    """
    if is_translation_task(task):
        # Smaller batch for generation
        return evaluate_translation_model(model_name, model_id, task, min(batch_size, 8), device)
    else:
        return evaluate_classification_model(model_name, model_id, task, batch_size, device)


def format_metrics(metrics: dict, task: str) -> str:
    """Format metrics for display based on task type."""
    if is_translation_task(task):
        return f"bleu={metrics['bleu']:.4f}"
    else:
        return f"acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, mcc={metrics['mcc']:.4f}"


def evaluate_task(task: str, model_names: list[str], batch_size: int = 32) -> dict:
    """Evaluate specified models on a task."""
    available = get_models_for_task(task)
    primary_metric = get_primary_metric(task)

    print(f"\n{'='*60}")
    print(f"Task: {task.upper()} (primary metric: {primary_metric})")
    print(f"{'='*60}")

    results = {}

    for name in model_names:
        if name not in available:
            print(f"  {name:<15} SKIPPED (not available for {task})")
            continue

        model_id = available[name]
        metrics = evaluate_model(name, model_id, task, batch_size)
        results[name] = {"model_id": model_id, "metrics": metrics}

        pm_value = metrics[primary_metric]
        extra = format_metrics(metrics, task)
        print(f"  {name:<15} {primary_metric}={pm_value:.4f} ({extra})")

    return results


def print_summary(all_results: dict):
    """Print summary table."""
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    tasks = list(all_results.keys())
    all_models = set()
    for task_results in all_results.values():
        all_models.update(task_results.keys())

    # Header
    print(f"{'Model':<15}", end="")
    for task in tasks:
        print(f" | {task.upper():^12}", end="")
    print()

    print(f"{'':<15}", end="")
    for task in tasks:
        pm = get_primary_metric(task)
        print(f" | {pm:^12}", end="")
    print()
    print("-" * 80)

    # Results
    for model in sorted(all_models):
        print(f"{model:<15}", end="")
        for task in tasks:
            if model in all_results[task]:
                pm = get_primary_metric(task)
                value = all_results[task][model]["metrics"][pm]
                print(f" | {value*100:>10.2f}%", end="")
            else:
                print(f" | {'--':^12}", end="")
        print()


def get_all_model_names() -> list[str]:
    """Get list of all available model names across all tasks."""
    names = set(CLASSIFICATION_BASE_MODELS.keys())
    names.add("bert-sft")
    names.update(TRANSLATION_BASE_MODELS.keys())
    return sorted(names)


def main():
    all_tasks = list(TASK_CONFIG.keys())
    all_model_names = get_all_model_names()

    parser = argparse.ArgumentParser(
        description="Evaluate baseline models on supported tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models: {', '.join(all_model_names)}
Available tasks: {', '.join(all_tasks)}

Examples:
    python scripts/eval_baselines.py                      # All tasks, all models
    python scripts/eval_baselines.py --task mnli          # Single task
    python scripts/eval_baselines.py --task mnli sst2     # Multiple tasks
    python scripts/eval_baselines.py --models bert-sft    # Just SFT models
        """,
    )
    parser.add_argument(
        "--task",
        type=str,
        nargs="*",
        default=all_tasks,
        choices=all_tasks,
        help=f"Tasks to evaluate (default: all)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=all_model_names,
        help=f"Models to evaluate (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    args = parser.parse_args()

    # Validate models
    for model in args.models:
        if model not in all_model_names:
            parser.error(f"Unknown model: {model}. Available: {', '.join(all_model_names)}")

    all_results = {}
    for task in args.task:
        all_results[task] = evaluate_task(task, args.models, args.batch_size)

    if len(args.task) > 1:
        print_summary(all_results)


if __name__ == "__main__":
    main()
