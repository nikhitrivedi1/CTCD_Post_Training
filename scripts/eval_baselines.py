#!/usr/bin/env python3
"""Evaluate baseline model accuracies on GLUE tasks (zero-shot, no training)."""

import argparse
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


def evaluate_model(model_name: str, task: str = "qqp", batch_size: int = 32):
    """Evaluate a model on a GLUE task without any training."""

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nEvaluating: {model_name}")
    print(f"Task: {task}, Device: {device}")

    # Determine num_labels
    num_labels = 2 if task == "qqp" else 3  # QQP=2, MNLI=3

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True  # For base models without classifier
    )
    model.to(device)
    model.eval()

    # Load dataset
    dataset = load_dataset("glue", task)

    if task == "qqp":
        eval_data = dataset["validation"]
        text_cols = ("question1", "question2")
    elif task == "mnli":
        eval_data = dataset["validation_matched"]
        text_cols = ("premise", "hypothesis")
    else:
        raise ValueError(f"Unknown task: {task}")

    # Filter out invalid labels
    eval_data = eval_data.filter(lambda x: x["label"] != -1)

    all_preds = []
    all_labels = []

    # Evaluate in batches
    for i in tqdm(range(0, len(eval_data), batch_size), desc="Evaluating"):
        batch = eval_data[i:i+batch_size]

        # Tokenize
        inputs = tokenizer(
            batch[text_cols[0]],
            batch[text_cols[1]],
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)

        # Apply label mapping for models with non-standard label order
        if ("textattack" in model_name) and task == "mnli":
            label_map = torch.tensor([2, 0, 1], device=device)  # [neutral, contradiction, entailment] -> [entailment, neutral, contradiction]
            preds = label_map[preds]

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["label"])

    # Compute accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()

    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Samples: {len(all_labels)}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline models on GLUE tasks")
    parser.add_argument("--task", type=str, default="qqp", choices=["qqp", "mnli"])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    # Models to evaluate
    if args.task == "qqp":
        models = [
            # ("distilbert-base-uncased", "Student (base)"),
            # ("bert-base-uncased", "Teacher (base)"),
            ("textattack/bert-base-uncased-QQP", "Teacher (SFT)"),
        ]
    else:  # mnli
        models = [
            # ("distilbert-base-uncased", "Student (base)"),
            # ("bert-base-uncased", "Teacher (base)"),
            ("textattack/bert-base-uncased-MNLI", "Teacher (SFT)"),
        ]

    print(f"\n{'='*60}")
    print(f"Zero-shot Baseline Evaluation: {args.task.upper()}")
    print(f"{'='*60}")

    results = []
    for model_name, role in models:
        acc = evaluate_model(model_name, args.task, args.batch_size)
        results.append((model_name, role, acc))

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<45} {'Role':<20} {'Accuracy':<10}")
    print("-" * 75)
    for model_name, role, acc in results:
        print(f"{model_name:<45} {role:<20} {acc*100:.2f}%")


if __name__ == "__main__":
    main()
