#!/usr/bin/env python
"""
Fine-tune classification head with frozen encoder.

Usage:
    python workbench/finetune_head.py --model outputs/checkpoint --task mnli
    python workbench/finetune_head.py --model outputs/checkpoint --task mnli --epochs 3 --lr 1e-3
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class HeadFinetuneConfig:
    """Configuration for head fine-tuning."""

    model_path: str = field(metadata={"help": "Path to model checkpoint"})
    task: str = field(default="mnli", metadata={"help": "Task name"})
    output_dir: str = field(default="workbench/results/head_finetune")
    num_epochs: int = field(default=3)
    learning_rate: float = field(default=1e-3)  # Higher LR for head-only
    batch_size: int = field(default=32)
    max_samples: Optional[int] = field(default=None)
    seed: int = field(default=42)


def freeze_encoder(model):
    """Freeze all parameters except the classification head."""
    frozen_count = 0
    trainable_count = 0

    for name, param in model.named_parameters():
        # Keep classifier/head trainable
        if "classifier" in name or "head" in name or "pooler" in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    logger.info(f"Frozen: {frozen_count:,} params")
    logger.info(f"Trainable: {trainable_count:,} params")

    return model


def load_task_data(task: str, tokenizer, max_samples: Optional[int] = None):
    """Load and tokenize task data."""
    if task == "mnli":
        dataset = load_dataset("glue", "mnli")
        text_cols = ("premise", "hypothesis")
        label_col = "label"
        train_split = "train"
        val_split = "validation_matched"
    elif task == "qqp":
        dataset = load_dataset("glue", "qqp")
        text_cols = ("question1", "question2")
        label_col = "label"
        train_split = "train"
        val_split = "validation"
    elif task == "sst2":
        dataset = load_dataset("glue", "sst2")
        text_cols = ("sentence",)
        label_col = "label"
        train_split = "train"
        val_split = "validation"
    elif task == "cola":
        dataset = load_dataset("glue", "cola")
        text_cols = ("sentence",)
        label_col = "label"
        train_split = "train"
        val_split = "validation"
    else:
        raise ValueError(f"Unknown task: {task}")

    def tokenize(examples):
        if len(text_cols) == 2:
            return tokenizer(
                examples[text_cols[0]],
                examples[text_cols[1]],
                truncation=True,
                max_length=128,
            )
        else:
            return tokenizer(
                examples[text_cols[0]],
                truncation=True,
                max_length=128,
            )

    train_data = dataset[train_split]
    val_data = dataset[val_split]

    if max_samples:
        train_data = train_data.select(range(min(max_samples, len(train_data))))
        val_data = val_data.select(range(min(max_samples // 5, len(val_data))))

    remove_cols = [c for c in train_data.column_names if c != label_col]
    train_data = train_data.map(tokenize, batched=True, remove_columns=remove_cols)
    val_data = val_data.map(tokenize, batched=True, remove_columns=remove_cols)

    train_data = train_data.rename_column(label_col, "labels")
    val_data = val_data.rename_column(label_col, "labels")

    return train_data, val_data


def compute_metrics(eval_pred):
    """Compute accuracy."""
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}


def run_head_finetune(config: HeadFinetuneConfig):
    """
    Fine-tune classification head with frozen encoder.

    Args:
        config: HeadFinetuneConfig with all parameters

    Returns:
        Dict with results
    """
    logger.info("=" * 60)
    logger.info("HEAD FINE-TUNING")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_path}")
    logger.info(f"Task: {config.task}")

    # Load model and tokenizer
    logger.info("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(config.model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    # Freeze encoder
    logger.info("Freezing encoder...")
    model = freeze_encoder(model)

    # Load data
    logger.info("Loading data...")
    train_data, val_data = load_task_data(config.task, tokenizer, config.max_samples)
    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Val samples: {len(val_data)}")

    # Evaluate before fine-tuning
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Training args
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=100,
        report_to="none",
        seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # Evaluate before training
    logger.info("Evaluating before fine-tuning...")
    pre_results = trainer.evaluate()
    logger.info(f"Pre-finetune accuracy: {pre_results['eval_accuracy']:.2%}")

    # Train
    logger.info("Training...")
    trainer.train()

    # Evaluate after
    logger.info("Evaluating after fine-tuning...")
    post_results = trainer.evaluate()
    logger.info(f"Post-finetune accuracy: {post_results['eval_accuracy']:.2%}")

    # Save
    trainer.save_model(str(output_dir / "final"))

    results = {
        "model": config.model_path,
        "task": config.task,
        "pre_finetune_accuracy": pre_results["eval_accuracy"],
        "post_finetune_accuracy": post_results["eval_accuracy"],
        "improvement": post_results["eval_accuracy"] - pre_results["eval_accuracy"],
        "config": {
            "num_epochs": config.num_epochs,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
        },
    }

    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Pre-finetune:  {results['pre_finetune_accuracy']:.2%}")
    logger.info(f"Post-finetune: {results['post_finetune_accuracy']:.2%}")
    logger.info(f"Improvement:   {results['improvement']:+.2%}")

    return results


def resolve_output_dir(args) -> Path:
    """
    Resolve output directory following forge.py conventions.

    Args:
        args: Parsed arguments

    Returns:
        Path to output directory
    """
    base_dir = Path("workbench/outputs")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%MZ")

    if args.run_name:
        # Manual naming
        if args.timestamp:
            run_name = f"{args.run_name}_{timestamp}"
        else:
            run_name = args.run_name
        output_dir = base_dir / run_name

        if output_dir.exists():
            if args.force:
                import shutil
                logger.info(f"--force specified. Deleting existing run: {output_dir}")
                shutil.rmtree(output_dir)
            else:
                logger.error(
                    f"\nError: Run '{args.run_name}' already exists at {output_dir}/\n\n"
                    f"Options:\n"
                    f"  Overwrite:  python workbench/finetune_head.py --run_name {args.run_name} --model ... --force\n"
                    f"  New name:   python workbench/finetune_head.py --run_name NEW_NAME --model ...\n"
                    f"  Timestamp:  python workbench/finetune_head.py --run_name {args.run_name} --model ... --timestamp\n"
                )
                sys.exit(1)
    else:
        # Auto-naming from model basename
        model_basename = Path(args.model).name
        run_name = f"{model_basename}_{args.task}_{timestamp}"
        output_dir = base_dir / run_name

    logger.info(f"Output directory: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune classification head with frozen encoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name. Use --force to overwrite existing.")
    parser.add_argument("--timestamp", action="store_true",
                        help="Append timestamp to --run_name")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing run directory")
    parser.add_argument("--model", type=str, required=True,
                        help="Model checkpoint to fine-tune")
    parser.add_argument("--task", type=str, default="mnli",
                        choices=["mnli", "qqp", "sst2", "cola"])
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3, higher for head-only)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max training samples (default: full dataset)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Resolve output directory
    output_dir = resolve_output_dir(args)

    config = HeadFinetuneConfig(
        model_path=args.model,
        task=args.task,
        output_dir=str(output_dir),
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    results = run_head_finetune(config)

    # Save results JSON in output directory
    output_path = output_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
