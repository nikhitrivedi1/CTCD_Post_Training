"""Main training script for knowledge distillation."""

import argparse
import os
import sys

# Add project root to path for running without package install
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import TrainingArguments

from src.data.datasets import load_dataset_for_task
from src.evaluation.metrics import get_compute_metrics
from src.models.loader import load_teacher_student
from src.training.callbacks import SLURMSignalCallback
from src.training.trainer import DistillationTrainer
from src.utils.checkpoint import TrainingState
from src.utils.config import load_config
from src.utils.environment import detect_environment, get_device


def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (for testing)")
    args = parser.parse_args()

    # Load configuration
    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Detect environment
    env = detect_environment()
    device = get_device()
    print(f"\n{'=' * 60}")
    print(f"Environment: {env}")
    print(f"Device: {device}")
    print(f"{'=' * 60}\n")

    # Initialize state tracking for SLURM job chaining
    state = TrainingState(args.output_dir)
    state.set_targets(epochs=config["training"]["num_epochs"])

    if state.is_complete():
        print("Training already complete!")
        return

    # Load dataset
    print("Loading dataset...")
    train_dataset, eval_dataset, tokenizer = load_dataset_for_task(config)
    print(f"Train size: {len(train_dataset)}")
    print(f"Eval size: {len(eval_dataset)}")

    # Determine number of labels for classification tasks
    task = config.get("data", {}).get("task", "").lower()
    if task == "mnli":
        num_labels = 3  # entailment, neutral, contradiction
    elif task == "qqp":
        num_labels = 2  # duplicate, not duplicate
    else:
        num_labels = None  # Not a classification task

    # Load teacher and student models
    print("\nLoading models...")
    teacher, student = load_teacher_student(config, device, tokenizer, num_labels)

    # Find checkpoint to resume from
    resume_checkpoint = state.get_resume_checkpoint(args.output_dir)
    if resume_checkpoint:
        print(f"\nResuming from checkpoint: {resume_checkpoint}")

    # Setup W&B
    os.environ.setdefault("WANDB_PROJECT", "knowledge-distillation")
    run_name = f"{config['model']['teacher']}_to_{config['model']['student']}_{task}"

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"].get("eval_batch_size", config["training"]["batch_size"]),
        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),
        learning_rate=config["training"]["learning_rate"],
        warmup_steps=config["training"]["warmup_steps"],
        weight_decay=config["training"]["weight_decay"],
        max_grad_norm=config["training"]["max_grad_norm"],
        logging_steps=config["logging"]["logging_steps"],
        save_steps=config["checkpointing"]["save_steps"],
        save_total_limit=config["checkpointing"]["save_total_limit"],
        eval_strategy=config.get("evaluation", {}).get("eval_strategy", "steps"),
        eval_steps=config.get("evaluation", {}).get("eval_steps", 500),
        report_to=config["logging"]["report_to"],
        fp16=config["training"].get("fp16", True) if device == "cuda" else False,
        dataloader_num_workers=config["training"].get("dataloader_num_workers", 4),
        run_name=run_name,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Get compute metrics function
    compute_metrics = get_compute_metrics(task, tokenizer=tokenizer)

    # Initialize trainer
    trainer = DistillationTrainer(
        teacher_model=teacher,
        temperature=config["distillation"]["temperature"],
        alpha=config["distillation"]["alpha"],
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Add SLURM signal handler
    if env == "slurm":
        print("Adding SLURM signal handler...")
        trainer.add_callback(SLURMSignalCallback(trainer))

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Update state
    state.save(
        epoch=trainer.state.epoch,
        step=trainer.state.global_step
    )

    # Save final model
    print("\nSaving final model...")
    trainer.save_model()

    # Evaluate
    print("\nFinal evaluation:")
    metrics = trainer.evaluate()
    print(metrics)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
