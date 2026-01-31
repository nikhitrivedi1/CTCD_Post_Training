"""Main training script for knowledge distillation."""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

import numpy as np
import torch

# Add project root to path for running without package install
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import environment first - this auto-loads .env file if it exists
# (override=False means Colab/HPC env vars take precedence)
from src.utils.environment import detect_environment, get_device

from transformers import TrainingArguments, set_seed


def set_reproducibility_seed(seed: int = 42):
    """Set seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # HuggingFace transformers seed
    # For fully deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from src.data.datasets import load_dataset_for_task
from src.evaluation.metrics import get_compute_metrics
from src.models.loader import load_teacher_student
from src.training.callbacks import SLURMSignalCallback
from src.training.trainer import DistillationTrainer
from src.utils.checkpoint import TrainingState
from src.utils.config import load_config


def generate_config_id(config_path: str) -> str:
    """Generate a config identifier from the config filename.

    configs/mnli/bert_distilbert/ablations/02_frozen_sft-teacher_kl.yaml → 02_frozen_sft-teacher_kl

    Full config path is stored in run_metadata for searchability.
    """
    return Path(config_path).stem


def truncate_for_wandb(run_id: str, max_length: int = 64) -> str:
    """Truncate run_id for W&B's 64-char limit, preserving timestamp suffix.

    Format: {config_name}_{timestamp} where timestamp is YYYY-MM-DD_HH-MMZ (17 chars).
    If truncation needed, shortens config_name while keeping full timestamp.
    """
    if len(run_id) <= max_length:
        return run_id

    timestamp_len = 17  # YYYY-MM-DD_HH-MMZ
    if len(run_id) > timestamp_len + 1:
        timestamp = run_id[-timestamp_len:]
        config_name = run_id[:-(timestamp_len + 1)]
        max_config_len = max_length - timestamp_len - 1
        return f"{config_name[:max_config_len]}_{timestamp}"

    return run_id[:max_length]


def load_run_metadata(output_dir: Path) -> dict | None:
    """Load run metadata if it exists."""
    metadata_path = output_dir / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    return None


def save_run_metadata(output_dir: Path, metadata: dict):
    """Save run metadata to output directory."""
    metadata_path = output_dir / "run_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def save_config_snapshot(output_dir: Path, config: dict):
    """Save full resolved config to output directory for reproducibility."""
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return config_path


def load_config_snapshot(output_dir: Path) -> dict | None:
    """Load stored config from output directory."""
    config_path = output_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file (required for new runs, ignored for resume)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name. With --new, creates new run. Without --new, resumes existing run.")
    parser.add_argument("--new", action="store_true",
                        help="Create a new run with --run_name (fails if exists). Without this, --run_name resumes.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="[Deprecated] Use --run_name instead. Direct output directory override.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (for testing)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Timestamp for new runs
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%MZ")

    # Determine output directory and run name
    # Priority: --output_dir (legacy) > --run_name > auto-generate (new run)
    if args.output_dir is not None:
        # Legacy mode: direct output_dir specification
        if args.new:
            print("Warning: --new is ignored when using --output_dir (deprecated)")
        output_dir = Path(args.output_dir)
        run_name = output_dir.name
        is_resume = output_dir.exists()
        if is_resume:
            print(f"Resuming from existing output directory: {output_dir}")
        elif args.config is None:
            sys.exit("\nError: --config is required for new runs.\n")
    elif args.run_name is not None:
        output_dir = Path(f"./outputs/{args.run_name}")
        run_name = args.run_name
        if args.new:
            # Explicit new run with custom name
            if args.config is None:
                sys.exit("\nError: --config is required for new runs.\n")
            if output_dir.exists():
                sys.exit(
                    f"\nError: Run '{args.run_name}' already exists at {output_dir}\n"
                    f"Remove --new to resume this run, or choose a different name.\n"
                )
            is_resume = False
            print(f"Starting new run: {run_name}")
        else:
            # Resume mode: run_name must exist
            if not output_dir.exists():
                sys.exit(
                    f"\nError: Run '{args.run_name}' not found at {output_dir}\n"
                    f"To start a new run with this name, add --new.\n"
                    f"To resume, ensure the run exists.\n"
                )
            is_resume = True
            print(f"Resuming run: {run_name}")
    else:
        # New run: auto-generate name from config
        if args.config is None:
            sys.exit("\nError: --config is required for new runs.\n")
        if args.new:
            print("Warning: --new without --run_name is redundant (auto-generated names are always new)")
        config_id = generate_config_id(args.config)
        run_name = f"{config_id}_{timestamp}"
        output_dir = Path(f"./outputs/{run_name}")
        is_resume = False
        print(f"Starting new run: {run_name}")

    # Convert to string for compatibility
    args.output_dir = str(output_dir)

    # Set reproducibility seed early
    set_reproducibility_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    # Load configuration
    if is_resume:
        # Resume: load stored config, error if --config provided
        if args.config is not None:
            sys.exit(
                f"\nError: --config cannot be used when resuming.\n"
                f"The stored config at {output_dir / 'config.yaml'} will be used.\n"
            )
        config = load_config_snapshot(output_dir)
        if config is None:
            sys.exit(f"\nError: No stored config found at {output_dir / 'config.yaml'}\n")
        print(f"Loaded stored config from: {output_dir / 'config.yaml'}")
    else:
        # New run: --config already validated above
        print(f"Loading config from {args.config}")
        config = load_config(args.config)

    print(f"Output directory: {args.output_dir}")

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

    # Handle run metadata
    metadata = load_run_metadata(output_dir)
    if is_resume and metadata:
        wandb_run_id = metadata.get("wandb_run_id")
        print(f"Resuming run: wandb_run_id={wandb_run_id}")
    else:
        config_id = generate_config_id(args.config)
        wandb_run_id = truncate_for_wandb(run_name)
        metadata = {
            "run_name": run_name,
            "wandb_run_id": wandb_run_id,
            "config": args.config,
            "config_id": config_id,
            "created": timestamp,
            "seed": args.seed,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        save_run_metadata(output_dir, metadata)
        save_config_snapshot(output_dir, config)
        print(f"Created run metadata: {output_dir / 'run_metadata.json'}")

    # Setup W&B
    os.environ.setdefault("WANDB_PROJECT", "knowledge-distillation")
    os.environ["WANDB_RUN_ID"] = wandb_run_id
    os.environ["WANDB_RESUME"] = "allow"

    report_to = config["logging"]["report_to"]
    if os.environ.get("WANDB_MODE") == "disabled":
        report_to = "none"
        print("W&B disabled - logging to console only")
    else:
        print(f"W&B run ID: {wandb_run_id}")

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
        report_to=report_to,
        fp16=config["training"].get("fp16", True) if device == "cuda" else False,
        dataloader_num_workers=config["training"].get("dataloader_num_workers", 4),
        dataloader_pin_memory=device == "cuda",  # Disable on MPS (not supported)
        run_name=wandb_run_id,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=args.seed,
        data_seed=args.seed,
    )

    # Get compute metrics function
    compute_metrics = get_compute_metrics(task, tokenizer=tokenizer)

    # Initialize trainer with configurable loss strategy
    trainer = DistillationTrainer(
        teacher_model=teacher,
        temperature=config["distillation"]["temperature"],
        alpha=config["distillation"]["alpha"],
        freeze_teacher=config["distillation"].get("freeze_teacher", True),
        loss_type=config["distillation"].get("loss_type", "kl_divergence"),
        teacher_label_map=config["distillation"].get("teacher_label_map"),
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

    # Update W&B with full config and sync metadata
    try:
        import wandb
        if wandb.run:
            # Log full config to W&B for dashboard visibility
            wandb.config.update(config, allow_val_change=True)
            # Sync metadata if W&B assigned a different ID (e.g., original was deleted)
            if wandb.run.id != metadata.get("wandb_run_id"):
                old_id = metadata.get("wandb_run_id")
                metadata["wandb_run_id"] = wandb.run.id
                metadata.setdefault("wandb_run_id_history", []).append(old_id)
                save_run_metadata(output_dir, metadata)
                print(f"W&B ID changed: {old_id} → {wandb.run.id}")
    except Exception:
        pass  # W&B not available or disabled

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
