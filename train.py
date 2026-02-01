"""Main training script for knowledge distillation."""

import argparse
import json
import logging
import os
import random
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import yaml

import numpy as np
import torch

# Module-level logger
logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False):
    """Configure logging level for the training run.

    Args:
        debug: If True, use DEBUG level; otherwise INFO level
    """
    level = logging.DEBUG if debug else logging.INFO
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" if debug else "%(message)s"

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set transformers logging
    import transformers
    transformers.logging.set_verbosity(logging.DEBUG if debug else logging.INFO)

    if debug:
        # Enable additional debug info
        logging.getLogger("src").setLevel(logging.DEBUG)
        logger.debug("DEBUG logging enabled")
    else:
        # Suppress verbose warnings in production
        warnings.filterwarnings("ignore")

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
from src.training.losses import (
    get_loss_strategy,
    CombinedDistillationLoss,
    FeatureLossStrategy,
    AttentionLossStrategy,
)
from src.training.layer_mapping import get_layer_mapping
from src.training.projections import create_projection_if_needed
from src.utils.checkpoint import TrainingState
from src.utils.config import load_config


def create_loss_strategy_from_config(
    distill_config: dict,
    teacher_model,
    student_model,
    device: str,
):
    """
    Create loss strategy from distillation config, supporting RFA combinations.

    Config format for single strategy:
        distillation:
          loss_type: kl_divergence
          alpha: 0.5

    Config format for combined strategies (RFA matrix):
        distillation:
          loss_strategies:
            - type: kl_divergence
              weight: 1.0
            - type: feature_mse
              weight: 0.5
            - type: attention_kl
              weight: 0.5
          alpha: 0.5
          layer_mapping:
            type: uniform

    Returns:
        Tuple of (loss_strategy, layer_mapping, projector)
    """
    alpha = distill_config.get("alpha", 0.5)
    temperature = distill_config.get("temperature", 2.0)

    # Check for combined strategies (RFA matrix)
    loss_strategies_config = distill_config.get("loss_strategies")
    if loss_strategies_config:
        # Multiple strategies: create CombinedDistillationLoss
        strategies = []
        weights = []

        for strat_config in loss_strategies_config:
            strat_type = strat_config["type"]
            strat_weight = strat_config.get("weight", 1.0)

            strategy = get_loss_strategy(
                strat_type,
                temperature=temperature,
                alpha=alpha,  # Each strategy gets same alpha
            )
            strategies.append(strategy)
            weights.append(strat_weight)

        combined = CombinedDistillationLoss(
            strategies=strategies,
            weights=weights,
            alpha=alpha,
        )
        loss_strategy = combined
    else:
        # Single strategy
        loss_type = distill_config.get("loss_type", "kl_divergence")
        loss_strategy = get_loss_strategy(
            loss_type,
            temperature=temperature,
            alpha=alpha,
        )

    # Setup layer mapping if needed (for feature/attention distillation)
    layer_mapping = None
    layer_mapping_config = distill_config.get("layer_mapping", {})
    if layer_mapping_config or loss_strategy.requires_hidden_states or loss_strategy.requires_attentions:
        mapping_type = layer_mapping_config.get("type", "uniform")

        # Get layer counts from model configs
        teacher_layers = getattr(teacher_model.config, "num_hidden_layers", 12)
        student_layers = getattr(student_model.config, "num_hidden_layers", 6)

        if mapping_type == "custom" and "mapping" in layer_mapping_config:
            # Explicit mapping provided
            layer_mapping = [
                (pair["teacher"], pair["student"])
                for pair in layer_mapping_config["mapping"]
            ]
        else:
            # Use a mapping strategy
            mapper = get_layer_mapping(mapping_type)
            layer_mapping = mapper.map(teacher_layers, student_layers)

        logger.debug(f"  Layer mapping ({mapping_type}): {layer_mapping}")

    # Setup projection if needed (for feature distillation with dimension mismatch)
    projector = None
    projection_config = distill_config.get("projection", {})
    if loss_strategy.requires_hidden_states:
        teacher_dim = getattr(teacher_model.config, "hidden_size", 768)
        student_dim = getattr(student_model.config, "hidden_size", 768)

        if teacher_dim != student_dim or projection_config.get("force", False):
            proj_type = projection_config.get("type", "linear")
            projector = create_projection_if_needed(
                teacher_dim=teacher_dim,
                student_dim=student_dim,
                projection_type=proj_type,
            )
            if projector is not None and device == "cuda":
                projector = projector.cuda()
            logger.debug(f"  Projection: {student_dim} -> {teacher_dim} ({proj_type})")

    return loss_strategy, layer_mapping, projector


def generate_config_id(config_path: str) -> str:
    """Generate a config identifier with task prefix.

    configs/mnli/bert_distilbert/ablations/02_frozen_sft-teacher_kl.yaml → mnli_02_frozen_sft-teacher_kl
    configs/qqp/bert_distilbert/ablations/00_no-teacher_ce-only.yaml → qqp_00_no-teacher_ce-only

    Full config path is stored in run_metadata for searchability.
    """
    path = Path(config_path)
    config_name = path.stem

    # Extract task from path: configs/{task}/...
    parts = path.parts
    if len(parts) > 1 and parts[0] == "configs":
        task = parts[1]
        return f"{task}_{config_name}"

    return config_name


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
                        help="Run name. Creates if not exists, resumes if exists. Use --force to overwrite.")
    parser.add_argument("--timestamp", action="store_true",
                        help="Append timestamp to --run_name (e.g., mnli_00 -> mnli_00_2026-01-31_12-00Z).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing run (deletes existing outputs and starts fresh).")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (for testing)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG level logging for verbose output")
    args = parser.parse_args()

    # Setup logging early
    setup_logging(debug=args.debug)

    # Timestamp for new runs
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%MZ")

    # Determine output directory and run name
    if args.run_name is not None:
        # Apply timestamp suffix if requested
        if args.timestamp:
            run_name = f"{args.run_name}_{timestamp}"
        else:
            run_name = args.run_name
        output_dir = Path(f"./outputs/{run_name}")
        if output_dir.exists():
            if args.force:
                # Force overwrite: delete existing and start fresh
                logger.info(f"--force specified. Deleting existing run: {output_dir}")
                shutil.rmtree(output_dir)
                if args.config is None:
                    sys.exit(
                        "\nError: --config is required for new runs.\n\n"
                        "Usage:\n"
                        "  python train.py --run_name NAME --config CONFIG [--force]\n"
                    )
                is_resume = False
                logger.info(f"Starting fresh run: {run_name}")
            else:
                # Resume existing run
                is_resume = True
                logger.info(f"Run '{run_name}' exists at {output_dir}/")
                logger.info(f"Resuming existing run. Use --force to overwrite.")
        else:
            # Create new run
            if args.config is None:
                sys.exit(
                    f"\nError: Run '{args.run_name}' does not exist and --config not provided.\n\n"
                    f"Usage:\n"
                    f"  New run:    python train.py --run_name {args.run_name} --config CONFIG\n"
                    f"  Resume:     python train.py --run_name EXISTING_RUN\n"
                    f"  Overwrite:  python train.py --run_name NAME --config CONFIG --force\n"
                )
            is_resume = False
            logger.info(f"Starting new run: {run_name}")
    else:
        # New run: auto-generate name from config
        if args.config is None:
            sys.exit(
                "\nError: No --run_name or --config provided.\n\n"
                "Usage:\n"
                "  New run:    python train.py --config CONFIG\n"
                "  Named run:  python train.py --run_name NAME --config CONFIG\n"
                "  Resume:     python train.py --run_name EXISTING_RUN\n"
            )
        if args.force:
            logger.warning("--force without --run_name is redundant (auto-generated names are always unique)")
        if args.timestamp:
            logger.warning("--timestamp without --run_name is redundant (auto-generated names include timestamps)")
        config_id = generate_config_id(args.config)
        run_name = f"{config_id}_{timestamp}"
        output_dir = Path(f"./outputs/{run_name}")
        is_resume = False
        logger.info(f"Starting new run: {run_name}")

    # Convert to string for compatibility
    args.output_dir = str(output_dir)

    # Set reproducibility seed early
    set_reproducibility_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")

    # Load configuration
    if is_resume:
        # Resume: load stored config, error if --config provided
        if args.config is not None:
            sys.exit(
                f"\nError: --config cannot be used when resuming.\n"
                f"The stored config at {output_dir / 'config.yaml'} will be used.\n\n"
                f"Usage:\n"
                f"  Resume:     python train.py --run_name {run_name}\n"
                f"  Overwrite:  python train.py --run_name {run_name} --config CONFIG --force\n"
            )
        config = load_config_snapshot(output_dir)
        if config is None:
            sys.exit(
                f"\nError: No stored config found at {output_dir / 'config.yaml'}\n"
                f"The run directory exists but appears corrupted.\n\n"
                f"Options:\n"
                f"  Delete and restart:  python train.py --run_name {run_name} --config CONFIG --force\n"
                f"  Or manually delete:  rm -rf {output_dir}\n"
            )
        logger.info(f"Loaded stored config from: {output_dir / 'config.yaml'}")
    else:
        # New run: --config already validated above
        logger.info(f"Loading config from {args.config}")
        config = load_config(args.config)

    logger.info(f"Output directory: {args.output_dir}")

    # Detect environment
    env = detect_environment()
    device = get_device()
    logger.info("=" * 60)
    logger.info(f"Environment: {env}")
    logger.info(f"Device: {device}")
    logger.info("=" * 60)

    # Initialize state tracking for SLURM job chaining
    state = TrainingState(args.output_dir)
    state.set_targets(epochs=config["training"]["num_epochs"])

    if state.is_complete():
        logger.info("Training already complete!")
        return

    # Load dataset
    logger.info("Loading dataset...")
    train_dataset, eval_dataset, tokenizer = load_dataset_for_task(config)
    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Eval size: {len(eval_dataset)}")

    # Determine number of labels for classification tasks
    task = config.get("data", {}).get("task", "").lower()
    if task == "mnli":
        num_labels = 3  # entailment, neutral, contradiction
    elif task == "qqp":
        num_labels = 2  # duplicate, not duplicate
    else:
        num_labels = None  # Not a classification task

    # Load teacher and student models
    logger.info("Loading models...")
    teacher, student = load_teacher_student(config, device, tokenizer, num_labels)

    # Find checkpoint to resume from
    resume_checkpoint = state.get_resume_checkpoint(args.output_dir)
    if resume_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")

    # Handle run metadata
    metadata = load_run_metadata(output_dir)
    if is_resume:
        if metadata:
            wandb_run_id = metadata.get("wandb_run_id")
            logger.info(f"Resuming run: wandb_run_id={wandb_run_id}")
        else:
            # Corrupted run: config.yaml exists but run_metadata.json missing
            # Regenerate metadata from available info
            wandb_run_id = truncate_for_wandb(run_name)
            metadata = {
                "run_name": run_name,
                "wandb_run_id": wandb_run_id,
                "config": "(recovered from config.yaml)",
                "created": "(unknown - recovered run)",
                "seed": args.seed,
            }
            save_run_metadata(output_dir, metadata)
            logger.warning(f"Recovered missing run_metadata.json for: {run_name}")
    else:
        # New run
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
        logger.info(f"Created run metadata: {output_dir / 'run_metadata.json'}")

    # Write .last_run for SLURM auto-resubmission (works with any run type)
    Path(".last_run").write_text(f"{run_name}\n{output_dir}\n")

    # Setup W&B
    wandb_project = "Test" if args.debug else "Knowledge Distillation"
    os.environ.setdefault("WANDB_PROJECT", wandb_project)
    os.environ["WANDB_RUN_ID"] = wandb_run_id
    os.environ["WANDB_RESUME"] = "allow"

    report_to = config["logging"]["report_to"]
    if os.environ.get("WANDB_MODE") == "disabled":
        report_to = "none"
        logger.info("W&B disabled - logging to console only")
    else:
        logger.info(f"W&B run ID: {wandb_run_id}")

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

    # Create loss strategy (supports single and combined RFA strategies)
    logger.info("Setting up distillation loss strategy...")
    loss_strategy, layer_mapping, projector = create_loss_strategy_from_config(
        config["distillation"],
        teacher,
        student,
        device,
    )

    # Initialize trainer with configurable loss strategy
    trainer = DistillationTrainer(
        teacher_model=teacher,
        temperature=config["distillation"]["temperature"],
        alpha=config["distillation"]["alpha"],
        freeze_teacher=config["distillation"].get("freeze_teacher", True),
        loss_strategy=loss_strategy,  # Use pre-built strategy (supports RFA combinations)
        teacher_label_map=config["distillation"].get("teacher_label_map"),
        layer_mapping=layer_mapping,
        projector=projector,
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Add SLURM signal handler
    if env == "slurm":
        logger.info("Adding SLURM signal handler...")
        trainer.add_callback(SLURMSignalCallback(trainer))

    # Train
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

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
                logger.info(f"W&B ID changed: {old_id} → {wandb.run.id}")
    except Exception:
        pass  # W&B not available or disabled

    # Update state
    state.save(
        epoch=trainer.state.epoch,
        step=trainer.state.global_step
    )

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()

    # Evaluate
    logger.info("Final evaluation:")
    metrics = trainer.evaluate()
    logger.info(metrics)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
