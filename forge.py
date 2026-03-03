"""Main training script for knowledge distillation."""

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
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
    format_str = "%(asctime)sZ - %(name)s - %(levelname)s - %(message)s" if debug else "%(asctime)sZ - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.Formatter.converter = time.gmtime

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


def add_file_logging(output_dir: Path, debug: bool = False):
    """Add file logging to capture all output to the run's output directory.

    Uses TeeWriter to capture stdout/stderr (print, tqdm, wandb) plus FileHandler
    for Python logging. Handles \\r so progress bars don't spam the log file.

    Args:
        output_dir: Path to the output directory
        debug: If True, use DEBUG level; otherwise INFO level
    """
    from utils.logging import setup_file_logging

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "output.log"

    setup_file_logging(log_file, update_transformers=True)
    logger.info(f"Logging to file: {log_file}")

# Add project root to path for running without package install
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import environment first - this auto-loads .env file if it exists
# (override=False means Colab/HPC env vars take precedence)
from src.utils.environment import detect_environment, get_device

from transformers import Trainer, TrainingArguments, set_seed, DataCollatorForSeq2Seq


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
from src.data.collators import VLMDataCollator
from src.evaluation import get_compute_metrics, run_evaluation
from src.tasks import get_num_labels, get_primary_metric, is_ocr_task
from src.models import load_models
from src.training.callbacks import (
    SLURMSignalCallback, TeacherEvalCallback, TranslationMetricsCallback, PeakMetricsCallback,
)
from src.training.trainer import DistillationTrainer
from src.training.losses import (
    get_loss_strategy,
    CombinedDistillationLoss,
    StagedDistillationLoss,
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
    Create loss strategy from distillation config.

    Supports three modes:
    1. Single strategy (loss_type)
    2. Combined strategies (loss_strategies) - for FAR matrix
    3. Staged strategies (stages) - for F→R, F→A→R progressions

    Config format for single strategy:
        distillation:
          loss_type: kl_divergence
          alpha: 0.5

    Config format for combined strategies (FAR matrix):
        distillation:
          loss_strategies:
            - type: kl_divergence
              weight: 1.0
            - type: feature_mse
              weight: 0.5

    Config format for staged distillation:
        distillation:
          stages:
            - strategy: feature_mse
              until_progress: 0.5
            - strategy: kl_divergence
              until_progress: 1.0

    Returns:
        Tuple of (loss_strategy, layer_mapping, projector)
    """
    alpha = distill_config.get("alpha", 0.5)
    temperature = distill_config.get("temperature", 2.0)

    # Check for staged distillation (F→R, F→A→R, etc.)
    stages_config = distill_config.get("stages")
    if stages_config:
        loss_strategy = StagedDistillationLoss(
            stages=stages_config,
            alpha=alpha,
            temperature=temperature,
        )
    # Check for combined strategies (FAR matrix)
    elif loss_strategies_config := distill_config.get("loss_strategies"):
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
            if not mapper.validate(teacher_layers, student_layers):
                raise ValueError(
                    f"Invalid layer mapping '{mapping_type}' for "
                    f"teacher={teacher_layers}, student={student_layers} layers"
                )
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
            if projector is not None:
                projector = projector.to(device)
            logger.debug(f"  Projection: {student_dim} -> {teacher_dim} ({proj_type})")

    return loss_strategy, layer_mapping, projector


def generate_config_id(config_path: str) -> str:
    """Generate a config identifier from path.

    Format: {task}_{experiment}_{config}

    Examples:
        configs/classification/mnli/bert_distilbert/staged/02_a-r.yaml → mnli_staged_02-a-r
        configs/translation/europarl/llama3p2-3b_llama3p2-1b/far/07_far.yaml → europarl_far_07_far

    Full config path is stored in run_metadata for searchability.
    """
    path = Path(config_path)
    config_name = path.stem  # e.g., "02_a-r"
    parts = path.parts

    # Find 'configs' in path (handles absolute paths)
    try:
        configs_idx = parts.index("configs")
        parts = parts[configs_idx:]
    except ValueError:
        return config_name

    # configs/{type}/{task}/{model_pair}/{experiment}/{config}.yaml
    if len(parts) >= 6:
        task = parts[2]        # mnli, sst2, cola, qqp, europarl
        experiment = parts[4]  # ablations, far, staged
        return f"{task}_{experiment}_{config_name}"
    elif len(parts) >= 3:
        # Fallback: just task + config
        task = parts[2] if len(parts) > 2 else parts[1]
        return f"{task}_{config_name}"

    return config_name


def generate_wandb_group(config_path: str) -> str:
    """Generate W&B group name from config path.

    Format: {task}_{experiment}

    Path structure: configs/{type}/{task}/{model_pair}/{experiment}/{config}.yaml

    Examples:
        configs/classification/mnli/bert_distilbert/staged/02_a-r.yaml → mnli_staged
        configs/translation/europarl/llama3p2-3b_llama3p2-1b/far/07_far.yaml → europarl_far
    """
    path = Path(config_path)
    parts = path.parts

    # Find 'configs' in path (handles absolute paths)
    try:
        configs_idx = parts.index("configs")
        parts = parts[configs_idx:]  # Slice from 'configs' onward
    except ValueError:
        return None

    # configs/{type}/{task}/{model_pair}/{experiment}/{config}.yaml
    if len(parts) >= 6:
        task = parts[2]        # mnli, sst2, cola, qqp, europarl
        experiment = parts[4]  # ablations, far, staged
        return f"{task}_{experiment}"

    return None


def generate_wandb_tags(config_path: str) -> list:
    """Generate W&B tags from config path.

    Returns tags for: task, experiment, model_pair

    Path structure: configs/{type}/{task}/{model_pair}/{experiment}/{config}.yaml

    Examples:
        configs/classification/mnli/bert_distilbert/staged/02_a-r.yaml
        → ["mnli", "staged", "bert_distilbert"]
    """
    path = Path(config_path)
    parts = path.parts

    # Find 'configs' in path (handles absolute paths)
    try:
        configs_idx = parts.index("configs")
        parts = parts[configs_idx:]  # Slice from 'configs' onward
    except ValueError:
        return []

    # configs/{type}/{task}/{model_pair}/{experiment}/{config}.yaml
    if len(parts) >= 6:
        task = parts[2]        # mnli, sst2, cola, qqp, europarl
        model_pair = parts[3]  # bert_distilbert, llama3p2-3b_llama3p2-1b
        experiment = parts[4]  # ablations, far, staged
        return [task, experiment, model_pair]

    return []


def truncate_for_wandb(run_id: str, max_length: int = 64) -> str:
    """Truncate run_id for W&B's 64-char limit, preserving timestamp suffix.

    Format: {task}_{experiment}_{config}_{timestamp}
    Timestamp format: YYYYMMDD-HHMMZ (14 chars)

    If truncation needed, shortens the prefix while keeping full timestamp.
    """
    if len(run_id) <= max_length:
        return run_id

    timestamp_len = 14  # YYYYMMDD-HHMMZ
    if len(run_id) > timestamp_len + 1:
        timestamp = run_id[-timestamp_len:]
        config_name = run_id[:-(timestamp_len + 1)]
        max_config_len = max_length - timestamp_len - 1
        return f"{config_name[:max_config_len]}_{timestamp}"

    return run_id[:max_length]


def load_incomplete_jobs() -> dict:
    """Load incomplete jobs registry.

    Returns:
        Dict mapping job_id to {"run_name": ..., "output_dir": ...}
    """
    path = Path(".incomplete_jobs")
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_incomplete_jobs(jobs: dict):
    """Save incomplete jobs registry atomically.

    Args:
        jobs: Dict mapping job_id to {"run_name": ..., "output_dir": ...}
    """
    path = Path(".incomplete_jobs")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(jobs, indent=2))
    tmp.rename(path)


def get_job_key() -> str:
    """Get unique key for current job (SLURM_JOB_ID or local_{pid})."""
    return os.environ.get("SLURM_JOB_ID") or f"local_{os.getpid()}"


def remove_incomplete_job(job_key: str = None):
    """Remove a job from incomplete jobs registry.

    Args:
        job_key: Job ID to remove. Defaults to current job.
    """
    if job_key is None:
        job_key = get_job_key()

    jobs = load_incomplete_jobs()
    if job_key in jobs:
        del jobs[job_key]
        save_incomplete_jobs(jobs)
        logger.info(f"Removed job from incomplete registry: {job_key}")


def add_incomplete_job(run_name: str, output_dir: Path):
    """Add current job to incomplete jobs registry.

    Uses SLURM_JOB_ID if available, otherwise falls back to local_{pid}.
    Also removes previous job entry if PREV_JOB_ID is set (for SLURM resubmission).

    Args:
        run_name: The actual run name (with timestamp if applicable)
        output_dir: Path to the output directory
    """
    # Remove previous job entry if this is a resubmission
    prev_job_id = os.environ.get("PREV_JOB_ID")
    if prev_job_id:
        remove_incomplete_job(prev_job_id)

    # Add current job
    job_key = get_job_key()
    jobs = load_incomplete_jobs()
    jobs[job_key] = {
        "type": "training",
        "run_name": run_name,
        "output_dir": str(output_dir),
    }
    save_incomplete_jobs(jobs)
    logger.info(f"Added job to incomplete registry: {job_key}")


def load_run_metadata(output_dir: Path) -> dict | None:
    """Load run metadata if it exists."""
    metadata_path = output_dir / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    return None


def delete_wandb_run(run_id: str, project: str = None) -> bool:
    """Delete a W&B run by ID.

    Args:
        run_id: The W&B run ID to delete
        project: W&B project name (defaults to WANDB_PROJECT env var)

    Returns:
        True if deleted, False if not found or error
    """
    try:
        import wandb
        api = wandb.Api()
        project = project or os.environ.get("WANDB_PROJECT", "Forge-Default")
        entity = os.environ.get("WANDB_ENTITY") or api.default_entity
        run_path = f"{entity}/{project}/{run_id}"
        run = api.run(run_path)
        run.delete()
        logger.info(f"Deleted W&B run: {run_path}")
        return True
    except Exception as e:
        logger.info(f"W&B run not deleted ({run_id}): {e}")
        return False


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


def _train(
    config, device, tokenizer, num_labels, task,
    train_dataset, eval_dataset, compute_metrics, data_collator,
    output_dir, run_name, args, state, resume_checkpoint,
    env, report_to, metadata,
):
    """Run the training phase with models scoped locally.

    All model references (teacher, student, trainer, loss_strategy, projector)
    are local to this function and GC'd on return.

    Args:
        config: Full configuration dictionary
        device: Device string ('cuda', 'mps', 'cpu')
        tokenizer: Tokenizer
        num_labels: Number of labels for classification (None for translation)
        task: Task name
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        compute_metrics: Metrics computation function
        data_collator: Data collator (or None for classification)
        output_dir: Output directory (Path)
        run_name: Run name for W&B
        args: Parsed CLI arguments
        state: TrainingState for checkpoint tracking
        resume_checkpoint: Path to resume checkpoint (or None)
        env: Environment string ('local', 'slurm', etc.)
        report_to: Logging target ('wandb', 'none', etc.)
        metadata: Run metadata dict
    """
    # Load teacher and student models
    logger.info("Loading models...")
    teacher, student = load_models(config, device, tokenizer, num_labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
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
        fp16=(config["training"].get("fp16", True) and student.dtype != torch.float16) if device == "cuda" else False,
        dataloader_num_workers=config["training"].get("dataloader_num_workers", 4) if device == "cuda" else 0,  # MPS requires 0 workers
        dataloader_pin_memory=device == "cuda",  # Disable on MPS (not supported)
        run_name=run_name,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        load_best_model_at_end=True,
        # Select best checkpoint by task metric (accuracy/f1/mcc), not loss.
        # Loss and metric can diverge; metric-based selection ensures we report the best achieved.
        # For translation: use loss since BLEU is computed separately by callback (generation-based),
        # not in eval metrics (which only has loss due to prediction_loss_only=True to avoid OOM).
        metric_for_best_model="eval_loss" if num_labels is None else (f"eval_{get_primary_metric(task)}" if get_primary_metric(task) else "eval_loss"),
        greater_is_better=False if num_labels is None else (get_primary_metric(task) is not None),  # Loss: lower is better; accuracy/f1/mcc: higher is better
        prediction_loss_only=num_labels is None,  # Seq2seq tasks: prevents OOM, metrics need generation anyway
        seed=args.seed,
        data_seed=args.seed,
        # VLM datasets store raw images + text; the collator handles tokenization.
        # Must keep these columns so the collator can process them.
        remove_unused_columns=not is_ocr_task(task),
    )

    # Choose trainer based on whether we have a teacher
    if teacher is None:
        # CE-only baseline: use standard HuggingFace Trainer
        logger.info("No teacher - using standard Trainer (CE-only mode)")
        trainer = Trainer(
            model=student,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )
    else:
        # Distillation: create loss strategy and use DistillationTrainer
        logger.info("Setting up distillation loss strategy...")
        loss_strategy, layer_mapping, projector = create_loss_strategy_from_config(
            config["distillation"],
            teacher,
            student,
            device,
        )

        trainer = DistillationTrainer(
            teacher_model=teacher,
            temperature=config["distillation"]["temperature"],
            alpha=config["distillation"]["alpha"],
            freeze_teacher=config["distillation"].get("freeze_teacher", True),
            loss_strategy=loss_strategy,
            teacher_label_map=config["distillation"].get("teacher_label_map"),
            layer_mapping=layer_mapping,
            projector=projector,
            model=student,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )

    # Add SLURM signal handler
    if env == "slurm":
        logger.info("Adding SLURM signal handler...")
        trainer.add_callback(SLURMSignalCallback(trainer))

    # Add teacher eval during training (co-training only, before PeakMetrics so peaks are tracked)
    if isinstance(trainer, DistillationTrainer) and not trainer.freeze_teacher:
        logger.info("Adding teacher eval callback (co-training)...")
        trainer.add_callback(TeacherEvalCallback(
            trainer, teacher_label_map=trainer.teacher_label_map,
        ))

    # Add peak metrics tracking
    logger.info("Adding peak metrics callback...")
    trainer.add_callback(PeakMetricsCallback())

    # Add translation metrics callback for seq2seq tasks
    if num_labels is None and config.get("evaluation", {}).get("compute_translation_metrics", False):
        num_samples = config.get("evaluation", {}).get("translation_metrics_samples", 200)
        source_lang = config.get("data", {}).get("source_lang", "en")
        target_lang = config.get("data", {}).get("target_lang", "fr")

        logger.info(f"Adding translation metrics callback ({num_samples} samples per epoch)...")
        metrics_callback = TranslationMetricsCallback(
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            source_lang=source_lang,
            target_lang=target_lang,
            num_samples=num_samples,
            max_new_tokens=config.get("evaluation", {}).get("max_new_tokens", 128),
        )
        trainer.add_callback(metrics_callback)

    # =========================================================================
    # Training
    # =========================================================================
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
            # Log primary metric (metric_for_best_model is set automatically by HF Trainer)
            wandb.config.update({
                "primary_metric": get_primary_metric(task) or "loss",
            }, allow_val_change=True)
            # Sync metadata if W&B assigned a different ID (e.g., original was deleted)
            if wandb.run.id != metadata.get("wandb_run_id"):
                old_id = metadata.get("wandb_run_id")
                metadata["wandb_run_id"] = wandb.run.id
                metadata.setdefault("wandb_run_id_history", []).append(old_id)
                save_run_metadata(output_dir, metadata)
                logger.info(f"W&B ID changed: {old_id} → {wandb.run.id}")
    except ImportError:
        pass  # W&B not installed
    except Exception as e:
        logger.debug(f"W&B config update failed: {e}")

    # Update state
    state.save(
        epoch=trainer.state.epoch,
        step=trainer.state.global_step
    )

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()

    # Save co-trained teacher (weights changed during training)
    if teacher is not None and not config["distillation"].get("freeze_teacher", True):
        teacher_dir = output_dir / "teacher"
        logger.info(f"Saving co-trained teacher to {teacher_dir}...")
        teacher.save_pretrained(teacher_dir)
        tokenizer.save_pretrained(teacher_dir)


def main():
    parser = argparse.ArgumentParser(description="Forge Training")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file (required for new runs, ignored for resume)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name. Creates if not exists, resumes if exists. Use --force to overwrite.")
    parser.add_argument("--timestamp", action="store_true",
                        help="Append timestamp to --run_name (e.g., mnli_00 -> mnli_00_2026-01-31_12-00Z).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing run (deletes existing outputs and starts fresh).")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (for testing)")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs from config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG level logging for verbose output")
    args = parser.parse_args()

    # Setup logging early
    setup_logging(debug=args.debug)

    # Timestamp for new runs
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%MZ")

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
                # Delete W&B run if it exists (before deleting metadata)
                old_metadata = load_run_metadata(output_dir)
                if old_metadata and old_metadata.get("wandb_run_id"):
                    delete_wandb_run(old_metadata["wandb_run_id"])
                shutil.rmtree(output_dir)
                if args.config is None:
                    sys.exit(
                        "\nError: --config is required for new runs.\n\n"
                        "Usage:\n"
                        "  python forge.py --run_name NAME --config CONFIG [--force]\n"
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
                    f"  New run:    python forge.py --run_name {args.run_name} --config CONFIG\n"
                    f"  Resume:     python forge.py --run_name EXISTING_RUN\n"
                    f"  Overwrite:  python forge.py --run_name NAME --config CONFIG --force\n"
                )
            is_resume = False
            logger.info(f"Starting new run: {run_name}")
    else:
        # New run: auto-generate name from config
        if args.config is None:
            sys.exit(
                "\nError: No --run_name or --config provided.\n\n"
                "Usage:\n"
                "  New run:    python forge.py --config CONFIG\n"
                "  Named run:  python forge.py --run_name NAME --config CONFIG\n"
                "  Resume:     python forge.py --run_name EXISTING_RUN\n"
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
                f"  Resume:     python forge.py --run_name {run_name}\n"
                f"  Overwrite:  python forge.py --run_name {run_name} --config CONFIG --force\n"
            )
        config = load_config_snapshot(output_dir)
        if config is None:
            sys.exit(
                f"\nError: No stored config found at {output_dir / 'config.yaml'}\n"
                f"The run directory exists but appears corrupted.\n\n"
                f"Options:\n"
                f"  Delete and restart:  python forge.py --run_name {run_name} --config CONFIG --force\n"
                f"  Or manually delete:  rm -rf {output_dir}\n"
            )
        logger.info(f"Loaded stored config from: {output_dir / 'config.yaml'}")
    else:
        # New run: --config already validated above
        logger.info(f"Loading config from {args.config}")
        config = load_config(args.config)

    # Apply CLI overrides to config
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
        logger.info(f"CLI override: num_epochs={args.epochs}")

    logger.info(f"Output directory: {args.output_dir}")

    # Add file logging to output directory
    add_file_logging(output_dir, debug=args.debug)

    # Detect environment
    env = detect_environment()
    device = get_device()
    logger.info("=" * 60)
    logger.info(f"Environment: {env}")
    logger.info(f"Device: {device}")
    logger.info("=" * 60)

    # Print resolved config for reference
    logger.info("")
    logger.info("=" * 60)
    logger.info("CONFIGURATION")
    logger.info("=" * 60)
    config_yaml = yaml.dump(config, default_flow_style=False, sort_keys=False)
    for line in config_yaml.split("\n"):
        logger.info(line)
    logger.info("=" * 60)
    logger.info("")

    # Initialize state tracking for SLURM job chaining
    state = TrainingState(args.output_dir)
    state.set_targets(epochs=config["training"]["num_epochs"])

    if state.is_complete():
        logger.info("Training already complete!")
        return

    # Load dataset
    logger.info("Loading dataset...")
    train_dataset, eval_dataset, tokenizer_or_processor = load_dataset_for_task(config)

    # Truncate training set if max_train_samples is set
    max_train_samples = config.get("data", {}).get("max_train_samples")
    if max_train_samples is not None and max_train_samples < len(train_dataset):
        train_dataset = train_dataset.select(range(max_train_samples))
        logger.info(f"Truncated training set to {max_train_samples} samples")

    # For VLMs, load_dataset_for_task returns an AutoProcessor (which contains a tokenizer).
    # Extract the tokenizer for use throughout the pipeline.
    task = config.get("data", {}).get("task", "").lower()
    processor = None
    if hasattr(tokenizer_or_processor, 'tokenizer'):
        processor = tokenizer_or_processor
        tokenizer = processor.tokenizer
    else:
        tokenizer = tokenizer_or_processor

    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Eval size: {len(eval_dataset) if eval_dataset is not None else 0}")

    # Determine number of labels for classification tasks
    num_labels = get_num_labels(task)

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
        # New run — include timestamp in W&B ID for uniqueness
        # (W&B tombstones deleted run IDs, so deterministic IDs can't be reused)
        config_id = generate_config_id(args.config)
        # Skip adding timestamp if run_name already contains one (--timestamp or auto-named)
        if timestamp in run_name:
            wandb_run_id = truncate_for_wandb(run_name)
        else:
            wandb_run_id = truncate_for_wandb(f"{run_name}_{timestamp}")
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

    # Track job in incomplete registry for SLURM auto-resubmission
    add_incomplete_job(run_name, output_dir)

    # Setup W&B
    wandb_project = "Test" if args.debug else "Forge-Default"
    os.environ.setdefault("WANDB_PROJECT", wandb_project)
    os.environ["WANDB_RUN_ID"] = wandb_run_id
    if is_resume:
        os.environ["WANDB_RESUME"] = "allow"
    else:
        os.environ.pop("WANDB_RESUME", None)
    # Use args.config for new runs, metadata["config"] for resume
    config_path = args.config or metadata.get("config")
    if config_path:
        wandb_group = generate_wandb_group(config_path)
        wandb_tags = generate_wandb_tags(config_path)
        logger.info(f"W&B group: {wandb_group}, tags: {wandb_tags}")
        if wandb_group:
            os.environ["WANDB_RUN_GROUP"] = wandb_group
        if wandb_tags:
            os.environ["WANDB_TAGS"] = ",".join(wandb_tags)

    report_to = config["logging"]["report_to"]
    if os.environ.get("WANDB_MODE") == "disabled":
        report_to = "none"
        logger.info("W&B disabled - logging to console only")
    else:
        logger.info(f"W&B run ID: {wandb_run_id}")

    # Initialize W&B early so pre-eval metrics are logged
    # (HF Trainer will reuse the existing run)
    if report_to == "wandb":
        try:
            import wandb
            wandb.init(
                project=os.environ.get("WANDB_PROJECT"),
                id=wandb_run_id,
                name=run_name,
                group=os.environ.get("WANDB_RUN_GROUP"),
                tags=os.environ.get("WANDB_TAGS", "").split(",") if os.environ.get("WANDB_TAGS") else None,
                resume="allow" if is_resume else None,
            )
        except Exception as e:
            logger.debug(f"W&B early init failed: {e}")

    # Get compute metrics function and data collator (shared by eval and training)
    compute_metrics = get_compute_metrics(task, tokenizer=tokenizer)
    data_collator = None
    if is_ocr_task(task):
        logger.info("Using VLMDataCollator for vision-language model inputs")
        data_collator = VLMDataCollator(processor=processor)
    elif num_labels is None:  # Translation/seq2seq task
        logger.info("Using DataCollatorForSeq2Seq for dynamic padding and label masking")
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8 if device == "cuda" else None,  # Optimize for GPU
        )

    # =========================================================================
    # Pre-training evaluation (models scoped inside run_evaluation)
    # =========================================================================
    student_pre = teacher_pre = None
    default_log_pre_post = num_labels is not None
    log_pre_post = config.get("evaluation", {}).get("log_pre_post_metrics", default_log_pre_post)
    if log_pre_post and num_labels is None:
        logger.warning("log_pre_post_metrics enabled for translation - this may take hours on large eval sets")
    if log_pre_post:
        set_reproducibility_seed(args.seed)
        student_pre, teacher_pre = run_evaluation(
            config, device, tokenizer, num_labels,
            eval_dataset, compute_metrics, data_collator,
            task, phase="pre",
            remove_unused_columns=not is_ocr_task(task),
        )

    # =========================================================================
    # Training (models scoped inside _train)
    # =========================================================================
    set_reproducibility_seed(args.seed)
    _train(
        config, device, tokenizer, num_labels, task,
        train_dataset, eval_dataset, compute_metrics, data_collator,
        output_dir, run_name, args, state, resume_checkpoint,
        env, report_to, metadata,
    )

    # =========================================================================
    # Post-training evaluation (models scoped inside run_evaluation)
    # =========================================================================
    set_reproducibility_seed(args.seed)
    student_post, teacher_post = run_evaluation(
        config, device, tokenizer, num_labels,
        eval_dataset, compute_metrics, data_collator,
        task, phase="post",
        checkpoint_dir=output_dir,
        evaluate_teacher=log_pre_post,
        prev_metrics=(student_pre, teacher_pre) if log_pre_post else None,
        remove_unused_columns=not is_ocr_task(task),
    )

    # Remove from incomplete registry
    remove_incomplete_job()

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
