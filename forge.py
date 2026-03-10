"""Main training script for fine-tuning, knowledge distillation, and reinforcement learning."""

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

from src.utils.jobs import get_job_key, locked_incomplete_jobs

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

from src.data.handlers import get_handler
from src.data.formatters import format_for_sft, format_for_rl
from src.data.collators import VLMDataCollator
from src.evaluation import get_compute_metrics, run_evaluation
from src.tasks import get_num_labels, get_primary_metric, get_task_type, is_ocr_task
from src.models import load_models
from src.training.callbacks import (
    SLURMSignalCallback, CheckpointMarkerCallback, StepMetricsCallback,
    TeacherEvalCallback, TranslationMetricsCallback, PeakMetricsCallback,
)
from src.training.trainer import FinetuningTrainer, DistillationTrainer
from src.training.rewards import get_reward_funcs
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


def resolve_config_path(config_arg: str) -> Path:
    """Validate and resolve a config path.

    Expects a full tab-completable path (e.g. configs/classification/mnli/...).
    Falls back to adding configs/ prefix and/or .yaml suffix if the literal path
    doesn't exist.

    Args:
        config_arg: Config path as provided by user.

    Returns:
        Resolved Path object.
    """
    candidates = [
        Path(config_arg),
        Path(f"configs/{config_arg}"),
        Path(f"{config_arg}.yaml"),
        Path(f"configs/{config_arg}.yaml"),
    ]
    for path in candidates:
        if path.is_file():
            return path

    sys.exit(
        f"\nError: Config file not found: {config_arg}\n\n"
        f"Tried:\n"
        + "\n".join(f"  {p}" for p in candidates)
        + "\n"
    )


def extract_stem(config_path: Path) -> str:
    """Extract the stem from a resolved config path.

    Strips the configs/ prefix and .yaml suffix to get the internal identity.

    Args:
        config_path: Resolved config path (from resolve_config_path).

    Returns:
        Stem string, e.g. "classification/mnli/distilbert/bert-base/ablations/02_kd_frozen-sft"

    Examples:
        configs/classification/mnli/distilbert/00_sft.yaml
            -> classification/mnli/distilbert/00_sft
        configs/classification/mnli/distilbert/bert-base/ablations/02_kd_frozen-sft.yaml
            -> classification/mnli/distilbert/bert-base/ablations/02_kd_frozen-sft
    """
    parts = config_path.parts

    # Find 'configs' in path (handles absolute paths)
    try:
        configs_idx = parts.index("configs")
        # Everything after 'configs', rejoin, strip .yaml
        relative = Path(*parts[configs_idx + 1:])
    except ValueError:
        # No configs/ prefix, just strip suffix
        relative = config_path

    return str(relative.with_suffix(""))


def generate_wandb_run_id(prefix: str, timestamp: str, max_length: int = 64) -> str:
    """Generate a W&B run ID from prefix and timestamp.

    Format: {prefix[:x]}-{timestamp}. Truncates prefix to fit the 64-char limit
    while preserving the full timestamp. Uses - not / as separator (W&B run IDs
    must not contain / \\ # ? % :). Always includes timestamp for uniqueness
    since W&B tombstones deleted run IDs.

    Args:
        prefix: The stem (auto-named) or run_name (named). Slashes are replaced with -.
        timestamp: Timestamp string (e.g. "20260304-1200Z").
        max_length: W&B run ID character limit.

    Returns:
        W&B-safe run ID string.
    """
    safe_prefix = prefix.replace("/", "-")
    full_id = f"{safe_prefix}-{timestamp}"
    if len(full_id) <= max_length:
        return full_id

    # Truncate prefix, keep full timestamp
    max_prefix_len = max_length - len(timestamp) - 1
    return f"{safe_prefix[:max_prefix_len]}-{timestamp}"


def generate_wandb_group(stem: str) -> str:
    """Generate W&B group name from stem.

    Returns the parent directory of the stem, which groups runs by their
    config location (e.g. model pair or experiment directory).

    Args:
        stem: Config stem from extract_stem().

    Returns:
        Group string, or None if stem has no parent.

    Examples:
        classification/mnli/distilbert/00_sft -> classification/mnli/distilbert
        classification/mnli/distilbert/bert-base/ablations/02_kd -> classification/mnli/distilbert/bert-base/ablations
    """
    parent = str(Path(stem).parent)
    return parent if parent != "." else None


def generate_wandb_tags(stem: str) -> list:
    """Generate W&B tags from stem path segments (excluding the leaf config name).

    Args:
        stem: Config stem from extract_stem().

    Returns:
        List of tag strings.

    Examples:
        classification/mnli/distilbert/00_sft -> ["classification", "mnli", "distilbert"]
        classification/mnli/distilbert/bert-base/ablations/02_kd -> ["classification", "mnli", "distilbert", "bert-base", "ablations"]
    """
    parts = Path(stem).parts
    if len(parts) > 1:
        return list(parts[:-1])
    return []


def remove_incomplete_job(job_key: str = None):
    """Remove a job from incomplete jobs registry.

    Args:
        job_key: Job ID to remove. Defaults to current job.
    """
    if job_key is None:
        job_key = get_job_key()

    with locked_incomplete_jobs() as jobs:
        if job_key in jobs:
            del jobs[job_key]
            logger.info(f"Removed job from incomplete registry: {job_key}")


def add_incomplete_job(resume_path: str, output_dir: Path):
    """Add current job to incomplete jobs registry.

    Uses SLURM_JOB_ID if available, otherwise falls back to local_{pid}.
    Also removes previous job entry if PREV_JOB_ID is set (for SLURM resubmission).

    Args:
        resume_path: Output dir relative to outputs/ (for --resume on resubmit)
        output_dir: Path to the output directory
    """
    job_key = get_job_key()
    prev_job_id = os.environ.get("PREV_JOB_ID")

    with locked_incomplete_jobs() as jobs:
        if prev_job_id and prev_job_id in jobs:
            del jobs[prev_job_id]
        jobs[job_key] = {
            "type": "training",
            "resume_path": resume_path,
            "output_dir": str(output_dir),
        }
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


def _save_base_ref(output_dir: Path, model_name: str, revision: str = None):
    """Save base model reference to base/ directory.

    For HF Hub models, saves a model_ref.json with the model path and
    commit revision. HF Hub weights are immutable per revision, so this
    is a complete reproducible reference without duplicating weights locally.

    For custom bases (e.g., merged from --adapter), the caller saves full
    weights to base/ directly instead of calling this function.

    Args:
        output_dir: Run output directory
        model_name: HF Hub model ID
        revision: Git commit hash from HF Hub
    """
    base_dir = output_dir / "base"
    base_dir.mkdir(exist_ok=True)
    ref = {"model_name_or_path": model_name}
    if revision:
        ref["revision"] = revision
    with open(base_dir / "model_ref.json", 'w') as f:
        json.dump(ref, f, indent=2)
    logger.info(f"Base ref saved: {model_name} (rev: {revision or 'unknown'})")


def _resolve_adapter(config, output_dir, task, processor=None, tokenizer=None):
    """Resolve --adapter into a base model for training.

    If adapter_path is set in config, loads the HF base model, merges the
    adapter, and caches the result in base/. On resume, loads from base/
    directly (skips re-download and re-merge).

    Args:
        config: Full config dict (reads model.name and model.adapter_path)
        output_dir: Run output directory (base/ is created here)
        task: Task string (determines VLM vs causal LM loading)
        processor: AutoProcessor for VLMs
        tokenizer: Tokenizer for causal LMs

    Returns:
        Model object (if adapter was merged) or model name string (if no adapter).
    """
    model_name = config["model"]["name"]
    adapter_path = config["model"].get("adapter_path")

    if not adapter_path:
        return model_name

    from transformers import AutoModelForImageTextToText, AutoModelForCausalLM
    base_dir = output_dir / "base"

    # Resume: base/ already has merged weights from first run
    if (base_dir / "config.json").exists():
        logger.info(f"Loading base model from {base_dir} (cached from prior run)")
        if is_ocr_task(task):
            return AutoModelForImageTextToText.from_pretrained(
                str(base_dir), torch_dtype="auto", attn_implementation="sdpa",
            )
        else:
            return AutoModelForCausalLM.from_pretrained(
                str(base_dir), torch_dtype="auto", attn_implementation="sdpa",
            )

    # First run: download base, merge adapter, save to base/
    from peft import PeftModel

    logger.info(f"Loading base model: {model_name}")
    if is_ocr_task(task):
        base_model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype="auto", attn_implementation="sdpa",
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", attn_implementation="sdpa",
        )

    logger.info(f"Loading and merging adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    logger.info("Adapter merged into base model")

    logger.info(f"Saving base model to {base_dir}")
    model.save_pretrained(base_dir)
    if processor is not None:
        processor.save_pretrained(base_dir)
    elif tokenizer is not None:
        tokenizer.save_pretrained(base_dir)
    logger.info("Base model saved")

    return model


def _sync_wandb(config, task, metadata, output_dir):
    """Sync config and metadata to W&B after training.

    Args:
        config: Full config dict
        task: Task string (for primary metric lookup)
        metadata: Run metadata dict (updated in-place if W&B ID changes)
        output_dir: Run output directory
    """
    try:
        import wandb
        if wandb.run:
            wandb.config.update(config, allow_val_change=True)
            primary = get_primary_metric(task)
            if primary:
                wandb.config.update({"primary_metric": primary}, allow_val_change=True)
            if wandb.run.id != metadata.get("wandb_run_id"):
                old_id = metadata.get("wandb_run_id")
                metadata["wandb_run_id"] = wandb.run.id
                metadata.setdefault("wandb_run_id_history", []).append(old_id)
                save_run_metadata(output_dir, metadata)
                logger.info(f"W&B ID changed: {old_id} -> {wandb.run.id}")
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"W&B config update failed: {e}")


def _save_final_model(trainer, config, output_dir, processor=None, tokenizer=None):
    """Save final model with unified structure (merged model + adapter + base ref).

    For LoRA: saves adapter, base reference, then merges and saves merged model.
    For Full FT: saves full model and base reference.
    Always saves processor/tokenizer.

    Args:
        trainer: HF Trainer instance (model accessible via trainer.model)
        config: Full config dict
        output_dir: Run output directory
        processor: AutoProcessor for VLMs (optional)
        tokenizer: Tokenizer for causal LMs (optional)
    """
    if hasattr(trainer.model, "merge_and_unload"):
        # LoRA: save adapter first (merge is destructive)
        logger.info("Saving LoRA adapter to output dir...")
        trainer.model.save_pretrained(output_dir)

        # Save base ref if not already saved with full weights (--adapter case)
        if not (output_dir / "base" / "config.json").exists():
            base_config = trainer.model.get_base_model().config
            _save_base_ref(
                output_dir,
                model_name=base_config._name_or_path,
                revision=getattr(base_config, "_commit_hash", None),
            )

        logger.info("Merging LoRA and saving merged model...")
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(output_dir)
    else:
        # Full FT: save complete model
        logger.info("Saving final model to output dir...")
        trainer.save_model(output_dir)
        # Save base ref if not already saved with full weights (--adapter case)
        if not (output_dir / "base" / "config.json").exists():
            _save_base_ref(
                output_dir,
                model_name=config["model"]["name"],
                revision=getattr(trainer.model.config, "_commit_hash", None),
            )

    # Save processor/tokenizer
    if processor is not None:
        processor.save_pretrained(output_dir)
    elif tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    logger.info("Final model saved")


def _train(
    config, device, tokenizer, num_labels, task,
    train_dataset, eval_dataset, compute_metrics, data_collator,
    output_dir, run_name, args, state, resume_checkpoint,
    env, report_to, metadata, processor=None,
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
        processor: AutoProcessor for VLMs (optional)
    """
    # Load teacher and student models
    logger.info("Loading models...")
    teacher, student = load_models(config, device, tokenizer, num_labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=int(config["training"]["num_epochs"]),
        per_device_train_batch_size=int(config["training"]["batch_size"]),
        per_device_eval_batch_size=int(config["training"].get("eval_batch_size", config["training"]["batch_size"])),
        gradient_accumulation_steps=int(config["training"].get("gradient_accumulation_steps", 1)),
        learning_rate=float(config["training"]["learning_rate"]),
        warmup_steps=int(config["training"]["warmup_steps"]),
        weight_decay=float(config["training"]["weight_decay"]),
        max_grad_norm=float(config["training"]["max_grad_norm"]),
        logging_steps=int(config["logging"]["logging_steps"]),
        save_steps=int(config["checkpointing"]["save_steps"]),
        save_total_limit=int(config["checkpointing"]["save_total_limit"]),
        eval_strategy=config.get("evaluation", {}).get("eval_strategy", "steps"),
        eval_steps=int(config.get("evaluation", {}).get("eval_steps", 500)),
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
        # CE-only baseline: use FinetuningTrainer
        logger.info("No teacher - using FinetuningTrainer (CE-only mode)")
        trainer = FinetuningTrainer(
            model=student,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
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
            freeze_teacher=config["model"].get("teacher", {}).get("freeze", True),
            loss_strategy=loss_strategy,
            teacher_label_map=config["distillation"].get("teacher_label_map"),
            layer_mapping=layer_mapping,
            projector=projector,
            model=student,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )

    # Log metrics to output.log in grep-friendly format
    trainer.add_callback(StepMetricsCallback())

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

    # Add checkpoint marker (LAST - runs after all other on_save hooks)
    trainer.add_callback(CheckpointMarkerCallback())

    # =========================================================================
    # Training
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    _sync_wandb(config, task, metadata, output_dir)

    # Update state
    state.save(
        epoch=trainer.state.epoch,
        step=trainer.state.global_step
    )

    # Save final model (skip if training was interrupted - checkpoint already saved)
    if state.is_complete():
        _save_final_model(trainer, config, output_dir, processor=processor, tokenizer=tokenizer)

        # Save co-trained teacher (weights changed during training)
        if teacher is not None and not config["model"].get("teacher", {}).get("freeze", True):
            teacher_dir = output_dir / "teacher"
            logger.info(f"Saving co-trained teacher to {teacher_dir}...")
            teacher.save_pretrained(teacher_dir)
            tokenizer.save_pretrained(teacher_dir)
    else:
        logger.info("Training interrupted - skipping final model save (checkpoint already saved)")


def _build_grpo_config(config, output_dir, run_name, args, device, report_to):
    """Build a GRPOConfig from the Forge YAML config.

    Args:
        config: Full config dict with 'rl' and 'training' sections.
        output_dir: Path to output directory.
        run_name: W&B display name.
        args: Parsed CLI arguments.
        device: Device string.
        report_to: Logging target.

    Returns:
        GRPOConfig instance.
    """
    from trl import GRPOConfig

    rl = config.get("rl", {})
    training = config["training"]

    grpo_args = GRPOConfig(
        output_dir=str(output_dir),
        # GRPO-specific
        num_generations=int(rl.get("num_generations", 8)),
        generation_batch_size=int(rl["generation_batch_size"]) if rl.get("generation_batch_size") is not None else None,
        max_completion_length=int(rl.get("max_completion_length", 256)),
        beta=float(rl.get("beta", 0.0)),
        temperature=float(rl.get("temperature", 1.0)),
        loss_type=rl.get("loss_type", "grpo"),
        num_iterations=int(rl.get("num_iterations", 1)),
        scale_rewards=rl.get("scale_rewards", "group"),
        # Training params from config
        num_train_epochs=int(training["num_epochs"]),
        per_device_train_batch_size=int(training.get("batch_size", 4)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        learning_rate=float(rl.get("learning_rate", training.get("learning_rate", 1e-6))),
        warmup_steps=int(training.get("warmup_steps", 200)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        max_grad_norm=float(training.get("max_grad_norm", 2.0)),
        # Logging & eval
        logging_steps=int(config["logging"]["logging_steps"]),
        save_steps=int(config["checkpointing"]["save_steps"]),
        save_total_limit=int(config["checkpointing"]["save_total_limit"]),
        eval_strategy=config.get("evaluation", {}).get("eval_strategy", "steps"),
        eval_steps=int(config.get("evaluation", {}).get("eval_steps", 500)),
        report_to=report_to,
        run_name=run_name,
        # Hardware
        bf16=(device == "cuda"),
        dataloader_num_workers=training.get("dataloader_num_workers", 4) if device == "cuda" else 0,
        dataloader_pin_memory=(device == "cuda"),
        # Overrides
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        seed=args.seed,
        data_seed=args.seed,
        # Log generated completions to W&B for inspection
        log_completions=rl.get("log_completions", True),
    )
    return grpo_args


def _train_rl(
    config, device, tokenizer, task,
    train_dataset, eval_dataset,
    output_dir, run_name, args, state, resume_checkpoint,
    env, report_to, metadata, processor=None,
):
    """Run RL (GRPO) training.

    Args:
        config: Full configuration dictionary.
        device: Device string.
        tokenizer: Tokenizer.
        task: Task name.
        train_dataset: Training dataset (prompt-based format).
        eval_dataset: Evaluation dataset.
        output_dir: Output directory (Path).
        run_name: Run name for W&B.
        args: Parsed CLI arguments.
        state: TrainingState for checkpoint tracking.
        resume_checkpoint: Path to resume checkpoint (or None).
        env: Environment string.
        report_to: Logging target.
        metadata: Run metadata dict.
        processor: AutoProcessor for VLM models (or None).
    """
    from trl import GRPOTrainer
    from peft import LoraConfig

    # Build reward functions from config
    reward_funcs = get_reward_funcs(config)
    logger.info(f"Reward functions: {[f.__name__ for f in reward_funcs]}")

    # Build GRPOConfig
    grpo_config = _build_grpo_config(config, output_dir, run_name, args, device, report_to)
    logger.info(f"GRPO config: num_generations={grpo_config.num_generations}, "
                f"max_completion_length={grpo_config.max_completion_length}, "
                f"beta={grpo_config.beta}, loss_type={grpo_config.loss_type}")

    # Model loading (resolves --adapter if set, with base/ caching for resume)
    model = _resolve_adapter(config, output_dir, task, processor, tokenizer)

    # LoRA config (if enabled)
    peft_config = None
    if config["training"].get("use_lora", False):
        lora_cfg = config["training"].get("lora", {})
        peft_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 16),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", None),
            task_type="CAUSAL_LM",
        )
        logger.info(f"LoRA enabled: r={peft_config.r}, alpha={peft_config.lora_alpha}")

    # Processing class (tokenizer or processor for VLMs)
    processing_class = processor if processor is not None else tokenizer

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        peft_config=peft_config,
    )

    # Log metrics to output.log in grep-friendly format
    trainer.add_callback(StepMetricsCallback())

    # Add SLURM signal handler
    if env == "slurm":
        logger.info("Adding SLURM signal handler...")
        trainer.add_callback(SLURMSignalCallback(trainer))

    # Add checkpoint marker (LAST - runs after all other on_save hooks)
    trainer.add_callback(CheckpointMarkerCallback())

    # Training
    logger.info("=" * 60)
    logger.info("Starting GRPO training...")
    logger.info("=" * 60)

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    _sync_wandb(config, task, metadata, output_dir)

    # Update state
    state.save(
        epoch=trainer.state.epoch,
        step=trainer.state.global_step,
    )

    # Save final model (skip if training was interrupted)
    if state.is_complete():
        _save_final_model(trainer, config, output_dir, processor=processor, tokenizer=tokenizer)
    else:
        logger.info("Training interrupted - skipping final model save (checkpoint already saved)")


def main():
    parser = argparse.ArgumentParser(description="Forge Training")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file (required for new runs)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to output dir to resume (e.g. outputs/classification/mnli/.../20260304-1200Z)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name for flat output dir (e.g. seed42 -> outputs/seed42/). Create-only.")
    parser.add_argument("--timestamp", action="store_true",
                        help="Append timestamp to --run_name (e.g. seed42 -> seed42_20260304-1200Z).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing named run (deletes outputs and W&B run, starts fresh).")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (for testing)")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs from config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to adapter to merge before training (e.g. outputs/my_sft_run/)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG level logging for verbose output")
    args = parser.parse_args()

    # Setup logging early
    setup_logging(debug=args.debug)

    # Timestamp for new runs
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%MZ")

    # Determine output directory and mode
    if args.resume:
        # =====================================================================
        # RESUME MODE
        # =====================================================================
        invalid = []
        if args.config:
            invalid.append("--config")
        if args.run_name:
            invalid.append("--run_name")
        if args.force:
            invalid.append("--force")
        if args.timestamp:
            invalid.append("--timestamp")
        if invalid:
            sys.exit(
                f"\nError: {', '.join(invalid)} cannot be used with --resume.\n\n"
                f"Usage:\n"
                f"  python forge.py --resume {args.resume}\n"
            )

        # Strip outputs/ prefix if present, then ensure outputs/ base
        resume_path = args.resume
        if resume_path.startswith("outputs/"):
            resume_path = resume_path[len("outputs/"):]
        resume_path = resume_path.rstrip("/")

        output_dir = Path(f"./outputs/{resume_path}")
        if not output_dir.exists():
            sys.exit(
                f"\nError: Output directory not found: {output_dir}\n\n"
                f"Usage:\n"
                f"  python forge.py --resume outputs/PATH_TO_RUN\n"
            )
        metadata_path = output_dir / "run_metadata.json"
        if not metadata_path.exists():
            sys.exit(
                f"\nError: No run_metadata.json found in {output_dir}.\n"
                f"Run may have crashed before setup completed.\n\n"
                f"Start fresh with:\n"
                f"  python forge.py --config CONFIG\n"
            )
        is_resume = True
        stem = None  # loaded from metadata later
        config_path = None
        logger.info(f"Resuming run from: {output_dir}")

    elif args.config:
        # =====================================================================
        # CREATE MODE
        # =====================================================================
        config_path = resolve_config_path(args.config)
        stem = extract_stem(config_path)

        if args.timestamp and not args.run_name:
            logger.warning("--timestamp without --run_name is redundant (auto-named runs always include timestamp)")
        if args.force and not args.run_name:
            logger.warning("--force without --run_name is redundant (auto-named runs are always unique)")

        if args.run_name:
            # Named run (flat output dir)
            if args.timestamp:
                display_name = f"{args.run_name}_{timestamp}"
            else:
                display_name = args.run_name
            output_dir = Path(f"./outputs/{display_name}")

            if output_dir.exists():
                if args.force:
                    logger.info(f"--force specified. Deleting existing run: {output_dir}")
                    old_metadata = load_run_metadata(output_dir)
                    if old_metadata and old_metadata.get("wandb_run_id"):
                        delete_wandb_run(old_metadata["wandb_run_id"])
                    shutil.rmtree(output_dir)
                else:
                    sys.exit(
                        f"\nError: Output directory already exists: {output_dir}\n\n"
                        f"Options:\n"
                        f"  Resume:     python forge.py --resume outputs/{display_name}\n"
                        f"  Overwrite:  python forge.py --run_name {args.run_name} --config {args.config} --force\n"
                    )
        else:
            # Auto-named run (hierarchical output dir)
            display_name = f"{stem}/{timestamp}"
            output_dir = Path(f"./outputs/{stem}/{timestamp}")

        is_resume = False
        logger.info(f"Starting new run: {display_name}")

    elif args.run_name:
        # --run_name without --config
        sys.exit(
            f"\nError: --config is required for new runs.\n\n"
            f"Usage:\n"
            f"  python forge.py --run_name {args.run_name} --config CONFIG\n"
        )

    else:
        # Nothing provided
        sys.exit(
            "\nError: No --config or --resume provided.\n\n"
            "Usage:\n"
            "  New run:    python forge.py --config configs/PATH.yaml\n"
            "  Named run:  python forge.py --config configs/PATH.yaml --run_name NAME\n"
            "  Resume:     python forge.py --resume outputs/PATH_TO_RUN\n"
        )

    # Convert to string for compatibility
    args.output_dir = str(output_dir)

    # Set reproducibility seed early
    set_reproducibility_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")

    # Load configuration
    if is_resume:
        # Resume: load stored config
        config = load_config_snapshot(output_dir)
        if config is None:
            sys.exit(
                f"\nError: No stored config found at {output_dir / 'config.yaml'}\n"
                f"The run directory exists but appears corrupted.\n\n"
                f"Options:\n"
                f"  Start fresh:  python forge.py --config CONFIG\n"
                f"  Delete dir:   rm -rf {output_dir}\n"
            )
        # Load stem from metadata (with backward compat)
        existing_metadata = load_run_metadata(output_dir)
        if existing_metadata:
            stem = existing_metadata.get("stem")
            if stem is None:
                # Backward compat: regenerate stem from stored config path
                old_config = existing_metadata.get("config", "")
                if old_config:
                    try:
                        stem = extract_stem(Path(old_config))
                    except Exception:
                        stem = output_dir.name
                else:
                    stem = output_dir.name
                # Migrate: add stem and display_name fields
                existing_metadata["stem"] = stem
                if "display_name" not in existing_metadata:
                    existing_metadata["display_name"] = existing_metadata.get("run_name", output_dir.name)
                save_run_metadata(output_dir, existing_metadata)
                logger.warning(f"Migrated metadata: added stem={stem}")
        logger.info(f"Loaded stored config from: {output_dir / 'config.yaml'}")
    else:
        # New run: --config already validated/resolved above
        logger.info(f"Loading config from {config_path}")
        config = load_config(str(config_path))

    # Apply CLI overrides to config
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
        logger.info(f"CLI override: num_epochs={args.epochs}")
    if args.adapter is not None:
        config.setdefault("model", {})["adapter_path"] = args.adapter
        logger.info(f"CLI override: adapter_path={args.adapter}")

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
    if args.max_steps > 0:
        state.set_targets(steps=args.max_steps)
    else:
        state.set_targets(epochs=config["training"]["num_epochs"])

    if state.is_pipeline_complete():
        logger.info("Training pipeline already complete!")
        return

    # Determine task and training mode
    task = config.get("data", {}).get("task", "").lower()
    has_rl = "rl" in config
    rl_algorithm = config.get("rl", {}).get("algorithm", "grpo") if has_rl else None

    if has_rl:
        logger.info(f"RL mode: algorithm={rl_algorithm}")

    # Find checkpoint to resume from
    resume_checkpoint = state.get_resume_checkpoint(args.output_dir)
    if resume_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")

    # Handle run metadata
    if is_resume:
        # existing_metadata already loaded in config loading section
        metadata = existing_metadata
        wandb_run_id = metadata.get("wandb_run_id")
        display_name = metadata.get("display_name") or metadata.get("run_name", output_dir.name)
        stem = metadata.get("stem", stem)  # stem may have been set by backward compat above
        logger.info(f"Resuming run: wandb_run_id={wandb_run_id}")
    else:
        # New run: generate W&B run ID with timestamp for uniqueness
        prefix = args.run_name if args.run_name else stem
        wandb_run_id = generate_wandb_run_id(prefix, timestamp)
        metadata = {
            "stem": stem,
            "display_name": display_name,
            "wandb_run_id": wandb_run_id,
            "config": str(config_path),
            "created": timestamp,
            "seed": args.seed,
        }
        # Lineage: track where this stage's base model came from
        if args.adapter:
            metadata["base_model"] = config["model"]["name"]
            metadata["adapter"] = args.adapter
        output_dir.mkdir(parents=True, exist_ok=True)
        save_run_metadata(output_dir, metadata)
        save_config_snapshot(output_dir, config)
        logger.info(f"Created run metadata: {output_dir / 'run_metadata.json'}")

    # Track job in incomplete registry for SLURM auto-resubmission
    resume_path = str(output_dir.relative_to("./outputs")) if str(output_dir).startswith("./outputs") else str(output_dir).replace("outputs/", "", 1)
    add_incomplete_job(resume_path, output_dir)

    # Setup W&B
    if args.debug:
        os.environ["WANDB_PROJECT"] = "Forge-Test"
    else:
        os.environ.setdefault("WANDB_PROJECT", "Forge-Default")
    os.environ["WANDB_RUN_ID"] = wandb_run_id
    if is_resume:
        os.environ["WANDB_RESUME"] = "allow"
    else:
        os.environ.pop("WANDB_RESUME", None)

    if stem:
        wandb_group = generate_wandb_group(stem)
        wandb_tags = generate_wandb_tags(stem)
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
                name=display_name,
                group=os.environ.get("WANDB_RUN_GROUP"),
                tags=os.environ.get("WANDB_TAGS", "").split(",") if os.environ.get("WANDB_TAGS") else None,
                resume="allow" if is_resume else None,
            )
        except Exception as e:
            logger.debug(f"W&B early init failed: {e}")

    if has_rl and rl_algorithm == "grpo":
        # =================================================================
        # RL TRAINING PATH (GRPO)
        # =================================================================
        logger.info("Loading RL dataset (prompt-based)...")
        dataset_name = config["data"].get("dataset", task)
        handler = get_handler(dataset_name)
        handler_kwargs = {}
        if "data_dir" in config.get("data", {}):
            handler_kwargs["data_dir"] = config["data"]["data_dir"]
        else:
            handler_kwargs["data_dir"] = f"data/{dataset_name}"
        if "max_train_samples" in config.get("data", {}):
            handler_kwargs["max_train_samples"] = config["data"]["max_train_samples"]
        if "include_reasoning" in config.get("data", {}):
            handler_kwargs["include_reasoning"] = config["data"]["include_reasoning"]
        # Load extra kwargs for specific handlers
        if "source_lang" in config.get("data", {}):
            handler_kwargs["source_lang"] = config["data"]["source_lang"]
        if "target_lang" in config.get("data", {}):
            handler_kwargs["target_lang"] = config["data"]["target_lang"]
        if "max_samples" in config.get("data", {}):
            handler_kwargs["max_samples"] = config["data"]["max_samples"]
        train_dataset, eval_dataset = handler.load(**handler_kwargs)
        task_type = get_task_type(task)
        train_dataset = format_for_rl(train_dataset, task_type)
        if eval_dataset is not None:
            eval_dataset = format_for_rl(eval_dataset, task_type)
        logger.info(f"Train size: {len(train_dataset)}")
        logger.info(f"Eval size: {len(eval_dataset) if eval_dataset is not None else 0}")

        # For VLM tasks, load the processor for tokenization
        processor = None
        tokenizer = None
        if is_ocr_task(task):
            from transformers import AutoProcessor
            processor_name = config["model"].get("processor", config["model"]["name"])
            processor = AutoProcessor.from_pretrained(processor_name)
            max_visual_tokens = config.get("data", {}).get("max_visual_tokens")
            if max_visual_tokens:
                ip = processor.image_processor
                pixels_per_token = (ip.patch_size * ip.merge_size) ** 2
                ip.max_pixels = max_visual_tokens * pixels_per_token
            tokenizer = processor.tokenizer
        else:
            from transformers import AutoTokenizer
            model_name = config["model"]["name"]
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        set_reproducibility_seed(args.seed)
        _train_rl(
            config, device, tokenizer, task,
            train_dataset, eval_dataset,
            output_dir, display_name, args, state, resume_checkpoint,
            env, report_to, metadata, processor=processor,
        )
    else:
        # =================================================================
        # SUPERVISED TRAINING PATH (SFT / KD)
        # =================================================================
        logger.info("Loading dataset...")

        # Load tokenizer/processor first (needed by SFT formatters)
        task_type = get_task_type(task)
        processor = None
        if is_ocr_task(task):
            from transformers import AutoProcessor
            processor_name = config["model"].get("processor", config["model"]["name"])
            logger.info(f"Loading processor: {processor_name}...")
            processor = AutoProcessor.from_pretrained(processor_name)
            max_visual_tokens = config.get("data", {}).get("max_visual_tokens")
            if max_visual_tokens:
                ip = processor.image_processor
                pixels_per_token = (ip.patch_size * ip.merge_size) ** 2
                ip.max_pixels = max_visual_tokens * pixels_per_token
            tokenizer = processor.tokenizer
        else:
            from transformers import AutoTokenizer
            teacher_cfg = config["model"].get("teacher", {})
            teacher_model = teacher_cfg.get("name") if isinstance(teacher_cfg, dict) else teacher_cfg
            if teacher_model is None:
                teacher_model = config["model"]["name"]
            tokenizer = AutoTokenizer.from_pretrained(teacher_model)
            from src.tasks import is_translation_task
            if "llama" in teacher_model.lower() or is_translation_task(task):
                tokenizer.padding_side = "left"
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

        # Load raw data via handler
        dataset_name = config["data"].get("dataset", task)
        handler = get_handler(dataset_name)
        handler_kwargs = {}
        if "data_dir" in config.get("data", {}):
            handler_kwargs["data_dir"] = config["data"]["data_dir"]
        elif dataset_name not in ("mnli", "qqp", "sst2", "cola", "europarl"):
            handler_kwargs["data_dir"] = f"data/{dataset_name}"
        if "max_train_samples" in config.get("data", {}):
            handler_kwargs["max_train_samples"] = config["data"]["max_train_samples"]
        if "include_reasoning" in config.get("data", {}):
            handler_kwargs["include_reasoning"] = config["data"]["include_reasoning"]
        if "source_lang" in config.get("data", {}):
            handler_kwargs["source_lang"] = config["data"]["source_lang"]
        if "target_lang" in config.get("data", {}):
            handler_kwargs["target_lang"] = config["data"]["target_lang"]
        if "max_samples" in config.get("data", {}):
            handler_kwargs["max_samples"] = config["data"]["max_samples"]
        train_dataset, eval_dataset = handler.load(**handler_kwargs)

        # Format for SFT
        max_length = config.get("data", {}).get("max_length", 1024)
        tokenizer_or_processor = processor if processor else tokenizer
        train_dataset = format_for_sft(train_dataset, task_type, tokenizer_or_processor, max_length)
        if eval_dataset is not None:
            eval_dataset = format_for_sft(eval_dataset, task_type, tokenizer_or_processor, max_length)

        # Resolve --adapter: merge into base/ on first run, reuse on resume
        adapter_path = config["model"].get("adapter_path")
        if adapter_path:
            base_dir = output_dir / "base"
            if not (base_dir / "config.json").exists():
                _resolve_adapter(config, output_dir, task, processor, tokenizer)
            config["model"]["name"] = str(base_dir)

        logger.info(f"Train size: {len(train_dataset)}")
        logger.info(f"Eval size: {len(eval_dataset) if eval_dataset is not None else 0}")

        num_labels = get_num_labels(task)

        # Get compute metrics function and data collator
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
                pad_to_multiple_of=8 if device == "cuda" else None,
            )

        # Pre-training evaluation
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

        # Training
        set_reproducibility_seed(args.seed)
        _train(
            config, device, tokenizer, num_labels, task,
            train_dataset, eval_dataset, compute_metrics, data_collator,
            output_dir, display_name, args, state, resume_checkpoint,
            env, report_to, metadata, processor=processor,
        )

        # Post-training evaluation (skip if training was interrupted to save time for W&B sync)
        if state.is_complete():
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
        else:
            logger.info("Training interrupted - skipping post-training evaluation")

    # Mark full pipeline complete and remove from incomplete registry
    if state.is_complete():
        state.mark_pipeline_complete()
        remove_incomplete_job()
        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        if state.state.get("target_steps"):
            logger.info(f"Training interrupted (step {state.state['completed_steps']}/{state.state['target_steps']})")
        else:
            logger.info(f"Training interrupted (epoch {state.state['completed_epochs']}/{state.state['target_epochs']})")
        logger.info("Job remains in incomplete registry for resubmission")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
