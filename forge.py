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
from datetime import UTC, datetime
from pathlib import Path

# Add project root to path for running without package install
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import transformers
import yaml
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    ProgressCallback,
    TrainingArguments,
    set_seed,
)

from src.data.collators import VLMDataCollator
from src.data.formatters import format_for_rl, format_for_sft
from src.data.handlers import get_handler
from src.evaluation import get_compute_metrics, run_evaluation
from src.models import load_model, needs_attention_outputs
from src.tasks import get_num_labels, get_primary_metric, get_task_type, is_ocr_task
from src.training.callbacks import (
    CheckpointMarkerCallback,
    PeakMetricsCallback,
    SLURMSignalCallback,
    StepMetricsCallback,
    StudentTeacherGradNormCallback,
    TeacherEvalCallback,
    TranslationMetricsCallback,
)
from src.training.losses import (
    build_distill_topology_strategy,
    build_layer_mapping_and_projector,
    build_teacher_response_only_strategy,
)
from src.training.rewards import get_reward_funcs
from src.training.trainer import DistillationTrainer, FinetuningTrainer
from src.utils.checkpoint import LATEST_CHECKPOINT_FILE, TrainingState
from src.utils.config import load_config, resolve_distillation_coefficients

# Auto-loads .env file if it exists (override=False means Colab/HPC env vars take precedence)
from src.utils.environment import detect_environment, get_device, get_num_workers
from src.utils.jobs import get_job_key, guard_duplicate_job, locked_incomplete_jobs

# Module-level logger
logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False):
    """Configure logging level for the training run.

    Args:
        debug: If True, use DEBUG level; otherwise INFO level
    """
    level = logging.DEBUG if debug else logging.INFO
    format_str = (
        "%(asctime)sZ - %(name)s - %(levelname)s - %(message)s"
        if debug
        else "%(asctime)sZ - %(levelname)s - %(message)s"
    )

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.Formatter.converter = time.gmtime

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


def set_reproducibility_seed(seed: int = 42):
    """Set seeds for reproducibility across all libraries.

    Args:
        seed: Seed value applied to Python's ``random``, NumPy, and PyTorch (CPU + CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # HuggingFace transformers seed
    # For fully deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Keys forwarded from config["data"] to handler.load(**kwargs)
_HANDLER_DATA_KEYS = (
    "max_train_samples",
    "include_reasoning",
    "sampling",
    "source_lang",
    "target_lang",
    "max_samples",
)


def _build_handler_kwargs(config: dict, dataset_name: str) -> dict:
    """Build kwargs dict for handler.load() from config["data"].

    Args:
        config: Full training config.
        dataset_name: Dataset identifier (e.g. "olmocr-mix", "formatting").

    Returns:
        Dict of keyword arguments to pass to handler.load().
    """
    data_cfg = config.get("data", {})
    kwargs = {"data_dir": data_cfg.get("data_dir", f"data/{dataset_name}")}

    for key in _HANDLER_DATA_KEYS:
        if key in data_cfg:
            kwargs[key] = data_cfg[key]

    return kwargs


def create_loss_strategy_from_config(
    distill_config: dict,
    model,
    teacher_model,
    device: str,
    freeze_teacher: bool = True,
):
    """
    Create loss strategy from distillation config.

    Args:
        distill_config: The ``distillation`` section of the training config.
        model: The student model (used to size layer mapping/projection).
        teacher_model: The teacher model (used to size layer mapping/projection).
        device: Device to move a newly created projector onto.
        freeze_teacher: Whether the teacher is frozen; when False, also builds a
            teacher-side loss strategy for co-distillation.

    Supports three modes:
    1. Single strategy (loss_type)
    2. Combined strategies (loss_strategies) - for FAR matrix
    3. Staged strategies (stages) - for F→R, F→A→R progressions

    Config format for single strategy:
        distillation:
          loss_type: kl_divergence
          alpha_h: 0.5
          alpha_s: 0.5

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
        Tuple of (loss_strategy, teacher_loss_strategy, layer_mapping, projector).
        ``teacher_loss_strategy`` is None when ``freeze_teacher`` is True.
    """
    coeffs = resolve_distillation_coefficients(distill_config, freeze_teacher=freeze_teacher)
    alpha_h = coeffs["alpha_h"]
    alpha_s = coeffs["alpha_s"]
    temperature = distill_config.get("temperature", 2.0)

    loss_strategy = build_distill_topology_strategy(
        distill_config,
        alpha_h,
        alpha_s,
        temperature=temperature,
    )
    teacher_loss_strategy = None
    if not freeze_teacher:
        teacher_loss_strategy = build_teacher_response_only_strategy(
            distill_config,
            coeffs["beta_h"],
            coeffs["beta_s"],
            temperature=temperature,
        )

    layer_mapping, projector = build_layer_mapping_and_projector(
        distill_config,
        loss_strategy,
        model,
        teacher_model,
        device,
    )
    if layer_mapping is not None:
        logger.debug(f"  Layer mapping: {layer_mapping}")
    if projector is not None:
        logger.debug("  Projection enabled for feature distillation")

    return loss_strategy, teacher_loss_strategy, layer_mapping, projector


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
        f"Tried:\n" + "\n".join(f"  {p}" for p in candidates) + "\n"
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
        relative = Path(*parts[configs_idx + 1 :])
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


def remove_incomplete_job(job_key: str | None = None):
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
        guard_duplicate_job(jobs, "resume_path", resume_path, job_key)
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


def delete_wandb_run(run_id: str, project: str | None = None) -> bool:
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
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def save_config_snapshot(output_dir: Path, config: dict):
    """Save full resolved config to output directory for reproducibility.

    Returns:
        Path to the written ``config.yaml``.
    """
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return config_path


def load_config_snapshot(output_dir: Path) -> dict | None:
    """Load stored config from output directory.

    Returns:
        The parsed config, or ``None`` if ``output_dir`` has no ``config.yaml`` yet.
    """
    config_path = output_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return None


def should_load_best_model(config: dict) -> bool:
    """Whether to enable load_best_model_at_end, based on checkpointing.final_model_save.

    Args:
        config: Full training config dict.

    Returns:
        True when checkpointing.final_model_save is "best" (the default) or "both".
        False when it is "last" - the run keeps whatever state training ended in.

    Raises:
        ValueError: If final_model_save is "best"/"both" but eval isn't actively
            running (steps/epoch), since there are no eval metrics to pick a best
            checkpoint from. Also raised if final_model_save is set to anything
            other than "best", "last", or "both".
    """
    final_model_save = config.get("checkpointing", {}).get("final_model_save", "best")
    if final_model_save not in ("best", "last", "both"):
        raise ValueError(
            f"checkpointing.final_model_save must be 'best', 'last', or 'both', got {final_model_save!r}"
        )
    if final_model_save == "last":
        return False

    eval_strategy = config.get("evaluation", {}).get("eval_strategy", "steps")
    if eval_strategy not in ("steps", "epoch"):
        raise ValueError(
            f"checkpointing.final_model_save is {final_model_save!r} but evaluation.eval_strategy is "
            f"{eval_strategy!r} - no eval metrics available to select a best checkpoint. Set "
            "checkpointing.final_model_save: last, or enable evaluation.eval_strategy: steps/epoch."
        )
    return True


def should_save_last_model(config: dict) -> bool:
    """Whether to additionally export the last-step model, based on final_model_save.

    Args:
        config: Full training config dict.

    Returns:
        True when checkpointing.final_model_save is "both".
    """
    return config.get("checkpointing", {}).get("final_model_save", "best") == "both"


def no_weights_mode(config: dict) -> bool:
    """Return True to skip saving final model weights after training.

    Intermediate checkpoints are still written during training (SLURM
    resumption works). On successful completion, checkpoints are removed
    by _remove_hf_checkpoint_dirs_after_success() and final weights are
    not saved. Results land in W&B only.

    Args:
        config: Full training config dict.

    Returns:
        True when checkpointing.no_weights is set to true.
    """
    return bool(config.get("checkpointing", {}).get("no_weights", False))


def warmup_kwargs_for_training_args(
    training: dict,
    default_warmup_steps: int = 0,
) -> dict:
    """Build warmup fields for HuggingFace TrainingArguments.

    Args:
        training: The ``training`` section of the config.
        default_warmup_steps: Fallback when neither ``warmup_ratio`` nor ``warmup_steps``
            is set in ``training`` (0 for standard fine-tuning, 200 for GRPO).

    If ``training.warmup_ratio`` is set (non-null), it is used as a fraction of
    total optimizer steps (stable when batch size or step count changes). Otherwise
    ``training.warmup_steps`` is used when non-null; if both ratio and steps are
    absent or null, ``default_warmup_steps`` is used (0 for standard fine-tuning;
    GRPO passes 200 to preserve prior defaults).

    When both ratio and steps appear in a merged config, ``warmup_ratio`` wins.

    Returns:
        Dict with either ``warmup_ratio`` or ``warmup_steps`` for unpacking into
        ``TrainingArguments`` (or ``GRPOConfig``).
    """
    ratio = training.get("warmup_ratio")
    if ratio is not None:
        return {"warmup_ratio": float(ratio)}
    steps = training.get("warmup_steps")
    if steps is not None:
        return {"warmup_steps": int(steps)}
    return {"warmup_steps": int(default_warmup_steps)}


def _save_base_ref(output_dir: Path, model_name: str, revision: str | None = None):
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
    with open(base_dir / "model_ref.json", "w") as f:
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

    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    base_dir = output_dir / "base"

    # Resume: base/ already has merged weights from first run
    if (base_dir / "config.json").exists():
        logger.info(f"Loading base model from {base_dir} (cached from prior run)")
        if is_ocr_task(task):
            return AutoModelForImageTextToText.from_pretrained(
                str(base_dir),
                torch_dtype="auto",
                attn_implementation="sdpa",
            )
        else:
            return AutoModelForCausalLM.from_pretrained(
                str(base_dir),
                torch_dtype="auto",
                attn_implementation="sdpa",
            )

    # First run: download base, merge adapter, save to base/
    from peft import PeftModel

    logger.info(f"Loading base model: {model_name}")
    if is_ocr_task(task):
        base_model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype="auto",
            attn_implementation="sdpa",
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            attn_implementation="sdpa",
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

    # Write model_ref.json for lineage tracking (adapter source path)
    ref = {"model_name_or_path": adapter_path}
    with open(base_dir / "model_ref.json", "w") as f:
        json.dump(ref, f, indent=2)
    logger.info(f"Base model saved (ref: {adapter_path})")

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


def _save_model(model, save_dir, model_name=None, processor=None, tokenizer=None):
    """Save a model with unified structure (merged model + adapter + base ref).

    For LoRA: saves adapter, base reference, then merges and saves merged model.
    For Full FT: saves full model and base reference.
    Always saves processor/tokenizer.

    Args:
        model: Model to save (may be a PeftModel with LoRA adapters)
        save_dir: Directory to save to
        model_name: Fallback model name for base ref (used for full FT)
        processor: AutoProcessor for VLMs (optional)
        tokenizer: Tokenizer for causal LMs (optional)
    """
    from peft import PeftModel

    save_dir = Path(save_dir)

    # Unwrap DDP/FSDP wrapper if present (multi-GPU training)
    model = getattr(model, "module", model)

    if isinstance(model, PeftModel):
        # LoRA: save adapter first (merge is destructive)
        logger.info("Saving LoRA adapter...")
        model.save_pretrained(save_dir)

        # Save base ref if not already saved with full weights (--adapter case)
        if not (save_dir / "base" / "config.json").exists():
            base_config = model.get_base_model().config
            _save_base_ref(
                save_dir,
                model_name=base_config._name_or_path,
                revision=getattr(base_config, "_commit_hash", None),
            )

        logger.info("Merging LoRA and saving merged model...")
        merged = model.merge_and_unload()
        merged.save_pretrained(save_dir)
    else:
        # Full FT: save complete model
        logger.info("Saving model...")
        model.save_pretrained(save_dir)
        if not (save_dir / "base" / "config.json").exists():
            _save_base_ref(
                save_dir,
                model_name=model_name or model.config._name_or_path,
                revision=getattr(model.config, "_commit_hash", None),
            )

    if processor is not None:
        processor.save_pretrained(save_dir)
    elif tokenizer is not None:
        tokenizer.save_pretrained(save_dir)
    logger.info(f"Model saved to {save_dir}")


def _save_final_model(trainer, config, output_dir, processor=None, tokenizer=None):
    """Save the student model after training. Delegates to _save_model.

    Args:
        trainer: HF Trainer instance (model accessible via trainer.model)
        config: Full config dict
        output_dir: Run output directory
        processor: AutoProcessor for VLMs (optional)
        tokenizer: Tokenizer for causal LMs (optional)
    """
    _save_model(
        trainer.model,
        output_dir,
        model_name=config["model"]["name"],
        processor=processor,
        tokenizer=tokenizer,
    )


def _save_last_model(trainer, config, output_dir, last_checkpoint, processor=None, tokenizer=None):
    """Reload the last completed checkpoint and export it to output_dir/last.

    Used when checkpointing.final_model_save is "both", after the best model has
    already been saved to output_dir (trainer.model holds the best-checkpoint
    weights at that point because load_best_model_at_end reloaded them at the end
    of trainer.train()). Reloads the last on-disk checkpoint via the same internal
    HF mechanism used for --resume, mutating trainer.model in place, then delegates
    to _save_model for the same merged/base-ref/processor export the best model gets.

    Args:
        trainer: HF Trainer instance (model mutated in place to last-checkpoint weights)
        config: Full config dict
        output_dir: Run output directory (last/ subdir created under this)
        last_checkpoint: Path to the last completed checkpoint dir, or None if none exist
        processor: AutoProcessor for VLMs (optional)
        tokenizer: Tokenizer for causal LMs (optional)
    """
    if last_checkpoint is None:
        logger.warning("final_model_save is 'both' but no checkpoint was found to export as 'last'")
        return

    logger.info(f"Reloading last checkpoint from {last_checkpoint} for last-model export...")
    trainer._load_from_checkpoint(last_checkpoint)
    _save_model(
        trainer.model,
        Path(output_dir) / "last",
        model_name=config["model"]["name"],
        processor=processor,
        tokenizer=tokenizer,
    )


def _remove_hf_checkpoint_dirs_after_success(output_dir: Path) -> None:
    """Remove Hugging Face ``checkpoint-*`` trees after a normal training finish.

    Called only when ``TrainingState.is_complete()`` so interrupted runs still keep
    checkpoints for ``--resume``. Preserves root model export, ``teacher/``,
    ``config.yaml``, ``run_metadata.json``, ``training_state.json``, and logs.
    Drops ``latest_checkpoint`` if any checkpoint dir was removed (stale pointer).
    """
    removed_any = False
    for path in sorted(output_dir.glob("checkpoint-*")):
        if not path.is_dir():
            continue
        try:
            int(path.name.split("-", 1)[1])
        except (ValueError, IndexError):
            continue
        shutil.rmtree(path)
        removed_any = True
    if removed_any:
        latest = output_dir / LATEST_CHECKPOINT_FILE
        if latest.exists():
            try:
                latest.unlink()
            except OSError as exc:
                logger.warning("Could not remove %s: %s", latest, exc)
        logger.info(
            "Removed HF checkpoint-* directories under %s (final weights kept at run root)",
            output_dir,
        )


def _remove_model_weights(output_dir: Path) -> None:
    """Delete model weight files and subdirs from output_dir after post-training eval.

    Keeps run metadata (config.yaml, run_metadata.json, training_state.json, output.log).
    Removes weight files (*.safetensors, *.bin, *.pt), HF config/tokenizer files,
    adapter files, and the teacher/, base/, and last/ subdirectories.

    Called in no_weights mode after post-training evaluation has completed so that
    results are already logged to W&B before weights are removed.

    Args:
        output_dir: Run output directory (Path).
    """
    _KEEP = {
        "config.yaml",
        "run_metadata.json",
        "training_state.json",
        "output.log",
        "wandb",
    }
    _WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt"}
    _HF_FILES = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
        "adapter_config.json",
    }
    removed = []
    for path in sorted(output_dir.iterdir()):
        if path.name in _KEEP:
            continue
        if path.is_dir() and path.name in ("teacher", "base", "last"):
            shutil.rmtree(path)
            removed.append(path.name)
        elif path.is_file() and (path.suffix in _WEIGHT_SUFFIXES or path.name in _HF_FILES):
            path.unlink()
            removed.append(path.name)
    if removed:
        logger.info(
            "no_weights mode: removed model files from %s: %s", output_dir, ", ".join(removed)
        )


def _train(
    config,
    device,
    tokenizer,
    num_labels,
    task,
    train_dataset,
    eval_dataset,
    compute_metrics,
    data_collator,
    output_dir,
    run_name,
    args,
    state,
    resume_checkpoint,
    env,
    report_to,
    metadata,
    processor=None,
):
    """Run the training phase with models scoped locally.

    All model references (model, teacher, trainer, loss_strategy, projector)
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

    Returns:
        ``True`` if training was interrupted by a SLURM signal (caller should exit
        without marking the job complete), ``False`` otherwise.
    """
    # Load model
    eager = needs_attention_outputs(config)
    if eager:
        logger.info("Using eager attention (required for attention distillation)")
    lora_config = config["training"]["lora"] if config.get("training", {}).get("use_lora") else None
    logger.info("Loading model...")
    model = load_model(
        config["model"]["name"],
        task,
        device,
        init=config["model"].get("init", "pretrained"),
        tokenizer=tokenizer,
        num_labels=num_labels,
        eager=eager,
        lora_config=lora_config,
    )

    # Load teacher (if applicable)
    teacher = None
    teacher_cfg = config["model"].get("teacher", {})
    teacher_name = teacher_cfg.get("name") if isinstance(teacher_cfg, dict) else teacher_cfg
    freeze_t = teacher_cfg.get("freeze", True) if isinstance(teacher_cfg, dict) else True
    distill_coeffs = resolve_distillation_coefficients(
        config.get("distillation", {}), freeze_teacher=freeze_t
    )
    if teacher_name and distill_coeffs["student_soft_weight"] > 0.0:
        teacher_load_path = teacher_name
        if resume_checkpoint and not freeze_t:
            candidate = Path(resume_checkpoint) / "teacher"
            if candidate.exists():
                teacher_load_path = str(candidate)
                logger.info(f"Resuming co-trained teacher from: {teacher_load_path}")
        logger.info("Loading teacher...")
        teacher = load_model(
            teacher_load_path, task, device, tokenizer=tokenizer, num_labels=num_labels, eager=eager
        )
        # Save teacher base reference for lineage tracking (mirrors base/model_ref.json)
        teacher_base_dir = output_dir / "teacher" / "base"
        teacher_base_dir.mkdir(parents=True, exist_ok=True)
        ref = {"model_name_or_path": teacher_name}
        revision = getattr(teacher.config, "_commit_hash", None)
        if revision:
            ref["revision"] = revision
        with open(teacher_base_dir / "model_ref.json", "w") as f:
            json.dump(ref, f, indent=2)
        logger.info(f"Teacher ref saved: {teacher_name}")
    else:
        reason = "no teacher configured" if not teacher_name else "alpha_s=0.0"
        logger.info(f"No teacher ({reason}) - CE-only mode")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=int(config["training"]["num_epochs"]),
        per_device_train_batch_size=int(config["training"]["batch_size"]),
        per_device_eval_batch_size=int(
            config["training"].get("eval_batch_size", config["training"]["batch_size"])
        ),
        gradient_accumulation_steps=int(config["training"].get("gradient_accumulation_steps", 1)),
        learning_rate=float(config["training"]["learning_rate"]),
        **warmup_kwargs_for_training_args(config["training"]),
        weight_decay=float(config["training"]["weight_decay"]),
        max_grad_norm=float(config["training"]["max_grad_norm"]),
        logging_steps=int(config["logging"]["logging_steps"]),
        save_steps=int(config["checkpointing"]["save_steps"]),
        save_total_limit=int(config["checkpointing"]["save_total_limit"]),
        eval_strategy=config.get("evaluation", {}).get("eval_strategy", "steps"),
        eval_steps=int(config.get("evaluation", {}).get("eval_steps", 500)),
        report_to=report_to,
        fp16=(
            (config["training"].get("fp16", True) and model.dtype != torch.float16)
            if device == "cuda"
            else False
        ),
        dataloader_num_workers=(
            config["training"].get("dataloader_num_workers", get_num_workers())
            if device == "cuda"
            else 0
        ),
        dataloader_pin_memory=device == "cuda",  # Disable on MPS (not supported)
        run_name=run_name,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        load_best_model_at_end=should_load_best_model(config),
        # Select best checkpoint by task metric (accuracy/f1/mcc), not loss.
        # Loss and metric can diverge; metric-based selection ensures we report the best achieved.
        # For translation: use loss since BLEU is computed separately by callback (generation-based),
        # not in eval metrics (which only has loss due to prediction_loss_only=True to avoid OOM).
        metric_for_best_model=(
            "eval_loss"
            if num_labels is None
            else (f"eval_{get_primary_metric(task)}" if get_primary_metric(task) else "eval_loss")
        ),
        greater_is_better=(
            False if num_labels is None else (get_primary_metric(task) is not None)
        ),  # Loss: lower is better; accuracy/f1/mcc: higher is better
        prediction_loss_only=num_labels
        is None,  # Seq2seq tasks: prevents OOM, metrics need generation anyway
        seed=args.seed,
        data_seed=args.seed,
        # VLM datasets store raw images + text; the collator handles tokenization.
        # Must keep these columns so the collator can process them.
        remove_unused_columns=not is_ocr_task(task),
        gradient_checkpointing=config["training"].get("gradient_checkpointing", False),
    )

    # Choose trainer based on whether we have a teacher
    if teacher is None:
        logger.info("No teacher - using FinetuningTrainer (CE-only mode)")
        trainer = FinetuningTrainer(
            model=model,
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
        loss_strategy, teacher_loss_strategy, layer_mapping, projector = (
            create_loss_strategy_from_config(
                config["distillation"],
                model,
                teacher,
                device,
                freeze_teacher=config["model"].get("teacher", {}).get("freeze", True),
            )
        )

        trainer = DistillationTrainer(
            teacher_model=teacher,
            temperature=config["distillation"]["temperature"],
            alpha_h=distill_coeffs["alpha_h"],
            alpha_s=distill_coeffs["alpha_s"],
            beta_h=distill_coeffs["beta_h"],
            beta_s=distill_coeffs["beta_s"],
            freeze_teacher=config["model"].get("teacher", {}).get("freeze", True),
            loss_strategy=loss_strategy,
            teacher_loss_strategy=teacher_loss_strategy,
            teacher_label_map=config["distillation"].get("teacher_label_map"),
            layer_mapping=layer_mapping,
            projector=projector,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )

    # Replace HF's default dict-dump logging with our clean format
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(StepMetricsCallback())

    # Add SLURM signal handler
    slurm_cb = None
    if env == "slurm":
        logger.info("Adding SLURM signal handler...")
        slurm_cb = SLURMSignalCallback(trainer)
        trainer.add_callback(slurm_cb)

    # Add teacher eval during training (co-training only, before PeakMetrics so peaks are tracked)
    if isinstance(trainer, DistillationTrainer) and not trainer.freeze_teacher:
        logger.info("Adding student/teacher grad norm logging (co-training)...")
        trainer.add_callback(StudentTeacherGradNormCallback(trainer))
        logger.info("Adding teacher eval callback (co-training)...")
        trainer.add_callback(
            TeacherEvalCallback(
                trainer,
                teacher_label_map=trainer.teacher_label_map,
            )
        )

    # Add peak metrics tracking
    logger.info("Adding peak metrics callback...")
    trainer.add_callback(PeakMetricsCallback())

    eval_cfg = config.get("evaluation", {})
    esp = eval_cfg.get("early_stopping_patience")
    if esp is not None and int(esp) > 0:
        if should_load_best_model(config):
            logger.info(f"Early stopping enabled (patience={int(esp)})")
            trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=int(esp)))
        else:
            logger.warning(
                "evaluation.early_stopping_patience is set but eval_strategy does not run "
                "evaluation; skipping EarlyStoppingCallback"
            )

    # Add translation metrics callback for seq2seq tasks
    if num_labels is None and config.get("evaluation", {}).get(
        "compute_translation_metrics", False
    ):
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
    state.save(epoch=trainer.state.epoch, step=trainer.state.global_step)

    signal_interrupted = slurm_cb is not None and slurm_cb.caught_signal

    # Save final model (skip if training was interrupted by SLURM signal - checkpoint already saved)
    if not signal_interrupted:
        _save_final_model(trainer, config, output_dir, processor=processor, tokenizer=tokenizer)

        # Save co-trained teacher (weights changed during training)
        if teacher is not None and not config["model"].get("teacher", {}).get("freeze", True):
            teacher_cfg = config["model"].get("teacher", {})
            teacher_name = teacher_cfg.get("name") if isinstance(teacher_cfg, dict) else teacher_cfg
            teacher_dir = output_dir / "teacher"
            logger.info(f"Saving co-trained teacher to {teacher_dir}...")
            _save_model(
                teacher,
                teacher_dir,
                model_name=teacher_name,
                processor=processor,
                tokenizer=tokenizer,
            )

        if should_save_last_model(config):
            last_checkpoint = state.get_resume_checkpoint(output_dir)
            _save_last_model(
                trainer, config, output_dir, last_checkpoint, processor=processor, tokenizer=tokenizer
            )

        _remove_hf_checkpoint_dirs_after_success(output_dir)
    else:
        logger.info("Training interrupted - skipping final model save (checkpoint already saved)")

    return signal_interrupted


def _build_grpo_config(
    config,
    output_dir,
    run_name,
    args,
    device,
    report_to,
    reward_weights=None,
    chat_template_kwargs=None,
):
    """Build a GRPOConfig from the Forge YAML config.

    Args:
        config: Full config dict with 'rl' and 'training' sections.
        output_dir: Path to output directory.
        run_name: W&B display name.
        args: Parsed CLI arguments.
        device: Device string.
        report_to: Logging target.
        reward_weights: Resolved reward weights from get_reward_funcs.
        chat_template_kwargs: Extra kwargs (e.g. ``enable_thinking``) forwarded to the
            chat template when formatting generations.

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
        generation_batch_size=(
            int(rl["generation_batch_size"])
            if rl.get("generation_batch_size") is not None
            else None
        ),
        max_completion_length=int(rl.get("max_completion_length", 256)),
        beta=float(rl.get("beta", 0.0)),
        temperature=float(rl.get("temperature", 1.0)),
        loss_type=rl.get("loss_type", "grpo"),
        reward_weights=reward_weights,
        num_iterations=int(rl.get("num_iterations", 1)),
        scale_rewards=rl.get("scale_rewards", "group"),
        # Training params from config
        num_train_epochs=int(training["num_epochs"]),
        per_device_train_batch_size=int(training.get("batch_size", 4)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        learning_rate=float(rl.get("learning_rate", training.get("learning_rate", 1e-6))),
        **warmup_kwargs_for_training_args(training, default_warmup_steps=200),
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
        gradient_checkpointing=training.get("gradient_checkpointing", False),
        bf16=(device == "cuda"),
        dataloader_num_workers=(
            training.get("dataloader_num_workers", get_num_workers()) if device == "cuda" else 0
        ),
        dataloader_pin_memory=(device == "cuda"),
        # Overrides
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        seed=args.seed,
        data_seed=args.seed,
        # Log generated completions to W&B for inspection
        log_completions=rl.get("log_completions", True),
        # Chat template kwargs (e.g. enable_thinking) from model profiles + config
        chat_template_kwargs=chat_template_kwargs,
    )
    return grpo_args


def _train_rl(
    config,
    device,
    tokenizer,
    task,
    train_dataset,
    eval_dataset,
    output_dir,
    run_name,
    args,
    state,
    resume_checkpoint,
    env,
    report_to,
    metadata,
    processor=None,
    chat_template_kwargs=None,
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
        chat_template_kwargs: Extra kwargs (e.g. ``enable_thinking``) forwarded to the
            chat template when formatting generations.

    Returns:
        ``True`` if training was interrupted by a SLURM signal (caller should exit
        without marking the job complete), ``False`` otherwise.
    """
    from peft import LoraConfig
    from trl import GRPOTrainer

    # Build reward functions from config
    reward_funcs, reward_weights = get_reward_funcs(config)
    logger.info(f"Reward functions: {[f.__name__ for f in reward_funcs]}")

    # Build GRPOConfig (pass resolved weights)
    grpo_config = _build_grpo_config(
        config,
        output_dir,
        run_name,
        args,
        device,
        report_to,
        reward_weights=reward_weights,
        chat_template_kwargs=chat_template_kwargs,
    )
    logger.info(
        f"GRPO config: num_generations={grpo_config.num_generations}, "
        f"max_completion_length={grpo_config.max_completion_length}, "
        f"beta={grpo_config.beta}, loss_type={grpo_config.loss_type}"
    )

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

    processing_class = processor if processor is not None else tokenizer

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        peft_config=peft_config,
    )

    # Replace HF's default dict-dump logging with our clean format
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(StepMetricsCallback())

    # Add SLURM signal handler
    slurm_cb = None
    if env == "slurm":
        logger.info("Adding SLURM signal handler...")
        slurm_cb = SLURMSignalCallback(trainer)
        trainer.add_callback(slurm_cb)

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

    signal_interrupted = slurm_cb is not None and slurm_cb.caught_signal

    # Save final model (skip if training was interrupted by SLURM signal)
    if not signal_interrupted:
        _save_final_model(trainer, config, output_dir, processor=processor, tokenizer=tokenizer)

        if should_save_last_model(config):
            last_checkpoint = state.get_resume_checkpoint(output_dir)
            _save_last_model(
                trainer, config, output_dir, last_checkpoint, processor=processor, tokenizer=tokenizer
            )

        _remove_hf_checkpoint_dirs_after_success(output_dir)
    else:
        logger.info("Training interrupted - skipping final model save (checkpoint already saved)")

    return signal_interrupted


def main():
    """CLI entry point: parse args, resolve a new-run or --resume output directory, then dispatch
    to ``_train`` (SFT/KD) or ``_train_rl`` (GRPO) based on the config.

    Registers the run in the SLURM incomplete-jobs registry for the duration of training and
    writes config/metadata snapshots to ``output_dir`` as a side effect.
    """
    parser = argparse.ArgumentParser(description="Forge Training")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config file (required for new runs)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to output dir to resume (e.g. outputs/classification/mnli/.../20260304-1200Z)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name for flat output dir (e.g. seed42 -> outputs/seed42/). Create-only.",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Append timestamp to --run_name (e.g. seed42 -> seed42_20260304-1200Z).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing named run (deletes outputs and W&B run, starts fresh).",
    )
    parser.add_argument(
        "--max_steps", type=int, default=-1, help="Max training steps (for testing)"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs from config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to adapter to merge before training (e.g. outputs/my_sft_run/)",
    )
    parser.add_argument(
        "--teacher_adapter",
        type=str,
        default=None,
        help="Path to teacher checkpoint/adapter (overrides model.teacher.name)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable DEBUG level logging for verbose output"
    )
    args = parser.parse_args()

    setup_logging(debug=args.debug)

    # Timestamp for new runs
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%MZ")

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
            resume_path = resume_path[len("outputs/") :]
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
            logger.warning(
                "--timestamp without --run_name is redundant (auto-named runs always include timestamp)"
            )
        if args.force and not args.run_name:
            logger.warning(
                "--force without --run_name is redundant (auto-named runs are always unique)"
            )

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
                    existing_metadata["display_name"] = existing_metadata.get(
                        "run_name", output_dir.name
                    )
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
    if args.teacher_adapter is not None:
        model_cfg = config.setdefault("model", {})
        if not isinstance(model_cfg.get("teacher"), dict):
            model_cfg["teacher"] = {}
        model_cfg["teacher"]["name"] = args.teacher_adapter
        logger.info(f"CLI override: teacher={args.teacher_adapter}")

    logger.info(f"Output directory: {args.output_dir}")

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
        if args.teacher_adapter:
            metadata["teacher_adapter"] = args.teacher_adapter
        output_dir.mkdir(parents=True, exist_ok=True)
        save_run_metadata(output_dir, metadata)
        save_config_snapshot(output_dir, config)
        logger.info(f"Created run metadata: {output_dir / 'run_metadata.json'}")

    # Track job in incomplete registry for SLURM auto-resubmission
    resume_path = (
        str(output_dir.relative_to("./outputs"))
        if str(output_dir).startswith("./outputs")
        else str(output_dir).replace("outputs/", "", 1)
    )
    add_incomplete_job(resume_path, output_dir)

    # Setup W&B
    if args.debug:
        os.environ["WANDB_PROJECT"] = "Forge-Test"
    else:
        wandb_project_yaml = config.get("logging", {}).get("wandb_project")
        if wandb_project_yaml:
            os.environ["WANDB_PROJECT"] = str(wandb_project_yaml)
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
                tags=(
                    os.environ.get("WANDB_TAGS", "").split(",")
                    if os.environ.get("WANDB_TAGS")
                    else None
                ),
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
        handler_kwargs = _build_handler_kwargs(config, dataset_name)
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

        # Resolve chat_template_kwargs: model profile defaults + config override
        from src.inference.model_profiles import get_chat_template_kwargs

        ct_kwargs = get_chat_template_kwargs(
            config["model"]["name"],
            explicit=config.get("model", {}).get("chat_template_kwargs"),
            task=task,
        )
        if ct_kwargs:
            logger.info(f"Chat template kwargs: {ct_kwargs}")

        set_reproducibility_seed(args.seed)
        signal_interrupted = _train_rl(
            config,
            device,
            tokenizer,
            task,
            train_dataset,
            eval_dataset,
            output_dir,
            display_name,
            args,
            state,
            resume_checkpoint,
            env,
            report_to,
            metadata,
            processor=processor,
            chat_template_kwargs=ct_kwargs or None,
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
            tokenizer_model = (
                teacher_cfg.get("name") if isinstance(teacher_cfg, dict) else teacher_cfg
            )
            if tokenizer_model is None:
                tokenizer_model = config["model"]["name"]
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
            from src.tasks import is_translation_task

            if "llama" in tokenizer_model.lower() or is_translation_task(task):
                tokenizer.padding_side = "left"
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

        # Load raw data via handler
        dataset_name = config["data"].get("dataset", task)
        handler = get_handler(dataset_name)
        handler_kwargs = _build_handler_kwargs(config, dataset_name)
        train_dataset, eval_dataset = handler.load(**handler_kwargs)

        # Format for SFT (cached to output_dir/.formatted_cache/ for fast restarts)
        max_length = config.get("data", {}).get("max_length", 1024)
        tokenizer_or_processor = processor if processor else tokenizer

        # Resolve chat_template_kwargs: model profile defaults + config override
        from src.inference.model_profiles import get_chat_template_kwargs

        ct_kwargs = get_chat_template_kwargs(
            config["model"]["name"],
            explicit=config.get("model", {}).get("chat_template_kwargs"),
            task=task,
        )
        if ct_kwargs:
            logger.info(f"Chat template kwargs: {ct_kwargs}")

        train_dataset = format_for_sft(
            train_dataset,
            task_type,
            tokenizer_or_processor,
            max_length,
            cache_dir=output_dir,
            split="train",
            chat_template_kwargs=ct_kwargs or None,
        )
        if eval_dataset is not None:
            eval_dataset = format_for_sft(
                eval_dataset,
                task_type,
                tokenizer_or_processor,
                max_length,
                cache_dir=output_dir,
                split="eval",
                chat_template_kwargs=ct_kwargs or None,
            )

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
        model_pre = teacher_pre = None
        default_log_pre_post = num_labels is not None
        log_pre_post = config.get("evaluation", {}).get(
            "log_pre_post_metrics", default_log_pre_post
        )
        if log_pre_post and num_labels is None:
            logger.warning(
                "log_pre_post_metrics enabled for translation - this may take hours on large eval sets"
            )
        if log_pre_post:
            set_reproducibility_seed(args.seed)
            model_pre, teacher_pre = run_evaluation(
                config,
                device,
                tokenizer,
                num_labels,
                eval_dataset,
                compute_metrics,
                data_collator,
                task,
                phase="pre",
                remove_unused_columns=not is_ocr_task(task),
            )

        # Training
        set_reproducibility_seed(args.seed)
        signal_interrupted = _train(
            config,
            device,
            tokenizer,
            num_labels,
            task,
            train_dataset,
            eval_dataset,
            compute_metrics,
            data_collator,
            output_dir,
            display_name,
            args,
            state,
            resume_checkpoint,
            env,
            report_to,
            metadata,
            processor=processor,
        )

        # Post-training evaluation (skip if training was interrupted by SLURM signal)
        if not signal_interrupted:
            set_reproducibility_seed(args.seed)
            model_post, teacher_post = run_evaluation(
                config,
                device,
                tokenizer,
                num_labels,
                eval_dataset,
                compute_metrics,
                data_collator,
                task,
                phase="post",
                checkpoint_dir=output_dir,
                evaluate_teacher=log_pre_post,
                prev_metrics=(model_pre, teacher_pre) if log_pre_post else None,
                remove_unused_columns=not is_ocr_task(task),
            )
            if no_weights_mode(config):
                _remove_model_weights(output_dir)
        else:
            logger.info("Training interrupted - skipping post-training evaluation")

    # Mark full pipeline complete and remove from incomplete registry
    if not signal_interrupted:
        state.mark_pipeline_complete()
        remove_incomplete_job()
        logger.info("=" * 60)
        if state.is_complete():
            logger.info("Training complete!")
        else:
            if state.state.get("target_steps"):
                logger.info(
                    f"Training complete (early stopping at step {state.state['completed_steps']}/{state.state['target_steps']})"
                )
            else:
                logger.info(
                    f"Training complete (early stopping at epoch {state.state['completed_epochs']:.2f}/{state.state['target_epochs']})"
                )
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        if state.state.get("target_steps"):
            logger.info(
                f"Training interrupted (step {state.state['completed_steps']}/{state.state['target_steps']})"
            )
        else:
            logger.info(
                f"Training interrupted (epoch {state.state['completed_epochs']}/{state.state['target_epochs']})"
            )
        logger.info("Job remains in incomplete registry for resubmission")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
