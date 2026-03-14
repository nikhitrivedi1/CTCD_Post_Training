"""Pre/post training evaluation orchestration.

Handles the full lifecycle of evaluation models: load → evaluate → return metrics.
Models are function-local and GC'd on return — no manual memory management needed.
"""

import inspect
import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from src.models import load_model
from src.tasks import get_primary_metric

logger = logging.getLogger(__name__)


def evaluate_model(
    model: torch.nn.Module,
    eval_dataset,
    tokenizer,
    compute_metrics,
    data_collator=None,
    batch_size: int = 32,
    device: str = None,
    loss_only: bool = False,
    remove_unused_columns: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a model using a direct torch inference loop.

    Uses a simple DataLoader + torch.no_grad() loop instead of creating a
    temporary HF Trainer. Creating a Trainer spawns a new Accelerator whose
    singleton AcceleratorState can overwrite the main Trainer's mixed-precision
    settings, corrupting subsequent training if called mid-run.

    Args:
        model: Model to evaluate
        eval_dataset: Evaluation dataset
        tokenizer: Tokenizer for the model
        compute_metrics: Metrics computation function
        data_collator: Optional data collator
        batch_size: Batch size for evaluation
        device: Device string (inferred from model if None)
        loss_only: If True, only compute loss (faster for seq2seq tasks)
        remove_unused_columns: If False, keep all dataset columns (needed for VLM tasks
            where the collator processes raw columns like 'image' and 'formatted_text')

    Returns:
        Dictionary of metric name -> value
    """
    from transformers import default_data_collator

    if device is None:
        device = str(next(model.parameters()).device)

    # Filter dataset columns to match model.forward() signature
    if remove_unused_columns and hasattr(eval_dataset, "column_names"):
        sig_cols = set(inspect.signature(model.forward).parameters.keys())
        sig_cols |= {"label", "label_ids"}  # HF convention
        cols_to_remove = [c for c in eval_dataset.column_names if c not in sig_cols]
        if cols_to_remove:
            eval_dataset = eval_dataset.remove_columns(cols_to_remove)

    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=data_collator or default_data_collator,
        shuffle=False,
        pin_memory="cuda" in str(device),
    )

    logger.info(f"Running evaluation ({len(eval_dataset)} samples)...")
    was_training = model.training
    model.eval()

    all_logits = []
    all_labels = []
    total_loss = 0.0
    num_samples = 0

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        bs = next(v.shape[0] for v in batch.values() if isinstance(v, torch.Tensor))
        if outputs.loss is not None:
            total_loss += outputs.loss.item() * bs
        num_samples += bs

        if not loss_only and hasattr(outputs, "logits"):
            all_logits.append(outputs.logits.cpu())
            if "labels" in batch:
                all_labels.append(batch["labels"].cpu())

    model.train(was_training)

    # Build metrics dict
    metrics = {}
    if num_samples > 0:
        metrics["loss"] = total_loss / num_samples

    if not loss_only and compute_metrics is not None and all_logits:
        all_logits_np = torch.cat(all_logits).numpy()
        all_labels_np = torch.cat(all_labels).numpy() if all_labels else None
        if all_labels_np is not None:
            metrics.update(compute_metrics((all_logits_np, all_labels_np)))

    return metrics


def log_pre_post_metrics(
    metrics: Dict[str, float],
    model_name: str,
    phase: str,
    to_wandb: bool = True,
) -> None:
    """
    Log pre/post training metrics with proper prefixes.

    Args:
        metrics: Dictionary of metrics
        model_name: "teacher" or "model"
        phase: "pre" or "post"
        to_wandb: Whether to log to W&B
    """
    prefix = f"{model_name}/{phase}"

    # Log to console
    logger.info(f"  {model_name.capitalize()} ({phase}-run):")
    for name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"    {name}: {value:.4f}")

    # Log to W&B
    if to_wandb:
        try:
            import wandb
            if wandb.run:
                wandb_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
                wandb.log(wandb_metrics, commit=False)
        except Exception as e:
            logger.debug(f"W&B logging failed: {e}")


def _remap_compute_metrics(compute_metrics, label_map: List[int]):
    """Wrap compute_metrics to remap logits before metric computation.

    Args:
        compute_metrics: Original compute_metrics function
        label_map: Index mapping — logits are reordered as logits[:, label_map]

    Returns:
        Wrapped compute_metrics function
    """
    def wrapped(eval_pred):
        logits, labels = eval_pred
        remapped = logits[:, label_map]
        return compute_metrics((remapped, labels))
    return wrapped


def evaluate_models(
    model: torch.nn.Module,
    teacher: Optional[torch.nn.Module],
    eval_dataset,
    tokenizer,
    compute_metrics,
    data_collator,
    eval_batch_size: int,
    device: str,
    phase: str,
    loss_only: bool = False,
    teacher_label_map: Optional[List[int]] = None,
    remove_unused_columns: bool = True,
) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
    """
    Evaluate model and teacher, logging metrics with the given phase prefix.

    Args:
        model: Model to evaluate
        teacher: Teacher model (or None)
        eval_dataset: Evaluation dataset
        tokenizer: Tokenizer
        compute_metrics: Metrics function
        data_collator: Data collator
        eval_batch_size: Batch size for eval
        device: Device string
        phase: "pre" or "post" (used for logging prefix)
        loss_only: If True, only compute loss (faster for seq2seq)
        teacher_label_map: Optional label index remapping for teacher logits
        remove_unused_columns: If False, keep all dataset columns (needed for VLM tasks)

    Returns:
        Tuple of (model_metrics, teacher_metrics)
    """
    eval_kwargs = dict(
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        batch_size=eval_batch_size,
        device=device,
        loss_only=loss_only,
        remove_unused_columns=remove_unused_columns,
    )

    logger.info(f"Evaluating model ({phase}-training)...")
    model_metrics = evaluate_model(model=model, **eval_kwargs)
    log_pre_post_metrics(model_metrics, "model", phase)

    teacher_metrics = None
    if teacher is not None:
        logger.info(f"Evaluating teacher ({phase}-training)...")
        teacher_eval_kwargs = eval_kwargs.copy()
        if teacher_label_map is not None:
            logger.info(f"  Remapping teacher logits for eval: {teacher_label_map}")
            teacher_eval_kwargs["compute_metrics"] = _remap_compute_metrics(
                compute_metrics, teacher_label_map
            )
        teacher_metrics = evaluate_model(model=teacher, **teacher_eval_kwargs)
        log_pre_post_metrics(teacher_metrics, "teacher", phase)

    return model_metrics, teacher_metrics


def log_transfer_metrics(
    task: str,
    model: Dict[str, float],
    teacher: Optional[Dict[str, float]],
    model_prev: Optional[Dict[str, float]] = None,
    teacher_prev: Optional[Dict[str, float]] = None,
) -> None:
    """
    Compute and log transfer metrics from evaluation results.

    Logs whatever it can from the available metrics:
    - Teacher advantage (teacher - model gap)
    - Model gain (current - previous)
    - Teacher gain (current - previous, co-training)
    - Transfer efficiency (model gain / pre-training teacher advantage)

    Args:
        task: Task name (for primary metric lookup)
        model: Current model metrics
        teacher: Current teacher metrics (or None)
        model_prev: Previous model metrics (or None)
        teacher_prev: Previous teacher metrics (or None)
    """
    primary_metric = get_primary_metric(task) or "loss"
    metrics = {}

    # Teacher-model gap at this point
    if teacher is not None and primary_metric in teacher and primary_metric in model:
        gap = teacher[primary_metric] - model[primary_metric]
        metrics["transfer/teacher_advantage"] = gap
        logger.info(f"  Teacher advantage ({primary_metric}): {gap:+.4f}")

    # Model gain (requires previous)
    if model_prev is not None and primary_metric in model and primary_metric in model_prev:
        model_gain = model[primary_metric] - model_prev[primary_metric]
        metrics["transfer/model_gain"] = model_gain
        logger.info(f"  Model gain ({primary_metric}): {model_gain:+.4f}")

        # Transfer efficiency (model gain / pre-training teacher advantage)
        if teacher_prev is not None and primary_metric in teacher_prev and primary_metric in model_prev:
            teacher_advantage_pre = teacher_prev[primary_metric] - model_prev[primary_metric]
            if abs(teacher_advantage_pre) > 1e-6:
                efficiency = model_gain / teacher_advantage_pre
                metrics["transfer/efficiency"] = efficiency
                logger.info(f"  Transfer efficiency: {efficiency:.2%}")

    # Teacher gain (co-training)
    if teacher is not None and teacher_prev is not None:
        if primary_metric in teacher and primary_metric in teacher_prev:
            teacher_gain = teacher[primary_metric] - teacher_prev[primary_metric]
            metrics["transfer/teacher_gain"] = teacher_gain
            logger.info(f"  Teacher gain ({primary_metric}): {teacher_gain:+.4f}")

    if metrics:
        try:
            import wandb
            if wandb.run:
                wandb.log(metrics, commit=False)
        except Exception as e:
            logger.debug(f"Transfer metrics logging failed: {e}")


def run_evaluation(
    config: dict,
    device: str,
    tokenizer,
    num_labels: Optional[int],
    eval_dataset,
    compute_metrics,
    data_collator,
    task: str,
    phase: str,
    checkpoint_dir=None,
    evaluate_teacher: bool = True,
    prev_metrics: Optional[Tuple[Dict, Optional[Dict]]] = None,
    remove_unused_columns: bool = True,
) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
    """
    Run a complete evaluation phase: load models, evaluate, return metrics.

    Models are local to this function and automatically GC'd on return.

    Args:
        config: Full configuration dictionary
        device: Device string ('cuda', 'mps', 'cpu')
        tokenizer: Tokenizer
        num_labels: Number of labels for classification (None for translation)
        eval_dataset: Evaluation dataset
        compute_metrics: Metrics computation function
        data_collator: Data collator (or None for classification)
        task: Task name (e.g., 'mnli', 'sst2')
        phase: "pre" or "post"
        checkpoint_dir: If provided, load models from checkpoint instead of config
        evaluate_teacher: Whether to evaluate the teacher model
        prev_metrics: Previous (model_metrics, teacher_metrics) for gain computation
        remove_unused_columns: If False, keep all dataset columns (needed for VLM tasks)

    Returns:
        Tuple of (model_metrics, teacher_metrics)
    """
    logger.info("=" * 60)
    logger.info(f"{'Pre' if phase == 'pre' else 'Post'}-training evaluation{' (baselines)' if phase == 'pre' else ''}...")
    logger.info("=" * 60)

    from pathlib import Path

    # Load model
    if checkpoint_dir is not None:
        model = load_model(str(checkpoint_dir), task, device,
                           tokenizer=tokenizer, num_labels=num_labels)
    else:
        lora_config = config["training"]["lora"] if config.get("training", {}).get("use_lora") else None
        model = load_model(config["model"]["name"], task, device,
                           init=config["model"].get("init", "pretrained"),
                           tokenizer=tokenizer, num_labels=num_labels,
                           lora_config=lora_config)

    # Load teacher (if applicable)
    teacher_for_eval = None
    if evaluate_teacher:
        teacher_cfg = config["model"].get("teacher", {})
        teacher_name = teacher_cfg.get("name") if isinstance(teacher_cfg, dict) else teacher_cfg
        alpha = config.get("distillation", {}).get("alpha", 0.5)
        if teacher_name and alpha > 0.0:
            if checkpoint_dir is not None:
                teacher_path = Path(checkpoint_dir) / "teacher"
                teacher_src = str(teacher_path) if teacher_path.exists() else teacher_name
            else:
                teacher_src = teacher_name
            teacher_for_eval = load_model(teacher_src, task, device,
                                          tokenizer=tokenizer, num_labels=num_labels)

    eval_batch_size = config["training"].get("eval_batch_size", config["training"]["batch_size"])
    loss_only = num_labels is None
    teacher_label_map = config.get("distillation", {}).get("teacher_label_map")

    model_metrics, teacher_metrics = evaluate_models(
        model=model,
        teacher=teacher_for_eval,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        eval_batch_size=eval_batch_size,
        device=device,
        phase=phase,
        loss_only=loss_only,
        teacher_label_map=teacher_label_map,
        remove_unused_columns=remove_unused_columns,
    )

    # Log transfer metrics if previous metrics provided
    if prev_metrics is not None:
        model_prev, teacher_prev = prev_metrics
        log_transfer_metrics(
            task, model_metrics, teacher_metrics,
            model_prev=model_prev, teacher_prev=teacher_prev,
        )

    return model_metrics, teacher_metrics
