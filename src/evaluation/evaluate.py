"""Pre/post training evaluation orchestration.

Handles the full lifecycle of evaluation models: load → evaluate → return metrics.
Models are function-local and GC'd on return — no manual memory management needed.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch

from src.models import load_models, load_models_from_checkpoint
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
) -> Dict[str, float]:
    """
    Evaluate a model standalone (without the full Trainer overhead).

    Uses a minimal Trainer just for evaluation. This is used to get
    pre-training and post-training metrics for both teacher and student.

    Args:
        model: Model to evaluate
        eval_dataset: Evaluation dataset
        tokenizer: Tokenizer for the model
        compute_metrics: Metrics computation function
        data_collator: Optional data collator
        batch_size: Batch size for evaluation
        device: Device string for MPS compatibility
        loss_only: If True, only compute loss (faster for seq2seq tasks)

    Returns:
        Dictionary of metric name -> value
    """
    from transformers import Trainer, TrainingArguments

    # Minimal training args just for evaluation
    eval_args = TrainingArguments(
        output_dir="/tmp/eval_temp",
        per_device_eval_batch_size=batch_size,
        report_to="none",  # Don't log to W&B during standalone eval
        disable_tqdm=True,
        dataloader_pin_memory=device == "cuda",  # Disable on MPS
        prediction_loss_only=loss_only,  # Maps to HF's prediction_loss_only
    )

    eval_trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # Run evaluation (HF Trainer.evaluate() leaves model in eval mode)
    training_mode = model.training
    metrics = eval_trainer.evaluate()
    model.train(training_mode)

    # Clean up prefix and return
    return {k.replace("eval_", ""): v for k, v in metrics.items()}


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
        model_name: "teacher" or "student"
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
    student: torch.nn.Module,
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
) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
    """
    Evaluate student and teacher, logging metrics with the given phase prefix.

    Args:
        student: Student model
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

    Returns:
        Tuple of (student_metrics, teacher_metrics)
    """
    eval_kwargs = dict(
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        batch_size=eval_batch_size,
        device=device,
        loss_only=loss_only,
    )

    logger.info(f"Evaluating student ({phase}-training)...")
    student_metrics = evaluate_model(model=student, **eval_kwargs)
    log_pre_post_metrics(student_metrics, "student", phase)

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

    return student_metrics, teacher_metrics


def log_transfer_metrics(
    task: str,
    student: Dict[str, float],
    teacher: Optional[Dict[str, float]],
    student_prev: Optional[Dict[str, float]] = None,
    teacher_prev: Optional[Dict[str, float]] = None,
) -> None:
    """
    Compute and log transfer metrics from evaluation results.

    Logs whatever it can from the available metrics:
    - Teacher advantage (teacher - student gap)
    - Student gain (current - previous)
    - Teacher gain (current - previous, co-training)
    - Transfer efficiency (student gain / pre-training teacher advantage)

    Args:
        task: Task name (for primary metric lookup)
        student: Current student metrics
        teacher: Current teacher metrics (or None)
        student_prev: Previous student metrics (or None)
        teacher_prev: Previous teacher metrics (or None)
    """
    primary_metric = get_primary_metric(task) or "loss"
    metrics = {}

    # Teacher-student gap at this point
    if teacher is not None and primary_metric in teacher and primary_metric in student:
        gap = teacher[primary_metric] - student[primary_metric]
        metrics["transfer/teacher_advantage"] = gap
        logger.info(f"  Teacher advantage ({primary_metric}): {gap:+.4f}")

    # Student gain (requires previous)
    if student_prev is not None and primary_metric in student and primary_metric in student_prev:
        student_gain = student[primary_metric] - student_prev[primary_metric]
        metrics["transfer/student_gain"] = student_gain
        logger.info(f"  Student gain ({primary_metric}): {student_gain:+.4f}")

        # Transfer efficiency (student gain / pre-training teacher advantage)
        if teacher_prev is not None and primary_metric in teacher_prev and primary_metric in student_prev:
            teacher_advantage_pre = teacher_prev[primary_metric] - student_prev[primary_metric]
            if abs(teacher_advantage_pre) > 1e-6:
                efficiency = student_gain / teacher_advantage_pre
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
        prev_metrics: Previous (student_metrics, teacher_metrics) for gain computation

    Returns:
        Tuple of (student_metrics, teacher_metrics)
    """
    logger.info("=" * 60)
    logger.info(f"{'Pre' if phase == 'pre' else 'Post'}-training evaluation{' (baselines)' if phase == 'pre' else ''}...")
    logger.info("=" * 60)

    # Load models
    if checkpoint_dir is not None:
        teacher, student = load_models_from_checkpoint(
            checkpoint_dir, config, device, tokenizer, num_labels,
        )
    else:
        teacher, student = load_models(config, device, tokenizer, num_labels)

    # Skip teacher eval if not requested
    teacher_for_eval = teacher if evaluate_teacher else None

    eval_batch_size = config["training"].get("eval_batch_size", config["training"]["batch_size"])
    loss_only = num_labels is None
    teacher_label_map = config.get("distillation", {}).get("teacher_label_map")

    student_metrics, teacher_metrics = evaluate_models(
        student=student,
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
    )

    # Log transfer metrics if previous metrics provided
    if prev_metrics is not None:
        student_prev, teacher_prev = prev_metrics
        log_transfer_metrics(
            task, student_metrics, teacher_metrics,
            student_prev=student_prev, teacher_prev=teacher_prev,
        )

    return student_metrics, teacher_metrics
