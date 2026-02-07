"""Attention analysis for comparing teacher-student attention patterns."""

import logging
from typing import Optional

import numpy as np

from workbench.utils import load_model, load_task_data, extract_attention_weights

logger = logging.getLogger(__name__)


def compute_attention_entropy(attn_weights: np.ndarray, attention_mask: np.ndarray = None) -> np.ndarray:
    """
    Compute entropy of attention distributions.

    Higher entropy = attention spread across many tokens (diffuse).
    Lower entropy = attention focused on few tokens (sharp).

    Args:
        attn_weights: Attention weights [N, num_heads, seq, seq]
        attention_mask: Optional mask to exclude padding [N, seq]

    Returns:
        Entropy per sample per head [N, num_heads]
    """
    # Attention from CLS token (position 0) to all tokens
    cls_attn = attn_weights[:, :, 0, :]  # [N, num_heads, seq]

    # Mask padding tokens if attention_mask provided
    if attention_mask is not None:
        mask = attention_mask[:, np.newaxis, :]  # [N, 1, seq]
        cls_attn = cls_attn * mask
        # Re-normalize after masking
        cls_attn = cls_attn / (cls_attn.sum(axis=-1, keepdims=True) + 1e-10)

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    cls_attn = np.clip(cls_attn, eps, 1.0)

    # Compute entropy: -sum(p * log(p))
    entropy = -np.sum(cls_attn * np.log(cls_attn), axis=-1)  # [N, num_heads]

    return entropy


def compute_layer_entropy_stats(layer_attentions: dict, attention_mask: np.ndarray = None) -> dict:
    """
    Compute entropy statistics per layer.

    Args:
        layer_attentions: Dict of layer_idx -> attention weights
        attention_mask: Optional mask to exclude padding [N, seq]

    Returns:
        Dict with entropy stats per layer
    """
    stats = {}

    for layer_idx, attn in layer_attentions.items():
        entropy = compute_attention_entropy(attn, attention_mask)

        # Average over samples, keep per-head info
        head_entropy = entropy.mean(axis=0)  # [num_heads]

        stats[f"layer_{layer_idx + 1}"] = {
            "mean_entropy": float(entropy.mean()),
            "std_entropy": float(entropy.std()),
            "per_head_entropy": head_entropy.tolist(),
            "max_head": int(np.argmax(head_entropy)),
            "min_head": int(np.argmin(head_entropy)),
        }

    return stats


def run_attention_analysis(
    teacher_model: str,
    student_model: str,
    task: str = "mnli",
    device: str = "cuda",
    max_samples: Optional[int] = 1000,
) -> dict:
    """
    Run attention entropy analysis comparing teacher and student.

    Args:
        teacher_model: Teacher model path or HF name
        student_model: Student model checkpoint path
        task: Task name
        device: Device to use
        max_samples: Max samples for analysis

    Returns:
        Dict with attention analysis results
    """
    logger.info("Loading models...")
    teacher, teacher_tok, teacher_layers = load_model(teacher_model, device)
    student, student_tok, student_layers = load_model(student_model, device)

    logger.info("Loading data...")
    texts, labels, task_cfg = load_task_data(task, max_samples)

    logger.info("Extracting teacher attention...")
    teacher_attn, teacher_mask = extract_attention_weights(
        teacher, teacher_tok, texts, device=device, batch_size=16
    )

    logger.info("Extracting student attention...")
    student_attn, student_mask = extract_attention_weights(
        student, student_tok, texts, device=device, batch_size=16
    )

    logger.info("Computing entropy statistics...")
    teacher_stats = compute_layer_entropy_stats(teacher_attn, teacher_mask)
    student_stats = compute_layer_entropy_stats(student_attn, student_mask)

    # Compute summary metrics
    teacher_mean_entropy = np.mean([s["mean_entropy"] for s in teacher_stats.values()])
    student_mean_entropy = np.mean([s["mean_entropy"] for s in student_stats.values()])

    # Layer-wise comparison (map student layers to teacher equivalents)
    layer_comparison = {}
    layer_ratio = teacher_layers / student_layers

    for s_layer in range(student_layers):
        s_key = f"layer_{s_layer + 1}"
        # Map to corresponding teacher layer (e.g., student 1 -> teacher 2)
        t_layer = int((s_layer + 1) * layer_ratio) - 1
        t_key = f"layer_{t_layer + 1}"

        if s_key in student_stats and t_key in teacher_stats:
            s_ent = student_stats[s_key]["mean_entropy"]
            t_ent = teacher_stats[t_key]["mean_entropy"]
            layer_comparison[s_key] = {
                "student_entropy": s_ent,
                "teacher_layer": t_key,
                "teacher_entropy": t_ent,
                "ratio": s_ent / t_ent if t_ent > 0 else 0,
            }

    return {
        "teacher": {
            "model": teacher_model,
            "layers": teacher_layers,
            "mean_entropy": float(teacher_mean_entropy),
            "layer_stats": teacher_stats,
        },
        "student": {
            "model": student_model,
            "layers": student_layers,
            "mean_entropy": float(student_mean_entropy),
            "layer_stats": student_stats,
        },
        "comparison": {
            "entropy_ratio": float(student_mean_entropy / teacher_mean_entropy),
            "layer_mapping": layer_comparison,
        },
        "samples": len(texts),
        "task": task,
    }
