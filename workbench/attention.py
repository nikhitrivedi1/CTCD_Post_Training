"""Attention analysis for comparing attention patterns between models."""

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
    model_name: str,
    reference_model: str,
    task: str = "mnli",
    device: str = "cuda",
    max_samples: Optional[int] = 1000,
) -> dict:
    """
    Run attention entropy analysis comparing two models.

    Args:
        model_name: Model checkpoint path
        reference_model: Reference model path or HF name
        task: Task name
        device: Device to use
        max_samples: Max samples for analysis

    Returns:
        Dict with attention analysis results
    """
    logger.info("Loading models...")
    model, model_tok, model_layers = load_model(model_name, device)
    ref, ref_tok, ref_layers = load_model(reference_model, device)

    logger.info("Loading data...")
    texts, labels, task_cfg = load_task_data(task, max_samples)

    logger.info("Extracting model attention...")
    model_attn, model_mask = extract_attention_weights(
        model, model_tok, texts, device=device, batch_size=16
    )

    logger.info("Extracting reference attention...")
    ref_attn, ref_mask = extract_attention_weights(
        ref, ref_tok, texts, device=device, batch_size=16
    )

    logger.info("Computing entropy statistics...")
    model_stats = compute_layer_entropy_stats(model_attn, model_mask)
    ref_stats = compute_layer_entropy_stats(ref_attn, ref_mask)

    # Compute summary metrics
    model_mean_entropy = np.mean([s["mean_entropy"] for s in model_stats.values()])
    ref_mean_entropy = np.mean([s["mean_entropy"] for s in ref_stats.values()])

    # Layer-wise comparison (map model layers to reference equivalents)
    layer_comparison = {}
    layer_ratio = ref_layers / model_layers

    for m_layer in range(model_layers):
        m_key = f"layer_{m_layer + 1}"
        # Map to corresponding reference layer (e.g., model 1 -> reference 2)
        r_layer = int((m_layer + 1) * layer_ratio) - 1
        r_key = f"layer_{r_layer + 1}"

        if m_key in model_stats and r_key in ref_stats:
            m_ent = model_stats[m_key]["mean_entropy"]
            r_ent = ref_stats[r_key]["mean_entropy"]
            layer_comparison[m_key] = {
                "model_entropy": m_ent,
                "reference_layer": r_key,
                "reference_entropy": r_ent,
                "ratio": m_ent / r_ent if r_ent > 0 else 0,
            }

    return {
        "model": {
            "model": model_name,
            "layers": model_layers,
            "mean_entropy": float(model_mean_entropy),
            "layer_stats": model_stats,
        },
        "reference": {
            "model": reference_model,
            "layers": ref_layers,
            "mean_entropy": float(ref_mean_entropy),
            "layer_stats": ref_stats,
        },
        "comparison": {
            "entropy_ratio": float(model_mean_entropy / ref_mean_entropy),
            "layer_mapping": layer_comparison,
        },
        "samples": len(texts),
        "task": task,
    }
