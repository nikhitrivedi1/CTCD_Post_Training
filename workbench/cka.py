"""CKA (Centered Kernel Alignment) analysis for representation similarity."""

import logging

import numpy as np

from workbench.utils import load_model, load_task_data, extract_hidden_states

logger = logging.getLogger(__name__)


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute CKA between two representation matrices.

    Args:
        X: Representations [N, D1]
        Y: Representations [N, D2]

    Returns:
        CKA similarity score (0 to 1)
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Gram matrices
    K = X @ X.T
    L = Y @ Y.T

    # Center Gram matrices: HKH where H = I - 1/n * 11^T
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K = H @ K @ H
    L = H @ L @ H

    # HSIC
    hsic_xy = (K * L).sum()
    hsic_xx = (K * K).sum()
    hsic_yy = (L * L).sum()

    return hsic_xy / (np.sqrt(hsic_xx) * np.sqrt(hsic_yy) + 1e-10)


def compute_cka_matrix(
    model_hiddens: dict,
    reference_hiddens: dict,
) -> np.ndarray:
    """
    Compute CKA similarity matrix between all model and reference layers.

    Args:
        model_hiddens: Dict of layer_idx -> representations
        reference_hiddens: Dict of layer_idx -> representations

    Returns:
        CKA matrix [model_layers, reference_layers]
    """
    model_layers = sorted(model_hiddens.keys())
    ref_layers = sorted(reference_hiddens.keys())

    cka_matrix = np.zeros((len(model_layers), len(ref_layers)))

    for i, m_layer in enumerate(model_layers):
        for j, r_layer in enumerate(ref_layers):
            cka_matrix[i, j] = compute_cka(
                model_hiddens[m_layer],
                reference_hiddens[r_layer]
            )

    return cka_matrix


def run_cka(
    model_path: str,
    reference_path: str,
    task: str,
    device: str = "cuda",
    max_samples: int = 1000,
):
    """
    Run CKA analysis comparing model and reference representations.

    Args:
        model_path: Path to model
        reference_path: Path to reference model
        task: Task name
        device: Device
        max_samples: Max samples (CKA is O(N^2), keep small)

    Returns:
        Dict with CKA matrix and metadata
    """
    logger.info(f"Loading model: {model_path}")
    model, model_tok, model_layers = load_model(model_path, device)

    logger.info(f"Loading reference: {reference_path}")
    ref_model, ref_tok, ref_layers = load_model(reference_path, device)

    texts, _, _ = load_task_data(task, max_samples)

    logger.info("Extracting model hidden states...")
    model_hiddens = extract_hidden_states(
        model, model_tok, texts, device=device
    )

    logger.info("Extracting reference hidden states...")
    ref_hiddens = extract_hidden_states(
        ref_model, ref_tok, texts, device=device
    )

    logger.info("Computing CKA matrix...")
    cka_matrix = compute_cka_matrix(model_hiddens, ref_hiddens)

    # Find best alignment for each model layer
    best_alignment = {}
    for m_idx in range(cka_matrix.shape[0]):
        r_idx = np.argmax(cka_matrix[m_idx])
        m_name = "embeddings" if m_idx == 0 else f"layer_{m_idx}"
        r_name = "embeddings" if r_idx == 0 else f"layer_{r_idx}"
        best_alignment[m_name] = {
            "reference_layer": r_name,
            "cka_score": float(cka_matrix[m_idx, r_idx])
        }

    return {
        "cka_matrix": cka_matrix.tolist(),
        "model_layers": model_layers + 1,  # +1 for embeddings
        "reference_layers": ref_layers + 1,
        "best_alignment": best_alignment,
    }
