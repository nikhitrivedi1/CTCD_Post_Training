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
    teacher_hiddens: dict,
    student_hiddens: dict,
) -> np.ndarray:
    """
    Compute CKA similarity matrix between all teacher and student layers.

    Args:
        teacher_hiddens: Dict of layer_idx -> representations
        student_hiddens: Dict of layer_idx -> representations

    Returns:
        CKA matrix [student_layers, teacher_layers]
    """
    student_layers = sorted(student_hiddens.keys())
    teacher_layers = sorted(teacher_hiddens.keys())

    cka_matrix = np.zeros((len(student_layers), len(teacher_layers)))

    for i, s_layer in enumerate(student_layers):
        for j, t_layer in enumerate(teacher_layers):
            cka_matrix[i, j] = compute_cka(
                student_hiddens[s_layer],
                teacher_hiddens[t_layer]
            )

    return cka_matrix


def run_cka(
    teacher_path: str,
    student_path: str,
    task: str,
    device: str = "cuda",
    max_samples: int = 1000,
):
    """
    Run CKA analysis comparing teacher and student representations.

    Args:
        teacher_path: Path to teacher model
        student_path: Path to student model
        task: Task name
        device: Device
        max_samples: Max samples (CKA is O(N^2), keep small)

    Returns:
        Dict with CKA matrix and metadata
    """
    logger.info(f"Loading teacher: {teacher_path}")
    teacher_model, teacher_tok, teacher_layers = load_model(teacher_path, device)

    logger.info(f"Loading student: {student_path}")
    student_model, student_tok, student_layers = load_model(student_path, device)

    texts, _, _ = load_task_data(task, max_samples)

    logger.info("Extracting teacher hidden states...")
    teacher_hiddens = extract_hidden_states(
        teacher_model, teacher_tok, texts, device=device
    )

    logger.info("Extracting student hidden states...")
    student_hiddens = extract_hidden_states(
        student_model, student_tok, texts, device=device
    )

    logger.info("Computing CKA matrix...")
    cka_matrix = compute_cka_matrix(teacher_hiddens, student_hiddens)

    # Find best alignment for each student layer
    best_alignment = {}
    for s_idx in range(cka_matrix.shape[0]):
        t_idx = np.argmax(cka_matrix[s_idx])
        s_name = "embeddings" if s_idx == 0 else f"layer_{s_idx}"
        t_name = "embeddings" if t_idx == 0 else f"layer_{t_idx}"
        best_alignment[s_name] = {
            "teacher_layer": t_name,
            "cka_score": float(cka_matrix[s_idx, t_idx])
        }

    return {
        "cka_matrix": cka_matrix.tolist(),
        "student_layers": student_layers + 1,  # +1 for embeddings
        "teacher_layers": teacher_layers + 1,
        "best_alignment": best_alignment,
    }
