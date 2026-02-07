"""Linear probing analysis for layer-wise task signal."""

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from workbench.utils import load_model, load_task_data, extract_hidden_states

logger = logging.getLogger(__name__)


def train_probe(hiddens: np.ndarray, labels: np.ndarray, train_ratio: float = 0.8):
    """
    Train a linear probe and evaluate.

    Args:
        hiddens: Hidden states [N, hidden_dim]
        labels: Labels [N]
        train_ratio: Train/val split ratio

    Returns:
        Accuracy on validation split
    """
    n_train = int(len(hiddens) * train_ratio)
    indices = np.random.permutation(len(hiddens))
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    X_train, y_train = hiddens[train_idx], labels[train_idx]
    X_val, y_val = hiddens[val_idx], labels[val_idx]

    clf = LogisticRegression(max_iter=5000, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    return accuracy


def run_probing(
    model_path: str,
    task: str,
    device: str = "cuda",
    max_samples: int = None,
    seed: int = 42,
):
    """
    Run linear probing analysis on a model.

    Args:
        model_path: Path to model (HuggingFace name or local checkpoint)
        task: Task name (mnli, qqp, sst2, cola)
        device: Device
        max_samples: Max samples for probing
        seed: Random seed

    Returns:
        Dict of layer_name -> accuracy
    """
    np.random.seed(seed)

    model, tokenizer, num_layers = load_model(model_path, device)
    texts, labels, _ = load_task_data(task, max_samples)

    layer_hiddens = extract_hidden_states(
        model, tokenizer, texts, device=device
    )
    labels_array = np.array(labels)

    results = {}
    for layer_idx in range(num_layers + 1):
        layer_name = "embeddings" if layer_idx == 0 else f"layer_{layer_idx}"
        accuracy = train_probe(layer_hiddens[layer_idx], labels_array)
        results[layer_name] = accuracy
        logger.info(f"  {layer_name}: {accuracy:.2%}")

    return results


def compare_probes(teacher_results: dict, student_results: dict):
    """
    Compare probing results between teacher and student.

    Args:
        teacher_results: Dict of layer -> accuracy for teacher
        student_results: Dict of layer -> accuracy for student

    Returns:
        Dict with comparison metrics
    """
    teacher_peak = max(teacher_results.items(), key=lambda x: x[1])
    student_peak = max(student_results.items(), key=lambda x: x[1])

    # Find where student saturates (first layer within 1% of peak)
    student_peak_acc = student_peak[1]
    saturation_layer = None
    for layer, acc in sorted(student_results.items(), key=lambda x: int(x[0].split("_")[-1]) if "_" in x[0] else 0):
        if acc >= student_peak_acc - 0.01:
            saturation_layer = layer
            break

    return {
        "teacher_peak": {"layer": teacher_peak[0], "accuracy": teacher_peak[1]},
        "student_peak": {"layer": student_peak[0], "accuracy": student_peak[1]},
        "student_saturation_layer": saturation_layer,
        "retention": student_peak[1] / teacher_peak[1] if teacher_peak[1] > 0 else 0,
    }
