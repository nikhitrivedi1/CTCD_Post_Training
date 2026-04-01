"""Configuration loading with YAML inheritance support."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Maximum inheritance depth to prevent circular references
MAX_INHERITANCE_DEPTH = 10


def load_config(config_path: str, _depth: int = 0) -> Dict[str, Any]:
    """
    Load a YAML configuration file with inheritance support.

    If the config contains an 'inherit' key, it will load the parent config
    first and then override with the current config's values.

    Args:
        config_path: Path to the YAML config file
        _depth: Internal counter to prevent circular inheritance (do not set manually)

    Returns:
        Dict containing the merged configuration

    Raises:
        RecursionError: If inheritance depth exceeds MAX_INHERITANCE_DEPTH
    """
    if _depth > MAX_INHERITANCE_DEPTH:
        raise RecursionError(
            f"Config inheritance depth exceeded {MAX_INHERITANCE_DEPTH}. "
            f"Check for circular inheritance in: {config_path}"
        )

    config_path = Path(config_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}

    # Handle inheritance (string or list of paths)
    if 'inherit' in config:
        parents = config['inherit']
        if isinstance(parents, str):
            parents = [parents]

        # Merge parents left-to-right, then overlay current config
        merged = {}
        for parent_ref in parents:
            parent_path = config_path.parent / parent_ref
            parent_config = load_config(str(parent_path), _depth=_depth + 1)
            merged = _deep_merge(merged, parent_config)

        del config['inherit']
        merged_config = _deep_merge(merged, config)
        return merged_config

    return config


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base dictionary
        override: Dictionary with values to override

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def resolve_distillation_coefficients(
    distillation: Optional[Dict[str, Any]] = None,
    *,
    freeze_teacher: bool = True,
) -> Dict[str, float]:
    """Resolve student (alpha_h, alpha_s) and teacher (beta_h, beta_s) distillation weights.

    Student:
        - ``alpha_h`` / ``alpha_s``: hard (CE) and soft (KD) weights; if one is missing, set it to
          ``1 - other``. Default 1.0 / 1.0 when neither key is present.

    Teacher (co-training, ``freeze_teacher=False``):
        - Default ``beta_h = 1``, ``beta_s = 0`` (CE-only teacher) when no beta keys.
        - If either ``beta_h`` or ``beta_s`` is set, infer the other as ``1 - other`` when missing.

    Returns:
        Dict with keys ``alpha_h``, ``alpha_s``, ``beta_h``, ``beta_s``, ``student_soft_weight``
        (``alpha_s``, for teacher loading / eval gating).
    """
    d = distillation or {}

    if "alpha_h" in d or "alpha_s" in d:
        alpha_s = float(d["alpha_s"]) if "alpha_s" in d else 1.0 - float(d["alpha_h"])
        alpha_h = float(d["alpha_h"]) if "alpha_h" in d else 1.0 - float(d["alpha_s"])
    else:
        alpha_h, alpha_s = 1.0, 1.0

    if freeze_teacher:
        beta_h, beta_s = 1.0, 0.0
    elif "beta_h" in d or "beta_s" in d:
        beta_s = float(d["beta_s"]) if "beta_s" in d else 1.0 - float(d["beta_h"])
        beta_h = float(d["beta_h"]) if "beta_h" in d else 1.0 - float(d["beta_s"])
    else:
        beta_h, beta_s = 1.0, 0.0

    return {
        "alpha_h": alpha_h,
        "alpha_s": alpha_s,
        "beta_h": beta_h,
        "beta_s": beta_s,
        "student_soft_weight": alpha_s,
    }
