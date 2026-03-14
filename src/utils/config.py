"""Configuration loading with YAML inheritance support."""

from pathlib import Path
from typing import Dict, Any

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
