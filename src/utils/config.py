"""Configuration loading with YAML inheritance support."""

from pathlib import Path
from typing import Dict, Any

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file with inheritance support.

    If the config contains an 'inherit' key, it will load the parent config
    first and then override with the current config's values.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Dict containing the merged configuration
    """
    config_path = Path(config_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Handle inheritance
    if 'inherit' in config:
        parent_path = config_path.parent / config['inherit']
        parent_config = load_config(str(parent_path))

        # Remove the inherit key and merge
        del config['inherit']
        merged_config = _deep_merge(parent_config, config)
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
