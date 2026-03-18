"""Model-family profiles for inference defaults.

Maps (model family, task) pairs to default chat_template_kwargs so callers
don't need to know per-family quirks. Explicit kwargs always override
profile defaults.

Lookup order:
    1. (family, task) - task-specific profile
    2. (family, "*")  - family-wide default
    3. {} - no profile

Usage:
    from src.inference.model_profiles import get_chat_template_kwargs

    # Family-wide default (thinking enabled)
    kwargs = get_chat_template_kwargs("Qwen/Qwen3.5-9B")
    # -> {"enable_thinking": True}

    # Task-specific override (OCR disables thinking)
    kwargs = get_chat_template_kwargs("Qwen/Qwen3.5-9B", task="ocr")
    # -> {"enable_thinking": False}

    # Explicit overrides win over everything
    kwargs = get_chat_template_kwargs("Qwen/Qwen3.5-9B", explicit={"enable_thinking": True})
    # -> {"enable_thinking": True}
"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# (model_family, task) -> default chat_template_kwargs.
# Use "*" as task wildcard for family-wide defaults.
# Lookup: (family, task) first, then (family, "*").
MODEL_PROFILES = {
    # Qwen3.5 (native multimodal text) supports enable_thinking in chat template.
    # Thinking enabled by default for reasoning tasks, disabled for OCR (clean output).
    ("Qwen3.5", "*"): {"enable_thinking": True},
    ("Qwen3.5", "ocr"): {"enable_thinking": False},
    # Qwen3-VL chat template does NOT support enable_thinking.
    # No profile entries needed - empty dict returned by default.
}


def _resolve_model_family(name_or_path):
    """Resolve model family from a model ID or checkpoint path.

    Resolution order for checkpoint paths:
    1. model.family from config.yaml (explicit, preferred)
    2. model.name from config.yaml (substring match, backward compat)

    For model IDs (not a path), returns the ID for substring matching.

    Args:
        name_or_path: HF model ID or checkpoint path.

    Returns:
        (family, exact) tuple. family is the string to match, exact is True
        if it came from model.family (use exact match) or False if from
        model.name / model ID (use substring match). Returns (None, False)
        if unresolvable.
    """
    p = Path(name_or_path)
    config_path = p / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            model_cfg = cfg.get("model", {})
            family = model_cfg.get("family", "")
            if family:
                logger.debug(f"Model family from config: {family}")
                return family, True
            name = model_cfg.get("name", "")
            if name:
                logger.debug(f"Model name from config: {name}")
                return name, False
        except Exception:
            pass
        return None, False

    # Not a path, treat as model ID directly
    return name_or_path, False


def _match_profile(resolved, exact, task=None):
    """Find the best matching profile for a resolved model family.

    Args:
        resolved: Model family string (from config or model ID).
        exact: If True, match family exactly. If False, substring match.
        task: Optional task name for task-specific lookup.

    Returns:
        Dict of chat_template_kwargs, or empty dict.
    """
    if not resolved:
        return {}

    def _find(target_task):
        for (pattern, profile_task), defaults in MODEL_PROFILES.items():
            if profile_task != target_task:
                continue
            family_match = (resolved == pattern) if exact else (pattern in resolved)
            if family_match:
                return dict(defaults)
        return None

    # Task-specific first, then wildcard
    if task:
        result = _find(task)
        if result is not None:
            return result
    return _find("*") or {}


def get_chat_template_kwargs(model_name, explicit=None, task=None):
    """Resolve chat_template_kwargs for a model.

    Merges profile defaults with explicit overrides (explicit wins).
    For checkpoint paths, reads config.yaml to get model.family (exact
    match) or model.name (substring match, backward compat).

    Args:
        model_name: Model name, HF ID, or checkpoint path.
        explicit: Caller-provided kwargs (from CLI flag, config, etc.).
        task: Task name for task-specific profile lookup.

    Returns:
        Merged dict of chat_template_kwargs, or empty dict.
    """
    resolved, exact = _resolve_model_family(model_name)
    profile = _match_profile(resolved, exact, task)

    if explicit:
        profile.update(explicit)

    return profile
