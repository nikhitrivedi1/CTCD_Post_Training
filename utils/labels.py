"""Label mapping utilities for handling model-specific label orderings.

Different pre-trained models may use different internal label orderings that don't
match the GLUE standard. This module provides a registry and utilities to handle
these mappings consistently across the codebase.

See docs/mnli-model-label-mapping.md for detailed documentation.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Label Mapping Registry
# =============================================================================
# Maps model IDs to their logit reordering needed to match GLUE standard labels.
#
# Entry format: model_id -> [idx for label 0, idx for label 1, idx for label 2]
# where the indices refer to the model's internal logit positions.
#
# GLUE standard label order:
#   MNLI: [entailment=0, neutral=1, contradiction=2]
#   QQP:  [not_duplicate=0, duplicate=1]
#   SST2: [negative=0, positive=1]
#   CoLA: [unacceptable=0, acceptable=1]
#
# Example: TextAttack MNLI uses [C,E,N] internally at positions [0,1,2]
# To get GLUE order [E,N,C], we need: new[0]=old[1], new[1]=old[2], new[2]=old[0]
# So the mapping is [1, 2, 0]

LABEL_MAP_REGISTRY: Dict[str, List[int]] = {
    # TextAttack models use [contradiction, entailment, neutral] internally
    "textattack/bert-base-uncased-MNLI": [1, 2, 0],
    "textattack/distilbert-base-uncased-MNLI": [1, 2, 0],
    # Facebook BART uses [contradiction, neutral, entailment]
    "facebook/bart-large-mnli": [2, 1, 0],
}

# GLUE standard label names per task
GLUE_LABEL_ORDER: Dict[str, List[str]] = {
    "mnli": ["entailment", "neutral", "contradiction"],
    "qqp": ["not_duplicate", "duplicate"],
    "sst2": ["negative", "positive"],
    "cola": ["unacceptable", "acceptable"],
}

# Class labels for display (index -> name)
CLASS_LABELS: Dict[str, Dict[int, str]] = {
    "mnli": {0: "entailment", 1: "neutral", 2: "contradiction"},
    "qqp": {0: "not_duplicate", 1: "duplicate"},
    "sst2": {0: "negative", 1: "positive"},
    "cola": {0: "unacceptable", 1: "acceptable"},
}


def get_label_map(model_id: str, task: str, config=None) -> Optional[List[int]]:
    """
    Get label mapping for a model to convert to GLUE standard order.

    Checks in order:
    1. Model's config.id2label (if semantic labels available)
    2. Registry of known model mappings
    3. Returns None if no mapping needed

    Args:
        model_id: HuggingFace model ID or local path
        task: Task name (mnli, qqp, etc.)
        config: Optional pre-loaded model config

    Returns:
        List of indices to reorder logits, or None if no mapping needed
    """
    # First check if model has semantic id2label in config
    if config is not None and hasattr(config, "id2label"):
        id2label = config.id2label
        # Check if labels are semantic (not generic LABEL_0, LABEL_1, etc.)
        labels = list(id2label.values())
        if labels and not labels[0].startswith("LABEL_"):
            expected_order = GLUE_LABEL_ORDER.get(task)
            if expected_order:
                # Build mapping from model order to GLUE order
                try:
                    label_map = []
                    for expected_label in expected_order:
                        # Find which model index has this label
                        for idx, model_label in id2label.items():
                            if model_label.lower() == expected_label.lower():
                                label_map.append(int(idx))
                                break
                    if len(label_map) == len(expected_order):
                        # Check if mapping is identity (no reordering needed)
                        if label_map == list(range(len(label_map))):
                            return None
                        logger.info(f"Inferred label map from config.id2label: {label_map}")
                        return label_map
                except (ValueError, KeyError):
                    pass

    # Fall back to registry
    if model_id in LABEL_MAP_REGISTRY:
        label_map = LABEL_MAP_REGISTRY[model_id]
        logger.info(f"Using registry label map for {model_id}: {label_map}")
        return label_map

    return None
