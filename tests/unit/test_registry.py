"""Tests for src/tasks/registry.py

Covers TASK_REGISTRY structure, type predicates mutual exclusivity,
and factory function behavior for known/unknown tasks.

Run:
    python -m pytest tests/unit/test_registry.py -v
"""

import pytest

from src.tasks.registry import (
    TASK_REGISTRY,
    get_default_reward,
    get_num_labels,
    get_primary_metric,
    get_task_type,
    is_classification_task,
    is_ocr_task,
    is_translation_task,
)

ALL_TASKS = sorted(TASK_REGISTRY.keys())
REQUIRED_KEYS = {"num_labels", "type", "primary_metric"}


# ===========================================================================
# 1. Registry structure
# ===========================================================================


class TestRegistryStructure:
    """Every task has required keys and consistent types."""

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_required_keys_present(self, task):
        missing = REQUIRED_KEYS - set(TASK_REGISTRY[task].keys())
        assert not missing, f"'{task}' missing: {missing}"

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_type_is_known_string(self, task):
        assert TASK_REGISTRY[task]["type"] in {"classification", "translation", "ocr"}

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_primary_metric_is_string(self, task):
        assert isinstance(TASK_REGISTRY[task]["primary_metric"], str)


# ===========================================================================
# 2. Type predicates are mutually exclusive
# ===========================================================================


class TestTypePredicates:
    """is_classification/translation/ocr are mutually exclusive for every task."""

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_exactly_one_type_predicate_true(self, task):
        flags = [
            is_classification_task(task),
            is_translation_task(task),
            is_ocr_task(task),
        ]
        assert sum(flags) == 1, f"'{task}' matched {sum(flags)} predicates"

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_predicate_matches_registry_type(self, task):
        task_type = TASK_REGISTRY[task]["type"]
        predicate_map = {
            "classification": is_classification_task,
            "translation": is_translation_task,
            "ocr": is_ocr_task,
        }
        assert predicate_map[task_type](task) is True

    def test_unknown_task_all_false(self):
        assert is_classification_task("nonexistent") is False
        assert is_translation_task("nonexistent") is False
        assert is_ocr_task("nonexistent") is False


# ===========================================================================
# 3. Factory functions
# ===========================================================================


class TestFactoryFunctions:
    """Factory functions return correct values for known tasks, None for unknown."""

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_get_num_labels_matches_registry(self, task):
        assert get_num_labels(task) == TASK_REGISTRY[task]["num_labels"]

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_get_task_type_matches_registry(self, task):
        assert get_task_type(task) == TASK_REGISTRY[task]["type"]

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_get_primary_metric_matches_registry(self, task):
        assert get_primary_metric(task) == TASK_REGISTRY[task]["primary_metric"]

    @pytest.mark.parametrize(
        "func",
        [
            get_num_labels,
            get_task_type,
            get_primary_metric,
            get_default_reward,
        ],
    )
    def test_unknown_task_returns_none(self, func):
        assert func("nonexistent_task") is None

    def test_get_default_reward_known_task(self):
        """Exercise get_default_reward for a task that has a default_reward."""
        # europarl has default_reward: "bleu"
        result = get_default_reward("europarl")
        assert result == "bleu"

    def test_get_default_reward_no_reward(self):
        """Classification tasks without default_reward return None."""
        result = get_default_reward("mnli")
        assert result is None
