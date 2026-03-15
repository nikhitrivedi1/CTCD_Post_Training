"""Tests for src/training/layer_mapping.py

Property-based tests parametrized across all 4 strategies, plus
strategy-specific invariants and factory tests.

Run:
    python -m pytest tests/unit/test_layer_mapping.py -v
"""

import pytest

from src.training.layer_mapping import (
    CustomMappingStrategy,
    FirstLastMappingStrategy,
    LastNMappingStrategy,
    UniformMappingStrategy,
    get_layer_mapping,
    LAYER_MAPPING_REGISTRY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=[
    UniformMappingStrategy,
    LastNMappingStrategy,
    FirstLastMappingStrategy,
], ids=["uniform", "last_n", "first_last"])
def strategy(request):
    """Non-custom strategies (no constructor args needed)."""
    return request.param()


LAYER_PAIRS = [(12, 6), (24, 8), (6, 3), (6, 6), (10, 5)]


# ---------------------------------------------------------------------------
# Cross-strategy property tests
# ---------------------------------------------------------------------------

class TestStrategyProperties:
    """Properties that hold for all non-custom strategies."""

    @pytest.mark.parametrize("t,s", LAYER_PAIRS)
    def test_output_length_equals_model_layers(self, strategy, t, s):
        assert len(strategy.map(t, s)) == s

    @pytest.mark.parametrize("t,s", LAYER_PAIRS)
    def test_teacher_indices_in_bounds(self, strategy, t, s):
        for t_idx, _ in strategy.map(t, s):
            assert 0 <= t_idx < t

    @pytest.mark.parametrize("t,s", LAYER_PAIRS)
    def test_model_indices_in_bounds(self, strategy, t, s):
        for _, s_idx in strategy.map(t, s):
            assert 0 <= s_idx < s

    @pytest.mark.parametrize("t,s", LAYER_PAIRS)
    def test_model_indices_are_sequential(self, strategy, t, s):
        model_indices = [s_idx for _, s_idx in strategy.map(t, s)]
        assert model_indices == list(range(s))

    @pytest.mark.parametrize("t,s", [(12, 6), (24, 8), (6, 6)])
    def test_validate_true_for_valid(self, strategy, t, s):
        assert strategy.validate(t, s) is True

    @pytest.mark.parametrize("t,s", [(3, 6), (2, 8), (1, 5)])
    def test_validate_false_when_teacher_lt_model(self, strategy, t, s):
        assert strategy.validate(t, s) is False

    def test_validate_false_when_model_zero(self, strategy):
        assert strategy.validate(12, 0) is False

    def test_zero_model_layers_returns_empty(self, strategy):
        assert strategy.map(12, 0) == []


# ---------------------------------------------------------------------------
# Strategy-specific tests
# ---------------------------------------------------------------------------

class TestUniform:
    def test_equal_layers_is_identity(self):
        mapping = UniformMappingStrategy().map(6, 6)
        assert all(t == s for t, s in mapping)

    def test_teacher_indices_evenly_spaced(self):
        mapping = UniformMappingStrategy().map(12, 6)
        t_indices = [t for t, _ in mapping]
        diffs = [t_indices[i+1] - t_indices[i] for i in range(len(t_indices) - 1)]
        assert len(set(diffs)) == 1, "Teacher indices should be evenly spaced"


class TestLastN:
    @pytest.mark.parametrize("t,s", [(12, 6), (24, 8), (10, 5)])
    def test_uses_final_teacher_layers(self, t, s):
        mapping = LastNMappingStrategy().map(t, s)
        t_indices = [t_idx for t_idx, _ in mapping]
        assert t_indices == list(range(t - s, t))

    def test_equal_layers_is_identity(self):
        mapping = LastNMappingStrategy().map(6, 6)
        assert all(t == s for t, s in mapping)


class TestFirstLast:
    @pytest.mark.parametrize("t,s", [(12, 6), (24, 8), (10, 5)])
    def test_first_pair_is_zero_zero(self, t, s):
        mapping = FirstLastMappingStrategy().map(t, s)
        assert mapping[0] == (0, 0)

    @pytest.mark.parametrize("t,s", [(12, 6), (24, 8), (10, 5)])
    def test_last_pair_is_final_layers(self, t, s):
        mapping = FirstLastMappingStrategy().map(t, s)
        assert mapping[-1] == (t - 1, s - 1)

    @pytest.mark.parametrize("t,s", [(12, 6), (24, 8), (10, 5)])
    def test_teacher_indices_monotonically_increasing(self, t, s):
        t_indices = [t_idx for t_idx, _ in FirstLastMappingStrategy().map(t, s)]
        assert t_indices == sorted(t_indices)
        assert len(set(t_indices)) == len(t_indices), "No duplicates"

    def test_single_model_maps_to_last_teacher(self):
        assert FirstLastMappingStrategy().map(12, 1) == [(11, 0)]

    def test_two_models_maps_first_and_last(self):
        assert FirstLastMappingStrategy().map(12, 2) == [(0, 0), (11, 1)]


class TestCustom:
    def test_preserves_user_indices_exactly(self):
        t_idx, s_idx = [2, 5, 11], [0, 1, 2]
        strategy = CustomMappingStrategy(teacher_indices=t_idx, model_indices=s_idx)
        assert strategy.map(12, 6) == list(zip(t_idx, s_idx))

    def test_map_ignores_layer_count_args(self):
        strategy = CustomMappingStrategy(teacher_indices=[0, 2], model_indices=[0, 1])
        assert strategy.map(4, 2) == strategy.map(100, 50)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            CustomMappingStrategy(teacher_indices=[1, 2, 3], model_indices=[0, 1])

    def test_validate_true_when_in_bounds(self):
        strategy = CustomMappingStrategy(teacher_indices=[0, 5, 11], model_indices=[0, 2, 5])
        assert strategy.validate(12, 6) is True

    def test_validate_false_teacher_out_of_bounds(self):
        strategy = CustomMappingStrategy(teacher_indices=[15], model_indices=[0])
        assert strategy.validate(12, 6) is False

    def test_validate_false_model_out_of_bounds(self):
        strategy = CustomMappingStrategy(teacher_indices=[0], model_indices=[10])
        assert strategy.validate(12, 6) is False

    def test_validate_false_negative_index(self):
        strategy = CustomMappingStrategy(teacher_indices=[-1], model_indices=[0])
        assert strategy.validate(6, 3) is False


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

class TestGetLayerMapping:
    @pytest.mark.parametrize("name,cls", [
        ("uniform", UniformMappingStrategy),
        ("last_n", LastNMappingStrategy),
        ("first_last", FirstLastMappingStrategy),
    ])
    def test_returns_correct_type(self, name, cls):
        assert isinstance(get_layer_mapping(name), cls)

    def test_custom_returns_correct_type(self):
        strategy = get_layer_mapping("custom", teacher_indices=[0], model_indices=[0])
        assert isinstance(strategy, CustomMappingStrategy)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown layer mapping"):
            get_layer_mapping("diagonal")

    def test_registry_has_four_strategies(self):
        assert set(LAYER_MAPPING_REGISTRY.keys()) == {"uniform", "last_n", "first_last", "custom"}
