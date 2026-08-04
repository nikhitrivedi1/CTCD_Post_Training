"""Tests for src/utils/config.py

Covers _deep_merge() and load_config() with YAML inheritance,
circular reference guard, and missing file handling.

Run:
    python -m pytest tests/unit/test_config.py -v
"""

import pytest

from src.utils.config import MAX_INHERITANCE_DEPTH, _deep_merge, load_config

# ===========================================================================
# 1. TestDeepMerge
# ===========================================================================


class TestDeepMerge:
    """Unit tests for _deep_merge() - no filesystem needed."""

    @pytest.mark.parametrize(
        "base,override,expected",
        [
            ({"a": 1, "b": 2}, {"b": 3}, {"a": 1, "b": 3}),
            ({"a": 1}, {"b": 2}, {"a": 1, "b": 2}),
            ({}, {"a": 1}, {"a": 1}),
            ({"a": 1}, {}, {"a": 1}),
            ({}, {}, {}),
            ({"key": "value"}, {"key": None}, {"key": None}),
            ({"x": [1, 2, 3]}, {"x": [4, 5]}, {"x": [4, 5]}),
        ],
        ids=[
            "scalar-override",
            "new-key-added",
            "empty-base",
            "empty-override",
            "both-empty",
            "none-overrides-value",
            "list-replaced-not-merged",
        ],
    )
    def test_flat_merge(self, base, override, expected):
        assert _deep_merge(base, override) == expected

    def test_nested_dict_merges_recursively(self):
        """Deep merge preserves sibling keys in nested dicts."""
        base = {"training": {"lr": 1e-4, "epochs": 10}}
        override = {"training": {"lr": 2e-5}}
        result = _deep_merge(base, override)
        assert result["training"]["lr"] == 2e-5
        assert result["training"]["epochs"] == 10

    def test_three_levels_deep(self):
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 99}}}
        result = _deep_merge(base, override)
        assert result["a"]["b"] == {"c": 99, "d": 2}

    @pytest.mark.parametrize(
        "base,override",
        [
            ({"key": 42}, {"key": {"nested": 1}}),
            ({"key": {"nested": 1}}, {"key": 42}),
        ],
        ids=["scalar-to-dict", "dict-to-scalar"],
    )
    def test_type_mismatch_override_wins(self, base, override):
        result = _deep_merge(base, override)
        assert result["key"] == override["key"]

    def test_inputs_not_mutated(self):
        base = {"a": 1, "nested": {"x": 10}}
        override = {"nested": {"x": 99}}
        base_copy, override_copy = dict(base), dict(override)
        _deep_merge(base, override)
        assert base == base_copy
        assert override == override_copy


# ===========================================================================
# 2. TestLoadConfig
# ===========================================================================


class TestLoadConfig:
    """Tests for load_config() - uses tmp_path for temp YAML files."""

    def test_simple_yaml_load(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("lr: 0.001\nepochs: 10\n")
        assert load_config(str(cfg)) == {"lr": 0.001, "epochs": 10}

    def test_child_inherits_parent(self, tmp_path):
        """Child values override parent, parent values preserved."""
        (tmp_path / "base.yaml").write_text("lr: 0.001\nepochs: 10\nbatch_size: 16\n")
        (tmp_path / "child.yaml").write_text("inherit: base.yaml\nlr: 0.0001\n")
        result = load_config(str(tmp_path / "child.yaml"))
        assert result["lr"] == 0.0001
        assert result["epochs"] == 10
        assert result["batch_size"] == 16
        assert "inherit" not in result

    def test_deep_merge_through_inheritance(self, tmp_path):
        """Nested dicts merge recursively, scalars overwrite."""
        (tmp_path / "base.yaml").write_text("training:\n  lr: 0.001\n  epochs: 10\n")
        (tmp_path / "child.yaml").write_text("inherit: base.yaml\ntraining:\n  lr: 0.0001\n")
        result = load_config(str(tmp_path / "child.yaml"))
        assert result["training"]["lr"] == 0.0001
        assert result["training"]["epochs"] == 10

    def test_multiple_inheritance_left_to_right(self, tmp_path):
        """inherit: [a.yaml, b.yaml] merges a first, then b, then child."""
        (tmp_path / "a.yaml").write_text("x: 1\ny: 2\n")
        (tmp_path / "b.yaml").write_text("y: 20\nz: 30\n")
        (tmp_path / "child.yaml").write_text("inherit:\n  - a.yaml\n  - b.yaml\nx: 100\n")
        result = load_config(str(tmp_path / "child.yaml"))
        assert result["x"] == 100  # child overrides a
        assert result["y"] == 20  # b overrides a
        assert result["z"] == 30  # from b

    def test_circular_inheritance_raises(self, tmp_path):
        """Max depth guard prevents infinite recursion."""
        prev = "level_0.yaml"
        (tmp_path / prev).write_text("base: true\n")
        for i in range(1, MAX_INHERITANCE_DEPTH + 2):
            name = f"level_{i}.yaml"
            (tmp_path / name).write_text(f"inherit: level_{i - 1}.yaml\nlevel: {i}\n")
            prev = name
        with pytest.raises(RecursionError):
            load_config(str(tmp_path / prev))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "nonexistent.yaml"))

    def test_relative_path_resolution(self, tmp_path):
        """Inherit path resolved relative to the child config's directory."""
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (tmp_path / "base.yaml").write_text("lr: 0.001\n")
        (subdir / "child.yaml").write_text("inherit: ../base.yaml\nepochs: 5\n")
        result = load_config(str(subdir / "child.yaml"))
        assert result == {"lr": 0.001, "epochs": 5}

    def test_max_inheritance_depth_is_ten(self):
        assert MAX_INHERITANCE_DEPTH == 10
