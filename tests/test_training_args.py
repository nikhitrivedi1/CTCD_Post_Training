"""Tests for training argument construction helpers in forge.py."""

import pytest
import yaml

from src.utils.config import load_config
from forge import should_load_best_model, warmup_kwargs_for_training_args


@pytest.fixture
def kd_config_tree(tmp_path):
    """Build a minimal KD config hierarchy in tmp_path.

    Structure mirrors configs/ocr/formatting/{model}/kd/{teacher}/:
        base.yaml -> model/base.yaml -> kd/base.yaml -> kd/teacher/base.yaml -> leaf.yaml
    """
    # Root base
    (tmp_path / "base.yaml").write_text(yaml.dump({
        "data": {"task": "ocr", "dataset": "formatting"},
        "training": {
            "num_epochs": 3,
            "use_lora": True,
            "lora": {"r": 16, "alpha": 16},
        },
        "evaluation": {"eval_strategy": "steps", "eval_steps": 50},
        "checkpointing": {"save_steps": 50},
    }))

    # Model base (student)
    model_dir = tmp_path / "student"
    model_dir.mkdir()
    (model_dir / "base.yaml").write_text(yaml.dump({
        "inherit": "../base.yaml",
        "model": {"name": "Qwen/Qwen3.5-0.8B"},
        "training": {"batch_size": 1, "learning_rate": "2e-6"},
        "evaluation": {"eval_strategy": "no"},
        "checkpointing": {"save_steps": 999999, "final_model_save": "last"},
    }))

    # KD base
    kd_dir = model_dir / "kd"
    kd_dir.mkdir()
    (kd_dir / "base.yaml").write_text(yaml.dump({
        "inherit": "../base.yaml",
    }))

    # Teacher base
    teacher_dir = kd_dir / "teacher"
    teacher_dir.mkdir()
    (teacher_dir / "base.yaml").write_text(yaml.dump({
        "inherit": "../base.yaml",
        "model": {"teacher": {"name": "Qwen/Qwen3.5-0.8B", "freeze": True}},
    }))

    # Leaf config (response_kl)
    (teacher_dir / "01_response_kl.yaml").write_text(yaml.dump({
        "inherit": "base.yaml",
        "distillation": {
            "loss_type": "response_kl",
            "temperature": 2.0,
            "alpha_h": 0.5,
            "alpha_s": 0.5,
        },
    }))

    return teacher_dir / "01_response_kl.yaml"


class TestWarmupKwargsForTrainingArgs:
    def test_warmup_ratio_takes_precedence(self):
        kw = warmup_kwargs_for_training_args(
            {"warmup_ratio": 0.05, "warmup_steps": 999},
        )
        assert kw == {"warmup_ratio": 0.05}

    def test_warmup_steps_when_no_ratio(self):
        kw = warmup_kwargs_for_training_args({"warmup_steps": 100})
        assert kw == {"warmup_steps": 100}

    def test_default_steps_when_neither(self):
        assert warmup_kwargs_for_training_args({}) == {"warmup_steps": 0}
        assert warmup_kwargs_for_training_args({}, default_warmup_steps=200) == {
            "warmup_steps": 200,
        }

    def test_null_ratio_falls_back_to_steps(self):
        kw = warmup_kwargs_for_training_args({"warmup_ratio": None, "warmup_steps": 50})
        assert kw == {"warmup_steps": 50}


class TestShouldLoadBestModel:
    """load_best_model_at_end reflects checkpointing.final_model_save (default 'best'),
    gated on eval actually running when 'best' is requested."""

    def test_steps_strategy(self):
        config = {"evaluation": {"eval_strategy": "steps"}}
        assert should_load_best_model(config) is True

    def test_epoch_strategy(self):
        config = {"evaluation": {"eval_strategy": "epoch"}}
        assert should_load_best_model(config) is True

    def test_no_strategy_raises(self):
        """Default final_model_save='best' with eval disabled has no metric to select from."""
        config = {"evaluation": {"eval_strategy": "no"}}
        with pytest.raises(ValueError):
            should_load_best_model(config)

    def test_none_strategy_raises(self):
        config = {"evaluation": {"eval_strategy": None}}
        with pytest.raises(ValueError):
            should_load_best_model(config)

    def test_missing_eval_strategy(self):
        """No eval_strategy key - defaults to steps (existing behavior)."""
        config = {"evaluation": {}}
        assert should_load_best_model(config) is True

    def test_missing_evaluation_section(self):
        """No evaluation section at all - defaults to steps."""
        config = {}
        assert should_load_best_model(config) is True

    def test_eval_no_from_config_file(self, kd_config_tree):
        """Config with eval_strategy: no and final_model_save: last should disable load_best_model."""
        config = load_config(str(kd_config_tree))
        assert should_load_best_model(config) is False

    def test_final_model_save_last_with_eval_running(self):
        """final_model_save: last always saves the last checkpoint, even with eval on."""
        config = {"evaluation": {"eval_strategy": "steps"}, "checkpointing": {"final_model_save": "last"}}
        assert should_load_best_model(config) is False

    def test_final_model_save_best_explicit(self):
        config = {"evaluation": {"eval_strategy": "steps"}, "checkpointing": {"final_model_save": "best"}}
        assert should_load_best_model(config) is True

    def test_final_model_save_best_with_eval_disabled_raises(self):
        """Can't select a 'best' checkpoint with no eval metrics to compare."""
        config = {"evaluation": {"eval_strategy": "no"}, "checkpointing": {"final_model_save": "best"}}
        with pytest.raises(ValueError, match="final_model_save"):
            should_load_best_model(config)

    def test_final_model_save_invalid_value_raises(self):
        config = {"checkpointing": {"final_model_save": "bogus"}}
        with pytest.raises(ValueError, match="final_model_save"):
            should_load_best_model(config)


class TestKDConfigResolution:
    """Verify KD config chain resolves correctly."""

    def test_student_model(self, kd_config_tree):
        config = load_config(str(kd_config_tree))
        assert config["model"]["name"] == "Qwen/Qwen3.5-0.8B"

    def test_teacher_model(self, kd_config_tree):
        config = load_config(str(kd_config_tree))
        assert config["model"]["teacher"]["name"] == "Qwen/Qwen3.5-0.8B"
        assert config["model"]["teacher"]["freeze"] is True

    def test_distillation_params(self, kd_config_tree):
        config = load_config(str(kd_config_tree))
        assert config["distillation"]["loss_type"] == "response_kl"
        assert config["distillation"]["temperature"] == 2.0
        assert config["distillation"]["alpha_h"] == 0.5
        assert config["distillation"]["alpha_s"] == 0.5

    def test_inherits_formatting_dataset(self, kd_config_tree):
        config = load_config(str(kd_config_tree))
        assert config["data"]["dataset"] == "formatting"

    def test_inherits_lora_settings(self, kd_config_tree):
        config = load_config(str(kd_config_tree))
        assert config["training"]["use_lora"] is True
        assert config["training"]["lora"]["r"] == 16
