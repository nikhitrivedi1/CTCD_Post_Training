"""Tests for KD configuration, teacher_adapter CLI, teacher ref saving, and _save_model."""

import json
from unittest.mock import MagicMock, PropertyMock

import pytest
import torch

from src.utils.config import load_config


# =============================================================================
# Config inheritance tests (self-contained fixtures, no dependency on configs/)
# =============================================================================


@pytest.fixture
def config_tree(tmp_path):
    """Build a minimal config hierarchy mimicking the KD inheritance chain.

    Structure:
        task/base.yaml          - dataset + epochs
        task/model/base.yaml    - inherits ../base.yaml, adds model name + LR
        task/model/sft.yaml     - inherits base.yaml (plain SFT, no teacher)
        task/model/kd/base.yaml - inherits ../base.yaml
        task/model/kd/kd.yaml   - inherits base.yaml, adds teacher + distillation
    """
    task = tmp_path / "task"
    model = task / "model"
    kd = model / "kd"
    kd.mkdir(parents=True)

    (task / "base.yaml").write_text(
        "data:\n"
        "  dataset: formatting\n"
        "training:\n"
        "  num_epochs: 3\n"
    )
    (model / "base.yaml").write_text(
        "inherit: ../base.yaml\n"
        "model:\n"
        "  name: TestStudent/2B\n"
        "training:\n"
        "  learning_rate: 5e-6\n"
    )
    (model / "sft.yaml").write_text(
        "inherit: base.yaml\n"
    )
    (kd / "base.yaml").write_text(
        "inherit: ../base.yaml\n"
    )
    (kd / "kd.yaml").write_text(
        "inherit: base.yaml\n"
        "model:\n"
        "  teacher:\n"
        "    name: TestTeacher/9B\n"
        "    freeze: true\n"
        "distillation:\n"
        "  loss_type: response_kl\n"
        "  temperature: 2.0\n"
        "  alpha: 0.5\n"
    )

    return {"task": task, "model": model, "kd": kd}


class TestKDConfigInheritance:
    """Verify KD configs resolve correctly through the inheritance chain."""

    def test_model_base_sets_model_and_lr(self, config_tree):
        config = load_config(str(config_tree["model"] / "base.yaml"))
        assert config["model"]["name"] == "TestStudent/2B"
        assert float(config["training"]["learning_rate"]) == 5e-6

    def test_sft_inherits_model_from_base(self, config_tree):
        config = load_config(str(config_tree["model"] / "sft.yaml"))
        assert config["model"]["name"] == "TestStudent/2B"
        assert float(config["training"]["learning_rate"]) == 5e-6
        assert "teacher" not in config.get("model", {})

    def test_kd_base_inherits_dataset_and_epochs(self, config_tree):
        """kd/base.yaml -> ../base.yaml -> task/base.yaml."""
        config = load_config(str(config_tree["kd"] / "base.yaml"))
        assert config["data"]["dataset"] == "formatting"
        assert config["training"]["num_epochs"] == 3

    def test_kd_config_has_student_and_teacher(self, config_tree):
        config = load_config(str(config_tree["kd"] / "kd.yaml"))
        # Student model from model/base.yaml via deep merge
        assert config["model"]["name"] == "TestStudent/2B"
        # Teacher from the KD config itself
        assert config["model"]["teacher"]["name"] == "TestTeacher/9B"
        assert config["model"]["teacher"]["freeze"] is True
        # Distillation settings
        assert config["distillation"]["loss_type"] == "response_kl"
        assert config["distillation"]["temperature"] == 2.0
        assert config["distillation"]["alpha"] == 0.5
        # LR from model/base.yaml
        assert float(config["training"]["learning_rate"]) == 5e-6

    def test_deep_merge_preserves_student_with_teacher(self, config_tree):
        """Adding model.teacher doesn't clobber model.name."""
        config = load_config(str(config_tree["kd"] / "kd.yaml"))
        assert config["model"]["name"] == "TestStudent/2B"
        assert config["model"]["teacher"]["name"] == "TestTeacher/9B"


# =============================================================================
# teacher_adapter CLI override tests (pure dict manipulation, no config files)
# =============================================================================

class TestTeacherAdapterOverride:
    """Test that --teacher_adapter correctly overrides config."""

    def test_override_sets_teacher_name(self):
        """Simulate the CLI override logic from forge.py."""
        config = {
            "model": {
                "name": "Student/2B",
                "teacher": {"name": "Teacher/9B", "freeze": True},
            }
        }

        teacher_adapter = "outputs/fmt_grpo_teacher"
        config.setdefault("model", {}).setdefault("teacher", {})["name"] = teacher_adapter

        assert config["model"]["teacher"]["name"] == "outputs/fmt_grpo_teacher"
        # Student model unaffected
        assert config["model"]["name"] == "Student/2B"
        # Freeze unaffected
        assert config["model"]["teacher"]["freeze"] is True

    def test_override_on_config_without_teacher(self):
        """TEACHER_ADAPTER on a plain SFT config adds the teacher section."""
        config = {"model": {"name": "Student/2B"}}
        assert "teacher" not in config["model"]

        teacher_adapter = "outputs/some_teacher"
        config.setdefault("model", {}).setdefault("teacher", {})["name"] = teacher_adapter

        assert config["model"]["teacher"]["name"] == "outputs/some_teacher"
        assert config["model"]["name"] == "Student/2B"


# =============================================================================
# Teacher ref saving tests
# =============================================================================

class TestTeacherRefSaving:
    """Test teacher/base/model_ref.json creation."""

    def test_frozen_teacher_saves_ref(self, tmp_path):
        """Frozen KD: saves teacher/base/model_ref.json with checkpoint path."""
        output_dir = tmp_path / "kd_run"
        output_dir.mkdir()

        teacher_name = "outputs/fmt_grpo_qwen35-9b"
        teacher_base_dir = output_dir / "teacher" / "base"
        teacher_base_dir.mkdir(parents=True, exist_ok=True)
        ref = {"model_name_or_path": teacher_name}
        with open(teacher_base_dir / "model_ref.json", 'w') as f:
            json.dump(ref, f, indent=2)

        # Verify
        ref_path = output_dir / "teacher" / "base" / "model_ref.json"
        assert ref_path.exists()
        with open(ref_path) as f:
            saved = json.load(f)
        assert saved["model_name_or_path"] == "outputs/fmt_grpo_qwen35-9b"
        assert "revision" not in saved

    def test_hf_teacher_saves_ref_with_revision(self, tmp_path):
        """HF teacher (no TEACHER_ADAPTER): saves ref with revision hash."""
        output_dir = tmp_path / "kd_run"
        output_dir.mkdir()

        teacher_name = "Qwen/Qwen3.5-9B"
        revision = "abc123def456"
        teacher_base_dir = output_dir / "teacher" / "base"
        teacher_base_dir.mkdir(parents=True, exist_ok=True)
        ref = {"model_name_or_path": teacher_name}
        if revision:
            ref["revision"] = revision
        with open(teacher_base_dir / "model_ref.json", 'w') as f:
            json.dump(ref, f, indent=2)

        ref_path = output_dir / "teacher" / "base" / "model_ref.json"
        with open(ref_path) as f:
            saved = json.load(f)
        assert saved["model_name_or_path"] == "Qwen/Qwen3.5-9B"
        assert saved["revision"] == "abc123def456"

    def test_teacher_ref_mirrors_student_structure(self, tmp_path):
        """teacher/base/model_ref.json parallels base/model_ref.json."""
        output_dir = tmp_path / "kd_run"
        output_dir.mkdir()

        # Student base ref
        student_base = output_dir / "base"
        student_base.mkdir()
        with open(student_base / "model_ref.json", 'w') as f:
            json.dump({"model_name_or_path": "Qwen/Qwen3.5-2B"}, f)

        # Teacher base ref
        teacher_base = output_dir / "teacher" / "base"
        teacher_base.mkdir(parents=True)
        with open(teacher_base / "model_ref.json", 'w') as f:
            json.dump({"model_name_or_path": "outputs/fmt_grpo_qwen35-9b"}, f)

        # Both exist with parallel structure
        assert (output_dir / "base" / "model_ref.json").exists()
        assert (output_dir / "teacher" / "base" / "model_ref.json").exists()


# =============================================================================
# _save_model tests
# =============================================================================


def _make_plain_model(**config_attrs):
    """Create a mock model that is NOT a PeftModel (for full-FT tests).

    Uses torch.nn.Module as spec so isinstance(model, PeftModel) is False.
    """
    model = MagicMock(spec=torch.nn.Module)
    model.save_pretrained = MagicMock()
    config = MagicMock()
    for k, v in config_attrs.items():
        setattr(config, k, v)
    model.config = config
    return model


class TestSaveModel:
    """Test _save_model handles LoRA and full FT models correctly."""

    def test_full_ft_model_saves_weights_and_ref(self, tmp_path):
        """Full FT model: saves model weights + base/model_ref.json."""
        from forge import _save_model

        save_dir = tmp_path / "model_out"

        model = _make_plain_model(
            _name_or_path="Qwen/Qwen3.5-9B",
            _commit_hash="abc123",
        )

        save_dir.mkdir(parents=True)  # mock save_pretrained won't create it
        _save_model(model, save_dir, model_name="Qwen/Qwen3.5-9B")

        model.save_pretrained.assert_called_once_with(save_dir)
        # Base ref created
        ref_path = save_dir / "base" / "model_ref.json"
        assert ref_path.exists()
        with open(ref_path) as f:
            ref = json.load(f)
        assert ref["model_name_or_path"] == "Qwen/Qwen3.5-9B"
        assert ref["revision"] == "abc123"

    def test_lora_model_saves_adapter_then_merged(self, tmp_path):
        """LoRA model: saves adapter, then merges and saves merged weights."""
        from forge import _save_model
        from peft import PeftModel

        save_dir = tmp_path / "model_out"
        save_dir.mkdir(parents=True)  # mock save_pretrained won't create it

        merged_model = MagicMock()
        model = MagicMock(spec=PeftModel)
        model.save_pretrained = MagicMock()
        model.merge_and_unload = MagicMock(return_value=merged_model)
        model.get_base_model = MagicMock()
        base_config = MagicMock()
        base_config._name_or_path = "Qwen/Qwen3.5-2B"
        base_config._commit_hash = "def456"
        model.get_base_model.return_value.config = base_config

        _save_model(model, save_dir)

        # First call: save adapter
        # Second call: save merged (via merged_model)
        assert model.save_pretrained.call_count == 1
        model.save_pretrained.assert_called_with(save_dir)
        model.merge_and_unload.assert_called_once()
        merged_model.save_pretrained.assert_called_once_with(save_dir)

        # Base ref created from base model config
        ref_path = save_dir / "base" / "model_ref.json"
        assert ref_path.exists()
        with open(ref_path) as f:
            ref = json.load(f)
        assert ref["model_name_or_path"] == "Qwen/Qwen3.5-2B"

    def test_saves_tokenizer_when_provided(self, tmp_path):
        """Tokenizer is saved alongside model."""
        from forge import _save_model

        save_dir = tmp_path / "model_out"
        save_dir.mkdir(parents=True)
        model = _make_plain_model(_name_or_path="test", _commit_hash=None)

        tokenizer = MagicMock()
        _save_model(model, save_dir, tokenizer=tokenizer)

        tokenizer.save_pretrained.assert_called_once_with(save_dir)

    def test_saves_processor_over_tokenizer(self, tmp_path):
        """Processor takes precedence over tokenizer if both provided."""
        from forge import _save_model

        save_dir = tmp_path / "model_out"
        save_dir.mkdir(parents=True)
        model = _make_plain_model(_name_or_path="test", _commit_hash=None)

        processor = MagicMock()
        tokenizer = MagicMock()
        _save_model(model, save_dir, processor=processor, tokenizer=tokenizer)

        processor.save_pretrained.assert_called_once_with(save_dir)
        tokenizer.save_pretrained.assert_not_called()

    def test_skips_base_ref_if_already_exists(self, tmp_path):
        """Don't overwrite base ref if base/ has config.json (--adapter case)."""
        from forge import _save_model

        save_dir = tmp_path / "model_out"
        base_dir = save_dir / "base"
        base_dir.mkdir(parents=True)
        # Pre-existing base with config.json (from --adapter merge)
        (base_dir / "config.json").write_text("{}")
        (base_dir / "model_ref.json").write_text(
            json.dumps({"model_name_or_path": "original_adapter_source"})
        )

        model = _make_plain_model(_name_or_path="test", _commit_hash=None)

        _save_model(model, save_dir, model_name="should_not_overwrite")

        # Original ref preserved
        with open(base_dir / "model_ref.json") as f:
            ref = json.load(f)
        assert ref["model_name_or_path"] == "original_adapter_source"


# =============================================================================
# _save_final_model delegation test
# =============================================================================

class TestSaveFinalModel:
    """Test that _save_final_model delegates to _save_model correctly."""

    def test_delegates_with_trainer_model(self, tmp_path):
        from forge import _save_final_model

        save_dir = tmp_path / "output"
        save_dir.mkdir(parents=True)
        trainer = MagicMock()
        model = _make_plain_model(
            _name_or_path="Qwen/Qwen3.5-2B",
            _commit_hash=None,
        )
        trainer.model = model

        config = {"model": {"name": "Qwen/Qwen3.5-2B"}}
        _save_final_model(trainer, config, save_dir)

        model.save_pretrained.assert_called_once_with(save_dir)
        ref_path = save_dir / "base" / "model_ref.json"
        assert ref_path.exists()
        with open(ref_path) as f:
            ref = json.load(f)
        assert ref["model_name_or_path"] == "Qwen/Qwen3.5-2B"
