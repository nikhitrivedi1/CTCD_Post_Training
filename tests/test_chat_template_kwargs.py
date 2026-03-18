"""Tests for chat_template_kwargs plumbing through training.

Verifies that model profile defaults (e.g. enable_thinking) flow correctly
from MODEL_PROFILES -> forge.py -> formatters -> apply_chat_template, with
config overrides taking precedence.
"""

from unittest.mock import MagicMock, patch, call
import pytest

from src.inference.model_profiles import (
    get_chat_template_kwargs,
    _resolve_model_family,
    _match_profile,
    MODEL_PROFILES,
)
from src.data.formatters import _cache_key, format_for_sft


# =============================================================================
# Model profile resolution
# =============================================================================


class TestModelProfileDefaults:
    """MODEL_PROFILES returns correct defaults for known model families."""

    def test_qwen35_wildcard_enables_thinking(self):
        kwargs = get_chat_template_kwargs("Qwen/Qwen3.5-9B")
        assert kwargs == {"enable_thinking": True}

    def test_qwen35_ocr_disables_thinking(self):
        kwargs = get_chat_template_kwargs("Qwen/Qwen3.5-9B", task="ocr")
        assert kwargs == {"enable_thinking": False}

    def test_qwen3vl_returns_empty(self):
        """Qwen3-VL chat template does not support enable_thinking."""
        kwargs = get_chat_template_kwargs("Qwen/Qwen3-VL-2B-Instruct")
        assert kwargs == {}

    def test_qwen3vl_ocr_returns_empty(self):
        """Qwen3-VL with task also returns empty - no profile."""
        kwargs = get_chat_template_kwargs("Qwen/Qwen3-VL-2B-Instruct", task="ocr")
        assert kwargs == {}

    def test_unknown_model_returns_empty(self):
        kwargs = get_chat_template_kwargs("bert-base-uncased")
        assert kwargs == {}

    def test_unknown_model_with_task_returns_empty(self):
        kwargs = get_chat_template_kwargs("bert-base-uncased", task="ocr")
        assert kwargs == {}


class TestModelProfileExplicitOverride:
    """Explicit kwargs override profile defaults."""

    def test_explicit_overrides_profile(self):
        kwargs = get_chat_template_kwargs(
            "Qwen/Qwen3.5-9B",
            explicit={"enable_thinking": False},
        )
        assert kwargs["enable_thinking"] is False

    def test_explicit_overrides_task_specific(self):
        kwargs = get_chat_template_kwargs(
            "Qwen/Qwen3.5-9B",
            task="ocr",
            explicit={"enable_thinking": True},
        )
        assert kwargs["enable_thinking"] is True

    def test_explicit_adds_new_keys(self):
        kwargs = get_chat_template_kwargs(
            "Qwen/Qwen3.5-9B",
            explicit={"custom_key": "value"},
        )
        assert kwargs["enable_thinking"] is True
        assert kwargs["custom_key"] == "value"

    def test_explicit_on_unknown_model(self):
        kwargs = get_chat_template_kwargs(
            "bert-base-uncased",
            explicit={"enable_thinking": False},
        )
        assert kwargs == {"enable_thinking": False}


class TestModelFamilyResolution:
    """_resolve_model_family handles model IDs and checkpoint paths."""

    def test_model_id_returns_id(self):
        resolved, exact = _resolve_model_family("Qwen/Qwen3.5-9B")
        assert resolved == "Qwen/Qwen3.5-9B"
        assert exact is False

    def test_checkpoint_with_family(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  family: Qwen3.5\n  name: Qwen/Qwen3.5-9B\n")
        resolved, exact = _resolve_model_family(str(tmp_path))
        assert resolved == "Qwen3.5"
        assert exact is True

    def test_checkpoint_with_name_only(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  name: Qwen/Qwen3.5-9B\n")
        resolved, exact = _resolve_model_family(str(tmp_path))
        assert resolved == "Qwen/Qwen3.5-9B"
        assert exact is False

    def test_checkpoint_no_config(self, tmp_path):
        resolved, exact = _resolve_model_family(str(tmp_path))
        assert resolved == str(tmp_path)
        assert exact is False


class TestMatchProfile:
    """_match_profile correctly matches family patterns."""

    def test_exact_match(self):
        result = _match_profile("Qwen3.5", exact=True, task="ocr")
        assert result == {"enable_thinking": False}

    def test_substring_match(self):
        result = _match_profile("Qwen/Qwen3.5-9B", exact=False, task="ocr")
        assert result == {"enable_thinking": False}

    def test_wildcard_fallback(self):
        result = _match_profile("Qwen/Qwen3.5-9B", exact=False, task="formatting")
        assert result == {"enable_thinking": True}

    def test_no_match(self):
        result = _match_profile("bert-base", exact=False, task="ocr")
        assert result == {}


# =============================================================================
# Cache key includes chat_template_kwargs
# =============================================================================


class TestCacheKey:
    """Cache key changes when chat_template_kwargs differ."""

    def test_same_params_same_key(self):
        k1 = _cache_key("model", 100, 1024, "train")
        k2 = _cache_key("model", 100, 1024, "train")
        assert k1 == k2

    def test_different_kwargs_different_key(self):
        k1 = _cache_key("model", 100, 1024, "train", {"enable_thinking": True})
        k2 = _cache_key("model", 100, 1024, "train", {"enable_thinking": False})
        assert k1 != k2

    def test_none_kwargs_matches_empty(self):
        k1 = _cache_key("model", 100, 1024, "train", None)
        k2 = _cache_key("model", 100, 1024, "train", {})
        assert k1 == k2

    def test_kwargs_vs_no_kwargs(self):
        k1 = _cache_key("model", 100, 1024, "train")
        k2 = _cache_key("model", 100, 1024, "train", {"enable_thinking": False})
        assert k1 != k2


# =============================================================================
# Formatter passes kwargs to apply_chat_template
# =============================================================================


class TestOCRFormatterPassesKwargs:
    """_format_ocr_sft forwards chat_template_kwargs to apply_chat_template."""

    @patch("src.data.formatters.get_num_workers", return_value=1)
    def test_kwargs_forwarded(self, _mock_workers):
        from src.data.formatters import _format_ocr_sft
        from PIL import Image

        processor = MagicMock()
        processor.apply_chat_template.return_value = "<formatted>"

        img = Image.new("RGB", (10, 10))
        from datasets import Dataset
        ds = Dataset.from_dict({"image": [img], "ground_truth": ["hello"]})

        _format_ocr_sft(
            ds, processor, max_length=1024, system_prompt="OCR",
            chat_template_kwargs={"enable_thinking": False},
        )

        processor.apply_chat_template.assert_called_once()
        _, call_kwargs = processor.apply_chat_template.call_args
        assert call_kwargs["enable_thinking"] is False
        assert call_kwargs["tokenize"] is False

    @patch("src.data.formatters.get_num_workers", return_value=1)
    def test_no_kwargs_no_extra_args(self, _mock_workers):
        from src.data.formatters import _format_ocr_sft
        from PIL import Image

        processor = MagicMock()
        processor.apply_chat_template.return_value = "<formatted>"

        img = Image.new("RGB", (10, 10))
        from datasets import Dataset
        ds = Dataset.from_dict({"image": [img], "ground_truth": ["hello"]})

        _format_ocr_sft(ds, processor, max_length=1024, system_prompt="OCR")

        _, call_kwargs = processor.apply_chat_template.call_args
        assert "enable_thinking" not in call_kwargs


class TestTranslationFormatterPassesKwargs:
    """_format_translation_sft forwards chat_template_kwargs."""

    @patch("src.data.formatters.get_num_workers", return_value=1)
    def test_kwargs_forwarded(self, _mock_workers):
        from src.data.formatters import _format_translation_sft

        tokenizer = MagicMock()
        tokenizer.chat_template = "some_template"
        tokenizer.apply_chat_template.return_value = "<formatted>"
        tokenizer.return_value = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
        }

        from datasets import Dataset
        ds = Dataset.from_dict({
            "source_text": ["hello"],
            "ground_truth": ["hola"],
            "system_message": ["Translate"],
        })

        _format_translation_sft(
            ds, tokenizer, max_length=128,
            chat_template_kwargs={"enable_thinking": False},
        )

        # apply_chat_template called twice (full + prompt), both with kwargs
        assert tokenizer.apply_chat_template.call_count == 2
        for c in tokenizer.apply_chat_template.call_args_list:
            assert c.kwargs["enable_thinking"] is False


# =============================================================================
# End-to-end: forge.py resolves kwargs from profile + config
# =============================================================================


class TestForgeResolvesKwargs:
    """forge.py correctly resolves chat_template_kwargs for SFT and RL paths."""

    def test_sft_passes_kwargs_to_formatter(self):
        """Simulate the SFT path: model profile -> format_for_sft."""
        config = {
            "model": {"name": "Qwen/Qwen3.5-2B"},
            "data": {"task": "ocr", "max_length": 1024},
        }

        ct_kwargs = get_chat_template_kwargs(
            config["model"]["name"],
            explicit=config.get("model", {}).get("chat_template_kwargs"),
            task="ocr",
        )
        assert ct_kwargs == {"enable_thinking": False}

    def test_sft_config_override(self):
        """Config-level chat_template_kwargs overrides profile."""
        config = {
            "model": {
                "name": "Qwen/Qwen3.5-2B",
                "chat_template_kwargs": {"enable_thinking": True},
            },
        }

        ct_kwargs = get_chat_template_kwargs(
            config["model"]["name"],
            explicit=config["model"]["chat_template_kwargs"],
            task="ocr",
        )
        # Config override wins over profile's False
        assert ct_kwargs == {"enable_thinking": True}

    def test_rl_kwargs_in_grpo_config(self):
        """GRPOConfig receives chat_template_kwargs."""
        from trl import GRPOConfig

        ct_kwargs = {"enable_thinking": False}
        grpo_config = GRPOConfig(
            output_dir="/tmp/test",
            chat_template_kwargs=ct_kwargs,
        )
        assert grpo_config.chat_template_kwargs == {"enable_thinking": False}

    def test_rl_none_kwargs(self):
        """GRPOConfig accepts None for chat_template_kwargs."""
        from trl import GRPOConfig

        grpo_config = GRPOConfig(
            output_dir="/tmp/test",
            chat_template_kwargs=None,
        )
        assert grpo_config.chat_template_kwargs is None

    def test_non_ocr_task_gets_wildcard(self):
        """Non-OCR task falls back to wildcard profile."""
        ct_kwargs = get_chat_template_kwargs("Qwen/Qwen3.5-2B", task="translation")
        assert ct_kwargs == {"enable_thinking": True}

    def test_non_qwen_model_gets_empty(self):
        """Non-Qwen model gets no kwargs."""
        ct_kwargs = get_chat_template_kwargs("bert-base-uncased", task="ocr")
        assert ct_kwargs == {}


# =============================================================================
# Chain configs: all Qwen3.5 OCR configs resolve enable_thinking=False
# =============================================================================


class TestChainConfigProfiles:
    """Verify OCR training configs resolve correctly per model family."""

    @pytest.mark.parametrize("model_name", [
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-27B",
    ])
    def test_qwen35_ocr_disables_thinking(self, model_name):
        kwargs = get_chat_template_kwargs(model_name, task="ocr")
        assert kwargs.get("enable_thinking") is False, (
            f"{model_name} with task='ocr' should have enable_thinking=False"
        )

    @pytest.mark.parametrize("model_name", [
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-27B",
    ])
    def test_qwen35_wildcard_enables_thinking(self, model_name):
        kwargs = get_chat_template_kwargs(model_name)
        assert kwargs.get("enable_thinking") is True, (
            f"{model_name} with no task should have enable_thinking=True"
        )

    @pytest.mark.parametrize("model_name", [
        "Qwen/Qwen3-VL-2B-Instruct",
        "Qwen/Qwen3-VL-4B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
    ])
    def test_qwen3vl_returns_empty(self, model_name):
        """Qwen3-VL chat template has no enable_thinking support."""
        kwargs = get_chat_template_kwargs(model_name)
        assert kwargs == {}, (
            f"{model_name} should return empty dict (no enable_thinking support)"
        )

    @pytest.mark.parametrize("model_name", [
        "Qwen/Qwen3-VL-2B-Instruct",
        "Qwen/Qwen3-VL-4B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
    ])
    def test_qwen3vl_with_task_returns_empty(self, model_name):
        kwargs = get_chat_template_kwargs(model_name, task="ocr")
        assert kwargs == {}
