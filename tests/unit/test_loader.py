"""Tests for src/models/loader.py - routing logic, loaders, and config parsing.

Covers:
- needs_attention_outputs() config parsing
- _detect_attn_implementation() device/eager routing
- _load_classifier() pretrained and fresh paths
- _load_causal_lm() dtype selection, embedding resize
- _load_vlm() dtype selection, fresh init warning
- _apply_lora() wraps model with LoRA config
- _resolve_checkpoint() local checkpoint detection
- _log_params() trainable vs total param logging
- load_model() full flow with task routing

All tests use mocks. No network calls, no downloads.

Run:
    python -m pytest tests/unit/test_loader.py -v
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.models.loader import (
    _apply_lora,
    _detect_attn_implementation,
    _load_causal_lm,
    _load_classifier,
    _load_vlm,
    _log_params,
    _resolve_checkpoint,
    load_model,
    needs_attention_outputs,
)

# ===========================================================================
# Helpers
# ===========================================================================


def _fake_model(**kwargs):
    """Minimal mock model for routing tests."""
    model = MagicMock()
    params = [MagicMock(numel=lambda: 1000, requires_grad=True)]
    model.parameters.return_value = params
    model.to.return_value = model
    model.config = MagicMock()
    model.config.vocab_size = 32000
    return model


# ===========================================================================
# needs_attention_outputs
# ===========================================================================


class TestNeedsAttentionOutputs:
    """Config parsing for attention/feature output requirements."""

    def test_empty_config(self):
        assert needs_attention_outputs({}) is False

    def test_no_distillation_key(self):
        assert needs_attention_outputs({"data": {"task": "mnli"}}) is False

    # --- loss_type ---

    def test_loss_type_attention_kl(self):
        config = {"distillation": {"loss_type": "attention_kl"}}
        assert needs_attention_outputs(config) is True

    def test_loss_type_response_kl(self):
        config = {"distillation": {"loss_type": "response_kl"}}
        assert needs_attention_outputs(config) is False

    def test_loss_type_feature_mse(self):
        """feature_mse does not contain 'attention', should be False."""
        config = {"distillation": {"loss_type": "feature_mse"}}
        assert needs_attention_outputs(config) is False

    # --- components ---

    def test_components_with_attention(self):
        config = {
            "distillation": {
                "components": [
                    {"type": "response_kl", "weight": 0.5},
                    {"type": "attention_kl", "weight": 0.5},
                ]
            }
        }
        assert needs_attention_outputs(config) is True

    def test_components_without_attention(self):
        config = {
            "distillation": {
                "components": [
                    {"type": "response_kl", "weight": 0.5},
                    {"type": "feature_mse", "weight": 0.5},
                ]
            }
        }
        assert needs_attention_outputs(config) is False

    def test_components_empty_list(self):
        config: dict = {"distillation": {"components": []}}
        assert needs_attention_outputs(config) is False

    # --- stages ---

    def test_stages_with_attention(self):
        config = {
            "distillation": {
                "stages": [
                    {"strategy": "feature_mse", "until": 0.5},
                    {"strategy": "attention_kl", "until": 1.0},
                ]
            }
        }
        assert needs_attention_outputs(config) is True

    def test_stages_without_attention(self):
        config = {
            "distillation": {
                "stages": [
                    {"strategy": "feature_mse", "until": 0.5},
                    {"strategy": "response_kl", "until": 1.0},
                ]
            }
        }
        assert needs_attention_outputs(config) is False

    def test_stages_empty_list(self):
        config: dict = {"distillation": {"stages": []}}
        assert needs_attention_outputs(config) is False


# ===========================================================================
# _detect_attn_implementation
# ===========================================================================


class TestDetectAttnImplementation:
    """Attention implementation selection based on (eager, device) combos."""

    @pytest.mark.parametrize("device", ["cuda", "cpu", "mps"])
    def test_eager_always_returns_eager(self, device):
        assert _detect_attn_implementation(eager_attention=True, device=device) == "eager"

    def test_cuda_without_eager_returns_sdpa(self):
        assert _detect_attn_implementation(eager_attention=False, device="cuda") == "sdpa"

    @pytest.mark.parametrize("device", ["cpu", "mps"])
    def test_non_cuda_without_eager_returns_none(self, device):
        assert _detect_attn_implementation(eager_attention=False, device=device) is None


# ===========================================================================
# _load_classifier
# ===========================================================================


class TestLoadClassifier:
    """Pretrained and fresh paths for sequence classification models."""

    @patch("src.models.loader.AutoModelForSequenceClassification")
    def test_pretrained_calls_from_pretrained(self, mock_auto):
        model = _fake_model()
        mock_auto.from_pretrained.return_value = model
        _load_classifier("bert-base-uncased", "pretrained", 3, "cpu")
        mock_auto.from_pretrained.assert_called_once_with("bert-base-uncased", num_labels=3)
        model.to.assert_called_once_with("cpu")

    @patch("src.models.loader.AutoConfig")
    @patch("src.models.loader.AutoModelForSequenceClassification")
    def test_fresh_calls_from_config(self, mock_auto, mock_config_cls):
        mock_config = MagicMock()
        mock_config_cls.from_pretrained.return_value = mock_config
        model = _fake_model()
        mock_auto.from_config.return_value = model
        _load_classifier("bert-base-uncased", "fresh", 3, "cpu")
        mock_config_cls.from_pretrained.assert_called_once_with("bert-base-uncased")
        assert mock_config.num_labels == 3
        mock_auto.from_config.assert_called_once_with(mock_config)

    @patch("src.models.loader.AutoConfig")
    def test_fresh_distilbert_uses_distilbert_class(self, mock_config_cls):
        mock_config = MagicMock()
        mock_config_cls.from_pretrained.return_value = mock_config
        model = _fake_model()
        with patch(
            "src.models.loader.DistilBertForSequenceClassification",
            create=True,
        ):
            # Patch the import inside the function
            with patch.dict("sys.modules", {}):
                import transformers

                with patch.object(
                    transformers,
                    "DistilBertForSequenceClassification",
                    create=True,
                ) as mock_distil_cls:
                    mock_distil_cls.return_value = model
                    _load_classifier("distilbert-base-uncased", "fresh", 2, "cpu")
                    mock_distil_cls.assert_called_once_with(mock_config)


# ===========================================================================
# _load_causal_lm
# ===========================================================================


class TestLoadCausalLM:
    """Causal LM loading: dtype selection and embedding resize."""

    @patch("src.models.loader.AutoModelForCausalLM")
    def test_pretrained_cuda_uses_fp16(self, mock_auto):
        model = _fake_model()
        mock_auto.from_pretrained.return_value = model
        tokenizer = MagicMock()
        tokenizer.__len__ = MagicMock(return_value=32000)
        _load_causal_lm("llama-1b", "pretrained", "cuda", tokenizer)
        call_kwargs = mock_auto.from_pretrained.call_args
        assert call_kwargs.kwargs["dtype"] == torch.float16

    @patch("src.models.loader.AutoModelForCausalLM")
    def test_pretrained_cpu_uses_fp32(self, mock_auto):
        model = _fake_model()
        mock_auto.from_pretrained.return_value = model
        tokenizer = MagicMock()
        tokenizer.__len__ = MagicMock(return_value=32000)
        _load_causal_lm("llama-1b", "pretrained", "cpu", tokenizer)
        call_kwargs = mock_auto.from_pretrained.call_args
        assert call_kwargs.kwargs["dtype"] == torch.float32

    @patch("src.models.loader.AutoModelForCausalLM")
    def test_embedding_resize_when_tokenizer_larger(self, mock_auto):
        model = _fake_model()
        model.config.vocab_size = 32000
        mock_auto.from_pretrained.return_value = model
        tokenizer = MagicMock()
        tokenizer.__len__ = MagicMock(return_value=32100)  # larger than vocab
        _load_causal_lm("llama-1b", "pretrained", "cpu", tokenizer)
        model.resize_token_embeddings.assert_called_once_with(32100)

    @patch("src.models.loader.AutoModelForCausalLM")
    def test_no_resize_when_tokenizer_matches(self, mock_auto):
        model = _fake_model()
        model.config.vocab_size = 32000
        mock_auto.from_pretrained.return_value = model
        tokenizer = MagicMock()
        tokenizer.__len__ = MagicMock(return_value=32000)
        _load_causal_lm("llama-1b", "pretrained", "cpu", tokenizer)
        model.resize_token_embeddings.assert_not_called()

    @patch("src.models.loader.AutoConfig")
    @patch("src.models.loader.AutoModelForCausalLM")
    def test_fresh_init_from_config(self, mock_auto, mock_config_cls):
        mock_config = MagicMock()
        mock_config_cls.from_pretrained.return_value = mock_config
        model = _fake_model()
        mock_auto.from_config.return_value = model
        tokenizer = MagicMock()
        tokenizer.__len__ = MagicMock(return_value=32000)
        _load_causal_lm("llama-1b", "fresh", "cpu", tokenizer)
        mock_auto.from_config.assert_called_once_with(mock_config)

    @patch("src.models.loader.AutoModelForCausalLM")
    def test_pretrained_no_device_map(self, mock_auto):
        """device_map removed - all device placement via .to(device)."""
        model = _fake_model()
        mock_auto.from_pretrained.return_value = model
        tokenizer = MagicMock()
        tokenizer.__len__ = MagicMock(return_value=32000)
        _load_causal_lm("llama-1b", "pretrained", "cpu", tokenizer)
        call_kwargs = mock_auto.from_pretrained.call_args
        assert "device_map" not in call_kwargs.kwargs
        model.to.assert_called_with("cpu")


# ===========================================================================
# _load_vlm
# ===========================================================================


class TestLoadVLM:
    """VLM loading: dtype selection and fresh init warning."""

    @patch("src.models.loader.AutoModelForImageTextToText", create=True)
    def test_cuda_uses_bf16(self, mock_auto):
        # Patch the import inside _load_vlm
        with patch.dict("sys.modules"):
            mock_auto_cls = MagicMock()
            model = _fake_model()
            mock_auto_cls.from_pretrained.return_value = model
            with patch(
                "transformers.AutoModelForImageTextToText",
                mock_auto_cls,
                create=True,
            ):
                _load_vlm("Qwen/Qwen3-VL-2B", "pretrained", "cuda")
                call_kwargs = mock_auto_cls.from_pretrained.call_args
                assert call_kwargs.kwargs["dtype"] == torch.bfloat16

    @patch("src.models.loader.AutoModelForImageTextToText", create=True)
    def test_cpu_uses_fp32(self, mock_auto):
        with patch.dict("sys.modules"):
            mock_auto_cls = MagicMock()
            model = _fake_model()
            mock_auto_cls.from_pretrained.return_value = model
            with patch(
                "transformers.AutoModelForImageTextToText",
                mock_auto_cls,
                create=True,
            ):
                _load_vlm("Qwen/Qwen3-VL-2B", "pretrained", "cpu")
                call_kwargs = mock_auto_cls.from_pretrained.call_args
                assert call_kwargs.kwargs["dtype"] == torch.float32

    @patch("src.models.loader.AutoModelForImageTextToText", create=True)
    def test_fresh_init_logs_warning(self, mock_auto, caplog):
        with patch.dict("sys.modules"):
            mock_auto_cls = MagicMock()
            model = _fake_model()
            mock_auto_cls.from_pretrained.return_value = model
            with patch(
                "transformers.AutoModelForImageTextToText",
                mock_auto_cls,
                create=True,
            ):
                import logging

                with caplog.at_level(logging.WARNING, logger="src.models.loader"):
                    _load_vlm("Qwen/Qwen3-VL-2B", "fresh", "cpu")
                assert "Fresh init not recommended" in caplog.text


# ===========================================================================
# _apply_lora
# ===========================================================================


class TestApplyLora:
    """LoRA adapter wrapping."""

    @patch("src.models.loader.get_peft_model")
    @patch("src.models.loader.LoraConfig")
    def test_apply_lora_wraps_model(self, mock_lora_config, mock_get_peft):
        model = _fake_model()
        wrapped = _fake_model()
        mock_get_peft.return_value = wrapped
        lora_cfg = {
            "r": 8,
            "alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            "dropout": 0.1,
        }
        result = _apply_lora(model, lora_cfg)
        mock_lora_config.assert_called_once_with(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        mock_get_peft.assert_called_once_with(model, mock_lora_config.return_value)
        assert result is wrapped


# ===========================================================================
# _log_params
# ===========================================================================


class TestLogParams:
    """Parameter logging doesn't crash."""

    def test_all_trainable(self, caplog):
        """When all params are trainable, logs single count."""
        import logging

        model = MagicMock()
        p = MagicMock()
        p.numel.return_value = 1000000
        p.requires_grad = True
        model.parameters.return_value = [p]
        with caplog.at_level(logging.INFO, logger="src.models.loader"):
            _log_params("test-model", model)
        assert "1.0M params" in caplog.text

    def test_partial_trainable(self, caplog):
        """When some params frozen, logs trainable/total."""
        import logging

        model = MagicMock()
        p1 = MagicMock()
        p1.numel.return_value = 500000
        p1.requires_grad = True
        p2 = MagicMock()
        p2.numel.return_value = 500000
        p2.requires_grad = False
        model.parameters.return_value = [p1, p2]
        with caplog.at_level(logging.INFO, logger="src.models.loader"):
            _log_params("test-model", model)
        assert "0.5M trainable" in caplog.text
        assert "1.0M total" in caplog.text


# ===========================================================================
# _resolve_checkpoint
# ===========================================================================


class TestResolveCheckpoint:
    """Local checkpoint detection for merged and adapter-only cases."""

    def test_nonexistent_dir_returns_none(self):
        result = _resolve_checkpoint("/nonexistent/path", "mnli", "cpu", num_labels=3)
        assert result is None

    def test_dir_without_weights_returns_none(self, tmp_path):
        result = _resolve_checkpoint(str(tmp_path), "mnli", "cpu", num_labels=3)
        assert result is None

    @patch("src.models.loader._load_classifier")
    def test_merged_safetensors_loads(self, mock_load, tmp_path):
        (tmp_path / "model.safetensors").write_text("fake")
        model = _fake_model()
        mock_load.return_value = model
        result = _resolve_checkpoint(str(tmp_path), "mnli", "cpu", num_labels=3)
        assert result is model
        mock_load.assert_called_once_with(str(tmp_path), "pretrained", 3, "cpu")

    @patch("src.models.loader._load_classifier")
    def test_merged_pytorch_bin_loads(self, mock_load, tmp_path):
        (tmp_path / "pytorch_model.bin").write_text("fake")
        model = _fake_model()
        mock_load.return_value = model
        result = _resolve_checkpoint(str(tmp_path), "mnli", "cpu", num_labels=3)
        assert result is model

    @patch("src.models.loader._load_vlm")
    def test_merged_ocr_task_routes_to_vlm(self, mock_load, tmp_path):
        (tmp_path / "model.safetensors").write_text("fake")
        model = _fake_model()
        mock_load.return_value = model
        result = _resolve_checkpoint(str(tmp_path), "ocr", "cpu")
        assert result is model
        mock_load.assert_called_once()

    @patch("src.models.loader._load_causal_lm")
    def test_merged_translation_routes_to_causal_lm(self, mock_load, tmp_path):
        (tmp_path / "model.safetensors").write_text("fake")
        model = _fake_model()
        mock_load.return_value = model
        tokenizer = MagicMock()
        result = _resolve_checkpoint(str(tmp_path), "europarl", "cpu", tokenizer=tokenizer)
        assert result is model
        mock_load.assert_called_once()

    @patch("src.models.loader._load_classifier")
    def test_adapter_only_merges(self, mock_load, tmp_path):
        import json

        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "bert-base"})
        )
        base_model = _fake_model()
        mock_load.return_value = base_model
        merged_model = _fake_model()
        peft_model = MagicMock()
        peft_model.merge_and_unload.return_value = merged_model

        with patch("src.inference.engine._resolve_base_model", return_value=("bert-base", None)):
            with patch("peft.PeftModel") as mock_peft_cls:
                mock_peft_cls.from_pretrained.return_value = peft_model
                result = _resolve_checkpoint(str(tmp_path), "mnli", "cpu", num_labels=3)
        assert result is merged_model


# ===========================================================================
# load_model full flow
# ===========================================================================


class TestLoadModelRouting:
    """Verify load_model routes to the correct internal loader by task type."""

    @patch("src.models.loader._resolve_checkpoint", return_value=None)
    @patch("src.models.loader._log_params")
    @patch("src.models.loader._load_classifier")
    def test_classification_routes_to_classifier(self, mock_load, mock_log, mock_ckpt):
        mock_load.return_value = _fake_model()
        load_model("bert-base", "mnli", "cpu", num_labels=3)
        mock_load.assert_called_once_with("bert-base", "pretrained", 3, "cpu")

    @patch("src.models.loader._resolve_checkpoint", return_value=None)
    @patch("src.models.loader._log_params")
    @patch("src.models.loader._load_causal_lm")
    def test_translation_routes_to_causal_lm(self, mock_load, mock_log, mock_ckpt):
        mock_load.return_value = _fake_model()
        tok = MagicMock()
        load_model("llama-1b", "europarl", "cpu", tokenizer=tok)
        mock_load.assert_called_once_with("llama-1b", "pretrained", "cpu", tok, False)

    @patch("src.models.loader._resolve_checkpoint", return_value=None)
    @patch("src.models.loader._log_params")
    @patch("src.models.loader._load_vlm")
    def test_ocr_routes_to_vlm(self, mock_load, mock_log, mock_ckpt):
        mock_load.return_value = _fake_model()
        load_model("Qwen/Qwen3-VL-2B", "ocr", "cpu")
        mock_load.assert_called_once_with("Qwen/Qwen3-VL-2B", "pretrained", "cpu", False)

    @patch("src.models.loader._resolve_checkpoint", return_value=None)
    @patch("src.models.loader._log_params")
    @patch("src.models.loader._load_classifier")
    @patch("src.models.loader._apply_lora")
    def test_lora_applied_when_config_provided(self, mock_lora, mock_load, mock_log, mock_ckpt):
        base_model = _fake_model()
        lora_model = _fake_model()
        mock_load.return_value = base_model
        mock_lora.return_value = lora_model
        lora_cfg = {"r": 8, "alpha": 16, "target_modules": ["q_proj"], "dropout": 0.1}
        load_model("bert-base", "mnli", "cpu", num_labels=3, lora_config=lora_cfg)
        mock_lora.assert_called_once_with(base_model, lora_cfg)
        # _log_params is called on the final model (after LoRA)
        mock_log.assert_called_once()

    @patch("src.models.loader._log_params")
    @patch("src.models.loader._resolve_checkpoint")
    def test_checkpoint_used_when_resolved(self, mock_ckpt, mock_log):
        ckpt_model = _fake_model()
        mock_ckpt.return_value = ckpt_model
        result = load_model("some/path", "mnli", "cpu", num_labels=3)
        assert result is ckpt_model

    @patch("src.models.loader._resolve_checkpoint", return_value=None)
    @patch("src.models.loader._log_params")
    @patch("src.models.loader._load_classifier")
    def test_fresh_init_passed_through(self, mock_load, mock_log, mock_ckpt):
        mock_load.return_value = _fake_model()
        load_model("bert-base", "mnli", "cpu", init="fresh", num_labels=3)
        mock_load.assert_called_once_with("bert-base", "fresh", 3, "cpu")
