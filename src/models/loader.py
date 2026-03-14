"""Model loading primitives."""

import logging

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoConfig,
    PreTrainedTokenizer
)

from src.tasks import is_translation_task, is_ocr_task

logger = logging.getLogger(__name__)


def _detect_attn_implementation(eager_attention: bool, device: str):
    """Select best attention implementation.

    Args:
        eager_attention: If True, force eager (needed for attention distillation).
        device: Device string.

    Returns:
        "eager", "sdpa", or None (let transformers choose).
    """
    if eager_attention:
        return "eager"
    if device == "cuda":
        return "sdpa"
    return None


# =============================================================================
# Primitives: load a single model
# =============================================================================

def _load_classifier(name: str, init: str, num_labels: int, device: str):
    """Load a single sequence classification model.

    Args:
        name: HuggingFace model ID (e.g., "bert-base-uncased")
        init: "pretrained" or "fresh"
        num_labels: Number of classification labels
        device: Device string

    Returns:
        Model on device
    """
    if init == "fresh":
        logger.info(f"Initializing fresh {name} model...")
        model_config = AutoConfig.from_pretrained(name)
        model_config.num_labels = num_labels
        # DistilBert doesn't register with AutoModel for from_config
        if "distilbert" in name.lower():
            from transformers import DistilBertForSequenceClassification
            model = DistilBertForSequenceClassification(model_config)
        else:
            model = AutoModelForSequenceClassification.from_config(model_config)
        return model.to(device)

    return AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=num_labels
    ).to(device)


def _load_causal_lm(name: str, init: str, device: str, tokenizer: PreTrainedTokenizer,
                     eager_attention: bool = False):
    """Load a single causal language model with embedding resize.

    Args:
        name: HuggingFace model ID (e.g., "meta-llama/Llama-3.2-1B-Instruct")
        init: "pretrained" or "fresh"
        device: Device string
        tokenizer: Tokenizer (for embedding resize check)
        eager_attention: Use eager attention (required for attention distillation)

    Returns:
        Model on device
    """
    dtype = torch.float16 if device == "cuda" else torch.float32

    attn_impl = _detect_attn_implementation(eager_attention, device)

    if init == "fresh":
        model_config = AutoConfig.from_pretrained(name)
        if attn_impl:
            model_config._attn_implementation = attn_impl
        model = AutoModelForCausalLM.from_config(model_config).to(dtype)
        if device == "cuda":
            model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            name,
            dtype=dtype,
            device_map=device if device == "cuda" else None,
            attn_implementation=attn_impl,
        )

    # Resize embeddings if tokenizer has custom tokens
    if len(tokenizer) > model.config.vocab_size:
        logger.info(f"Resizing embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    return model


def _load_vlm(name: str, init: str, device: str, eager_attention: bool = False):
    """Load a Vision-Language Model (Qwen2.5-VL or Qwen3-VL family).

    Auto-detects the correct model class based on the model config.

    Args:
        name: HuggingFace model ID (e.g., "Qwen/Qwen3-VL-2B-Instruct")
        init: "pretrained" (fresh init not supported for VLMs)
        device: Device string
        eager_attention: Use eager attention (for attention distillation)

    Returns:
        Model on device
    """
    from transformers import AutoModelForImageTextToText

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    attn_impl = _detect_attn_implementation(eager_attention, device)

    kwargs = {
        "dtype": dtype,
        "device_map": "auto" if device == "cuda" else None,
    }

    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    if init == "fresh":
        logger.warning("Fresh init not recommended for VLMs, using pretrained")

    logger.info(f"Loading VLM: {name} (dtype={dtype})...")
    model = AutoModelForImageTextToText.from_pretrained(name, **kwargs)
    logger.info(f"Loaded VLM: {name}")

    return model


def _apply_lora(model, lora_cfg: dict):
    """Wrap model with LoRA adapters.

    Args:
        model: Base model to wrap
        lora_cfg: LoRA config dict with keys: r, alpha, target_modules, dropout

    Returns:
        PEFT-wrapped model
    """
    logger.info("Applying LoRA...")
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def _log_params(name: str, model):
    """Log model parameter count."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable == total:
        logger.info(f"  {name} ({total / 1e6:.1f}M params)")
    else:
        logger.info(f"  {name} ({trainable / 1e6:.1f}M trainable / {total / 1e6:.1f}M total)")


def needs_attention_outputs(config: dict) -> bool:
    """Check if config requires attention outputs for distillation."""
    distill = config.get("distillation", {})

    # Check loss_type
    if "attention" in distill.get("loss_type", ""):
        return True

    # Check components (composite loss)
    for comp in distill.get("components", []):
        if "attention" in comp.get("type", ""):
            return True

    # Check stages (staged distillation)
    for stage in distill.get("stages", []):
        if "attention" in stage.get("strategy", ""):
            return True

    return False


def _resolve_checkpoint(name: str, task: str, device: str,
                        eager_attention: bool = False,
                        tokenizer: PreTrainedTokenizer = None,
                        num_labels: int = None):
    """Load a model from a local checkpoint, handling both merged and adapter-only cases.

    Mirrors the InferenceEngine loading pattern: checks for merged weights first,
    falls back to adapter resolution (base model + LoRA merge).

    Args:
        name: Local path to model directory (run root or checkpoint-N/)
        task: Task string (routes to VLM vs classifier vs causal LM)
        device: Device string
        eager_attention: Use eager attention (for attention distillation)
        tokenizer: Tokenizer (needed for causal LM embedding resize check)
        num_labels: Number of labels (needed for classifiers)

    Returns:
        Loaded model, or None if path is not a local checkpoint.
    """
    from pathlib import Path
    path = Path(name)
    if not path.is_dir():
        return None

    has_merged = (
        (path / "model.safetensors").exists()
        or (path / "pytorch_model.bin").exists()
    )
    has_adapter = (path / "adapter_config.json").exists()

    if not has_merged and not has_adapter:
        return None

    def _load(model_name):
        if is_ocr_task(task):
            return _load_vlm(model_name, "pretrained", device, eager_attention)
        elif is_translation_task(task):
            return _load_causal_lm(model_name, "pretrained", device, tokenizer, eager_attention)
        else:
            return _load_classifier(model_name, "pretrained", num_labels, device)

    if has_merged:
        return _load(name)

    # Adapter-only: resolve base, load, merge
    from src.inference.engine import _resolve_base_model
    from peft import PeftModel

    base_model_id, revision = _resolve_base_model(path)
    logger.info(f"Resolving adapter checkpoint: base={base_model_id}, adapter={path}")

    base = _load(base_model_id)
    model = PeftModel.from_pretrained(base, str(path))
    model = model.merge_and_unload()
    logger.info("Adapter merged into base model")
    return model


# =============================================================================
# Public API
# =============================================================================

def load_model(name: str, task: str, device: str, *,
               init: str = "pretrained",
               tokenizer: PreTrainedTokenizer = None,
               num_labels: int = None,
               eager: bool = False,
               lora_config: dict = None) -> torch.nn.Module:
    """Load a single model, routing by task type.

    Tries local checkpoint first (merged or adapter-only), falls back to Hub.

    Args:
        name: Model name (HF ID) or local checkpoint path
        task: Task string (routes to VLM, causal LM, or classifier)
        device: Device string
        init: "pretrained" or "fresh"
        tokenizer: Tokenizer (needed for causal LM embedding resize)
        num_labels: Number of labels (needed for classifiers)
        eager: Use eager attention (for attention distillation)
        lora_config: LoRA config dict (if provided, wraps model with adapters)

    Returns:
        Loaded model on device
    """
    model = _resolve_checkpoint(name, task, device, eager,
                                tokenizer=tokenizer, num_labels=num_labels)
    if model is None:
        if is_ocr_task(task):
            model = _load_vlm(name, init, device, eager)
        elif is_translation_task(task):
            model = _load_causal_lm(name, init, device, tokenizer, eager)
        else:
            model = _load_classifier(name, init, num_labels, device)

    if lora_config:
        model = _apply_lora(model, lora_config)

    _log_params(name, model)
    return model

