"""Load teacher and student model pairs for knowledge distillation."""

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

    if init == "fresh":
        model_config = AutoConfig.from_pretrained(name)
        if eager_attention:
            model_config._attn_implementation = "eager"
        model = AutoModelForCausalLM.from_config(model_config).to(dtype)
        if device == "cuda":
            model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=dtype,
            device_map=device if device == "cuda" else None,
            attn_implementation="eager" if eager_attention else None,
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

    kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto" if device == "cuda" else None,
    }

    if eager_attention:
        kwargs["attn_implementation"] = "eager"

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


def _log_params(role: str, name: str, model):
    """Log model parameter count."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable == total:
        logger.info(f"{role}: {name} ({total / 1e6:.1f}M params)")
    else:
        logger.info(f"{role}: {name} ({trainable / 1e6:.1f}M trainable / {total / 1e6:.1f}M total)")


def _needs_attention_outputs(config: dict) -> bool:
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


# =============================================================================
# Compositions: load teacher + student pairs
# =============================================================================

def load_models(config: dict, device: str, tokenizer: PreTrainedTokenizer, num_labels: int = None):
    """
    Load teacher and student models based on config.

    Routes to classifier or causal LM loader based on task type.
    Returns (None, student) if no teacher is needed (CE-only baselines).

    Args:
        config: Configuration dictionary
        device: Device to load models on ('cuda', 'mps', 'cpu')
        tokenizer: Tokenizer (may have been resized with custom tokens)
        num_labels: Number of labels for classification tasks

    Returns:
        Tuple of (teacher, student). Teacher is None if not specified.
    """
    teacher_name = config["model"].get("teacher")
    student_name = config["model"]["student"]
    student_init = config["model"].get("student_init", "pretrained")
    task = config.get("data", {}).get("task", "").lower()
    alpha = config.get("distillation", {}).get("alpha", 0.5)

    translation = is_translation_task(task)
    ocr = is_ocr_task(task)
    eager = _needs_attention_outputs(config)
    if eager:
        logger.info("Using eager attention (required for attention distillation)")

    # No teacher needed for CE-only baselines
    if teacher_name is None or alpha == 0.0:
        reason = "teacher is null" if teacher_name is None else "alpha=0.0"
        logger.info(f"No teacher needed ({reason}) - CE-only mode")
        teacher = None
    elif ocr:
        teacher = _load_vlm(teacher_name, "pretrained", device, eager)
    elif translation:
        teacher = _load_causal_lm(teacher_name, "pretrained", device, tokenizer, eager)
    else:
        teacher = _load_classifier(teacher_name, "pretrained", num_labels, device)

    # Load student
    if ocr:
        student = _load_vlm(student_name, student_init, device, eager)
        if config.get("training", {}).get("use_lora", False):
            student = _apply_lora(student, config["training"]["lora"])
    elif translation:
        student = _load_causal_lm(student_name, student_init, device, tokenizer, eager)
        if config.get("training", {}).get("use_lora", False):
            student = _apply_lora(student, config["training"]["lora"])
    else:
        student = _load_classifier(student_name, student_init, num_labels, device)

    # Log
    if teacher is not None:
        _log_params("Teacher", teacher_name, teacher)
    _log_params("Student", student_name, student)

    return teacher, student


def load_models_from_checkpoint(
    checkpoint_dir: str,
    config: dict,
    device: str,
    tokenizer: PreTrainedTokenizer,
    num_labels: int = None,
):
    """Load teacher and student from saved checkpoints.

    Student is loaded from checkpoint_dir, teacher from checkpoint_dir/teacher
    (if it exists, i.e., co-trained). Frozen teachers are reloaded from the
    original model ID since their weights don't change.

    Args:
        checkpoint_dir: Path to the training output directory
        config: Configuration dictionary (for model type routing)
        device: Device to load models on
        tokenizer: Tokenizer
        num_labels: Number of labels for classification tasks

    Returns:
        Tuple of (teacher, student). Teacher is None if not applicable.
    """
    from pathlib import Path

    task = config.get("data", {}).get("task", "").lower()
    translation = is_translation_task(task)
    ocr = is_ocr_task(task)
    teacher_name = config["model"].get("teacher")
    alpha = config.get("distillation", {}).get("alpha", 0.5)
    checkpoint_dir = Path(checkpoint_dir)

    # Load student from checkpoint
    student_path = str(checkpoint_dir)
    if ocr:
        student = _load_vlm(student_path, "pretrained", device)
    elif translation:
        student = _load_causal_lm(student_path, "pretrained", device, tokenizer)
    else:
        student = _load_classifier(student_path, "pretrained", num_labels, device)
    _log_params("Student (checkpoint)", student_path, student)

    # Load teacher
    teacher = None
    if teacher_name is not None and alpha > 0.0:
        teacher_path = checkpoint_dir / "teacher"
        if teacher_path.exists():
            # Co-trained teacher saved alongside student
            teacher_src = str(teacher_path)
        else:
            # Frozen teacher: reload from original model ID
            teacher_src = teacher_name

        if ocr:
            teacher = _load_vlm(teacher_src, "pretrained", device)
        elif translation:
            teacher = _load_causal_lm(teacher_src, "pretrained", device, tokenizer)
        else:
            teacher = _load_classifier(teacher_src, "pretrained", num_labels, device)
        _log_params("Teacher (checkpoint)", teacher_src, teacher)

    return teacher, student



