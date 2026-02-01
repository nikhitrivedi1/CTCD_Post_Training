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

logger = logging.getLogger(__name__)


def load_teacher_student(config: dict, device: str, tokenizer: PreTrainedTokenizer, num_labels: int = None):
    """
    Load teacher and student models based on config.

    Handles both:
    - BERT/DistilBERT for classification (GLUE tasks)
    - Llama 3B/1B for translation (with optional LoRA)

    Args:
        config: Configuration dictionary
        device: Device to load models on ('cuda', 'mps', 'cpu')
        tokenizer: Tokenizer (may have been resized with custom tokens)
        num_labels: Number of labels for classification (required for GLUE)

    Returns:
        Tuple of (teacher_model, student_model)
    """
    teacher_name = config["model"]["teacher"]
    student_name = config["model"]["student"]
    task = config.get("data", {}).get("task", "").lower()

    # Determine model type
    if "llama" in teacher_name.lower() or task == "translation":
        return _load_translation_models(config, device, tokenizer)
    else:
        return _load_classification_models(config, device, tokenizer, num_labels)


def _load_classification_models(config: dict, device: str, tokenizer: PreTrainedTokenizer, num_labels: int):
    """
    Load BERT/DistilBERT models for classification tasks.

    Args:
        config: Configuration dictionary
        device: Device to load models on
        tokenizer: Tokenizer
        num_labels: Number of classification labels

    Returns:
        Tuple of (teacher, student)
    """
    teacher_name = config["model"]["teacher"]
    student_name = config["model"]["student"]
    student_init = config["model"].get("student_init", "pretrained")

    # Load teacher (pretrained)
    teacher = AutoModelForSequenceClassification.from_pretrained(
        teacher_name,
        num_labels=num_labels
    ).to(device)
    teacher.eval()  # Always in eval mode

    # Load student
    if student_init == "fresh":
        # Fresh initialization: load config, create new model with random weights
        logger.info(f"Initializing fresh {student_name} model...")
        student_config = AutoConfig.from_pretrained(student_name)
        student_config.num_labels = num_labels

        # Import the appropriate model class
        if "distilbert" in student_name.lower():
            from transformers import DistilBertForSequenceClassification
            student = DistilBertForSequenceClassification(student_config)
        else:
            student = AutoModelForSequenceClassification.from_config(student_config)

        student = student.to(device)
    else:
        # Load pretrained student
        student = AutoModelForSequenceClassification.from_pretrained(
            student_name,
            num_labels=num_labels
        ).to(device)

    logger.info(f"Teacher: {teacher_name} ({sum(p.numel() for p in teacher.parameters()) / 1e6:.1f}M params)")
    logger.info(f"Student: {student_name} ({sum(p.numel() for p in student.parameters()) / 1e6:.1f}M params)")

    return teacher, student


def _load_translation_models(config: dict, device: str, tokenizer: PreTrainedTokenizer):
    """
    Load Llama 3B/1B models for translation with LoRA support.

    Args:
        config: Configuration dictionary
        device: Device to load models on
        tokenizer: Tokenizer (may have custom tokens)

    Returns:
        Tuple of (teacher, student)
    """
    teacher_name = config["model"]["teacher"]
    student_name = config["model"]["student"]
    student_init = config["model"].get("student_init", "pretrained")
    use_lora = config.get("training", {}).get("use_lora", False)

    # Load teacher (pretrained, fp16)
    logger.info(f"Loading teacher: {teacher_name}...")
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None
    )

    # Resize embeddings if custom tokens added
    if len(tokenizer) > teacher.config.vocab_size:
        logger.info(f"Resizing teacher embeddings: {teacher.config.vocab_size} -> {len(tokenizer)}")
        teacher.resize_token_embeddings(len(tokenizer))

    teacher.eval()

    # Load student
    logger.info(f"Loading student: {student_name}...")
    if student_init == "fresh":
        # Fresh initialization
        student_config = AutoConfig.from_pretrained(student_name)
        student = AutoModelForCausalLM.from_config(student_config)
        student = student.to(torch.float16 if device == "cuda" else torch.float32)
        if device == "cuda":
            student = student.to(device)
    else:
        student = AutoModelForCausalLM.from_pretrained(
            student_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device if device == "cuda" else None
        )

    # Resize student embeddings
    if len(tokenizer) > student.config.vocab_size:
        logger.info(f"Resizing student embeddings: {student.config.vocab_size} -> {len(tokenizer)}")
        student.resize_token_embeddings(len(tokenizer))

    # Apply LoRA to student if specified
    if use_lora:
        logger.info("Applying LoRA to student model...")
        lora_config = LoraConfig(
            r=config["training"]["lora"]["r"],
            lora_alpha=config["training"]["lora"]["alpha"],
            target_modules=config["training"]["lora"]["target_modules"],
            lora_dropout=config["training"]["lora"]["dropout"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        student = get_peft_model(student, lora_config)
        student.print_trainable_parameters()

    logger.info(f"Teacher: {teacher_name} ({sum(p.numel() for p in teacher.parameters()) / 1e6:.1f}M params)")
    if use_lora:
        trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        total = sum(p.numel() for p in student.parameters())
        logger.info(f"Student: {student_name} ({trainable / 1e6:.1f}M trainable / {total / 1e6:.1f}M total params)")
    else:
        logger.info(f"Student: {student_name} ({sum(p.numel() for p in student.parameters()) / 1e6:.1f}M params)")

    return teacher, student
