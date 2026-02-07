"""Shared utilities for analysis toolkit."""

import inspect
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from datasets import load_dataset

# Re-export label utilities from shared location
from utils.labels import get_label_map, CLASS_LABELS, LABEL_MAP_REGISTRY

logger = logging.getLogger(__name__)


# Task configurations
TASK_CONFIG = {
    "mnli": {
        "dataset": ("glue", "mnli"),
        "text_cols": ["premise", "hypothesis"],
        "label_col": "label",
        "num_labels": 3,
        "eval_split": "validation_matched",
    },
    "qqp": {
        "dataset": ("glue", "qqp"),
        "text_cols": ["question1", "question2"],
        "label_col": "label",
        "num_labels": 2,
        "eval_split": "validation",
    },
    "sst2": {
        "dataset": ("glue", "sst2"),
        "text_cols": ["sentence"],
        "label_col": "label",
        "num_labels": 2,
        "eval_split": "validation",
    },
    "cola": {
        "dataset": ("glue", "cola"),
        "text_cols": ["sentence"],
        "label_col": "label",
        "num_labels": 2,
        "eval_split": "validation",
    },
}


def load_model(model_path: str, device: str = "cuda"):
    """
    Load a model for analysis.

    Args:
        model_path: HuggingFace model name or local checkpoint path
        device: Device to load on

    Returns:
        Tuple of (model, tokenizer, num_layers)
    """
    checkpoint_path = Path(model_path)

    if checkpoint_path.exists():
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path, config=config)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = model.to(device)
    model.eval()

    if hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    else:
        num_layers = model.config.n_layers

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Loaded {model_path}: {num_layers} layers, {param_count:.1f}M params")

    return model, tokenizer, num_layers


def load_task_data(task: str, max_samples: int = None):
    """
    Load task data for analysis.

    Args:
        task: Task name (mnli, qqp, sst2, cola)
        max_samples: Limit samples

    Returns:
        Tuple of (texts, labels, task_config)
    """
    if task not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {task}. Supported: {list(TASK_CONFIG.keys())}")

    task_cfg = TASK_CONFIG[task]
    dataset = load_dataset(*task_cfg["dataset"])
    eval_data = dataset[task_cfg["eval_split"]]

    text_cols = task_cfg["text_cols"]
    if len(text_cols) == 2:
        texts = [(ex[text_cols[0]], ex[text_cols[1]]) for ex in eval_data]
    else:
        texts = [ex[text_cols[0]] for ex in eval_data]

    labels = [ex[task_cfg["label_col"]] for ex in eval_data]

    # Filter invalid labels
    valid_indices = [i for i, label in enumerate(labels) if label >= 0]
    texts = [texts[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]

    if max_samples and len(texts) > max_samples:
        texts = texts[:max_samples]
        labels = labels[:max_samples]

    logger.info(f"Loaded {len(texts)} samples for {task}")

    return texts, labels, task_cfg


def extract_hidden_states(
    model,
    tokenizer,
    texts: list,
    batch_size: int = 32,
    max_length: int = 128,
    device: str = "cuda",
    show_progress: bool = True,
):
    """
    Extract [CLS] hidden states from all layers.

    Args:
        model: Transformer model
        tokenizer: Tokenizer
        texts: List of text inputs (tuples for pair tasks)
        batch_size: Batch size for extraction
        max_length: Max sequence length
        device: Device
        show_progress: Show progress bar

    Returns:
        Dict of layer_idx -> np.array[N, hidden_dim]
    """
    # Tokenize
    if isinstance(texts[0], tuple):
        encodings = tokenizer(
            [t[0] for t in texts],
            [t[1] for t in texts],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
    else:
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

    # Build tensor list, including token_type_ids if model accepts them
    forward_params = set(inspect.signature(model.forward).parameters.keys())
    tensors = [encodings["input_ids"], encodings["attention_mask"]]
    has_token_type_ids = "token_type_ids" in forward_params and "token_type_ids" in encodings
    if has_token_type_ids:
        tensors.append(encodings["token_type_ids"])

    dataset = TensorDataset(*tensors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    num_layers = model.config.num_hidden_layers
    layer_hiddens = {i: [] for i in range(num_layers + 1)}  # +1 for embeddings

    iterator = tqdm(dataloader, desc="Extracting") if show_progress else dataloader

    with torch.no_grad():
        for batch in iterator:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if has_token_type_ids:
                model_inputs["token_type_ids"] = batch[2].to(device)

            outputs = model(
                **model_inputs,
                output_hidden_states=True,
            )

            for layer_idx, hidden in enumerate(outputs.hidden_states):
                cls_hidden = hidden[:, 0, :].cpu().numpy()
                layer_hiddens[layer_idx].append(cls_hidden)

    for layer_idx in layer_hiddens:
        layer_hiddens[layer_idx] = np.concatenate(layer_hiddens[layer_idx], axis=0)

    return layer_hiddens


def extract_attention_weights(
    model,
    tokenizer,
    texts: list,
    batch_size: int = 32,
    max_length: int = 128,
    device: str = "cuda",
    show_progress: bool = True,
):
    """
    Extract attention weights from all layers.

    Args:
        model: Transformer model
        tokenizer: Tokenizer
        texts: List of text inputs
        batch_size: Batch size
        max_length: Max sequence length
        device: Device
        show_progress: Show progress bar

    Returns:
        Tuple of (layer_attentions, attention_masks) where:
            layer_attentions: Dict of layer_idx -> np.array[N, num_heads, seq, seq]
            attention_masks: np.array[N, seq] indicating non-padding tokens
    """
    if isinstance(texts[0], tuple):
        encodings = tokenizer(
            [t[0] for t in texts],
            [t[1] for t in texts],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
    else:
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

    # Build tensor list, including token_type_ids if model accepts them
    forward_params = set(inspect.signature(model.forward).parameters.keys())
    tensors = [encodings["input_ids"], encodings["attention_mask"]]
    has_token_type_ids = "token_type_ids" in forward_params and "token_type_ids" in encodings
    if has_token_type_ids:
        tensors.append(encodings["token_type_ids"])

    dataset = TensorDataset(*tensors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    num_layers = model.config.num_hidden_layers
    layer_attentions = {i: [] for i in range(num_layers)}
    all_masks = []

    iterator = tqdm(dataloader, desc="Extracting attention") if show_progress else dataloader

    with torch.no_grad():
        for batch in iterator:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if has_token_type_ids:
                model_inputs["token_type_ids"] = batch[2].to(device)

            outputs = model(
                **model_inputs,
                output_attentions=True,
            )

            all_masks.append(attention_mask.cpu().numpy())
            for layer_idx, attn in enumerate(outputs.attentions):
                layer_attentions[layer_idx].append(attn.cpu().numpy())

    for layer_idx in layer_attentions:
        layer_attentions[layer_idx] = np.concatenate(layer_attentions[layer_idx], axis=0)
    attention_masks = np.concatenate(all_masks, axis=0)

    return layer_attentions, attention_masks
