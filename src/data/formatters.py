"""Dataset formatters for SFT and RL training.

Handlers load raw data. Formatters shape it for the training mode:
  format_for_sft() - chat-templated text, ready for tokenization
  format_for_rl()  - prompt message dicts + ground_truth/tests for reward computation

These are thin wrappers that bridge handler output to trainer input.
"""

import logging

from datasets import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RL formatting
# ---------------------------------------------------------------------------


def format_for_rl(
    dataset: Dataset,
    task_type: str,
    system_prompt: str | None = None,
) -> Dataset:
    """Shape raw samples into RL (GRPO) format.

    RL format:
      - 'prompt': list of message dicts (unfilled, for model generation)
      - 'ground_truth' or 'tests': reference for reward computation
      - 'image': PIL Image (OCR tasks only)

    Args:
        dataset: Raw dataset from handler.load() with 'ground_truth'/'tests' etc.
        task_type: One of "ocr", "translation", "classification".
        system_prompt: Override system prompt (uses default per task_type if None).

    Returns:
        Dataset with RL-compatible columns.
    """
    if task_type == "ocr":
        return _format_ocr_rl(dataset, system_prompt)
    elif task_type == "translation":
        return _format_translation_rl(dataset)
    else:
        raise ValueError(f"RL formatting not supported for task type '{task_type}'")


def _format_ocr_rl(dataset: Dataset, system_prompt: str | None = None) -> Dataset:
    """Format OCR dataset for RL training.

    Args:
        dataset: Raw dataset with 'image' and 'ground_truth' (or 'tests') columns.
        system_prompt: Override OCR system prompt.

    Returns:
        Dataset with 'prompt', 'image', and 'ground_truth'/'tests' columns.
    """
    from src.data.ocr_utils import OCR_SYSTEM_PROMPT
    prompt_text = system_prompt or OCR_SYSTEM_PROMPT

    # Detect whether this is a ground_truth or tests-based dataset
    has_tests = "tests" in dataset.column_names
    keep_col = "tests" if has_tests else "ground_truth"

    def format_fn(example):
        return {
            "prompt": [{"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image"},
            ]}],
        }

    remove_cols = [c for c in dataset.column_names if c not in {"image", keep_col}]
    return dataset.map(format_fn, remove_columns=remove_cols)


def _format_translation_rl(dataset: Dataset) -> Dataset:
    """Format translation dataset for RL training.

    Args:
        dataset: Raw dataset with 'source_text', 'ground_truth', 'system_message'.

    Returns:
        Dataset with 'prompt' and 'ground_truth' columns.
    """
    def format_fn(example):
        return {
            "prompt": [
                {"role": "system", "content": example["system_message"]},
                {"role": "user", "content": example["source_text"]},
            ],
        }

    remove_cols = [c for c in dataset.column_names if c not in {"ground_truth"}]
    return dataset.map(format_fn, remove_columns=remove_cols)


# ---------------------------------------------------------------------------
# SFT formatting
# ---------------------------------------------------------------------------


def format_for_sft(
    dataset: Dataset,
    task_type: str,
    tokenizer_or_processor=None,
    max_length: int = 1024,
    system_prompt: str | None = None,
) -> Dataset:
    """Shape raw samples into SFT format.

    SFT format varies by task:
      - OCR: 'formatted_text' (chat-templated) + 'image' (PIL) -> VLMDataCollator
      - Translation: 'input_ids', 'attention_mask', 'labels' (prompt-masked)
      - Classification: 'input_ids', 'attention_mask', 'labels' (tokenized)

    Args:
        dataset: Raw dataset from handler.load().
        task_type: One of "ocr", "translation", "classification".
        tokenizer_or_processor: Tokenizer (text tasks) or AutoProcessor (VLM tasks).
        max_length: Maximum sequence length.
        system_prompt: Override system prompt.

    Returns:
        Dataset formatted for SFT training.
    """
    if task_type == "ocr":
        return _format_ocr_sft(dataset, tokenizer_or_processor, max_length, system_prompt)
    elif task_type == "translation":
        return _format_translation_sft(dataset, tokenizer_or_processor, max_length)
    elif task_type == "classification":
        return _format_classification_sft(dataset, tokenizer_or_processor, max_length)
    else:
        raise ValueError(f"SFT formatting not supported for task type '{task_type}'")


def _format_ocr_sft(dataset: Dataset, processor, max_length: int, system_prompt: str | None) -> Dataset:
    """Format OCR dataset for SFT training.

    Args:
        dataset: Raw dataset with 'image' and 'ground_truth' columns.
        processor: AutoProcessor for VLM models.
        max_length: Max sequence length (used for text truncation).
        system_prompt: Override OCR system prompt.

    Returns:
        Dataset with 'formatted_text' and 'image' columns.
    """
    from src.data.ocr_utils import OCR_SYSTEM_PROMPT
    prompt_text = system_prompt or OCR_SYSTEM_PROMPT

    def preprocess(example):
        text = example.get("ground_truth") or ""
        max_text_chars = max_length * 4
        if len(text) > max_text_chars:
            text = text[:max_text_chars]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image", "image": example["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            },
        ]
        formatted_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"formatted_text": formatted_text}

    return dataset.map(
        preprocess,
        remove_columns=[c for c in dataset.column_names if c != "image"],
    )


def _format_translation_sft(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    """Format translation dataset for SFT training with prompt masking.

    Args:
        dataset: Raw dataset with 'source_text', 'ground_truth', 'system_message'.
        tokenizer: HuggingFace tokenizer.
        max_length: Max sequence length.

    Returns:
        Tokenized dataset with prompt-masked labels.
    """
    def format_fn(example):
        source_text = example["source_text"]
        target_text = example["ground_truth"]
        system_message = example["system_message"]

        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": source_text},
                {"role": "assistant", "content": target_text},
            ]
            formatted_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            prompt_messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": source_text},
            ]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = f"{system_message}\n\n{source_text}\n"
            formatted_text = prompt_text + target_text

        model_inputs = tokenizer(
            formatted_text, max_length=max_length, truncation=True, return_tensors=None
        )

        # Prompt masking: find longest common prefix between prompt and full tokenization
        prompt_ids = tokenizer(
            prompt_text, max_length=max_length, truncation=True, return_tensors=None
        )["input_ids"]
        input_ids = model_inputs["input_ids"]
        max_prefix = min(len(prompt_ids), len(input_ids))
        prompt_len = 0
        for i in range(max_prefix):
            if prompt_ids[i] == input_ids[i]:
                prompt_len += 1
            else:
                break

        labels = input_ids.copy()
        labels[:prompt_len] = [-100] * prompt_len
        model_inputs["labels"] = labels
        return model_inputs

    return dataset.map(format_fn, remove_columns=dataset.column_names)


def _format_classification_sft(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    """Format classification dataset for SFT training.

    GLUE datasets are already loaded with 'labels' column from the handler.
    This just tokenizes the text columns.

    Args:
        dataset: Raw dataset with text columns and 'labels'.
        tokenizer: HuggingFace tokenizer.
        max_length: Max sequence length.

    Returns:
        Tokenized dataset.
    """
    # Detect text columns (everything except 'labels')
    text_cols = [c for c in dataset.column_names if c != "labels"]

    if len(text_cols) == 2:
        # Sentence pair task (MNLI, QQP)
        col_a, col_b = text_cols[0], text_cols[1]

        def preprocess(examples):
            return tokenizer(
                examples[col_a], examples[col_b],
                truncation=True, max_length=max_length, padding="max_length",
            )
    else:
        # Single sentence task (SST-2, CoLA)
        col = text_cols[0]

        def preprocess(examples):
            return tokenizer(
                examples[col],
                truncation=True, max_length=max_length, padding="max_length",
            )

    return dataset.map(preprocess, batched=True, remove_columns=text_cols)
