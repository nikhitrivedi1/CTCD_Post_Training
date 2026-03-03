"""Dataset loaders for GLUE tasks (MNLI, QQP, SST-2, CoLA), Europarl translation, and OCR."""

from datasets import concatenate_datasets, load_dataset, load_from_disk
from transformers import PreTrainedTokenizer

from src.tasks import TASKS, is_translation_task, is_ocr_task


def load_mnli(tokenizer: PreTrainedTokenizer, max_length: int = 128):
    """
    Load MNLI (Multi-Genre Natural Language Inference) dataset.

    MNLI is a 3-class sentence entailment task: given a premise and hypothesis,
    classify as entailment, contradiction, or neutral.

    Args:
        tokenizer: HuggingFace tokenizer for text preprocessing
        max_length: Maximum sequence length for tokenization

    Returns:
        Tuple of (train_dataset, eval_dataset, tokenizer)
    """
    dataset = load_dataset("glue", "mnli")

    def preprocess(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

    tokenized = dataset.map(preprocess, batched=True, remove_columns=["premise", "hypothesis", "idx"])

    # Rename label column to labels for HuggingFace Trainer
    tokenized = tokenized.rename_column("label", "labels")

    train_dataset = tokenized["train"]
    # Use matched validation set
    eval_dataset = tokenized["validation_matched"]

    return train_dataset, eval_dataset, tokenizer


def load_qqp(tokenizer: PreTrainedTokenizer, max_length: int = 128):
    """
    Load QQP (Quora Question Pairs) dataset.

    QQP is a binary classification task to determine if two questions
    are semantically equivalent.

    Args:
        tokenizer: HuggingFace tokenizer for text preprocessing
        max_length: Maximum sequence length for tokenization

    Returns:
        Tuple of (train_dataset, eval_dataset, tokenizer)
    """
    dataset = load_dataset("glue", "qqp")

    def preprocess(examples):
        return tokenizer(
            examples["question1"],
            examples["question2"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

    tokenized = dataset.map(preprocess, batched=True, remove_columns=["question1", "question2", "idx"])

    # Rename label column to labels for HuggingFace Trainer
    tokenized = tokenized.rename_column("label", "labels")

    # Filter out examples with -1 labels (unlabeled)
    train_dataset = tokenized["train"].filter(lambda x: x["labels"] != -1)
    eval_dataset = tokenized["validation"].filter(lambda x: x["labels"] != -1)

    return train_dataset, eval_dataset, tokenizer


def load_sst2(tokenizer: PreTrainedTokenizer, max_length: int = 128):
    """
    Load SST-2 (Stanford Sentiment Treebank) dataset.

    SST-2 is a binary sentiment classification task on single sentences.

    Args:
        tokenizer: HuggingFace tokenizer for text preprocessing
        max_length: Maximum sequence length for tokenization

    Returns:
        Tuple of (train_dataset, eval_dataset, tokenizer)
    """
    dataset = load_dataset("glue", "sst2")

    def preprocess(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

    tokenized = dataset.map(preprocess, batched=True, remove_columns=["sentence", "idx"])

    # Rename label column to labels for HuggingFace Trainer
    tokenized = tokenized.rename_column("label", "labels")

    train_dataset = tokenized["train"]
    eval_dataset = tokenized["validation"]

    return train_dataset, eval_dataset, tokenizer


def load_cola(tokenizer: PreTrainedTokenizer, max_length: int = 128):
    """
    Load CoLA (Corpus of Linguistic Acceptability) dataset.

    CoLA is a binary classification task for grammatical acceptability.
    Note: Primary metric is Matthews correlation coefficient (MCC).

    Args:
        tokenizer: HuggingFace tokenizer for text preprocessing
        max_length: Maximum sequence length for tokenization

    Returns:
        Tuple of (train_dataset, eval_dataset, tokenizer)
    """
    dataset = load_dataset("glue", "cola")

    def preprocess(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

    tokenized = dataset.map(preprocess, batched=True, remove_columns=["sentence", "idx"])

    # Rename label column to labels for HuggingFace Trainer
    tokenized = tokenized.rename_column("label", "labels")

    train_dataset = tokenized["train"]
    eval_dataset = tokenized["validation"]

    return train_dataset, eval_dataset, tokenizer


def load_europarl(
    tokenizer: PreTrainedTokenizer,
    max_samples: int = 45715,
    max_length: int = 256,
    source_lang: str = "en",
    target_lang: str = "fr"
):
    """
    Load Europarl translation dataset for any language pair.

    Loads a subset of the Europarl dataset with custom formatting:
    - Formats with chat template (if available) or simple prompt
    - Relies on model's built-in EOS token for stopping
    - Splits into 70/20/10 train/test/val

    Args:
        tokenizer: HuggingFace tokenizer
        max_samples: Number of examples to use (paper uses 45,715)
        max_length: Maximum sequence length
        source_lang: Source language code (e.g., 'en')
        target_lang: Target language code (e.g., 'fr')

    Returns:
        Tuple of (train_dataset, val_dataset, tokenizer)
    """

    # Load Europarl dataset (Helsinki-NLP version with config-based language pairs)
    lang_pair = f"{source_lang}-{target_lang}"
    dataset = load_dataset("Helsinki-NLP/europarl", lang_pair, split="train")

    # Subsample to match paper (45,715 examples)
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    # Split: 70% train, 20% test, 10% val
    splits = dataset.train_test_split(test_size=0.3, seed=42)
    train_data = splits["train"]
    temp_splits = splits["test"].train_test_split(test_size=0.333, seed=42)  # 0.1/0.3 ≈ 0.333
    val_data = temp_splits["train"]  # This is ~20% of total
    test_data = temp_splits["test"]  # This is ~10% of total

    # Language name mapping for prompts
    lang_names = {
        "en": "English", "fr": "French", "de": "German", "es": "Spanish",
        "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "da": "Danish",
        "sv": "Swedish", "fi": "Finnish", "el": "Greek", "cs": "Czech",
        "pl": "Polish", "hu": "Hungarian", "ro": "Romanian", "bg": "Bulgarian"
    }
    source_lang_name = lang_names.get(source_lang, source_lang.upper())
    target_lang_name = lang_names.get(target_lang, target_lang.upper())

    # Prompt template with dynamic language names
    system_message = f"You are a professional translator. Translate the provided text from {source_lang_name} to {target_lang_name}, remaining true to the source text. Do not add any additional commentary or conversational elements to your response."
    prompt_template = f"""{system_message}

{{source_text}}"""

    def format_translation(example):
        """Format with chat template (if available) or simple prompt.

        Applies prompt masking: labels are set to -100 for all tokens before
        the assistant's response (system prompt, source text, chat template
        boilerplate). Only the target translation tokens contribute to the
        CE loss. Note: KL distillation loss operates on all logit positions
        regardless of label masking — this fix scopes the task loss only.
        """
        source_text = example["translation"][source_lang]
        target_text = example["translation"][target_lang]

        # Try using chat template if available (Llama, Mistral, etc.)
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": source_text},
                {"role": "assistant", "content": target_text}
            ]
            # Format with chat template (returns string)
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            # Also format the prompt-only portion (without target text) to find
            # where the assistant's content begins in the token sequence.
            # add_generation_prompt=True appends the assistant header so the
            # boundary sits right before the target translation tokens.
            prompt_messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": source_text},
            ]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback to simple prompt for models without chat template
            prompt_text = prompt_template.format(
                source_text=source_text
            ) + "\n"
            formatted_text = prompt_text + target_text

        # Tokenize without padding (DataCollator will handle padding dynamically)
        model_inputs = tokenizer(
            formatted_text,
            max_length=max_length,
            truncation=True,
            return_tensors=None
        )

        # Tokenize the prompt portion so we can find its boundary in the full
        # sequence. We take the length of the longest common prefix between
        # prompt_ids and input_ids so that only tokens we are certain belong
        # to the prompt are masked.
        # This handles BPE prefix inconsistency in both directions:
        # over-tokenization (prompt_ids longer) and under-tokenization
        # (prompt_ids shorter due to cross-boundary merges).
        prompt_ids = tokenizer(
            prompt_text,
            max_length=max_length,
            truncation=True,
            return_tensors=None,
        )["input_ids"]
        input_ids = model_inputs["input_ids"]
        max_prefix = min(len(prompt_ids), len(input_ids))
        prompt_len = 0
        for i in range(max_prefix):
            if prompt_ids[i] == input_ids[i]:
                prompt_len += 1
            else:
                break

        # Create labels with prompt masking:
        # - tokens 0..prompt_len-1 → -100 (system prompt, source text, headers)
        # - tokens prompt_len..end → input_ids (target translation + EOS)
        # DataCollatorForSeq2Seq will additionally mask padding with -100
        labels = input_ids.copy()
        labels[:prompt_len] = [-100] * prompt_len
        model_inputs["labels"] = labels

        return model_inputs

    train_tokenized = train_data.map(format_translation, remove_columns=train_data.column_names)
    val_tokenized = val_data.map(format_translation, remove_columns=val_data.column_names)

    return train_tokenized, val_tokenized, tokenizer


def load_ocr(
    processor,
    data_dir: str = "data/ocr/documents",
    max_train_samples: int | None = None,
    max_length: int = 1024,
    include_reasoning: bool = False,
):
    """
    Load pre-processed OCR dataset (images + text pairs).

    Expects a local dataset created by scripts/prepare_ocr_data.py with
    'image' (PIL) and 'text' (markdown) columns.

    Args:
        processor: AutoProcessor for Qwen VL models (handles image + text)
        data_dir: Path to pre-processed dataset directory (with train/ and eval/ subdirs)
        max_train_samples: Maximum training samples (truncate before preprocessing)
        max_length: Maximum token sequence length
        include_reasoning: If True, use 'reasoning_and_output' column instead of 'text'

    Returns:
        Tuple of (train_dataset, eval_dataset, processor)
    """
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)

    data_path = Path(data_dir)
    logger.info(f"Loading pre-processed dataset from {data_path}...")

    if (data_path / "train").exists():
        # Single dataset directory (e.g., data/ocr/documents/train/)
        train_dataset = load_from_disk(str(data_path / "train"))
        eval_dataset = load_from_disk(str(data_path / "eval")) if (data_path / "eval").exists() else None
    else:
        # Multi-subset directory (e.g., data/ocr/olmocr-mix-full/{00_documents,01_books,...}/train/)
        train_parts, eval_parts = [], []
        for subset_dir in sorted(data_path.iterdir()):
            if not subset_dir.is_dir() or not (subset_dir / "train").exists():
                continue
            logger.info(f"  Loading subset: {subset_dir.name}")
            train_parts.append(load_from_disk(str(subset_dir / "train")))
            if (subset_dir / "eval").exists():
                eval_parts.append(load_from_disk(str(subset_dir / "eval")))
        if not train_parts:
            raise FileNotFoundError(f"No train splits found in {data_path}")
        train_dataset = concatenate_datasets(train_parts)
        eval_dataset = concatenate_datasets(eval_parts) if eval_parts else None

    if max_train_samples and len(train_dataset) > max_train_samples:
        train_dataset = train_dataset.select(range(max_train_samples))

    eval_count = len(eval_dataset) if eval_dataset is not None else 0
    logger.info(f"OCR dataset loaded: {len(train_dataset)} train, {eval_count} eval")

    text_column = "reasoning_and_output" if include_reasoning else "text"

    # OCR prompt — matches olmOCR's document extraction prompt
    system_prompt = (
        "Below is an image of a document page. Extract all text content "
        "and output it in clean markdown format, preserving the document's "
        "structure (headings, lists, tables, etc.)."
    )

    def preprocess(example):
        """Format chat messages; image processing deferred to collation time.

        We only apply the chat template here to produce formatted text.
        The VLMDataCollator handles tokenization and image processing,
        avoiding Arrow serialization issues with variable-size pixel tensors.
        """
        text = example[text_column] or ""

        # Truncate the OCR text to limit sequence length.
        # Conservative estimate: ~1 token per 4 chars.
        max_text_chars = max_length * 4
        if len(text) > max_text_chars:
            text = text[:max_text_chars]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt},
                    {"type": "image", "image": example["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": text},
                ],
            },
        ]

        # Apply chat template to get text with image placeholders
        formatted_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        return {"formatted_text": formatted_text}

    # Keep image column, replace text columns with formatted_text
    train_columns = train_dataset.column_names
    eval_columns = eval_dataset.column_names if eval_dataset is not None else []

    logger.info(f"Formatting train split ({len(train_dataset)} samples) with chat template...")
    train_dataset = train_dataset.map(
        preprocess,
        remove_columns=[c for c in train_columns if c != "image"],
    )
    if eval_dataset is not None:
        logger.info(f"Formatting eval split ({len(eval_dataset)} samples) with chat template...")
        eval_dataset = eval_dataset.map(
            preprocess,
            remove_columns=[c for c in eval_columns if c != "image"],
        )

    return train_dataset, eval_dataset, processor


def load_dataset_for_task(config: dict):
    """
    Load dataset based on config specification.

    Args:
        config: Configuration dictionary with 'data' section

    Returns:
        Tuple of (train_dataset, eval_dataset, tokenizer)
    """
    from transformers import AutoTokenizer, AutoProcessor

    # Determine task type
    task_name = config.get("data", {}).get("task", "").lower()

    # Validate task exists in registry
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Supported: {list(TASKS.keys())}")

    # OCR tasks use AutoProcessor (handles both image and text)
    if is_ocr_task(task_name):
        import logging
        _logger = logging.getLogger(__name__)
        processor_name = config["model"].get("processor", config["model"]["student"])
        _logger.info(f"Loading processor: {processor_name}...")
        processor = AutoProcessor.from_pretrained(processor_name)
        return load_ocr(
            processor=processor,
            data_dir=config.get("data", {}).get("data_dir", "data/ocr/documents"),
            max_train_samples=config.get("data", {}).get("max_train_samples"),
            max_length=config.get("data", {}).get("max_length", 1024),
            include_reasoning=config.get("data", {}).get("include_reasoning", False),
        )

    teacher_model = config["model"]["teacher"]

    # Load appropriate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(teacher_model)

    # Set padding side for causal LM (translation)
    if "llama" in teacher_model.lower() or is_translation_task(task_name):
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Load dataset based on task
    # Note: Each task has unique preprocessing, so explicit dispatch is needed
    if task_name == "mnli":
        return load_mnli(tokenizer)
    elif task_name == "qqp":
        return load_qqp(tokenizer)
    elif task_name == "sst2":
        return load_sst2(tokenizer)
    elif task_name == "cola":
        return load_cola(tokenizer)
    elif task_name == "europarl":
        max_samples = config.get("data", {}).get("max_samples", 45715)
        source_lang = config.get("data", {}).get("source_lang", "en")
        target_lang = config.get("data", {}).get("target_lang", "fr")
        return load_europarl(
            tokenizer,
            max_samples=max_samples,
            source_lang=source_lang,
            target_lang=target_lang
        )
