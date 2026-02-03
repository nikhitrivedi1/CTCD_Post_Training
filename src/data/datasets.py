"""Dataset loaders for GLUE tasks (MNLI, QQP) and Europarl translation."""

from datasets import load_dataset
from transformers import PreTrainedTokenizer


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
        """Format with chat template (if available) or simple prompt."""
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
        else:
            # Fallback to simple prompt for models without chat template
            formatted_text = prompt_template.format(
                source_text=source_text
            ) + "\n" + target_text

        # Tokenize without padding (DataCollator will handle padding dynamically)
        model_inputs = tokenizer(
            formatted_text,
            max_length=max_length,
            truncation=True,
            return_tensors=None
        )

        # For causal LM, labels are same as input_ids (model predicts next token)
        # DataCollatorForSeq2Seq will automatically mask padding with -100
        model_inputs["labels"] = model_inputs["input_ids"].copy()

        return model_inputs

    train_tokenized = train_data.map(format_translation, remove_columns=train_data.column_names)
    val_tokenized = val_data.map(format_translation, remove_columns=val_data.column_names)

    return train_tokenized, val_tokenized, tokenizer


def load_dataset_for_task(config: dict):
    """
    Load dataset based on config specification.

    Args:
        config: Configuration dictionary with 'data' section

    Returns:
        Tuple of (train_dataset, eval_dataset, tokenizer)
    """
    from transformers import AutoTokenizer

    # Determine task type
    task_name = config.get("data", {}).get("task", "").lower()
    teacher_model = config["model"]["teacher"]

    # Load appropriate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(teacher_model)

    # Set padding side for causal LM (translation)
    if "llama" in teacher_model.lower() or task_name == "translation":
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Load dataset based on task
    if task_name == "mnli":
        return load_mnli(tokenizer)
    elif task_name == "qqp":
        return load_qqp(tokenizer)
    elif task_name == "translation" or task_name == "europarl":
        max_samples = config.get("data", {}).get("max_samples", 45715)
        source_lang = config.get("data", {}).get("source_lang", "en")
        target_lang = config.get("data", {}).get("target_lang", "fr")
        return load_europarl(
            tokenizer,
            max_samples=max_samples,
            source_lang=source_lang,
            target_lang=target_lang
        )
    else:
        raise ValueError(f"Unknown task: {task_name}")
