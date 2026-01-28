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


def load_europarl(tokenizer: PreTrainedTokenizer, max_samples: int = 45715, max_length: int = 256):
    """
    Load Europarl EN-FR translation dataset.

    Loads a subset of the Europarl dataset with custom formatting:
    - Adds a custom <|end_of_translation|> token to prevent hallucination
    - Formats with a professional translator prompt template
    - Splits into 70/20/10 train/test/val

    Args:
        tokenizer: HuggingFace tokenizer (should be Llama tokenizer)
        max_samples: Number of examples to use (paper uses 45,715)
        max_length: Maximum sequence length

    Returns:
        Tuple of (train_dataset, val_dataset, tokenizer)
    """
    # Add custom end token to prevent hallucination in small models
    special_token = "<|end_of_translation|>"
    if special_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [special_token]})

    # Load Europarl dataset
    dataset = load_dataset("europarl_bilingual", lang1="en", lang2="fr", split="train")

    # Subsample to match paper (45,715 examples)
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    # Split: 70% train, 20% test, 10% val
    splits = dataset.train_test_split(test_size=0.3, seed=42)
    train_data = splits["train"]
    temp_splits = splits["test"].train_test_split(test_size=0.333, seed=42)  # 0.1/0.3 ≈ 0.333
    val_data = temp_splits["train"]  # This is ~20% of total
    test_data = temp_splits["test"]  # This is ~10% of total

    # Prompt template from paper
    prompt_template = """You are a professional translator. Translate the provided text from English to French, remaining true to the source text. Do not add any additional commentary or conversational elements to your response.

{source_text}

{special_token}"""

    def format_translation(example):
        """Format with prompt and add end token."""
        source_text = example["translation"]["en"]
        target_text = example["translation"]["fr"]

        # Create input with prompt
        input_text = prompt_template.format(
            source_text=source_text,
            special_token=special_token
        )

        # Target includes the French text plus end token
        full_target = target_text + " " + special_token

        # Tokenize (left padding for causal LM)
        model_inputs = tokenizer(
            input_text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None
        )

        # Tokenize labels
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                full_target,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors=None
            )

        model_inputs["labels"] = labels["input_ids"]

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
        return load_europarl(tokenizer, max_samples=max_samples)
    else:
        raise ValueError(f"Unknown task: {task_name}")
