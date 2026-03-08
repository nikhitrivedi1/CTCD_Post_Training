"""Dataset loaders for RL (GRPO) training.

GRPO needs raw prompts (not tokenized input_ids). These loaders reuse the
same underlying datasets as the supervised loaders but format them differently:

- 'prompt' column: the input text/messages for generation
- 'ground_truth' column: the reference output for reward computation

The GRPOTrainer handles tokenization internally during generation.
"""

import logging

from datasets import load_dataset, load_from_disk, concatenate_datasets

logger = logging.getLogger(__name__)


def load_europarl_rl(
    max_samples: int = 45715,
    source_lang: str = "en",
    target_lang: str = "fr",
):
    """Load Europarl for GRPO training (prompts + ground truth).

    Args:
        max_samples: Number of examples to use.
        source_lang: Source language code.
        target_lang: Target language code.

    Returns:
        Tuple of (train_dataset, eval_dataset).
    """
    lang_pair = f"{source_lang}-{target_lang}"
    dataset = load_dataset("Helsinki-NLP/europarl", lang_pair, split="train")

    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    # Same split ratios as supervised loader: 70/20/10
    splits = dataset.train_test_split(test_size=0.3, seed=42)
    train_data = splits["train"]
    temp_splits = splits["test"].train_test_split(test_size=0.333, seed=42)
    val_data = temp_splits["train"]

    lang_names = {
        "en": "English", "fr": "French", "de": "German", "es": "Spanish",
        "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "da": "Danish",
        "sv": "Swedish", "fi": "Finnish", "el": "Greek", "cs": "Czech",
        "pl": "Polish", "hu": "Hungarian", "ro": "Romanian", "bg": "Bulgarian",
    }
    source_name = lang_names.get(source_lang, source_lang.upper())
    target_name = lang_names.get(target_lang, target_lang.upper())

    system_message = (
        f"You are a professional translator. Translate the provided text from "
        f"{source_name} to {target_name}, remaining true to the source text. "
        f"Do not add any additional commentary or conversational elements to your response."
    )

    def format_for_rl(example):
        source_text = example["translation"][source_lang]
        target_text = example["translation"][target_lang]
        prompt = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": source_text},
        ]
        return {"prompt": prompt, "ground_truth": target_text}

    train_dataset = train_data.map(
        format_for_rl,
        remove_columns=train_data.column_names,
    )
    eval_dataset = val_data.map(
        format_for_rl,
        remove_columns=val_data.column_names,
    )

    logger.info(f"Europarl RL dataset: {len(train_dataset)} train, {len(eval_dataset)} eval")
    return train_dataset, eval_dataset


def load_ocr_rl(
    data_dir: str = "data/ocr/documents",
    max_train_samples: int | None = None,
    include_reasoning: bool = False,
):
    """Load OCR dataset for GRPO training (prompts + ground truth).

    Args:
        data_dir: Path to pre-processed dataset directory.
        max_train_samples: Maximum training samples.
        include_reasoning: If True, use reasoning_and_output column.

    Returns:
        Tuple of (train_dataset, eval_dataset).
    """
    from pathlib import Path

    data_path = Path(data_dir)

    if (data_path / "train").exists():
        train_dataset = load_from_disk(str(data_path / "train"))
        eval_dataset = load_from_disk(str(data_path / "eval")) if (data_path / "eval").exists() else None
    else:
        train_parts, eval_parts = [], []
        for subset_dir in sorted(data_path.iterdir()):
            if not subset_dir.is_dir() or not (subset_dir / "train").exists():
                continue
            train_parts.append(load_from_disk(str(subset_dir / "train")))
            if (subset_dir / "eval").exists():
                eval_parts.append(load_from_disk(str(subset_dir / "eval")))
        if not train_parts:
            raise FileNotFoundError(f"No train splits found in {data_path}")
        train_dataset = concatenate_datasets(train_parts)
        eval_dataset = concatenate_datasets(eval_parts) if eval_parts else None

    if max_train_samples and len(train_dataset) > max_train_samples:
        train_dataset = train_dataset.select(range(max_train_samples))

    text_column = "reasoning_and_output" if include_reasoning else "text"

    from src.data.ocr_utils import OCR_SYSTEM_PROMPT

    def format_for_rl(example):
        text = example.get(text_column) or ""
        prompt = [{"role": "user", "content": [
            {"type": "text", "text": OCR_SYSTEM_PROMPT},
            {"type": "image"},
        ]}]
        return {"prompt": prompt, "ground_truth": text}

    train_columns = train_dataset.column_names
    train_dataset = train_dataset.map(
        format_for_rl,
        remove_columns=[c for c in train_columns if c != "image"],
    )

    if eval_dataset is not None:
        eval_columns = eval_dataset.column_names
        eval_dataset = eval_dataset.map(
            format_for_rl,
            remove_columns=[c for c in eval_columns if c != "image"],
        )

    eval_count = len(eval_dataset) if eval_dataset is not None else 0
    logger.info(f"OCR RL dataset: {len(train_dataset)} train, {eval_count} eval")
    return train_dataset, eval_dataset


def load_synthmix_rl(
    data_dir: str = "data/ocr/synthmix",
    max_train_samples: int | None = None,
    eval_fraction: float = 0.1,
):
    """Load olmOCR-synthmix for GRPO training with unit test rewards.

    Groups bench_data/*.jsonl tests by (pdf, page), pairs with pre-rendered
    images from images/ directory. Each sample has a prompt, image, and list
    of test assertions for the unit_test reward function.

    Args:
        data_dir: Path to synthmix dataset root (with bench_data/ and images/).
        max_train_samples: Maximum training samples.
        eval_fraction: Fraction of samples to hold out for eval.

    Returns:
        Tuple of (train_dataset, eval_dataset).
    """
    import json
    import glob as globmod
    from pathlib import Path
    from collections import defaultdict
    from datasets import Dataset
    from PIL import Image

    data_path = Path(data_dir)
    bench_dir = data_path / "bench_data"
    images_dir = data_path / "images"

    if not bench_dir.exists():
        raise FileNotFoundError(
            f"bench_data/ not found in {data_dir}. Download with:\n"
            f"  python -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('allenai/olmOCR-synthmix-1025', local_dir='{data_dir}', repo_type='dataset')\""
        )
    if not images_dir.exists():
        raise FileNotFoundError(
            f"images/ not found in {data_dir}. Run:\n"
            f"  python scripts/data/convert_synthmix.py --data_dir {data_dir}"
        )

    # Group tests by (pdf, page)
    tests_by_page = defaultdict(list)
    for jsonl_path in sorted(globmod.glob(str(bench_dir / "*.jsonl"))):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                test = json.loads(line)
                key = (test["pdf"], test["page"])
                tests_by_page[key].append(test)

    from src.data.ocr_utils import OCR_SYSTEM_PROMPT, pdf_image_name

    # Build samples: each (pdf, page) with its image and tests
    samples = []
    missing_images = 0
    for (pdf_rel, page_num), test_list in sorted(tests_by_page.items()):
        image_path = images_dir / pdf_image_name(pdf_rel, page_num)

        if not image_path.exists():
            missing_images += 1
            continue

        with Image.open(image_path) as im:
            image = im.convert("RGB")
            image.load()
        samples.append({
            "prompt": [{"role": "user", "content": [
                {"type": "text", "text": OCR_SYSTEM_PROMPT},
                {"type": "image"},
            ]}],
            "image": image,
            "tests": test_list,
        })

    if missing_images > 0:
        logger.warning(f"Skipped {missing_images} pages with missing images")
    logger.info(f"Loaded {len(samples)} pages with {sum(len(s['tests']) for s in samples)} total tests")

    if not samples:
        raise ValueError(f"No valid samples found in {data_dir}")

    # Split into train/eval
    dataset = Dataset.from_list(samples)
    splits = dataset.train_test_split(test_size=eval_fraction, seed=42)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]

    if max_train_samples and len(train_dataset) > max_train_samples:
        train_dataset = train_dataset.select(range(max_train_samples))

    logger.info(f"Synthmix RL dataset: {len(train_dataset)} train, {len(eval_dataset)} eval")
    return train_dataset, eval_dataset


def load_omnidocbench_rl(
    data_dir: str = "data/ocr/omnidocbench",
    max_train_samples: int | None = None,
    eval_fraction: float = 0.1,
):
    """Load OmniDocBench for GRPO training with edit similarity rewards.

    Reads pre-converted markdown ground truth (from convert_omnidocbench.py)
    and pairs with images. Each sample has a prompt, image, and ground_truth.

    Args:
        data_dir: Path to OmniDocBench dataset root (with images/ and markdown/).
        max_train_samples: Maximum training samples.
        eval_fraction: Fraction of samples to hold out for eval.

    Returns:
        Tuple of (train_dataset, eval_dataset).
    """
    from pathlib import Path
    from datasets import Dataset
    from PIL import Image

    from src.data.ocr_utils import OCR_SYSTEM_PROMPT

    data_path = Path(data_dir)
    images_dir = data_path / "images"
    markdown_dir = data_path / "markdown"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"images/ not found in {data_dir}. Download with:\n"
            f"  python -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('opendatalab/OmniDocBench', local_dir='{data_dir}', repo_type='dataset')\""
        )
    if not markdown_dir.exists():
        raise FileNotFoundError(
            f"markdown/ not found in {data_dir}. Run:\n"
            f"  python scripts/data/convert_omnidocbench.py --data_dir {data_dir}"
        )

    # Each markdown file corresponds to an image (same stem)
    samples = []
    for md_path in sorted(markdown_dir.glob("*.md")):
        image_path = images_dir / (md_path.stem + ".jpg")
        if not image_path.exists():
            # Try other extensions
            for ext in (".png", ".jpeg"):
                alt = images_dir / (md_path.stem + ext)
                if alt.exists():
                    image_path = alt
                    break
            else:
                continue

        with open(md_path) as f:
            ground_truth = f.read().strip()

        if not ground_truth:
            continue

        with Image.open(image_path) as im:
            image = im.convert("RGB")
            image.load()
        samples.append({
            "prompt": [{"role": "user", "content": [
                {"type": "text", "text": OCR_SYSTEM_PROMPT},
                {"type": "image"},
            ]}],
            "image": image,
            "ground_truth": ground_truth,
        })

    logger.info(f"Loaded {len(samples)} OmniDocBench pages with markdown GT")

    if not samples:
        raise ValueError(f"No valid samples found in {data_dir}")

    dataset = Dataset.from_list(samples)
    splits = dataset.train_test_split(test_size=eval_fraction, seed=42)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]

    if max_train_samples and len(train_dataset) > max_train_samples:
        train_dataset = train_dataset.select(range(max_train_samples))

    logger.info(f"OmniDocBench RL dataset: {len(train_dataset)} train, {len(eval_dataset)} eval")
    return train_dataset, eval_dataset


def load_rl_dataset(config: dict):
    """Load dataset for RL training based on config.

    Args:
        config: Configuration dictionary with 'data' section.

    Returns:
        Tuple of (train_dataset, eval_dataset).
    """
    task = config["data"]["task"]

    if task == "europarl":
        return load_europarl_rl(
            max_samples=config["data"].get("max_samples", 45715),
            source_lang=config["data"].get("source_lang", "en"),
            target_lang=config["data"].get("target_lang", "fr"),
        )
    elif task == "ocr":
        dataset_name = config["data"].get("dataset", "")
        if dataset_name == "synthmix":
            return load_synthmix_rl(
                data_dir=config["data"].get("data_dir", "data/ocr/synthmix"),
                max_train_samples=config["data"].get("max_train_samples"),
            )
        if dataset_name == "omnidocbench":
            return load_omnidocbench_rl(
                data_dir=config["data"].get("data_dir", "data/ocr/omnidocbench"),
                max_train_samples=config["data"].get("max_train_samples"),
            )
        return load_ocr_rl(
            data_dir=config["data"].get("data_dir", "data/ocr/documents"),
            max_train_samples=config["data"].get("max_train_samples"),
            include_reasoning=config["data"].get("include_reasoning", False),
        )
    else:
        raise ValueError(
            f"RL training not supported for task '{task}'. "
            f"Supported: europarl, ocr. "
            f"Classification tasks use encoder models that cannot generate."
        )
