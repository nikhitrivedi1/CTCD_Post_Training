"""Europarl translation dataset handler.

Public parallel translation corpus from Helsinki-NLP. Used for
translation fine-tuning and RL training.

Pipeline: download from HF Hub at load time (small dataset, no prep needed).
"""

import logging

from datasets import Dataset, load_dataset

from .base import DatasetHandler

logger = logging.getLogger(__name__)

# Language name mapping for prompts
LANG_NAMES = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "da": "Danish",
    "sv": "Swedish", "fi": "Finnish", "el": "Greek", "cs": "Czech",
    "pl": "Polish", "hu": "Hungarian", "ro": "Romanian", "bg": "Bulgarian",
}


class EuroparlHandler(DatasetHandler):
    """Handler for Helsinki-NLP/europarl translation dataset."""

    def prepare(self, data_dir: str, **kwargs) -> None:
        """No-op. Europarl is downloaded inline from HF Hub during load().

        Args:
            data_dir: Not used (dataset streams from HF Hub).
        """
        logger.info("Europarl requires no preparation (downloaded from HF Hub at load time)")

    def validate(self, data_dir: str) -> dict:
        """Validate by checking HF Hub availability.

        Args:
            data_dir: Not used.

        Returns:
            Validation results dict.
        """
        try:
            ds = load_dataset("Helsinki-NLP/europarl", "en-fr", split="train", streaming=True)
            # Just check we can iterate
            next(iter(ds))
            return {"ok": True, "details": "HF Hub dataset accessible"}
        except Exception as e:
            return {"ok": False, "details": f"Cannot access dataset: {e}"}

    def load(
        self,
        data_dir: str = "",
        max_train_samples: int | None = None,
        eval_fraction: float = 0.1,
        source_lang: str = "en",
        target_lang: str = "fr",
        max_samples: int = 45715,
        **kwargs,
    ) -> tuple[Dataset, Dataset | None]:
        """Load Europarl as raw source/target text pairs.

        Args:
            data_dir: Not used (dataset from HF Hub).
            max_train_samples: Cap on training samples (applied after split).
            eval_fraction: Not used (fixed 70/20/10 split).
            source_lang: Source language code.
            target_lang: Target language code.
            max_samples: Total samples to load.

        Returns:
            Tuple of (train_dataset, eval_dataset).
        """
        lang_pair = f"{source_lang}-{target_lang}"
        dataset = load_dataset("Helsinki-NLP/europarl", lang_pair, split="train")

        if len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        # Fixed 70/20/10 split matching original loaders
        splits = dataset.train_test_split(test_size=0.3, seed=42)
        train_data = splits["train"]
        temp_splits = splits["test"].train_test_split(test_size=0.333, seed=42)
        val_data = temp_splits["train"]

        source_name = LANG_NAMES.get(source_lang, source_lang.upper())
        target_name = LANG_NAMES.get(target_lang, target_lang.upper())

        system_message = (
            f"You are a professional translator. Translate the provided text from "
            f"{source_name} to {target_name}, remaining true to the source text. "
            f"Do not add any additional commentary or conversational elements to your response."
        )

        def format_raw(example):
            return {
                "source_text": example["translation"][source_lang],
                "ground_truth": example["translation"][target_lang],
                "system_message": system_message,
            }

        train_dataset = train_data.map(format_raw, remove_columns=train_data.column_names)
        eval_dataset = val_data.map(format_raw, remove_columns=val_data.column_names)

        if max_train_samples and len(train_dataset) > max_train_samples:
            train_dataset = train_dataset.select(range(max_train_samples))

        logger.info(f"Europarl: {len(train_dataset)} train, {len(eval_dataset)} eval")
        return train_dataset, eval_dataset
