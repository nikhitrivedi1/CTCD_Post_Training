"""CLINC150 intent classification dataset handler.

Handles DeepPavlov/clinc150 from HF Hub: 150-way single-utterance intent
classification. Downloaded from HF Hub at load time (no local prep needed).

Dataset ships fixed train/validation/test splits (15.2k/3.1k/5.5k rows) under
the "default" config, with columns `utterance` (text) and `label` (int, 0-149).
This handler uses the train and validation splits directly, like GLUE - no
eval_fraction synthesis is needed. The test split is held out, unused (same
convention as the GLUE handler).

Each split also includes out-of-scope utterances with `label=None` (~200 in
train, ~100 in validation) - these are filtered out, same as GLUE's QQP
handler drops sentinel `-1` labels, since num_labels=150 has no OOS class.

Pipeline: no prep needed, download inline during load().
"""

import logging

from datasets import Dataset, load_dataset

from .base import DatasetHandler

logger = logging.getLogger(__name__)

_HF_DATASET = "DeepPavlov/clinc150"
_HF_SUBSET = "default"


class CLINC150Handler(DatasetHandler):
    """Handler for the CLINC150 intent classification dataset."""

    def prepare(self, data_dir: str, **kwargs) -> None:
        """No-op. CLINC150 is downloaded from HF Hub during load().

        Args:
            data_dir: Not used.
        """
        logger.info("clinc150 requires no preparation (downloaded from HF Hub at load time)")

    def validate(self, data_dir: str) -> dict:
        """Validate by checking HF Hub availability.

        Args:
            data_dir: Not used.

        Returns:
            Validation results dict.
        """
        try:
            ds = load_dataset(_HF_DATASET, _HF_SUBSET, split="train", streaming=True)
            next(iter(ds))
            return {"ok": True, "details": "clinc150 accessible on HF Hub"}
        except Exception as e:
            return {"ok": False, "details": f"Cannot access clinc150: {e}"}

    def load(
        self,
        data_dir: str = "",
        max_train_samples: int | None = None,
        eval_fraction: float = 0.1,
        **kwargs,
    ) -> tuple[Dataset, Dataset | None]:
        """Load CLINC150 as raw text + label samples.

        Returns unprocessed samples. Tokenization happens downstream
        in the formatter (format_for_sft).

        Args:
            data_dir: Not used (dataset from HF Hub).
            max_train_samples: Cap on training samples.
            eval_fraction: Not used (CLINC150 has fixed train/validation splits).

        Returns:
            Tuple of (train_dataset, eval_dataset).
        """
        dataset = load_dataset(_HF_DATASET, _HF_SUBSET)

        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

        # Drop out-of-scope utterances (label=None); no OOS class in num_labels=150
        train_dataset = train_dataset.filter(lambda x: x["label"] is not None)
        eval_dataset = eval_dataset.filter(lambda x: x["label"] is not None)

        # Rename label -> labels for HF Trainer compatibility
        train_dataset = train_dataset.rename_column("label", "labels")
        eval_dataset = eval_dataset.rename_column("label", "labels")

        if max_train_samples and len(train_dataset) > max_train_samples:
            train_dataset = train_dataset.select(range(max_train_samples))

        logger.info(f"clinc150: {len(train_dataset)} train, {len(eval_dataset)} eval")
        return train_dataset, eval_dataset
