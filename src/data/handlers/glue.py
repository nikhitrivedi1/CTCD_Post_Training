"""GLUE classification dataset handler.

Handles MNLI, QQP, SST-2, CoLA, MRPC, QNLI, RTE, and WNLI from the GLUE benchmark.
All are downloaded from HF Hub at load time.

Pipeline: no prep needed, download inline during load().
"""

import logging

from datasets import Dataset, load_dataset

from .base import DatasetHandler

logger = logging.getLogger(__name__)

# Task-specific config: column names, subset, eval split
_GLUE_TASKS = {
    "mnli": {
        "subset": "mnli",
        "input_columns": ("premise", "hypothesis"),
        "eval_split": "validation_matched",
    },
    "qqp": {
        "subset": "qqp",
        "input_columns": ("question1", "question2"),
        "eval_split": "validation",
        "filter_labels": True,  # Filter -1 labels
    },
    "sst2": {
        "subset": "sst2",
        "input_columns": ("sentence",),
        "eval_split": "validation",
    },
    "cola": {
        "subset": "cola",
        "input_columns": ("sentence",),
        "eval_split": "validation",
    },
    "mrpc": {
        "subset": "mrpc",
        "input_columns": ("sentence1", "sentence2"),
        # Note that test split is available for MRPC
        "eval_split": "validation",
    },
    "qnli": {
        "subset": "qnli",
        "input_columns": ("question", "sentence"),
        "eval_split": "validation",
    },
    "rte": {
        "subset": "rte",
        "input_columns": ("sentence1", "sentence2"),
        "eval_split": "validation",
    },
    "wnli": {
        "subset": "wnli",
        "input_columns": ("sentence1", "sentence2"),
        "eval_split": "validation",
    },
}


class GLUEHandler(DatasetHandler):
    """Handler for GLUE classification tasks (MNLI, QQP, SST-2, CoLA, MRPC, QNLI, RTE, WNLI).

    Instantiated per-task: GLUEHandler(task="mnli"), GLUEHandler(task="qqp"), etc.
    """

    def __init__(self, task: str = "mnli"):
        """Initialize with a specific GLUE task.

        Args:
            task: One of "mnli", "qqp", "sst2", "cola", "mrpc", "qnli", "rte", "wnli".
        """
        if task not in _GLUE_TASKS:
            raise ValueError(f"Unknown GLUE task: '{task}'. Supported: {list(_GLUE_TASKS.keys())}")
        self.task = task
        self._config = _GLUE_TASKS[task]

    def prepare(self, data_dir: str, **kwargs) -> None:
        """No-op. GLUE datasets are downloaded from HF Hub during load().

        Args:
            data_dir: Not used.
        """
        logger.info(
            f"GLUE/{self.task} requires no preparation (downloaded from HF Hub at load time)"
        )

    def validate(self, data_dir: str) -> dict:
        """Validate by checking HF Hub availability.

        Args:
            data_dir: Not used.

        Returns:
            Validation results dict.
        """
        try:
            ds = load_dataset("glue", self._config["subset"], split="train", streaming=True)
            next(iter(ds))
            return {"ok": True, "details": f"GLUE/{self.task} accessible on HF Hub"}
        except Exception as e:
            return {"ok": False, "details": f"Cannot access GLUE/{self.task}: {e}"}

    def load(
        self,
        data_dir: str = "",
        max_train_samples: int | None = None,
        eval_fraction: float = 0.1,
        **kwargs,
    ) -> tuple[Dataset, Dataset | None]:
        """Load GLUE task as raw text + label samples.

        Returns unprocessed samples. Tokenization happens downstream
        in the formatter (format_for_sft).

        Args:
            data_dir: Not used (dataset from HF Hub).
            max_train_samples: Cap on training samples.
            eval_fraction: Not used (GLUE has fixed splits).

        Returns:
            Tuple of (train_dataset, eval_dataset).
        """
        dataset = load_dataset("glue", self._config["subset"])

        train_dataset = dataset["train"]
        eval_dataset = dataset[self._config["eval_split"]]

        # QQP has some -1 labels (unlabeled)
        if self._config.get("filter_labels"):
            train_dataset = train_dataset.filter(lambda x: x["label"] != -1)
            eval_dataset = eval_dataset.filter(lambda x: x["label"] != -1)

        # Rename label -> labels for HF Trainer compatibility
        train_dataset = train_dataset.rename_column("label", "labels")
        eval_dataset = eval_dataset.rename_column("label", "labels")

        # Drop idx column (not needed)
        if "idx" in train_dataset.column_names:
            train_dataset = train_dataset.remove_columns(["idx"])
            eval_dataset = eval_dataset.remove_columns(["idx"])

        if max_train_samples and len(train_dataset) > max_train_samples:
            train_dataset = train_dataset.select(range(max_train_samples))

        logger.info(f"GLUE/{self.task}: {len(train_dataset)} train, {len(eval_dataset)} eval")
        return train_dataset, eval_dataset
