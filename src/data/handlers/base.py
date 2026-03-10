"""Base class for all dataset handlers.

Every dataset (public or synthetic) implements the same interface:
  prepare()  - download/generate + convert (idempotent)
  validate() - check integrity
  load()     - read raw samples from disk -> (train, eval) HF Datasets
"""

from abc import ABC, abstractmethod

from datasets import Dataset


class DatasetHandler(ABC):
    """Common interface for all dataset handlers.

    Public datasets implement prepare() as download + convert.
    Synthetic datasets implement prepare() as generate (if data missing).
    Both are idempotent: re-running skips already-completed work.
    """

    @abstractmethod
    def prepare(self, data_dir: str, **kwargs) -> None:
        """Download/generate and convert data. Idempotent.

        Args:
            data_dir: Root directory for this dataset's files.
        """

    @abstractmethod
    def validate(self, data_dir: str) -> dict:
        """Check dataset integrity.

        Args:
            data_dir: Root directory for this dataset's files.

        Returns:
            Dict with validation results: {"ok": bool, "details": str, ...}
        """

    @abstractmethod
    def load(
        self,
        data_dir: str,
        max_train_samples: int | None = None,
        eval_fraction: float = 0.1,
        **kwargs,
    ) -> tuple[Dataset, Dataset | None]:
        """Load raw samples from disk.

        Returns unformatted samples. SFT vs RL formatting is handled
        downstream by formatters, not here.

        Args:
            data_dir: Root directory for this dataset's files.
            max_train_samples: Cap on training samples.
            eval_fraction: Fraction held out for eval (if not pre-split).

        Returns:
            Tuple of (train_dataset, eval_dataset). eval may be None.
        """
