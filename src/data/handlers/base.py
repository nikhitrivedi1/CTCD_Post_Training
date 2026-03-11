"""Base class for all dataset handlers.

Every dataset (public or synthetic) implements the same interface:
  prepare()     - download/generate + convert (idempotent)
  validate()    - check integrity
  load()        - read raw samples from disk -> (train, eval) HF Datasets
  load_paths()  - list file paths without loading into memory (for bench)
"""

from abc import ABC, abstractmethod
from pathlib import Path

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

    def load_paths(
        self,
        data_dir: str,
        split: str | None = None,
    ) -> list[dict[str, Path]]:
        """List image/GT file paths without loading into memory.

        Memory-efficient alternative to load() for bench scripts that
        process pages one at a time. Override in handlers with image data.

        Args:
            data_dir: Root directory for this dataset's files.
            split: Optional split filter ("train", "eval"). None returns all.

        Returns:
            List of dicts with "image_path" and "gt_path" as Path objects.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support load_paths()")
