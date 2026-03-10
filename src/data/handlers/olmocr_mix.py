"""olmOCR-mix dataset handler.

Public OCR dataset from allenai (olmOCR-mix-1025). Contains 260K+ document
pages across 4 subsets with OCR text annotations.

Pipeline: download + render PDFs (via prepare_ocr_data.py) -> load HF dataset from disk.
"""

import logging
import subprocess
import sys
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk

from .base import DatasetHandler

logger = logging.getLogger(__name__)


class OlmOCRMixHandler(DatasetHandler):
    """Handler for olmOCR-mix-1025 (allenai/olmOCR-mix-1025).

    prepare() delegates to scripts/data/prepare_ocr_data.py which handles
    the complex pipelined download/render/build process (659 lines).
    load() reads the resulting HF Dataset from disk.
    """

    def prepare(
        self,
        data_dir: str,
        subset: str | None = None,
        max_samples: int | None = None,
        **kwargs,
    ) -> None:
        """Download and render olmOCR-mix using the existing pipeline script.

        Args:
            data_dir: Output directory (e.g. data/ocr/olmocr-mix-5k).
            subset: Specific subset (e.g. "00_documents") or None for all.
            max_samples: Max samples per subset.
        """
        data_path = Path(data_dir)

        # Check if already prepared (has train/ split)
        has_train = (data_path / "train").exists()
        if not has_train:
            # Check multi-subset layout
            has_train = any(
                (d / "train").exists()
                for d in data_path.iterdir()
                if d.is_dir()
            ) if data_path.exists() else False

        if has_train:
            logger.info(f"olmOCR-mix data already exists at {data_dir}, skipping")
            return

        logger.info(f"Preparing olmOCR-mix at {data_dir}...")

        script = Path(__file__).resolve().parents[3] / "scripts" / "data" / "olmocr_mix_pipeline.py"
        if not script.exists():
            raise FileNotFoundError(f"Preparation script not found: {script}")

        cmd = [sys.executable, str(script), "--output", str(data_dir)]
        if subset:
            cmd.extend(["--subset", subset])
        if max_samples:
            cmd.extend(["--max_samples", str(max_samples)])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Preparation failed:\n{result.stderr}")
            raise RuntimeError(f"olmOCR-mix preparation failed: {result.stderr[:500]}")

        logger.info("Preparation complete")

    def validate(self, data_dir: str) -> dict:
        """Check that HF dataset splits exist on disk.

        Args:
            data_dir: Root directory.

        Returns:
            Validation results dict.
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            return {"ok": False, "details": f"Directory not found: {data_dir}"}

        # Check single-dir or multi-subset layout
        train_count = 0
        eval_count = 0
        subsets = []

        if (data_path / "train").exists():
            ds = load_from_disk(str(data_path / "train"))
            train_count = len(ds)
            if (data_path / "eval").exists():
                eval_count = len(load_from_disk(str(data_path / "eval")))
        else:
            for subset_dir in sorted(data_path.iterdir()):
                if not subset_dir.is_dir() or not (subset_dir / "train").exists():
                    continue
                subsets.append(subset_dir.name)
                train_count += len(load_from_disk(str(subset_dir / "train")))
                if (subset_dir / "eval").exists():
                    eval_count += len(load_from_disk(str(subset_dir / "eval")))

        result = {
            "ok": train_count > 0,
            "train": train_count,
            "eval": eval_count,
            "subsets": len(subsets) if subsets else 1,
            "details": f"{train_count} train, {eval_count} eval" +
                       (f" across {len(subsets)} subsets" if subsets else ""),
        }
        return result

    def load(
        self,
        data_dir: str,
        max_train_samples: int | None = None,
        eval_fraction: float = 0.1,
        include_reasoning: bool = False,
        **kwargs,
    ) -> tuple[Dataset, Dataset | None]:
        """Load pre-processed HF dataset from disk.

        Supports both single-directory and multi-subset layouts.
        Returns raw samples with 'image' and 'ground_truth' columns.

        Args:
            data_dir: Root directory with train/ and eval/ or subset subdirs.
            max_train_samples: Cap on training samples.
            eval_fraction: Not used (data is pre-split).
            include_reasoning: Use 'reasoning_and_output' column instead of 'text'.

        Returns:
            Tuple of (train_dataset, eval_dataset).
        """
        data_path = Path(data_dir)

        if (data_path / "train").exists():
            train_dataset = load_from_disk(str(data_path / "train"))
            eval_dataset = load_from_disk(str(data_path / "eval")) if (data_path / "eval").exists() else None
        else:
            train_parts, eval_parts = [], []
            for subset_dir in sorted(data_path.iterdir()):
                if not subset_dir.is_dir() or not (subset_dir / "train").exists():
                    continue
                logger.info(f"  Loading subset: {subset_dir.name}")
                train_parts.append(load_from_disk(str(subset_dir / "train")))
                if (subset_dir / "eval").exists():
                    eval_parts.append(load_from_disk(str(subset_dir / "eval")))
            if not train_parts:
                raise FileNotFoundError(f"No train splits found in {data_path}. Run: prepare.py --dataset olmocr-mix")
            train_dataset = concatenate_datasets(train_parts)
            eval_dataset = concatenate_datasets(eval_parts) if eval_parts else None

        if max_train_samples and len(train_dataset) > max_train_samples:
            train_dataset = train_dataset.select(range(max_train_samples))

        # Normalize column name: 'text' or 'reasoning_and_output' -> 'ground_truth'
        text_col = "reasoning_and_output" if include_reasoning else "text"

        def normalize(example):
            return {"ground_truth": example.get(text_col) or ""}

        keep_cols = {"image"}
        train_dataset = train_dataset.map(
            normalize,
            remove_columns=[c for c in train_dataset.column_names if c not in keep_cols],
        )
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                normalize,
                remove_columns=[c for c in eval_dataset.column_names if c not in keep_cols],
            )

        eval_count = len(eval_dataset) if eval_dataset is not None else 0
        logger.info(f"olmOCR-mix: {len(train_dataset)} train, {eval_count} eval")
        return train_dataset, eval_dataset
