"""Synthmix dataset handler.

Public OCR dataset from allenai (olmOCR-synthmix-1025). Contains PDF pages
with unit test assertions for GRPO training.

Pipeline: download from HF -> render PDFs to images -> load.
"""

import json
import logging
import os
from collections import defaultdict
from pathlib import Path

from datasets import Dataset
from PIL import Image

from .base import DatasetHandler

logger = logging.getLogger(__name__)


class SynthmixHandler(DatasetHandler):
    """Handler for olmOCR-synthmix-1025 (allenai/olmOCR-synthmix-1025)."""

    def prepare(self, data_dir: str, dpi: int = 150, workers: int | None = None, **kwargs) -> None:
        """Download from HuggingFace and render PDFs to images.

        Args:
            data_dir: Root directory (e.g. data/ocr/synthmix).
            dpi: Render resolution for PDF pages.
            workers: Parallel workers for rendering.
        """
        import glob as globmod
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from src.data.ocr_utils import render_pdf_page, pdf_image_name

        data_path = Path(data_dir)
        bench_dir = data_path / "bench_data"
        images_dir = data_path / "images"

        # Step 1: Download if needed
        if not bench_dir.exists():
            logger.info("Downloading olmOCR-synthmix-1025 from HuggingFace...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                "allenai/olmOCR-synthmix-1025",
                local_dir=str(data_path),
                repo_type="dataset",
            )
            logger.info("Download complete")

        # Step 2: Render PDFs to images
        pdf_base = bench_dir / "pdfs"
        if not pdf_base.exists():
            raise FileNotFoundError(f"bench_data/pdfs/ not found in {data_dir}")

        os.makedirs(images_dir, exist_ok=True)
        workers = workers or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or os.cpu_count() or 1

        # Collect unique pages from JSONL
        pages = set()
        for jsonl_path in sorted(globmod.glob(str(bench_dir / "*.jsonl"))):
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    test = json.loads(line)
                    pages.add((test["pdf"], test["page"]))
        pages = sorted(pages)
        logger.info(f"Found {len(pages)} unique pages to render")

        # Build jobs, skipping already-rendered
        jobs = []
        skipped = 0
        for pdf_rel, page_num in pages:
            out_path = str(images_dir / pdf_image_name(pdf_rel, page_num))
            if os.path.exists(out_path):
                skipped += 1
                continue
            pdf_path = str(pdf_base / pdf_rel)
            if not os.path.exists(pdf_path):
                continue
            jobs.append((pdf_path, page_num, dpi, out_path))

        if not jobs:
            logger.info(f"All {skipped} images already exist, nothing to render")
            return

        logger.info(f"Rendering {len(jobs)} pages ({skipped} already exist) with {workers} workers...")

        def _render_one(args):
            pdf_path, page_num, dpi, out_path = args
            try:
                image = render_pdf_page(pdf_path, page_num, dpi=dpi)
                image.save(out_path, "JPEG", quality=90)
                return True
            except Exception:
                return False

        rendered = errors = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_render_one, job): job for job in jobs}
            for future in as_completed(futures):
                if future.result():
                    rendered += 1
                else:
                    errors += 1

        logger.info(f"Done: {rendered} rendered, {skipped} existed, {errors} errors")

    def validate(self, data_dir: str) -> dict:
        """Check that bench_data and images exist and are aligned.

        Args:
            data_dir: Root directory.

        Returns:
            Validation results dict.
        """
        import glob as globmod

        data_path = Path(data_dir)
        bench_dir = data_path / "bench_data"
        images_dir = data_path / "images"

        if not bench_dir.exists():
            return {"ok": False, "details": f"bench_data/ not found in {data_dir}"}
        if not images_dir.exists():
            return {"ok": False, "details": f"images/ not found in {data_dir}"}

        # Count tests and pages
        tests = 0
        pages = set()
        for jsonl_path in sorted(globmod.glob(str(bench_dir / "*.jsonl"))):
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    t = json.loads(line)
                    tests += 1
                    pages.add((t["pdf"], t["page"]))

        images = list(images_dir.glob("*.jpg"))
        result = {
            "ok": len(images) >= len(pages) and len(pages) > 0,
            "pages": len(pages),
            "tests": tests,
            "images": len(images),
        }
        details = f"{len(pages)} pages, {tests} tests, {len(images)} images"
        if len(images) < len(pages):
            details += f", {len(pages) - len(images)} missing images"
        result["details"] = details
        return result

    def load(
        self,
        data_dir: str,
        max_train_samples: int | None = None,
        eval_fraction: float = 0.1,
        **kwargs,
    ) -> tuple[Dataset, Dataset | None]:
        """Load image + test assertion pairs as raw samples.

        Args:
            data_dir: Root directory with bench_data/ and images/.
            max_train_samples: Cap on training samples.
            eval_fraction: Fraction held out for eval.

        Returns:
            Tuple of (train_dataset, eval_dataset).
        """
        import glob as globmod

        from src.data.ocr_utils import pdf_image_name

        data_path = Path(data_dir)
        bench_dir = data_path / "bench_data"
        images_dir = data_path / "images"

        if not bench_dir.exists():
            raise FileNotFoundError(f"bench_data/ not found in {data_dir}. Run: prepare.py --dataset synthmix")
        if not images_dir.exists():
            raise FileNotFoundError(f"images/ not found in {data_dir}. Run: prepare.py --dataset synthmix")

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

        # Build samples
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

            samples.append({"image": image, "tests": test_list})

        if missing_images > 0:
            logger.warning(f"Skipped {missing_images} pages with missing images")

        if not samples:
            raise ValueError(f"No valid samples found in {data_dir}")

        logger.info(f"Loaded {len(samples)} pages with {sum(len(s['tests']) for s in samples)} tests")

        dataset = Dataset.from_list(samples)
        splits = dataset.train_test_split(test_size=eval_fraction, seed=42)
        train_dataset = splits["train"]
        eval_dataset = splits["test"]

        if max_train_samples and len(train_dataset) > max_train_samples:
            train_dataset = train_dataset.select(range(max_train_samples))

        logger.info(f"Synthmix: {len(train_dataset)} train, {len(eval_dataset)} eval")
        return train_dataset, eval_dataset
