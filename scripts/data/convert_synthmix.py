"""Convert olmOCR-synthmix PDFs to images for GRPO training.

Renders each PDF page referenced in bench_data/*.jsonl to a JPG image.
Run this on the cluster after downloading the dataset.

Usage:
    python scripts/data/convert_synthmix.py --data_dir data/ocr/synthmix
    python scripts/data/convert_synthmix.py --data_dir data/ocr/synthmix --dpi 200
"""

import argparse
import glob
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add project root for src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ocr_utils import render_pdf_page, pdf_image_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def collect_pages_from_jsonl(jsonl_dir: str) -> list[tuple[str, int]]:
    """Collect unique (pdf_rel, page) pairs from JSONL files in a directory.

    Args:
        jsonl_dir: Path to directory containing *.jsonl files.

    Returns:
        Sorted list of (pdf_relative_path, page_number) tuples.
    """
    pages = set()
    for jsonl_path in sorted(glob.glob(os.path.join(jsonl_dir, "*.jsonl"))):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                test = json.loads(line)
                pages.add((test["pdf"], test["page"]))
    return sorted(pages)


def _render_one(args: tuple) -> str | None:
    """Render a single PDF page to JPG. Top-level for pickling.

    Args:
        args: Tuple of (pdf_path, page_num, dpi, out_path).

    Returns:
        out_path on success, None on failure.
    """
    pdf_path, page_num, dpi, out_path = args
    try:
        image = render_pdf_page(pdf_path, page_num, dpi=dpi)
        image.save(out_path, "JPEG", quality=90)
        return out_path
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Convert synthmix PDFs to images")
    parser.add_argument("--data_dir", type=str, default="data/ocr/synthmix",
                        help="Path to synthmix dataset root")
    parser.add_argument("--dpi", type=int, default=150, help="Render resolution")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: SLURM_CPUS_PER_TASK or all CPUs)")
    args = parser.parse_args()

    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or os.cpu_count() or 1

    bench_dir = os.path.join(args.data_dir, "bench_data")
    pdf_base = os.path.join(bench_dir, "pdfs")
    images_dir = os.path.join(args.data_dir, "images")

    if not os.path.isdir(bench_dir):
        raise FileNotFoundError(f"bench_data/ not found in {args.data_dir}")
    if not os.path.isdir(pdf_base):
        raise FileNotFoundError(f"bench_data/pdfs/ not found in {args.data_dir}")

    os.makedirs(images_dir, exist_ok=True)

    pages = collect_pages_from_jsonl(bench_dir)
    logger.info(f"Found {len(pages)} unique (pdf, page) pairs to render")

    # Build jobs, skipping already-rendered images
    jobs = []
    skipped = 0
    missing = 0
    for pdf_rel, page_num in pages:
        out_path = os.path.join(images_dir, pdf_image_name(pdf_rel, page_num))
        if os.path.exists(out_path):
            skipped += 1
            continue
        pdf_path = os.path.join(pdf_base, pdf_rel)
        if not os.path.exists(pdf_path):
            logger.warning(f"PDF not found: {pdf_path}")
            missing += 1
            continue
        jobs.append((pdf_path, page_num, args.dpi, out_path))

    logger.info(f"To render: {len(jobs)}, already exist: {skipped}, missing PDFs: {missing}")

    if not jobs:
        logger.info("Nothing to render")
        return

    # Render in parallel
    rendered = 0
    errors = 0
    logger.info(f"Rendering with {workers} workers...")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_render_one, job): job[3] for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                rendered += 1
                if rendered % 100 == 0:
                    logger.info(f"Rendered {rendered}/{len(jobs)} images...")
            else:
                errors += 1

    logger.info(f"Done: {rendered} rendered, {skipped} already existed, {errors} errors")
    logger.info(f"Images saved to {images_dir}")


if __name__ == "__main__":
    main()
