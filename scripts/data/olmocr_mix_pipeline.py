"""Pre-process olmOCR-mix-1025: download dataset, render PDFs to images, save as HF dataset.

Pipelined two-phase approach:

  Phase 1 — Render PDFs to PNG files on disk (pipelined, O(1) memory)
    Three-stage pipeline that overlaps I/O with compute:
      1. Background download: Thread pool downloads tarballs; rendering starts as each arrives
      2. Double-buffer: Extract tarball N+1 in background while rendering tarball N
      3. Background cleanup: shutil.rmtree runs in a thread, never blocks rendering

    With 64 workers on H200 nodes, this yields ~2.4x speedup over the serial version
    by keeping CPUs busy during download/extract/cleanup I/O.

  Phase 2 — Build HuggingFace dataset (streaming, O(1) memory)
    Dataset.from_generator() reads JSONL + PNG files one at a time, writes Arrow
    incrementally. Never holds more than one image in memory.

NOTE: This script is specific to allenai/olmOCR-mix-1025's structure:
  - 4 subsets: 00_documents (232K), 01_books (17K), 02_loc_transcripts (10K), 03_national_archives (10K)
  - PDFs packed in tarballs referenced by pdf_relpath column
  - --max_samples selects the first N rows from the train split

Usage:
    python scripts/data/prepare_ocr_data.py --subset 00_documents --output data/ocr/olmocr-mix-5k --max_samples 5000
    python scripts/data/prepare_ocr_data.py --output data/ocr/olmocr-mix-full  # all subsets
"""

import argparse
import base64
import json
import logging
import os
import shutil
import tarfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from datasets import Dataset, Features, Image, Value, load_dataset, load_dataset_builder
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1: Render PDFs → PNG files on disk
# ---------------------------------------------------------------------------


def _render_and_save_png(args: tuple) -> dict | None:
    """Render a single PDF page to PNG and save to disk. No PIL involved.

    Called by ProcessPoolExecutor — must be a picklable top-level function.

    Args:
        args: Tuple of (pdf_path, page_number, target_dim, output_png_path)

    Returns:
        Dict with metadata if successful, None if rendering failed
    """
    pdf_path, page_number, target_dim, output_png_path = args
    try:
        from olmocr.data.renderpdf import render_pdf_to_base64png

        image_base64 = render_pdf_to_base64png(
            pdf_path, page_number, target_longest_image_dim=target_dim
        )
        png_bytes = base64.b64decode(image_base64)
        with open(output_png_path, "wb") as f:
            f.write(png_bytes)
        return {"png_path": output_png_path}
    except Exception as e:
        logger.warning(f"Failed to render {pdf_path} page {page_number}: {e}")
        return None


def _find_tarball(tarball_name: str, cache_dir: str | None) -> str | None:
    """Download/locate a tarball via HuggingFace Hub.

    Args:
        tarball_name: HF-relative path (e.g., 'pdf_tarballs/00_documents_train_00000.tar.gz')
        cache_dir: HuggingFace cache directory

    Returns:
        Absolute path to the tarball file or None
    """
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            repo_id="allenai/olmOCR-mix-1025",
            filename=tarball_name,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        return path
    except Exception as e:
        logger.warning(f"Failed to download tarball {tarball_name}: {e}")
        return None


def _get_source_metadata(repo_id: str, subset: str) -> dict:
    """Fetch provenance metadata from the source HuggingFace dataset.

    Args:
        repo_id: HuggingFace dataset ID
        subset: Dataset subset/config name

    Returns:
        Dict with source dataset metadata
    """
    try:
        builder = load_dataset_builder(repo_id, subset)
        info = builder.info
        splits_info = {}
        if info.splits:
            for name, split in info.splits.items():
                splits_info[name] = {
                    "num_examples": split.num_examples,
                    "num_bytes": split.num_bytes,
                }
        return {
            "repo_id": repo_id,
            "subset": subset,
            "version": str(info.version),
            "description": info.description,
            "license": info.license,
            "features": {k: str(v) for k, v in (info.features or {}).items()},
            "splits": splits_info,
            "download_size": info.download_size,
            "dataset_size": info.dataset_size,
        }
    except Exception as e:
        logger.warning(f"Could not fetch source metadata: {e}")
        return {"repo_id": repo_id, "subset": subset, "error": str(e)}


def _extract_tarball(tarball_path: str, extract_dir: str) -> bool:
    """Extract a tarball to a directory. Returns True on success."""
    os.makedirs(extract_dir, exist_ok=True)
    try:
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        return True
    except Exception as e:
        logger.warning(f"  Failed to extract {tarball_path}: {e}")
        return False


class _TarballDownloader:
    """Background tarball downloader with per-tarball blocking.

    Downloads start immediately on construction via a thread pool.
    Call .get(name) to block until a specific tarball is ready.
    Provides the same .get() interface as a dict for drop-in use.
    """

    def __init__(self, tarball_names: list[str], cache_dir: str | None, max_threads: int = 4):
        self._results: dict[str, str | None] = {}
        self._events: dict[str, threading.Event] = {
            name: threading.Event() for name in tarball_names
        }
        self._lock = threading.Lock()
        self._completed = 0
        self._total = len(tarball_names)
        self._pool = ThreadPoolExecutor(max_workers=max_threads)

        for name in tarball_names:
            future = self._pool.submit(_find_tarball, name, cache_dir)
            future.add_done_callback(lambda f, n=name: self._on_complete(n, f))

    def _on_complete(self, name: str, future):
        self._results[name] = future.result()
        with self._lock:
            self._completed += 1
            count = self._completed
        status = "ok" if self._results[name] else "FAILED"
        logger.info(f"  Downloaded {count}/{self._total}: {name} [{status}]")
        self._events[name].set()

    def get(self, name: str, default=None) -> str | None:
        """Block until this tarball is downloaded, return its local path."""
        if name not in self._events:
            return default
        self._events[name].wait()
        return self._results.get(name, default)

    def shutdown(self):
        """Wait for all remaining downloads and release thread pool."""
        self._pool.shutdown(wait=True)


def _collect_tarball_names(split_data) -> list[str]:
    """Extract unique tarball names from a dataset split."""
    names = set()
    for row in split_data:
        pdf_relpath = row["pdf_relpath"]
        if ":" in pdf_relpath:
            names.add(pdf_relpath.split(":", 1)[0])
    return sorted(names)


def render_to_pngs(
    split_data,
    split_name: str,
    png_dir: str,
    jsonl_path: str,
    target_dim: int,
    workers: int,
    cache_dir: str | None,
    downloader: "_TarballDownloader | None" = None,
) -> tuple[int, int]:
    """Phase 1: Render all PDF pages to PNG files and write metadata JSONL.

    Three-stage pipeline that overlaps I/O with compute:
      1. Background download: Thread pool downloads tarballs; rendering starts
         as each arrives (shared downloader or creates one internally)
      2. Double-buffer: Extract tarball N+1 in a background thread while
         ProcessPoolExecutor renders tarball N
      3. Background cleanup: shutil.rmtree runs in a thread, never blocks

    Args:
        split_data: HuggingFace dataset split with pdf_relpath, natural_text, etc.
        split_name: Split name for logging
        png_dir: Directory to write PNG files
        jsonl_path: Path to write metadata JSONL
        target_dim: Target longest image dimension for rendering
        workers: Number of parallel rendering workers
        cache_dir: HuggingFace cache directory
        downloader: Optional shared background downloader (renders start as tarballs arrive)

    Returns:
        Tuple of (total rendered, total skipped)
    """
    os.makedirs(png_dir, exist_ok=True)

    # Group rows by tarball
    tarball_groups: dict[str | None, list[tuple[int, dict, str]]] = {}
    for idx, row in enumerate(split_data):
        pdf_relpath = row["pdf_relpath"]
        if ":" in pdf_relpath:
            tarball_name, inner_path = pdf_relpath.split(":", 1)
        else:
            tarball_name = None
            inner_path = pdf_relpath

        if tarball_name not in tarball_groups:
            tarball_groups[tarball_name] = []
        tarball_groups[tarball_name].append((idx, row, inner_path))

    # Filter to valid tarballs
    tarball_order = [name for name in tarball_groups if name is not None]
    num_tarballs = len(tarball_order)
    total_rendered = 0
    total_skipped = sum(len(entries) for name, entries in tarball_groups.items() if name is None)

    logger.info(f"[{split_name}] {num_tarballs} tarballs, {workers} render workers")

    # --- Stage 1: Set up tarball source (shared downloader or create one) ---
    owns_downloader = downloader is None
    if downloader is not None:
        tarball_paths = downloader
        logger.info(f"[{split_name}] Using shared downloader (renders start as tarballs arrive)")
    else:
        logger.info(f"[{split_name}] Starting background download of {num_tarballs} tarballs...")
        tarball_paths = _TarballDownloader(tarball_order, cache_dir)

    # --- Stage 2+3: Double-buffered extract/render + background cleanup ---
    # We use alternating extraction directories (_extract_a, _extract_b) so that
    # extraction of tarball N+1 can overlap with rendering of tarball N.
    extract_dirs = [
        os.path.join(png_dir, "_extract_a"),
        os.path.join(png_dir, "_extract_b"),
    ]
    # Track cleanup threads per buffer to avoid race conditions:
    # when prefetch starts on buffer N, any prior cleanup on that buffer must finish first.
    buffer_cleanup: list[threading.Thread | None] = [None, None]

    def _cleanup_in_background(path: str, buf_idx: int):
        """Remove a directory tree in a background thread."""
        t = threading.Thread(target=shutil.rmtree, args=(path,), kwargs={"ignore_errors": True})
        t.start()
        buffer_cleanup[buf_idx] = t

    prefetch_thread = None
    prefetch_tb_idx = None

    def _start_prefetch(tb_idx: int, buf_idx: int):
        """Download (if needed) + extract tarball in a background thread.

        The thread blocks on downloader.get() until the tarball is downloaded,
        then waits for any prior cleanup on this buffer, then extracts.
        This keeps the main thread free to continue rendering.
        """
        nonlocal prefetch_thread, prefetch_tb_idx
        prefetch_tb_idx = tb_idx

        def _download_and_extract():
            name = tarball_order[tb_idx]
            path = tarball_paths.get(name)  # blocks until downloaded
            if path is None:
                return
            # Wait for any prior cleanup on this buffer to finish
            if buffer_cleanup[buf_idx] is not None:
                buffer_cleanup[buf_idx].join()
                buffer_cleanup[buf_idx] = None
            target_dir = extract_dirs[buf_idx]
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir, ignore_errors=True)
            _extract_tarball(path, target_dir)

        prefetch_thread = threading.Thread(target=_download_and_extract)
        prefetch_thread.start()

    # Kick off extraction of tarball 0
    if num_tarballs > 0:
        _start_prefetch(0, 0)

    with open(jsonl_path, "w") as jsonl_f:
        for tb_idx, tarball_name in enumerate(tarball_order):
            entries = tarball_groups[tarball_name]

            # Wait for this tarball's prefetch (download + extraction) to finish
            current_buf = tb_idx % 2
            if prefetch_thread is not None and prefetch_tb_idx == tb_idx:
                prefetch_thread.join()

            tb_path = tarball_paths.get(tarball_name)

            logger.info(
                f"[{split_name}] Tarball {tb_idx + 1}/{num_tarballs}: "
                f"{tarball_name} ({len(entries)} samples)"
            )

            if tb_path is None:
                total_skipped += len(entries)
                # Start prefetch of next tarball if available
                if tb_idx + 1 < num_tarballs:
                    _start_prefetch(tb_idx + 1, (tb_idx + 1) % 2)
                continue

            extract_dir = extract_dirs[current_buf]

            # Start prefetch of NEXT tarball into the OTHER buffer
            if tb_idx + 1 < num_tarballs:
                _start_prefetch(tb_idx + 1, (tb_idx + 1) % 2)

            if not os.path.exists(extract_dir):
                logger.warning(f"  Extraction dir missing for {tarball_name}, skipping")
                total_skipped += len(entries)
                continue

            # Build render jobs
            render_jobs = []
            job_metadata = {}
            for idx, row, inner_path in entries:
                pdf_path = os.path.join(extract_dir, inner_path)
                if not os.path.exists(pdf_path):
                    total_skipped += 1
                    continue

                png_filename = f"{idx:06d}.png"
                png_path = os.path.join(png_dir, png_filename)
                render_jobs.append((pdf_path, 1, target_dim, png_path))
                job_metadata[png_path] = {
                    "png": png_filename,
                    "text": row["natural_text"],
                    "id": row["id"],
                    "page_number": row["page_number"],
                    "primary_language": row.get("primary_language", ""),
                    "is_table": row.get("is_table", False),
                    "is_diagram": row.get("is_diagram", False),
                }

            if not render_jobs:
                _cleanup_in_background(extract_dir, current_buf)
                continue

            # Render in parallel
            batch_rendered = 0
            t_render = time.monotonic()
            if workers > 1:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(_render_and_save_png, job): job[3] for job in render_jobs
                    }
                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f"  Rendering {split_name}",
                        unit="pages",
                    ):
                        png_path = futures[future]
                        result = future.result()
                        if result is not None:
                            meta = job_metadata[png_path]
                            jsonl_f.write(json.dumps(meta) + "\n")
                            batch_rendered += 1
                        else:
                            total_skipped += 1
            else:
                for job in tqdm(render_jobs, desc=f"  Rendering {split_name}", unit="pages"):
                    result = _render_and_save_png(job)
                    if result is not None:
                        meta = job_metadata[job[3]]
                        jsonl_f.write(json.dumps(meta) + "\n")
                        batch_rendered += 1
                    else:
                        total_skipped += 1

            jsonl_f.flush()
            render_elapsed = time.monotonic() - t_render

            total_rendered += batch_rendered
            logger.info(
                f"  Rendered: {batch_rendered} in {render_elapsed:.1f}s "
                f"({batch_rendered / max(render_elapsed, 0.1):.0f} pages/s), "
                f"cumulative: {total_rendered}"
            )

            # Clean up extracted PDFs in background (don't block next render)
            _cleanup_in_background(extract_dir, current_buf)

    # Wait for any remaining cleanup threads
    for t in buffer_cleanup:
        if t is not None:
            t.join()

    # Shut down internal downloader if we created one
    if owns_downloader:
        tarball_paths.shutdown()

    return total_rendered, total_skipped


# ---------------------------------------------------------------------------
# Phase 2: Build HuggingFace dataset from PNG files + JSONL
# ---------------------------------------------------------------------------


def build_hf_dataset(png_dir: str, jsonl_path: str, output_path: str) -> int:
    """Phase 2: Build a HuggingFace dataset from rendered PNGs and metadata JSONL.

    Uses Dataset.from_generator() for O(1) memory — reads one PNG at a time,
    writes Arrow incrementally.

    Args:
        png_dir: Directory containing PNG files
        jsonl_path: Path to metadata JSONL (one JSON object per line)
        output_path: Where to save the final HF dataset

    Returns:
        Number of samples in the dataset
    """
    # Count lines for progress reporting
    with open(jsonl_path) as f:
        num_samples = sum(1 for _ in f)
    logger.info(f"  Building Arrow dataset from {num_samples} samples...")

    def generator():
        with open(jsonl_path) as f:
            for line in f:
                meta = json.loads(line)
                png_path = os.path.join(png_dir, meta["png"])
                with open(png_path, "rb") as img_f:
                    image_bytes = img_f.read()
                yield {
                    "image": {"bytes": image_bytes, "path": meta["png"]},
                    "text": meta["text"],
                    "id": meta["id"],
                    "page_number": meta["page_number"],
                    "primary_language": meta["primary_language"],
                    "is_table": meta["is_table"],
                    "is_diagram": meta["is_diagram"],
                }

    features = Features(
        {
            "image": Image(),
            "text": Value("string"),
            "id": Value("string"),
            "page_number": Value("int64"),
            "primary_language": Value("string"),
            "is_table": Value("bool"),
            "is_diagram": Value("bool"),
        }
    )

    dataset = Dataset.from_generator(generator, features=features)
    dataset.save_to_disk(output_path)
    return len(dataset)


# ---------------------------------------------------------------------------
# Main: orchestrate both phases
# ---------------------------------------------------------------------------

ALL_SUBSETS = ["00_documents", "01_books", "02_loc_transcripts", "03_national_archives"]


def process_subset(
    subset: str,
    output_dir: str,
    max_samples: int | None = None,
    target_dim: int = 1288,
    cache_dir: str | None = None,
    workers: int = 1,
):
    """Process an olmOCR-mix-1025 subset into a local image+text HF dataset.

    Phase 1: Render all PDF pages to PNG files on disk (parallel, O(1) memory).
    Phase 2: Build HF dataset from PNGs via streaming generator (O(1) memory).

    Args:
        subset: Dataset subset name (e.g., '00_documents')
        output_dir: Where to save the processed dataset
        max_samples: Maximum number of samples to process (None for all)
        target_dim: Target longest image dimension for rendering
        cache_dir: HuggingFace cache directory
        workers: Number of parallel workers for PDF rendering
    """
    repo_id = "allenai/olmOCR-mix-1025"

    logger.info("Fetching source dataset metadata...")
    source_metadata = _get_source_metadata(repo_id, subset)

    logger.info(f"Loading {repo_id} subset: {subset}")
    dataset = load_dataset(repo_id, subset, cache_dir=cache_dir)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Staging area for PNGs + JSONL (inside output dir so it's on the same filesystem)
    staging_dir = output_path / "_staging"
    staging_dir.mkdir(exist_ok=True)

    # Prepare split data and pre-download ALL tarballs across splits at once
    splits = {}
    all_tarball_names = []
    for split_name in ["train", "eval"]:
        if split_name not in dataset:
            continue
        split_data = dataset[split_name]
        if max_samples and split_name == "train":
            split_data = split_data.select(range(min(max_samples, len(split_data))))
        splits[split_name] = split_data
        all_tarball_names.extend(_collect_tarball_names(split_data))

    # Start background downloads (non-blocking) — rendering starts as tarballs arrive
    all_tarball_names = sorted(set(all_tarball_names))
    logger.info(f"Starting background download of {len(all_tarball_names)} tarballs...")
    downloader = _TarballDownloader(all_tarball_names, cache_dir)

    split_counts = {}
    for split_name, split_data in splits.items():
        logger.info(f"Processing {split_name}: {len(split_data)} samples")

        png_dir = str(staging_dir / split_name / "images")
        jsonl_path = str(staging_dir / split_name / "metadata.jsonl")
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

        # Phase 1: Render
        logger.info(f"=== Phase 1: Rendering {split_name} ===")
        rendered, skipped = render_to_pngs(
            split_data=split_data,
            split_name=split_name,
            png_dir=png_dir,
            jsonl_path=jsonl_path,
            target_dim=target_dim,
            workers=workers,
            cache_dir=cache_dir,
            downloader=downloader,
        )
        logger.info(f"  Phase 1 complete: {rendered} rendered, {skipped} skipped")

        if rendered == 0:
            logger.warning(f"  No samples rendered for {split_name}, skipping dataset build")
            split_counts[split_name] = {"processed": 0, "skipped": skipped}
            continue

        # Phase 2: Build HF dataset
        logger.info(f"=== Phase 2: Building {split_name} dataset ===")
        split_output = str(output_path / split_name)
        num_samples = build_hf_dataset(png_dir, jsonl_path, split_output)
        logger.info(f"  Phase 2 complete: {num_samples} samples saved to {split_output}")
        split_counts[split_name] = {"processed": num_samples, "skipped": skipped}

    # Ensure all background downloads are complete
    downloader.shutdown()

    # Clean up staging area
    logger.info("Cleaning up staging area...")
    shutil.rmtree(str(staging_dir), ignore_errors=True)

    # Save provenance metadata
    metadata = {
        "source": source_metadata,
        "preparation": {
            "script": "scripts/data/prepare_ocr_data.py",
            "timestamp": datetime.now(UTC).isoformat(),
            "params": {
                "subset": subset,
                "max_samples": max_samples,
                "target_dim": target_dim,
                "workers": workers,
            },
        },
        "output": {
            "path": str(output_path),
            "splits": split_counts,
        },
    }
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Provenance metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Pre-process olmOCR-mix-1025 for training")
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        choices=ALL_SUBSETS + ["all"],
        help="Dataset subset to process ('all' for full 268K)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/ocr/olmocr-mix-full",
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum training samples to process per subset (None for all)",
    )
    parser.add_argument(
        "--target_dim", type=int, default=1288, help="Target longest image dimension in pixels"
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="HuggingFace cache directory")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for PDF rendering (default: all CPUs)",
    )
    args = parser.parse_args()

    # Prefer SLURM allocation, then explicit arg, then all CPUs
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or os.cpu_count() or 1

    with logging_redirect_tqdm():
        logger.info(f"Using {workers} workers")

        subsets = ALL_SUBSETS if args.subset == "all" else [args.subset]
        for subset in subsets:
            output_dir = os.path.join(args.output, subset)
            logger.info(f"=== Subset: {subset} -> {output_dir} ===")
            process_subset(
                subset=subset,
                output_dir=output_dir,
                max_samples=args.max_samples,
                target_dim=args.target_dim,
                cache_dir=args.cache_dir,
                workers=workers,
            )

        logger.info("Done!")


if __name__ == "__main__":
    main()
