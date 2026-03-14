"""Evaluate VLM models on OmniDocBench formatting quality.

Runs any HuggingFace VLM (or LoRA checkpoint) against OmniDocBench's 1,355 pages,
scoring predictions across four formatting dimensions: text, tables, formulas,
and reading order. Produces per-page and aggregate scores.

Usage:
    # New run
    python scripts/benchmarks/omnidoc_bench.py --model Qwen/Qwen3-VL-2B-Instruct

    # With explicit output dir
    python scripts/benchmarks/omnidoc_bench.py --model Qwen/Qwen3-VL-2B-Instruct --output outputs/bench/omnidoc/my-run

    # Resume (model loaded from metadata.json)
    python scripts/benchmarks/omnidoc_bench.py --output outputs/bench/omnidoc/my-run

    # Sharded run
    python scripts/benchmarks/omnidoc_bench.py --model X --output D --shard 1/4

    # Score only (after all shards complete)
    python scripts/benchmarks/omnidoc_bench.py --score_only --output outputs/bench/omnidoc/my-run

"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.handlers import get_handler
from src.utils.environment import get_num_workers
from src.evaluation.bench_utils import (
    add_common_bench_args,
    apply_shard,
    build_metadata,
    cleanup_temp_files,
    is_shutdown_requested,
    register_bench_job,
    remove_bench_job,
    save_bench_output,
    save_metadata,
    setup_logging,
    validate_bench_args,
)
from src.evaluation.document_metrics import (
    score_page,
    compute_aggregate,
)
from src.inference import InferenceEngine

setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = "data/ocr/omnidocbench"
DEFAULT_OUTPUT_BASE = "outputs/bench/omnidoc"
DEFAULT_MAX_NEW_TOKENS = 2048
OCR_PROMPT = (
    "Below is an image of a document page. Extract all text content "
    "and output it in clean markdown format, preserving the document's "
    "structure (headings, lists, tables, etc.)."
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_pages(data_dir: str):
    """Load OmniDocBench image/GT paths via dataset handler.

    Args:
        data_dir: Path to OmniDocBench dataset root.

    Returns:
        List of dicts with "image_path", "gt_path" (Path), and "stem" (str).
    """
    handler = get_handler("omnidocbench")
    handler.prepare(data_dir)
    pages = handler.load_paths(data_dir)
    # Add stem for convenience (used as file key throughout)
    for p in pages:
        p["stem"] = p["image_path"].stem
    logger.info(f"Loaded {len(pages)} pages with GT markdown")
    return pages


# Scoring functions imported from src/evaluation/document_metrics


def _score_page_worker(args):
    """Score one page. Top-level function for ProcessPoolExecutor pickling.

    Args:
        args: Tuple of (stem, pred_path, gt_path).

    Returns:
        Tuple of (stem, scores_dict).
    """
    stem, pred_path, gt_path = args
    try:
        with open(pred_path) as f:
            pred_md = f.read()
        with open(gt_path) as f:
            gt_md = f.read()
        scores = score_page(pred_md, gt_md)
        return stem, scores
    except Exception as e:
        logger.error(f"Error scoring {stem}: {e}")
        return stem, {"text": 0.0, "table": None, "formula": None, "reading_order": None, "error": str(e)}


def score_all(pages, markdowns_dir: str, num_workers: int = 1):
    """Score all pages against GT markdown.

    Args:
        pages: List of (stem, image_path, gt_path) tuples.
        markdowns_dir: Directory with predicted .md files.
        num_workers: Parallel workers for scoring.

    Returns:
        List of (stem, scores_dict) tuples for scored pages.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    work_items = []
    missing = 0
    for page in pages:
        stem, gt_path = page["stem"], page["gt_path"]
        pred_path = os.path.join(markdowns_dir, stem + ".md")
        if not os.path.exists(pred_path):
            missing += 1
            continue
        work_items.append((stem, pred_path, gt_path))

    if missing > 0:
        logger.info(f"{missing} pages missing predictions, {len(work_items)} to score")

    results = []
    if num_workers > 1 and len(work_items) > 10:
        logger.info(f"Scoring with {num_workers} parallel workers")
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_score_page_worker, item): item[0] for item in work_items}
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Scoring", unit="pg")
            for future in pbar:
                stem, scores = future.result()
                results.append((stem, scores))
    else:
        for item in (pbar := tqdm(work_items, desc="Scoring", unit="pg")):
            stem, scores = _score_page_worker(item)
            results.append((stem, scores))

    return results


# compute_aggregate imported from src/evaluation/document_metrics


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_results(aggregates: dict, num_scored: int, num_total: int):
    """Print formatted results table.

    Args:
        aggregates: Dict from compute_aggregate.
        num_scored: Number of pages scored.
        num_total: Total pages.
    """
    logger.info("=" * 60)
    logger.info("OmniDocBench Results")
    logger.info("=" * 60)
    logger.info(f"Pages scored: {num_scored}/{num_total}")
    logger.info("-" * 60)

    metric_names = {
        "text": "Text (Edit Dist)",
        "table": "Tables (TEDS)",
        "formula": "Formulas (Edit Dist)",
        "reading_order": "Reading Order (Edit Dist)",
    }
    for key, name in metric_names.items():
        val = aggregates.get(key)
        count = aggregates.get("counts", {}).get(key, 0)
        if val is not None:
            logger.info(f"  {name:<30} {val*100:>6.2f}%  (n={count})")
        else:
            logger.info(f"  {name:<30}    N/A")

    logger.info("-" * 60)
    overall = aggregates.get("overall", 0.0)
    logger.info(f"  {'Overall':<30} {overall*100:>6.2f}%")
    logger.info("=" * 60)


def save_bench_results(output_dir: str, aggregates: dict, page_results: list,
                      metadata: dict, num_scored: int, num_total: int):
    """Save results, metadata, and per-page scores to output directory.

    Args:
        output_dir: Output directory path.
        aggregates: Aggregate scores dict.
        page_results: Per-page (stem, scores) list.
        metadata: Run metadata dict.
        num_scored: Number of pages scored.
        num_total: Total pages.
    """
    results = {
        "aggregates": aggregates,
        "num_scored": num_scored,
        "num_total": num_total,
    }
    save_bench_output(output_dir, results, metadata)

    # Per-page details
    with open(os.path.join(output_dir, "page_scores.jsonl"), "w") as f:
        for stem, scores in sorted(page_results, key=lambda x: x[0]):
            f.write(json.dumps({"page": stem, **scores}) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM on OmniDocBench formatting")
    add_common_bench_args(parser, max_new_tokens=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Path to OmniDocBench data root")
    args = parser.parse_args()

    output_dir, markdowns_dir = validate_bench_args(args, parser, DEFAULT_OUTPUT_BASE)

    # Load pages via handler (prepare + load_paths)
    pages = load_pages(args.data_dir)

    # Apply sharding
    if args.shard:
        if not args.output:
            parser.error("--shard requires --output so all shards write to the same directory")
        pages = apply_shard(pages, args.shard)

    # Register in incomplete jobs
    if not args.score_only:
        register_bench_job(output_dir, "omnidoc", args.shard)

    # Generation phase
    if not args.score_only:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        engine = InferenceEngine.load(
            model_id=args.model, checkpoint_path=args.checkpoint,
            model_type="vlm", device=device,
        )

        metadata = build_metadata(args, engine, "omnidoc", {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "prompt": OCR_PROMPT,
        })
        save_metadata(output_dir, metadata)

        cleanup_temp_files(markdowns_dir)

        # Count already done
        already_done = sum(
            1 for p in pages
            if os.path.exists(os.path.join(markdowns_dir, p["stem"] + ".md"))
        )

        logger.info(f"Starting generation ({already_done} done, {len(pages) - already_done} remaining)...")
        generated = 0
        errors = 0

        # Build work list (skip already done)
        pending = []
        for page in pages:
            stem, image_path = page["stem"], page["image_path"]
            md_path = os.path.join(markdowns_dir, stem + ".md")
            if not os.path.exists(md_path):
                pending.append((stem, image_path, md_path))

        def _load_image(image_path):
            """Load and convert image (runs in background thread)."""
            with Image.open(image_path) as im:
                image = im.convert("RGB")
                image.load()
            return image

        pbar = tqdm(total=len(pages), desc="Generating", unit="pg", initial=already_done)
        first_generate = True

        # Prefetch: load next image while GPU generates current one
        prefetch_pool = ThreadPoolExecutor(max_workers=1)
        prefetch_future = None

        for idx, (stem, image_path, md_path) in enumerate(pending):
            if is_shutdown_requested():
                logger.info(f"Shutting down gracefully. Generated {generated} this session.")
                break

            try:
                # Get current image (from prefetch or direct load)
                if prefetch_future is not None:
                    image = prefetch_future.result()
                else:
                    image = _load_image(image_path)

                # Start prefetching next image
                if idx + 1 < len(pending):
                    prefetch_future = prefetch_pool.submit(_load_image, pending[idx + 1][1])
                else:
                    prefetch_future = None

                if first_generate:
                    pbar.clear()
                results = engine.generate(
                    [(image, OCR_PROMPT)], max_new_tokens=args.max_new_tokens
                )
                if first_generate:
                    first_generate = False
                    pbar.refresh()

                markdown = results[0]
                tmp_path = md_path + ".tmp"
                with open(tmp_path, "w") as f:
                    f.write(markdown)
                os.replace(tmp_path, md_path)
                generated += 1

            except Exception as e:
                if is_shutdown_requested():
                    logger.info(f"Interrupted on {stem} - will retry on resume")
                    break
                logger.error(f"Failed on {stem}: {e}")
                errors += 1
                prefetch_future = None

            pbar.set_postfix(gen=generated, err=errors)
            pbar.update(1)

        prefetch_pool.shutdown(wait=False)

        pbar.close()
        logger.info(f"Generation complete: {generated} generated, {already_done} skipped, {errors} errors")

        cleanup_temp_files(markdowns_dir)

        save_metadata(output_dir, metadata)

        if is_shutdown_requested():
            logger.info(f"Interrupted. Auto-resubmit will resume from: {output_dir}")
            sys.exit(0)
    else:
        logger.info("Score-only mode - skipping generation")
        metadata = {}
        meta_path = os.path.join(output_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)

    # Scoring phase (skip for sharded runs)
    if args.shard and not args.score_only:
        remove_bench_job()
        logger.info(f"Shard {args.shard} complete - run with --score_only to score after all shards finish")
        return

    # Reload all pages for scoring (not just this shard)
    all_pages = load_pages(args.data_dir)
    if not all_pages:
        logger.error(f"No pages found in {args.data_dir}")
        return

    available = sum(
        1 for p in all_pages
        if os.path.exists(os.path.join(markdowns_dir, p["stem"] + ".md"))
    )
    logger.info(f"Scoring {available}/{len(all_pages)} pages ({available/len(all_pages)*100:.1f}% available)")

    num_workers = get_num_workers()

    page_results = score_all(all_pages, markdowns_dir, num_workers=num_workers)
    aggregates = compute_aggregate(page_results)

    print_results(aggregates, available, len(all_pages))
    save_bench_results(output_dir, aggregates, page_results, metadata, available, len(all_pages))

    remove_bench_job()


if __name__ == "__main__":
    main()
