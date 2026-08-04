"""Formatting-specific benchmark for OCR models.

Standalone eval that measures inline formatting preservation (bold, italic,
strikethrough) on document pages. Runs independently from OmniDocBench and
olmOCR-Bench to avoid contaminating those standardized metrics.

Uses the same dataset handlers as training for consistency.

Usage:
    # Default: all eval pages
    python scripts/benchmarks/formatting_bench.py --checkpoint outputs/fmt_sft

    # Stratified sample (faster, for diagnostic runs)
    python scripts/benchmarks/formatting_bench.py --checkpoint outputs/fmt_sft --max_pages 60

    # New run with base model
    python scripts/benchmarks/formatting_bench.py --model Qwen/Qwen3-VL-2B-Instruct

    # With explicit output dir
    python scripts/benchmarks/formatting_bench.py --checkpoint outputs/fmt_sft --output outputs/bench/formatting/my-run

    # Resume (model loaded from metadata.json)
    python scripts/benchmarks/formatting_bench.py --output outputs/bench/formatting/my-run

    # Sharded run
    python scripts/benchmarks/formatting_bench.py --checkpoint X --output D --shard 1/4

    # Score only (after all shards complete)
    python scripts/benchmarks/formatting_bench.py --score_only --output outputs/bench/formatting/my-run

    # Score existing predictions against GT files (ad-hoc)
    python scripts/benchmarks/formatting_bench.py --pred_dir outputs/bench/omnidoc/my-run/markdowns --gt_dir data/ocr/omnidocbench/markdown
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ocr_utils import OCR_SYSTEM_PROMPT
from src.evaluation.bench_utils import (
    add_common_bench_args,
    apply_shard,
    build_metadata,
    generate_markdowns,
    is_shutdown_requested,
    register_bench_job,
    remove_bench_job,
    save_bench_output,
    save_metadata,
    setup_logging,
    validate_bench_args,
)
from src.evaluation.formatting_metrics import (
    compute_aggregate,
    score_formatting,
)
from src.inference import InferenceEngine
from src.utils.environment import get_num_workers

setup_logging()
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_BASE = "outputs/bench/formatting"
DEFAULT_MAX_NEW_TOKENS = 2048


# Scoring functions imported from src/evaluation/formatting_metrics


# ---------------------------------------------------------------------------
# Data loading (delegates to handlers for consistency)
# ---------------------------------------------------------------------------


def load_eval_paths(data_dir: str, dataset: str = "formatting") -> list[dict]:
    """Load eval file paths using the registered dataset handler.

    Args:
        data_dir: Root data directory for the dataset.
        dataset: Dataset name in the handler registry.

    Returns:
        List of dicts with "image_path" and "gt_path" as Path objects.
    """
    from src.data.handlers import get_handler

    handler = get_handler(dataset)
    handler.prepare(data_dir)
    paths = handler.load_paths(data_dir, split="eval")
    if not paths:
        raise ValueError(f"No eval samples found in {data_dir}")

    logger.info(f"Loaded {len(paths)} eval paths from {dataset} ({data_dir})")
    return paths


# ---------------------------------------------------------------------------
# Scoring from existing predictions (ad-hoc --pred_dir/--gt_dir mode)
# ---------------------------------------------------------------------------


def _score_formatting_worker(args):
    """Score one page's formatting. Top-level function for ProcessPoolExecutor pickling.

    Args:
        args: Tuple of (stem, pred_path, gt_path).

    Returns:
        Dict with formatting scores and stem.
    """
    stem, pred_path, gt_path = args
    try:
        pred_md = Path(pred_path).read_text(errors="replace")
        gt_md = Path(gt_path).read_text(errors="replace").strip()
        scores = score_formatting(pred_md, gt_md)
        scores["stem"] = stem
        return scores
    except Exception as e:
        logger.error(f"Error scoring {stem}: {e}")
        return {"stem": stem, "error": str(e)}


def _run_scoring(work_items: list[tuple], num_workers: int) -> list[dict]:
    """Run formatting scoring in parallel or serial.

    Args:
        work_items: List of (stem, pred_path, gt_path) tuples.
        num_workers: Number of parallel workers.

    Returns:
        List of per-page result dicts.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    results = []
    if num_workers > 1 and len(work_items) > 10:
        logger.info(f"Scoring with {num_workers} parallel workers")
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_score_formatting_worker, item): item[0] for item in work_items}
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Scoring formatting"
            ):
                results.append(future.result())
    else:
        for item in tqdm(work_items, desc="Scoring formatting"):
            results.append(_score_formatting_worker(item))
    return results


def score_from_files(pred_dir: str, gt_dir: str, num_workers: int = 1) -> list[dict]:
    """Score pre-existing prediction files against GT.

    Args:
        pred_dir: Directory with predicted .md files.
        gt_dir: Directory with ground truth .md files.
        num_workers: Number of parallel workers for scoring.

    Returns:
        List of per-page result dicts.
    """
    gt_files = {p.stem: p for p in Path(gt_dir).glob("*.md")}
    pred_files = {p.stem: p for p in Path(pred_dir).glob("*.md")}

    common = sorted(set(gt_files) & set(pred_files))
    if not common:
        logger.warning(f"No matching files between {pred_dir} and {gt_dir}")
        return []

    logger.info(f"Scoring {len(common)} pages ({len(gt_files)} GT, {len(pred_files)} pred)")
    work_items = [(stem, str(pred_files[stem]), str(gt_files[stem])) for stem in common]
    return _run_scoring(work_items, num_workers)


# ---------------------------------------------------------------------------
# Scoring from markdowns directory (handler paths provide GT)
# ---------------------------------------------------------------------------


def score_from_markdowns(paths: list[dict], markdowns_dir: str, num_workers: int = 1) -> list[dict]:
    """Score generated markdowns against GT from handler paths.

    Args:
        paths: Eval path dicts with "image_path" and "gt_path".
        markdowns_dir: Directory with generated .md files.
        num_workers: Number of parallel workers for scoring.

    Returns:
        List of per-page result dicts.
    """
    work_items = []
    for entry in paths:
        stem = entry["image_path"].stem
        md_path = os.path.join(markdowns_dir, f"{stem}.md")
        if not os.path.exists(md_path):
            continue
        work_items.append((stem, md_path, str(entry["gt_path"])))
    return _run_scoring(work_items, num_workers)


# compute_aggregate imported from src/evaluation/formatting_metrics


def select_stratified(paths: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Sample n pages stratified by formatting type presence.

    Ensures rare formatting types are represented by reserving ~1/3 of slots
    for pages that contain the least common type. Deterministic via fixed seed.

    Args:
        paths: Eval path dicts with "gt_path".
        n: Total pages to select.
        seed: Random seed for reproducibility.

    Returns:
        List of n (or fewer, if dataset is small) path dicts.
    """
    import random
    import re

    if len(paths) <= n:
        return paths

    def has_rare_type(gt_path) -> bool:
        try:
            return bool(re.search(r"~~[^~]+~~", Path(gt_path).read_text(errors="replace")))
        except Exception:
            return False

    rare_pages = [p for p in paths if has_rare_type(p["gt_path"])]
    other_pages = [p for p in paths if not has_rare_type(p["gt_path"])]

    rng = random.Random(seed)
    n_rare = min(len(rare_pages), max(1, n // 3))
    n_other = n - n_rare

    selected = rng.sample(rare_pages, n_rare) + rng.sample(
        other_pages, min(n_other, len(other_pages))
    )
    logger.info(
        f"Stratified sample: {n_rare} rare-type + {len(selected) - n_rare} other = {len(selected)} pages"
    )
    return selected


def print_results(agg: dict):
    """Print formatted results.

    Args:
        agg: Aggregate results dict.
    """
    scores = agg["scores"]
    counts = agg["counts"]

    logger.info("=" * 60)
    logger.info("Formatting Benchmark Results")
    logger.info("=" * 60)
    logger.info(f"Total pages:              {agg['total_pages']}")
    logger.info(f"Pages with formatting:    {agg['pages_with_formatting']}")
    logger.info("-" * 60)

    metric_names = {
        "bold": "Bold",
        "italic": "Italic",
        "strikethrough": "Strikethrough",
        "overall": "Overall",
    }
    logger.info(f"  {'Metric':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}  {'n':>5}")
    logger.info(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10}  {'-' * 5}")
    for key, name in metric_names.items():
        val = scores.get(key)
        n = counts.get(key, 0)
        if val is not None:
            if isinstance(val, dict):
                p, r, f1 = val["precision"], val["recall"], val["f1"]
                logger.info(
                    f"  {name:<20} {p * 100:>9.2f}% {r * 100:>9.2f}% {f1 * 100:>9.2f}%  {n:>5}"
                )
            else:
                # Old format (f1 only)
                logger.info(f"  {name:<20} {'':>10} {'':>10} {val * 100:>9.2f}%  {n:>5}")
        else:
            logger.info(f"  {name:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    logger.info("-" * 60)
    gt = agg["gt_span_totals"]
    pred = agg["pred_span_totals"]
    logger.info(
        f"GT spans:   bold={gt['bold']}, italic={gt['italic']}, strike={gt['strikethrough']}"
    )
    logger.info(
        f"Pred spans: bold={pred['bold']}, italic={pred['italic']}, strike={pred['strikethrough']}"
    )

    hall = agg.get("hallucination", {})
    if hall.get("total_pred", 0) > 0:
        logger.info("-" * 60)
        logger.info("Predicted span breakdown:")
        logger.info(f"  Matched:        {hall['matched']:>5}  (correct formatting)")
        logger.info(f"  Over-formatted: {hall['over_formatted']:>5}  (real text, wrong formatting)")
        logger.info(f"  Hallucinated:   {hall['hallucinated']:>5}  (fabricated content)")
        logger.info(f"  Hallucination rate: {hall['hallucination_rate'] * 100:.2f}%")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_formatting_results(
    output_dir: str, aggregates: dict, page_results: list, metadata: dict | None = None
):
    """Save results, metadata, and per-page scores to output directory.

    Args:
        output_dir: Output directory path.
        aggregates: Aggregate scores dict.
        page_results: Per-page result dicts.
        metadata: Run metadata dict.
    """
    results = {"aggregates": aggregates}
    save_bench_output(output_dir, results, metadata)

    with open(os.path.join(output_dir, "page_scores.jsonl"), "w") as f:
        for r in sorted(page_results, key=lambda x: x.get("stem", "")):
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Formatting-specific benchmark")
    parser.add_argument(
        "--dataset",
        type=str,
        default="formatting",
        help="Dataset handler name (default: formatting)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/ocr/formatting",
        help="Data root directory (default: data/ocr/formatting)",
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default=None,
        help="Directory with predicted .md files (ad-hoc score mode)",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=None,
        help="Directory with ground truth .md files (for --pred_dir mode)",
    )

    add_common_bench_args(parser, max_new_tokens=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--max_pages",
        type=int,
        default=None,
        help="Limit pages, sampled stratified by formatting type. Omit to run all eval pages.",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Mode 1: Ad-hoc score from file directories (--pred_dir / --gt_dir)
    # -----------------------------------------------------------------------
    if args.pred_dir:
        if not args.gt_dir:
            parser.error("--gt_dir required for --pred_dir mode")
        num_workers = get_num_workers()
        results = score_from_files(args.pred_dir, args.gt_dir, num_workers=num_workers)
        if not results:
            logger.warning("No results to aggregate")
            sys.exit(1)
        agg = compute_aggregate(results)
        print_results(agg)
        if args.output:
            save_formatting_results(args.output, agg, results)
        return

    # -----------------------------------------------------------------------
    # Mode 2: Full bench (inference + score, or score_only)
    # -----------------------------------------------------------------------
    output_dir, markdowns_dir = validate_bench_args(args, parser, DEFAULT_OUTPUT_BASE)

    # Generation phase
    if not args.score_only:
        # Load eval paths via handler
        paths = load_eval_paths(args.data_dir, args.dataset)
        if args.max_pages:
            paths = select_stratified(paths, args.max_pages)
        # Apply sharding
        if args.shard:
            if not args.output:
                parser.error("--shard requires --output so all shards write to the same directory")
            paths = apply_shard(paths, args.shard)

        register_bench_job(output_dir, "formatting", args.shard)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        engine = InferenceEngine.load(
            model_id=args.model,
            checkpoint_path=args.checkpoint,
            model_type="vlm",
            device=device,
            task="ocr",
        )

        metadata = build_metadata(
            args,
            engine,
            "formatting",
            {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
            },
        )
        save_metadata(output_dir, metadata)

        # Build work items
        work_items = [
            {
                "md_path": os.path.join(markdowns_dir, p["image_path"].stem + ".md"),
                "image_path": p["image_path"],
            }
            for p in paths
        ]

        def _load_image(item):
            with Image.open(item["image_path"]) as im:
                image = im.convert("RGB")
                image.load()
            return image

        generate_markdowns(
            engine,
            work_items,
            OCR_SYSTEM_PROMPT,
            args.max_new_tokens,
            markdowns_dir,
            image_fn=_load_image,
        )
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
        logger.info(
            f"Shard {args.shard} complete - run with --score_only to score after all shards finish"
        )
        return

    # Reload all paths for scoring (not just this shard)
    all_paths = load_eval_paths(args.data_dir, args.dataset)
    if args.max_pages:
        all_paths = select_stratified(all_paths, args.max_pages)
    num_workers = get_num_workers()
    results = score_from_markdowns(all_paths, markdowns_dir, num_workers=num_workers)
    if not results:
        logger.warning("No results to aggregate")
        sys.exit(1)

    agg = compute_aggregate(results)
    print_results(agg)
    save_formatting_results(output_dir, agg, results, metadata)

    remove_bench_job()


if __name__ == "__main__":
    main()
