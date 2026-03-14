"""Evaluate VLM models on olmOCR-Bench.

Runs any HuggingFace VLM (or LoRA checkpoint) against the olmOCR-Bench benchmark,
producing standardized pass/fail scores comparable to published results.

Usage:
    # New run (auto-derives output dir)
    python scripts/benchmarks/olmocr_bench.py --model allenai/olmOCR-2-7B-1025-FP8

    # New run with explicit output dir
    python scripts/benchmarks/olmocr_bench.py --model Qwen/Qwen3-VL-2B-Instruct --output outputs/bench/olmocr/my-run

    # Resume (model loaded from metadata.json)
    python scripts/benchmarks/olmocr_bench.py --output outputs/bench/olmocr/my-run

    # Sharded run
    python scripts/benchmarks/olmocr_bench.py --model X --output D --shard 1/4

    # Score only (after all shards complete)
    python scripts/benchmarks/olmocr_bench.py --score_only --output outputs/bench/olmocr/my-run

"""

import argparse
import glob
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path

import torch
import pypdfium2
from PIL import Image
from tqdm import tqdm

# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
from src.inference import InferenceEngine

setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_BENCH_DIR = "data/olmocr-bench"
DEFAULT_OUTPUT_BASE = "outputs/bench/olmocr"
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_DPI = 150
OCR_PROMPT = "Extract all text from this document page. Output clean markdown."

# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------
def render_pdf_page(pdf_path: str, page_num: int, dpi: int = 150) -> Image.Image:
    """Render a single PDF page to a PIL Image.

    Args:
        pdf_path: Path to PDF file
        page_num: 1-indexed page number
        dpi: Render resolution

    Returns:
        PIL Image of the rendered page
    """
    pdf_doc = pypdfium2.PdfDocument(pdf_path)
    if page_num < 1 or page_num > len(pdf_doc):
        raise ValueError(f"Page {page_num} out of range for {pdf_path} ({len(pdf_doc)} pages)")
    page = pdf_doc[page_num - 1]
    bitmap = page.render(scale=dpi / 72)
    image = bitmap.to_pil()
    pdf_doc.close()
    return image


# ---------------------------------------------------------------------------
# Benchmark data loading
# ---------------------------------------------------------------------------
def load_bench_data(bench_dir: str):
    """Load olmOCR-Bench test cases and resolve PDF paths.

    Dataset layout (flat JSONL + nested PDFs):
        bench_data/
          arxiv_math.jsonl
          headers_footers.jsonl
          ...
          pdfs/
            arxiv_math/*.pdf
            headers_footers/*.pdf
            ...

    Each test's "pdf" field is a relative path like "arxiv_math/2503.04048_pg46.pdf"
    which maps to bench_data/pdfs/arxiv_math/2503.04048_pg46.pdf.

    Args:
        bench_dir: Path to olmOCR-bench dataset root

    Returns:
        Tuple of (all_tests, pdf_base_dir)
        - all_tests: flat list of test dicts (with _category and _jsonl fields)
        - pdf_base_dir: absolute path to bench_data/pdfs/ for resolving test["pdf"]
    """
    bench_data_dir = os.path.join(bench_dir, "bench_data")
    if not os.path.isdir(bench_data_dir):
        raise FileNotFoundError(
            f"bench_data/ not found in {bench_dir}. Run:\n"
            f"  git clone https://huggingface.co/datasets/allenai/olmOCR-bench {bench_dir}"
        )

    pdf_base_dir = os.path.join(bench_data_dir, "pdfs")
    all_tests = []

    # JSONL files sit at bench_data/ root (one per category)
    for jsonl_path in sorted(glob.glob(os.path.join(bench_data_dir, "*.jsonl"))):
        category = os.path.splitext(os.path.basename(jsonl_path))[0]
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                test = json.loads(line)
                test["_category"] = category
                test["_jsonl"] = os.path.basename(jsonl_path)
                all_tests.append(test)

    logger.info(f"Loaded {len(all_tests)} tests from "
                f"{len(set(t['_category'] for t in all_tests))} categories")
    return all_tests, os.path.abspath(pdf_base_dir)


def pdf_to_md_name(pdf_rel: str, page: int) -> str:
    """Convert a test's pdf relative path + page to a flat markdown filename.

    Args:
        pdf_rel: Relative pdf path from test, e.g. "arxiv_math/2503.04048_pg46.pdf"
        page: Page number

    Returns:
        Flat filename like "arxiv_math_2503.04048_pg46_pg1.md"
    """
    stem = os.path.splitext(pdf_rel)[0].replace("/", "_").replace("\\", "_")
    return f"{stem}_pg{page}.md"


def collect_unique_pages(all_tests):
    """Collect unique (pdf_rel, page) pairs from test cases.

    Args:
        all_tests: List of test dicts

    Returns:
        Sorted list of (pdf_relative_path, page_number) tuples
    """
    pages = set()
    for test in all_tests:
        pages.add((test["pdf"], test["page"]))
    return sorted(pages)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _ensure_playwright_chromium():
    """Auto-install Playwright's Chromium if not already present.

    Playwright bundles its own patched Chromium — this can't be installed via
    conda/pip. We detect and install it automatically on first run.

    Returns:
        True if Chromium is available (or was just installed), False otherwise.
    """
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            # Try to get the executable — raises if not installed
            p.chromium.executable_path
        return True
    except Exception:
        pass

    # Chromium not found — try to install it
    logger.info("Playwright Chromium not found. Installing...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            logger.info("Playwright Chromium installed successfully")
            return True
        else:
            logger.warning(f"Playwright Chromium install failed: {result.stderr.strip()}")
            return False
    except Exception as e:
        logger.warning(f"Could not install Playwright Chromium: {e}")
        return False


def _check_olmocr_scoring():
    """Check olmocr bench scoring availability.

    Tries three tiers:
    1. Full olmocr with Playwright Chromium (auto-installs if missing)
    2. olmocr without math tests (mock katex if Playwright unavailable)
    3. Fallback text-only scoring

    Returns:
        Tuple of (load_single_test function or None, set of skipped test types, mock_katex bool)
    """
    skip_types = set()
    try:
        from olmocr.bench.tests import load_single_test
        logger.info("Using olmocr.bench.tests for scoring (exact comparability)")
        return load_single_test, skip_types, False
    except ImportError as e:
        if "playwright" in str(e) or "katex" in str(e):
            # Playwright package exists but Chromium binary may be missing,
            # or katex import fails because Playwright isn't working
            if _ensure_playwright_chromium():
                # Retry import after installing Chromium
                try:
                    from olmocr.bench.tests import load_single_test
                    logger.info("Using olmocr.bench.tests for scoring (exact comparability)")
                    return load_single_test, skip_types, False
                except ImportError:
                    pass

            # Chromium unavailable - fall back to scoring without math
            logger.warning("Playwright Chromium unavailable - math tests will be skipped")
            try:
                import unittest.mock
                katex_mock = unittest.mock.MagicMock()
                sys.modules["olmocr.bench.katex"] = katex_mock
                sys.modules["olmocr.bench.katex.render"] = katex_mock
                from olmocr.bench.tests import load_single_test
                skip_types.add("math")
                return load_single_test, skip_types, True
            except Exception:
                pass
        logger.warning("olmocr not installed - using fallback scoring (text presence/absence only)")
        return None, skip_types, False


def _score_single_test(args):
    """Score one test against its markdown. Top-level function for pickling.

    Args:
        args: Tuple of (test_dict, markdowns_dir, use_olmocr, mock_katex)

    Returns:
        Tuple of (test, result_dict)
    """
    test, markdowns_dir, use_olmocr, mock_katex = args

    # Apply katex mock in worker if parent needed it
    if use_olmocr and mock_katex:
        import unittest.mock
        katex_mock = unittest.mock.MagicMock()
        sys.modules.setdefault("olmocr.bench.katex", katex_mock)
        sys.modules.setdefault("olmocr.bench.katex.render", katex_mock)

    md_name = pdf_to_md_name(test["pdf"], test["page"])
    md_path = os.path.join(markdowns_dir, md_name)

    if not os.path.exists(md_path):
        return test, {"test_id": test.get("id", "unknown"), "status": "missing", "result": "skip"}

    with open(md_path) as f:
        md_content = f.read()

    if use_olmocr:
        try:
            from olmocr.bench.tests import load_single_test
            test_data = {k: v for k, v in test.items() if not k.startswith("_")}
            test_obj = load_single_test(test_data)
            passed, explanation = test_obj.run(md_content)
            return test, {"test_id": test.get("id", "unknown"), "status": "scored",
                          "result": "pass" if passed else "fail", "explanation": explanation}
        except Exception as e:
            return test, {"test_id": test.get("id", "unknown"), "status": "scored",
                          "result": "fail", "explanation": f"error: {e}"}
    else:
        return test, _fallback_score(test, md_content)


def score_tests(all_tests, markdowns_dir: str, num_workers: int = 1):
    """Score all tests against generated markdowns using olmocr test classes.

    Tries to import olmocr.bench.tests for exact comparability.
    Falls back to simple text presence/absence checks if olmocr is not installed.
    If playwright is missing, math tests are skipped gracefully.

    Args:
        all_tests: List of test dicts from JSONL
        markdowns_dir: Directory containing generated .md files
        num_workers: Number of parallel workers for scoring (default: 1)

    Returns:
        Tuple of (results_by_category, results_by_jsonl, individual_results)
    """
    load_single_test, skip_types, mock_katex = _check_olmocr_scoring()
    use_olmocr = load_single_test is not None

    results_by_category = defaultdict(lambda: {"passed": 0, "failed": 0, "missing": 0, "total": 0})
    results_by_jsonl = defaultdict(lambda: {"passed": 0, "failed": 0, "missing": 0, "total": 0})
    individual_results = []

    # Filter out unsupported test types
    scoreable_tests = []
    skipped_count = 0
    for test in all_tests:
        if test.get("type", "") in skip_types:
            skipped_count += 1
        else:
            scoreable_tests.append(test)

    pass_count = 0
    fail_count = 0
    work_items = [(test, markdowns_dir, use_olmocr, mock_katex) for test in scoreable_tests]

    if num_workers > 1:
        logger.info(f"Scoring with {num_workers} parallel workers")
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_score_single_test, item): i for i, item in enumerate(work_items)}
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Scoring tests", unit="test")
            for future in pbar:
                test_dict, result = future.result()
                _accumulate_score(test_dict, result, results_by_category, results_by_jsonl, individual_results)
                if result["result"] == "pass":
                    pass_count += 1
                elif result["result"] != "skip":
                    fail_count += 1
                pbar.set_postfix(passed=pass_count, failed=fail_count)
    else:
        for item in (pbar := tqdm(work_items, desc="Scoring tests", unit="test")):
            test_dict, result = _score_single_test(item)
            _accumulate_score(test_dict, result, results_by_category, results_by_jsonl, individual_results)
            if result["result"] == "pass":
                pass_count += 1
            elif result["result"] != "skip":
                fail_count += 1
            pbar.set_postfix(passed=pass_count, failed=fail_count)

    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} tests (unsupported types: {skip_types})")

    return dict(results_by_category), dict(results_by_jsonl), individual_results


def _accumulate_score(test, result, results_by_category, results_by_jsonl, individual_results):
    """Accumulate a single test result into category/jsonl counters."""
    category = test["_category"]
    jsonl = test["_jsonl"]
    results_by_category[category]["total"] += 1
    results_by_jsonl[jsonl]["total"] += 1
    if result["result"] == "pass":
        results_by_category[category]["passed"] += 1
        results_by_jsonl[jsonl]["passed"] += 1
    elif result["result"] == "skip":
        results_by_category[category]["missing"] += 1
        results_by_jsonl[jsonl]["missing"] += 1
    else:
        results_by_category[category]["failed"] += 1
        results_by_jsonl[jsonl]["failed"] += 1
    individual_results.append(result)


def _fallback_score(test: dict, md_content: str) -> dict:
    """Simple fallback scoring when olmocr is not installed.

    Only handles text presence/absence. Returns False for table/math/order tests.

    Args:
        test: Test dict from JSONL
        md_content: Generated markdown content

    Returns:
        Dict with test_id, status, result, explanation
    """
    test_type = test.get("type", "")
    test_id = test.get("id", "unknown")

    if test_type == "present":
        text = test.get("text", "")
        if not test.get("case_sensitive", True):
            passed = text.lower() in md_content.lower()
        else:
            passed = text in md_content
        return {"test_id": test_id, "status": "scored", "result": "pass" if passed else "fail",
                "explanation": "fallback: simple text match"}

    elif test_type == "absent":
        text = test.get("text", "")
        if not test.get("case_sensitive", True):
            passed = text.lower() not in md_content.lower()
        else:
            passed = text not in md_content
        return {"test_id": test_id, "status": "scored", "result": "pass" if passed else "fail",
                "explanation": "fallback: simple text absence"}

    else:
        return {"test_id": test_id, "status": "scored", "result": "fail",
                "explanation": f"fallback: {test_type} not supported"}


def compute_overall_score(results_by_jsonl: dict) -> tuple:
    """Compute overall scores as average of per-JSONL pass rates.

    Returns two scores:
    - score_of_total: missing pages count as failed (conservative, comparable)
    - score_of_scored: only scored pages in denominator (reflects actual quality)

    This matches olmOCR-Bench's published scoring: each JSONL category gets
    equal weight regardless of test count.

    Args:
        results_by_jsonl: Dict mapping jsonl filename to {passed, failed, missing, total}

    Returns:
        Tuple of (score_of_total, score_of_scored) as percentages (0-100)
    """
    total_scores = []
    scored_scores = []
    for jsonl, counts in results_by_jsonl.items():
        if counts["total"] > 0:
            total_scores.append(counts["passed"] / counts["total"])
        scored = counts["passed"] + counts["failed"]
        if scored > 0:
            scored_scores.append(counts["passed"] / scored)

    score_of_total = sum(total_scores) / len(total_scores) * 100 if total_scores else 0.0
    score_of_scored = sum(scored_scores) / len(scored_scores) * 100 if scored_scores else 0.0
    return score_of_total, score_of_scored


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_results_table(results_by_category: dict, score_of_total: float, score_of_scored: float):
    """Print a formatted results table to stdout.

    Args:
        results_by_category: Dict mapping category to {passed, failed, missing, total}
        score_of_total: Overall score with missing=failed
        score_of_scored: Overall score of scored tests only
    """
    logger.info("=" * 78)
    logger.info("olmOCR-Bench Results")
    logger.info("=" * 78)

    header = f"{'Category':<22} {'Pass':>6} {'Fail':>6} {'Miss':>6} {'Total':>6}  {'Scored':>8} {'Total':>8}"
    logger.info(header)
    logger.info("-" * 78)

    for category in sorted(results_by_category.keys()):
        c = results_by_category[category]
        scored = c["passed"] + c["failed"]
        pct_scored = c["passed"] / scored * 100 if scored > 0 else 0
        pct_total = c["passed"] / c["total"] * 100 if c["total"] > 0 else 0
        logger.info(f"{category:<22} {c['passed']:>6} {c['failed']:>6} {c['missing']:>6} {c['total']:>6}  {pct_scored:>7.1f}% {pct_total:>7.1f}%")

    total_passed = sum(c["passed"] for c in results_by_category.values())
    total_failed = sum(c["failed"] for c in results_by_category.values())
    total_missing = sum(c["missing"] for c in results_by_category.values())
    total_tests = sum(c["total"] for c in results_by_category.values())
    total_scored = total_passed + total_failed
    agg_scored = total_passed / total_scored * 100 if total_scored > 0 else 0
    agg_total = total_passed / total_tests * 100 if total_tests > 0 else 0
    logger.info("-" * 78)
    logger.info(f"{'Aggregate':<22} {total_passed:>6} {total_failed:>6} {total_missing:>6} {total_tests:>6}  {agg_scored:>7.1f}% {agg_total:>7.1f}%")
    logger.info(f"{'Per-JSONL Average':<22} {'':>6} {'':>6} {'':>6} {'':>6}  {score_of_scored:>7.1f}% {score_of_total:>7.1f}%")
    logger.info("=" * 78)


def save_olmocr_results(output_dir: str, results_by_category: dict, results_by_jsonl: dict,
                        score_of_total: float, score_of_scored: float,
                        metadata: dict, individual_results: list):
    """Save results, metadata, and per-test details to output directory.

    Args:
        output_dir: Output directory path.
        results_by_category: Per-category results.
        results_by_jsonl: Per-JSONL results.
        score_of_total: Overall score with missing=failed.
        score_of_scored: Overall score of scored tests only.
        metadata: Run metadata dict.
        individual_results: Per-test results.
    """
    def _cat_summary(c):
        scored = c["passed"] + c["failed"]
        return {
            "passed": c["passed"], "failed": c["failed"], "missing": c["missing"], "total": c["total"],
            "score_of_scored": round(c["passed"] / scored * 100, 2) if scored > 0 else 0,
            "score_of_total": round(c["passed"] / c["total"] * 100, 2) if c["total"] > 0 else 0,
        }

    results = {
        "score_of_total": round(score_of_total, 2),
        "score_of_scored": round(score_of_scored, 2),
        "by_category": {cat: _cat_summary(c) for cat, c in sorted(results_by_category.items())},
        "by_jsonl": {jsonl: _cat_summary(c) for jsonl, c in sorted(results_by_jsonl.items())},
    }
    save_bench_output(output_dir, results, metadata)

    # Per-test details (large, useful for debugging)
    with open(os.path.join(output_dir, "test_details.jsonl"), "w") as f:
        for r in individual_results:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM on olmOCR-Bench")
    add_common_bench_args(parser, max_new_tokens=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--bench_dir", type=str, default=DEFAULT_BENCH_DIR, help="Path to olmOCR-bench data")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="PDF render DPI")
    args = parser.parse_args()

    output_dir, markdowns_dir = validate_bench_args(args, parser, DEFAULT_OUTPUT_BASE)

    # Load benchmark data
    logger.info(f"Loading olmOCR-Bench from {args.bench_dir}...")
    all_tests, pdf_base_dir = load_bench_data(args.bench_dir)
    unique_pages = collect_unique_pages(all_tests)

    # Apply sharding
    if args.shard:
        if not args.output:
            parser.error("--shard requires --output so all shards write to the same directory")
        unique_pages = apply_shard(unique_pages, args.shard)

    logger.info(f"Unique (pdf, page) pairs to process: {len(unique_pages)}")

    # Register in incomplete jobs for SLURM auto-resubmission
    if not args.score_only:
        register_bench_job(output_dir, "olmocr", args.shard)

    # Generation phase
    if not args.score_only:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        engine = InferenceEngine.load(
            model_id=args.model, checkpoint_path=args.checkpoint,
            model_type="vlm", device=device,
        )

        metadata = build_metadata(args, engine, "olmocr", {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "dpi": args.dpi,
            "prompt": OCR_PROMPT,
        })
        save_metadata(output_dir, metadata)

        cleanup_temp_files(markdowns_dir)

        # Count already-generated pages for accurate progress bar
        already_done = sum(
            1 for pdf_rel, page_num in unique_pages
            if os.path.exists(os.path.join(markdowns_dir, pdf_to_md_name(pdf_rel, page_num)))
        )

        logger.info(f"Starting generation ({already_done} already done, {len(unique_pages) - already_done} remaining)...")
        generated = 0
        errors = 0

        # Build work list (skip already done and missing PDFs)
        pending = []
        for pdf_rel, page_num in unique_pages:
            md_name = pdf_to_md_name(pdf_rel, page_num)
            md_path = os.path.join(markdowns_dir, md_name)
            if os.path.exists(md_path):
                continue
            pdf_abs = os.path.join(pdf_base_dir, pdf_rel)
            if not os.path.exists(pdf_abs):
                logger.warning(f"PDF not found: {pdf_abs}")
                errors += 1
                continue
            pending.append((pdf_rel, page_num, pdf_abs, md_path))

        def _render_page(pdf_abs, page_num):
            """Render PDF page to image (runs in background thread)."""
            return render_pdf_page(pdf_abs, page_num, dpi=args.dpi)

        pbar = tqdm(total=len(unique_pages), desc="Generating", unit="pg", initial=already_done + errors)
        first_generate = True

        # Prefetch: render next page while GPU generates current one
        prefetch_pool = ThreadPoolExecutor(max_workers=1)
        prefetch_future = None

        for idx, (pdf_rel, page_num, pdf_abs, md_path) in enumerate(pending):
            if is_shutdown_requested():
                logger.info(f"Shutting down gracefully. Generated {generated} this session.")
                break

            try:
                # Get current image (from prefetch or direct render)
                if prefetch_future is not None:
                    image = prefetch_future.result()
                else:
                    image = _render_page(pdf_abs, page_num)

                # Start prefetching next page
                if idx + 1 < len(pending):
                    next_pdf_abs, next_page_num = pending[idx + 1][2], pending[idx + 1][1]
                    prefetch_future = prefetch_pool.submit(_render_page, next_pdf_abs, next_page_num)
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
                    logger.info(f"Interrupted on {pdf_rel} page {page_num} - will retry on resume")
                    break
                logger.error(f"Failed on {pdf_rel} page {page_num}: {e}")
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
            # Leave registry entry as incomplete — shell will auto-resubmit
            logger.info(f"Interrupted. Auto-resubmit will resume from: {output_dir}")
            sys.exit(0)
    else:
        logger.info("Score-only mode — skipping generation")
        metadata = {}
        meta_path = os.path.join(output_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)

    # Scoring phase (skip for sharded runs — score with --score_only after all shards complete)
    if args.shard and not args.score_only:
        remove_bench_job()
        logger.info(f"Shard {args.shard} complete — run with --score_only to score after all shards finish")
        return

    # Completeness check: report how many markdowns are available
    all_unique = collect_unique_pages(all_tests)
    available = sum(
        1 for pdf_rel, page_num in all_unique
        if os.path.exists(os.path.join(markdowns_dir, pdf_to_md_name(pdf_rel, page_num)))
    )
    pct_available = available / len(all_unique) * 100
    missing = len(all_unique) - available
    logger.info(f"Scoring {available}/{len(all_unique)} pages ({pct_available:.1f}% available, {missing} missing)")
    num_workers = get_num_workers()
    results_by_category, results_by_jsonl, individual_results = score_tests(all_tests, markdowns_dir, num_workers=num_workers)
    score_of_total, score_of_scored = compute_overall_score(results_by_jsonl)
    print_results_table(results_by_category, score_of_total, score_of_scored)
    save_olmocr_results(output_dir, results_by_category, results_by_jsonl,
                        score_of_total, score_of_scored, metadata, individual_results)

    # Mark complete in registry
    remove_bench_job()


if __name__ == "__main__":
    main()
