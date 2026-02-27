"""Evaluate VLM models on olmOCR-Bench.

Runs any HuggingFace VLM (or LoRA checkpoint) against the olmOCR-Bench benchmark,
producing standardized pass/fail scores comparable to published results.

Usage:
    # New run (auto-derives output dir)
    python scripts/eval_olmocr_bench.py --model allenai/olmOCR-2-7B-1025-FP8

    # New run with explicit output dir
    python scripts/eval_olmocr_bench.py --model Qwen/Qwen3-VL-2B-Instruct --output_dir outputs/bench/my-run

    # Resume (model loaded from metadata.json)
    python scripts/eval_olmocr_bench.py --output_dir outputs/bench/my-run

    # Sharded run
    python scripts/eval_olmocr_bench.py --model X --output_dir D --shard 1/4

    # Score only (after all shards complete)
    python scripts/eval_olmocr_bench.py --score_only --output_dir outputs/bench/my-run
"""

import argparse
import glob
import json
import logging
import os
import signal
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import pypdfium2
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from tqdm import tqdm

# Add project root to path for olmocr imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_BENCH_DIR = "data/olmocr-bench"
DEFAULT_OUTPUT_BASE = "outputs/bench"
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_DPI = 150
OCR_PROMPT = "Extract all text from this document page. Output clean markdown."

# Graceful shutdown flag — set by SIGTERM handler
_shutdown_requested = False


def _handle_sigterm(signum, frame):
    """Handle SIGTERM from SLURM — request graceful shutdown after current page."""
    global _shutdown_requested
    _shutdown_requested = True
    logger.warning("SIGTERM received — finishing current page then shutting down")


signal.signal(signal.SIGTERM, _handle_sigterm)


# ---------------------------------------------------------------------------
# Incomplete jobs registry (shared .incomplete_jobs file, same as distill.py)
# ---------------------------------------------------------------------------
_INCOMPLETE_JOBS = Path(".incomplete_jobs")


def _load_incomplete_jobs() -> dict:
    """Load incomplete jobs registry.

    Returns:
        Dict mapping job_id to job info dict.
    """
    if _INCOMPLETE_JOBS.exists():
        try:
            return json.loads(_INCOMPLETE_JOBS.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_incomplete_jobs(jobs: dict):
    """Save incomplete jobs registry atomically.

    Args:
        jobs: Dict mapping job_id to job info.
    """
    tmp = _INCOMPLETE_JOBS.with_suffix(".tmp")
    tmp.write_text(json.dumps(jobs, indent=2))
    tmp.rename(_INCOMPLETE_JOBS)


def _register_bench_job(output_dir: str, shard: str = None):
    """Register current bench job in incomplete registry.

    Removes previous job entry if PREV_JOB_ID is set (SLURM resubmission handoff).

    Args:
        output_dir: Output directory path.
        shard: Shard spec (e.g. "1/4") or None.
    """
    prev = os.environ.get("PREV_JOB_ID")
    if prev:
        jobs = _load_incomplete_jobs()
        if prev in jobs:
            del jobs[prev]
            _save_incomplete_jobs(jobs)

    key = os.environ.get("SLURM_JOB_ID") or f"local_{os.getpid()}"
    jobs = _load_incomplete_jobs()
    jobs[key] = {"type": "bench", "output_dir": output_dir, "shard": shard or ""}
    _save_incomplete_jobs(jobs)
    logger.info(f"Registered bench job in incomplete registry: {key}")


def _remove_bench_job():
    """Remove current job from incomplete registry (marks it complete)."""
    key = os.environ.get("SLURM_JOB_ID") or f"local_{os.getpid()}"
    jobs = _load_incomplete_jobs()
    if key in jobs:
        del jobs[key]
        _save_incomplete_jobs(jobs)
        logger.info(f"Removed job from incomplete registry: {key}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_processor(model_id: str = None, checkpoint_path: str = None, device: str = "cuda"):
    """Load a VLM model and processor.

    Args:
        model_id: HuggingFace model ID (e.g. 'Qwen/Qwen3-VL-2B-Instruct')
        checkpoint_path: Path to LoRA checkpoint (mutually exclusive with model_id)
        device: Device to load model on

    Returns:
        Tuple of (model, processor, model_name, commit_hash)
    """
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if checkpoint_path:
        # Load LoRA checkpoint — adapter_config.json specifies base model
        checkpoint_path = Path(checkpoint_path)
        adapter_config_path = checkpoint_path / "adapter_config.json"
        if not adapter_config_path.exists():
            raise FileNotFoundError(f"No adapter_config.json in {checkpoint_path}")

        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_id = adapter_config["base_model_name_or_path"]

        logger.info(f"Loading base model: {base_model_id}")
        model = AutoModelForImageTextToText.from_pretrained(
            base_model_id, torch_dtype=dtype, device_map="auto" if device == "cuda" else None
        )
        logger.info(f"Loading LoRA adapter: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, str(checkpoint_path))
        model = model.merge_and_unload()

        processor = AutoProcessor.from_pretrained(base_model_id)
        # Derive name: use dir name for run outputs, parent name for checkpoint-N/
        if checkpoint_path.name.startswith("checkpoint-"):
            model_name = checkpoint_path.parent.name
        else:
            model_name = checkpoint_path.name
        commit_hash = f"local:{checkpoint_path}"
    else:
        logger.info(f"Loading model: {model_id}")
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto" if device == "cuda" else None
        )
        processor = AutoProcessor.from_pretrained(model_id)
        model_name = model_id.split("/")[-1]
        commit_hash = getattr(model.config, "_commit_hash", "unknown")

    model.eval()
    logger.info(f"Model loaded: {model_name} ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")
    return model, processor, model_name, commit_hash


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
# Generation
# ---------------------------------------------------------------------------
def generate_markdown(model, processor, image: Image.Image, device: str, max_new_tokens: int) -> str:
    """Generate markdown from a document image.

    Args:
        model: VLM model
        processor: AutoProcessor
        image: PIL Image of the document page
        device: Device string
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated markdown text
    """
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": OCR_PROMPT},
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    # Strip input tokens, decode only generated portion
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    markdown = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return markdown


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
        Tuple of (load_single_test function or None, set of skipped test types)
    """
    skip_types = set()
    try:
        from olmocr.bench.tests import load_single_test
        logger.info("Using olmocr.bench.tests for scoring (exact comparability)")
        return load_single_test, skip_types
    except ImportError as e:
        if "playwright" in str(e) or "katex" in str(e):
            # Playwright package exists but Chromium binary may be missing,
            # or katex import fails because Playwright isn't working
            if _ensure_playwright_chromium():
                # Retry import after installing Chromium
                try:
                    from olmocr.bench.tests import load_single_test
                    logger.info("Using olmocr.bench.tests for scoring (exact comparability)")
                    return load_single_test, skip_types
                except ImportError:
                    pass

            # Chromium unavailable — fall back to scoring without math
            logger.warning("Playwright Chromium unavailable — math tests will be skipped")
            try:
                import unittest.mock
                katex_mock = unittest.mock.MagicMock()
                sys.modules["olmocr.bench.katex"] = katex_mock
                sys.modules["olmocr.bench.katex.render"] = katex_mock
                from olmocr.bench.tests import load_single_test
                skip_types.add("math")
                return load_single_test, skip_types
            except Exception:
                pass
        logger.warning("olmocr not installed — using fallback scoring (text presence/absence only)")
        return None, skip_types


def score_tests(all_tests, markdowns_dir: str):
    """Score all tests against generated markdowns using olmocr test classes.

    Tries to import olmocr.bench.tests for exact comparability.
    Falls back to simple text presence/absence checks if olmocr is not installed.
    If playwright is missing, math tests are skipped gracefully.

    Args:
        all_tests: List of test dicts from JSONL
        markdowns_dir: Directory containing generated .md files

    Returns:
        Tuple of (results_by_category, results_by_jsonl, individual_results)
    """
    load_single_test, skip_types = _check_olmocr_scoring()
    use_olmocr = load_single_test is not None

    results_by_category = defaultdict(lambda: {"passed": 0, "failed": 0, "missing": 0, "total": 0})
    results_by_jsonl = defaultdict(lambda: {"passed": 0, "failed": 0, "missing": 0, "total": 0})
    individual_results = []

    skipped_count = 0
    for test in all_tests:
        test_type = test.get("type", "")

        # Skip unsupported test types (e.g. math without playwright)
        if test_type in skip_types:
            skipped_count += 1
            continue

        md_name = pdf_to_md_name(test["pdf"], test["page"])
        md_path = os.path.join(markdowns_dir, md_name)

        if not os.path.exists(md_path):
            result = {"test_id": test.get("id", "unknown"), "status": "missing", "result": "skip"}
        else:
            with open(md_path) as f:
                md_content = f.read()

            if use_olmocr:
                try:
                    # Strip internal fields before passing to olmocr (dataclass rejects extras)
                    test_data = {k: v for k, v in test.items() if not k.startswith("_")}
                    test_obj = load_single_test(test_data)
                    passed, explanation = test_obj.run(md_content)
                    result = {"test_id": test.get("id", "unknown"), "status": "scored",
                              "result": "pass" if passed else "fail", "explanation": explanation}
                except Exception as e:
                    result = {"test_id": test.get("id", "unknown"), "status": "scored",
                              "result": "fail", "explanation": f"error: {e}"}
            else:
                result = _fallback_score(test, md_content)

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

    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} tests (unsupported types: {skip_types})")

    return dict(results_by_category), dict(results_by_jsonl), individual_results


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


def save_results(output_dir: str, results_by_category: dict, results_by_jsonl: dict,
                 score_of_total: float, score_of_scored: float,
                 metadata: dict, individual_results: list):
    """Save results and metadata to output directory.

    Args:
        output_dir: Output directory path
        results_by_category: Per-category results
        results_by_jsonl: Per-JSONL results
        score_of_total: Overall score with missing=failed
        score_of_scored: Overall score of scored tests only
        metadata: Run metadata dict
        individual_results: Per-test results
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

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save individual test results (large file, but useful for debugging)
    with open(os.path.join(output_dir, "test_details.jsonl"), "w") as f:
        for r in individual_results:
            f.write(json.dumps(r) + "\n")

    logger.info(f"Results saved to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _model_name_from_args(model_id: str = None, checkpoint_path: str = None) -> str:
    """Extract a short model name from args.

    Args:
        model_id: HuggingFace model ID
        checkpoint_path: LoRA checkpoint path

    Returns:
        Short model name string.
    """
    if checkpoint_path:
        cp = Path(checkpoint_path)
        if cp.name.startswith("checkpoint-"):
            return cp.parent.name
        return cp.name
    return model_id.split("/")[-1]


def derive_output_dir(model_id: str = None, checkpoint_path: str = None) -> str:
    """Derive default output directory from model/checkpoint.

    Args:
        model_id: HuggingFace model ID
        checkpoint_path: LoRA checkpoint path

    Returns:
        Output directory path
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%MZ")
    name = _model_name_from_args(model_id, checkpoint_path)
    return os.path.join(DEFAULT_OUTPUT_BASE, name, timestamp)


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM on olmOCR-Bench")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--model", type=str, help="HuggingFace model ID")
    group.add_argument("--checkpoint", type=str, help="Path to LoRA checkpoint")

    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory. If it exists with metadata.json, resumes automatically.")
    parser.add_argument("--shard", type=str, default=None, help="Shard spec N/M (e.g. 1/4 = first of 4 shards)")
    parser.add_argument("--bench_dir", type=str, default=DEFAULT_BENCH_DIR, help="Path to olmOCR-bench data")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="PDF render DPI")
    parser.add_argument("--score_only", action="store_true", help="Skip generation, only score existing markdowns")
    args = parser.parse_args()

    # Validate required args based on mode
    if args.score_only:
        if not args.output_dir:
            parser.error("--score_only requires --output_dir")

    # Output directory (auto-detect resume from existing metadata.json)
    if args.output_dir:
        output_dir = args.output_dir
        meta_path = os.path.join(output_dir, "metadata.json")
        if os.path.isdir(output_dir) and os.path.exists(meta_path):
            # Resume: load model from metadata if not provided via args
            with open(meta_path) as f:
                prev_meta = json.load(f)
            if not args.model and not args.checkpoint:
                stored_id = prev_meta.get("model_id", "")
                if not stored_id:
                    parser.error("metadata.json has no model_id — specify --model or --checkpoint")
                if stored_id.startswith("local:"):
                    args.checkpoint = stored_id.replace("local:", "")
                else:
                    args.model = stored_id
            else:
                # Both provided — validate match
                curr_model = _model_name_from_args(args.model, args.checkpoint)
                prev_model = prev_meta.get("model_name", "")
                if prev_model and curr_model != prev_model:
                    parser.error(
                        f"Model mismatch: '{curr_model}' vs stored '{prev_model}'"
                    )
            logger.info(f"Resuming into existing dir: {output_dir}")
        elif not args.score_only and not args.model and not args.checkpoint:
            # Explicit dir but no metadata — new run, model required
            parser.error("--model or --checkpoint required for new runs")
    else:
        # No --output_dir: score_only already caught above, so this is a new generation run
        if not args.model and not args.checkpoint:
            parser.error("--model or --checkpoint required for new runs")
        output_dir = derive_output_dir(args.model, args.checkpoint)
    markdowns_dir = os.path.join(output_dir, "markdowns")
    os.makedirs(markdowns_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load benchmark data
    logger.info(f"Loading olmOCR-Bench from {args.bench_dir}...")
    all_tests, pdf_base_dir = load_bench_data(args.bench_dir)
    unique_pages = collect_unique_pages(all_tests)

    # Apply sharding
    if args.shard:
        if not args.output_dir:
            parser.error("--shard requires --output_dir so all shards write to the same directory")
        shard_idx, num_shards = [int(x) for x in args.shard.split("/")]
        if shard_idx < 1 or shard_idx > num_shards:
            parser.error(f"Invalid shard spec: {args.shard} (must be N/M where 1 <= N <= M)")
        shard_size = len(unique_pages) // num_shards
        start = (shard_idx - 1) * shard_size
        end = len(unique_pages) if shard_idx == num_shards else shard_idx * shard_size
        unique_pages = unique_pages[start:end]
        logger.info(f"Shard {shard_idx}/{num_shards}: pages {start}-{end} ({len(unique_pages)} pages)")

    logger.info(f"Unique (pdf, page) pairs to process: {len(unique_pages)}")

    # Register in incomplete jobs for SLURM auto-resubmission
    if not args.score_only:
        _register_bench_job(output_dir, args.shard)

    # Generation phase
    if not args.score_only:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, processor, model_name, commit_hash = load_model_and_processor(
            model_id=args.model, checkpoint_path=args.checkpoint, device=device
        )

        # Load existing metadata on resume, or create fresh
        meta_path = os.path.join(output_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)
        else:
            metadata = {
                "model_id": args.model or str(args.checkpoint),
                "model_name": model_name,
                "commit_hash": commit_hash,
                "generation_params": {
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": False,
                    "dpi": args.dpi,
                    "prompt": OCR_PROMPT,
                },
                "slurm_jobs": [],
            }

        # Track SLURM job ID (ordered list, first to latest)
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id:
            jobs = metadata.setdefault("slurm_jobs", [])
            jobs.append({
                "job_id": slurm_job_id,
                "node": os.environ.get("SLURM_NODELIST", "unknown"),
                "started": datetime.now(timezone.utc).isoformat(),
            })

        # Clean up stale .tmp files from interrupted writes
        for tmp_file in glob.glob(os.path.join(markdowns_dir, "*.tmp")):
            os.remove(tmp_file)
            logger.info(f"Cleaned up stale temp file: {os.path.basename(tmp_file)}")

        # Count already-generated pages for accurate progress bar
        already_done = sum(
            1 for pdf_rel, page_num in unique_pages
            if os.path.exists(os.path.join(markdowns_dir, pdf_to_md_name(pdf_rel, page_num)))
        )

        logger.info(f"Starting generation ({already_done} already done, {len(unique_pages) - already_done} remaining)...")
        generated = 0
        errors = 0

        pbar = tqdm(total=len(unique_pages), desc="Generating", unit="pg", initial=already_done)
        for pdf_rel, page_num in unique_pages:
            md_name = pdf_to_md_name(pdf_rel, page_num)
            md_path = os.path.join(markdowns_dir, md_name)

            # Graceful shutdown on SIGTERM (SLURM time limit)
            if _shutdown_requested:
                logger.info(f"Shutting down gracefully. Generated {generated} this session.")
                break

            # Skip if already generated (resume support)
            if os.path.exists(md_path):
                continue

            # Resolve PDF path: test["pdf"] is relative to bench_data/pdfs/
            pdf_abs = os.path.join(pdf_base_dir, pdf_rel)
            if not os.path.exists(pdf_abs):
                logger.warning(f"PDF not found: {pdf_abs}")
                errors += 1
                pbar.set_postfix(gen=generated, err=errors)
                pbar.update(1)
                continue

            try:
                image = render_pdf_page(pdf_abs, page_num, dpi=args.dpi)
                markdown = generate_markdown(model, processor, image, device, args.max_new_tokens)

                # Atomic write: temp file + rename so partial writes don't look complete on resume
                tmp_path = md_path + ".tmp"
                with open(tmp_path, "w") as f:
                    f.write(markdown)
                os.rename(tmp_path, md_path)
                generated += 1

            except Exception as e:
                if _shutdown_requested:
                    logger.info(f"Interrupted on {pdf_rel} page {page_num} — will retry on resume")
                    break
                logger.error(f"Failed on {pdf_rel} page {page_num}: {e}")
                errors += 1

            pbar.set_postfix(gen=generated, err=errors)
            pbar.update(1)

        pbar.close()
        logger.info(f"Generation complete: {generated} generated, {already_done} skipped, {errors} errors")

        # Clean up any .tmp files left from this session (e.g. interrupted write)
        for tmp_file in glob.glob(os.path.join(markdowns_dir, "*.tmp")):
            os.remove(tmp_file)

        # Save metadata (always, even on shutdown)
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        if _shutdown_requested:
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
        _remove_bench_job()
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
    results_by_category, results_by_jsonl, individual_results = score_tests(all_tests, markdowns_dir)
    score_of_total, score_of_scored = compute_overall_score(results_by_jsonl)
    print_results_table(results_by_category, score_of_total, score_of_scored)
    save_results(output_dir, results_by_category, results_by_jsonl,
                 score_of_total, score_of_scored, metadata, individual_results)

    # Mark complete in registry
    _remove_bench_job()


if __name__ == "__main__":
    main()
