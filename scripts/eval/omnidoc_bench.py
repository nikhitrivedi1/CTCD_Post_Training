"""Evaluate VLM models on OmniDocBench formatting quality.

Runs any HuggingFace VLM (or LoRA checkpoint) against OmniDocBench's 1,355 pages,
scoring predictions across four formatting dimensions: text, tables, formulas,
and reading order. Produces per-page and aggregate scores.

Usage:
    # New run
    python scripts/eval/omnidoc_bench.py --model Qwen/Qwen3-VL-2B-Instruct

    # With explicit output dir
    python scripts/eval/omnidoc_bench.py --model Qwen/Qwen3-VL-2B-Instruct --output_dir outputs/bench/omnidoc/my-run

    # Resume (model loaded from metadata.json)
    python scripts/eval/omnidoc_bench.py --output_dir outputs/bench/omnidoc/my-run

    # Sharded run
    python scripts/eval/omnidoc_bench.py --model X --output_dir D --shard 1/4

    # Score only (after all shards complete)
    python scripts/eval/omnidoc_bench.py --score_only --output_dir outputs/bench/omnidoc/my-run

"""

import argparse
import glob
import json
import logging
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import InferenceEngine
from src.utils.jobs import get_job_key, locked_incomplete_jobs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
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

# Graceful shutdown flag
_shutdown_requested = False


def _handle_sigterm(signum, frame):
    """Handle SIGTERM for graceful SLURM shutdown."""
    global _shutdown_requested
    _shutdown_requested = True
    logger.warning("SIGTERM received - finishing current page then shutting down")


signal.signal(signal.SIGTERM, _handle_sigterm)


# ---------------------------------------------------------------------------
# Incomplete jobs registry
# ---------------------------------------------------------------------------
def _register_bench_job(output_dir: str, shard: str = None):
    """Register current bench job in incomplete registry.

    Args:
        output_dir: Output directory path.
        shard: Shard spec (e.g. "1/4") or None.
    """
    key = get_job_key()
    prev = os.environ.get("PREV_JOB_ID")

    with locked_incomplete_jobs() as jobs:
        if prev and prev in jobs:
            del jobs[prev]
        jobs[key] = {"type": "omnidocbench", "output_dir": output_dir, "shard": shard or ""}
    logger.info(f"Registered bench job in incomplete registry: {key}")


def _remove_bench_job():
    """Remove current job from incomplete registry (marks it complete)."""
    key = get_job_key()
    with locked_incomplete_jobs() as jobs:
        if key in jobs:
            del jobs[key]
            logger.info(f"Removed job from incomplete registry: {key}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_pages(data_dir: str):
    """Load OmniDocBench image paths and GT markdown paths.

    Expects:
        data_dir/images/*.jpg   - document page images
        data_dir/markdown/*.md  - ground truth markdown (from convert_omnidocbench.py)

    Args:
        data_dir: Path to OmniDocBench dataset root.

    Returns:
        Sorted list of (image_stem, image_path, gt_markdown_path) tuples.
    """
    images_dir = os.path.join(data_dir, "images")
    markdown_dir = os.path.join(data_dir, "markdown")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"images/ not found in {data_dir}. Download with:\n"
            f"  python -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('opendatalab/OmniDocBench', local_dir='{data_dir}', repo_type='dataset')\""
        )
    if not os.path.isdir(markdown_dir):
        raise FileNotFoundError(
            f"markdown/ not found in {data_dir}. Run:\n"
            f"  python scripts/data/convert_omnidocbench.py --data_dir {data_dir}"
        )

    pages = []
    for img_path in sorted(glob.glob(os.path.join(images_dir, "*"))):
        ext = os.path.splitext(img_path)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            continue
        stem = Path(img_path).stem
        gt_path = os.path.join(markdown_dir, stem + ".md")
        if os.path.exists(gt_path):
            pages.append((stem, img_path, gt_path))

    logger.info(f"Loaded {len(pages)} pages with GT markdown")
    return pages


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _normalized_edit_distance(pred: str, gt: str) -> float:
    """Compute 1 - normalized Levenshtein distance (similarity in [0, 1]).

    Args:
        pred: Predicted string.
        gt: Ground truth string.

    Returns:
        Similarity score in [0, 1].
    """
    from rapidfuzz.distance import Levenshtein

    if not gt:
        return 1.0 if not pred else 0.0
    dist = Levenshtein.distance(pred, gt)
    max_len = max(len(pred), len(gt))
    return 1.0 - (dist / max_len) if max_len > 0 else 1.0


def _extract_text_blocks(text: str) -> list[str]:
    """Split markdown into text blocks (paragraphs separated by blank lines).

    Args:
        text: Markdown text.

    Returns:
        List of non-trivial text blocks.
    """
    blocks = []
    for block in text.split("\n\n"):
        block = block.strip()
        if block and len(block) > 10:
            blocks.append(block)
    return blocks


def _extract_markdown_tables(text: str) -> list[list[list[str]]]:
    """Extract tables from markdown as list of row-cell grids.

    Args:
        text: Markdown text.

    Returns:
        List of tables, each a list of rows, each a list of cell strings.
    """
    import re

    tables = []
    current_table = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("|") and line.endswith("|"):
            inner = line.strip("|")
            parts = [p.strip() for p in inner.split("|")]
            if all(re.match(r"^:?-+:?$", p) for p in parts if p):
                continue
            cells = [c.strip() for c in line.split("|")[1:-1]]
            current_table.append(cells)
        else:
            if current_table:
                tables.append(current_table)
                current_table = []
    if current_table:
        tables.append(current_table)
    return tables


def _extract_latex(text: str) -> list[str]:
    """Extract LaTeX formulas from markdown text.

    Args:
        text: Markdown text with $$...$$ or $...$ formulas.

    Returns:
        List of cleaned LaTeX strings.
    """
    import re

    formulas = []
    for match in re.finditer(r"\$\$(.*?)\$\$", text, re.DOTALL):
        formulas.append(_clean_latex(match.group(1)))
    for match in re.finditer(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", text):
        formulas.append(_clean_latex(match.group(1)))
    return formulas


def _clean_latex(latex: str) -> str:
    """Normalize LaTeX for comparison."""
    import re

    latex = latex.strip()
    latex = re.sub(r"\s+", " ", latex)
    latex = latex.replace(r"\left", "").replace(r"\right", "")
    latex = latex.replace(r"\displaystyle", "").replace(r"\textstyle", "")
    return latex.strip()


def score_text(pred: str, gt: str) -> float:
    """Score text extraction quality via normalized edit distance on text blocks.

    Compares concatenated text blocks (excluding tables and formulas).

    Args:
        pred: Predicted markdown.
        gt: Ground truth markdown.

    Returns:
        Similarity score in [0, 1].
    """
    pred_blocks = _extract_text_blocks(pred)
    gt_blocks = _extract_text_blocks(gt)

    if not gt_blocks:
        return 1.0 if not pred_blocks else 0.0

    pred_text = "\n".join(pred_blocks)
    gt_text = "\n".join(gt_blocks)
    return _normalized_edit_distance(pred_text, gt_text)


def score_tables(pred: str, gt: str) -> float | None:
    """Score table formatting quality via cell-level F1.

    Args:
        pred: Predicted markdown.
        gt: Ground truth markdown.

    Returns:
        Score in [0, 1], or None if GT has no tables.
    """
    from rapidfuzz.distance import Levenshtein

    gt_tables = _extract_markdown_tables(gt)
    if not gt_tables:
        return None

    pred_tables = _extract_markdown_tables(pred)
    if not pred_tables:
        return 0.0

    scores = []
    for i, gt_table in enumerate(gt_tables):
        if i < len(pred_tables):
            gt_cells = [cell for row in gt_table for cell in row if cell.strip()]
            pred_cells = [cell for row in pred_tables[i] for cell in row if cell.strip()]

            if not gt_cells:
                scores.append(1.0 if not pred_cells else 0.0)
                continue

            matched = 0.0
            used = set()
            for gt_cell in gt_cells:
                best_sim = 0.0
                best_idx = -1
                for j, pred_cell in enumerate(pred_cells):
                    if j in used:
                        continue
                    dist = Levenshtein.distance(gt_cell, pred_cell)
                    max_len = max(len(gt_cell), len(pred_cell))
                    sim = 1.0 - (dist / max_len) if max_len > 0 else 1.0
                    if sim > best_sim:
                        best_sim = sim
                        best_idx = j
                if best_idx >= 0 and best_sim > 0.3:
                    matched += best_sim
                    used.add(best_idx)

            precision = matched / len(pred_cells) if pred_cells else 0.0
            recall = matched / len(gt_cells)
            if precision + recall == 0:
                scores.append(0.0)
            else:
                scores.append(2 * precision * recall / (precision + recall))
        else:
            scores.append(0.0)

    return sum(scores) / len(scores)


def score_formulas(pred: str, gt: str) -> float | None:
    """Score formula quality via normalized edit distance on cleaned LaTeX.

    Args:
        pred: Predicted markdown.
        gt: Ground truth markdown.

    Returns:
        Score in [0, 1], or None if GT has no formulas.
    """
    gt_formulas = _extract_latex(gt)
    if not gt_formulas:
        return None

    pred_formulas = _extract_latex(pred)
    if not pred_formulas:
        return 0.0

    scores = []
    for i, gt_f in enumerate(gt_formulas):
        if i < len(pred_formulas):
            scores.append(_normalized_edit_distance(pred_formulas[i], gt_f))
        else:
            scores.append(0.0)

    return sum(scores) / len(scores)


def score_reading_order(pred: str, gt: str) -> float | None:
    """Score reading order via LCS ratio on matched text blocks.

    Args:
        pred: Predicted markdown.
        gt: Ground truth markdown.

    Returns:
        Score in [0, 1], or None if GT has insufficient blocks.
    """
    from rapidfuzz.distance import Levenshtein

    gt_blocks = _extract_text_blocks(gt)
    if len(gt_blocks) < 2:
        return None

    pred_blocks = _extract_text_blocks(pred)
    if not pred_blocks:
        return 0.0

    # Match pred blocks to GT blocks by content similarity
    gt_prefixes = [b[:50] for b in gt_blocks]
    pred_prefixes = [b[:50] for b in pred_blocks]

    order = []
    used = set()
    for pred_p in pred_prefixes:
        best_sim = 0.0
        best_idx = -1
        for j, gt_p in enumerate(gt_prefixes):
            if j in used:
                continue
            dist = Levenshtein.distance(pred_p, gt_p)
            max_len = max(len(pred_p), len(gt_p))
            sim = 1.0 - (dist / max_len) if max_len > 0 else 1.0
            if sim > best_sim:
                best_sim = sim
                best_idx = j
        if best_idx >= 0 and best_sim > 0.3:
            order.append(best_idx)
            used.add(best_idx)

    if not order:
        return 0.0

    # LCS length against natural order
    gt_order = list(range(len(gt_blocks)))
    m, n = len(order), len(gt_order)
    dp = [0] * (n + 1)
    for i in range(m):
        prev = 0
        for j in range(n):
            temp = dp[j + 1]
            if order[i] == gt_order[j]:
                dp[j + 1] = prev + 1
            else:
                dp[j + 1] = max(dp[j + 1], dp[j])
            prev = temp

    return dp[n] / len(gt_blocks)


def score_page(pred_md: str, gt_md: str) -> dict:
    """Score a single page across all four metrics.

    Args:
        pred_md: Predicted markdown.
        gt_md: Ground truth markdown.

    Returns:
        Dict with keys: text, table, formula, reading_order.
        Values are floats in [0, 1] or None if metric not applicable.
    """
    return {
        "text": score_text(pred_md, gt_md),
        "table": score_tables(pred_md, gt_md),
        "formula": score_formulas(pred_md, gt_md),
        "reading_order": score_reading_order(pred_md, gt_md),
    }


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
    for stem, _, gt_path in pages:
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


def compute_aggregate(results: list[tuple[str, dict]]) -> dict:
    """Compute aggregate scores from per-page results.

    For each metric, averages across pages where the metric is applicable
    (not None). This avoids penalizing pages without tables or formulas.

    Args:
        results: List of (stem, scores_dict) tuples.

    Returns:
        Dict with per-metric averages and overall score.
    """
    metrics = ["text", "table", "formula", "reading_order"]
    sums = {m: 0.0 for m in metrics}
    counts = {m: 0 for m in metrics}

    for _, scores in results:
        for m in metrics:
            val = scores.get(m)
            if val is not None:
                sums[m] += val
                counts[m] += 1

    averages = {}
    for m in metrics:
        if counts[m] > 0:
            averages[m] = round(sums[m] / counts[m], 4)
        else:
            averages[m] = None

    # Overall: average of available metric averages
    valid = [v for v in averages.values() if v is not None]
    averages["overall"] = round(sum(valid) / len(valid), 4) if valid else 0.0
    averages["counts"] = counts

    return averages


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
    logger.info("OmniDocBench Formatting Results")
    logger.info("=" * 60)
    logger.info(f"Pages scored: {num_scored}/{num_total}")
    logger.info("-" * 60)

    metric_names = {
        "text": "Text (Edit Distance)",
        "table": "Tables (Cell F1)",
        "formula": "Formulas (LaTeX Edit Dist)",
        "reading_order": "Reading Order (LCS)",
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


def save_results(output_dir: str, aggregates: dict, page_results: list,
                 metadata: dict, num_scored: int, num_total: int):
    """Save results and metadata to output directory.

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

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Per-page details
    with open(os.path.join(output_dir, "page_scores.jsonl"), "w") as f:
        for stem, scores in sorted(page_results, key=lambda x: x[0]):
            f.write(json.dumps({"page": stem, **scores}) + "\n")

    logger.info(f"Results saved to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _model_name_from_args(model_id: str = None, checkpoint_path: str = None) -> str:
    """Extract a short model name from args.

    Args:
        model_id: HuggingFace model ID.
        checkpoint_path: LoRA checkpoint path.

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
        model_id: HuggingFace model ID.
        checkpoint_path: LoRA checkpoint path.

    Returns:
        Output directory path.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%MZ")
    name = _model_name_from_args(model_id, checkpoint_path)
    return os.path.join(DEFAULT_OUTPUT_BASE, name, timestamp)


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM on OmniDocBench formatting")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--model", type=str, help="HuggingFace model ID")
    group.add_argument("--checkpoint", type=str, help="Path to LoRA checkpoint")

    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory. If it exists with metadata.json, resumes automatically.")
    parser.add_argument("--shard", type=str, default=None, help="Shard spec N/M (e.g. 1/4)")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Path to OmniDocBench data root")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--score_only", action="store_true",
                        help="Skip generation, only score existing markdowns")
    args = parser.parse_args()

    # Validate required args
    if args.score_only and not args.output_dir:
        parser.error("--score_only requires --output_dir")

    # Output directory (auto-detect resume from existing metadata.json)
    if args.output_dir:
        output_dir = args.output_dir
        meta_path = os.path.join(output_dir, "metadata.json")
        if os.path.isdir(output_dir) and os.path.exists(meta_path):
            with open(meta_path) as f:
                prev_meta = json.load(f)
            if not args.model and not args.checkpoint:
                stored_id = prev_meta.get("model_id", "")
                if not stored_id:
                    parser.error("metadata.json has no model_id - specify --model or --checkpoint")
                if stored_id.startswith("local:"):
                    args.checkpoint = stored_id.replace("local:", "")
                else:
                    args.model = stored_id
            else:
                curr_model = _model_name_from_args(args.model, args.checkpoint)
                prev_model = prev_meta.get("model_name", "")
                if prev_model and curr_model != prev_model:
                    parser.error(f"Model mismatch: '{curr_model}' vs stored '{prev_model}'")
            logger.info(f"Resuming into existing dir: {output_dir}")
        elif not args.score_only and not args.model and not args.checkpoint:
            parser.error("--model or --checkpoint required for new runs")
    else:
        if not args.model and not args.checkpoint:
            parser.error("--model or --checkpoint required for new runs")
        output_dir = derive_output_dir(args.model, args.checkpoint)

    markdowns_dir = os.path.join(output_dir, "markdowns")
    os.makedirs(markdowns_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load dataset
    logger.info(f"Loading OmniDocBench from {args.data_dir}...")
    pages = load_pages(args.data_dir)

    # Apply sharding
    if args.shard:
        if not args.output_dir:
            parser.error("--shard requires --output_dir so all shards write to the same directory")
        shard_idx, num_shards = [int(x) for x in args.shard.split("/")]
        if shard_idx < 1 or shard_idx > num_shards:
            parser.error(f"Invalid shard spec: {args.shard}")
        total = len(pages)
        shard_size = total // num_shards
        start = (shard_idx - 1) * shard_size
        end = total if shard_idx == num_shards else shard_idx * shard_size
        pages = pages[start:end]
        logger.info(f"Shard {shard_idx}/{num_shards}: pages {start+1}-{end} of {total}")

    # Register in incomplete jobs
    if not args.score_only:
        _register_bench_job(output_dir, args.shard)

    # Generation phase
    if not args.score_only:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        engine = InferenceEngine.load(
            model_id=args.model, checkpoint_path=args.checkpoint,
            model_type="vlm", device=device,
        )

        meta_path = os.path.join(output_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)
        else:
            metadata = {
                "model_id": args.model or f"local:{args.checkpoint}",
                "model_name": engine.model_name,
                "commit_hash": engine.commit_hash,
                "benchmark": "omnidocbench",
                "generation_params": {
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": False,
                    "prompt": OCR_PROMPT,
                },
                "slurm_jobs": [],
            }

        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id:
            jobs = metadata.setdefault("slurm_jobs", [])
            jobs.append({
                "job_id": slurm_job_id,
                "node": os.environ.get("SLURM_NODELIST", "unknown"),
                "started": datetime.now(timezone.utc).isoformat(),
            })

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Clean stale temp files
        for tmp_file in glob.glob(os.path.join(markdowns_dir, "*.tmp")):
            os.remove(tmp_file)

        # Count already done
        already_done = sum(
            1 for stem, _, _ in pages
            if os.path.exists(os.path.join(markdowns_dir, stem + ".md"))
        )

        logger.info(f"Starting generation ({already_done} done, {len(pages) - already_done} remaining)...")
        generated = 0
        errors = 0

        # Build work list (skip already done)
        pending = []
        for stem, image_path, _ in pages:
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
            if _shutdown_requested:
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
                if _shutdown_requested:
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

        for tmp_file in glob.glob(os.path.join(markdowns_dir, "*.tmp")):
            os.remove(tmp_file)

        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        if _shutdown_requested:
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
        _remove_bench_job()
        logger.info(f"Shard {args.shard} complete - run with --score_only to score after all shards finish")
        return

    # Reload all pages for scoring (not just this shard)
    all_pages = load_pages(args.data_dir)
    if not all_pages:
        logger.error(f"No pages found in {args.data_dir}")
        return

    available = sum(
        1 for stem, _, _ in all_pages
        if os.path.exists(os.path.join(markdowns_dir, stem + ".md"))
    )
    logger.info(f"Scoring {available}/{len(all_pages)} pages ({available/len(all_pages)*100:.1f}% available)")

    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    available_cpus = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 1)
    num_workers = max(1, available_cpus // 2)

    page_results = score_all(all_pages, markdowns_dir, num_workers=num_workers)
    aggregates = compute_aggregate(page_results)

    print_results(aggregates, available, len(all_pages))
    save_results(output_dir, aggregates, page_results, metadata, available, len(all_pages))

    _remove_bench_job()


if __name__ == "__main__":
    main()
