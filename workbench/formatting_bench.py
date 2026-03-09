"""Formatting-specific benchmark for OCR models.

Standalone eval that measures inline formatting preservation (bold, italic,
strikethrough) on document pages. Runs independently from OmniDocBench and
olmOCR-Bench to avoid contaminating those standardized metrics.

Can score against:
1. OmniDocBench predictions (from existing omnidoc_bench runs)
2. Live model inference on any image set with formatting-rich GT

Usage:
    # Score existing OmniDocBench predictions
    python workbench/formatting_bench.py --pred_dir outputs/bench/omnidoc/my-run/markdowns --gt_dir data/ocr/omnidocbench/markdown

    # Run live inference + score
    python workbench/formatting_bench.py --model Qwen/Qwen3-VL-2B-Instruct --gt_dir data/ocr/omnidocbench/markdown --images_dir data/ocr/omnidocbench/images

    # Score with adapter
    python workbench/formatting_bench.py --model Qwen/Qwen3-VL-2B-Instruct --adapter outputs/ablations_00/checkpoint-XXX --gt_dir data/ocr/omnidocbench/markdown --images_dir data/ocr/omnidocbench/images
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Formatting extraction and scoring
# ---------------------------------------------------------------------------

def extract_formatting_spans(text: str) -> dict[str, set[str]]:
    """Extract inline formatting spans from markdown text.

    Args:
        text: Markdown text.

    Returns:
        Dict with keys 'bold', 'italic', 'strikethrough', each a set of spans.
    """
    bold = set(re.findall(r"\*\*(.+?)\*\*", text))
    italic = set(re.findall(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", text))
    strikethrough = set(re.findall(r"~~(.+?)~~", text))
    return {"bold": bold, "italic": italic, "strikethrough": strikethrough}


def count_formatting(text: str) -> dict[str, int]:
    """Count formatting spans by type.

    Args:
        text: Markdown text.

    Returns:
        Dict with counts per type plus total.
    """
    spans = extract_formatting_spans(text)
    counts = {k: len(v) for k, v in spans.items()}
    counts["total"] = sum(counts.values())
    return counts


def fuzzy_set_f1(pred_spans: set[str], gt_spans: set[str], threshold: float = 0.5) -> float:
    """Compute F1 between two sets of text spans using fuzzy matching.

    Args:
        pred_spans: Predicted spans.
        gt_spans: Ground truth spans.
        threshold: Minimum Levenshtein similarity to count as match.

    Returns:
        F1 score in [0, 1].
    """
    from rapidfuzz.distance import Levenshtein

    if not gt_spans:
        return 1.0 if not pred_spans else 0.0
    if not pred_spans:
        return 0.0

    matched = 0.0
    used = set()
    pred_list = list(pred_spans)
    for gt_s in gt_spans:
        best_sim = 0.0
        best_idx = -1
        for j, pred_s in enumerate(pred_list):
            if j in used:
                continue
            dist = Levenshtein.distance(gt_s.lower(), pred_s.lower())
            max_len = max(len(gt_s), len(pred_s))
            sim = 1.0 - (dist / max_len) if max_len > 0 else 1.0
            if sim > best_sim:
                best_sim = sim
                best_idx = j
        if best_idx >= 0 and best_sim >= threshold:
            matched += best_sim
            used.add(best_idx)

    precision = matched / len(pred_list)
    recall = matched / len(gt_spans)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_formatting(pred_md: str, gt_md: str) -> dict:
    """Score inline formatting preservation for a single page.

    Args:
        pred_md: Predicted markdown.
        gt_md: Ground truth markdown.

    Returns:
        Dict with per-type F1 scores, overall F1, and span counts.
        Returns None for types not present in GT.
    """
    gt_spans = extract_formatting_spans(gt_md)
    pred_spans = extract_formatting_spans(pred_md)

    all_gt = gt_spans["bold"] | gt_spans["italic"] | gt_spans["strikethrough"]
    all_pred = pred_spans["bold"] | pred_spans["italic"] | pred_spans["strikethrough"]

    result = {
        "gt_counts": {k: len(v) for k, v in gt_spans.items()},
        "pred_counts": {k: len(v) for k, v in pred_spans.items()},
    }

    # Per-type F1
    for fmt_type in ["bold", "italic", "strikethrough"]:
        if gt_spans[fmt_type]:
            result[fmt_type] = fuzzy_set_f1(pred_spans[fmt_type], gt_spans[fmt_type])
        else:
            result[fmt_type] = None

    # Overall F1 (all types combined)
    if all_gt:
        result["overall"] = fuzzy_set_f1(all_pred, all_gt)
    else:
        result["overall"] = None

    return result


# ---------------------------------------------------------------------------
# Scoring from existing predictions
# ---------------------------------------------------------------------------

def score_from_files(pred_dir: str, gt_dir: str) -> list[dict]:
    """Score pre-existing prediction files against GT.

    Args:
        pred_dir: Directory with predicted .md files.
        gt_dir: Directory with ground truth .md files.

    Returns:
        List of per-page result dicts.
    """
    from tqdm import tqdm

    gt_files = {p.stem: p for p in Path(gt_dir).glob("*.md")}
    pred_files = {p.stem: p for p in Path(pred_dir).glob("*.md")}

    common = sorted(set(gt_files) & set(pred_files))
    if not common:
        print(f"No matching files between {pred_dir} and {gt_dir}")
        return []

    print(f"Scoring {len(common)} pages ({len(gt_files)} GT, {len(pred_files)} pred)")

    results = []
    for stem in tqdm(common, desc="Scoring formatting"):
        gt_md = gt_files[stem].read_text(errors="replace")
        pred_md = pred_files[stem].read_text(errors="replace")
        scores = score_formatting(pred_md, gt_md)
        scores["stem"] = stem
        results.append(scores)

    return results


# ---------------------------------------------------------------------------
# Live inference + scoring
# ---------------------------------------------------------------------------

def score_with_inference(model: str, adapter: str, images_dir: str, gt_dir: str,
                         max_new_tokens: int, max_pages: int = None) -> list[dict]:
    """Run model inference on images and score formatting.

    Args:
        model: Model name or path.
        adapter: Optional LoRA adapter path.
        images_dir: Directory with document images.
        gt_dir: Directory with GT markdown files.
        max_new_tokens: Max tokens to generate.
        max_pages: Optional limit on pages to score.

    Returns:
        List of per-page result dicts.
    """
    from PIL import Image
    from tqdm import tqdm

    from src.data.ocr_utils import OCR_SYSTEM_PROMPT
    from src.inference import InferenceEngine

    gt_files = {p.stem: p for p in Path(gt_dir).glob("*.md")}

    # Match images to GT by stem
    exts = {".jpg", ".jpeg", ".png"}
    image_files = {}
    for p in Path(images_dir).iterdir():
        if p.suffix.lower() in exts and p.stem in gt_files:
            image_files[p.stem] = p

    common = sorted(set(gt_files) & set(image_files))
    if max_pages:
        common = common[:max_pages]

    if not common:
        print(f"No matching images+GT found")
        return []

    print(f"Running inference on {len(common)} pages")
    engine = InferenceEngine(model, adapter_path=adapter)

    results = []
    for stem in tqdm(common, desc="Inference + scoring"):
        image = Image.open(image_files[stem]).convert("RGB")
        outputs = engine.generate([(image, OCR_SYSTEM_PROMPT)], max_new_tokens=max_new_tokens)
        pred_md = outputs[0] if outputs else ""

        gt_md = gt_files[stem].read_text(errors="replace")
        scores = score_formatting(pred_md, gt_md)
        scores["stem"] = stem
        results.append(scores)

    return results


# ---------------------------------------------------------------------------
# Aggregation and display
# ---------------------------------------------------------------------------

def compute_aggregate(results: list[dict]) -> dict:
    """Compute aggregate formatting scores.

    Args:
        results: List of per-page score dicts.

    Returns:
        Dict with per-type and overall averages.
    """
    metrics = ["bold", "italic", "strikethrough", "overall"]
    sums = {m: 0.0 for m in metrics}
    counts = {m: 0 for m in metrics}

    total_gt = {"bold": 0, "italic": 0, "strikethrough": 0}
    total_pred = {"bold": 0, "italic": 0, "strikethrough": 0}
    pages_with_formatting = 0

    for r in results:
        for m in metrics:
            val = r.get(m)
            if val is not None:
                sums[m] += val
                counts[m] += 1
        for k in total_gt:
            total_gt[k] += r.get("gt_counts", {}).get(k, 0)
            total_pred[k] += r.get("pred_counts", {}).get(k, 0)
        if r.get("overall") is not None:
            pages_with_formatting += 1

    averages = {}
    for m in metrics:
        averages[m] = round(sums[m] / counts[m], 4) if counts[m] > 0 else None

    return {
        "scores": averages,
        "counts": counts,
        "total_pages": len(results),
        "pages_with_formatting": pages_with_formatting,
        "gt_span_totals": total_gt,
        "pred_span_totals": total_pred,
    }


def print_results(agg: dict):
    """Print formatted results.

    Args:
        agg: Aggregate results dict.
    """
    scores = agg["scores"]
    counts = agg["counts"]

    print(f"\n{'=' * 60}")
    print(f"Formatting Benchmark Results")
    print(f"{'=' * 60}")
    print(f"Total pages:              {agg['total_pages']}")
    print(f"Pages with formatting:    {agg['pages_with_formatting']}")
    print(f"{'-' * 60}")

    metric_names = {
        "bold": "Bold (F1)",
        "italic": "Italic (F1)",
        "strikethrough": "Strikethrough (F1)",
        "overall": "Overall (F1)",
    }
    for key, name in metric_names.items():
        val = scores.get(key)
        n = counts.get(key, 0)
        if val is not None:
            print(f"  {name:<25} {val*100:>6.2f}%  (n={n})")
        else:
            print(f"  {name:<25}    N/A")

    print(f"{'-' * 60}")
    gt = agg["gt_span_totals"]
    pred = agg["pred_span_totals"]
    print(f"GT spans:   bold={gt['bold']}, italic={gt['italic']}, strike={gt['strikethrough']}")
    print(f"Pred spans: bold={pred['bold']}, italic={pred['italic']}, strike={pred['strikethrough']}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Formatting-specific benchmark")
    parser.add_argument("--pred_dir", type=str, default=None,
                        help="Directory with predicted .md files (score-only mode)")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory with ground truth .md files")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name/path for live inference")
    parser.add_argument("--adapter", type=str, default=None,
                        help="LoRA adapter path")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Directory with document images (for live inference)")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_pages", type=int, default=None,
                        help="Limit number of pages to score")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results JSON to this path")
    args = parser.parse_args()

    if args.pred_dir:
        # Score-only mode: use existing predictions
        results = score_from_files(args.pred_dir, args.gt_dir)
    elif args.model:
        # Live inference mode
        if not args.images_dir:
            print("--images_dir required for live inference mode")
            sys.exit(1)
        results = score_with_inference(
            args.model, args.adapter, args.images_dir, args.gt_dir,
            args.max_new_tokens, args.max_pages,
        )
    else:
        print("Provide either --pred_dir (score-only) or --model (live inference)")
        sys.exit(1)

    if not results:
        print("No results to aggregate")
        sys.exit(1)

    agg = compute_aggregate(results)
    print_results(agg)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        save_data = {
            "aggregates": agg,
            "per_page": results,
        }
        with open(args.output, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
