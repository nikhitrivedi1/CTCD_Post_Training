"""Failure mode analysis for formatting bench results.

Analyzes GT vs pred markdown files to identify specific error categories:
hallucinated spans, over-formatting, type swaps, table drops, merge errors,
and missing text.
"""

import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rapidfuzz.distance import Levenshtein
from src.evaluation.formatting_metrics import (
    extract_formatting_spans,
    _strip_all_formatting,
    _strip_inner_markers,
)

CATEGORIES = [
    "00_text", "01_tables", "02_academic", "03_financial", "04_legal",
    "05_technical", "06_books", "07_slides", "08_news", "09_forms",
]

DATA_DIR = Path("data/ocr/formatting")
PRED_DIR = Path("outputs/bench/formatting/fmt_sft-lora_qwen35-9b/markdowns")
THRESHOLD = 0.5


def load_pages():
    """Load all GT and pred markdown pairs.

    Returns:
        List of (stem, gt_md, pred_md) tuples.
    """
    pages = []
    for cat in CATEGORIES:
        gt_dir = DATA_DIR / cat / "eval" / "markdown"
        if not gt_dir.exists():
            continue
        for gt_file in sorted(gt_dir.glob("*.md")):
            stem = gt_file.stem
            pred_file = PRED_DIR / f"{stem}.md"
            if not pred_file.exists():
                print(f"WARNING: missing pred for {stem}")
                continue
            gt_md = gt_file.read_text()
            pred_md = pred_file.read_text()
            pages.append((stem, gt_md, pred_md))
    return pages


def fuzzy_match_with_details(pred_spans, gt_spans, threshold=THRESHOLD):
    """Match pred to GT spans, returning matched pairs and unmatched on both sides.

    Returns:
        matched: list of (gt_text, pred_text, similarity)
        unmatched_pred: list of pred texts with no GT match
        unmatched_gt: list of GT texts with no pred match
    """
    if not pred_spans and not gt_spans:
        return [], [], []

    pred_list = sorted(pred_spans)
    gt_list = sorted(gt_spans)
    used_pred = set()
    matched = []
    matched_gt = set()

    for i, gt_s in enumerate(gt_list):
        best_sim = 0.0
        best_idx = -1
        for j, pred_s in enumerate(pred_list):
            if j in used_pred:
                continue
            dist = Levenshtein.distance(gt_s.lower(), pred_s.lower())
            max_len = max(len(gt_s), len(pred_s))
            sim = 1.0 - (dist / max_len) if max_len > 0 else 1.0
            if sim > best_sim:
                best_sim = sim
                best_idx = j
        if best_idx >= 0 and best_sim >= threshold:
            matched.append((gt_s, pred_list[best_idx], best_sim))
            used_pred.add(best_idx)
            matched_gt.add(i)

    unmatched_pred = [pred_list[j] for j in range(len(pred_list)) if j not in used_pred]
    unmatched_gt = [gt_list[i] for i in range(len(gt_list)) if i not in matched_gt]
    return matched, unmatched_pred, unmatched_gt


def analyze_all(pages):
    """Run all failure mode analyses across all pages."""

    # Accumulators
    all_hallucinated = []  # (stem, fmt_type, text)
    all_over_formatted = []  # (stem, fmt_type, text)
    all_type_swaps = []  # (stem, gt_type, pred_type, text)
    all_table_drops = []  # (stem, fmt_type, text)
    all_merge_errors = []  # (stem, fmt_type, pred_text, matched_gt_texts)
    all_missing_text = []  # (stem, fmt_type, text)

    for stem, gt_md, pred_md in pages:
        gt_spans = extract_formatting_spans(gt_md)
        pred_spans = extract_formatting_spans(pred_md)
        gt_plain = _strip_all_formatting(gt_md)
        pred_plain = _strip_all_formatting(pred_md)

        # Per-type matching
        per_type_matched_pred = set()  # pred texts matched within their own type
        per_type_matched_gt = set()  # gt texts matched within their own type

        for fmt_type in ["bold", "italic", "strikethrough"]:
            matched, unmatched_p, unmatched_g = fuzzy_match_with_details(
                pred_spans[fmt_type], gt_spans[fmt_type]
            )
            for gt_s, pred_s, sim in matched:
                per_type_matched_pred.add((fmt_type, pred_s))
                per_type_matched_gt.add((fmt_type, gt_s))

            # Unmatched pred spans for this type
            for p_text in unmatched_p:
                # Check if it's a hallucination or over-formatting
                if p_text.lower() in gt_plain.lower():
                    all_over_formatted.append((stem, fmt_type, p_text))
                else:
                    # Could be a type swap - check other GT types
                    swapped = False
                    for other_type in ["bold", "italic", "strikethrough"]:
                        if other_type == fmt_type:
                            continue
                        for gt_s in gt_spans[other_type]:
                            dist = Levenshtein.distance(gt_s.lower(), p_text.lower())
                            max_len = max(len(gt_s), len(p_text))
                            sim = 1.0 - (dist / max_len) if max_len > 0 else 1.0
                            if sim >= THRESHOLD:
                                all_type_swaps.append((stem, other_type, fmt_type, p_text))
                                swapped = True
                                break
                        if swapped:
                            break
                    if not swapped:
                        if p_text.lower() in gt_plain.lower():
                            all_over_formatted.append((stem, fmt_type, p_text))
                        else:
                            all_hallucinated.append((stem, fmt_type, p_text))

            # Unmatched GT spans
            for g_text in unmatched_g:
                # Check if inside a table row
                if "|" in g_text or any(
                    g_text in line for line in gt_md.split("\n") if "|" in line
                ):
                    all_table_drops.append((stem, fmt_type, g_text))

                # Check if text missing entirely from pred
                if g_text.lower() not in pred_plain.lower():
                    all_missing_text.append((stem, fmt_type, g_text))

        # Merge errors: pred span that contains multiple GT spans
        for fmt_type in ["bold", "italic", "strikethrough"]:
            for p_text in pred_spans[fmt_type]:
                contained_gt = []
                for g_text in gt_spans[fmt_type]:
                    if len(g_text) >= 3 and g_text.lower() in p_text.lower() and g_text.lower() != p_text.lower():
                        contained_gt.append(g_text)
                if len(contained_gt) >= 2:
                    all_merge_errors.append((stem, fmt_type, p_text, contained_gt))

    return {
        "hallucinated": all_hallucinated,
        "over_formatted": all_over_formatted,
        "type_swaps": all_type_swaps,
        "table_drops": all_table_drops,
        "merge_errors": all_merge_errors,
        "missing_text": all_missing_text,
    }


def print_results(results):
    """Print formatted analysis results."""

    print("=" * 80)
    print("FAILURE MODE ANALYSIS: fmt_sft-lora_qwen35-9b")
    print("=" * 80)

    # 1. Hallucinated spans
    hall = results["hallucinated"]
    print(f"\n{'=' * 80}")
    print(f"1. HALLUCINATED SPANS ({len(hall)} total)")
    print(f"   Pred spans where plain text doesn't appear anywhere in GT document")
    print(f"{'=' * 80}")
    for i, (stem, fmt_type, text) in enumerate(hall):
        trunc = text[:100] + "..." if len(text) > 100 else text
        print(f"  [{i+1:>3}] {stem} | {fmt_type:<15} | {trunc}")

    # 2. Over-formatted spans
    ovf = results["over_formatted"]
    print(f"\n{'=' * 80}")
    print(f"2. OVER-FORMATTED SPANS ({len(ovf)} total)")
    print(f"   Pred spans matching real GT text but shouldn't be formatted")
    print(f"{'=' * 80}")
    # Count by text
    text_counts = Counter(text for _, _, text in ovf)
    type_counts = Counter(fmt_type for _, fmt_type, _ in ovf)
    print(f"\n  By formatting type:")
    for ft, c in type_counts.most_common():
        print(f"    {ft}: {c}")
    print(f"\n  Top 20 most common over-formatted texts:")
    for i, (text, count) in enumerate(text_counts.most_common(20)):
        trunc = text[:80] + "..." if len(text) > 80 else text
        # Find which types this text appears under
        types_for_text = set(ft for _, ft, t in ovf if t == text)
        print(f"  [{i+1:>3}] (x{count}) [{', '.join(types_for_text)}] {trunc}")

    # 3. Type swaps
    swaps = results["type_swaps"]
    print(f"\n{'=' * 80}")
    print(f"3. TYPE SWAPS ({len(swaps)} total)")
    print(f"   GT spans matched by pred but under wrong formatting type")
    print(f"{'=' * 80}")
    swap_matrix = Counter((gt_t, pred_t) for _, gt_t, pred_t, _ in swaps)
    print(f"\n  Confusion matrix (GT type -> Pred type):")
    for (gt_t, pred_t), count in swap_matrix.most_common():
        print(f"    {gt_t} -> {pred_t}: {count}")
    # Show top 10 examples for most common swap direction
    if swap_matrix:
        top_gt, top_pred = swap_matrix.most_common(1)[0][0]
        print(f"\n  Top 10 examples of most common swap ({top_gt} -> {top_pred}):")
        examples = [(s, t) for s, gt, pr, t in swaps if gt == top_gt and pr == top_pred]
        for i, (stem, text) in enumerate(examples[:10]):
            trunc = text[:80] + "..." if len(text) > 80 else text
            print(f"  [{i+1:>3}] {stem} | {trunc}")

    # Also show examples for second most common if exists
    if len(swap_matrix) >= 2:
        sec_gt, sec_pred = swap_matrix.most_common(2)[1][0]
        print(f"\n  Top 10 examples of second most common swap ({sec_gt} -> {sec_pred}):")
        examples = [(s, t) for s, gt, pr, t in swaps if gt == sec_gt and pr == sec_pred]
        for i, (stem, text) in enumerate(examples[:10]):
            trunc = text[:80] + "..." if len(text) > 80 else text
            print(f"  [{i+1:>3}] {stem} | {trunc}")

    # 4. Table formatting drops
    td = results["table_drops"]
    print(f"\n{'=' * 80}")
    print(f"4. TABLE FORMATTING DROPS ({len(td)} total)")
    print(f"   Missed GT spans inside markdown table rows")
    print(f"{'=' * 80}")
    type_dist = Counter(ft for _, ft, _ in td)
    print(f"\n  By formatting type:")
    for ft, c in type_dist.most_common():
        print(f"    {ft}: {c}")
    print(f"\n  5 examples:")
    for i, (stem, fmt_type, text) in enumerate(td[:5]):
        trunc = text[:80] + "..." if len(text) > 80 else text
        print(f"  [{i+1}] {stem} | {fmt_type:<15} | {trunc}")

    # 5. Merge errors
    me = results["merge_errors"]
    print(f"\n{'=' * 80}")
    print(f"5. MERGE ERRORS ({len(me)} total)")
    print(f"   Pred span contains multiple GT spans merged together")
    print(f"{'=' * 80}")
    for i, (stem, fmt_type, pred_text, gt_texts) in enumerate(me[:5]):
        p_trunc = pred_text[:80] + "..." if len(pred_text) > 80 else pred_text
        print(f"  [{i+1}] {stem} | {fmt_type}")
        print(f"       Pred: {p_trunc}")
        print(f"       GT spans ({len(gt_texts)}): {gt_texts[:5]}")

    # 6. Text missing entirely
    mt = results["missing_text"]
    print(f"\n{'=' * 80}")
    print(f"6. TEXT MISSING ENTIRELY ({len(mt)} total)")
    print(f"   GT formatted text not found anywhere in pred output")
    print(f"{'=' * 80}")
    type_dist = Counter(ft for _, ft, _ in mt)
    print(f"\n  By formatting type:")
    for ft, c in type_dist.most_common():
        print(f"    {ft}: {c}")
    print(f"\n  Top 10 examples:")
    for i, (stem, fmt_type, text) in enumerate(mt[:10]):
        trunc = text[:80] + "..." if len(text) > 80 else text
        print(f"  [{i+1:>3}] {stem} | {fmt_type:<15} | {trunc}")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Hallucinated spans:     {len(hall)}")
    print(f"  Over-formatted spans:   {len(ovf)}")
    print(f"  Type swaps:             {len(swaps)}")
    print(f"  Table formatting drops: {len(td)}")
    print(f"  Merge errors:           {len(me)}")
    print(f"  Text missing entirely:  {len(mt)}")


if __name__ == "__main__":
    print("Loading pages...")
    pages = load_pages()
    print(f"Loaded {len(pages)} pages")
    print("Running analysis...")
    results = analyze_all(pages)
    print_results(results)
