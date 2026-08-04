"""Formatting bench evaluation metrics.

Measures inline formatting preservation (bold, italic, strikethrough)
between predicted and ground truth markdown. Computes precision, recall,
and F1 using fuzzy Levenshtein matching of formatting spans.

Used by:
    scripts/benchmarks/formatting_bench.py - Formatting bench scoring
    src/training/rewards.py                - inline_formatting_reward (GRPO)
"""

import re

from rapidfuzz.distance import Levenshtein


def _strip_inner_markers(span: str) -> str:
    """Strip nested formatting markers from a captured span.

    Args:
        span: Raw text captured between outer formatting markers.

    Returns:
        Text with inner bold/italic/strikethrough markers removed.
    """
    span = re.sub(r"\*{1,3}", "", span)
    span = re.sub(r"~~", "", span)
    return span.strip()


def extract_formatting_spans(text: str) -> dict[str, set[str]]:
    """Extract inline formatting spans from markdown text.

    Handles nested formatting: ``***~~hello~~***`` registers under bold,
    italic, and strikethrough. Inner markers are stripped so each type
    stores the plain text content.

    Args:
        text: Markdown text.

    Returns:
        Dict with keys 'bold', 'italic', 'strikethrough', each a set of
        the plain text content within those markers.
    """
    bold = set()
    italic = set()
    strikethrough = set()

    # Bold+italic (*** ... ***) - must match before ** and *
    for raw in re.findall(r"\*{3}(.+?)\*{3}", text):
        clean = _strip_inner_markers(raw)
        if clean:
            bold.add(clean)
            italic.add(clean)

    # Bold (** ... **) - exclude ***
    for raw in re.findall(r"(?<!\*)\*{2}(?!\*)(.+?)(?<!\*)\*{2}(?!\*)", text):
        clean = _strip_inner_markers(raw)
        if clean:
            bold.add(clean)

    # Italic (* ... *) - exclude ** and ***
    for raw in re.findall(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", text):
        clean = _strip_inner_markers(raw)
        if clean:
            italic.add(clean)

    # Strikethrough (~~ ... ~~)
    for raw in re.findall(r"~~(.+?)~~", text):
        clean = _strip_inner_markers(raw)
        if clean:
            strikethrough.add(clean)

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


def fuzzy_set_match(pred_spans: set[str], gt_spans: set[str], threshold: float = 0.5) -> dict:
    """Compute precision, recall, and F1 between two sets of text spans.

    Uses fuzzy Levenshtein matching: each GT span is matched to the most
    similar unmatched predicted span. A match counts if similarity >= threshold.

    Args:
        pred_spans: Predicted formatting spans.
        gt_spans: Ground truth formatting spans.
        threshold: Minimum Levenshtein similarity to count as match.

    Returns:
        Dict with 'precision', 'recall', 'f1' keys, each in [0, 1].
    """
    zero = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not gt_spans:
        perfect = {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        return perfect if not pred_spans else zero
    if not pred_spans:
        return zero

    matched = 0.0
    used = set()
    pred_list = sorted(pred_spans)
    for gt_s in sorted(gt_spans):
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
        return zero
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _strip_all_formatting(text: str) -> str:
    """Strip all inline formatting markers from markdown text.

    Args:
        text: Markdown text with formatting markers.

    Returns:
        Plain text with bold/italic/strikethrough markers removed.
    """
    text = re.sub(r"\*{1,3}", "", text)
    text = re.sub(r"~~", "", text)
    return text


def _classify_unmatched_spans(
    pred_spans: set[str],
    gt_spans: set[str],
    gt_plain: str,
    threshold: float = 0.5,
) -> dict:
    """Classify unmatched predicted spans as over-formatting or hallucination.

    For each predicted span that doesn't fuzzy-match any GT formatting span,
    checks whether its text appears in the full GT plain text. If it does,
    the model over-formatted real text. If not, the model hallucinated content.

    Args:
        pred_spans: Predicted formatting spans (plain text).
        gt_spans: Ground truth formatting spans (plain text).
        gt_plain: Full GT document with formatting markers stripped.
        threshold: Minimum Levenshtein similarity for matching.

    Returns:
        Dict with 'total_pred', 'matched', 'over_formatted', 'hallucinated',
        and 'hallucination_rate' (hallucinated / total_pred, or 0 if none).
    """
    if not pred_spans:
        return {
            "total_pred": 0,
            "matched": 0,
            "over_formatted": 0,
            "hallucinated": 0,
            "hallucination_rate": 0.0,
        }

    # Find which pred spans matched GT formatting spans (same logic as fuzzy_set_match)
    used = set()
    pred_list = sorted(pred_spans)
    matched_indices = set()
    for gt_s in sorted(gt_spans):
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
            matched_indices.add(best_idx)
            used.add(best_idx)

    # Classify unmatched spans
    gt_lower = gt_plain.lower()
    over_formatted = 0
    hallucinated = 0
    for j, pred_s in enumerate(pred_list):
        if j in matched_indices:
            continue
        # Check if this text exists anywhere in the GT document
        if pred_s.lower() in gt_lower:
            over_formatted += 1
        else:
            hallucinated += 1

    total = len(pred_list)
    return {
        "total_pred": total,
        "matched": len(matched_indices),
        "over_formatted": over_formatted,
        "hallucinated": hallucinated,
        "hallucination_rate": hallucinated / total if total > 0 else 0.0,
    }


def score_formatting(pred_md: str, gt_md: str) -> dict:
    """Score inline formatting preservation for a single page.

    Args:
        pred_md: Predicted markdown.
        gt_md: Ground truth markdown.

    Returns:
        Dict with per-type scores (precision, recall, F1), overall scores,
        span counts, and hallucination breakdown. Returns None for types
        not present in GT.
    """
    gt_spans = extract_formatting_spans(gt_md)
    pred_spans = extract_formatting_spans(pred_md)

    all_gt = gt_spans["bold"] | gt_spans["italic"] | gt_spans["strikethrough"]
    all_pred = pred_spans["bold"] | pred_spans["italic"] | pred_spans["strikethrough"]

    result = {
        "gt_counts": {k: len(v) for k, v in gt_spans.items()},
        "pred_counts": {k: len(v) for k, v in pred_spans.items()},
    }

    for fmt_type in ["bold", "italic", "strikethrough"]:
        if gt_spans[fmt_type]:
            result[fmt_type] = fuzzy_set_match(pred_spans[fmt_type], gt_spans[fmt_type])
        else:
            result[fmt_type] = None

    if all_gt:
        result["overall"] = fuzzy_set_match(all_pred, all_gt)
    else:
        result["overall"] = None

    # Hallucination breakdown: classify unmatched predicted spans
    gt_plain = _strip_all_formatting(gt_md)
    result["hallucination"] = _classify_unmatched_spans(
        all_pred,
        all_gt,
        gt_plain,
    )

    return result


def compute_aggregate(results: list[dict]) -> dict:
    """Compute aggregate formatting scores.

    Args:
        results: List of per-page score dicts.

    Returns:
        Dict with per-type and overall averages.
    """
    metrics = ["bold", "italic", "strikethrough", "overall"]
    stat_keys = ["precision", "recall", "f1"]
    sums = {m: dict.fromkeys(stat_keys, 0.0) for m in metrics}
    counts = dict.fromkeys(metrics, 0)

    total_gt = {"bold": 0, "italic": 0, "strikethrough": 0}
    total_pred = {"bold": 0, "italic": 0, "strikethrough": 0}
    pages_with_formatting = 0

    # Hallucination aggregation
    hall_keys = ["total_pred", "matched", "over_formatted", "hallucinated"]
    hall_totals = dict.fromkeys(hall_keys, 0)

    for r in results:
        for m in metrics:
            val = r.get(m)
            if val is not None:
                # Support both old format (float = f1 only) and new (dict with p/r/f1)
                if isinstance(val, dict):
                    for s in stat_keys:
                        sums[m][s] += val.get(s, 0.0)
                else:
                    sums[m]["f1"] += val
                counts[m] += 1
        for k in total_gt:
            total_gt[k] += r.get("gt_counts", {}).get(k, 0)
            total_pred[k] += r.get("pred_counts", {}).get(k, 0)
        if r.get("overall") is not None:
            pages_with_formatting += 1
        hall = r.get("hallucination", {})
        for k in hall_keys:
            hall_totals[k] += hall.get(k, 0)

    scores = {}
    for m in metrics:
        if counts[m] > 0:
            scores[m] = {s: round(sums[m][s] / counts[m], 4) for s in stat_keys}
        else:
            scores[m] = None

    total_p = hall_totals["total_pred"]
    hall_totals["hallucination_rate"] = round(
        hall_totals["hallucinated"] / total_p if total_p > 0 else 0.0, 4
    )

    return {
        "scores": scores,
        "counts": counts,
        "total_pages": len(results),
        "pages_with_formatting": pages_with_formatting,
        "gt_span_totals": total_gt,
        "pred_span_totals": total_pred,
        "hallucination": hall_totals,
    }
