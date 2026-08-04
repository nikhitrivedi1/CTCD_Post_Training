"""Document quality evaluation metrics.

Scoring functions for document conversion quality:
- Text: normalized edit distance on text blocks
- Tables: TEDS (Tree Edit Distance-based Similarity) on HTML tables
- Formulas: normalized edit distance on cleaned LaTeX
- Reading order: normalized edit distance on block sequence
- Overall: average of (text_similarity, TEDS, formula_similarity)

Reference: OmniDocBench (CVPR 2025), https://github.com/opendatalab/OmniDocBench

Note: The official pipeline uses CDM (Character Distance Metric) for formulas,
which requires a LaTeX rendering pipeline (Node.js + KaTeX). We use normalized
edit distance on cleaned LaTeX as an approximation, which the authors also
report as a secondary metric.
"""

import logging
import re

from rapidfuzz.distance import Levenshtein

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Edit distance utilities
# ---------------------------------------------------------------------------


def normalized_edit_distance(pred: str, gt: str) -> float:
    """Compute normalized edit distance (error rate) in [0, 1].

    Args:
        pred: Predicted string.
        gt: Ground truth string.

    Returns:
        Normalized distance in [0, 1]. 0 = identical.
    """
    if not gt and not pred:
        return 0.0
    if not gt or not pred:
        return 1.0
    dist = Levenshtein.distance(pred, gt)
    max_len = max(len(pred), len(gt))
    return dist / max_len if max_len > 0 else 0.0


def edit_similarity(pred: str, gt: str) -> float:
    """Compute edit similarity (1 - normalized edit distance) in [0, 1].

    Args:
        pred: Predicted string.
        gt: Ground truth string.

    Returns:
        Similarity in [0, 1]. 1 = identical.
    """
    return 1.0 - normalized_edit_distance(pred, gt)


# ---------------------------------------------------------------------------
# Block extraction
# ---------------------------------------------------------------------------


def extract_text_blocks(text: str, min_length: int = 10) -> list[str]:
    """Split markdown into text blocks (paragraphs separated by blank lines).

    Args:
        text: Markdown text.
        min_length: Minimum block length to include.

    Returns:
        List of non-trivial text blocks.
    """
    blocks = []
    for block in text.split("\n\n"):
        block = block.strip()
        if block and len(block) >= min_length:
            blocks.append(block)
    return blocks


def extract_markdown_tables(text: str) -> list[str]:
    """Extract raw markdown table strings from text.

    Args:
        text: Markdown text.

    Returns:
        List of markdown table strings (including pipe rows).
    """
    tables = []
    current_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            current_lines.append(stripped)
        else:
            if current_lines:
                tables.append("\n".join(current_lines))
                current_lines = []
    if current_lines:
        tables.append("\n".join(current_lines))
    return tables


def extract_latex(text: str) -> list[str]:
    """Extract LaTeX formulas from markdown text.

    Args:
        text: Markdown text with $$...$$ or $...$ formulas.

    Returns:
        List of cleaned LaTeX strings.
    """
    formulas = []
    for match in re.finditer(r"\$\$(.*?)\$\$", text, re.DOTALL):
        formulas.append(_clean_latex(match.group(1)))
    for match in re.finditer(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", text):
        formulas.append(_clean_latex(match.group(1)))
    return formulas


def _clean_latex(latex: str) -> str:
    """Normalize LaTeX for comparison."""
    latex = latex.strip()
    latex = re.sub(r"\s+", " ", latex)
    latex = latex.replace(r"\left", "").replace(r"\right", "")
    latex = latex.replace(r"\displaystyle", "").replace(r"\textstyle", "")
    return latex.strip()


# ---------------------------------------------------------------------------
# Markdown table -> HTML conversion
# ---------------------------------------------------------------------------


def markdown_table_to_html(md_table: str) -> str:
    """Convert a markdown table string to HTML.

    Args:
        md_table: Markdown table string with pipe-delimited rows.

    Returns:
        HTML table string suitable for TEDS scoring.
    """
    rows = []
    for line in md_table.split("\n"):
        line = line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        # Skip separator rows (|---|---|)
        if all(re.match(r"^:?-+:?$", c) for c in cells if c):
            continue
        rows.append(cells)

    if not rows:
        return "<table></table>"

    html_rows = []
    for row in rows:
        cells_html = "".join(f"<td>{cell}</td>" for cell in row)
        html_rows.append(f"<tr>{cells_html}</tr>")

    return f"<table>{''.join(html_rows)}</table>"


# ---------------------------------------------------------------------------
# TEDS (Tree Edit Distance-based Similarity)
# ---------------------------------------------------------------------------


def _check_teds_deps():
    """Check if TEDS dependencies are available.

    Returns:
        True if apted and lxml are importable.
    """
    try:
        import apted  # noqa: F401
        from lxml import html  # noqa: F401

        return True
    except ImportError:
        return False


class _TableTree:
    """Tree node for TEDS computation, compatible with apted."""

    def __init__(self, tag, colspan=None, rowspan=None, content=None):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = []

    @property
    def name(self):
        return self.tag


class _TEDSConfig:
    """Custom apted config for table tree comparison.

    Implements the apted Config interface: rename/delete/insert costs,
    children accessor, and valuecls for cost accumulation.
    """

    # apted uses valuecls as a constructor: valuecls() -> 0, valuecls(n) -> n
    valuecls = int

    def __init__(self, structure_only=False):
        self.structure_only = structure_only

    def rename(self, node1, node2):
        """Cost of renaming node1 to node2."""
        if node1.tag != node2.tag:
            return 1.0
        if node1.colspan != node2.colspan or node1.rowspan != node2.rowspan:
            return 1.0
        if self.structure_only:
            return 0.0
        if node1.tag == "td" and (node1.content or node2.content):
            s1 = node1.content or ""
            s2 = node2.content or ""
            if not s1 and not s2:
                return 0.0
            max_len = max(len(s1), len(s2))
            return Levenshtein.distance(s1, s2) / max_len if max_len > 0 else 0.0
        return 0.0

    def children(self, node):
        """Return children of a node."""
        return node.children

    def delete(self, node):
        """Cost of deleting a node."""
        return 1.0

    def insert(self, node):
        """Cost of inserting a node."""
        return 1.0


def _html_to_tree(html_str: str) -> "_TableTree | None":
    """Parse an HTML table string into a _TableTree.

    Args:
        html_str: HTML table string.

    Returns:
        Root _TableTree node, or None if parsing fails.
    """
    from lxml import html

    try:
        doc = html.fromstring(
            html_str, parser=html.HTMLParser(remove_comments=True, encoding="utf-8")
        )
    except Exception:
        return None

    tables = doc.xpath("body/table") if doc.xpath("body/table") else doc.xpath(".//table")
    if not tables:
        # The string might be a bare <table> element
        if doc.tag == "table":
            table_el = doc
        else:
            return None
    else:
        table_el = tables[0]

    def _build(element):
        if element.tag == "td" or element.tag == "th":
            # Collect all text content in the cell
            content = element.text_content().strip()
            node = _TableTree(
                tag="td",
                colspan=int(element.get("colspan", "1")),
                rowspan=int(element.get("rowspan", "1")),
                content=content,
            )
        else:
            node = _TableTree(tag=element.tag)
            for child in element:
                child_node = _build(child)
                if child_node is not None:
                    node.children.append(child_node)
        return node

    return _build(table_el)


def _count_nodes(tree: "_TableTree") -> int:
    """Count total nodes in a tree."""
    count = 1
    for child in tree.children:
        count += _count_nodes(child)
    return count


def teds_score(pred_html: str, gt_html: str, structure_only: bool = False) -> float:
    """Compute TEDS between two HTML table strings.

    Tree Edit Distance-based Similarity: 1 - (tree_edit_distance / max_nodes).
    Uses the APTED algorithm for optimal tree edit distance.

    Args:
        pred_html: Predicted HTML table.
        gt_html: Ground truth HTML table.
        structure_only: If True, ignore cell content (compare structure only).

    Returns:
        TEDS score in [0, 1]. 1 = identical.
    """
    from apted import APTED

    if not pred_html or not gt_html:
        return 0.0

    pred_tree = _html_to_tree(pred_html)
    gt_tree = _html_to_tree(gt_html)

    if pred_tree is None or gt_tree is None:
        return 0.0

    n_pred = _count_nodes(pred_tree)
    n_gt = _count_nodes(gt_tree)
    n_nodes = max(n_pred, n_gt)

    if n_nodes == 0:
        return 1.0

    config = _TEDSConfig(structure_only=structure_only)
    distance = APTED(pred_tree, gt_tree, config).compute_edit_distance()
    return max(0.0, 1.0 - float(distance) / n_nodes)


# ---------------------------------------------------------------------------
# Hungarian matching
# ---------------------------------------------------------------------------


def hungarian_match(
    pred_items: list[str], gt_items: list[str], similarity_fn=None
) -> list[tuple[int, int, float]]:
    """Match predicted items to GT items using the Hungarian algorithm.

    Finds the optimal 1-to-1 matching that maximizes total similarity.

    Args:
        pred_items: List of predicted strings.
        gt_items: List of ground truth strings.
        similarity_fn: Function(pred, gt) -> float in [0, 1].
            Defaults to edit_similarity.

    Returns:
        List of (pred_idx, gt_idx, score) tuples for matched pairs.
        Unmatched GT items are not included (count as 0).
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if not pred_items or not gt_items:
        return []

    if similarity_fn is None:
        similarity_fn = edit_similarity

    # Build similarity matrix
    n_pred = len(pred_items)
    n_gt = len(gt_items)
    sim_matrix = np.zeros((n_pred, n_gt))

    for i, pred in enumerate(pred_items):
        for j, gt in enumerate(gt_items):
            sim_matrix[i, j] = similarity_fn(pred, gt)

    # Hungarian algorithm minimizes cost, so use 1 - similarity as cost
    cost_matrix = 1.0 - sim_matrix
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

    matches = []
    for pi, gi in zip(pred_indices, gt_indices, strict=True):
        matches.append((pi, gi, sim_matrix[pi, gi]))

    return matches


# ---------------------------------------------------------------------------
# Per-page scoring functions
# ---------------------------------------------------------------------------


def score_text(pred: str, gt: str) -> float:
    """Score text extraction quality via normalized edit distance.

    Args:
        pred: Predicted markdown.
        gt: Ground truth markdown.

    Returns:
        Similarity score in [0, 1].
    """
    pred_blocks = extract_text_blocks(pred)
    gt_blocks = extract_text_blocks(gt)

    if not gt_blocks:
        return 1.0 if not pred_blocks else 0.0

    pred_text = "\n".join(pred_blocks)
    gt_text = "\n".join(gt_blocks)
    return edit_similarity(pred_text, gt_text)


def score_tables(pred: str, gt: str) -> float | None:
    """Score table quality via TEDS with Hungarian matching.

    Extracts markdown tables, converts to HTML, matches predicted tables
    to GT tables using Hungarian algorithm, then scores each pair with TEDS.

    Args:
        pred: Predicted markdown.
        gt: Ground truth markdown.

    Returns:
        Average TEDS score in [0, 1], or None if GT has no tables.
    """
    gt_tables = extract_markdown_tables(gt)
    if not gt_tables:
        return None

    pred_tables = extract_markdown_tables(pred)
    if not pred_tables:
        return 0.0

    if not _check_teds_deps():
        logger.warning("apted/lxml not installed, falling back to edit distance for tables")
        return _score_tables_fallback(pred_tables, gt_tables)

    # Convert to HTML
    pred_html = [markdown_table_to_html(t) for t in pred_tables]
    gt_html = [markdown_table_to_html(t) for t in gt_tables]

    # Hungarian matching using TEDS as similarity
    matches = hungarian_match(pred_html, gt_html, similarity_fn=teds_score)

    # Sum matched scores, unmatched GT tables score 0
    matched_gt = set()
    total_score = 0.0
    for _, gi, score in matches:
        matched_gt.add(gi)
        total_score += score

    return total_score / len(gt_tables)


def _score_tables_fallback(pred_tables: list[str], gt_tables: list[str]) -> float:
    """Fallback table scoring using edit distance when apted is not available.

    Args:
        pred_tables: List of predicted markdown table strings.
        gt_tables: List of GT markdown table strings.

    Returns:
        Average edit similarity across matched tables.
    """
    matches = hungarian_match(pred_tables, gt_tables, similarity_fn=edit_similarity)

    total_score = 0.0
    for _, _, score in matches:
        total_score += score

    return total_score / len(gt_tables)


def score_formulas(pred: str, gt: str) -> float | None:
    """Score formula quality via edit distance with Hungarian matching.

    Args:
        pred: Predicted markdown.
        gt: Ground truth markdown.

    Returns:
        Average edit similarity in [0, 1], or None if GT has no formulas.
    """
    gt_formulas = extract_latex(gt)
    if not gt_formulas:
        return None

    pred_formulas = extract_latex(pred)
    if not pred_formulas:
        return 0.0

    matches = hungarian_match(pred_formulas, gt_formulas, similarity_fn=edit_similarity)

    total_score = 0.0
    for _, _, score in matches:
        total_score += score

    return total_score / len(gt_formulas)


def score_reading_order(pred: str, gt: str) -> float | None:
    """Score reading order via normalized edit distance on matched block sequence.

    Matches predicted text blocks to GT blocks by content similarity,
    then compares the ordering using edit distance.

    Args:
        pred: Predicted markdown.
        gt: Ground truth markdown.

    Returns:
        Similarity score in [0, 1], or None if GT has insufficient blocks.
    """
    gt_blocks = extract_text_blocks(gt)
    if len(gt_blocks) < 2:
        return None

    pred_blocks = extract_text_blocks(pred)
    if not pred_blocks:
        return 0.0

    # Match pred blocks to GT blocks by prefix similarity
    gt_prefixes = [b[:50] for b in gt_blocks]
    pred_prefixes = [b[:50] for b in pred_blocks]

    # Build order sequence: for each pred block, find best matching GT index
    order = []
    used = set()
    for pred_p in pred_prefixes:
        best_sim = 0.0
        best_idx = -1
        for j, gt_p in enumerate(gt_prefixes):
            if j in used:
                continue
            sim = edit_similarity(pred_p, gt_p)
            if sim > best_sim:
                best_sim = sim
                best_idx = j
        if best_idx >= 0 and best_sim > 0.3:
            order.append(best_idx)
            used.add(best_idx)

    if not order:
        return 0.0

    # Compare predicted order to natural GT order using edit distance.
    # Only compare indices that were matched.
    gt_matched_order = sorted(order)

    # Normalize: edit distance between the predicted ordering and sorted ordering
    order_str = " ".join(str(i) for i in order)
    natural_str = " ".join(str(i) for i in gt_matched_order)

    return edit_similarity(order_str, natural_str)


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


def compute_aggregate(results: list[tuple[str, dict]]) -> dict:
    """Compute aggregate scores from per-page results.

    For each metric, averages across pages where the metric is applicable.
    Overall follows the official OmniDocBench formula: average of text,
    table, and formula scores (reading order is reported separately).

    Args:
        results: List of (stem, scores_dict) tuples.

    Returns:
        Dict with per-metric averages, overall score, and counts.
    """
    metrics = ["text", "table", "formula", "reading_order"]
    sums = dict.fromkeys(metrics, 0.0)
    counts = dict.fromkeys(metrics, 0)

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

    # Official overall: average of text, table, formula (not reading order)
    overall_metrics = ["text", "table", "formula"]
    valid = [averages[m] for m in overall_metrics if averages[m] is not None]
    averages["overall"] = round(sum(valid) / len(valid), 4) if valid else 0.0
    averages["counts"] = counts

    return averages
