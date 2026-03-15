"""Tests for OmniDocBench evaluation metrics."""

import pytest

from src.evaluation.document_metrics import (
    normalized_edit_distance,
    edit_similarity,
    extract_text_blocks,
    extract_markdown_tables,
    extract_latex,
    markdown_table_to_html,
    teds_score,
    hungarian_match,
    score_text,
    score_tables,
    score_formulas,
    score_reading_order,
    score_page,
    compute_aggregate,
)


# ---------------------------------------------------------------------------
# Edit distance utilities
# ---------------------------------------------------------------------------
class TestEditDistance:
    def test_identical(self):
        assert normalized_edit_distance("hello", "hello") == 0.0
        assert edit_similarity("hello", "hello") == 1.0

    def test_completely_different(self):
        assert normalized_edit_distance("abc", "xyz") == 1.0
        assert edit_similarity("abc", "xyz") == 0.0

    def test_partial_match(self):
        dist = normalized_edit_distance("hello", "helo")
        assert 0.0 < dist < 1.0
        assert edit_similarity("hello", "helo") == pytest.approx(1.0 - dist)

    def test_empty_strings(self):
        assert normalized_edit_distance("", "") == 0.0
        assert edit_similarity("", "") == 1.0

    def test_one_empty(self):
        assert normalized_edit_distance("hello", "") == 1.0
        assert normalized_edit_distance("", "hello") == 1.0


# ---------------------------------------------------------------------------
# Block extraction
# ---------------------------------------------------------------------------
class TestExtractTextBlocks:
    def test_basic(self):
        text = "First paragraph here.\n\nSecond paragraph here."
        blocks = extract_text_blocks(text)
        assert len(blocks) == 2

    def test_short_blocks_excluded(self):
        text = "Short.\n\nThis is a long enough paragraph."
        blocks = extract_text_blocks(text)
        assert len(blocks) == 1

    def test_empty(self):
        assert extract_text_blocks("") == []


class TestExtractMarkdownTables:
    def test_single_table(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        tables = extract_markdown_tables(md)
        assert len(tables) == 1

    def test_two_tables(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |\n\nSome text\n\n| X | Y |\n|---|---|\n| 3 | 4 |"
        tables = extract_markdown_tables(md)
        assert len(tables) == 2

    def test_no_tables(self):
        assert extract_markdown_tables("Just plain text") == []


class TestExtractLatex:
    def test_display_math(self):
        formulas = extract_latex("Here is $$E = mc^2$$ a formula")
        assert len(formulas) == 1
        assert "E = mc^2" in formulas[0]

    def test_inline_math(self):
        formulas = extract_latex("Here is $x + y$ inline")
        assert len(formulas) == 1

    def test_no_formulas(self):
        assert extract_latex("No math here") == []


# ---------------------------------------------------------------------------
# Markdown -> HTML table conversion
# ---------------------------------------------------------------------------
class TestMarkdownTableToHtml:
    def test_basic_table(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        html = markdown_table_to_html(md)
        assert "<table>" in html
        assert "<tr>" in html
        assert "<td>A</td>" in html
        assert "<td>1</td>" in html
        assert "<td>2</td>" in html

    def test_separator_excluded(self):
        md = "| H1 | H2 |\n|---|---|\n| C1 | C2 |"
        html = markdown_table_to_html(md)
        # Separator row should not appear as a data row
        assert html.count("<tr>") == 2  # header + 1 data row

    def test_empty_table(self):
        html = markdown_table_to_html("")
        assert html == "<table></table>"


# ---------------------------------------------------------------------------
# TEDS
# ---------------------------------------------------------------------------
class TestTEDS:
    @pytest.fixture(autouse=True)
    def check_deps(self):
        pytest.importorskip("apted")
        pytest.importorskip("lxml")

    def test_identical_tables(self):
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"
        assert teds_score(html, html) == 1.0

    def test_completely_different(self):
        pred = "<table><tr><td>X</td></tr></table>"
        gt = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
        score = teds_score(pred, gt)
        assert 0.0 <= score < 0.5

    def test_empty_input(self):
        assert teds_score("", "<table><tr><td>A</td></tr></table>") == 0.0
        assert teds_score("<table><tr><td>A</td></tr></table>", "") == 0.0

    def test_similar_content(self):
        pred = "<table><tr><td>Hello</td><td>World</td></tr></table>"
        gt = "<table><tr><td>Hello</td><td>World!</td></tr></table>"
        score = teds_score(pred, gt)
        assert score > 0.8

    def test_structure_only(self):
        pred = "<table><tr><td>AAA</td><td>BBB</td></tr></table>"
        gt = "<table><tr><td>XXX</td><td>YYY</td></tr></table>"
        # Same structure, different content
        struct_score = teds_score(pred, gt, structure_only=True)
        content_score = teds_score(pred, gt, structure_only=False)
        assert struct_score == 1.0
        assert content_score < 1.0


# ---------------------------------------------------------------------------
# Hungarian matching
# ---------------------------------------------------------------------------
class TestHungarianMatch:
    def test_exact_matches(self):
        matches = hungarian_match(["hello", "world"], ["hello", "world"])
        scores = {(m[0], m[1]): m[2] for m in matches}
        assert scores[(0, 0)] == 1.0 or scores[(0, 1)] == 1.0

    def test_best_assignment(self):
        # pred[0] matches gt[1] better, pred[1] matches gt[0] better
        matches = hungarian_match(["aaa", "bbb"], ["bbb", "aaa"])
        matched_gt = {m[1] for m in matches}
        assert len(matched_gt) == 2  # both GT items matched

    def test_empty_pred(self):
        assert hungarian_match([], ["hello"]) == []

    def test_empty_gt(self):
        assert hungarian_match(["hello"], []) == []

    def test_unequal_sizes(self):
        matches = hungarian_match(["a"], ["a", "b", "c"])
        assert len(matches) == 1  # only 1 pred, so at most 1 match


# ---------------------------------------------------------------------------
# Per-page scoring
# ---------------------------------------------------------------------------
class TestScoreText:
    def test_identical(self):
        text = "This is a long enough paragraph for testing purposes."
        assert score_text(text, text) == 1.0

    def test_no_gt_blocks(self):
        assert score_text("", "") == 1.0
        # Pred has content but GT doesn't -> 0
        assert score_text("Some extra text here.", "") == 0.0

    def test_different(self):
        pred = "Completely different content here for testing."
        gt = "This is the original text that should be matched."
        score = score_text(pred, gt)
        assert 0.0 <= score < 1.0


class TestScoreTables:
    @pytest.fixture(autouse=True)
    def check_deps(self):
        pytest.importorskip("apted")
        pytest.importorskip("lxml")

    def test_no_gt_tables(self):
        assert score_tables("| A | B |", "plain text") is None

    def test_no_pred_tables(self):
        assert score_tables("plain text", "| A | B |\n|---|---|\n| 1 | 2 |") == 0.0

    def test_identical_tables(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        score = score_tables(md, md)
        assert score == 1.0

    def test_similar_tables(self):
        pred = "| A | B |\n|---|---|\n| 1 | 2 |"
        gt = "| A | B |\n|---|---|\n| 1 | 3 |"
        score = score_tables(pred, gt)
        assert 0.5 < score < 1.0


class TestScoreFormulas:
    def test_no_gt_formulas(self):
        assert score_formulas("$x$", "no math") is None

    def test_no_pred_formulas(self):
        assert score_formulas("no math", "$x + y$") == 0.0

    def test_identical(self):
        md = "Here is $E = mc^2$ a formula"
        assert score_formulas(md, md) == 1.0

    def test_partial_match(self):
        pred = "Here is $E = mc^2$"
        gt = "Here is $E = mc^3$"
        score = score_formulas(pred, gt)
        assert 0.5 < score < 1.0


class TestScoreReadingOrder:
    def test_insufficient_blocks(self):
        assert score_reading_order("short", "short") is None

    def test_no_pred(self):
        gt = "First paragraph text.\n\nSecond paragraph text."
        assert score_reading_order("", gt) == 0.0

    def test_correct_order(self):
        gt = "First paragraph text.\n\nSecond paragraph text.\n\nThird paragraph text."
        assert score_reading_order(gt, gt) == 1.0

    def test_reversed_order(self):
        gt = "First paragraph text.\n\nSecond paragraph text.\n\nThird paragraph text."
        pred = "Third paragraph text.\n\nSecond paragraph text.\n\nFirst paragraph text."
        score = score_reading_order(pred, gt)
        assert score < 1.0


# ---------------------------------------------------------------------------
# score_page
# ---------------------------------------------------------------------------
class TestScorePage:
    def test_returns_all_keys(self):
        result = score_page("some text content here.", "some text content here.")
        assert "text" in result
        assert "table" in result
        assert "formula" in result
        assert "reading_order" in result


# ---------------------------------------------------------------------------
# compute_aggregate
# ---------------------------------------------------------------------------
class TestComputeAggregate:
    def test_basic(self):
        results = [
            ("p1", {"text": 0.8, "table": 0.6, "formula": None, "reading_order": 0.7}),
            ("p2", {"text": 0.9, "table": None, "formula": 0.5, "reading_order": 0.8}),
        ]
        agg = compute_aggregate(results)
        assert agg["text"] == pytest.approx(0.85)
        assert agg["counts"]["text"] == 2
        assert agg["counts"]["table"] == 1
        assert agg["counts"]["formula"] == 1

    def test_overall_excludes_reading_order(self):
        results = [
            ("p1", {"text": 0.6, "table": 0.4, "formula": 0.2, "reading_order": 1.0}),
        ]
        agg = compute_aggregate(results)
        # Overall = avg(text, table, formula), not reading_order
        expected = (0.6 + 0.4 + 0.2) / 3
        assert agg["overall"] == pytest.approx(expected, abs=0.001)

    def test_all_none(self):
        results = [
            ("p1", {"text": None, "table": None, "formula": None, "reading_order": None}),
        ]
        agg = compute_aggregate(results)
        assert agg["overall"] == 0.0
