"""Tests for formatting bench scoring functions."""

import pytest
from src.evaluation.formatting_metrics import (
    extract_formatting_spans,
    fuzzy_set_match,
    score_formatting,
    compute_aggregate,
)


# ---------------------------------------------------------------------------
# extract_formatting_spans
# ---------------------------------------------------------------------------
class TestExtractFormattingSpans:
    def test_bold(self):
        spans = extract_formatting_spans("This is **bold** text.")
        assert spans["bold"] == {"bold"}

    def test_italic(self):
        spans = extract_formatting_spans("This is *italic* text.")
        assert spans["italic"] == {"italic"}

    def test_strikethrough(self):
        spans = extract_formatting_spans("This is ~~struck~~ text.")
        assert spans["strikethrough"] == {"struck"}

    def test_mixed(self):
        spans = extract_formatting_spans("**bold** and *italic* and ~~strike~~")
        assert spans["bold"] == {"bold"}
        assert spans["italic"] == {"italic"}
        assert spans["strikethrough"] == {"strike"}

    def test_no_formatting(self):
        spans = extract_formatting_spans("plain text")
        assert spans["bold"] == set()
        assert spans["italic"] == set()
        assert spans["strikethrough"] == set()

    def test_multiple_same_type(self):
        spans = extract_formatting_spans("**one** and **two** and **three**")
        assert spans["bold"] == {"one", "two", "three"}


# ---------------------------------------------------------------------------
# fuzzy_set_match
# ---------------------------------------------------------------------------
class TestFuzzySetMatch:
    def test_identical_sets(self):
        result = fuzzy_set_match({"hello", "world"}, {"hello", "world"})
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_no_overlap(self):
        result = fuzzy_set_match({"aaa"}, {"zzz"})
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_partial_overlap(self):
        # pred has "hello" (matches) and "extra" (no match)
        # gt has "hello" (matched) and "world" (not matched)
        result = fuzzy_set_match({"hello", "extra"}, {"hello", "world"})
        assert result["precision"] == pytest.approx(0.5, abs=0.05)
        assert result["recall"] == pytest.approx(0.5, abs=0.05)
        assert 0.3 < result["f1"] < 0.8

    def test_high_precision_low_recall(self):
        # pred: 1 span, gt: 3 spans. 1 match.
        result = fuzzy_set_match({"hello"}, {"hello", "world", "foo"})
        assert result["precision"] == 1.0
        assert result["recall"] == pytest.approx(1.0 / 3, abs=0.01)

    def test_low_precision_high_recall(self):
        # pred: 3 spans, gt: 1 span. 1 match.
        result = fuzzy_set_match({"hello", "aaa", "bbb"}, {"hello"})
        assert result["precision"] == pytest.approx(1.0 / 3, abs=0.01)
        assert result["recall"] == 1.0

    def test_empty_gt(self):
        result = fuzzy_set_match({"hello"}, set())
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_empty_pred(self):
        result = fuzzy_set_match(set(), {"hello"})
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_both_empty(self):
        result = fuzzy_set_match(set(), set())
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_fuzzy_match(self):
        """Slightly different spans should still match above threshold."""
        result = fuzzy_set_match({"Introduction"}, {"Introductions"})
        assert result["f1"] > 0.5
        assert result["precision"] > 0.5
        assert result["recall"] > 0.5


# ---------------------------------------------------------------------------
# score_formatting
# ---------------------------------------------------------------------------
class TestScoreFormatting:
    def test_exact_match(self):
        md = "**bold** and *italic* and ~~strike~~"
        result = score_formatting(md, md)
        assert result["bold"]["f1"] == 1.0
        assert result["bold"]["precision"] == 1.0
        assert result["bold"]["recall"] == 1.0
        assert result["italic"]["f1"] == 1.0
        assert result["overall"]["f1"] == 1.0

    def test_no_formatting_in_either(self):
        result = score_formatting("plain", "plain")
        assert result["bold"] is None
        assert result["italic"] is None
        assert result["strikethrough"] is None
        assert result["overall"] is None

    def test_no_formatting_in_gt(self):
        result = score_formatting("**bold**", "plain text")
        assert result["bold"] is None
        assert result["overall"] is None

    def test_no_formatting_in_pred(self):
        result = score_formatting("plain text", "**bold** text")
        assert result["bold"]["f1"] == 0.0
        assert result["bold"]["precision"] == 0.0
        assert result["bold"]["recall"] == 0.0

    def test_span_counts(self):
        result = score_formatting("**a** **b**", "**a** **b** **c**")
        assert result["gt_counts"]["bold"] == 3
        assert result["pred_counts"]["bold"] == 2

    def test_type_independence(self):
        """Bold score shouldn't affect italic score."""
        result = score_formatting("**bold**", "**bold** and *italic*")
        assert result["bold"]["f1"] == 1.0
        assert result["italic"]["f1"] == 0.0


# ---------------------------------------------------------------------------
# compute_aggregate
# ---------------------------------------------------------------------------
class TestComputeAggregate:
    def test_single_page(self):
        pages = [score_formatting("**a**", "**a**")]
        agg = compute_aggregate(pages)
        assert agg["scores"]["bold"]["f1"] == 1.0
        assert agg["scores"]["bold"]["precision"] == 1.0
        assert agg["scores"]["bold"]["recall"] == 1.0
        assert agg["counts"]["bold"] == 1

    def test_averages_across_pages(self):
        page1 = score_formatting("**a**", "**a**")  # bold f1=1.0
        page2 = score_formatting("plain", "**a**")  # bold f1=0.0
        agg = compute_aggregate([page1, page2])
        assert agg["scores"]["bold"]["f1"] == pytest.approx(0.5)
        assert agg["counts"]["bold"] == 2

    def test_skips_none_types(self):
        """Pages without a type in GT should not count toward that type."""
        page1 = score_formatting("**a**", "**a**")  # bold only in GT
        page2 = score_formatting("*b*", "*b*")  # italic only in GT
        agg = compute_aggregate([page1, page2])
        assert agg["counts"]["bold"] == 1
        assert agg["counts"]["italic"] == 1
        assert agg["counts"]["strikethrough"] == 0
        assert agg["scores"]["strikethrough"] is None

    def test_span_totals(self):
        page1 = score_formatting("**a** **b**", "**a** **b** **c**")
        page2 = score_formatting("**x**", "**x** **y**")
        agg = compute_aggregate([page1, page2])
        assert agg["gt_span_totals"]["bold"] == 5
        assert agg["pred_span_totals"]["bold"] == 3

    def test_backward_compat_float(self):
        """Old format with float scores should still aggregate."""
        old_page = {"bold": 0.8, "italic": None, "strikethrough": None,
                    "overall": 0.8, "gt_counts": {"bold": 1, "italic": 0, "strikethrough": 0},
                    "pred_counts": {"bold": 1, "italic": 0, "strikethrough": 0}}
        agg = compute_aggregate([old_page])
        assert agg["scores"]["bold"]["f1"] == pytest.approx(0.8)
        # precision/recall unknown from old format, should be 0
        assert agg["scores"]["bold"]["precision"] == 0.0
