"""Tests for reward functions and scoring helpers."""

import pytest
from src.training.rewards import (
    _score_ocr,
    _score_format,
    _score_table_cell_f1,
    _score_formula_cdm,
    _score_reading_order,
    _score_inline_formatting,
    _extract_text_blocks,
    _extract_markdown_tables,
    _extract_latex,
    _match_block_order,
    _lcs_length,
    _score_parallel,
    ocr_reward,
    format_reward,
    table_cell_f1_reward,
    formula_cdm_reward,
    reading_order_reward,
    inline_formatting_reward,
    ParallelRewardGroup,
)
from src.evaluation.formatting_metrics import (
    extract_formatting_spans as _extract_formatting_spans,
    fuzzy_set_match as _fuzzy_set_match,
)


# ---------------------------------------------------------------------------
# _score_ocr
# ---------------------------------------------------------------------------
class TestScoreOcr:
    def test_identical(self):
        assert _score_ocr("hello world", "hello world") == 1.0

    def test_completely_different(self):
        score = _score_ocr("aaaa", "zzzz")
        assert score < 0.1

    def test_partial_match(self):
        score = _score_ocr("hello world", "hello")
        assert 0.3 < score < 0.8

    def test_empty_reference_empty_completion(self):
        """Both empty should be a perfect match."""
        assert _score_ocr("", "") == 1.0

    def test_empty_reference_nonempty_completion(self):
        """Empty reference but model generated text should score 0."""
        assert _score_ocr("some output", "") == 0.0

    def test_empty_completion_nonempty_reference(self):
        """Model generated nothing but GT exists."""
        assert _score_ocr("", "expected text") == 0.0


# ---------------------------------------------------------------------------
# _extract_text_blocks
# ---------------------------------------------------------------------------
class TestExtractTextBlocks:
    def test_basic(self):
        text = "First paragraph here.\n\nSecond paragraph here."
        blocks = _extract_text_blocks(text)
        assert len(blocks) == 2

    def test_returns_full_blocks(self):
        """Blocks should not be truncated."""
        long_block = "A" * 200
        text = f"{long_block}\n\nAnother block here."
        blocks = _extract_text_blocks(text)
        assert blocks[0] == long_block

    def test_filters_short_blocks(self):
        text = "Short.\n\nThis is a longer block of text."
        blocks = _extract_text_blocks(text)
        assert len(blocks) == 1
        assert "longer" in blocks[0]

    def test_empty_text(self):
        assert _extract_text_blocks("") == []


# ---------------------------------------------------------------------------
# _match_block_order (uses 50-char prefixes internally)
# ---------------------------------------------------------------------------
class TestMatchBlockOrder:
    def test_same_order(self):
        blocks = ["First block of text.", "Second block of text.", "Third block of text."]
        order = _match_block_order(blocks, blocks)
        assert order == [0, 1, 2]

    def test_reversed_order(self):
        gt = ["First block of text.", "Second block of text.", "Third block of text."]
        pred = list(reversed(gt))
        order = _match_block_order(pred, gt)
        assert order == [2, 1, 0]

    def test_long_blocks_matched_by_prefix(self):
        """Blocks longer than 50 chars should still match via prefix."""
        gt = ["A" * 100 + " suffix1", "B" * 100 + " suffix2"]
        pred = ["A" * 100 + " different", "B" * 100 + " different"]
        order = _match_block_order(pred, gt)
        assert order == [0, 1]


# ---------------------------------------------------------------------------
# _score_format
# ---------------------------------------------------------------------------
class TestScoreFormat:
    def test_empty(self):
        assert _score_format("") == 0.0

    def test_full_markdown(self):
        md = "# Heading\n\n- list item\n\n| col1 | col2 |\n| - | - |\n| a | b |\n\nSome text that is reasonably long enough to pass the length check."
        assert _score_format(md) == 1.0

    def test_partial(self):
        md = "# Just a heading\n\nSome longer text that passes the length threshold easily."
        score = _score_format(md)
        assert 0.4 <= score <= 0.8

    def test_inline_formatting_bonus(self):
        """Bold/italic markers should contribute to the score."""
        md = "This has **bold text** and some longer content to pass threshold."
        score = _score_format(md)
        # non-empty (0.2) + length (0.2) + inline formatting (0.2) = 0.6
        assert score >= 0.6

    def test_max_still_capped(self):
        """Score should not exceed 1.0 even with all criteria met."""
        md = "# Heading\n\n- **bold item**\n\n| col1 | col2 |\n| - | - |\n| a | b |\n\nSome longer text here."
        assert _score_format(md) == 1.0


# ---------------------------------------------------------------------------
# _score_table_cell_f1
# ---------------------------------------------------------------------------
class TestScoreTableCellF1:
    def test_no_gt_tables(self):
        """No tables in GT should return 0.5 (neutral)."""
        assert _score_table_cell_f1("| a | b |", "plain text") == 0.5

    def test_identical_tables(self):
        table = "| a | b |\n| - | - |\n| 1 | 2 |"
        score = _score_table_cell_f1(table, table)
        assert score > 0.9

    def test_no_pred_tables(self):
        gt = "| a | b |\n| - | - |\n| 1 | 2 |"
        assert _score_table_cell_f1("no tables here", gt) == 0.0

    def test_empty_reference(self):
        assert _score_table_cell_f1("| a |", "") == 0.0


# ---------------------------------------------------------------------------
# _score_formula_cdm
# ---------------------------------------------------------------------------
class TestScoreFormulaCdm:
    def test_no_gt_formulas(self):
        assert _score_formula_cdm("$x^2$", "no formulas") == 0.5

    def test_identical_formulas(self):
        text = "The equation is $E = mc^2$ here."
        score = _score_formula_cdm(text, text)
        assert score > 0.9

    def test_no_pred_formulas(self):
        assert _score_formula_cdm("no formulas", "$x^2$") == 0.0

    def test_empty_reference(self):
        assert _score_formula_cdm("$x$", "") == 0.0


# ---------------------------------------------------------------------------
# _score_reading_order
# ---------------------------------------------------------------------------
class TestScoreReadingOrder:
    def test_correct_order(self):
        text = "First paragraph text.\n\nSecond paragraph text.\n\nThird paragraph text."
        score = _score_reading_order(text, text)
        assert score == 1.0

    def test_reversed_order(self):
        gt = "First paragraph text.\n\nSecond paragraph text.\n\nThird paragraph text."
        pred = "Third paragraph text.\n\nSecond paragraph text.\n\nFirst paragraph text."
        score = _score_reading_order(pred, gt)
        assert score < 0.5

    def test_no_gt_blocks(self):
        """GT with text but no blocks >10 chars returns 0.5."""
        assert _score_reading_order("some text here.", "short") == 0.5

    def test_empty_reference(self):
        """Empty reference string returns 0.0 (guard clause)."""
        assert _score_reading_order("some text here.", "") == 0.0


# ---------------------------------------------------------------------------
# _score_parallel
# ---------------------------------------------------------------------------
class TestScoreParallel:
    def test_with_references(self):
        results = _score_parallel(_score_ocr, ["hello", "world"], ["hello", "earth"])
        assert len(results) == 2
        assert results[0] == 1.0
        assert results[1] < 1.0

    def test_without_references(self):
        results = _score_parallel(_score_format, ["# Heading\n\nLong text that passes the threshold."])
        assert len(results) == 1
        assert results[0] > 0.0

    def test_unwraps_conversational_format(self):
        """Completions in list-of-dict format should be unwrapped."""
        conv = [{"role": "assistant", "content": "hello world"}]
        results = _score_parallel(_score_ocr, [conv], ["hello world"])
        assert results[0] == 1.0


# ---------------------------------------------------------------------------
# ParallelRewardGroup
# ---------------------------------------------------------------------------
class TestParallelRewardGroup:
    def test_wraps_and_caches(self):
        calls = {"a": 0, "b": 0}

        def reward_a(completions, **kwargs):
            calls["a"] += 1
            return [1.0] * len(completions)

        def reward_b(completions, **kwargs):
            calls["b"] += 1
            return [0.5] * len(completions)

        group = ParallelRewardGroup([reward_a, reward_b])
        wrappers = group.wrap()
        assert len(wrappers) == 2

        completions = ["test"]
        result_a = wrappers[0](completions=completions)
        result_b = wrappers[1](completions=completions)

        assert result_a == [1.0]
        assert result_b == [0.5]
        # Both should have been called exactly once (concurrent on first call)
        assert calls["a"] == 1
        assert calls["b"] == 1

    def test_new_batch_reruns(self):
        call_count = {"n": 0}

        def reward_fn(completions, **kwargs):
            call_count["n"] += 1
            return [1.0] * len(completions)

        group = ParallelRewardGroup([reward_fn])
        wrappers = group.wrap()

        # Keep both lists alive so id() can't be reused
        batch1 = ["batch1"]
        batch2 = ["batch2"]

        wrappers[0](completions=batch1)
        assert call_count["n"] == 1

        wrappers[0](completions=batch2)
        assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# Public reward functions (end-to-end through _score_parallel)
# ---------------------------------------------------------------------------
class TestRewardFunctionsE2E:
    def test_ocr_reward(self):
        results = ocr_reward(completions=["hello"], ground_truth=["hello"])
        assert results == [1.0]

    def test_format_reward(self):
        results = format_reward(completions=["# Title\n\n- item\n\nLong enough text."])
        assert len(results) == 1
        assert results[0] > 0.0

    def test_table_cell_f1_reward(self):
        table = "| a | b |\n| - | - |\n| 1 | 2 |"
        results = table_cell_f1_reward(completions=[table], ground_truth=[table])
        assert results[0] > 0.9

    def test_formula_cdm_reward(self):
        text = "We have $x^2 + y^2 = z^2$ here."
        results = formula_cdm_reward(completions=[text], ground_truth=[text])
        assert results[0] > 0.9

    def test_reading_order_reward(self):
        text = "Block one is here.\n\nBlock two is here.\n\nBlock three is here."
        results = reading_order_reward(completions=[text], ground_truth=[text])
        assert results[0] == 1.0

    def test_inline_formatting_reward(self):
        text = "This has **bold** and *italic* formatting."
        results = inline_formatting_reward(completions=[text], ground_truth=[text])
        assert results[0] > 0.9


# ---------------------------------------------------------------------------
# _extract_formatting_spans
# ---------------------------------------------------------------------------
class TestExtractFormattingSpans:
    def test_bold(self):
        spans = _extract_formatting_spans("This is **bold** text.")
        assert "bold" in spans["bold"]

    def test_italic(self):
        spans = _extract_formatting_spans("This is *italic* text.")
        assert "italic" in spans["italic"]

    def test_strikethrough(self):
        spans = _extract_formatting_spans("This is ~~struck~~ text.")
        assert "struck" in spans["strikethrough"]

    def test_mixed(self):
        text = "**bold** and *italic* and ~~strike~~"
        spans = _extract_formatting_spans(text)
        assert len(spans["bold"]) == 1
        assert len(spans["italic"]) == 1
        assert len(spans["strikethrough"]) == 1

    def test_no_formatting(self):
        spans = _extract_formatting_spans("Plain text with no formatting.")
        assert not spans["bold"]
        assert not spans["italic"]
        assert not spans["strikethrough"]

    def test_empty(self):
        spans = _extract_formatting_spans("")
        assert not spans["bold"]


# ---------------------------------------------------------------------------
# fuzzy_set_match (from src.evaluation.formatting_metrics)
# ---------------------------------------------------------------------------
class TestFuzzySetMatch:
    def test_identical_sets(self):
        result = _fuzzy_set_match({"hello", "world"}, {"hello", "world"})
        assert result["f1"] == 1.0

    def test_no_overlap(self):
        result = _fuzzy_set_match({"aaa"}, {"zzz"})
        assert result["f1"] == 0.0

    def test_partial_overlap(self):
        result = _fuzzy_set_match({"hello", "extra"}, {"hello", "world"})
        assert 0.3 < result["f1"] < 0.8

    def test_empty_gt(self):
        assert _fuzzy_set_match({"hello"}, set())["f1"] == 0.0

    def test_empty_pred(self):
        assert _fuzzy_set_match(set(), {"hello"})["f1"] == 0.0

    def test_both_empty(self):
        assert _fuzzy_set_match(set(), set())["f1"] == 1.0

    def test_fuzzy_match(self):
        """Slightly different spans should still match."""
        result = _fuzzy_set_match({"Introduction"}, {"Introductions"})
        assert result["f1"] > 0.5


# ---------------------------------------------------------------------------
# _score_inline_formatting
# ---------------------------------------------------------------------------
class TestInlineFormattingScore:
    def test_identical(self):
        text = "This has **bold** and *italic* text."
        score = _score_inline_formatting(text, text)
        assert score > 0.9

    def test_no_gt_formatting(self):
        """GT without formatting returns 0.0 (no reward contribution)."""
        assert _score_inline_formatting("**bold**", "plain text") == 0.0

    def test_missing_pred_formatting(self):
        """Pred without formatting when GT has it returns 0.0."""
        assert _score_inline_formatting("plain text", "**bold** text") == 0.0

    def test_empty_reference(self):
        assert _score_inline_formatting("**bold**", "") == 0.0

    def test_partial_formatting(self):
        """Pred has some but not all formatting."""
        pred = "**bold** and plain"
        gt = "**bold** and *italic*"
        score = _score_inline_formatting(pred, gt)
        assert 0.3 < score < 0.8

    def test_strikethrough(self):
        text = "This was ~~removed~~ from the doc."
        score = _score_inline_formatting(text, text)
        assert score > 0.9
