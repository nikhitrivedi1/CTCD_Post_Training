"""Tests for src/data/collators.py

Covers VLMDataCollator: prompt masking, padding masking, label creation,
and output dict structure.

Run:
    python -m pytest tests/unit/test_collators.py -v
"""

from unittest.mock import MagicMock

import pytest
import torch

from src.data.collators import VLMDataCollator

# ===========================================================================
# Fixtures
# ===========================================================================

IM_START_ID = 151644  # Qwen's <|im_start|> token ID


@pytest.fixture
def mock_processor():
    """Mock processor that simulates Qwen VL chat format tokenization.

    Produces input_ids with <|im_start|> markers at known positions,
    simulating: <|im_start|>system\n...<|im_start|>user\n...<|im_start|>assistant\n<content>
    """
    processor = MagicMock()
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.convert_tokens_to_ids = lambda t: IM_START_ID if t == "<|im_start|>" else 0
    # "assistant\n" encodes to 3 tokens
    tokenizer.encode = MagicMock(return_value=[10, 20, 30])

    processor.tokenizer = tokenizer
    return processor


def _make_batch_inputs(sequences, pad_id=0):
    """Build a processor return dict from a list of token-id lists.

    Pads shorter sequences with pad_id.

    Args:
        sequences: List of lists of token IDs
        pad_id: Padding token ID

    Returns:
        Dict with input_ids and attention_mask tensors
    """
    max_len = max(len(s) for s in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    attention_mask = (input_ids != pad_id).long()
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# ===========================================================================
# Tests
# ===========================================================================


class TestVLMDataCollator:
    """VLMDataCollator prompt masking and label creation."""

    def test_prompt_tokens_masked(self, mock_processor):
        """Tokens before the last <|im_start|>assistant\\n are masked with -100."""
        # Sequence: [im_start, sys, tokens, im_start, user, tokens, im_start, asst_prefix(3), content, content]
        # positions:    0        1     2       3        4     5        6         7,8,9           10      11
        seq = [IM_START_ID, 1, 2, IM_START_ID, 3, 4, IM_START_ID, 10, 20, 30, 99, 100]
        inputs = _make_batch_inputs([seq])
        mock_processor.side_effect = lambda **kw: inputs
        mock_processor.return_value = inputs

        collator = VLMDataCollator(mock_processor)
        features = [{"image": MagicMock(), "formatted_text": "dummy"}]
        batch = collator(features)

        labels = batch["labels"][0]
        # Everything up to position 9 (inclusive) should be -100
        # content_start = positions[-1] + 1 + prefix_len = 6 + 1 + 3 = 10
        assert (labels[:10] == -100).all(), f"Prompt not masked: {labels[:10]}"

    def test_content_tokens_preserved(self, mock_processor):
        """Content tokens after assistant turn match input_ids."""
        seq = [IM_START_ID, 1, 2, IM_START_ID, 3, 4, IM_START_ID, 10, 20, 30, 99, 100]
        inputs = _make_batch_inputs([seq])
        mock_processor.side_effect = lambda **kw: inputs
        mock_processor.return_value = inputs

        collator = VLMDataCollator(mock_processor)
        features = [{"image": MagicMock(), "formatted_text": "dummy"}]
        batch = collator(features)

        labels = batch["labels"][0]
        # Content starts at position 10
        assert labels[10].item() == 99
        assert labels[11].item() == 100

    def test_padding_tokens_masked(self, mock_processor):
        """Padding tokens (pad_id=0) are masked with -100."""
        # Two sequences of different lengths to force padding
        seq1 = [IM_START_ID, 1, IM_START_ID, 10, 20, 30, 99, 100, 101]
        seq2 = [IM_START_ID, 1, IM_START_ID, 10, 20, 30, 99]
        inputs = _make_batch_inputs([seq1, seq2], pad_id=0)
        mock_processor.side_effect = lambda **kw: inputs
        mock_processor.return_value = inputs

        collator = VLMDataCollator(mock_processor)
        features = [
            {"image": MagicMock(), "formatted_text": "longer"},
            {"image": MagicMock(), "formatted_text": "short"},
        ]
        batch = collator(features)

        labels_2 = batch["labels"][1]
        # seq2 has pad tokens at positions 7 and 8
        assert labels_2[7].item() == -100
        assert labels_2[8].item() == -100

    def test_labels_shape_matches_input_ids(self, mock_processor):
        """Labels tensor has the same shape as input_ids."""
        seq = [IM_START_ID, 1, IM_START_ID, 10, 20, 30, 99]
        inputs = _make_batch_inputs([seq])
        mock_processor.side_effect = lambda **kw: inputs
        mock_processor.return_value = inputs

        collator = VLMDataCollator(mock_processor)
        features = [{"image": MagicMock(), "formatted_text": "dummy"}]
        batch = collator(features)

        assert batch["labels"].shape == batch["input_ids"].shape

    def test_returns_dict_with_expected_keys(self, mock_processor):
        """Output dict has input_ids, attention_mask, and labels."""
        seq = [IM_START_ID, 1, IM_START_ID, 10, 20, 30, 99]
        inputs = _make_batch_inputs([seq])
        mock_processor.side_effect = lambda **kw: inputs
        mock_processor.return_value = inputs

        collator = VLMDataCollator(mock_processor)
        features = [{"image": MagicMock(), "formatted_text": "dummy"}]
        batch = collator(features)

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch

    def test_single_im_start_no_masking(self, mock_processor):
        """With fewer than 2 <|im_start|> tokens, no prompt masking occurs."""
        seq = [IM_START_ID, 10, 20, 30, 99, 100]
        inputs = _make_batch_inputs([seq])
        mock_processor.side_effect = lambda **kw: inputs
        mock_processor.return_value = inputs

        collator = VLMDataCollator(mock_processor)
        features = [{"image": MagicMock(), "formatted_text": "dummy"}]
        batch = collator(features)

        labels = batch["labels"][0]
        # No prompt masking (only 1 im_start), so content tokens should be preserved
        # Only pad tokens (if any) get masked
        assert labels[0].item() == IM_START_ID  # not masked

    def test_batch_size_two(self, mock_processor):
        """Collator handles multi-sample batches correctly."""
        seq1 = [IM_START_ID, 1, IM_START_ID, 10, 20, 30, 50, 51]
        seq2 = [IM_START_ID, 2, IM_START_ID, 10, 20, 30, 60, 61]
        inputs = _make_batch_inputs([seq1, seq2])
        mock_processor.side_effect = lambda **kw: inputs
        mock_processor.return_value = inputs

        collator = VLMDataCollator(mock_processor)
        features = [
            {"image": MagicMock(), "formatted_text": "sample1"},
            {"image": MagicMock(), "formatted_text": "sample2"},
        ]
        batch = collator(features)

        assert batch["labels"].shape[0] == 2
        # Both should have content at position 6+
        # content_start = 2 + 1 + 3 = 6
        assert batch["labels"][0, 6].item() == 50
        assert batch["labels"][1, 6].item() == 60
