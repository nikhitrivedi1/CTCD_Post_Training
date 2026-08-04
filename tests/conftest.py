"""Shared fixtures for Core tests.

All fixtures are CPU-only, constructed in code, nothing stored on disk.
Naming: model_logits/target_logits for pure math, model/teacher for distillation.
"""

from unittest.mock import MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# Primitive dimensions
# ---------------------------------------------------------------------------


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def num_classes():
    return 3


@pytest.fixture
def vocab_size():
    return 128


@pytest.fixture
def hidden_size():
    return 64


@pytest.fixture
def seq_len():
    return 16


# ---------------------------------------------------------------------------
# Tensor fixtures — classification logits (batch, num_classes)
# ---------------------------------------------------------------------------


@pytest.fixture
def model_logits(batch_size, num_classes):
    torch.manual_seed(42)
    return torch.randn(batch_size, num_classes, requires_grad=True)


@pytest.fixture
def target_logits(batch_size, num_classes):
    torch.manual_seed(99)
    return torch.randn(batch_size, num_classes).detach()


@pytest.fixture
def ce_loss():
    return torch.tensor(1.5, requires_grad=True)


# ---------------------------------------------------------------------------
# Tensor fixtures — language model logits (batch, seq_len, vocab_size)
# ---------------------------------------------------------------------------


@pytest.fixture
def lm_model_logits(batch_size, seq_len, vocab_size):
    torch.manual_seed(42)
    return torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)


@pytest.fixture
def lm_target_logits(batch_size, seq_len, vocab_size):
    torch.manual_seed(99)
    return torch.randn(batch_size, seq_len, vocab_size).detach()


# ---------------------------------------------------------------------------
# Hidden state fixtures — (batch, seq_len, hidden_size) per layer
# ---------------------------------------------------------------------------


@pytest.fixture
def hidden_states_model(batch_size, seq_len, hidden_size):
    """7 layers: embedding + 6 transformer layers."""
    torch.manual_seed(10)
    return tuple(torch.randn(batch_size, seq_len, hidden_size) for _ in range(7))


@pytest.fixture
def hidden_states_target(batch_size, seq_len, hidden_size):
    """13 layers: embedding + 12 transformer layers."""
    torch.manual_seed(20)
    return tuple(torch.randn(batch_size, seq_len, hidden_size) for _ in range(13))


# ---------------------------------------------------------------------------
# Attention weight fixtures — (batch, heads, seq, seq), softmax-normalized
# ---------------------------------------------------------------------------


@pytest.fixture
def attention_weights_model(batch_size, seq_len):
    """6 layers x 6 heads."""
    torch.manual_seed(30)
    w = torch.rand(6, batch_size, 6, seq_len, seq_len)
    return tuple(w[i] / w[i].sum(dim=-1, keepdim=True) for i in range(6))


@pytest.fixture
def attention_weights_target(batch_size, seq_len):
    """12 layers x 12 heads."""
    torch.manual_seed(40)
    w = torch.rand(12, batch_size, 12, seq_len, seq_len)
    return tuple(w[i] / w[i].sum(dim=-1, keepdim=True) for i in range(12))


# ---------------------------------------------------------------------------
# Mock model output fixtures (for loss strategy tests)
# ---------------------------------------------------------------------------


def _make_mock_outputs(logits, hidden_states=None, attentions=None, loss=None):
    m = MagicMock()
    m.logits = logits
    m.hidden_states = hidden_states
    m.attentions = attentions
    m.loss = loss
    return m


@pytest.fixture
def mock_model_outputs(model_logits, hidden_states_model, attention_weights_model, ce_loss):
    return _make_mock_outputs(model_logits, hidden_states_model, attention_weights_model, ce_loss)


@pytest.fixture
def mock_target_outputs(target_logits, hidden_states_target, attention_weights_target):
    return _make_mock_outputs(
        target_logits, hidden_states_target, attention_weights_target, torch.tensor(1.2)
    )


# ---------------------------------------------------------------------------
# Layer mapping fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def bert_distilbert_mapping():
    """Standard 12-teacher -> 6-model uniform mapping."""
    return [(1, 0), (3, 1), (5, 2), (7, 3), (9, 4), (11, 5)]


# ---------------------------------------------------------------------------
# Tiny BERT fixtures (for trainer/evaluation tests needing HF Trainer compat)
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_bert_model():
    """2-layer BertForSequenceClassification, random init, no download."""
    from transformers import BertConfig, BertForSequenceClassification

    cfg = BertConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        num_labels=3,
        vocab_size=1000,
        attn_implementation="eager",
    )
    return BertForSequenceClassification(cfg)


@pytest.fixture
def tiny_bert_teacher():
    """4-layer BertForSequenceClassification, random init, no download."""
    from transformers import BertConfig, BertForSequenceClassification

    cfg = BertConfig(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=128,
        num_labels=3,
        vocab_size=1000,
        attn_implementation="eager",
    )
    return BertForSequenceClassification(cfg)
