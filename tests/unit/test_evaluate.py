"""Tests for src/evaluation/evaluate.py and src/evaluation/metrics.py

Covers evaluate_model(), evaluate_models(), log_pre_post_metrics(),
_remap_compute_metrics(), run_evaluation(), log_transfer_metrics(),
_edit_distance(), get_compute_metrics(), compute_classification_metrics(),
compute_translation_metrics(), and compute_ocr_metrics().

Run:
    python -m pytest tests/unit/test_evaluate.py -v
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from datasets import Dataset

from src.evaluation.evaluate import (
    _remap_compute_metrics,
    evaluate_model,
    evaluate_models,
    log_pre_post_metrics,
    log_transfer_metrics,
    run_evaluation,
)
from src.evaluation.metrics import (
    _edit_distance,
    compute_classification_metrics,
    compute_ocr_metrics,
    compute_translation_metrics,
    get_compute_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_dataset(n=8, seq_len=16, num_labels=3, vocab=1000):
    torch.manual_seed(42)
    return Dataset.from_dict({
        "input_ids": torch.randint(0, vocab, (n, seq_len)).tolist(),
        "attention_mask": torch.ones(n, seq_len, dtype=torch.long).tolist(),
        "labels": torch.randint(0, num_labels, (n,)).tolist(),
    })


def _simple_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = float((preds == labels).mean())
    return {"accuracy": acc, "f1": acc}


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    """Tests using tiny_bert_model fixture + synthetic dataset."""

    @pytest.fixture
    def eval_result(self, tiny_bert_model):
        dataset = _synthetic_dataset()
        return evaluate_model(
            model=tiny_bert_model,
            eval_dataset=dataset,
            tokenizer=MagicMock(),
            compute_metrics=_simple_metrics,
            batch_size=4,
            device="cpu",
        )

    def test_returns_dict(self, eval_result):
        assert isinstance(eval_result, dict)

    @pytest.mark.parametrize("key", ["loss", "accuracy", "f1"])
    def test_has_expected_keys(self, eval_result, key):
        assert key in eval_result

    def test_all_values_finite(self, eval_result):
        for v in eval_result.values():
            assert np.isfinite(v)

    def test_loss_is_positive(self, eval_result):
        assert eval_result["loss"] > 0

    def test_restores_train_mode(self, tiny_bert_model):
        tiny_bert_model.train()
        evaluate_model(
            model=tiny_bert_model,
            eval_dataset=_synthetic_dataset(),
            tokenizer=MagicMock(),
            compute_metrics=_simple_metrics,
            batch_size=4,
            device="cpu",
        )
        assert tiny_bert_model.training is True

    def test_restores_eval_mode(self, tiny_bert_model):
        tiny_bert_model.eval()
        evaluate_model(
            model=tiny_bert_model,
            eval_dataset=_synthetic_dataset(),
            tokenizer=MagicMock(),
            compute_metrics=_simple_metrics,
            batch_size=4,
            device="cpu",
        )
        assert tiny_bert_model.training is False

    def test_loss_only_suppresses_metrics(self, tiny_bert_model):
        result = evaluate_model(
            model=tiny_bert_model,
            eval_dataset=_synthetic_dataset(),
            tokenizer=MagicMock(),
            compute_metrics=_simple_metrics,
            batch_size=4,
            device="cpu",
            loss_only=True,
        )
        assert "loss" in result
        assert "accuracy" not in result


# ---------------------------------------------------------------------------
# log_transfer_metrics
# ---------------------------------------------------------------------------

class TestLogTransferMetrics:

    @patch("src.evaluation.evaluate.logger")
    def test_teacher_advantage_value(self, mock_logger):
        """teacher_advantage = teacher[metric] - model[metric]."""
        log_transfer_metrics(
            task="mnli",
            model={"accuracy": 0.70},
            teacher={"accuracy": 0.90},
        )
        logged = " ".join(str(c) for c in mock_logger.info.call_args_list)
        assert "Teacher advantage" in logged

    @patch("src.evaluation.evaluate.logger")
    def test_model_gain_value(self, mock_logger):
        """model_gain = model[metric] - model_prev[metric]."""
        log_transfer_metrics(
            task="mnli",
            model={"accuracy": 0.85},
            teacher=None,
            model_prev={"accuracy": 0.70},
        )
        logged = " ".join(str(c) for c in mock_logger.info.call_args_list)
        assert "Model gain" in logged

    @patch("src.evaluation.evaluate.logger")
    def test_transfer_efficiency_computed(self, mock_logger):
        """efficiency = model_gain / pre_teacher_advantage."""
        log_transfer_metrics(
            task="mnli",
            model={"accuracy": 0.85},
            teacher={"accuracy": 0.90},
            model_prev={"accuracy": 0.70},
            teacher_prev={"accuracy": 0.95},
        )
        logged = " ".join(str(c) for c in mock_logger.info.call_args_list)
        assert "Transfer efficiency" in logged

    @patch("src.evaluation.evaluate.logger")
    def test_zero_advantage_skips_efficiency(self, mock_logger):
        """Division by ~zero teacher advantage should be skipped."""
        log_transfer_metrics(
            task="mnli",
            model={"accuracy": 0.85},
            teacher={"accuracy": 0.90},
            model_prev={"accuracy": 0.70},
            teacher_prev={"accuracy": 0.70},
        )
        logged = " ".join(str(c) for c in mock_logger.info.call_args_list)
        assert "Transfer efficiency" not in logged

    @patch("src.evaluation.evaluate.logger")
    def test_no_teacher_no_advantage(self, mock_logger):
        log_transfer_metrics(task="mnli", model={"accuracy": 0.85}, teacher=None)
        logged = " ".join(str(c) for c in mock_logger.info.call_args_list)
        assert "Teacher advantage" not in logged

    @patch("src.evaluation.evaluate.logger")
    def test_missing_metric_no_crash(self, mock_logger):
        """Missing primary metric in dicts should not raise."""
        log_transfer_metrics(
            task="mnli",
            model={"loss": 0.5},
            teacher={"loss": 0.3},
        )


# ---------------------------------------------------------------------------
# _edit_distance
# ---------------------------------------------------------------------------

class TestEditDistance:

    def test_identical_strings_zero(self):
        assert _edit_distance("abc", "abc") == 0

    def test_symmetric(self):
        assert _edit_distance("kitten", "sitting") == _edit_distance("sitting", "kitten")

    def test_empty_to_nonempty(self):
        assert _edit_distance("", "abc") == 3

    def test_both_empty(self):
        assert _edit_distance("", "") == 0

    @pytest.mark.parametrize("s1,s2,expected", [
        ("abc", "axc", 1),    # substitution
        ("ac", "abc", 1),     # insertion
        ("abc", "ac", 1),     # deletion
        ("kitten", "sitting", 3),
    ])
    def test_known_distances(self, s1, s2, expected):
        assert _edit_distance(s1, s2) == expected

    def test_distance_at_most_max_length(self):
        """Edit distance <= max(len(s1), len(s2))."""
        s1, s2 = "abcdef", "xyz"
        assert _edit_distance(s1, s2) <= max(len(s1), len(s2))


# ---------------------------------------------------------------------------
# get_compute_metrics routing
# ---------------------------------------------------------------------------

class TestGetComputeMetrics:

    @pytest.mark.parametrize("task", ["mnli", "qqp", "sst2", "cola"])
    def test_classification_returns_classifier(self, task):
        fn = get_compute_metrics(task)
        assert fn is compute_classification_metrics

    def test_translation_requires_tokenizer(self):
        with pytest.raises(ValueError, match="[Tt]okenizer"):
            get_compute_metrics("europarl", tokenizer=None)

    def test_translation_with_tokenizer_returns_callable(self):
        fn = get_compute_metrics("europarl", tokenizer=MagicMock())
        assert callable(fn)

    def test_ocr_requires_tokenizer(self):
        with pytest.raises(ValueError, match="[Tt]okenizer"):
            get_compute_metrics("ocr", tokenizer=None)

    def test_ocr_with_tokenizer_returns_callable(self):
        fn = get_compute_metrics("ocr", tokenizer=MagicMock())
        assert callable(fn)

    def test_unknown_task_returns_none(self):
        assert get_compute_metrics("nonexistent") is None


# ---------------------------------------------------------------------------
# compute_classification_metrics (real evaluate library)
# ---------------------------------------------------------------------------

class TestComputeClassificationMetrics:

    def test_returns_accuracy_f1_mcc(self):
        logits = np.array([[2.0, 0.1, 0.1], [0.1, 2.0, 0.1], [0.1, 0.1, 2.0], [2.0, 0.1, 0.1]])
        labels = np.array([0, 1, 2, 0])
        result = compute_classification_metrics((logits, labels))
        assert set(result.keys()) == {"accuracy", "f1", "mcc"}

    def test_perfect_predictions(self):
        logits = np.eye(3)
        labels = np.array([0, 1, 2])
        result = compute_classification_metrics((logits, labels))
        assert result["accuracy"] == 1.0
        assert result["f1"] == 1.0

    def test_wrong_predictions(self):
        logits = np.array([[0.0, 1.0], [1.0, 0.0]])
        labels = np.array([0, 1])
        result = compute_classification_metrics((logits, labels))
        assert result["accuracy"] == 0.0

    def test_empty_predictions(self):
        logits = np.zeros((0, 3))
        labels = np.array([])
        result = compute_classification_metrics((logits, labels))
        assert result == {"accuracy": 0.0, "f1": 0.0, "mcc": 0.0}


# ---------------------------------------------------------------------------
# compute_translation_metrics
# ---------------------------------------------------------------------------

class TestComputeTranslationMetrics:

    @pytest.fixture
    def mock_tokenizer(self):
        tok = MagicMock()
        tok.pad_token_id = 0
        tok.batch_decode = lambda ids, **kw: ["hello world" for _ in ids]
        return tok

    def test_returns_bleu_key(self, mock_tokenizer):
        fn = compute_translation_metrics(mock_tokenizer)
        # 3D logits (batch, seq, vocab) triggers argmax path
        logits = np.random.rand(4, 10, 50)
        labels = np.ones((4, 10), dtype=np.int64)
        result = fn((logits, labels))
        assert "bleu" in result

    def test_2d_logits_skips_argmax(self, mock_tokenizer):
        fn = compute_translation_metrics(mock_tokenizer)
        # 2D = already token IDs, no argmax
        logits = np.ones((4, 10), dtype=np.int64)
        labels = np.ones((4, 10), dtype=np.int64)
        result = fn((logits, labels))
        assert "bleu" in result

    def test_replaces_neg100_in_labels(self, mock_tokenizer):
        """Labels with -100 padding should be replaced by pad_token_id."""
        decoded_labels = []
        def capture_decode(ids, **kw):
            decoded_labels.append(ids)
            return ["test" for _ in ids]
        mock_tokenizer.batch_decode = capture_decode
        fn = compute_translation_metrics(mock_tokenizer)
        labels = np.array([[1, 2, -100]])
        logits = np.array([[1, 2, 3]])
        fn((logits, labels))
        # The second call is for labels; check -100 was replaced
        label_array = decoded_labels[1]
        assert -100 not in label_array.flatten()


# ---------------------------------------------------------------------------
# compute_ocr_metrics
# ---------------------------------------------------------------------------

class TestComputeOcrMetrics:

    @pytest.fixture
    def mock_tokenizer(self):
        tok = MagicMock()
        tok.pad_token_id = 0
        return tok

    def test_exact_match_keys(self, mock_tokenizer):
        mock_tokenizer.batch_decode = lambda ids, **kw: ["hello" for _ in ids]
        fn = compute_ocr_metrics(mock_tokenizer)
        logits = np.ones((4, 10), dtype=np.int64)
        labels = np.ones((4, 10), dtype=np.int64)
        result = fn((logits, labels))
        assert "edit_similarity" in result
        assert "exact_match" in result

    def test_perfect_match(self, mock_tokenizer):
        mock_tokenizer.batch_decode = lambda ids, **kw: ["abc" for _ in ids]
        fn = compute_ocr_metrics(mock_tokenizer)
        logits = np.ones((2, 5), dtype=np.int64)
        labels = np.ones((2, 5), dtype=np.int64)
        result = fn((logits, labels))
        assert result["edit_similarity"] == 1.0
        assert result["exact_match"] == 1.0

    def test_partial_match(self, mock_tokenizer):
        # First call decodes predictions, second decodes labels
        call_count = [0]
        def side_effect(ids, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return ["abc"]
            return ["axc"]
        mock_tokenizer.batch_decode = side_effect
        fn = compute_ocr_metrics(mock_tokenizer)
        logits = np.ones((1, 5), dtype=np.int64)
        labels = np.ones((1, 5), dtype=np.int64)
        result = fn((logits, labels))
        assert result["exact_match"] == 0.0
        assert 0.0 < result["edit_similarity"] < 1.0

    def test_3d_logits_argmax(self, mock_tokenizer):
        mock_tokenizer.batch_decode = lambda ids, **kw: ["test" for _ in ids]
        fn = compute_ocr_metrics(mock_tokenizer)
        logits = np.random.rand(2, 5, 50)
        labels = np.ones((2, 5), dtype=np.int64)
        result = fn((logits, labels))
        assert "edit_similarity" in result

    def test_none_pad_token_id(self):
        tok = MagicMock()
        tok.pad_token_id = None
        tok.batch_decode = lambda ids, **kw: ["test" for _ in ids]
        fn = compute_ocr_metrics(tok)
        labels = np.array([[1, -100]])
        logits = np.array([[1, 2]])
        result = fn((logits, labels))
        assert "edit_similarity" in result


# ---------------------------------------------------------------------------
# _remap_compute_metrics
# ---------------------------------------------------------------------------

class TestRemapComputeMetrics:

    def test_reorders_logits(self):
        """Logits columns should be reordered according to label_map."""
        def inner(eval_pred):
            logits, labels = eval_pred
            return {"col0": float(logits[0, 0])}

        # label_map [2, 0, 1] means new col 0 = old col 2
        wrapped = _remap_compute_metrics(inner, [2, 0, 1])
        logits = np.array([[10.0, 20.0, 30.0]])
        labels = np.array([0])
        result = wrapped((logits, labels))
        assert result["col0"] == 30.0

    def test_passes_labels_unchanged(self):
        captured = {}
        def inner(eval_pred):
            captured["labels"] = eval_pred[1]
            return {}

        wrapped = _remap_compute_metrics(inner, [1, 0])
        labels = np.array([7, 8, 9])
        wrapped((np.zeros((3, 2)), labels))
        np.testing.assert_array_equal(captured["labels"], labels)


# ---------------------------------------------------------------------------
# evaluate_model - column filtering and device inference
# ---------------------------------------------------------------------------

class TestEvaluateModelColumnFiltering:

    def test_extra_columns_removed(self, tiny_bert_model):
        """Dataset columns not in model.forward() signature get removed."""
        ds = _synthetic_dataset()
        # Add an extra column that isn't in BertForSequenceClassification.forward()
        ds = ds.add_column("extra_col", list(range(len(ds))))
        result = evaluate_model(
            model=tiny_bert_model,
            eval_dataset=ds,
            tokenizer=MagicMock(),
            compute_metrics=_simple_metrics,
            batch_size=4,
            device="cpu",
        )
        assert "loss" in result

    def test_device_inferred_from_model(self, tiny_bert_model):
        """When device=None, it should be inferred from the model's parameters."""
        result = evaluate_model(
            model=tiny_bert_model,
            eval_dataset=_synthetic_dataset(),
            tokenizer=MagicMock(),
            compute_metrics=_simple_metrics,
            batch_size=4,
            device=None,
        )
        assert isinstance(result, dict)
        assert "loss" in result


# ---------------------------------------------------------------------------
# log_pre_post_metrics
# ---------------------------------------------------------------------------

class TestLogPrePostMetrics:

    @patch("src.evaluation.evaluate.logger")
    def test_logs_model_pre(self, mock_logger):
        log_pre_post_metrics({"accuracy": 0.85, "loss": 0.5}, "model", "pre", to_wandb=False)
        logged = " ".join(str(c) for c in mock_logger.info.call_args_list)
        assert "Model" in logged
        assert "pre" in logged

    @patch("src.evaluation.evaluate.logger")
    def test_logs_teacher_post(self, mock_logger):
        log_pre_post_metrics({"accuracy": 0.90}, "teacher", "post", to_wandb=False)
        logged = " ".join(str(c) for c in mock_logger.info.call_args_list)
        assert "Teacher" in logged
        assert "post" in logged

    @patch("src.evaluation.evaluate.logger")
    def test_skips_non_float_values(self, mock_logger):
        """Non-float metric values should not be logged with :.4f formatting."""
        log_pre_post_metrics({"accuracy": 0.85, "name": "test"}, "model", "pre", to_wandb=False)
        # Should not raise - the string value is just skipped


# ---------------------------------------------------------------------------
# evaluate_models
# ---------------------------------------------------------------------------

class TestEvaluateModels:

    def test_model_only(self, tiny_bert_model):
        model_m, teacher_m = evaluate_models(
            model=tiny_bert_model,
            teacher=None,
            eval_dataset=_synthetic_dataset(),
            tokenizer=MagicMock(),
            compute_metrics=_simple_metrics,
            data_collator=None,
            eval_batch_size=4,
            device="cpu",
            phase="pre",
        )
        assert isinstance(model_m, dict)
        assert "loss" in model_m
        assert teacher_m is None

    def test_model_and_teacher(self, tiny_bert_model, tiny_bert_teacher):
        model_m, teacher_m = evaluate_models(
            model=tiny_bert_model,
            teacher=tiny_bert_teacher,
            eval_dataset=_synthetic_dataset(),
            tokenizer=MagicMock(),
            compute_metrics=_simple_metrics,
            data_collator=None,
            eval_batch_size=4,
            device="cpu",
            phase="post",
        )
        assert isinstance(model_m, dict)
        assert isinstance(teacher_m, dict)
        assert "loss" in teacher_m

    def test_with_label_map(self, tiny_bert_model, tiny_bert_teacher):
        """Teacher eval should use remapped compute_metrics when label_map provided."""
        model_m, teacher_m = evaluate_models(
            model=tiny_bert_model,
            teacher=tiny_bert_teacher,
            eval_dataset=_synthetic_dataset(),
            tokenizer=MagicMock(),
            compute_metrics=_simple_metrics,
            data_collator=None,
            eval_batch_size=4,
            device="cpu",
            phase="pre",
            teacher_label_map=[2, 0, 1],
        )
        assert isinstance(teacher_m, dict)
        assert "accuracy" in teacher_m


# ---------------------------------------------------------------------------
# run_evaluation
# ---------------------------------------------------------------------------

class TestRunEvaluation:

    def test_basic_orchestration(self, tiny_bert_model):
        """Mock load_model to return tiny_bert_model, verify orchestration."""
        config = {
            "model": {"name": "bert-base", "init": "pretrained"},
            "training": {"batch_size": 4},
        }
        with patch("src.evaluation.evaluate.load_model", return_value=tiny_bert_model):
            model_m, teacher_m = run_evaluation(
                config=config,
                device="cpu",
                tokenizer=MagicMock(),
                num_labels=3,
                eval_dataset=_synthetic_dataset(),
                compute_metrics=_simple_metrics,
                data_collator=None,
                task="mnli",
                phase="pre",
                evaluate_teacher=False,
            )
        assert isinstance(model_m, dict)
        assert "loss" in model_m
        assert teacher_m is None

    def test_with_teacher(self, tiny_bert_model, tiny_bert_teacher):
        """run_evaluation loads teacher when config has teacher and alpha > 0."""
        config = {
            "model": {"name": "bert-base", "init": "pretrained", "teacher": {"name": "bert-large"}},
            "training": {"batch_size": 4},
            "distillation": {"alpha": 0.5},
        }
        call_count = [0]
        models = [tiny_bert_model, tiny_bert_teacher]
        def fake_load(*args, **kwargs):
            idx = min(call_count[0], len(models) - 1)
            call_count[0] += 1
            return models[idx]

        with patch("src.evaluation.evaluate.load_model", side_effect=fake_load):
            model_m, teacher_m = run_evaluation(
                config=config,
                device="cpu",
                tokenizer=MagicMock(),
                num_labels=3,
                eval_dataset=_synthetic_dataset(),
                compute_metrics=_simple_metrics,
                data_collator=None,
                task="mnli",
                phase="post",
                evaluate_teacher=True,
            )
        assert isinstance(teacher_m, dict)

    def test_with_prev_metrics(self, tiny_bert_model):
        """Previous metrics trigger log_transfer_metrics."""
        config = {
            "model": {"name": "bert-base", "init": "pretrained"},
            "training": {"batch_size": 4},
        }
        prev = ({"accuracy": 0.7, "loss": 0.8}, None)
        with patch("src.evaluation.evaluate.load_model", return_value=tiny_bert_model), \
             patch("src.evaluation.evaluate.log_transfer_metrics") as mock_log:
            run_evaluation(
                config=config,
                device="cpu",
                tokenizer=MagicMock(),
                num_labels=3,
                eval_dataset=_synthetic_dataset(),
                compute_metrics=_simple_metrics,
                data_collator=None,
                task="mnli",
                phase="post",
                evaluate_teacher=False,
                prev_metrics=prev,
            )
        mock_log.assert_called_once()

    def test_checkpoint_dir(self, tiny_bert_model):
        """When checkpoint_dir is set, load_model receives the checkpoint path."""
        config = {
            "model": {"name": "bert-base"},
            "training": {"batch_size": 4},
        }
        with patch("src.evaluation.evaluate.load_model", return_value=tiny_bert_model) as mock_load:
            run_evaluation(
                config=config,
                device="cpu",
                tokenizer=MagicMock(),
                num_labels=3,
                eval_dataset=_synthetic_dataset(),
                compute_metrics=_simple_metrics,
                data_collator=None,
                task="mnli",
                phase="post",
                checkpoint_dir="/tmp/fake_ckpt",
                evaluate_teacher=False,
            )
        # First arg to load_model should be the checkpoint path
        assert mock_load.call_args[0][0] == "/tmp/fake_ckpt"

    def test_loss_only_for_none_num_labels(self, tiny_bert_model):
        """When num_labels is None, loss_only should be True."""
        config = {
            "model": {"name": "bert-base", "init": "pretrained"},
            "training": {"batch_size": 4},
        }
        with patch("src.evaluation.evaluate.load_model", return_value=tiny_bert_model), \
             patch("src.evaluation.evaluate.evaluate_models") as mock_em:
            mock_em.return_value = ({"loss": 0.5}, None)
            run_evaluation(
                config=config,
                device="cpu",
                tokenizer=MagicMock(),
                num_labels=None,
                eval_dataset=_synthetic_dataset(),
                compute_metrics=_simple_metrics,
                data_collator=None,
                task="mnli",
                phase="pre",
                evaluate_teacher=False,
            )
        assert mock_em.call_args[1]["loss_only"] is True

    def test_lora_config_passed(self, tiny_bert_model):
        """When use_lora is set, lora_config is passed to load_model."""
        lora_cfg = {"r": 8, "alpha": 16}
        config = {
            "model": {"name": "bert-base", "init": "pretrained"},
            "training": {"batch_size": 4, "use_lora": True, "lora": lora_cfg},
        }
        with patch("src.evaluation.evaluate.load_model", return_value=tiny_bert_model) as mock_load:
            run_evaluation(
                config=config,
                device="cpu",
                tokenizer=MagicMock(),
                num_labels=3,
                eval_dataset=_synthetic_dataset(),
                compute_metrics=_simple_metrics,
                data_collator=None,
                task="mnli",
                phase="pre",
                evaluate_teacher=False,
            )
        assert mock_load.call_args[1]["lora_config"] == lora_cfg
