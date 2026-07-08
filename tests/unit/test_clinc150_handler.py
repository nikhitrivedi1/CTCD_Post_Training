"""Tests for src/data/handlers/clinc150.py

Covers CLINC150Handler prepare/validate/load behavior using a mocked
load_dataset - no network calls.

Run:
    python -m pytest tests/unit/test_clinc150_handler.py -v
"""

from unittest.mock import patch

from datasets import Dataset, DatasetDict

from src.data.handlers import DATASET_REGISTRY, get_handler
from src.data.handlers.clinc150 import CLINC150Handler


def _fake_dataset_dict(train_size=10, eval_size=5, num_oos=0):
    def _rows(n, num_oos):
        labels = [i % 150 for i in range(n)] + [None] * num_oos
        return {"utterance": [f"utterance {i}" for i in range(n + num_oos)], "label": labels}

    return DatasetDict(
        {
            "train": Dataset.from_dict(_rows(train_size, num_oos)),
            "validation": Dataset.from_dict(_rows(eval_size, num_oos)),
            "test": Dataset.from_dict(_rows(eval_size, num_oos)),
        }
    )


class TestCLINC150HandlerPrepare:
    def test_prepare_does_not_raise(self):
        CLINC150Handler().prepare("unused")


class TestCLINC150HandlerValidate:
    def test_validate_ok_when_accessible(self):
        with patch("src.data.handlers.clinc150.load_dataset") as mock_load:
            mock_load.return_value = iter([{"utterance": "book a flight", "label": 0}])
            result = CLINC150Handler().validate("unused")
        assert result["ok"] is True

    def test_validate_fails_gracefully_on_error(self):
        with patch("src.data.handlers.clinc150.load_dataset", side_effect=RuntimeError("network down")):
            result = CLINC150Handler().validate("unused")
        assert result["ok"] is False
        assert "network down" in result["details"]


class TestCLINC150HandlerLoad:
    def test_load_uses_train_and_validation_splits(self):
        with patch("src.data.handlers.clinc150.load_dataset", return_value=_fake_dataset_dict(10, 5)):
            train, eval_ = CLINC150Handler().load()
        assert len(train) == 10
        assert len(eval_) == 5

    def test_load_renames_label_to_labels(self):
        with patch("src.data.handlers.clinc150.load_dataset", return_value=_fake_dataset_dict(10, 5)):
            train, eval_ = CLINC150Handler().load()
        assert "labels" in train.column_names
        assert "label" not in train.column_names
        assert "labels" in eval_.column_names
        assert "label" not in eval_.column_names

    def test_load_max_train_samples_caps_dataset(self):
        with patch("src.data.handlers.clinc150.load_dataset", return_value=_fake_dataset_dict(10, 5)):
            train, _ = CLINC150Handler().load(max_train_samples=3)
        assert len(train) == 3

    def test_load_no_cap_when_under_limit(self):
        with patch("src.data.handlers.clinc150.load_dataset", return_value=_fake_dataset_dict(10, 5)):
            train, _ = CLINC150Handler().load(max_train_samples=1000)
        assert len(train) == 10

    def test_load_filters_out_of_scope_none_labels(self):
        with patch("src.data.handlers.clinc150.load_dataset", return_value=_fake_dataset_dict(10, 5, num_oos=3)):
            train, eval_ = CLINC150Handler().load()
        assert len(train) == 10
        assert len(eval_) == 5
        assert None not in train["labels"]
        assert None not in eval_["labels"]


class TestCLINC150DatasetRegistry:
    def test_task_in_registry(self):
        assert "clinc150" in DATASET_REGISTRY

    def test_registry_entry_is_clinc150_handler(self):
        assert isinstance(DATASET_REGISTRY["clinc150"], CLINC150Handler)

    def test_get_handler_returns_correct_instance(self):
        assert isinstance(get_handler("clinc150"), CLINC150Handler)
