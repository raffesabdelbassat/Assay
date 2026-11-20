import json

import pytest

from assay.dataset import load_dataset


def test_load_dataset_and_reject_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"id": "one", "input": "Q", "expected": "A"}) + "\n", encoding="utf-8")
    assert load_dataset(path)[0].id == "one"
    path.write_text('{"id":"one","input":"Q"}\n{"id":"one","input":"Q2"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_dataset(path)
