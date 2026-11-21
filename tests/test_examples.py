from pathlib import Path

from assay.config import load_config
from assay.dataset import load_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_example_config_and_dataset_are_runnable() -> None:
    config = load_config(ROOT / "config.example.yaml")
    examples = load_dataset(ROOT / "data" / "sample.jsonl")

    assert config.provider == "anthropic"
    assert config.model == "claude-haiku-5-5"
    assert len(examples) == 10
    assert len({example.id for example in examples}) == 10
