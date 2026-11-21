from pathlib import Path

from assay.config import load_comparison_config, load_config, load_price_catalog
from assay.dataset import load_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_example_config_and_dataset_are_runnable() -> None:
    config = load_config(ROOT / "config.example.yaml")
    examples = load_dataset(ROOT / "data" / "sample.jsonl")

    assert config.provider == "anthropic"
    assert config.model == "claude-haiku-5-5"
    assert len(examples) == 10
    assert len({example.id for example in examples}) == 10


def test_comparison_examples_cover_three_providers_and_prices() -> None:
    config = load_comparison_config(ROOT / "comparison.example.yaml")
    prices = load_price_catalog(ROOT / config.prices_file)
    target_keys = {f"{target.provider}:{target.model}" for target in config.models}
    assert {target.provider for target in config.models} == {"anthropic", "openai", "google"}
    assert set(config.prompts) == {"v1", "v2"}
    assert target_keys <= set(prices.models)
