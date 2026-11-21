from pathlib import Path

import pytest
from pydantic import ValidationError

from assay.config import load_comparison_config, load_price_catalog
from assay.models import AssayConfig


def test_prompt_template_accepts_supported_fields() -> None:
    config = AssayConfig(prompt_template="{id}: {input} Expected: {expected}")
    assert config.prompt_template.startswith("{id}")


def test_prompt_template_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError, match="Unknown prompt template field"):
        AssayConfig(prompt_template="{question}")


def test_prompt_template_rejects_unclosed_brace() -> None:
    with pytest.raises(ValidationError, match="Invalid prompt template"):
        AssayConfig(prompt_template="{input")


def test_config_rejects_unknown_settings_instead_of_ignoring_them() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AssayConfig.model_validate({"generation": {"concurency": 2}})


def test_comparison_config_loads_named_prompts_models_and_limits(tmp_path) -> None:
    path = tmp_path / "comparison.yaml"
    path.write_text(
        "models:\n  - provider: ollama\n    model: llama3\n"
        "prompts:\n  v1:\n    prompt_template: '{input}'\n"
        "provider_limits:\n  ollama:\n    concurrency: 2\n    requests_per_minute: 30\n",
        encoding="utf-8",
    )
    config = load_comparison_config(path)
    assert config.models[0].provider == "ollama"
    assert config.provider_limits["ollama"].requests_per_minute == 30


def test_price_catalog_loads_rates_per_model(tmp_path) -> None:
    path = tmp_path / "prices.yaml"
    path.write_text(
        "models:\n  'openai:gpt-5.5':\n    input_per_token: 0.000001\n    output_per_token: 0.000002\n",
        encoding="utf-8",
    )
    catalog = load_price_catalog(path)
    assert catalog.models["openai:gpt-5.5"].output_per_token == 0.000002


def test_pairwise_example_config_loads_its_rubric_and_prices() -> None:
    project = Path(__file__).resolve().parents[1]
    config = load_comparison_config(project / "comparison-pairwise.example.yaml")
    prices = load_price_catalog(project / config.prices_file)
    assert config.pairwise is True
    assert config.judge is not None
    assert "Factual accuracy" in config.judge.rubric
    assert "openai:gpt-5.5" in prices.models
