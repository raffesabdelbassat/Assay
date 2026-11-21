import pytest
from pydantic import ValidationError

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
