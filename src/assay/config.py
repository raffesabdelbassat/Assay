"""Configuration file loading and validation."""

from pathlib import Path
from typing import Any

import yaml

from assay.models import AssayConfig, ComparisonConfig, PriceCatalog


def load_config(path: str | Path) -> AssayConfig:
    """Load an Assay YAML config and validate it with Pydantic."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return AssayConfig.model_validate(raw)


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return raw


def load_comparison_config(path: str | Path) -> ComparisonConfig:
    return ComparisonConfig.model_validate(_load_yaml_mapping(path))


def load_price_catalog(path: str | Path) -> PriceCatalog:
    return PriceCatalog.model_validate(_load_yaml_mapping(path))
