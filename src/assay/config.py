"""Configuration file loading and validation."""

from pathlib import Path
from typing import Any

import yaml

from assay.models import AssayConfig


def load_config(path: str | Path) -> AssayConfig:
    """Load an Assay YAML config and validate it with Pydantic."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return AssayConfig.model_validate(raw)
