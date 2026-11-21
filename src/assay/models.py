"""Shared validated data models used by datasets, providers, and storage."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DatasetExample(BaseModel):
    """One JSONL evaluation item."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    input: str
    expected: str | None = None


class GenerationSettings(BaseModel):
    """Sampling and concurrency settings for a model call."""

    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, gt=0)
    concurrency: int = Field(default=5, gt=0)
    retries: int = Field(default=3, ge=0)
    retry_backoff_seconds: float = Field(default=0.5, ge=0.0)
    timeout_seconds: float = Field(default=60.0, gt=0.0)


class PriceSettings(BaseModel):
    """USD per token, captured into each run."""

    input_per_token: float = Field(default=0.0, ge=0.0)
    output_per_token: float = Field(default=0.0, ge=0.0)


class ScorerSettings(BaseModel):
    """Configuration for the built-in scorer plugin names."""

    exact_match: bool = True
    contains_text: bool = True
    regex_match: bool = True
    valid_json: bool = True
    case_sensitive: bool = True
    strip_whitespace: bool = True
    regex_pattern: str | None = None

    @field_validator("regex_pattern")
    @classmethod
    def validate_regex_pattern(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                re.compile(value)
            except re.error as exc:
                raise ValueError(f"Invalid scorer regex pattern: {exc}") from exc
        return value


class AssayConfig(BaseModel):
    """Complete run configuration loaded from YAML."""

    provider: str = "anthropic"
    model: str = "claude-haiku-5-5"
    system_prompt: str = "You are a helpful assistant."
    prompt_template: str = "{input}"
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    prices: PriceSettings = Field(default_factory=PriceSettings)
    scorers: ScorerSettings = Field(default_factory=ScorerSettings)
    database: str = "assay.sqlite3"


class ProviderResponse(BaseModel):
    """Normalized model response from a provider."""

    output: str
    input_tokens: int = 0
    output_tokens: int = 0


class ScoreResult(BaseModel):
    """A scorer's result; score is normalized to the 0..1 range."""

    scorer: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    details: dict[str, Any] = Field(default_factory=dict)
