"""Shared validated data models used by datasets, providers, and storage."""

from __future__ import annotations

import re
import string
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DatasetExample(BaseModel):
    """One JSONL evaluation item."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    input: str
    expected: str | None = None


class GenerationSettings(BaseModel):
    """Sampling and concurrency settings for a model call."""

    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, gt=0)
    concurrency: int = Field(default=5, gt=0)
    retries: int = Field(default=3, ge=0)
    retry_backoff_seconds: float = Field(default=0.5, ge=0.0)
    timeout_seconds: float = Field(default=60.0, gt=0.0)


class PriceSettings(BaseModel):
    """USD per token, captured into each run."""

    model_config = ConfigDict(extra="forbid")

    input_per_token: float = Field(default=0.0, ge=0.0)
    output_per_token: float = Field(default=0.0, ge=0.0)


class JudgeSettings(BaseModel):
    """LLM-based scorer settings and its model-specific prices."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    rubric: str | None = Field(default=None, min_length=1)
    rubric_file: str | None = None
    when: Literal["always", "simple_failure", "missing_expected"] = "always"
    pass_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    temperature: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, gt=0)
    concurrency: int = Field(default=2, gt=0)
    retries: int = Field(default=2, ge=0)
    retry_backoff_seconds: float = Field(default=0.5, ge=0.0)
    timeout_seconds: float = Field(default=60.0, gt=0.0)
    prices: PriceSettings = Field(default_factory=PriceSettings)

    @model_validator(mode="after")
    def require_rubric(self) -> "JudgeSettings":
        if not self.rubric and not self.rubric_file:
            raise ValueError("Set either judge.rubric or judge.rubric_file")
        return self


class ScorerSettings(BaseModel):
    """Configuration for the built-in scorer plugin names."""

    model_config = ConfigDict(extra="forbid")

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


class ModelTarget(BaseModel):
    """One provider/model pair in a comparison run."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str


class PromptVersion(BaseModel):
    """A named system and user prompt pair."""

    model_config = ConfigDict(extra="forbid")

    system_prompt: str = "You are a helpful assistant."
    prompt_template: str = "{input}"

    @field_validator("prompt_template")
    @classmethod
    def validate_prompt_template(cls, value: str) -> str:
        return AssayConfig.validate_prompt_template(value)


class ProviderLimit(BaseModel):
    """Per-provider in-flight and request-rate limits."""

    model_config = ConfigDict(extra="forbid")

    concurrency: int = Field(default=5, gt=0)
    requests_per_minute: int | None = Field(default=None, gt=0)


class ComparisonConfig(BaseModel):
    """Models and named prompts to cross-evaluate on one dataset."""

    model_config = ConfigDict(extra="forbid")

    models: list[ModelTarget] = Field(min_length=1)
    prompts: dict[str, PromptVersion] = Field(min_length=1)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    provider_limits: dict[str, ProviderLimit] = Field(default_factory=dict)
    scorers: ScorerSettings = Field(default_factory=ScorerSettings)
    judge: JudgeSettings | None = None
    prices_file: str = "prices.yaml"
    database: str = "assay.sqlite3"


class PriceCatalog(BaseModel):
    """Per-model token prices keyed as ``provider:model``."""

    model_config = ConfigDict(extra="forbid")

    models: dict[str, PriceSettings]


class AssayConfig(BaseModel):
    """Complete run configuration loaded from YAML."""

    model_config = ConfigDict(extra="forbid")

    provider: str = "anthropic"
    model: str = "claude-haiku-5-5"
    system_prompt: str = "You are a helpful assistant."
    prompt_template: str = "{input}"
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    prices: PriceSettings = Field(default_factory=PriceSettings)
    scorers: ScorerSettings = Field(default_factory=ScorerSettings)
    judge: JudgeSettings | None = None
    prompt_version: str | None = None
    comparison_id: str | None = None
    database: str = "assay.sqlite3"

    @field_validator("prompt_template")
    @classmethod
    def validate_prompt_template(cls, value: str) -> str:
        allowed_fields = {"input", "expected", "id"}
        try:
            for _, field_name, _, _ in string.Formatter().parse(value):
                if field_name is not None and field_name not in allowed_fields:
                    raise ValueError(
                        f"Unknown prompt template field {field_name!r}; use {', '.join(sorted(allowed_fields))}"
                    )
        except ValueError as exc:
            raise ValueError(f"Invalid prompt template: {exc}") from exc
        return value


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
    provider: str | None = None
    model: str | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost: float = Field(default=0.0, ge=0.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
