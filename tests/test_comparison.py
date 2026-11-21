import pytest
import time

from assay.comparison import run_comparison
from assay.models import ComparisonConfig, DatasetExample, PriceCatalog, ProviderResponse
from assay.storage import Storage


class FakeProvider:
    def __init__(self, name: str) -> None:
        self.name = name

    async def generate(self, *, model: str, system_prompt: str, prompt: str,
                       temperature: float, max_tokens: int) -> ProviderResponse:
        return ProviderResponse(output="Paris", input_tokens=10, output_tokens=2)


@pytest.mark.asyncio
async def test_comparison_runs_models_and_named_prompts_without_api(tmp_path) -> None:
    config = ComparisonConfig.model_validate({
        "models": [
            {"provider": "anthropic", "model": "haiku"},
            {"provider": "openai", "model": "gpt"},
            {"provider": "ollama", "model": "llama"},
        ],
        "prompts": {
            "v1": {"prompt_template": "{input}"},
            "v2": {"system_prompt": "Be brief", "prompt_template": "Answer: {input}"},
        },
        "database": str(tmp_path / "comparison.sqlite3"),
        "provider_limits": {"ollama": {"concurrency": 1, "requests_per_minute": 6000}},
    })
    prices = PriceCatalog.model_validate({
        "models": {
            "anthropic:haiku": {"input_per_token": 0.001, "output_per_token": 0.002},
            "openai:gpt": {"input_per_token": 0.002, "output_per_token": 0.003},
            "ollama:llama": {"input_per_token": 0, "output_per_token": 0},
        }
    })
    storage = Storage(config.database)
    comparison_id, results = await run_comparison(
        [DatasetExample(id="capital", input="Capital of France?", expected="Paris")],
        config,
        prices,
        lambda provider, timeout: FakeProvider(provider),
        storage=storage,
    )

    assert len(results) == 6
    assert {row["prompt_version"] for row in results} == {"v1", "v2"}
    assert {row["accuracy"] for row in results} == {1}
    rows = storage.connection.execute("SELECT COUNT(*) AS n FROM runs WHERE comparison_id=?", (comparison_id,)).fetchone()
    assert rows["n"] == 6
    assert next(row["total_cost"] for row in results if row["model"] == "haiku") == 0.014
    storage.close()


@pytest.mark.asyncio
async def test_comparison_requires_explicit_prices_for_every_model(tmp_path) -> None:
    config = ComparisonConfig.model_validate({
        "models": [{"provider": "openai", "model": "gpt"}],
        "prompts": {"v1": {"prompt_template": "{input}"}},
        "database": str(tmp_path / "missing-prices.sqlite3"),
    })
    with pytest.raises(ValueError, match="Missing price entries for: openai:gpt"):
        await run_comparison(
            [DatasetExample(id="one", input="question")],
            config,
            PriceCatalog(models={}),
            lambda provider, timeout: FakeProvider(provider),
        )


@pytest.mark.asyncio
async def test_provider_rate_limit_is_shared_across_models(tmp_path) -> None:
    request_times: list[float] = []

    class TimedProvider(FakeProvider):
        async def generate(self, *, model: str, system_prompt: str, prompt: str,
                           temperature: float, max_tokens: int) -> ProviderResponse:
            request_times.append(time.monotonic())
            return ProviderResponse(output="ok")

    config = ComparisonConfig.model_validate({
        "models": [
            {"provider": "openai", "model": "gpt-a"},
            {"provider": "openai", "model": "gpt-b"},
        ],
        "prompts": {"v1": {"prompt_template": "{input}"}},
        "provider_limits": {"openai": {"concurrency": 2, "requests_per_minute": 600}},
        "database": str(tmp_path / "rate-limit.sqlite3"),
    })
    prices = PriceCatalog(models={
        "openai:gpt-a": {"input_per_token": 0, "output_per_token": 0},
        "openai:gpt-b": {"input_per_token": 0, "output_per_token": 0},
    })

    await run_comparison(
        [DatasetExample(id="one", input="question")],
        config,
        prices,
        lambda provider, timeout: TimedProvider(provider),
    )

    assert len(request_times) == 2
    assert request_times[1] - request_times[0] >= 0.08
