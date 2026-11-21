import asyncio
import json

import pytest

from assay.models import AssayConfig, DatasetExample, ProviderResponse
from assay.runner import run_evaluation
from assay.storage import Storage


class FakeProvider:
    name = "fake"

    async def generate(self, *, model: str, system_prompt: str, prompt: str,
                       temperature: float, max_tokens: int) -> ProviderResponse:
        return ProviderResponse(output="42", input_tokens=12, output_tokens=3)


@pytest.mark.asyncio
async def test_runner_persists_results_and_uses_cache(tmp_path) -> None:
    config = AssayConfig(database=str(tmp_path / "assay.sqlite3"), prompt_template="{input}",
                         system_prompt="sys", prices={"input_per_token": 0.01, "output_per_token": 0.02})
    examples = [DatasetExample(id="answer", input="What is the answer?", expected="42")]
    storage = Storage(config.database)
    first_id, first = await run_evaluation(examples, config, FakeProvider(), storage=storage)
    second_id, second = await run_evaluation(examples, config, FakeProvider(), storage=storage)
    assert first_id != second_id
    assert first["accuracy"] == second["accuracy"] == 1
    assert first["total_cost"] > 0
    assert second["total_cost"] == 0
    response = storage.connection.execute("SELECT attempts,cache_hit FROM responses ORDER BY id DESC LIMIT 1").fetchone()
    assert response["attempts"] == 0
    assert response["cache_hit"] == 1
    config_json = storage.connection.execute("SELECT config_json FROM runs WHERE id=?", (first_id,)).fetchone()[0]
    assert json.loads(config_json)["system_prompt"] == "sys"
    storage.close()


@pytest.mark.asyncio
async def test_retry_wait_does_not_occupy_concurrency_slot(tmp_path) -> None:
    call_order: list[str] = []
    attempts = 0

    class RetryProvider:
        async def generate(self, *, model: str, system_prompt: str, prompt: str,
                           temperature: float, max_tokens: int) -> ProviderResponse:
            nonlocal attempts
            call_order.append(prompt)
            if prompt == "A" and attempts == 0:
                attempts += 1
                raise TimeoutError("temporary failure")
            return ProviderResponse(output="ok")

    config = AssayConfig(
        database=str(tmp_path / "retry.sqlite3"),
        prompt_template="{input}",
        generation={"concurrency": 1, "retries": 1, "retry_backoff_seconds": 0.05},
        scorers={"exact_match": False, "contains_text": False, "regex_match": False, "valid_json": False},
    )
    storage = Storage(config.database)
    await run_evaluation(
        [DatasetExample(id="a", input="A"), DatasetExample(id="b", input="B")],
        config,
        RetryProvider(),
        no_cache=True,
        storage=storage,
    )

    assert call_order == ["A", "B", "A"]
    row = storage.connection.execute(
        "SELECT attempts,error FROM responses r JOIN examples e ON e.id=r.example_id WHERE e.example_key='a'"
    ).fetchone()
    assert row["attempts"] == 2
    assert row["error"] is None
    storage.close()


@pytest.mark.asyncio
async def test_failed_model_call_counts_as_incorrect_in_accuracy(tmp_path) -> None:
    class FailingProvider:
        async def generate(self, *, model: str, system_prompt: str, prompt: str,
                           temperature: float, max_tokens: int) -> ProviderResponse:
            raise RuntimeError("provider unavailable")

    config = AssayConfig(
        database=str(tmp_path / "failed.sqlite3"),
        generation={"retries": 0},
    )
    storage = Storage(config.database)
    _, summary = await run_evaluation(
        [DatasetExample(id="failed", input="Question", expected="Answer")],
        config,
        FailingProvider(),
        no_cache=True,
        storage=storage,
    )
    assert summary["accuracy"] == 0
    status = storage.connection.execute("SELECT status FROM runs ORDER BY id DESC LIMIT 1").fetchone()["status"]
    assert status == "completed_with_errors"
    storage.close()


@pytest.mark.asyncio
async def test_runner_respects_configured_concurrency_limit(tmp_path) -> None:
    active_calls = 0
    max_active_calls = 0

    class SlowProvider:
        async def generate(self, *, model: str, system_prompt: str, prompt: str,
                           temperature: float, max_tokens: int) -> ProviderResponse:
            nonlocal active_calls, max_active_calls
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
            await asyncio.sleep(0.02)
            active_calls -= 1
            return ProviderResponse(output="done")

    config = AssayConfig(
        database=str(tmp_path / "concurrency.sqlite3"),
        generation={"concurrency": 2, "retries": 0},
        scorers={"exact_match": False, "contains_text": False, "regex_match": False, "valid_json": False},
    )
    storage = Storage(config.database)
    await run_evaluation(
        [DatasetExample(id=str(index), input=f"question {index}") for index in range(6)],
        config,
        SlowProvider(),
        no_cache=True,
        storage=storage,
    )
    assert max_active_calls == 2
    storage.close()


@pytest.mark.asyncio
async def test_no_cache_bypasses_reads_and_writes(tmp_path) -> None:
    calls = 0

    class CountingProvider:
        async def generate(self, *, model: str, system_prompt: str, prompt: str,
                           temperature: float, max_tokens: int) -> ProviderResponse:
            nonlocal calls
            calls += 1
            return ProviderResponse(output=f"answer-{calls}", input_tokens=1, output_tokens=1)

    config = AssayConfig(database=str(tmp_path / "no-cache.sqlite3"), prompt_template="{input}",
                         prices={"input_per_token": 0.01, "output_per_token": 0.01})
    example = DatasetExample(id="one", input="Question")
    provider = CountingProvider()
    storage = Storage(config.database)
    await run_evaluation([example], config, provider, storage=storage)
    _, uncached = await run_evaluation([example], config, provider, no_cache=True, storage=storage)
    _, cached_again = await run_evaluation([example], config, provider, storage=storage)

    assert calls == 2
    assert uncached["total_cost"] == 0.02
    assert cached_again["total_cost"] == 0
    assert storage.connection.execute("SELECT COUNT(*) FROM response_cache").fetchone()[0] == 1
    storage.close()


@pytest.mark.asyncio
async def test_runner_does_not_retry_permanent_api_errors(tmp_path) -> None:
    class BadRequestError(Exception):
        status_code = 400

    calls = 0

    class BadRequestProvider:
        async def generate(self, *, model: str, system_prompt: str, prompt: str,
                           temperature: float, max_tokens: int) -> ProviderResponse:
            nonlocal calls
            calls += 1
            raise BadRequestError("invalid model")

    config = AssayConfig(
        database=str(tmp_path / "bad-request.sqlite3"),
        generation={"retries": 3, "retry_backoff_seconds": 0.0},
    )
    storage = Storage(config.database)
    await run_evaluation(
        [DatasetExample(id="bad", input="Question")],
        config,
        BadRequestProvider(),
        no_cache=True,
        storage=storage,
    )
    attempts = storage.connection.execute("SELECT attempts FROM responses").fetchone()["attempts"]
    assert calls == attempts == 1
    storage.close()
