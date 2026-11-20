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
