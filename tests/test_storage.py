from assay.models import AssayConfig, DatasetExample, ProviderResponse, ScoreResult
from assay.storage import Storage


def test_storage_records_run_scores_and_zero_cost_cache_hit(tmp_path) -> None:
    db = Storage(tmp_path / "test.sqlite3")
    config = AssayConfig()
    run_id = db.start_run(config, "now")
    example_id = db.add_example(run_id, DatasetExample(id="k", input="Q", expected="A"))
    response_id = db.add_response(example_db_id=example_id, provider="anthropic", model="m", prompt="Q",
        system_prompt="", response=ProviderResponse(output="A", input_tokens=10, output_tokens=2), cost=0,
        latency_ms=3.0, attempts=0, cache_hit=True, error=None)
    db.add_scores(response_id, [ScoreResult(scorer="exact_match", passed=True, score=1)])
    db.finish_run(run_id, "done")
    assert db.summary(run_id)["accuracy"] == 1
    db.close()
