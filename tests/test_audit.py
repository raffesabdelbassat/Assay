import json

from typer.testing import CliRunner

from assay.cli import app
from assay.models import AssayConfig, DatasetExample, JudgeSettings, ProviderResponse, ScoreResult
from assay.storage import Storage


def test_human_audit_export_rows_are_blind_and_agreement_uses_run_threshold(tmp_path) -> None:
    database = tmp_path / "audit.sqlite3"
    storage = Storage(database)
    config = AssayConfig(judge=JudgeSettings(
        provider="openai", model="judge", rubric="Is it helpful?", pass_threshold=0.6,
    ))
    run_id = storage.start_run(config, "now")
    response_ids = []
    for index in range(30):
        example_id = storage.add_example(run_id, DatasetExample(id=f"q-{index}", input=f"Question {index}"))
        response_id = storage.add_response(
            example_db_id=example_id, provider="anthropic", model="candidate", prompt="prompt",
            system_prompt="", response=ProviderResponse(output=f"Answer {index}"), cost=0,
            latency_ms=0, attempts=1, cache_hit=False, error=None,
        )
        response_ids.append(response_id)
        judge_score = 0.8 if index < 20 else 0.3
        storage.add_scores(response_id, [ScoreResult(
            scorer="llm_judge", passed=judge_score >= 0.6, score=judge_score,
            details={"reason": "stored judge explanation"},
        )])

    blind_rows = storage.judge_audit_items(run_id, 40)
    assert len(blind_rows) == 30
    assert set(blind_rows[0]) == {"response_id", "question", "answer"}
    labels = {response_id: (0.8 if index < 18 else 0.3) for index, response_id in enumerate(response_ids)}
    agreement = storage.judge_audit_agreement(run_id, labels)
    assert agreement["count"] == 30
    assert agreement["agreement"] == 28 / 30
    assert agreement["mean_absolute_error"] == 2 * 0.5 / 30
    storage.close()


def test_audit_cli_exports_blind_labels_and_reports_agreement(tmp_path) -> None:
    database = tmp_path / "audit-cli.sqlite3"
    storage = Storage(database)
    config = AssayConfig(judge=JudgeSettings(provider="openai", model="judge", rubric="Quality?"))
    run_id = storage.start_run(config, "now")
    for index in range(30):
        example_id = storage.add_example(run_id, DatasetExample(id=f"q-{index}", input=f"Question {index}"))
        response_id = storage.add_response(
            example_db_id=example_id, provider="anthropic", model="candidate", prompt="prompt",
            system_prompt="", response=ProviderResponse(output=f"Answer {index}"), cost=0,
            latency_ms=0, attempts=1, cache_hit=False, error=None,
        )
        storage.add_scores(response_id, [ScoreResult(
            scorer="llm_judge", passed=True, score=0.9, details={"reason": "do not export"},
        )])
    storage.close()

    output = tmp_path / "blind.jsonl"
    runner = CliRunner()
    exported = runner.invoke(app, ["audit", "export", "--run-id", str(run_id), "--output", str(output),
                                   "--database", str(database), "--limit", "30"])
    assert exported.exit_code == 0, exported.output
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 30
    assert set(rows[0]) == {"response_id", "question", "answer", "human_score"}
    assert all(row["human_score"] is None for row in rows)
    assert "reason" not in output.read_text(encoding="utf-8")

    for row in rows:
        row["human_score"] = 0.9
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    analyzed = runner.invoke(app, ["audit", "agreement", "--run-id", str(run_id), "--labels", str(output),
                                   "--database", str(database)])
    assert analyzed.exit_code == 0, analyzed.output
    assert "Pass/fail agreement  100.0%" in analyzed.output
