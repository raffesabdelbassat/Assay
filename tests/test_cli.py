from pathlib import Path

from typer.testing import CliRunner

from assay import cli
from assay.models import ProviderResponse


class FakeProvider:
    name = "fake"

    async def generate(self, *, model: str, system_prompt: str, prompt: str,
                       temperature: float, max_tokens: int) -> ProviderResponse:
        return ProviderResponse(output="Paris", input_tokens=5, output_tokens=2)


def test_cli_runs_full_evaluation_with_fake_provider(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text('{"id":"capital","input":"Capital of France?","expected":"Paris"}\n', encoding="utf-8")
    database = tmp_path / "runs.sqlite3"
    config = tmp_path / "config.yaml"
    config.write_text(
        "provider: anthropic\n"
        "model: claude-haiku-5-5\n"
        "prompt_template: '{input}'\n"
        f"database: '{database.as_posix()}'\n"
        "prices:\n  input_per_token: 0.0001\n  output_per_token: 0.0002\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "create_provider", lambda settings: FakeProvider())

    result = CliRunner().invoke(cli.app, ["run", str(dataset), "--config", str(config)])

    assert result.exit_code == 0, result.stdout
    assert "Accuracy            100.0%" in result.stdout
    assert "Total cost          $0.000900" in result.stdout
    assert database.exists()


def test_compare_cli_runs_three_fake_providers(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text('{"id":"capital","input":"Capital of France?","expected":"Paris"}\n', encoding="utf-8")
    prices = tmp_path / "prices.yaml"
    prices.write_text(
        "models:\n"
        "  'anthropic:a': {input_per_token: 0.001, output_per_token: 0.001}\n"
        "  'openai:b': {input_per_token: 0.002, output_per_token: 0.002}\n"
        "  'ollama:c': {input_per_token: 0, output_per_token: 0}\n",
        encoding="utf-8",
    )
    database = tmp_path / "compare.sqlite3"
    config = tmp_path / "compare.yaml"
    config.write_text(
        "models:\n"
        "  - {provider: anthropic, model: a}\n"
        "  - {provider: openai, model: b}\n"
        "  - {provider: ollama, model: c}\n"
        "prompts:\n  v1: {prompt_template: '{input}'}\n"
        "prices_file: prices.yaml\n"
        f"database: '{database.as_posix()}'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "create_comparison_provider", lambda name, timeout: FakeProvider())

    result = CliRunner().invoke(cli.app, ["compare", str(dataset), "--config", str(config)])

    assert result.exit_code == 0, result.stdout
    assert "anthropic" in result.stdout and "openai" in result.stdout and "ollama" in result.stdout
    assert "Best accuracy:" in result.stdout and "Fastest:" in result.stdout and "Cheapest:" in result.stdout
    assert database.exists()
