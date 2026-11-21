"""Command-line interface for running evaluations."""

import asyncio
import json
from pathlib import Path

import typer

from assay.config import load_comparison_config, load_config, load_price_catalog
from assay.comparison import run_comparison
from assay.dataset import load_dataset
from assay.models import AssayConfig, ComparisonConfig
from assay.providers.base import Provider
from assay.providers.factory import create_provider as provider_factory
from assay.report import format_comparison_report
from assay.runner import run_evaluation

app = typer.Typer(help="Evaluate an LLM against a JSONL dataset.")
audit_app = typer.Typer(help="Blindly audit judge scores against human labels.")
app.add_typer(audit_app, name="audit")


def create_provider(settings: AssayConfig) -> Provider:
    """Create the configured provider; provider SDKs load only when selected."""
    return provider_factory(settings.provider, timeout_seconds=settings.generation.timeout_seconds)


def create_comparison_provider(provider: str, timeout_seconds: float) -> Provider:
    return provider_factory(provider, timeout_seconds=timeout_seconds)


@app.command()
def run(
    dataset: Path = typer.Argument(..., exists=True, readable=True, help="Input JSONL dataset"),
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True, help="YAML config file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Skip reading and writing the response cache"),
) -> None:
    """Run an evaluation and print its summary."""
    settings = load_config(config)
    examples = load_dataset(dataset)
    try:
        provider = create_provider(settings)
        judge_provider = None
        if settings.judge is not None:
            judge_provider = create_comparison_provider(
                settings.judge.provider,
                settings.judge.timeout_seconds,
            )
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    run_id, summary = asyncio.run(run_evaluation(
        examples, settings, provider, no_cache=no_cache, judge_provider=judge_provider,
    ))
    typer.echo(f"Run {run_id}")
    typer.echo("Metric              Value")
    typer.echo(f"Accuracy            {summary['accuracy'] if summary['accuracy'] is not None else 0:.1%}")
    if settings.judge is not None:
        judge_score = summary.get("average_judge_score")
        typer.echo(f"Judge average score {judge_score:.2f}" if judge_score is not None else "Judge average score n/a")
    typer.echo(f"Average latency     {summary['average_latency_ms'] or 0:.1f} ms")
    typer.echo(f"Total cost          ${summary['total_cost'] or 0:.6f}")


@app.command()
def compare(
    dataset: Path = typer.Argument(..., exists=True, readable=True, help="Input JSONL dataset"),
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True, help="Comparison YAML config"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Skip reading and writing the response cache"),
    pairwise: bool = typer.Option(False, "--pairwise", help="Compare candidate answers with the configured judge"),
) -> None:
    """Run every configured model/prompt pair and compare results."""
    settings = load_comparison_config(config)
    price_path = Path(settings.prices_file)
    if not price_path.is_absolute():
        price_path = config.parent / price_path
    prices = load_price_catalog(price_path)
    examples = load_dataset(dataset)
    try:
        comparison_id, results = asyncio.run(run_comparison(
            examples,
            settings,
            prices,
            create_comparison_provider,
            no_cache=no_cache,
            pairwise=pairwise,
        ))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Comparison {comparison_id}")
    typer.echo(format_comparison_report(results))


@audit_app.command("export")
def audit_export(
    run_id: int = typer.Option(..., "--run-id", min=1, help="Run id containing judge scores"),
    output: Path = typer.Option(..., "--output", "-o", help="JSONL file to fill in by hand"),
    database: Path = typer.Option(Path("assay.sqlite3"), "--database", "-d", help="Assay SQLite database"),
    limit: int = typer.Option(50, "--limit", min=30, max=50, help="Export 30 to 50 examples"),
    seed: int = typer.Option(0, "--seed", help="Seed used to reproduce the random sample"),
) -> None:
    """Export blind questions and answers for manual judge review."""
    from assay.storage import Storage

    storage = Storage(database)
    try:
        rows = storage.judge_audit_items(run_id, limit, seed=seed)
    finally:
        storage.close()
    if len(rows) < 30:
        raise typer.BadParameter(f"Run {run_id} has only {len(rows)} judge-scored answers; at least 30 are required")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps({
                "response_id": row["response_id"],
                "question": row["question"],
                "answer": row["answer"],
                "human_score": None,
            }, ensure_ascii=False) + "\n")
    typer.echo(
        f"Exported {len(rows)} blind answers to {output} (seed {seed}); "
        "fill human_score with a value from 0 to 1."
    )


@audit_app.command("agreement")
def audit_agreement(
    run_id: int = typer.Option(..., "--run-id", min=1, help="Run id containing judge scores"),
    labels: Path = typer.Option(..., "--labels", exists=True, readable=True, help="Completed audit JSONL file"),
    database: Path = typer.Option(Path("assay.sqlite3"), "--database", "-d", help="Assay SQLite database"),
) -> None:
    """Compare filled human scores with stored judge scores."""
    from assay.storage import Storage

    human_scores: dict[int, float] = {}
    try:
        with labels.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ValueError("each line must be a JSON object")
                    if item.get("human_score") is None:
                        continue
                    score = float(item["human_score"])
                    response_id = int(item["response_id"])
                    if not 0 <= score <= 1:
                        raise ValueError("human_score must be between 0 and 1")
                    if response_id in human_scores:
                        raise ValueError(f"duplicate response_id {response_id}")
                    human_scores[response_id] = score
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise typer.BadParameter(f"Invalid labels JSONL line {line_number}: {exc}") from exc
    except OSError as exc:
        raise typer.BadParameter(f"Could not read labels file: {exc}") from exc
    if not 30 <= len(human_scores) <= 50:
        raise typer.BadParameter(f"Provide 30 to 50 completed human_score labels; found {len(human_scores)}")
    storage = Storage(database)
    try:
        report = storage.judge_audit_agreement(run_id, human_scores)
    finally:
        storage.close()
    if report["count"] != len(human_scores):
        raise typer.BadParameter(f"Only {report['count']} labels match judge-scored responses in run {run_id}")
    typer.echo(f"Compared answers     {report['count']}")
    typer.echo(f"Pass/fail agreement  {report['agreement']:.1%}")
    typer.echo(f"Mean absolute error  {report['mean_absolute_error']:.3f} (0–1 scale)")


if __name__ == "__main__":
    app()
