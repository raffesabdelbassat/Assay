"""Command-line interface for running evaluations."""

import asyncio
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
        ))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Comparison {comparison_id}")
    typer.echo(format_comparison_report(results))


if __name__ == "__main__":
    app()
