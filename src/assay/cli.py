"""Command-line interface for running evaluations."""

import asyncio
from pathlib import Path

import typer

from assay.config import load_config
from assay.dataset import load_dataset
from assay.models import AssayConfig
from assay.providers.base import Provider
from assay.runner import run_evaluation

app = typer.Typer(help="Evaluate an LLM against a JSONL dataset.")


def create_provider(settings: AssayConfig) -> Provider:
    """Create the configured provider; provider SDKs load only when selected."""
    if settings.provider == "anthropic":
        from assay.providers.anthropic import AnthropicProvider

        return AnthropicProvider(timeout_seconds=settings.generation.timeout_seconds)
    raise ValueError(f"Unsupported provider: {settings.provider}")


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
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    run_id, summary = asyncio.run(run_evaluation(examples, settings, provider, no_cache=no_cache))
    typer.echo(f"Run {run_id}")
    typer.echo("Metric              Value")
    typer.echo(f"Accuracy            {summary['accuracy'] if summary['accuracy'] is not None else 0:.1%}")
    typer.echo(f"Average latency     {summary['average_latency_ms'] or 0:.1f} ms")
    typer.echo(f"Total cost          ${summary['total_cost'] or 0:.6f}")


if __name__ == "__main__":
    app()
