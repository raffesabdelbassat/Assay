"""Text comparison report and per-metric winners."""

from __future__ import annotations

from typing import Any


def _winner(results: list[dict[str, Any]], metric: str, *, maximize: bool) -> dict[str, Any] | None:
    candidates = [row for row in results if row.get(metric) is not None]
    if not candidates:
        return None
    return (max if maximize else min)(candidates, key=lambda row: row[metric])


def format_comparison_report(results: list[dict[str, Any]]) -> str:
    """Render all model/prompt metrics and label the best row per metric."""
    lines = [
        "Provider     Model                    Prompt  Accuracy  Avg latency  Total cost",
        "------------ ------------------------ ------- --------- ------------ ----------",
    ]
    for row in results:
        accuracy = row.get("accuracy")
        lines.append(
            f"{row['provider']:<12} {row['model']:<24} {row['prompt_version']:<7} "
            f"{f'{accuracy:.1%}' if accuracy is not None else 'n/a':>9}  "
            f"{(row.get('average_latency_ms') or 0):>10.1f} ms  "
            f"${(row.get('total_cost') or 0):.6f}"
        )

    pairwise_rows = [row for row in results if "pairwise_wins" in row]
    if pairwise_rows:
        lines.extend([
            "",
            "Pairwise judge (wins / losses / ties; judge cost share)",
        ])
        for row in pairwise_rows:
            lines.append(
                f"{row['provider']}/{row['model']} prompt={row['prompt_version']}: "
                f"{row['pairwise_wins']} / {row['pairwise_losses']} / {row['pairwise_ties']}; "
                f"${row['pairwise_judge_cost']:.6f}"
            )

    winner_specs = [
        ("Best accuracy", "accuracy", True, lambda value: f"{value:.1%}"),
        ("Fastest", "average_latency_ms", False, lambda value: f"{value:.1f} ms"),
        ("Cheapest", "total_cost", False, lambda value: f"${value:.6f}"),
    ]
    lines.append("")
    for label, metric, maximize, formatter in winner_specs:
        row = _winner(results, metric, maximize=maximize)
        if row is None:
            lines.append(f"{label}: n/a")
        else:
            identity = f"{row['provider']}/{row['model']} prompt={row['prompt_version']}"
            lines.append(f"{label}: {identity} ({formatter(row[metric])})")
    return "\n".join(lines)
