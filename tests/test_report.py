from assay.report import format_comparison_report


def test_report_shows_side_by_side_metrics_and_each_winner() -> None:
    report = format_comparison_report([
        {"provider": "openai", "model": "gpt-a", "prompt_version": "v1", "accuracy": 0.9,
         "average_latency_ms": 500.0, "total_cost": 0.03},
        {"provider": "ollama", "model": "local-b", "prompt_version": "v2", "accuracy": 0.8,
         "average_latency_ms": 120.0, "total_cost": 0.0},
    ])
    assert "Best accuracy: openai/gpt-a prompt=v1" in report
    assert "Fastest: ollama/local-b prompt=v2" in report
    assert "Cheapest: ollama/local-b prompt=v2" in report
    assert "Avg latency" in report
