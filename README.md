# Assay

Assay is a small Python 3.11+ command-line tool for evaluating model answers against a JSONL dataset. It supports an Anthropic provider, configurable prompts, pluggable scorers, async concurrency and retries, response caching, and SQLite run history.

## Install

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

Set `ANTHROPIC_API_KEY` in your environment, then copy `config.example.yaml` to `config.yaml` and set token prices if you want cost estimates. The default model is `claude-haiku-5-5` and is configurable.

## Run

```bash
assay run data/sample.jsonl --config config.yaml
assay run data/sample.jsonl --config config.yaml --no-cache
```

Each dataset line needs an `id` and an `input`; `expected` is optional. `id` is saved as `example_key` and supports comparing the same example across runs. Built-in scorers are `exact_match`, `contains_text`, `regex_match`, and `valid_json`. Regex uses the configured `regex_pattern` when present, otherwise it treats `expected` as the pattern. Scorers requiring an expected value return a failed score when it is absent.

SQLite stores the complete configuration (including prompts and scorer settings), run-time prices, examples, outputs, token counts, estimated cost, latency, attempts, scorer results, and response cache. Cache hits have zero cost in the run where they are reused. The cache key includes provider, model, system prompt, rendered prompt, temperature, and max tokens.

## Add a scorer

Subclass `assay.scorers.base.Scorer`, set a stable `name`, and implement `score(output, expected)` to return a `ScoreResult`. Add it to `build_scorers` or construct a scorer list in your own runner integration.

## Development

```bash
pytest
```
