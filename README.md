# Assay

Assay is a Python 3.11+ command-line tool for evaluating and comparing model answers against a JSONL dataset. It supports Anthropic, OpenAI, Google Gen AI, and Ollama providers, named prompt versions, per-provider limits, model-specific prices, async retries, caching, and SQLite history.

## Install

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,openai,google,ollama]"
```

Provider SDKs are optional extras. Set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_API_KEY` in the environment for the providers you use. Ollama needs a running local server and a model pulled locally. Tests use fake providers and never make API calls.

## Run

```bash
assay run data/sample.jsonl --config config.yaml
assay run data/sample.jsonl --config config.yaml --no-cache
assay compare data/sample.jsonl --config comparison.example.yaml
```

The comparison config runs every configured model against every named prompt version. Its `provider_limits` section can cap concurrent requests and requests per minute for each provider. The separate prices file uses `provider:model` keys and USD per token rates. Every comparison target must have an explicit price entry; use zero rates only when the model has no metered provider cost. Review and refresh `prices.example.yaml` before relying on cost rankings. The included cloud rates were checked on 2026-11-21: [Anthropic](https://www.anthropic.com/claude-haiku-5-5), [OpenAI](https://developers.openai.com/api/docs/models/gpt-5.5), and [Google](https://ai.google.dev/gemini-api/docs/pricing).

The report shows each provider/model/prompt combination with accuracy, average latency, and total cost, then names the most accurate, fastest, and cheapest configuration.

## AI judge scorer

Add a `judge` section to a single-run config (see `judge.example.yaml`) to grade open-ended answers. Configure a separate provider/model, rubric, pass threshold, generation limits, and that judge model's per-token prices. The judge receives only the question, candidate answer, and rubric; it does not receive the dataset's `expected` answer. Its JSON `score` and `reason` are stored with the response. When enabled, summary accuracy is the fraction of judge scores at or above `pass_threshold`, and the CLI also prints the average numeric judge score. Judge tokens, latency, and cost are included in run totals.

Use `assay run data/sample.jsonl --config judge.example.yaml` after setting both `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`. This command makes real API calls; tests use fake providers instead.

Each dataset line needs an `id` and an `input`; `expected` is optional. `id` is saved as `example_key` and supports comparing the same example across runs. Built-in scorers are `exact_match`, `contains_text`, `regex_match`, and `valid_json`. Regex uses the configured `regex_pattern` when present, otherwise it treats `expected` as the pattern. Scorers requiring an expected value return a failed score when it is absent.

SQLite stores the complete configuration (including named prompts and scorer settings), comparison and run IDs, run-time prices, examples, outputs, token counts, estimated cost, latency, attempts, scorer results, and response cache. Cache hits have zero cost in the run where they are reused. The cache key includes provider, model, system prompt, rendered prompt, temperature, and max tokens.

The included comparison prices were checked on 2026-11-21: [Anthropic](https://www.anthropic.com/claude-haiku-5-5), [OpenAI GPT-5.5](https://developers.openai.com/api/docs/models/gpt-5.5), and [Google](https://ai.google.dev/gemini-api/docs/pricing). The judge example uses the [OpenAI pricing page](https://developers.openai.com/api/docs/pricing). Refresh rates before using cost rankings.

## Add a scorer

Subclass `assay.scorers.base.Scorer`, set a stable `name`, and implement `score(output, expected)` to return a `ScoreResult`. Add it to `build_scorers` or construct a scorer list in your own runner integration.

## Development

```bash
pytest
```
