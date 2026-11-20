"""Async evaluation orchestration, retry handling, and response caching."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any

from assay.models import AssayConfig, DatasetExample, ProviderResponse
from assay.providers.base import Provider
from assay.scorers import build_scorers
from assay.storage import Storage


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cache_key(config: AssayConfig, prompt: str) -> str:
    """Hash all model-call inputs that can affect the response."""
    payload = {"provider": config.provider, "model": config.model, "system_prompt": config.system_prompt,
               "prompt": prompt, "temperature": config.generation.temperature,
               "max_tokens": config.generation.max_tokens}
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


async def run_evaluation(examples: list[DatasetExample], config: AssayConfig, provider: Provider,
                         *, no_cache: bool = False, storage: Storage | None = None) -> tuple[int, dict[str, Any]]:
    """Run dataset items with a concurrency cap and persist the results."""
    db = storage or Storage(config.database)
    run_id = db.start_run(config, _now())
    example_rows = [(example, db.add_example(run_id, example)) for example in examples]
    semaphore = asyncio.Semaphore(config.generation.concurrency)
    scorers = build_scorers(config.scorers)

    async def process(example: DatasetExample, example_db_id: int) -> None:
        prompt = config.prompt_template.format(input=example.input, expected=example.expected or "", id=example.id)
        key = cache_key(config, prompt)
        start = time.perf_counter()
        attempts = 0
        cached = None if no_cache else db.get_cached(key)
        error: str | None = None
        response: ProviderResponse | None = cached
        if cached is None:
            async with semaphore:
                for attempt in range(config.generation.retries + 1):
                    attempts = attempt + 1
                    try:
                        response = await asyncio.wait_for(provider.generate(
                            model=config.model, system_prompt=config.system_prompt, prompt=prompt,
                            temperature=config.generation.temperature, max_tokens=config.generation.max_tokens,
                        ), timeout=config.generation.timeout_seconds)
                        break
                    except Exception as exc:  # provider and timeout errors are retried uniformly
                        error = f"{type(exc).__name__}: {exc}"
                        if attempt < config.generation.retries:
                            await asyncio.sleep(config.generation.retry_backoff_seconds * (2 ** attempt))
                else:
                    response = None
        latency_ms = (time.perf_counter() - start) * 1000
        cost = 0.0
        if response is not None and cached is None:
            cost = (response.input_tokens * config.prices.input_per_token +
                    response.output_tokens * config.prices.output_per_token)
            if not no_cache:
                db.put_cached(key, provider=config.provider, model=config.model, system_prompt=config.system_prompt,
                              prompt=prompt, temperature=config.generation.temperature,
                              max_tokens=config.generation.max_tokens, response=response, created_at=_now())
        response_id = db.add_response(example_db_id=example_db_id, provider=config.provider, model=config.model,
                                      prompt=prompt, system_prompt=config.system_prompt, response=response, cost=cost,
                                      latency_ms=latency_ms, attempts=attempts, cache_hit=cached is not None, error=error)
        if response is not None:
            db.add_scores(response_id, [scorer.score(response.output, example.expected) for scorer in scorers])

    try:
        await asyncio.gather(*(process(example, db_id) for example, db_id in example_rows))
        db.finish_run(run_id, _now())
    except Exception as exc:
        db.finish_run(run_id, _now(), status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if storage is None:
            db.close()
    return run_id, db.summary(run_id) if storage is not None else _summary_after_close(config.database, run_id)


def _summary_after_close(path: str, run_id: int) -> dict[str, Any]:
    db = Storage(path)
    try:
        return db.summary(run_id)
    finally:
        db.close()
