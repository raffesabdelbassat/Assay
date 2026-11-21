"""Execute model and prompt combinations with shared provider limits."""

from __future__ import annotations

import asyncio
from typing import Any, Callable
from uuid import uuid4

from assay.models import (
    AssayConfig,
    ComparisonConfig,
    DatasetExample,
    PriceCatalog,
    ProviderLimit,
    ProviderResponse,
)
from assay.providers.base import Provider
from assay.runner import run_evaluation
from assay.storage import Storage


class LimitedProvider(Provider):
    """Apply a shared concurrency cap and minimum inter-request spacing."""

    def __init__(self, inner: Provider, limits: ProviderLimit) -> None:
        self.inner = inner
        self.name = inner.name
        self.semaphore = asyncio.Semaphore(limits.concurrency)
        self.interval = 60.0 / limits.requests_per_minute if limits.requests_per_minute else 0.0
        self._rate_lock = asyncio.Lock()
        self._last_request = 0.0

    async def generate(self, *, model: str, system_prompt: str, prompt: str,
                       temperature: float, max_tokens: int) -> ProviderResponse:
        loop = asyncio.get_running_loop()
        async with self._rate_lock:
            wait = self.interval - (loop.time() - self._last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = loop.time()
        async with self.semaphore:
            return await self.inner.generate(
                model=model,
                system_prompt=system_prompt,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )


ProviderFactory = Callable[[str, float], Provider]


async def run_comparison(
    examples: list[DatasetExample],
    config: ComparisonConfig,
    prices: PriceCatalog,
    provider_factory: ProviderFactory,
    *,
    no_cache: bool = False,
    storage: Storage | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Run every configured model against every named prompt version."""
    comparison_id = str(uuid4())
    missing_prices = [
        f"{target.provider}:{target.model}"
        for target in config.models
        if f"{target.provider}:{target.model}" not in prices.models
    ]
    if config.judge is not None:
        judge_key = f"{config.judge.provider}:{config.judge.model}"
        if judge_key not in prices.models:
            missing_prices.append(judge_key)
    if missing_prices:
        raise ValueError(f"Missing price entries for: {', '.join(missing_prices)}")
    provider_names = {target.provider for target in config.models}
    if config.judge is not None:
        provider_names.add(config.judge.provider)
    limits = {name: config.provider_limits.get(name, ProviderLimit()) for name in provider_names}
    providers: dict[str, LimitedProvider] = {}
    for provider_name, provider_limit in limits.items():
        raw_provider = provider_factory(provider_name, config.generation.timeout_seconds)
        providers[provider_name] = LimitedProvider(raw_provider, provider_limit)

    db = storage or Storage(config.database)

    tasks = []
    for target in config.models:
        for version, prompt in config.prompts.items():
            limit = limits[target.provider]
            run_config = AssayConfig(
                provider=target.provider,
                model=target.model,
                system_prompt=prompt.system_prompt,
                prompt_template=prompt.prompt_template,
                generation=config.generation.model_copy(update={"concurrency": limit.concurrency}),
                prices=prices.models[f"{target.provider}:{target.model}"],
                scorers=config.scorers,
                judge=(config.judge.model_copy(update={
                    "prices": prices.models[f"{config.judge.provider}:{config.judge.model}"],
                }) if config.judge is not None else None),
                prompt_version=version,
                comparison_id=comparison_id,
                database=config.database,
            )
            tasks.append((target, version, run_evaluation(
                examples, run_config, providers[target.provider], no_cache=no_cache, storage=db,
                judge_provider=providers[config.judge.provider] if config.judge is not None else None,
            )))

    try:
        summaries = await asyncio.gather(*(task for _, _, task in tasks))
        results = [
            {"provider": target.provider, "model": target.model, "prompt_version": version,
             "run_id": run_id, **summary}
            for (target, version, _), (run_id, summary) in zip(tasks, summaries, strict=True)
        ]
        return comparison_id, results
    finally:
        if storage is None:
            db.close()
