"""Execute model and prompt combinations with shared provider limits."""

from __future__ import annotations

import asyncio
from itertools import combinations
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
from assay.scorers.pairwise import PairwiseJudge
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
    pairwise: bool | None = None,
    storage: Storage | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Run every configured model against every named prompt version."""
    comparison_id = str(uuid4())
    pairwise_enabled = config.pairwise if pairwise is None else pairwise
    if pairwise_enabled and len(config.models) < 2:
        raise ValueError("Pairwise mode requires at least two candidate models")
    if pairwise_enabled and config.judge is None:
        raise ValueError("Pairwise mode requires judge settings")
    missing_prices = [
        f"{target.provider}:{target.model}"
        for target in config.models
        if f"{target.provider}:{target.model}" not in prices.models
    ]
    if config.judge is not None:
        judge_key = f"{config.judge.provider}:{config.judge.model}"
        if judge_key not in prices.models:
            missing_prices.append(judge_key)
        if config.judge.require_different_provider:
            same_provider = sorted({target.provider for target in config.models if target.provider == config.judge.provider})
            if same_provider:
                raise ValueError(f"Judge provider must differ from candidate providers: {', '.join(same_provider)}")
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
                }) if config.judge is not None and not pairwise_enabled else None),
                prompt_version=version,
                comparison_id=comparison_id,
                database=config.database,
            )
            tasks.append((target, version, run_evaluation(
                examples, run_config, providers[target.provider], no_cache=no_cache, storage=db,
                judge_provider=(providers[config.judge.provider]
                                if config.judge is not None and not pairwise_enabled else None),
            )))

    try:
        summaries = await asyncio.gather(*(task for _, _, task in tasks))
        results = [
            {"provider": target.provider, "model": target.model, "prompt_version": version,
             "run_id": run_id, **summary}
            for (target, version, _), (run_id, summary) in zip(tasks, summaries, strict=True)
        ]
        if pairwise_enabled and config.judge is not None:
            judge_settings = config.judge.model_copy(update={
                "prices": prices.models[f"{config.judge.provider}:{config.judge.model}"],
            })
            pairwise_judge = PairwiseJudge(judge_settings, providers[config.judge.provider])
            stored_outputs = db.comparison_outputs(comparison_id)
            output_map = {
                (item["example_key"], item["prompt_version"], item["provider"], item["model"]): item
                for item in stored_outputs
            }

            async def compare_pair(example_key: str, version: str, target_a, target_b) -> None:
                left = output_map.get((example_key, version, target_a.provider, target_a.model))
                right = output_map.get((example_key, version, target_b.provider, target_b.model))
                if left is None or right is None:
                    return
                pair_result = await pairwise_judge.compare(left["input"], left["output"], right["output"])
                db.add_pairwise_score(
                    comparison_id=comparison_id,
                    example_key=example_key,
                    prompt_version=version,
                    provider_a=target_a.provider,
                    model_a=target_a.model,
                    provider_b=target_b.provider,
                    model_b=target_b.model,
                    judge_provider=judge_settings.provider,
                    judge_model=judge_settings.model,
                    result=pair_result,
                )

            pair_jobs = [
                compare_pair(example_key, version, target_a, target_b)
                for version in config.prompts
                for target_a, target_b in combinations(config.models, 2)
                for example_key in {item["example_key"] for item in stored_outputs if item["prompt_version"] == version}
            ]
            await asyncio.gather(*pair_jobs)
            pair_scores = db.pairwise_scores(comparison_id)
            for row in results:
                wins = losses = ties = 0
                judge_cost = 0.0
                for pair in pair_scores:
                    if pair["prompt_version"] != row["prompt_version"]:
                        continue
                    identity = (row["provider"], row["model"])
                    if identity == (pair["provider_a"], pair["model_a"]):
                        outcome = pair["winner"]
                    elif identity == (pair["provider_b"], pair["model_b"]):
                        outcome = {"a": "b", "b": "a", "tie": "tie"}[pair["winner"]]
                    else:
                        continue
                    judge_cost += pair["cost"] / 2
                    wins += outcome == "a"
                    losses += outcome == "b"
                    ties += outcome == "tie"
                row["pairwise_wins"] = wins
                row["pairwise_losses"] = losses
                row["pairwise_ties"] = ties
                row["pairwise_judge_cost"] = judge_cost
        return comparison_id, results
    finally:
        if storage is None:
            db.close()
