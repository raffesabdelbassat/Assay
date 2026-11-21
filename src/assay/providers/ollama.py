"""Ollama local model async adapter."""

from typing import Any

from assay.models import ProviderResponse
from assay.providers.base import Provider


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, *, timeout_seconds: float = 60.0, client: Any | None = None) -> None:
        if client is None:
            try:
                from ollama import AsyncClient
            except ImportError as exc:
                raise RuntimeError("Install Assay with `pip install -e '.[ollama]'` to use Ollama") from exc
            client = AsyncClient(timeout=timeout_seconds)
        self.client = client

    async def generate(self, *, model: str, system_prompt: str, prompt: str,
                       temperature: float, max_tokens: int) -> ProviderResponse:
        response = await self.client.chat(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        message = response.message
        return ProviderResponse(
            output=message.content or "",
            input_tokens=getattr(response, "prompt_eval_count", 0) or 0,
            output_tokens=getattr(response, "eval_count", 0) or 0,
        )
