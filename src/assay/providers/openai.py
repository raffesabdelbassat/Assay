"""OpenAI Responses API adapter."""

from typing import Any

from assay.models import ProviderResponse
from assay.providers.base import Provider


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, *, timeout_seconds: float = 60.0, client: Any | None = None) -> None:
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise RuntimeError("Install Assay with `pip install -e '.[openai]'` to use OpenAI") from exc
            client = AsyncOpenAI(timeout=timeout_seconds)
        self.client = client

    async def generate(self, *, model: str, system_prompt: str, prompt: str,
                       temperature: float, max_tokens: int) -> ProviderResponse:
        request: dict[str, Any] = {
            "model": model,
            "instructions": system_prompt,
            "input": prompt,
            "max_output_tokens": max_tokens,
        }
        if temperature != 1.0:
            request["temperature"] = temperature
        response = await self.client.responses.create(**request)
        usage = response.usage
        return ProviderResponse(
            output=response.output_text or "",
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )
