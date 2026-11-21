"""Anthropic Messages API adapter."""

from typing import Any

from assay.models import ProviderResponse
from assay.providers.base import Provider


class AnthropicProvider(Provider):
    """Calls Anthropic using its official asynchronous Python client."""

    name = "anthropic"

    def __init__(self, *, timeout_seconds: float = 60.0, client: Any | None = None) -> None:
        if client is None:
            import anthropic

            client = anthropic.AsyncAnthropic(timeout=timeout_seconds)
        self.client = client

    async def generate(
        self,
        *,
        model: str,
        system_prompt: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        request: dict[str, Any] = {
            "model": model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if temperature != 1.0:
            request["extra_body"] = {"temperature": temperature}
        response = await self.client.messages.create(**request)
        output = "".join(block.text for block in response.content if block.type == "text")
        return ProviderResponse(
            output=output,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
