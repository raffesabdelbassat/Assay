"""Anthropic Messages API adapter."""

import anthropic

from assay.models import ProviderResponse
from assay.providers.base import Provider


class AnthropicProvider(Provider):
    """Calls Anthropic using its official asynchronous Python client."""

    name = "anthropic"

    def __init__(self, *, timeout_seconds: float = 60.0) -> None:
        self.client = anthropic.AsyncAnthropic(timeout=timeout_seconds)

    async def generate(
        self,
        *,
        model: str,
        system_prompt: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        response = await self.client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        output = "".join(block.text for block in response.content if block.type == "text")
        return ProviderResponse(
            output=output,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
