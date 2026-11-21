"""Google Gen AI async adapter."""

from typing import Any

from assay.models import ProviderResponse
from assay.providers.base import Provider


class GoogleProvider(Provider):
    name = "google"

    def __init__(self, *, timeout_seconds: float = 60.0, client: Any | None = None,
                 config_builder: Any | None = None) -> None:
        del timeout_seconds  # the SDK's async transport owns its timeout configuration
        if client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError("Install Assay with `pip install -e '.[google]'` to use Google") from exc
            client = genai.Client()
        self.client = client
        self.config_builder = config_builder

    async def generate(self, *, model: str, system_prompt: str, prompt: str,
                       temperature: float, max_tokens: int) -> ProviderResponse:
        config_builder = self.config_builder
        if config_builder is None:
            from google.genai import types

            config_builder = types.GenerateContentConfig

        response = await self.client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config_builder(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        usage = response.usage_metadata
        return ProviderResponse(
            output=response.text or "",
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        )
