from types import SimpleNamespace

import pytest

from assay.providers.anthropic import AnthropicProvider


class FakeMessages:
    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    async def create(self, **kwargs: object) -> SimpleNamespace:
        self.request = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hello")],
            usage=SimpleNamespace(input_tokens=4, output_tokens=2),
        )


class FakeClient:
    def __init__(self) -> None:
        self.messages = FakeMessages()


@pytest.mark.asyncio
async def test_anthropic_provider_omits_default_temperature() -> None:
    client = FakeClient()
    provider = AnthropicProvider(client=client)
    result = await provider.generate(
        model="claude-haiku-5-5",
        system_prompt="system",
        prompt="question",
        temperature=1.0,
        max_tokens=40,
    )

    assert result.output == "hello"
    assert result.input_tokens == 4
    assert "temperature" not in client.messages.request
    assert "extra_body" not in client.messages.request


@pytest.mark.asyncio
async def test_anthropic_provider_sends_non_default_temperature_in_extra_body() -> None:
    client = FakeClient()
    provider = AnthropicProvider(client=client)
    await provider.generate(
        model="claude-haiku-4-5-20251001",
        system_prompt="system",
        prompt="question",
        temperature=0.3,
        max_tokens=40,
    )

    assert client.messages.request["extra_body"] == {"temperature": 0.3}
    assert "temperature" not in client.messages.request
