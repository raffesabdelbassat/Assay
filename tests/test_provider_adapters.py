from types import SimpleNamespace

import pytest

from assay.providers.google import GoogleProvider
from assay.providers.ollama import OllamaProvider
from assay.providers.openai import OpenAIProvider


@pytest.mark.asyncio
async def test_openai_adapter_normalizes_response_and_usage() -> None:
    class Responses:
        async def create(self, **kwargs):
            assert kwargs["instructions"] == "system"
            return SimpleNamespace(output_text="answer", usage=SimpleNamespace(input_tokens=3, output_tokens=2))

    provider = OpenAIProvider(client=SimpleNamespace(responses=Responses()))
    result = await provider.generate(model="m", system_prompt="system", prompt="question", temperature=1.0, max_tokens=20)
    assert (result.output, result.input_tokens, result.output_tokens) == ("answer", 3, 2)


@pytest.mark.asyncio
async def test_google_adapter_normalizes_response_and_usage() -> None:
    class Models:
        async def generate_content(self, **kwargs):
            assert kwargs["contents"] == "question"
            return SimpleNamespace(text="answer", usage_metadata=SimpleNamespace(prompt_token_count=4, candidates_token_count=1))

    client = SimpleNamespace(aio=SimpleNamespace(models=Models()))
    provider = GoogleProvider(client=client, config_builder=lambda **kwargs: kwargs)
    result = await provider.generate(model="m", system_prompt="system", prompt="question", temperature=1.0, max_tokens=20)
    assert (result.output, result.input_tokens, result.output_tokens) == ("answer", 4, 1)


@pytest.mark.asyncio
async def test_ollama_adapter_normalizes_response_and_usage() -> None:
    class Client:
        async def chat(self, **kwargs):
            assert kwargs["messages"][0]["role"] == "system"
            return SimpleNamespace(message=SimpleNamespace(content="answer"), prompt_eval_count=5, eval_count=2)

    result = await OllamaProvider(client=Client()).generate(
        model="m", system_prompt="system", prompt="question", temperature=0.4, max_tokens=20,
    )
    assert (result.output, result.input_tokens, result.output_tokens) == ("answer", 5, 2)
