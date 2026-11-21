"""Provider adapter factory."""

from assay.providers.base import Provider


def create_provider(name: str, *, timeout_seconds: float = 60.0) -> Provider:
    """Load only the optional SDK for the requested provider."""
    if name == "anthropic":
        from assay.providers.anthropic import AnthropicProvider

        return AnthropicProvider(timeout_seconds=timeout_seconds)
    if name == "openai":
        from assay.providers.openai import OpenAIProvider

        return OpenAIProvider(timeout_seconds=timeout_seconds)
    if name == "google":
        from assay.providers.google import GoogleProvider

        return GoogleProvider(timeout_seconds=timeout_seconds)
    if name == "ollama":
        from assay.providers.ollama import OllamaProvider

        return OllamaProvider(timeout_seconds=timeout_seconds)
    raise ValueError(f"Unsupported provider: {name}")
