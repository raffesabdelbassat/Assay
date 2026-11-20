"""Provider interface for asynchronous text generation."""

from abc import ABC, abstractmethod

from assay.models import ProviderResponse


class Provider(ABC):
    """A model backend that returns normalized responses."""

    name: str

    @abstractmethod
    async def generate(
        self,
        *,
        model: str,
        system_prompt: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        """Generate one response."""
