"""Base class for adding scorer plugins."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable

from assay.models import ScoreResult


class Scorer(ABC):
    """A scorer evaluates a model output against an example."""

    name: str

    @abstractmethod
    def score(self, output: str, expected: str | None) -> ScoreResult | Awaitable[ScoreResult]:
        """Return a normalized score and useful details."""
