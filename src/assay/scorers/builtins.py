"""Built-in exact, substring, regex, and JSON validity scorers."""

import json
import re

from assay.models import ScoreResult, ScorerSettings
from assay.scorers.base import Scorer


class ExactMatchScorer(Scorer):
    name = "exact_match"

    def __init__(self, *, case_sensitive: bool, strip_whitespace: bool) -> None:
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    def _normalize(self, value: str) -> str:
        if self.strip_whitespace:
            value = value.strip()
        return value if self.case_sensitive else value.casefold()

    def score(self, output: str, expected: str | None) -> ScoreResult:
        passed = expected is not None and self._normalize(output) == self._normalize(expected)
        return ScoreResult(scorer=self.name, passed=passed, score=float(passed), details={"expected_available": expected is not None})


class ContainsTextScorer(Scorer):
    name = "contains_text"

    def __init__(self, *, case_sensitive: bool) -> None:
        self.case_sensitive = case_sensitive

    def score(self, output: str, expected: str | None) -> ScoreResult:
        if expected is None:
            passed = False
        elif self.case_sensitive:
            passed = expected in output
        else:
            passed = expected.casefold() in output.casefold()
        return ScoreResult(scorer=self.name, passed=passed, score=float(passed), details={"expected_available": expected is not None})


class RegexMatchScorer(Scorer):
    name = "regex_match"

    def __init__(self, *, pattern: str | None, case_sensitive: bool) -> None:
        self.pattern = pattern
        self.flags = 0 if case_sensitive else re.IGNORECASE

    def score(self, output: str, expected: str | None) -> ScoreResult:
        pattern = self.pattern if self.pattern is not None else expected
        passed = pattern is not None and re.search(pattern, output, self.flags) is not None
        return ScoreResult(scorer=self.name, passed=passed, score=float(passed), details={"pattern": pattern})


class ValidJsonScorer(Scorer):
    name = "valid_json"

    def score(self, output: str, expected: str | None) -> ScoreResult:
        try:
            json.loads(output)
            passed = True
        except (json.JSONDecodeError, TypeError):
            passed = False
        return ScoreResult(scorer=self.name, passed=passed, score=float(passed))


def build_scorers(settings: ScorerSettings) -> list[Scorer]:
    """Construct enabled scorers in a stable order."""
    scorers: list[Scorer] = []
    if settings.exact_match:
        scorers.append(ExactMatchScorer(case_sensitive=settings.case_sensitive, strip_whitespace=settings.strip_whitespace))
    if settings.contains_text:
        scorers.append(ContainsTextScorer(case_sensitive=settings.case_sensitive))
    if settings.regex_match:
        scorers.append(RegexMatchScorer(pattern=settings.regex_pattern, case_sensitive=settings.case_sensitive))
    if settings.valid_json:
        scorers.append(ValidJsonScorer())
    return scorers
