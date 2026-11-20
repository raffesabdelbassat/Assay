from assay.models import ScorerSettings
from assay.scorers import build_scorers


def test_builtin_scorers() -> None:
    scorers = build_scorers(ScorerSettings(regex_pattern=r"\d+"))
    result = {s.name: s.score('{"n": 42}', "42").passed for s in scorers}
    assert result == {"exact_match": False, "contains_text": True, "regex_match": True, "valid_json": True}
