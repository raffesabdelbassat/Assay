import json

import pytest

from assay.models import JudgeSettings, ProviderResponse
from assay.scorers.judge import JudgeScorer


class FakeJudgeProvider:
    name = "fake"

    def __init__(self, output: str = '{"score":0.9,"reason":"Concise and accurate."}') -> None:
        self.output = output
        self.last_call: dict[str, object] | None = None

    async def generate(self, **kwargs) -> ProviderResponse:
        self.last_call = kwargs
        return ProviderResponse(output=self.output, input_tokens=100, output_tokens=20)


@pytest.mark.asyncio
async def test_judge_sends_question_answer_rubric_and_returns_reason() -> None:
    provider = FakeJudgeProvider()
    settings = JudgeSettings(
        provider="openai",
        model="gpt-6.2-sol",
        rubric="A good answer is accurate and concise.",
        pass_threshold=0.8,
        prices={"input_per_token": 0.000001, "output_per_token": 0.000002},
    )
    scorer = JudgeScorer(settings, provider)

    result = await scorer.score_example("Summarize this.", "A concise summary.", "gold must not be sent")

    payload_text = provider.last_call["prompt"].split("Evaluation data: ", 1)[1]
    payload = json.loads(payload_text)
    assert payload == {
        "question": "Summarize this.",
        "answer": "A concise summary.",
        "rubric": "A good answer is accurate and concise.",
    }
    assert result.scorer == "llm_judge"
    assert result.passed is True
    assert result.details["reason"] == "Concise and accurate."
    assert result.cost == pytest.approx(0.00014)
    assert result.input_tokens == 100 and result.output_tokens == 20


@pytest.mark.asyncio
async def test_judge_rejects_malformed_verdict_json() -> None:
    scorer = JudgeScorer(
        JudgeSettings(provider="openai", model="judge", rubric="Accuracy matters.", retries=0),
        FakeJudgeProvider("score 1 because it looks good"),
    )
    with pytest.raises(ValueError, match="invalid JSON verdict"):
        await scorer.score_example("question", "answer", None)
