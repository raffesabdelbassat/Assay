import json

import pytest

from assay.models import JudgeSettings, ProviderResponse
from assay.scorers.pairwise import PairwiseJudge


class AlwaysChoosePresentedA:
    name = "fake"

    async def generate(self, *, prompt: str, **kwargs) -> ProviderResponse:
        payload = json.loads(prompt.split("Evaluation data: ", 1)[1])
        assert payload["question"] == "Question"
        return ProviderResponse(output='{"winner":"A","reason":"A is clearer."}', input_tokens=20, output_tokens=5)


@pytest.mark.asyncio
async def test_pairwise_random_order_maps_votes_back_to_original_answers() -> None:
    judge = PairwiseJudge(
        JudgeSettings(provider="openai", model="judge", rubric="Prefer concise accuracy.",
                      randomize_pair_order=True, repeats=3, timeout_seconds=1),
        AlwaysChoosePresentedA(),
        randomizer=lambda: True,
    )

    result = await judge.compare("Question", "answer from A", "answer from B")

    assert result.winner == "b"
    assert "A is clearer." in result.reason
    assert result.input_tokens == 60
    assert result.output_tokens == 15
