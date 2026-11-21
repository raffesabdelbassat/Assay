from assay.models import AssayConfig, GenerationSettings
from assay.runner import cache_key


def test_cache_key_is_stable_and_includes_all_response_inputs() -> None:
    base = AssayConfig(
        provider="anthropic",
        model="model-a",
        system_prompt="system-a",
        prompt_template="{input}",
        generation=GenerationSettings(temperature=0.1, max_tokens=128),
    )
    original = cache_key(base, "question")
    assert cache_key(base, "question") == original

    variants = [
        base.model_copy(update={"provider": "other"}),
        base.model_copy(update={"model": "model-b"}),
        base.model_copy(update={"system_prompt": "system-b"}),
        base.model_copy(update={"generation": GenerationSettings(temperature=0.2, max_tokens=128)}),
        base.model_copy(update={"generation": GenerationSettings(temperature=0.1, max_tokens=256)}),
    ]
    assert all(cache_key(variant, "question") != original for variant in variants)
    assert cache_key(base, "different question") != original
