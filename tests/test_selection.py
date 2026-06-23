from __future__ import annotations

from typing import Any

from humor_gen.reranker import RerankResult
from humor_gen.selection import generate_many_and_select


class FakeRunner:
    def __init__(self, sampled: list[str], greedy: str = ""):
        self.sampled = iter(sampled)
        self.greedy = greedy
        self.overrides: list[dict[str, Any] | None] = []

    def generate_joke(
        self,
        item: dict[str, str],
        contexts: list[str] | None = None,
        generation_overrides: dict[str, Any] | None = None,
    ) -> str:
        self.overrides.append(generation_overrides)
        if generation_overrides == {"do_sample": False}:
            return self.greedy
        return next(self.sampled)


class FakeReranker:
    scorer_type = "llm_judge"

    def rank_with_details(
        self,
        item: dict[str, str],
        candidates: list[str],
        contexts: list[str] | None = None,
    ) -> RerankResult:
        return RerankResult(
            winner_index=1,
            winner=candidates[1],
            scorer_type=self.scorer_type,
            reasoning="The second valid candidate has the stronger punchline.",
            scores=[2.0, 4.5],
        )


def test_selects_highest_scoring_valid_candidate() -> None:
    item = {
        "id": "test",
        "input_type": "word_pair",
        "headline": "",
        "word1": "umbrella",
        "word2": "lasagna",
    }
    candidates = [
        "This candidate forgets both required words completely and is invalid.",
        "My umbrella met lasagna at dinner and they discussed the weather.",
        "Umbrella asked lasagna: is this dinner, or very edible rain insurance?",
    ]
    runner = FakeRunner(candidates)

    result = generate_many_and_select(
        runner,
        item,
        contexts=[],
        selection_cfg={"num_candidates": 3, "fallback": "greedy"},
        reranker=FakeReranker(),
    )

    assert result.valid is True
    assert result.joke == candidates[2]
    assert result.fallback_used is False
    assert len(result.candidates) == 3
    assert result.score == 4.5
    assert result.reranker_metadata["scorer_type"] == "llm_judge"


def test_uses_greedy_fallback_when_all_samples_are_invalid() -> None:
    item = {
        "id": "test",
        "input_type": "word_pair",
        "headline": "",
        "word1": "umbrella",
        "word2": "lasagna",
    }
    runner = FakeRunner(
        ["Invalid candidate.", "Still invalid."],
        greedy="Umbrella met lasagna: dinner finally came with weather protection.",
    )

    result = generate_many_and_select(
        runner,
        item,
        contexts=[],
        selection_cfg={"num_candidates": 2, "fallback": "greedy"},
    )

    assert result.fallback_used is True
    assert result.valid is True
    assert result.candidates[-1].fallback is True
    assert runner.overrides[-1] == {"do_sample": False}
