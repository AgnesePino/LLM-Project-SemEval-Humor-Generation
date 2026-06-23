from __future__ import annotations

from typing import Any

from humor_gen.reranker import PreferenceRewardScorer


class FakeJudgeRunner:
    def __init__(self, response: str):
        self.response = response
        self.overrides: dict[str, Any] | None = None

    def generate_text(
        self,
        prompt: str,
        generation_overrides: dict[str, Any] | None = None,
    ) -> str:
        self.overrides = generation_overrides
        return self.response


def test_llm_judge_selects_json_winner() -> None:
    runner = FakeJudgeRunner(
        """```json
        {
          "winner_index": 1,
          "reasoning": "The second candidate has a coherent and surprising reversal.",
          "scores": [
            {"index": 0, "overall": 2.5},
            {"index": 1, "overall": 4.5}
          ]
        }
        ```"""
    )
    scorer = PreferenceRewardScorer("llm_judge", llm_runner=runner)
    candidates = ["First valid joke.", "Second valid joke."]

    result = scorer.rank_with_details(
        {"input_type": "headline", "headline": "Robots learn office humor"},
        candidates,
        contexts=["Robots can classify laughter."],
    )

    assert result.winner == candidates[1]
    assert result.scores == [2.5, 4.5]
    assert runner.overrides == {"do_sample": False, "max_new_tokens": 384}


def test_llm_judge_uses_first_valid_candidate_on_invalid_json() -> None:
    runner = FakeJudgeRunner("not-json")
    scorer = PreferenceRewardScorer("llm_judge", llm_runner=runner)
    candidates = ["First valid joke.", "Second valid joke."]

    result = scorer.rank_with_details(
        {"input_type": "headline", "headline": "Robots learn office humor"},
        candidates,
    )

    assert result.winner == candidates[0]
    assert result.fallback_used is True


def test_deberta_mode_requires_a_reward_checkpoint() -> None:
    try:
        PreferenceRewardScorer("deberta_reward")
    except ValueError as exc:
        assert "reward_model_name" in str(exc)
    else:
        raise AssertionError("Expected a missing-checkpoint ValueError.")
