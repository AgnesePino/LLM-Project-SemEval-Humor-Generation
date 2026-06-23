from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

from humor_gen.reranker import RerankResult
from humor_gen.validate import validate_joke


class JokeGeneratorProtocol(Protocol):
    def generate_joke(
        self,
        item: dict[str, str],
        contexts: list[str] | None = None,
        generation_overrides: dict[str, Any] | None = None,
    ) -> str:
        ...


class RerankerProtocol(Protocol):
    scorer_type: str

    def rank_with_details(
        self,
        item: dict[str, str],
        candidates: list[str],
        contexts: list[str] | None = None,
    ) -> RerankResult:
        ...


@dataclass(frozen=True)
class CandidateEvaluation:
    joke: str
    valid: bool
    errors: list[str]
    score: float | None
    fallback: bool = False


@dataclass(frozen=True)
class SelectionResult:
    joke: str
    valid: bool
    errors: list[str]
    score: float | None
    candidates: list[CandidateEvaluation]
    fallback_used: bool
    reranker_metadata: dict[str, Any]


def generate_many_and_select(
    runner: JokeGeneratorProtocol,
    item: dict[str, str],
    contexts: list[str] | None,
    selection_cfg: dict[str, Any],
    max_words: int = 45,
    reranker: RerankerProtocol | None = None,
) -> SelectionResult:
    """Generate candidates, discard invalid ones, and select the best valid joke."""
    num_candidates = max(1, int(selection_cfg.get("num_candidates", 5)))
    evaluations = [
        _evaluate_candidate(
            runner.generate_joke(item, contexts=contexts),
            item,
            max_words=max_words,
        )
        for _ in range(num_candidates)
    ]

    valid_indexes = [index for index, candidate in enumerate(evaluations) if candidate.valid]
    if valid_indexes:
        valid_candidates = [evaluations[index].joke for index in valid_indexes]
        reranker_metadata: dict[str, Any]
        if len(valid_candidates) > 1 and reranker is not None:
            rerank_result = reranker.rank_with_details(item, valid_candidates, contexts=contexts)
            winner_evaluation_index = valid_indexes[rerank_result.winner_index]
            if rerank_result.scores is not None:
                for valid_index, reward in zip(valid_indexes, rerank_result.scores):
                    evaluations[valid_index] = replace(evaluations[valid_index], score=reward)
            reranker_metadata = rerank_result.to_metadata()
        else:
            winner_evaluation_index = valid_indexes[0]
            reranker_metadata = {
                "scorer_type": reranker.scorer_type if reranker is not None else "none",
                "reasoning": (
                    "Only one valid candidate; semantic reranking was skipped."
                    if len(valid_candidates) == 1
                    else "No semantic reranker was configured; selected the first valid candidate."
                ),
                "fallback_used": False,
            }
        winner = evaluations[winner_evaluation_index]
        return SelectionResult(
            joke=winner.joke,
            valid=True,
            errors=[],
            score=winner.score,
            candidates=evaluations,
            fallback_used=False,
            reranker_metadata=reranker_metadata,
        )

    fallback_mode = str(selection_cfg.get("fallback", "greedy")).casefold()
    if fallback_mode == "greedy":
        fallback_joke = runner.generate_joke(
            item,
            contexts=contexts,
            generation_overrides={"do_sample": False},
        )
        fallback = _evaluate_candidate(
            fallback_joke,
            item,
            max_words=max_words,
            fallback=True,
        )
        evaluations.append(fallback)
        return SelectionResult(
            joke=fallback.joke,
            valid=fallback.valid,
            errors=fallback.errors,
            score=fallback.score,
            candidates=evaluations,
            fallback_used=True,
            reranker_metadata={
                "scorer_type": reranker.scorer_type if reranker is not None else "none",
                "reasoning": "No sampled candidate passed validation; semantic reranking was skipped.",
                "fallback_used": True,
            },
        )

    # "first" is a zero-cost fallback when a sixth greedy generation is undesirable.
    first = evaluations[0]
    return SelectionResult(
        joke=first.joke,
        valid=False,
        errors=first.errors,
        score=None,
        candidates=evaluations,
        fallback_used=True,
        reranker_metadata={
            "scorer_type": reranker.scorer_type if reranker is not None else "none",
            "reasoning": "No candidate passed validation; selected the first sampled candidate.",
            "fallback_used": True,
        },
    )


def _evaluate_candidate(
    joke: str,
    item: dict[str, str],
    max_words: int,
    fallback: bool = False,
) -> CandidateEvaluation:
    valid, errors = validate_joke(joke, item, max_words=max_words)
    return CandidateEvaluation(joke, valid, errors, score=None, fallback=fallback)
