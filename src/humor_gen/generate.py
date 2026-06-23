from __future__ import annotations

import logging
from typing import Any, Protocol

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - convenience fallback for bare local mock runs
    def tqdm(iterable, **_: Any):
        return iterable

from humor_gen.data import build_prompt_input, load_dataset
from humor_gen.models import get_runner
from humor_gen.reranker import PreferenceRewardScorer
from humor_gen.selection import generate_many_and_select
from humor_gen.utils import output_input_text, require_gpu_for_real_run, require_hf_token, resolve_model_config
from humor_gen.validate import validate_joke

LOGGER = logging.getLogger(__name__)


class RetrieverProtocol(Protocol):
    def retrieve(self, query: str, k: int) -> list[str]:
        ...


def generate_dataset(
    model_key: str,
    input_path: str,
    generation_cfg: dict[str, Any],
    models_config_path: str,
    mock: bool,
    method: str = "base",
    retriever: RetrieverProtocol | None = None,
    k: int = 0,
    rag_apply_to: str = "all",
    limit: int | None = None,
    selection_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    items = load_dataset(input_path)
    if limit is not None:
        items = items[:limit]

    model_cfg = resolve_model_config(model_key, models_config_path)
    require_gpu_for_real_run(mock)
    require_hf_token(model_cfg, mock)

    runner = get_runner(model_cfg, generation_cfg, mock)

    gen_opts = generation_cfg.get("generation", generation_cfg)
    max_words = gen_opts.get("max_words", 45)

    selection_cfg = selection_cfg or {}
    selection_enabled = bool(selection_cfg.get("enabled", False))
    num_candidates = int(selection_cfg.get("num_candidates", 1))
    fallback_mode = selection_cfg.get("fallback", "greedy")

    reranker_cfg = selection_cfg.get("reranker", {})
    reranker_type = reranker_cfg.get("type", "llm_judge").casefold()
    judge_max_new_tokens = int(reranker_cfg.get("judge_max_new_tokens", 384))

    rows: list[dict[str, Any]] = []

    for item in tqdm(items, desc=f"Generating {model_key}/{method}"):
        contexts = _retrieve_contexts(retriever, item, k, rag_apply_to)

        if selection_enabled and num_candidates > 1:
            row = _generate_best_of_n_row(
                item=item,
                runner=runner,
                model_key=model_key,
                model_cfg=model_cfg,
                mock=mock,
                method=method,
                contexts=contexts,
                k=k,
                max_words=max_words,
                num_candidates=num_candidates,
                fallback_mode=fallback_mode,
                reranker_type=reranker_type,
                judge_max_new_tokens=judge_max_new_tokens,
            )
        else:
            joke = runner.generate_joke(item, contexts=contexts)
            valid, errors = validate_joke(joke, item, max_words=max_words)

            row = _build_output_row(
                item=item,
                model_key=model_key,
                model_cfg=model_cfg,
                mock=mock,
                method=method,
                contexts=contexts,
                k=k,
                joke=joke,
                valid=valid,
                errors=errors,
                extra_metadata={},
            )

        rows.append(row)

    return rows

def _generate_best_of_n_row(
    item: dict[str, Any],
    runner: Any,
    model_key: str,
    model_cfg: dict[str, Any],
    mock: bool,
    method: str,
    contexts: list[str],
    k: int,
    max_words: int,
    num_candidates: int,
    fallback_mode: str,
    reranker_type: str,
    judge_max_new_tokens: int,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []

    for candidate_index in range(num_candidates):
        candidate = _generate_candidate(
            runner=runner,
            item=item,
            contexts=contexts,
            max_words=max_words,
            candidate_index=candidate_index,
            fallback=False,
            greedy=False,
        )
        candidates.append(candidate)

    valid_candidates = [
        candidate for candidate in candidates
        if candidate.get("valid")
    ]

    fallback_used = False
    winner_index_among_valid: int | None = None
    judge_reasoning = ""
    comparisons: list[dict[str, Any]] = []

    if valid_candidates:
        if reranker_type == "llm_judge":
            winner_index_among_valid, judge_reasoning, comparisons = _select_with_llm_judge(
                runner=runner,
                item=item,
                valid_candidates=valid_candidates,
                judge_max_new_tokens=judge_max_new_tokens,
            )
            selected_candidate = valid_candidates[winner_index_among_valid]
        else:
            winner_index_among_valid = _select_with_heuristic(valid_candidates)
            selected_candidate = valid_candidates[winner_index_among_valid]
            judge_reasoning = (
                f"Reranker type '{reranker_type}' is not implemented yet; "
                "used heuristic fallback among valid candidates."
            )
    else:
        fallback_used = True

        if fallback_mode == "greedy":
            selected_candidate = _generate_candidate(
                runner=runner,
                item=item,
                contexts=contexts,
                max_words=max_words,
                candidate_index=num_candidates,
                fallback=True,
                greedy=True,
            )
            candidates.append(selected_candidate)
            judge_reasoning = "No valid sampled candidates were found; used greedy fallback."
        else:
            selected_candidate = max(candidates, key=_heuristic_candidate_score)
            judge_reasoning = (
                "No valid sampled candidates were found; selected the best invalid "
                "candidate according to the heuristic score."
            )

    return _build_output_row(
        item=item,
        model_key=model_key,
        model_cfg=model_cfg,
        mock=mock,
        method=method,
        contexts=contexts,
        k=k,
        joke=selected_candidate["generated_joke"],
        valid=selected_candidate["valid"],
        errors=selected_candidate["constraint_errors"],
        extra_metadata={
            "candidates": candidates,
            "fallback_used": fallback_used,
            "selection": {
                "enabled": True,
                "num_candidates": num_candidates,
                "fallback": fallback_mode,
            },
            "reranker": {
                "type": reranker_type,
                "winner_index_among_valid": winner_index_among_valid,
                "reasoning": judge_reasoning,
                "num_sampled_candidates": num_candidates,
                "num_valid_candidates": len(valid_candidates),
                "comparisons": comparisons,
            },
        },
    )


def _generate_candidate(
    runner: Any,
    item: dict[str, Any],
    contexts: list[str],
    max_words: int,
    candidate_index: int,
    fallback: bool,
    greedy: bool,
) -> dict[str, Any]:
    if greedy:
        joke = runner.generate_joke(
            item,
            contexts=contexts,
            generation_overrides={
                "do_sample": False,
                "temperature": None,
                "top_p": None,
            },
        )
    else:
        joke = runner.generate_joke(item, contexts=contexts)

    valid, errors = validate_joke(joke, item, max_words=max_words)

    return {
        "candidate_index": candidate_index,
        "generated_joke": joke,
        "valid": valid,
        "constraint_errors": errors,
        "fallback": fallback,
        "score": _heuristic_candidate_score(
            {
                "generated_joke": joke,
                "valid": valid,
                "constraint_errors": errors,
            }
        ),
    }


def _select_with_llm_judge(
    runner: Any,
    item: dict[str, Any],
    valid_candidates: list[dict[str, Any]],
    judge_max_new_tokens: int,
) -> tuple[int, str, list[dict[str, Any]]]:
    if len(valid_candidates) == 1:
        valid_candidates[0]["score"] = 1.0
        return 0, "Only one valid candidate was available; selected it without LLM judging.", []

    scores = [0.0 for _ in valid_candidates]
    comparisons: list[dict[str, Any]] = []

    best_index = 0

    for challenger_index in range(1, len(valid_candidates)):
        current_best_index = best_index

        joke_a = valid_candidates[current_best_index]["generated_joke"]
        joke_b = valid_candidates[challenger_index]["generated_joke"]

        try:
            judge_result = runner.judge(
                item,
                joke_a,
                joke_b,
                generation_overrides={
                    "do_sample": False,
                    "max_new_tokens": judge_max_new_tokens,
                },
            )

            winner = str(judge_result.get("winner", "")).strip().upper()
            reason = str(
                judge_result.get("reason")
                or judge_result.get("reasoning")
                or ""
            )
            
            if winner not in {"A", "B", "TIE"}:
                winner_index = judge_result.get("winner_index")
            
                if winner_index == 0:
                    winner = "A"
                elif winner_index == 1:
                    winner = "B"
                else:
                    winner = "TIE"

        except Exception as exc:
            winner = _heuristic_pairwise_winner(
                valid_candidates[current_best_index],
                valid_candidates[challenger_index],
            )
            reason = f"LLM judge failed; heuristic fallback used. Error: {exc}"

        if winner == "A":
            scores[current_best_index] += 1.0
            selected_after_comparison = current_best_index
        elif winner == "B":
            scores[challenger_index] += 1.0
            best_index = challenger_index
            selected_after_comparison = challenger_index
        else:
            scores[current_best_index] += 0.5
            scores[challenger_index] += 0.5
            selected_after_comparison = current_best_index

        comparisons.append(
            {
                "candidate_a_index_among_valid": current_best_index,
                "candidate_b_index_among_valid": challenger_index,
                "winner": winner,
                "reason": reason,
                "selected_after_comparison": selected_after_comparison,
            }
        )

    for index, candidate in enumerate(valid_candidates):
        candidate["score"] = scores[index]

    reasoning = (
        f"Selected candidate {best_index} among {len(valid_candidates)} valid candidates "
        f"using LLM-as-a-judge pairwise tournament. Final score: {scores[best_index]}."
    )

    return best_index, reasoning, comparisons


def _select_with_heuristic(valid_candidates: list[dict[str, Any]]) -> int:
    best_index = 0
    best_score = _heuristic_candidate_score(valid_candidates[0])

    for index, candidate in enumerate(valid_candidates[1:], start=1):
        score = _heuristic_candidate_score(candidate)
        if score > best_score:
            best_score = score
            best_index = index

    return best_index


def _heuristic_pairwise_winner(
    candidate_a: dict[str, Any],
    candidate_b: dict[str, Any],
) -> str:
    score_a = _heuristic_candidate_score(candidate_a)
    score_b = _heuristic_candidate_score(candidate_b)

    if abs(score_a - score_b) < 0.01:
        return "tie"

    return "A" if score_a > score_b else "B"


def _heuristic_candidate_score(candidate: dict[str, Any]) -> float:
    joke = candidate.get("generated_joke", "") or ""
    errors = candidate.get("constraint_errors", []) or []
    word_count = len(joke.split())

    score = 0.0

    if candidate.get("valid"):
        score += 100.0

    score -= 20.0 * len(errors)

    if 8 <= word_count <= 25:
        score += 10.0

    score -= abs(word_count - 18) * 0.2

    return score


def _build_output_row(
    item: dict[str, Any],
    model_key: str,
    model_cfg: dict[str, Any],
    mock: bool,
    method: str,
    contexts: list[str],
    k: int,
    joke: str,
    valid: bool,
    errors: list[str],
    extra_metadata: dict[str, Any],
) -> dict[str, Any]:
    metadata = {
        "model_id": model_cfg["hf_id"],
        "mock": mock,
        "prompt_input": build_prompt_input(item),
        "rag_contexts": contexts,
        "rag_k": k if contexts else 0,
        "full_prompt": item.get("metadata", {}).get("full_prompt", ""),
    }

    metadata.update(extra_metadata)

    return {
        "id": item.get("id") or item.get("ID"),
        "input_type": item["input_type"],
        "input": output_input_text(item),
        "model": model_key,
        "method": method,
        "generated_joke": joke,
        "valid": valid,
        "constraint_errors": errors,
        "metadata": metadata,
    }

def _retrieve_contexts(retriever: RetrieverProtocol | None, item: dict[str, str], k: int, apply_to: str) -> list[str]:
    if retriever is None or k <= 0:
        return []
        
    input_type = item["input_type"]
    
    if apply_to == "headline" and input_type != "headline":
        return []
    if apply_to in {"word_pair", "word-pair", "word_inclusion"} and input_type != "word_pair":
        return []
        
    if input_type == "headline":
        headline_text = item.get("headline", "")
        query = f"Background facts and context about: {headline_text}"
    else:
        query = f"Meaning, usage, and related concepts for: {item.get('word1', '')} and {item.get('word2', '')}"
        
    return retriever.retrieve(query, k)
