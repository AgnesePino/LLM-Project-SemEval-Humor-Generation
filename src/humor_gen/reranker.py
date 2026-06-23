from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from humor_gen.prompts import build_semantic_judge_prompt

LOGGER = logging.getLogger(__name__)


class TextGenerationProtocol(Protocol):
    def generate_text(
        self,
        prompt: str,
        generation_overrides: dict[str, Any] | None = None,
    ) -> str:
        ...


@dataclass(frozen=True)
class RerankResult:
    winner_index: int
    winner: str
    scorer_type: str
    reasoning: str
    scores: list[float | None] | None = None
    raw_output: str | None = None
    fallback_used: bool = False

    def to_metadata(self) -> dict[str, Any]:
        return {
            "scorer_type": self.scorer_type,
            "winner_index_among_valid": self.winner_index,
            "reasoning": self.reasoning,
            "scores": self.scores,
            "raw_output": self.raw_output,
            "fallback_used": self.fallback_used,
        }


class PreferenceRewardScorer:
    """Listwise LLM judge or pointwise Hugging Face reward-model reranker."""

    SUPPORTED_TYPES = {"llm_judge", "deberta_reward"}

    def __init__(
        self,
        scorer_type: str,
        *,
        llm_runner: TextGenerationProtocol | None = None,
        reward_model_name: str | None = None,
        device: str = "auto",
        max_length: int = 512,
        positive_label_index: int = -1,
        judge_max_new_tokens: int = 384,
    ) -> None:
        scorer_type = scorer_type.strip().casefold()
        if scorer_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported scorer_type={scorer_type!r}. "
                f"Choose one of {sorted(self.SUPPORTED_TYPES)}."
            )
        if scorer_type == "llm_judge" and llm_runner is None:
            raise ValueError("llm_runner is required when scorer_type='llm_judge'.")
        if scorer_type == "deberta_reward" and not reward_model_name:
            raise ValueError(
                "reward_model_name must identify a preference-fine-tuned sequence classifier "
                "when scorer_type='deberta_reward'."
            )

        self.scorer_type = scorer_type
        self.llm_runner = llm_runner
        self.reward_model_name = reward_model_name
        self.device = device
        self.max_length = int(max_length)
        self.positive_label_index = int(positive_label_index)
        self.judge_max_new_tokens = int(judge_max_new_tokens)
        self._reward_tokenizer: Any | None = None
        self._reward_model: Any | None = None
        self._reward_device: Any | None = None

    def rank_candidates(
        self,
        item: dict[str, str],
        candidates: list[str],
        contexts: list[str] | None = None,
    ) -> str:
        """Return the highest-ranked candidate without rewriting it."""
        return self.rank_with_details(item, candidates, contexts=contexts).winner

    def rank_with_details(
        self,
        item: dict[str, str],
        candidates: list[str],
        contexts: list[str] | None = None,
    ) -> RerankResult:
        if not candidates:
            raise ValueError("Cannot rank an empty candidate list.")
        if len(candidates) == 1:
            return RerankResult(
                winner_index=0,
                winner=candidates[0],
                scorer_type=self.scorer_type,
                reasoning="Only one valid candidate; semantic reranking was skipped.",
            )
        if self.scorer_type == "llm_judge":
            return self._rank_with_llm_judge(item, candidates, contexts)
        return self._rank_with_deberta(item, candidates, contexts)

    def _rank_with_llm_judge(
        self,
        item: dict[str, str],
        candidates: list[str],
        contexts: list[str] | None,
    ) -> RerankResult:
        assert self.llm_runner is not None
        prompt = build_semantic_judge_prompt(item, candidates, contexts)
        raw_output = self.llm_runner.generate_text(
            prompt,
            generation_overrides={
                "do_sample": False,
                "max_new_tokens": self.judge_max_new_tokens,
            },
        )
        try:
            payload = _parse_json_object(raw_output)
            winner_index = int(payload["winner_index"])
            if not 0 <= winner_index < len(candidates):
                raise ValueError(f"winner_index={winner_index} is outside the candidate range.")
            reasoning = str(payload.get("reasoning", "")).strip()
            scores = _extract_overall_scores(payload.get("scores"), len(candidates))
            return RerankResult(
                winner_index=winner_index,
                winner=candidates[winner_index],
                scorer_type=self.scorer_type,
                reasoning=reasoning,
                scores=scores,
                raw_output=raw_output,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning("Invalid LLM-judge response; selecting the first valid candidate: %s", exc)
            return RerankResult(
                winner_index=0,
                winner=candidates[0],
                scorer_type=self.scorer_type,
                reasoning=f"Judge output was invalid; used first-valid fallback ({type(exc).__name__}).",
                raw_output=raw_output,
                fallback_used=True,
            )

    def _rank_with_deberta(
        self,
        item: dict[str, str],
        candidates: list[str],
        contexts: list[str] | None,
    ) -> RerankResult:
        self._ensure_reward_model()
        import torch

        task_text = _build_reward_input(item, contexts)
        encoded = self._reward_tokenizer(
            [task_text] * len(candidates),
            candidates,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(self._reward_device) for name, tensor in encoded.items()}
        with torch.no_grad():
            logits = self._reward_model(**encoded).logits.detach().float().cpu()

        if logits.ndim == 1:
            rewards = logits
        elif logits.shape[-1] == 1:
            rewards = logits[:, 0]
        else:
            rewards = logits[:, self.positive_label_index]

        scores = [float(score) for score in rewards.tolist()]
        winner_index = max(range(len(scores)), key=scores.__getitem__)
        return RerankResult(
            winner_index=winner_index,
            winner=candidates[winner_index],
            scorer_type=self.scorer_type,
            reasoning="Selected the candidate with the highest reward-model logit.",
            scores=scores,
        )

    def _ensure_reward_model(self) -> None:
        if self._reward_model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._reward_device = torch.device(device)
        self._reward_tokenizer = AutoTokenizer.from_pretrained(self.reward_model_name)
        self._reward_model = AutoModelForSequenceClassification.from_pretrained(self.reward_model_name)
        self._reward_model.to(self._reward_device)
        self._reward_model.eval()


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise TypeError("Judge output must be a JSON object.")
    return payload


def _extract_overall_scores(raw_scores: Any, candidate_count: int) -> list[float | None] | None:
    if not isinstance(raw_scores, list):
        return None
    scores: list[float | None] = [None] * candidate_count
    for entry in raw_scores:
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry["index"])
            overall = float(entry["overall"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= index < candidate_count:
            scores[index] = overall
    return scores


def _build_reward_input(item: dict[str, str], contexts: list[str] | None) -> str:
    if item["input_type"] == "headline":
        task = f"Write a humorous punchline for this headline: {item.get('headline', '')}"
    else:
        task = (
            "Write a joke containing both required words: "
            f"{item.get('word1', '-')} | {item.get('word2', '-')}"
        )
    if contexts:
        task += "\nBackground context:\n" + "\n".join(f"- {context}" for context in contexts)
    return task
