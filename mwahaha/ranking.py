from __future__ import annotations

import re

from .clients import LLMClient
from .humor import HumorScorer
from .io import parse_jsonish
from .prompts import SYSTEM_JUDGE, build_score_prompt
from .schema import Candidate, TaskInput
from .validation import clean_text, count_words


def score_candidates(
    client: LLMClient,
    item: TaskInput,
    candidates: list[Candidate],
    humor_scorer: HumorScorer | None = None,
    humor_weight: float = 0.25,
) -> list[Candidate]:
    valid_candidates = [candidate for candidate in candidates if candidate.valid]
    if not valid_candidates:
        return candidates
    humor_scores: list[float | None] = [None for _ in valid_candidates]
    if humor_scorer is not None and humor_weight > 0:
        humor_scores = humor_scorer.score([candidate.text for candidate in valid_candidates])
    for candidate in valid_candidates:
        prompt = build_score_prompt(item, candidate.text)
        raw = client.chat(
            [{"role": "system", "content": SYSTEM_JUDGE}, {"role": "user", "content": prompt}],
            temperature=0.35,
            max_tokens=120,
            seed=candidate.seed + 7777,
        )
        candidate.judge_raw = raw
        candidate.judge_score = parse_score(raw)
    for candidate, humor_score in zip(valid_candidates, humor_scores):
        candidate.humor_score = humor_score
        if humor_score is None:
            candidate.score = candidate.judge_score
        else:
            candidate.score = (1.0 - humor_weight) * candidate.judge_score + humor_weight * (10.0 * humor_score)
    return sorted(valid_candidates, key=lambda c: c.score, reverse=True) + [c for c in candidates if not c.valid]


def parse_score(raw: str) -> float:
    data = parse_jsonish(raw)
    if isinstance(data, dict):
        try:
            return float(data.get("score", 0.0))
        except (TypeError, ValueError):
            pass
    match = re.search(r"\b(10(?:\.0)?|[0-9](?:\.\d+)?)\b", raw)
    return float(match.group(1)) if match else 0.0


def pairwise_tournament(client: LLMClient, item: TaskInput, candidates: list[Candidate], top_k: int) -> Candidate:
    valid = [candidate for candidate in candidates if candidate.valid]
    if not valid:
        return fallback_candidate(item, candidates)
    bracket = valid[:top_k]
    champion = bracket[0]
    for challenger in bracket[1:]:
        winner = judge_pair(client, item, champion.text, challenger.text, champion.seed + challenger.seed)
        if winner == "B":
            champion = challenger
    return champion


def fallback_candidate(item: TaskInput, candidates: list[Candidate]) -> Candidate:
    if candidates:
        best = sorted(candidates, key=lambda c: (len(c.text) > 0, -len(c.invalid_reason)), reverse=True)[0]
        return best
    if item.kind == "word_inclusion":
        text = f"{item.word1} and {item.word2} walked into a joke, but the punchline filed a noise complaint."
    else:
        text = f"{item.headline} sounds like the news got tired and started writing its own punchlines."
    return Candidate(clean_text(text), "fallback", 0, 0.0)


def judge_pair(client: LLMClient, item: TaskInput, candidate_a: str, candidate_b: str, seed: int) -> str:
    if item.kind == "word_inclusion":
        context = f"Required words: {item.word1}, {item.word2}"
    else:
        context = f"Headline: {item.headline}"
    prompt = (
        "/no_think\n"
        f"{context}\n\n"
        f"Candidate A: {candidate_a}\n"
        f"Candidate B: {candidate_b}\n\n"
        "Choose the candidate more likely to win a human pairwise humor battle. "
        'Return JSON exactly like {"winner": "A", "reason": "short reason"}'
    )
    raw = client.chat(
        [{"role": "system", "content": SYSTEM_JUDGE}, {"role": "user", "content": prompt}],
        temperature=0.25,
        max_tokens=100,
        seed=seed + 9999,
    )
    data = parse_jsonish(raw)
    if isinstance(data, dict):
        winner = str(data.get("winner", "")).strip().upper()
        if winner in {"A", "B"}:
            return winner
    return "A" if "A" in raw[:20].upper() else "B"


def output_style_metrics(inputs: list[TaskInput], output_by_id: dict[str, str]) -> dict[str, Any]:
    texts = [output_by_id.get(item.id, "") for item in inputs]
    return {
        "rows": len(texts),
        "i_tried": sum(1 for text in texts if re.search(r"^i tried\b", text, flags=re.IGNORECASE)),
        "i_asked": sum(1 for text in texts if re.search(r"^i asked\b", text, flags=re.IGNORECASE)),
        "i_told": sum(1 for text in texts if re.search(r"^i told\b", text, flags=re.IGNORECASE)),
        "turns_out": sum(1 for text in texts if re.search(r"\bturns out\b", text, flags=re.IGNORECASE)),
        "quote_marks": sum(1 for text in texts if '"' in text or "\u201c" in text or "\u201d" in text),
        "long_over_200": sum(1 for text in texts if len(text) > 200),
        "avg_chars": round(sum(len(text) for text in texts) / len(texts), 1) if texts else 0.0,
        "avg_words": round(sum(count_words(text) for text in texts) / len(texts), 1) if texts else 0.0,
    }
