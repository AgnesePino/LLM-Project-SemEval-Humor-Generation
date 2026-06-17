from __future__ import annotations

import re

ENGLISH_HINT_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "with",
    "for",
    "is",
    "are",
    "was",
    "were",
    "why",
    "when",
    "because",
}


def validate_joke(joke: str, item: dict[str, str], max_words: int = 45) -> tuple[bool, list[str]]:
    errors: list[str] = []
    text = (joke or "").strip()
    lowered = text.casefold()
    if not text:
        errors.append("empty_output")
    if len(text.split()) > max_words:
        errors.append("too_long")
    if "\n" in text.strip():
        errors.append("multiple_lines")
    if _has_banned_preface(lowered):
        errors.append("contains_preface")
    if _looks_like_explanation(lowered):
        errors.append("contains_explanation")
    if text and not _looks_english(text):
        errors.append("not_english_like")
    if item["input_type"] == "word_pair":
        if not _contains_word(lowered, item["word1"]):
            errors.append("missing_word1")
        if not _contains_word(lowered, item["word2"]):
            errors.append("missing_word2")
    if item["input_type"] == "headline":
        if not _has_headline_overlap(lowered, item.get("headline", "")):
            errors.append("low_headline_relevance")
    return not errors, errors


def clean_joke(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^(sure[,.!]?\s*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(here('s| is)\s+a\s+joke[:\s-]*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(joke|answer|response)\s*:\s*", "", text, flags=re.IGNORECASE)
    return " ".join(text.strip().strip('"').split())


def _has_banned_preface(lowered: str) -> bool:
    prefixes = ("sure", "here is a joke", "here's a joke", "joke:", "answer:", "response:")
    return lowered.startswith(prefixes)


def _looks_like_explanation(lowered: str) -> bool:
    return any(marker in lowered for marker in ("this joke", "the humor", "explanation:", "the punchline is"))


def _contains_word(lowered: str, word: str) -> bool:
    word = str(word).strip().casefold()
    if not word:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(word)}s?(?![a-z0-9])"
    return re.search(pattern, lowered) is not None


def _looks_english(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z']+", text.casefold())
    if not tokens:
        return False
    ascii_ratio = sum(ch.isascii() for ch in text) / max(len(text), 1)
    hint_ratio = sum(tok in ENGLISH_HINT_WORDS for tok in tokens) / max(len(tokens), 1)
    return ascii_ratio > 0.9 and (hint_ratio > 0.08 or len(tokens) <= 8)


def _has_headline_overlap(lowered: str, headline: str) -> bool:
    headline_terms = {
        token
        for token in re.findall(r"[A-Za-z]{4,}", headline.casefold())
        if token not in ENGLISH_HINT_WORDS
    }
    if not headline_terms:
        return True
    return any(term in lowered for term in headline_terms)
