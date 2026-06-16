from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    reasons: list[str]


def validate_candidate(candidate, item):
    reasons = []
    normalized = candidate.casefold()

    if not candidate.strip():
        reasons.append("empty_output")

    if len(candidate.split()) > 60:
        reasons.append("too_long")

    if item.get("type") == "word_inclusion":
        for word_key in ("word1", "word2"):
            word = item[word_key]
            if not _contains_word(normalized, word):
                reasons.append(f"missing_{word_key}")

    if item.get("type") == "news_headline":
        if "headline" not in item:
            reasons.append("missing_headline_input")

    return ValidationResult(is_valid=not reasons, reasons=reasons)


def _contains_word(normalized_candidate, word):
    normalized_word = str(word).strip().casefold()
    if not normalized_word:
        return False

    pattern = rf"(?<![A-Za-z0-9]){re.escape(normalized_word)}(?![A-Za-z0-9])"
    return re.search(pattern, normalized_candidate) is not None
