from __future__ import annotations

import csv
import re
from pathlib import Path

from .schema import TaskInput


BOILERPLATE_PATTERNS = [
    r"^\s*here('?s| is)\s+(a|the)\s+joke\s*[:\-]",
    r"^\s*joke\s*[:\-]",
    r"^\s*final joke\s*[:\-]",
    r"^\s*answer\s*[:\-]",
]


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:\w+)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \u201c\u201d\"")
    return text


def validate_candidate(item: TaskInput, text: str) -> tuple[bool, str]:
    if not text:
        return False, "empty"
    if len(text) > 900:
        return False, "over_900_chars"
    if "\t" in text or "\r" in text or "\n" in text:
        return False, "contains_tab_or_newline"
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in BOILERPLATE_PATTERNS):
        return False, "boilerplate"
    if item.kind == "word_inclusion":
        assert item.word1 is not None and item.word2 is not None
        missing = [word for word in [item.word1, item.word2] if not contains_verbatim(text, word)]
        if missing:
            return False, f"missing_required_words:{','.join(missing)}"
    else:
        assert item.headline is not None
        if not headline_related(item.headline, text):
            return False, "weak_headline_overlap"
    return True, ""


def contains_verbatim(text: str, word: str) -> bool:
    word = word.strip()
    if not word:
        return True
    if re.search(r"[A-Za-z0-9_]", word):
        return re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text, flags=re.IGNORECASE) is not None
    return word in text


def headline_related(headline: str, text: str) -> bool:
    h_tokens = important_tokens(headline)
    if not h_tokens:
        return True
    text_lower = text.lower()
    hits = 0
    for token in h_tokens:
        if any(variant and variant in text_lower for variant in token_variants(token)):
            hits += 1
    return hits >= 1


def important_tokens(text: str) -> list[str]:
    stop = {
        "the", "a", "an", "and", "or", "but", "for", "to", "of", "in", "on", "at", "by", "with",
        "from", "as", "is", "are", "was", "were", "be", "been", "being", "this", "that", "after",
        "over", "under", "new", "says", "said", "will", "can", "could", "may", "might", "has", "have",
        "most", "more", "use", "made", "his", "what", "does", "mean", "rather", "than", "each", "other",
        "something", "people", "experts", "advise",
    }
    tokens = re.findall(r"[a-z0-9]{2,}", text.lower())
    return [token for token in tokens if token not in stop][:8]


def token_variants(token: str) -> set[str]:
    variants = {token}
    if len(token) > 4 and token.endswith("ing"):
        variants.add(token[:-3])
    if len(token) > 3 and token.endswith("es"):
        variants.add(token[:-2])
    if len(token) > 3 and token.endswith("s"):
        variants.add(token[:-1])
    return variants


def count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def validate_output(input_rows: list[TaskInput], output_path: Path) -> list[str]:
    errors: list[str] = []
    with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["id", "text"]:
            errors.append(f"Invalid header: {reader.fieldnames!r}; expected ['id', 'text']")
        output_rows = list(reader)
    expected_ids = [row.id for row in input_rows]
    seen: set[str] = set()
    output_by_id: dict[str, str] = {}
    for row in output_rows:
        item_id = row.get("id", "")
        if item_id in seen:
            errors.append(f"Duplicate id: {item_id}")
        seen.add(item_id)
        output_by_id[item_id] = row.get("text", "")
    missing = [item_id for item_id in expected_ids if item_id not in output_by_id]
    extra = [item_id for item_id in output_by_id if item_id not in set(expected_ids)]
    if missing:
        errors.append(f"Missing ids: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if extra:
        errors.append(f"Unexpected ids: {extra[:10]}{'...' if len(extra) > 10 else ''}")
    for item in input_rows:
        if item.id not in output_by_id:
            continue
        valid, reason = validate_candidate(item, output_by_id[item.id])
        if not valid:
            errors.append(f"{item.id}: {reason}")
    return errors
