from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_dataset(path: str | Path) -> list[dict[str, str]]:
    """Load TSV, CSV or JSONL and normalize rows to the internal task format."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = _load_jsonl(path)
    elif suffix in {".tsv", ".csv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        rows = _load_delimited(path, delimiter)
    else:
        raise ValueError(f"Unsupported dataset format '{suffix}'. Use TSV, CSV or JSONL.")
    return [normalize_row(row, idx) for idx, row in enumerate(rows, start=1)]


def normalize_row(row: dict[str, Any], idx: int = 0) -> dict[str, str]:
    row = {str(k).strip(): "" if v is None else str(v).strip() for k, v in row.items()}
    item_id = row.get("id") or row.get("ID") or row.get("example_id") or f"row_{idx:05d}"
    input_type = _canonical_type(row)
    
    # Estensione protetta: supporta sia le chiavi interne che quelle ufficiali SemEval (news_headline / news_title)
    headline = _clean_missing(row.get("headline") or row.get("news_headline") or row.get("news_title") or row.get("title") or "")
    word1 = _clean_missing(row.get("word1") or row.get("word_1") or row.get("word_a") or row.get("word_inclusion") or "")
    word2 = _clean_missing(row.get("word2") or row.get("word_2") or row.get("word_b") or "")
    
    if input_type == "headline" and not headline:
        raise ValueError(f"Row {item_id} is headline-based but has no headline.")
    if input_type == "word_pair" and (not word1 or not word2):
        raise ValueError(f"Row {item_id} is word_pair but does not contain both words.")
        
    return {
        "id": item_id,
        "input_type": input_type,
        "headline": headline,
        "word1": word1,
        "word2": word2,
    }


def build_prompt_input(item: dict[str, str]) -> str:
    if item["input_type"] == "headline":
        return item["headline"]
    return f"{item['word1']} / {item['word2']}"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def _load_delimited(path: Path, delimiter: str) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter=delimiter))


def _canonical_type(row: dict[str, str]) -> str:
    raw = (row.get("input_type") or row.get("type") or row.get("task") or "").strip().casefold()
    if raw in {"headline", "news_headline", "headline-based", "title"}:
        return "headline"
    if raw in {"word_pair", "word-pair", "word_inclusion", "word inclusion", "words"}:
        return "word_pair"
        
    headline = _clean_missing(row.get("headline") or row.get("news_headline") or row.get("news_title") or row.get("title") or "")
    word1 = _clean_missing(row.get("word1") or row.get("word_1") or row.get("word_inclusion") or "")
    word2 = _clean_missing(row.get("word2") or row.get("word_2") or "")
    
    if headline:
        return "headline"
    if word1 and word2:
        return "word_pair"
    raise ValueError(f"Cannot infer input_type from row with id={row.get('id', '<missing>')}")


def _clean_missing(value: str) -> str:
    value = str(value).strip()
    return "" if value in {"-", "—", "nan", "None", "null"} else value