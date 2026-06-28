from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .schema import Candidate, TaskInput
from .validation import clean_text, validate_candidate


ID_COLUMNS = ["id", "ID", "Id", "uid", "sample_id", "example_id"]
WORD1_COLUMNS = ["word1", "word_1", "word_a", "first_word", "w1"]
WORD2_COLUMNS = ["word2", "word_2", "word_b", "second_word", "w2"]
WORDS_COLUMNS = ["words", "required_words", "keywords", "constraints"]
HEADLINE_COLUMNS = ["headline", "title", "news_title", "prompt", "text", "context", "input"]
MISSING_VALUES = {"", "-", "--", "nan", "none", "null", "n/a", "na"}


def read_inputs(path: Path) -> list[TaskInput]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return read_json_inputs(path)
    return read_delimited_inputs(path)


def read_json_inputs(path: Path) -> list[TaskInput]:
    rows: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        rows = data if isinstance(data, list) else data.get("data", data.get("rows", []))
    return [row_to_input({str(k): "" if v is None else str(v) for k, v in row.items()}, idx) for idx, row in enumerate(rows)]


def read_delimited_inputs(path: Path) -> list[TaskInput]:
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:4096]
    delimiter = "\t" if sample.count("\t") >= sample.count(",") else ","
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames:
            return [row_to_input({k: (v or "") for k, v in row.items()}, idx) for idx, row in enumerate(reader)]

    rows: list[TaskInput] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader2 = csv.reader(handle, delimiter=delimiter)
        for idx, row in enumerate(reader2):
            if not row:
                continue
            rows.append(row_to_input({f"col{i}": value for i, value in enumerate(row)}, idx))
    return rows


def row_to_input(row: dict[str, str], idx: int) -> TaskInput:
    item_id = first_non_empty(row, ID_COLUMNS) or first_non_empty(row, ["col0"]) or str(idx)
    word1 = first_non_empty(row, WORD1_COLUMNS)
    word2 = first_non_empty(row, WORD2_COLUMNS)
    if word1 and word2:
        return TaskInput(item_id, "word_inclusion", word1=word1.strip(), word2=word2.strip(), raw=row)

    words_value = first_non_empty(row, WORDS_COLUMNS)
    if words_value:
        parsed_words = parse_two_words(words_value)
        if parsed_words:
            return TaskInput(item_id, "word_inclusion", word1=parsed_words[0], word2=parsed_words[1], raw=row)

    if "col1" in row and "col2" in row and row["col1"].strip() and row["col2"].strip():
        if looks_like_single_word(row["col1"]) and looks_like_single_word(row["col2"]):
            return TaskInput(item_id, "word_inclusion", word1=row["col1"].strip(), word2=row["col2"].strip(), raw=row)

    headline = first_non_empty(row, HEADLINE_COLUMNS)
    if not headline:
        non_id_values = [v for k, v in row.items() if k not in ID_COLUMNS and k != "col0" and v.strip()]
        headline = " ".join(non_id_values).strip()
    if not headline:
        raise ValueError(f"Could not infer input type for row {idx}: {row}")
    return TaskInput(item_id, "news_headline", headline=headline.strip(), raw=row)


def first_non_empty(row: dict[str, str], keys: Iterable[str]) -> str | None:
    for key in keys:
        if key in row and is_present_value(row[key]):
            return row[key]
    lower_map = {k.lower(): v for k, v in row.items()}
    for key in keys:
        value = lower_map.get(key.lower())
        if value and is_present_value(value):
            return value
    return None


def is_present_value(value: str) -> bool:
    return value.strip().lower() not in MISSING_VALUES


def parse_two_words(value: str) -> tuple[str, str] | None:
    cleaned = value.strip().strip("[](){}")
    if not cleaned:
        return None
    parts = [p.strip().strip("'\"") for p in re.split(r"[,;/|]+|\s+and\s+", cleaned) if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None


def looks_like_single_word(value: str) -> bool:
    return bool(re.fullmatch(r"[\w\-']{1,40}", value.strip(), flags=re.UNICODE))


def read_output_map(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {row.get("id", ""): row.get("text", "") for row in reader}


def parse_jsonish(raw: str) -> Any:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def write_output(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "text"])
        for item_id, text in rows:
            writer.writerow([item_id, clean_text(text)])


def candidate_record(item: TaskInput, candidate: Candidate) -> dict[str, Any]:
    return {
        "id": item.id,
        "kind": item.kind,
        "text": candidate.text,
        "style": candidate.style,
        "seed": candidate.seed,
        "temperature": candidate.temperature,
        "source_model": candidate.source_model,
        "source_backend": candidate.source_backend,
        "source_base_url": candidate.source_base_url,
        "valid": candidate.valid,
        "invalid_reason": candidate.invalid_reason,
    }


def write_candidate_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_candidate_pool(pool_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not pool_dir.exists():
        return records
    for path in sorted(pool_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                if isinstance(data, dict):
                    data["_pool_file"] = str(path)
                    records.append(data)
    return records


def candidate_from_record(item: TaskInput, record: dict[str, Any]) -> Candidate:
    text = clean_text(str(record.get("text", "")))
    valid, reason = validate_candidate(item, text)
    if not valid and record.get("invalid_reason"):
        reason = str(record.get("invalid_reason"))
    return Candidate(
        text=text,
        style=str(record.get("style", "pool")),
        seed=int(record.get("seed", 0) or 0),
        temperature=float(record.get("temperature", 0.0) or 0.0),
        source_model=str(record.get("source_model", "")),
        source_backend=str(record.get("source_backend", "")),
        source_base_url=str(record.get("source_base_url", "")),
        valid=valid,
        invalid_reason=reason,
    )


def slugify_model_alias(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return slug[:120] or "model"


def input_order_index(inputs: list[TaskInput], item_id: str) -> int:
    for index, item in enumerate(inputs):
        if item.id == item_id:
            return index
    return len(inputs)


def write_diagnostics(path: Path, item: TaskInput, candidates: list[Candidate], winner_text: str) -> None:
    payload = {
        "input": {
            "id": item.id,
            "kind": item.kind,
            "word1": item.word1,
            "word2": item.word2,
            "headline": item.headline,
            "raw": item.raw,
        },
        "winner": winner_text,
        "candidates": [candidate.__dict__ for candidate in candidates],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:120] or "item"
