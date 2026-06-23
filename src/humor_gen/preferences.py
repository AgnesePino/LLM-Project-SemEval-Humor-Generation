from __future__ import annotations

from pathlib import Path
from typing import Any

from humor_gen.prompts import build_preference_prompt
from humor_gen.utils import read_jsonl


def build_preferences(judgments_path: str, outputs_dir: str) -> list[dict[str, Any]]:
    judgments = read_jsonl(judgments_path)
    items = _items_from_outputs(outputs_dir)
    rows: list[dict[str, Any]] = []
    for judgment in judgments:
        winner = judgment.get("winner")
        if winner == "tie":
            continue
        if winner == judgment.get("model_a"):
            chosen, rejected = judgment["joke_a"], judgment["joke_b"]
        elif winner == judgment.get("model_b"):
            chosen, rejected = judgment["joke_b"], judgment["joke_a"]
        else:
            continue
        item = items.get(judgment["id"])
        prompt = build_preference_prompt(item) if item else ""
        rows.append(
            {
                "id": judgment["id"],
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "judge": judgment.get("judge"),
                "source_models": {
                    "chosen": winner,
                    "rejected": judgment["model_b"] if winner == judgment["model_a"] else judgment["model_a"],
                },
                "method": judgment.get("method"),
                "reason": judgment.get("reason", ""),
            }
        )
    return rows


def _items_from_outputs(outputs_dir: str) -> dict[str, dict[str, str]]:
    items: dict[str, dict[str, str]] = {}
    # rglob so generation outputs stored under subfolders (baseline/, rag/) resolve correctly.
    for path in sorted(Path(outputs_dir).rglob("*.jsonl")):
        for row in read_jsonl(path):
            if row["id"] in items:
                continue
            item = {"id": row["id"], "input_type": row["input_type"], "headline": "", "word1": "", "word2": ""}
            if row["input_type"] == "headline":
                item["headline"] = row.get("input", "")
            else:
                parts = [part.strip() for part in row.get("input", "").split("|")]
                item["word1"] = parts[0] if parts else ""
                item["word2"] = parts[1] if len(parts) > 1 else ""
            items[row["id"]] = item
    return items
