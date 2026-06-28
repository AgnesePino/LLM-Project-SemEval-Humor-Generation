from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskInput:
    id: str
    kind: str
    word1: str | None = None
    word2: str | None = None
    headline: str | None = None
    raw: dict[str, str] | None = None


@dataclass
class Candidate:
    text: str
    style: str
    seed: int
    temperature: float
    source_model: str = ""
    source_backend: str = ""
    source_base_url: str = ""
    score: float = 0.0
    judge_score: float = 0.0
    judge_raw: str = ""
    humor_score: float | None = None
    valid: bool = True
    invalid_reason: str = ""
