from __future__ import annotations

import argparse
from typing import Any


class HumorScorer:
    def score(self, texts: list[str]) -> list[float]:
        raise NotImplementedError


class NullHumorScorer(HumorScorer):
    def score(self, texts: list[str]) -> list[float]:
        return [0.0 for _ in texts]


class TransformersHumorScorer(HumorScorer):
    def __init__(self, model_ids: list[str], device: int):
        if not model_ids:
            raise ValueError("At least one humor model is required")
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "The humor scorer requires transformers. Install with: pip install torch transformers"
            ) from exc
        self.pipelines = []
        for model_id in model_ids:
            tokenizer_id = fallback_tokenizer_id(model_id)
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
            except OSError:
                if tokenizer_id is None:
                    raise
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
            self.pipelines.append(pipeline("text-classification", model=model, tokenizer=tokenizer, device=device))

    def score(self, texts: list[str]) -> list[float]:
        if not texts:
            return []
        per_model_scores: list[list[float]] = []
        for classifier in self.pipelines:
            raw_outputs = classifier(texts, truncation=True, top_k=None)
            per_model_scores.append([extract_humor_probability(output) for output in raw_outputs])
        averaged: list[float] = []
        for index in range(len(texts)):
            averaged.append(sum(scores[index] for scores in per_model_scores) / len(per_model_scores))
        return averaged


def extract_humor_probability(output: Any) -> float:
    if isinstance(output, dict):
        output = [output]
    if output and isinstance(output[0], list):
        output = output[0]
    if not isinstance(output, list):
        return 0.0

    best_unknown = 0.0
    for entry in output:
        label = str(entry.get("label", "")).lower()
        score = float(entry.get("score", 0.0))
        compact = label.replace("_", "").replace("-", "")
        if "humor" in compact or "funny" in compact or compact in {"label1", "1", "positive", "pos"}:
            if not any(marker in compact for marker in ["nohumor", "nonhumor", "notfunny", "negative", "neg"]):
                return score
        if any(marker in compact for marker in ["nohumor", "nonhumor", "notfunny", "negative", "neg"]) or compact in {
            "label0",
            "0",
        }:
            return 1.0 - score
        best_unknown = max(best_unknown, score)
    return best_unknown


def fallback_tokenizer_id(model_id: str) -> str | None:
    if model_id.startswith("Humor-Research/humor-detection-"):
        return "roberta-base"
    return None


def make_humor_scorer(args: argparse.Namespace) -> HumorScorer | None:
    model_ids = getattr(args, "humor_model", None) or []
    if not model_ids:
        return None
    return TransformersHumorScorer(model_ids, args.humor_device)
