from __future__ import annotations

import logging
from typing import Any, Protocol

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - convenience fallback for bare local mock runs
    def tqdm(iterable, **_: Any):
        return iterable

from humor_gen.data import build_prompt_input, load_dataset
from humor_gen.models import get_runner
from humor_gen.utils import output_input_text, require_gpu_for_real_run, require_hf_token, resolve_model_config
from humor_gen.validate import validate_joke

LOGGER = logging.getLogger(__name__)


class RetrieverProtocol(Protocol):
    def retrieve(self, query: str, k: int) -> list[str]:
        ...


def generate_dataset(
    model_key: str,
    input_path: str,
    generation_cfg: dict[str, Any],
    models_config_path: str,
    mock: bool,
    method: str = "base",
    retriever: RetrieverProtocol | None = None,
    k: int = 0,
    rag_apply_to: str = "all",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    items = load_dataset(input_path)
    if limit is not None:
        items = items[:limit]
    model_cfg = resolve_model_config(model_key, models_config_path)
    require_gpu_for_real_run(mock)
    require_hf_token(model_cfg, mock)
    runner = get_runner(model_cfg, generation_cfg, mock)
    max_words = generation_cfg.get("validation", {}).get("max_words", 45)
    rows: list[dict[str, Any]] = []
    for item in tqdm(items, desc=f"Generating {model_key}/{method}"):
        contexts = _retrieve_contexts(retriever, item, k, rag_apply_to)
        joke = runner.generate_joke(item, contexts=contexts)
        valid, errors = validate_joke(joke, item, max_words=max_words)
        rows.append(
            {
                "id": item["id"],
                "input_type": item["input_type"],
                "input": output_input_text(item),
                "model": model_key,
                "method": method,
                "generated_joke": joke,
                "valid": valid,
                "constraint_errors": errors,
                "metadata": {
                    "model_id": model_cfg["hf_id"],
                    "mock": mock,
                    "prompt_input": build_prompt_input(item),
                    "rag_contexts": contexts,
                    "rag_k": k if contexts else 0,
                },
            }
        )
    return rows


def _retrieve_contexts(retriever: RetrieverProtocol | None, item: dict[str, str], k: int, apply_to: str) -> list[str]:
    if retriever is None or k <= 0:
        return []
    if apply_to == "headline" and item["input_type"] != "headline":
        return []
    if apply_to in {"word_pair", "word-pair"} and item["input_type"] != "word_pair":
        return []
    if item["input_type"] == "headline":
        query = f"Background facts and context about: {item['headline']}"
    else:
        query = f"Meaning, usage, and related concepts for: {item['word1']} and {item['word2']}"
    return retriever.retrieve(query, k)
