from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - convenience fallback for bare local mock runs
    def tqdm(iterable, **_: Any):
        return iterable

from humor_gen.data import load_dataset
from humor_gen.models import get_runner
from humor_gen.utils import read_jsonl, require_gpu_for_real_run, require_hf_token, resolve_model_config


def run_tournament(
    input_dir: str,
    output_input_path: str | None,
    method: str,
    generation_cfg: dict[str, Any],
    models_config: dict[str, Any],
    models_config_path: str,
    mock: bool,
    seed: int = 13,
) -> list[dict[str, Any]]:
    require_gpu_for_real_run(mock)
    outputs = load_generation_outputs(input_dir, method)
    items = _load_items_from_outputs(outputs) if output_input_path is None else {row["id"]: row for row in load_dataset(output_input_path)}
    rows: list[dict[str, Any]] = []
    rng = random.Random(seed)
    
    # === VARIABILI DI CONTROLLO MEMORIA VRAM ===
    current_judge = None
    runner = None

    for matchup in models_config.get("judge_tournament", []):
        model_a = matchup["model_a"]
        model_b = matchup["model_b"]
        judge_key = matchup["judge"]
        if model_a not in outputs or model_b not in outputs:
            continue
            
        # === STRATEGIA DI SVUOTAMENTO VRAM DINAMICO ===
        # Se il giudice cambia rispetto al match precedente, liberiamo la scheda video
        if judge_key != current_judge and runner is not None:
            del runner
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            runner = None
            current_judge = None

        # Carichiamo il modello in GPU solo se non è già presente in memoria
        if runner is None:
            judge_cfg = resolve_model_config(judge_key, models_config_path)
            require_hf_token(judge_cfg, mock)
            runner = get_runner(judge_cfg, generation_cfg, mock)
            current_judge = judge_key

        shared_ids = sorted(set(outputs[model_a]) & set(outputs[model_b]) & set(items))
        for item_id in tqdm(shared_ids, desc=f"Judging {model_a} vs {model_b} by {judge_key}"):
            true_left = {"model": model_a, "joke": outputs[model_a][item_id]["generated_joke"]}
            true_right = {"model": model_b, "joke": outputs[model_b][item_id]["generated_joke"]}
            pair = [true_left, true_right]
            rng.shuffle(pair)
            judgment = runner.judge(items[item_id], pair[0]["joke"], pair[1]["joke"])
            winner = _map_winner(judgment.get("winner", "tie"), pair)
            rows.append(
                {
                    "id": item_id,
                    "judge": judge_key,
                    "model_a": pair[0]["model"],
                    "model_b": pair[1]["model"],
                    "joke_a": pair[0]["joke"],
                    "joke_b": pair[1]["joke"],
                    "winner": winner,
                    "reason": judgment.get("reason", ""),
                    "method": method,
                    "scores": judgment.get("scores", {}),
                    "metadata": {
                        "blind_labels": {"A": pair[0]["model"], "B": pair[1]["model"]},
                        "original_pair": [model_a, model_b],
                    },
                }
            )
            
    # Pulizia finale di cortesia al termine di tutto il torneo
    if runner is not None:
        del runner
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return rows


def load_generation_outputs(input_dir: str, method: str) -> dict[str, dict[str, dict[str, Any]]]:
    outputs: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted(Path(input_dir).rglob("*.jsonl")):
        for row in read_jsonl(path):
            if row.get("method") != method:
                continue
            model = row.get("model")
            if model:
                outputs.setdefault(model, {})[row["id"]] = row
    return outputs


def _load_items_from_outputs(outputs: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, str]]:
    items = {}
    for row in itertools.chain.from_iterable(model_rows.values() for model_rows in outputs.values()):
        input_type = row["input_type"]
        item = {"id": row["id"], "input_type": input_type, "headline": "", "word1": "", "word2": ""}
        if input_type == "headline":
            item["headline"] = row.get("input", "")
        else:
            words = [part.strip() for part in row.get("input", "").split("|")]
            item["word1"] = words[0] if words else ""
            item["word2"] = words[1] if len(words) > 1 else ""
        items[row["id"]] = item
    return items


def _map_winner(label: str, pair: list[dict[str, str]]) -> str:
    label = str(label).strip().casefold()
    if label == "a":
        return pair[0]["model"]
    if label == "b":
        return pair[1]["model"]
    return "tie"


def parse_judge_json(text: str) -> dict[str, Any]:
    return json.loads(text)