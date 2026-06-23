from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml


def require_project_venv() -> None:
    project_root = Path(__file__).resolve().parents[2]
    expected_venv = (project_root / ".venv").resolve()
    active_prefix = Path(sys.prefix).resolve()
    if active_prefix != expected_venv:
        raise RuntimeError(
            "This command must run inside the project virtual environment. "
            "Run 'source .venv/bin/activate' first."
        )


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: str | Path, overwrite: bool = False) -> None:
    path = Path(path)
    ensure_parent(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(row: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def check_output_path(path: str | Path, overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")
    ensure_parent(path)


def require_gpu_for_real_run(mock: bool) -> None:
    if mock:
        return
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Real model execution requires torch. Install requirements.txt in the project .venv.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available. Check the local NVIDIA driver or use --mock.")


def require_hf_token(model_cfg: dict[str, Any], mock: bool) -> None:
    if mock or not model_cfg.get("requires_hf_token"):
        return
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError(
            f"{model_cfg.get('display_name', 'This model')} may require a HuggingFace token. "
            "Set HF_TOKEN in the local environment after accepting the model license."
        )


def resolve_model_config(model_key: str, models_config_path: str | Path = "configs/models.yaml") -> dict[str, Any]:
    cfg = load_yaml(models_config_path)
    models = cfg.get("models", {})
    if model_key not in models:
        raise KeyError(f"Model '{model_key}' not found in {models_config_path}. Available: {sorted(models)}")
    model_cfg = dict(models[model_key])
    model_cfg["key"] = model_key
    return model_cfg


def output_input_text(item: dict[str, Any]) -> str:
    if item["input_type"] == "headline":
        return item.get("headline", "")
    return f"{item.get('word1', '')} | {item.get('word2', '')}"
