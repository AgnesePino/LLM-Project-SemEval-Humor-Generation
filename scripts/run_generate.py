#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from humor_gen.generate import generate_dataset
from humor_gen.utils import check_output_path, load_yaml, require_project_venv, setup_logging, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate baseline jokes with one configured model.")
    parser.add_argument("--model", required=True, help="Model key from configs/models.yaml.")
    parser.add_argument("--input", required=True, help="Input TSV/CSV/JSONL dataset.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--config", default="configs/generation.yaml", help="Generation YAML config.")
    parser.add_argument("--models-config", default=None, help="Optional models YAML override.")
    parser.add_argument("--mock", action="store_true", help="Run without loading real LLMs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of examples to process.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_project_venv()
    setup_logging(args.verbose)
    cfg = load_yaml(args.config)
    models_config_path = args.models_config or cfg.get("models_config", "configs/models.yaml")
    check_output_path(args.output, overwrite=args.overwrite)
    rows = generate_dataset(
        model_key=args.model,
        input_path=args.input,
        generation_cfg=cfg,
        models_config_path=models_config_path,
        mock=args.mock,
        method="base",
        limit=args.limit,
    )
    write_jsonl(rows, args.output, overwrite=True)


if __name__ == "__main__":
    main()
