#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from humor_gen.judge import run_tournament
from humor_gen.utils import check_output_path, load_yaml, setup_logging, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run blind round-robin LLM-as-a-judge tournament.")
    parser.add_argument("--input-dir", required=True, help="Directory containing generation JSONL files.")
    parser.add_argument("--output", required=True, help="Judgment JSONL path.")
    parser.add_argument("--method", default="base", help="Generation method to compare.")
    parser.add_argument("--dataset", default=None, help="Optional original dataset path for judge prompts.")
    parser.add_argument("--config", default="configs/generation.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    generation_cfg = load_yaml(args.config)
    models_cfg = load_yaml(args.models_config)
    check_output_path(args.output, overwrite=args.overwrite)
    rows = run_tournament(
        input_dir=args.input_dir,
        output_input_path=args.dataset,
        method=args.method,
        generation_cfg=generation_cfg,
        models_config=models_cfg,
        models_config_path=args.models_config,
        mock=args.mock,
        seed=args.seed,
    )
    write_jsonl(rows, args.output, overwrite=True)


if __name__ == "__main__":
    main()
