#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from humor_gen.dpo import run_dpo_training
from humor_gen.utils import load_yaml, require_project_venv, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional DPO/LoRA training on a local CUDA GPU.")
    parser.add_argument("--config", default="configs/dpo.yaml")
    parser.add_argument("--mock", action="store_true", help="Create a placeholder checkpoint without training.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_project_venv()
    setup_logging(args.verbose)
    cfg = load_yaml(args.config)
    run_dpo_training(cfg, mock=args.mock)


if __name__ == "__main__":
    main()
