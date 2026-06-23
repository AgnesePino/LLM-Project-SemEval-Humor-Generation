#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from humor_gen.preferences import build_preferences
from humor_gen.utils import check_output_path, setup_logging, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a DPO-style preference dataset from judgments.")
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--outputs-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    check_output_path(args.output, overwrite=args.overwrite)
    rows = build_preferences(args.judgments, args.outputs_dir)
    write_jsonl(rows, args.output, overwrite=True)


if __name__ == "__main__":
    main()
