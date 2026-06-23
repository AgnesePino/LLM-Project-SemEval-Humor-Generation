#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from humor_gen.metrics import create_figures, generation_summary, load_generated_dir, load_judged_dir, win_rate_summary, write_csv
from humor_gen.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CSV summaries and figures from generation/judgment outputs.")
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--judged-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated = load_generated_dir(args.generated_dir)
    judged = load_judged_dir(args.judged_dir)
    write_csv(generation_summary(generated), str(output / "generation_summary.csv"))
    write_csv(win_rate_summary(judged), str(output / "win_rate_summary.csv"))
    create_figures(generated, judged, str(output))


if __name__ == "__main__":
    main()
