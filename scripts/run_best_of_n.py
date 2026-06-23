#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from humor_gen.data import load_dataset
from humor_gen.models import get_runner
from humor_gen.reranker import PreferenceRewardScorer
from humor_gen.selection import generate_many_and_select
from humor_gen.utils import (
    check_output_path,
    load_yaml,
    output_input_text,
    require_gpu_for_real_run,
    require_hf_token,
    resolve_model_config,
    setup_logging,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate N candidates and keep a lightweight reranked best joke.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/generation.yaml")
    parser.add_argument("--models-config", default=None)
    parser.add_argument("--scorer-type", choices=["llm_judge", "deberta_reward"], default="llm_judge")
    parser.add_argument("--reward-model-name", default=None)
    parser.add_argument("--fallback", choices=["greedy", "first"], default="greedy")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    cfg = load_yaml(args.config)
    models_config_path = args.models_config or cfg.get("models_config", "configs/models.yaml")
    model_cfg = resolve_model_config(args.model, models_config_path)
    require_gpu_for_real_run(args.mock)
    require_hf_token(model_cfg, args.mock)
    runner = get_runner(model_cfg, cfg, args.mock)
    reranker = PreferenceRewardScorer(
        scorer_type=args.scorer_type,
        llm_runner=runner if args.scorer_type == "llm_judge" else None,
        reward_model_name=args.reward_model_name,
    )
    check_output_path(args.output, overwrite=args.overwrite)
    rows = []
    for item in load_dataset(args.input):
        selection = generate_many_and_select(
            runner,
            item,
            contexts=[],
            selection_cfg={"num_candidates": args.n, "fallback": args.fallback},
            max_words=int(cfg.get("validation", {}).get("max_words", 45)),
            reranker=reranker,
        )
        rows.append(
            {
                "id": item["id"],
                "input_type": item["input_type"],
                "input": output_input_text(item),
                "model": args.model,
                "method": "best_of_n",
                "generated_joke": selection.joke,
                "valid": selection.valid,
                "constraint_errors": selection.errors,
                "metadata": {
                    "n": args.n,
                    "mock": args.mock,
                    "reranker": selection.reranker_metadata,
                    "candidates": [
                        {
                            "generated_joke": candidate.joke,
                            "valid": candidate.valid,
                            "constraint_errors": candidate.errors,
                            "score": candidate.score,
                            "fallback": candidate.fallback,
                        }
                        for candidate in selection.candidates
                    ],
                },
            }
        )
    write_jsonl(rows, args.output, overwrite=True)


if __name__ == "__main__":
    main()
