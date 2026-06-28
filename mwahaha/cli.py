from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

from .clients import make_client
from .generation import dedupe_candidates, generate_baseline, generate_candidate_pool_for_item, generate_candidates
from .humor import make_humor_scorer
from .io import (
    candidate_from_record,
    candidate_record,
    input_order_index,
    read_candidate_pool,
    read_inputs,
    read_output_map,
    safe_filename,
    slugify_model_alias,
    write_candidate_records,
    write_diagnostics,
    write_output,
)
from .ranking import fallback_candidate, judge_pair, pairwise_tournament, score_candidates
from .refine import cmd_refine
from .validation import clean_text, validate_candidate, validate_output


def cmd_evaluate(args: argparse.Namespace) -> int:
    client = make_client(args)
    inputs = read_inputs(Path(args.input))
    if args.limit:
        inputs = inputs[: args.limit]
    output_by_id = read_output_map(Path(args.output))
    wins = 0
    losses = 0
    skipped = 0
    for index, item in enumerate(inputs, start=1):
        final = output_by_id.get(item.id)
        if not final:
            skipped += 1
            continue
        baseline = generate_baseline(client, item, args.seed + index * 1000)
        valid, _reason = validate_candidate(item, baseline)
        if not valid:
            skipped += 1
            continue
        winner = judge_pair(client, item, final, baseline, args.seed + index * 2000)
        if winner == "A":
            wins += 1
        else:
            losses += 1
        print(f"[{index}/{len(inputs)}] {item.id}: {'WIN' if winner == 'A' else 'LOSS'}", flush=True)
    total = wins + losses
    win_rate = wins / total if total else 0.0
    print(f"Final vs baseline wins: {wins}/{total} ({win_rate:.1%}); skipped={skipped}")
    if total and win_rate < args.min_win_rate:
        print(
            f"ERROR: win rate {win_rate:.1%} is below threshold {args.min_win_rate:.1%}",
            file=sys.stderr,
        )
        return 3
    return 0


def cmd_generate_candidates(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    inputs = read_inputs(Path(args.input))
    if args.limit:
        inputs = inputs[: args.limit]
    model_alias = args.model_alias or slugify_model_alias(args.model)
    pool_dir = Path(args.pool_dir)
    pool_path = pool_dir / f"{slugify_model_alias(model_alias)}.jsonl"
    existing_counts: dict[str, int] = {}
    if args.resume and pool_path.exists():
        for record in read_candidate_pool(pool_dir):
            if record.get("source_model") != model_alias:
                continue
            item_id = str(record.get("id", ""))
            existing_counts[item_id] = existing_counts.get(item_id, 0) + 1
        print(f"Resume enabled: loaded candidate counts for {len(existing_counts)} ids from {pool_path}.")

    client = make_client(args)
    for index, item in enumerate(inputs, start=1):
        existing = existing_counts.get(item.id, 0)
        if args.resume and existing >= args.variants_per_entry:
            print(f"[{index}/{len(inputs)}] skipping {item.id} ({model_alias}; {existing} candidates)", flush=True)
            continue
        print(f"[{index}/{len(inputs)}] generating candidates for {item.id} with {model_alias}", flush=True)
        candidates = generate_candidate_pool_for_item(
            client,
            item,
            args.variants_per_entry,
            args.seed + index * 1000,
            model_alias,
            args.backend,
            args.base_url,
            start_variant=existing if args.resume else 0,
        )
        records = [candidate_record(item, candidate) for candidate in candidates]
        write_candidate_records(pool_path, records)
        existing_counts[item.id] = existing + len(records)
        if args.sleep:
            time.sleep(args.sleep)

    print(f"Wrote candidate pool file {pool_path}.")
    return 0


def cmd_rank_candidates(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    inputs = read_inputs(Path(args.input))
    if args.limit:
        inputs = inputs[: args.limit]
    output_path = Path(args.output)
    output_by_id: dict[str, str] = {}
    if args.resume and output_path.exists():
        output_by_id = read_output_map(output_path)
        print(f"Resume enabled: loaded {len(output_by_id)} existing rows from {output_path}.")
    outputs: list[tuple[str, str]] = [(item.id, output_by_id[item.id]) for item in inputs if item.id in output_by_id]
    diagnostics_dir = Path(args.diagnostics_dir) if args.diagnostics_dir else None
    if diagnostics_dir:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

    records = read_candidate_pool(Path(args.pool_dir))
    records_by_id: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        records_by_id.setdefault(str(record.get("id", "")), []).append(record)
    print(f"Loaded {len(records)} candidates for {len(records_by_id)} ids from {args.pool_dir}.")

    client = make_client(args)
    humor_scorer = make_humor_scorer(args)
    for index, item in enumerate(inputs, start=1):
        if item.id in output_by_id:
            print(f"[{index}/{len(inputs)}] skipping {item.id} (already ranked)", flush=True)
            continue
        pool_records = records_by_id.get(item.id, [])
        candidates = dedupe_candidates([candidate_from_record(item, record) for record in pool_records])
        if not candidates:
            candidates = [fallback_candidate(item, [])]
        print(f"[{index}/{len(inputs)}] ranking {item.id} ({len(candidates)} candidates)", flush=True)
        ranked = score_candidates(client, item, candidates, humor_scorer, args.humor_weight)
        winner = pairwise_tournament(client, item, ranked, args.rerank_top_k)
        winner_text = clean_text(winner.text)
        outputs.append((item.id, winner_text))
        output_by_id[item.id] = winner_text
        if diagnostics_dir:
            write_diagnostics(diagnostics_dir / f"{safe_filename(item.id)}.json", item, ranked, winner_text)
        write_output(output_path, sorted(outputs, key=lambda pair: input_order_index(inputs, pair[0])))
        if args.sleep:
            time.sleep(args.sleep)

    write_output(output_path, sorted(outputs, key=lambda pair: input_order_index(inputs, pair[0])))
    errors = validate_output(inputs, output_path)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Wrote {output_path} with {len(outputs)} rows.")
    return 0


def run_pipeline(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    client = make_client(args)
    humor_scorer = make_humor_scorer(args)
    inputs = read_inputs(Path(args.input))
    if args.limit:
        inputs = inputs[: args.limit]
    output_path = Path(args.output)
    output_by_id: dict[str, str] = {}
    if args.resume and output_path.exists():
        output_by_id = read_output_map(output_path)
        print(f"Resume enabled: loaded {len(output_by_id)} existing rows from {output_path}.")
    outputs: list[tuple[str, str]] = [(item.id, output_by_id[item.id]) for item in inputs if item.id in output_by_id]
    diagnostics_dir = Path(args.diagnostics_dir) if args.diagnostics_dir else None
    if diagnostics_dir:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

    for index, item in enumerate(inputs, start=1):
        if item.id in output_by_id:
            print(f"[{index}/{len(inputs)}] skipping {item.id} (already generated)", flush=True)
            continue
        print(f"[{index}/{len(inputs)}] generating {item.id} ({item.kind})", flush=True)
        candidates = generate_candidates(client, item, args.variants_per_style, args.seed + index * 1000)
        ranked = score_candidates(client, item, candidates, humor_scorer, args.humor_weight)
        winner = pairwise_tournament(client, item, ranked, args.rerank_top_k)
        winner_text = clean_text(winner.text)
        outputs.append((item.id, winner_text))
        output_by_id[item.id] = winner_text
        if diagnostics_dir:
            write_diagnostics(diagnostics_dir / f"{safe_filename(item.id)}.json", item, ranked, winner_text)
        write_output(output_path, sorted(outputs, key=lambda pair: input_order_index(inputs, pair[0])))
        if args.sleep:
            time.sleep(args.sleep)

    write_output(output_path, sorted(outputs, key=lambda pair: input_order_index(inputs, pair[0])))
    errors = validate_output(inputs, output_path)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Wrote {output_path} with {len(outputs)} rows.")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    inputs = read_inputs(Path(args.input))
    errors = validate_output(inputs, Path(args.output))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print("Validation OK.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MWAHAHA Task A English local open-source pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Generate task-a-en.tsv")
    run.add_argument("--input", required=True, help="Task A English input file: TSV, CSV, JSON, or JSONL")
    run.add_argument("--output", default="task-a-en.tsv", help="Output TSV path")
    run.add_argument("--backend", choices=["openai", "ollama", "mock"], default="openai")
    run.add_argument("--base-url", default="http://localhost:1234/v1", help="OpenAI-compatible base URL or Ollama URL")
    run.add_argument("--model", default="qwen3-14b", help="Local model name")
    run.add_argument("--variants-per-style", type=int, default=4, help="4 gives 24 candidates total")
    run.add_argument("--rerank-top-k", type=int, default=6)
    run.add_argument("--seed", type=int, default=202601)
    run.add_argument("--timeout", type=int, default=180)
    run.add_argument("--sleep", type=float, default=0.0, help="Optional pause between examples")
    run.add_argument("--limit", type=int, default=0, help="Debug only: process first N examples")
    run.add_argument("--diagnostics-dir", default="", help="Optional directory for per-example candidate dumps")
    run.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing rows in --output and save progress after every generated item.",
    )
    run.add_argument(
        "--humor-model",
        action="append",
        default=[],
        help="Optional Hugging Face humor classifier. Can be repeated for an ensemble.",
    )
    run.add_argument(
        "--humor-weight",
        type=float,
        default=0.25,
        help="Weight of humor classifier score in ranking; 0 disables its influence.",
    )
    run.add_argument(
        "--humor-device",
        type=int,
        default=0,
        help="Transformers pipeline device for humor classifiers: 0 GPU, -1 CPU.",
    )
    run.set_defaults(func=run_pipeline)

    gen_pool = subparsers.add_parser(
        "generate-candidates",
        help="Generate sequential candidate pools with one loaded model.",
    )
    gen_pool.add_argument("--input", required=True, help="Task A English input file: TSV, CSV, JSON, or JSONL")
    gen_pool.add_argument("--pool-dir", required=True, help="Directory where model JSONL candidate files are stored")
    gen_pool.add_argument("--backend", choices=["openai", "ollama", "mock"], default="openai")
    gen_pool.add_argument("--base-url", default="http://localhost:1234/v1", help="OpenAI-compatible base URL or Ollama URL")
    gen_pool.add_argument("--model", default="qwen3-14b", help="Currently loaded local model name")
    gen_pool.add_argument("--model-alias", default="", help="Stable source name stored in the candidate pool")
    gen_pool.add_argument("--variants-per-entry", type=int, default=3, help="Candidates generated per input row for this model")
    gen_pool.add_argument("--seed", type=int, default=202601)
    gen_pool.add_argument("--timeout", type=int, default=180)
    gen_pool.add_argument("--sleep", type=float, default=0.0, help="Optional pause between examples")
    gen_pool.add_argument("--limit", type=int, default=0, help="Debug only: process first N examples")
    gen_pool.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing candidate records for this model alias and append only missing rows.",
    )
    gen_pool.set_defaults(func=cmd_generate_candidates)

    rank_pool = subparsers.add_parser(
        "rank-candidates",
        help="Rank all JSONL candidates in a pool and write a final TSV.",
    )
    rank_pool.add_argument("--input", required=True, help="Task A English input file")
    rank_pool.add_argument("--pool-dir", required=True, help="Directory containing model JSONL candidate files")
    rank_pool.add_argument("--output", default="task-a-en.ensemble.tsv", help="Output TSV path")
    rank_pool.add_argument("--backend", choices=["openai", "ollama", "mock"], default="openai")
    rank_pool.add_argument("--base-url", default="http://localhost:1234/v1", help="Judge base URL")
    rank_pool.add_argument("--model", default="qwen3-14b", help="Judge model name")
    rank_pool.add_argument("--rerank-top-k", type=int, default=6)
    rank_pool.add_argument("--seed", type=int, default=202601)
    rank_pool.add_argument("--timeout", type=int, default=180)
    rank_pool.add_argument("--sleep", type=float, default=0.0, help="Optional pause between examples")
    rank_pool.add_argument("--limit", type=int, default=0, help="Debug only: process first N examples")
    rank_pool.add_argument("--diagnostics-dir", default="", help="Optional directory for per-example ranking dumps")
    rank_pool.add_argument("--resume", action="store_true", help="Reuse existing rows in --output")
    rank_pool.add_argument(
        "--humor-model",
        action="append",
        default=[],
        help="Optional Hugging Face humor classifier. Can be repeated for an ensemble.",
    )
    rank_pool.add_argument(
        "--humor-weight",
        type=float,
        default=0.20,
        help="Weight of humor classifier score in ranking; 0 disables its influence.",
    )
    rank_pool.add_argument(
        "--humor-device",
        type=int,
        default=0,
        help="Transformers pipeline device for humor classifiers: 0 GPU, -1 CPU.",
    )
    rank_pool.set_defaults(func=cmd_rank_candidates)

    validate = subparsers.add_parser("validate", help="Validate an existing task-a-en.tsv")
    validate.add_argument("--input", required=True, help="Task A English input file")
    validate.add_argument("--output", default="task-a-en.tsv", help="Output TSV path")
    validate.set_defaults(func=cmd_validate)

    evaluate = subparsers.add_parser("evaluate", help="Judge final output against official-baseline-style generations")
    evaluate.add_argument("--input", required=True, help="Task A English input file")
    evaluate.add_argument("--output", default="task-a-en.tsv", help="Output TSV path")
    evaluate.add_argument("--backend", choices=["openai", "ollama", "mock"], default="openai")
    evaluate.add_argument("--base-url", default="http://localhost:1234/v1", help="OpenAI-compatible base URL or Ollama URL")
    evaluate.add_argument("--model", default="qwen3-14b", help="Local model name")
    evaluate.add_argument("--timeout", type=int, default=180)
    evaluate.add_argument("--seed", type=int, default=202601)
    evaluate.add_argument("--limit", type=int, default=0, help="Debug only: process first N examples")
    evaluate.add_argument("--min-win-rate", type=float, default=0.55)
    evaluate.set_defaults(func=cmd_evaluate)

    refine = subparsers.add_parser("refine", help="Targeted second-pass refinement of an existing output")
    refine.add_argument("--input", required=True, help="Task A English input file")
    refine.add_argument(
        "--incumbent-output",
        required=True,
        help="Current validated output TSV to use as the incumbent.",
    )
    refine.add_argument(
        "--output",
        default="task-a-en.refined.tsv",
        help="Refined output TSV path. The incumbent file is not overwritten.",
    )
    refine.add_argument("--backend", choices=["openai", "ollama", "mock"], default="openai")
    refine.add_argument("--base-url", default="http://localhost:1234/v1", help="OpenAI-compatible base URL or Ollama URL")
    refine.add_argument("--model", default="qwen3-14b", help="Local model name")
    refine.add_argument("--variants-per-style", type=int, default=2, help="2 gives 12 refinement candidates total")
    refine.add_argument("--rerank-top-k", type=int, default=4)
    refine.add_argument("--seed", type=int, default=202602)
    refine.add_argument("--timeout", type=int, default=180)
    refine.add_argument("--sleep", type=float, default=0.0, help="Optional pause between examples")
    refine.add_argument("--limit", type=int, default=0, help="Debug only: process first N selected targets")
    refine.add_argument("--dry-run", action="store_true", help="List selected targets without generating")
    refine.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing refined output and skip ids with refinement diagnostics.",
    )
    refine.add_argument(
        "--diagnostics-dir",
        default="",
        help="Existing run diagnostics used only for target selection by low score.",
    )
    refine.add_argument(
        "--refine-diagnostics-dir",
        default="",
        help="Optional directory for per-example refinement diagnostics.",
    )
    refine.add_argument(
        "--target-ids",
        default="",
        help="Optional comma/space-separated ids or path to a text file. Overrides automatic target selection.",
    )
    refine.add_argument("--max-targets", type=int, default=120, help="Maximum automatically selected targets; 0 means all")
    refine.add_argument("--low-score-threshold", type=float, default=7.0)
    refine.add_argument(
        "--include-word-inclusion",
        action="store_true",
        help="Allow automatic targeting of word-inclusion rows. Default targets headline rows only.",
    )
    refine.add_argument("--pairwise-votes", type=int, default=3, help="Incumbent-vs-challenger judge votes")
    refine.add_argument(
        "--replace-votes",
        type=int,
        default=2,
        help="Minimum challenger votes required to replace the incumbent.",
    )
    refine.add_argument(
        "--humor-model",
        action="append",
        default=[],
        help="Optional Hugging Face humor classifier. Can be repeated for an ensemble.",
    )
    refine.add_argument(
        "--humor-weight",
        type=float,
        default=0.20,
        help="Weight of humor classifier score in ranking; 0 disables its influence.",
    )
    refine.add_argument(
        "--humor-device",
        type=int,
        default=0,
        help="Transformers pipeline device for humor classifiers: 0 GPU, -1 CPU.",
    )
    refine.add_argument("--report", default="", help="Optional JSON report path")
    refine.set_defaults(func=cmd_refine)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
