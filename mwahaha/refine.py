from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from .clients import LLMClient, make_client
from .generation import generate_refine_candidates, normalize_for_dedupe
from .humor import make_humor_scorer
from .io import read_output_map, read_inputs, safe_filename, write_output
from .ranking import judge_pair, pairwise_tournament, score_candidates, output_style_metrics
from .schema import Candidate, TaskInput
from .validation import clean_text, count_words, validate_candidate, validate_output


REFINE_WEAK_START_PATTERNS = [
    r"^i tried\b",
    r"^i asked\b",
    r"^i told\b",
]
REFINE_WEAK_ANYWHERE_PATTERNS = [
    r"\bturns out\b",
]


def refine_style_penalty(text: str) -> float:
    lowered = text.strip().lower()
    penalty = 0.0
    if any(re.search(pattern, lowered) for pattern in REFINE_WEAK_START_PATTERNS):
        penalty += 2.0
    if any(re.search(pattern, lowered) for pattern in REFINE_WEAK_ANYWHERE_PATTERNS):
        penalty += 1.25
    if '"' in text or "\u201c" in text or "\u201d" in text:
        penalty += 0.6
    words = count_words(text)
    if words > 28:
        penalty += min(1.5, (words - 28) * 0.12)
    if len(text) > 180:
        penalty += 0.5
    if len(text) > 220:
        penalty += 0.8
    generic_markers = [
        "the kind of headline",
        "sounds like the news",
        "my coffee",
        "press secretary",
        "existential crisis",
    ]
    if any(marker in lowered for marker in generic_markers):
        penalty += 1.0
    return penalty


def apply_refine_style_penalties(candidates: list[Candidate]) -> list[Candidate]:
    for candidate in candidates:
        if candidate.valid:
            candidate.score -= refine_style_penalty(candidate.text)
    return sorted(candidates, key=lambda c: (c.valid, c.score), reverse=True)


def parse_target_ids(value: str) -> set[str]:
    if not value:
        return set()
    possible_path = Path(value)
    if possible_path.exists():
        raw = possible_path.read_text(encoding="utf-8-sig")
    else:
        raw = value
    return {part.strip() for part in re.split(r"[\s,;]+", raw) if part.strip()}


def diagnostic_winner_score(diagnostics_dir: Path | None, item_id: str, text: str) -> float | None:
    if diagnostics_dir is None:
        return None
    path = diagnostics_dir / f"{safe_filename(item_id)}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    candidates = data.get("candidates", [])
    winner = normalize_for_dedupe(str(data.get("winner", text)))
    text_key = normalize_for_dedupe(text)
    for candidate in candidates:
        candidate_text = normalize_for_dedupe(str(candidate.get("text", "")))
        if candidate_text in {winner, text_key}:
            try:
                return float(candidate.get("score", 0.0))
            except (TypeError, ValueError):
                return None
    return None


def refine_target_reasons(
    item: TaskInput,
    text: str,
    diagnostics_dir: Path | None,
    low_score_threshold: float,
    include_word_inclusion: bool,
) -> list[str]:
    if item.kind == "word_inclusion" and not include_word_inclusion:
        return []
    reasons: list[str] = []
    valid, reason = validate_candidate(item, text)
    if not valid:
        reasons.append(f"invalid:{reason}")
    lowered = text.strip().lower()
    if any(re.search(pattern, lowered) for pattern in REFINE_WEAK_START_PATTERNS):
        reasons.append("weak_first_person_start")
    if any(re.search(pattern, lowered) for pattern in REFINE_WEAK_ANYWHERE_PATTERNS):
        reasons.append("turns_out")
    if len(text) > 200:
        reasons.append("long_over_200")
    if '"' in text or "\u201c" in text or "\u201d" in text:
        reasons.append("quote_marks")
    score = diagnostic_winner_score(diagnostics_dir, item.id, text)
    if score is not None and score < low_score_threshold:
        reasons.append(f"low_score:{score:.2f}")
    return reasons


def refine_reason_priority(reasons: list[str], text: str) -> tuple[int, int]:
    weights = {
        "invalid": 100,
        "weak_first_person_start": 80,
        "long_over_200": 70,
        "turns_out": 60,
        "low_score": 45,
        "quote_marks": 25,
    }
    total = 0
    for reason in reasons:
        key = reason.split(":", 1)[0]
        total += weights.get(key, 0)
    return total, len(text)


def select_refine_targets(
    inputs: list[TaskInput],
    output_by_id: dict[str, str],
    diagnostics_dir: Path | None,
    explicit_target_ids: set[str],
    max_targets: int,
    low_score_threshold: float,
    include_word_inclusion: bool,
) -> list[tuple[TaskInput, list[str]]]:
    if explicit_target_ids:
        selected = []
        for item in inputs:
            if item.id in explicit_target_ids:
                selected.append((item, ["explicit"]))
        return selected[:max_targets] if max_targets else selected

    targets: list[tuple[TaskInput, list[str]]] = []
    for item in inputs:
        text = output_by_id.get(item.id, "")
        if not text:
            continue
        reasons = refine_target_reasons(item, text, diagnostics_dir, low_score_threshold, include_word_inclusion)
        if reasons:
            targets.append((item, reasons))
    targets.sort(
        key=lambda pair: refine_reason_priority(pair[1], output_by_id.get(pair[0].id, "")),
        reverse=True,
    )
    return targets[:max_targets] if max_targets else targets


def judge_incumbent_vs_challenger(
    client: LLMClient,
    item: TaskInput,
    incumbent: str,
    challenger: str,
    seed: int,
    votes: int,
) -> tuple[int, list[str]]:
    challenger_wins = 0
    raw_votes: list[str] = []
    for vote_index in range(votes):
        winner = judge_pair(client, item, incumbent, challenger, seed + vote_index * 101)
        raw_votes.append(winner)
        if winner == "B":
            challenger_wins += 1
    return challenger_wins, raw_votes


def write_refine_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def read_refine_item_report(refine_diagnostics_dir: Path | None, item_id: str) -> dict[str, Any] | None:
    if refine_diagnostics_dir is None:
        return None
    path = refine_diagnostics_dir / f"{safe_filename(item_id)}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def cmd_refine(args: argparse.Namespace) -> int:
    if args.pairwise_votes < 1:
        print("ERROR: --pairwise-votes must be at least 1", file=sys.stderr)
        return 2
    if args.replace_votes < 1 or args.replace_votes > args.pairwise_votes:
        print("ERROR: --replace-votes must be between 1 and --pairwise-votes", file=sys.stderr)
        return 2
    random.seed(args.seed)
    inputs = read_inputs(Path(args.input))
    incumbent_path = Path(args.incumbent_output)
    output_path = Path(args.output)
    original_incumbent_by_id = read_output_map(incumbent_path)
    output_by_id = dict(original_incumbent_by_id)
    if args.resume and output_path.exists():
        output_by_id.update(read_output_map(output_path))
        print(f"Resume enabled: loaded refined rows from {output_path}.")
    diagnostics_dir = Path(args.diagnostics_dir) if args.diagnostics_dir else None
    refine_diagnostics_dir = Path(args.refine_diagnostics_dir) if args.refine_diagnostics_dir else None
    if refine_diagnostics_dir:
        refine_diagnostics_dir.mkdir(parents=True, exist_ok=True)

    incumbent_errors = validate_output(inputs, incumbent_path)
    if incumbent_errors:
        print(f"WARNING: incumbent has {len(incumbent_errors)} validation issue(s). Refinement will still run.")
        for error in incumbent_errors[:10]:
            print(f"WARNING: {error}")

    explicit_target_ids = parse_target_ids(args.target_ids)
    targets = select_refine_targets(
        inputs,
        output_by_id,
        diagnostics_dir,
        explicit_target_ids,
        args.max_targets,
        args.low_score_threshold,
        args.include_word_inclusion,
    )
    if args.limit:
        targets = targets[: args.limit]

    if args.dry_run:
        print(f"Selected {len(targets)} refinement target(s).")
        for item, reasons in targets:
            text = output_by_id.get(item.id, "")
            print(f"{item.id}\t{item.kind}\t{len(text)} chars\t{','.join(reasons)}")
        metrics = output_style_metrics(inputs, output_by_id)
        print(json.dumps({"current_output_metrics": metrics}, ensure_ascii=False, indent=2))
        return 0

    client = make_client(args)
    humor_scorer = make_humor_scorer(args)
    outputs = dict(output_by_id)
    replacements = 0
    report: dict[str, Any] = {
        "input": str(Path(args.input)),
        "incumbent_output": str(incumbent_path),
        "refined_output": str(output_path),
        "target_count": len(targets),
        "replacement_count": 0,
        "pairwise_votes": args.pairwise_votes,
        "required_challenger_votes": args.replace_votes,
        "items": [],
    }

    print(f"Selected {len(targets)} refinement target(s).", flush=True)
    for index, (item, reasons) in enumerate(targets, start=1):
        incumbent = output_by_id.get(item.id, "")
        if args.resume:
            previous_item_report = read_refine_item_report(refine_diagnostics_dir, item.id)
            if previous_item_report is not None:
                report["items"].append(previous_item_report)
                print(f"[{index}/{len(targets)}] skipping {item.id} (already refined)", flush=True)
                continue
        print(f"[{index}/{len(targets)}] refining {item.id}: {', '.join(reasons)}", flush=True)
        candidates = generate_refine_candidates(
            client,
            item,
            incumbent,
            args.variants_per_style,
            args.seed + index * 5000,
        )
        ranked = score_candidates(client, item, candidates, humor_scorer, args.humor_weight)
        ranked = apply_refine_style_penalties(ranked)
        challenger = pairwise_tournament(client, item, ranked, args.rerank_top_k)
        challenger_text = clean_text(challenger.text)
        valid, invalid_reason = validate_candidate(item, challenger_text)
        challenger_votes = 0
        vote_trace: list[str] = []
        replaced = False
        if valid:
            challenger_votes, vote_trace = judge_incumbent_vs_challenger(
                client,
                item,
                incumbent,
                challenger_text,
                args.seed + index * 7000,
                args.pairwise_votes,
            )
            replaced = challenger_votes >= args.replace_votes
        if replaced:
            outputs[item.id] = challenger_text
            replacements += 1
        item_report = {
            "id": item.id,
            "kind": item.kind,
            "reasons": reasons,
            "incumbent": incumbent,
            "challenger": challenger_text,
            "challenger_valid": valid,
            "invalid_reason": invalid_reason,
            "challenger_votes": challenger_votes,
            "vote_trace": vote_trace,
            "replaced": replaced,
            "top_candidates": [candidate.__dict__ for candidate in ranked[: args.rerank_top_k]],
        }
        report["items"].append(item_report)
        if refine_diagnostics_dir:
            write_refine_report(refine_diagnostics_dir / f"{safe_filename(item.id)}.json", item_report)
        write_output(output_path, [(item.id, outputs[item.id]) for item in inputs if item.id in outputs])
        report["replacement_count"] = replacements
        report["output_metrics"] = output_style_metrics(inputs, outputs)
        if args.report:
            write_refine_report(Path(args.report), report)
        if args.sleep:
            time.sleep(args.sleep)

    rows = [(item.id, outputs[item.id]) for item in inputs if item.id in outputs]
    write_output(output_path, rows)
    errors = validate_output(inputs, output_path)
    report["replacement_count"] = replacements
    final_output_by_id = read_output_map(output_path)
    report["total_changed_from_incumbent"] = sum(
        1 for item in inputs if final_output_by_id.get(item.id) != original_incumbent_by_id.get(item.id)
    )
    report["validation_errors"] = errors
    report["output_metrics"] = output_style_metrics(inputs, final_output_by_id)
    if args.report:
        write_refine_report(Path(args.report), report)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Wrote {output_path} with {len(rows)} rows; replacements={replacements}/{len(targets)}.")
    return 0
