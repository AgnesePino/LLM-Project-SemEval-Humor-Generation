#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from humor_gen.generate import generate_dataset
from humor_gen.rag import build_retriever
from humor_gen.utils import check_output_path, load_yaml, setup_logging, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RAG-enhanced jokes with one configured model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rag-config", default="configs/rag.yaml")
    parser.add_argument("--generation-config", default=None)
    parser.add_argument("--models-config", default=None)
    parser.add_argument("--k", type=int, default=None, help="Number of retrieved contexts, usually 2 or 4.")
    parser.add_argument("--n-docs", type=int, default=None, help="Number of HF Wikipedia documents to load.")
    parser.add_argument("--apply-to", choices=["all", "headline", "word_pair"], default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of examples to process.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    rag_cfg = load_yaml(args.rag_config)
    if args.n_docs is not None:
        rag_cfg.setdefault("retriever", {})["n_docs"] = args.n_docs
    generation_cfg = load_yaml(args.generation_config or rag_cfg.get("generation_config", "configs/generation.yaml"))
    models_config_path = args.models_config or rag_cfg.get("models_config", "configs/models.yaml")
    k = args.k if args.k is not None else rag_cfg.get("retriever", {}).get("default_k", 2)
    allowed_k = rag_cfg.get("retriever", {}).get("allowed_k", [2, 4])
    if k not in allowed_k:
        raise ValueError(f"k={k} is not allowed by config. Allowed values: {allowed_k}")
    apply_to = args.apply_to or rag_cfg.get("retriever", {}).get("apply_to", "all")
    check_output_path(args.output, overwrite=args.overwrite)
    retriever = build_retriever(args.input, rag_cfg, mock=args.mock)
    rows = generate_dataset(
        model_key=args.model,
        input_path=args.input,
        generation_cfg=generation_cfg,
        models_config_path=models_config_path,
        mock=args.mock,
        method="rag",
        retriever=retriever,
        k=k,
        rag_apply_to=apply_to,
        limit=args.limit,
    )
    write_jsonl(rows, args.output, overwrite=True)


if __name__ == "__main__":
    main()
