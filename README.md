# Humor Generation – SemEval 2026 Task 1 (MWAHAHA)

**LLM for Software Engineering – Course Project (2025/2026)**  
*Politecnico di Torino*

## Overview

This repository implements a humor generation system for SemEval 2026 – Task 1 (MWAHAHA), Subtask A.
It generates creative English jokes under textual constraints and improves them through retrieval,
best-of-N selection with an LLM judge, and optional DPO preference fine-tuning.

The project runs locally from command-line scripts inside the repository `.venv`. Real model runs
require a CUDA-capable NVIDIA GPU; deterministic `--mock` runs can be used to test the pipeline
without loading a model.

## Task Description

Both supported input types produce one joke or punchline:

1. **Word Inclusion** – given two words, generate a joke that naturally includes both.
2. **News Title Humor** – given a news headline, generate a related punchline.

The official task supports English, Spanish, and Chinese; this project works in **English only**.

## Local setup

Prerequisites:

- Linux with Python 3.12 and `venv` support;
- for real model execution, a recent NVIDIA GPU, compatible driver, and CUDA runtime;
- enough disk space for Python packages, Hugging Face models, datasets, and checkpoints.

Create the project environment and install runtime dependencies:

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
```

For development tools and tests, use:

```bash
./scripts/setup_venv.sh --dev
source .venv/bin/activate
```

The setup script uses `python3` by default. To select another Python 3.12 executable:

```bash
PYTHON=python3.12 ./scripts/setup_venv.sh
```

All commands below must be run from the repository root with `.venv` active. The CLI scripts reject
execution from a different Python environment to prevent accidental dependency mismatches.

For gated Hugging Face models, accept the model licence and expose the token only in the local shell:

```bash
export HF_TOKEN="your_token_here"
```

Do not commit tokens or `.env` files.

## CLI workflow

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `run_generate.py` | Baseline generation and validation |
| 2 | `run_rag_generate.py` | Retrieval-augmented generation |
| 3 | `run_best_of_n.py` | Generate and rerank multiple candidates |
| 4 | `run_judge.py` | Blind model tournament with an LLM judge |
| 5 | `build_preferences.py` | Build DPO preference pairs from judgments |
| 6 | `run_dpo.py` | Optional DPO/LoRA training |
| 7 | `analyze_results.py` | Create CSV summaries and figures |

### Baseline generation

```bash
python scripts/run_generate.py --model llama \
  --input data/raw/task-a-en.tsv \
  --output data/generated/baseline/llama_base.jsonl --overwrite
```

Add `--mock` for a deterministic run that does not load the real model. `--limit N` processes only
the first `N` inputs.

### RAG and best-of-N generation

```bash
python scripts/run_rag_generate.py --model llama \
  --input data/raw/task-a-en.tsv \
  --output data/generated/rag/llama_rag.jsonl \
  --rag-config configs/rag.yaml --k 4 --apply-to headline --overwrite
```

Best-of-N is controlled by the `selection` block in `configs/rag.yaml`. When enabled, each output row
stores the sampled candidates and judge decision in `metadata.candidates` and `metadata.reranker`.

It can also be run without retrieval:

```bash
python scripts/run_best_of_n.py --model llama \
  --input data/raw/task-a-en.tsv --n 5 \
  --output data/generated/llama_best_of_n.jsonl --overwrite
```

### Blind tournament and preference data

```bash
python scripts/run_judge.py --input-dir data/generated/baseline \
  --output data/judged/base_judgments.jsonl --method base --overwrite

python scripts/build_preferences.py \
  --judgments data/judged/base_judgments.jsonl \
  --outputs-dir data/generated \
  --output data/final/preferences.jsonl --overwrite
```

The judge rebuilds items from generated outputs when `--dataset` is omitted, keeping example IDs
aligned with the compared jokes.

### Optional DPO/LoRA training

```bash
python scripts/run_dpo.py --config configs/dpo.yaml
```

Training uses the preferences and output directory configured in `configs/dpo.yaml`. It requires a
working local CUDA environment and sufficient GPU memory.

### Results analysis

```bash
python scripts/analyze_results.py \
  --generated-dir data/generated \
  --judged-dir data/judged \
  --output-dir reports/figures
```

## Mock smoke test

The execution path can be checked without a GPU:

```bash
python scripts/run_generate.py --model llama \
  --input data/raw/sample_task_a.tsv \
  --output /tmp/llama_base_demo.jsonl --mock --overwrite

python scripts/run_best_of_n.py --model llama \
  --input data/raw/sample_task_a.tsv --n 5 \
  --output /tmp/llama_best_of_n.jsonl --mock --overwrite
```

## Tests

After installing development dependencies:

```bash
PYTHONPATH=src python -m pytest -q
python -m ruff check src scripts tests
```

## Repository structure

```text
configs/                 # YAML configs: models, generation, RAG, DPO
data/
  raw/                   # Official-style TSV and small JSONL samples
  processed/             # Optional RAG corpus
  generated/             # Baseline, RAG and best-of-N outputs
  judged/                # Blind tournament judgments
  final/                 # DPO preference dataset
scripts/                 # Local CLI entry points and .venv bootstrap
src/humor_gen/           # Reusable pipeline library
reports/                 # Final report and generated figures
tests/                   # Unit tests
requirements.txt         # Runtime and GPU/model dependencies
requirements-dev.txt     # Runtime plus test and lint tools
```

## Model presets

Model presets are defined in `configs/models.yaml`:

| Preset | Model | Notes |
|--------|-------|-------|
| `llama` | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | 4-bit; may require `HF_TOKEN` |
| `qwen` | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | 4-bit |
| `mistral` | `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` | 4-bit |

## Data policy

Do not commit official CodaBench datasets or full prediction dumps. The repository keeps small
samples, code, and a few illustrative generated files. Generated samples marked `"mock": true` are
deterministic fixtures used to demonstrate output structure.
