# SemEval 2026 - Task 1: MWAHAHA, Humor Generation

This repository contains our local pipeline for **SemEval 2026 Task 1: MWAHAHA**, specifically **Task A English**, which focuses on constrained humor generation.

The task has two input types:

- a **news headline**, where the joke must stay related to the headline;
- a **pair of required words**, where both words must appear verbatim in the generated joke.

The system follows a **generation-and-selection approach**: it generates several candidate jokes for each input, validates the task constraints, scores valid candidates, and selects the final joke through reranking.

## Repository Overview

The command-line entrypoint is:

```text
mwahaha_task_a_en.py
```

This file is a small compatibility wrapper. The implementation lives in the `mwahaha/` package, and the CLI commands are defined in `mwahaha/cli.py`.

The main commands are used to:

- generate joke candidates;
- validate generated outputs;
- rank and rerank candidates;
- optionally refine weak or repetitive jokes;
- prepare a CodaBench-compatible submission file.

The project can be used in two main configurations:

1. **Single-model pipeline**  
   One local LLM generates multiple candidates for each input.

2. **Ensemble pipeline**  
   Three different LLMs generate candidates separately. Their outputs are merged, validated, scored, and reranked together.

## Main Features

- Local execution with open-source LLMs.
- Support for headline-based and word-inclusion inputs.
- Single-model generation for baseline runs.
- Sequential multi-model ensemble generation for more diverse candidates.
- Constraint validation for task rules.
- LLM-based scoring and pairwise reranking.
- Optional humor classifiers as an auxiliary scoring signal.
- Optional targeted refinement for future cleanup passes.

## Project Workflow

The general workflow is:

```text
Input file
   ↓
Candidate generation
   ├── single-model generation
   └── ensemble generation
   ↓
Constraint validation
   ↓
Candidate scoring
   ↓
Pairwise reranking
   ↓
Final output TSV
```

In practice, the pipeline:

1. loads the Task A English input file;
2. generates multiple joke candidates for each input;
3. validates each candidate against the task constraints;
4. scores valid candidates using an LLM judge and optional humor classifiers;
5. compares the best candidates using pairwise reranking;
6. writes one final joke for each input row.

## Current Outputs

The repository currently contains:

```text
submission/task-a-en.single_ministral.tsv
submission/task-a-en.ensemble.tsv
```

The ensemble output is the main submission candidate. The single-model output is kept as a baseline for comparison.

For CodaBench, the selected TSV must be packaged under the name `task-a-en.tsv`:

```powershell
Copy-Item .\submission\task-a-en.ensemble.tsv .\submission\task-a-en.tsv -Force
Compress-Archive -Path .\submission\task-a-en.tsv -DestinationPath .\submission.zip -Force
```

## Setup

### Requirements

The project requires:

- Python 3.10+
- LM Studio with the local server enabled

Recommended hardware for the local runs used in this project:

```text
NVIDIA RTX 4070 12GB VRAM
32GB RAM
```

The core CLI uses the Python standard library for local API calls. Optional humor classifiers require `torch` and `transformers`.

### Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Model Setup

The pipeline uses local open-source LLMs served through an API.

### LM Studio

In LM Studio, load the model you want to use and start the local OpenAI-compatible server at:

```text
http://localhost:1234/v1
```

Then check the available model name:

```powershell
Invoke-WebRequest -Uri http://localhost:1234/v1/models -UseBasicParsing
```

Use the returned model name with the `--model` argument.

## Running the Pipeline

### Single-Model Run

The single-model setup uses one LLM to generate all candidates. In this project, the single-model baseline uses **Ministral 3 14B Reasoning**.

```powershell
python .\mwahaha_task_a_en.py run `
  --input .\data\input\task-a-en.tsv `
  --output .\submission\task-a-en.single_ministral.tsv `
  --backend openai `
  --base-url http://localhost:1234/v1 `
  --model mistralai/ministral-3-14b-reasoning `
  --variants-per-style 2 `
  --rerank-top-k 3 `
  --timeout 300 `
  --diagnostics-dir .\diagnostics_single_ministral `
  --resume
```

This command reads the input file, generates candidates, validates them, scores and reranks them, and writes the final baseline output.

### Ensemble Run

The ensemble configuration generates a larger candidate pool with multiple models. The models are run sequentially because they may not fit in memory at the same time.

Example with Qwen:

```powershell
python .\mwahaha_task_a_en.py generate-candidates `
  --input .\data\input\task-a-en.tsv `
  --pool-dir .\candidate_pools\ensemble_v1 `
  --backend openai `
  --base-url http://localhost:1234/v1 `
  --model qwen/qwen3-14b `
  --model-alias qwen3-14b `
  --variants-per-entry 3 `
  --timeout 300 `
  --resume
```

Repeat the same command for the other ensemble models:

```text
google/gemma-4-12b-qat              -> gemma-12b-qat
mistralai/ministral-3-14b-reasoning -> ministral-3-14b-reasoning
```

After all candidate pools are generated, reload Qwen and run the global ranking step:

```powershell
python .\mwahaha_task_a_en.py rank-candidates `
  --input .\data\input\task-a-en.tsv `
  --pool-dir .\candidate_pools\ensemble_v1 `
  --output .\submission\task-a-en.ensemble.tsv `
  --backend openai `
  --base-url http://localhost:1234/v1 `
  --model qwen/qwen3-14b `
  --rerank-top-k 6 `
  --timeout 300 `
  --diagnostics-dir .\diagnostics_ensemble `
  --humor-model Humor-Research/humor-detection-comb-23 `
  --humor-model mohameddhiab/humor-no-humor `
  --humor-weight 0.20 `
  --humor-device -1 `
  --resume
```

The notebook [`notebooks/mwahaha_ensemble_runs.ipynb`](notebooks/mwahaha_ensemble_runs.ipynb) contains the same ensemble workflow split into cells.

## Validation

The validation command checks whether a generated TSV satisfies the formal task requirements.

```powershell
python .\mwahaha_task_a_en.py validate `
  --input .\data\input\task-a-en.tsv `
  --output .\submission\task-a-en.ensemble.tsv
```

The validator checks that:

- the file has the correct header;
- all expected IDs are present;
- there are no duplicated or extra IDs;
- the text field is not empty;
- there are no tabs or line breaks inside the text field;
- the output is not too long;
- the joke does not contain boilerplate such as `Joke:` or `Here is a joke:`;
- required words are included verbatim for word-inclusion examples;
- headline-based jokes keep a minimum lexical connection to the headline.

## Reranking

The final selection does not use BLEU, ROUGE, or text similarity metrics, because there is no single correct joke for each input. Instead, the system approximates human preference using an LLM judge.

The reranking process has three steps:

```text
Validation
   ↓
Numerical scoring
   ↓
Pairwise tournament
```

First, invalid candidates are removed or pushed to the bottom. Then valid candidates are scored by an LLM judge. If enabled, humor classifiers provide an auxiliary score. Finally, the best candidates are compared in a small pairwise tournament, and the winner becomes the final joke for that input.

More details are available in [`RERANKING.md`](RERANKING.md).

## Optional Refinement

The CLI includes a `refine` command for targeted second-pass cleanup of an existing output. It can target jokes that are weak, too long, repetitive, or stylistically risky.

This repository does not currently include a refined output artifact. The command is kept as an optional follow-up tool.

Example:

```powershell
python .\mwahaha_task_a_en.py refine `
  --input .\data\input\task-a-en.tsv `
  --incumbent-output .\submission\task-a-en.ensemble.tsv `
  --output .\submission\task-a-en.refined.tsv `
  --backend openai `
  --base-url http://localhost:1234/v1 `
  --model mistralai/ministral-3-14b-reasoning `
  --variants-per-style 2 `
  --rerank-top-k 4 `
  --diagnostics-dir .\diagnostics_ensemble `
  --refine-diagnostics-dir .\refine_diagnostics `
  --report .\refine_report.json `
  --resume
```

A new candidate replaces the existing joke only if it wins a conservative pairwise comparison.

## Output Format

The final submission must be a TSV file with this format:

```text
id	text
en_2001	...
en_2002	...
```

For CodaBench, the file inside the final ZIP archive must be named:

```text
task-a-en.tsv
```

## File Structure

```text
.
├── mwahaha_task_a_en.py              # CLI compatibility wrapper
├── mwahaha/                          # Pipeline implementation
├── requirements.txt                  # Python dependencies
├── README.md                         # Main project guide
├── RERANKING.md                      # Validation, scoring, and reranking details
├── REPORT_ENSEMBLE.md                # Report about the ensemble run
├── README_DIAGNOSTICS_COMPARISON.md  # Single-model vs ensemble comparison
├── data/
│   ├── input/
│   │   └── task-a-en.tsv             # Task A English input file
│   └── baseline/
│       └── task-a-en-baseline.tsv    # Baseline file for local analysis
├── submission/
│   ├── task-a-en.single_ministral.tsv
│   └── task-a-en.ensemble.tsv
├── candidate_pools/
│   └── ensemble_v1/                  # Candidate pools generated by the ensemble models
├── diagnostics_ensemble/
│   └── ...                           # Diagnostics for the ensemble run
├── diagnostics_single_ministral/
│   └── ...                           # Diagnostics for the single-model run
└── notebooks/
    └── mwahaha_ensemble_runs.ipynb   # Operational ensemble notebook
```

Generated packaging/refinement files such as `submission/task-a-en.tsv`, `submission.zip`, `submission/task-a-en.refined.tsv`, `refine_report.json`, and `refine_diagnostics/` are created only if the corresponding packaging or refinement commands are run.

## Main Files

### `mwahaha_task_a_en.py`

Compatibility wrapper for the command-line interface.

### `mwahaha/`

Python package containing the clients, prompts, validation, generation, ranking, diagnostics I/O, and optional refinement logic.

### `RERANKING.md`

Explains how candidates are validated, scored, sorted, and compared through pairwise reranking.

### `REPORT_ENSEMBLE.md`

Contains the experimental report for the ensemble run, including model win shares, candidate validity, output style metrics, and refinement suggestions.

### `README_DIAGNOSTICS_COMPARISON.md`

Compares the single-model Ministral run with the multi-model ensemble and highlights the trade-off between compactness and judged quality.

## Models

The ensemble configuration uses:

- Qwen3-14B
- Gemma 12B QAT
- Ministral 3 14B Reasoning

The single-model baseline uses **Ministral 3 14B Reasoning**. This is the 14B model used in our experiments and the strongest single contributor inside the ensemble.

Optional humor classifiers can be used as an auxiliary signal, but they do not decide the final output alone.

## Acknowledgments

- SemEval 2026 Task 1: MWAHAHA organizers.
- Hugging Face for open-source models and humor classifiers.
- LM Studio for local LLM inference.
- The open-source LLM community.
