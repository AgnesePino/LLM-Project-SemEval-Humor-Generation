# SemEval 2026 - Task 1: MWAHAHA, Humor Generation

This repository contains our local pipeline for **SemEval 2026 Task 1: MWAHAHA**, specifically **Task A English**, which focuses on constrained humor generation.

The goal of the task is to generate a funny text starting from one of two possible inputs:

- a **news headline**, where the joke must remain related to the headline;
- a **pair of required words**, where both words must appear verbatim in the generated joke.

The system follows a **generation-and-selection approach**. Instead of producing only one joke directly, it generates several candidate jokes for each input, checks whether they satisfy the task constraints, scores them, and finally selects the best one through reranking.

## Repository Overview

The repository is organized around one main script:

```text
mwahaha_task_a_en.py
```

This script contains the main commands used to:

- generate joke candidates;
- validate generated outputs;
- rank and rerank candidates;
- refine weak or repetitive jokes;
- create a final submission file.

The project can be used in two main configurations:

1. **Single-model pipeline**  
   One local LLM generates multiple candidates for each input.

2. **Ensemble pipeline**  
   Three different LLMs generate candidates separately. Their outputs are then merged, validated, scored, and reranked together.

After candidate generation, both configurations use the same validation, scoring, and reranking pipeline.

## Main Features

- **Local execution** with open-source LLMs.
- **Support for headline-based and word-inclusion inputs**.
- **Single-model generation** for simpler runs.
- **Sequential multi-model ensemble generation** for more diverse candidates.
- **Constraint validation** to ensure outputs follow the task rules.
- **LLM-based scoring** to estimate candidate quality.
- **Optional humor classifiers** as an auxiliary scoring signal.
- **Pairwise reranking** to select the final joke.
- **Targeted refinement** to improve weak or repetitive outputs.
- **CodaBench-ready submission packaging**.

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
Optional refinement
   ↓
Final submission
```

In practice, the pipeline works as follows:

1. Load the Task A English input file.
2. Generate multiple joke candidates for each input.
3. Validate each candidate against the task constraints.
4. Score valid candidates using an LLM judge and optional humor classifiers.
5. Compare the best candidates using pairwise reranking.
6. Select one final joke for each input.
7. Save the final `task-a-en.tsv` file.
8. Package the file into `submission.zip`.

## Setup

### Requirements

The project requires:

- Python 3.10+
- A local LLM server, such as:
  - LM Studio
  - llama.cpp server
  - Ollama

Recommended hardware:

```text
NVIDIA RTX 4070 12GB VRAM
32GB RAM
```

The basic script can run with the Python standard library. Optional humor classifiers require additional libraries such as `torch` and `transformers`.

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

### LM Studio or llama.cpp

Start an OpenAI-compatible server at:

```text
http://localhost:1234/v1
```

Then check the available model name:

```powershell
Invoke-WebRequest -Uri http://localhost:1234/v1/models -UseBasicParsing
```

Use the returned model name with the `--model` argument.

### Ollama

The pipeline can also run through Ollama, usually at:

```text
http://localhost:11434
```

Example:

```powershell
python .\mwahaha_task_a_en.py run `
  --input .\data\input\task-a-en.tsv `
  --output .\submission\task-a-en.tsv `
  --backend ollama `
  --base-url http://localhost:11434 `
  --model qwen3:14b
```

## Running the Pipeline

### Single-Model Run

The single-model configuration uses one LLM to generate all candidates. In this project, the single-model setup uses **Ministral 3 14B Reasoning**.

```powershell
python .\mwahaha_task_a_en.py run `
  --input .\data\input\task-a-en.tsv `
  --output .\submission\task-a-en.tsv `
  --backend openai `
  --base-url http://localhost:1234/v1 `
  --model mistralai/ministral-3-14b-reasoning `
  --variants-per-style 2 `
  --rerank-top-k 3 `
  --timeout 300 `
  --diagnostics-dir .\diagnostics `
  --resume
```

This command:

- reads the input file;
- generates multiple jokes per row;
- validates the candidates;
- scores and reranks them;
- writes the final output file.

### Ensemble Run

The ensemble configuration generates a larger candidate pool using multiple models.

The models are run sequentially because they may not fit in memory at the same time. Each model writes its candidates to disk, and the final ranking step merges and reranks all candidates.

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
  --resume
```

The same command can be repeated with other models by changing `--model` and `--model-alias`.

After all candidate pools are generated, run the global ranking step:

```powershell
python .\mwahaha_task_a_en.py rank-candidates `
  --input .\data\input\task-a-en.tsv `
  --pool-dir .\candidate_pools\ensemble_v1 `
  --output .\submission\task-a-en.ensemble.tsv `
  --backend openai `
  --base-url http://localhost:1234/v1 `
  --model qwen/qwen3-14b `
  --rerank-top-k 6 `
  --diagnostics-dir .\diagnostics_ensemble `
  --humor-weight 0.20 `
  --resume
```

## Validation

The validation command checks whether a generated submission satisfies the formal task requirements.

```powershell
python .\mwahaha_task_a_en.py validate `
  --input .\data\input\task-a-en.tsv `
  --output .\submission\task-a-en.tsv
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

The final selection does not use BLEU, ROUGE, or text similarity metrics.

This is because humor is subjective and there is no single correct joke for each input. Instead, the system approximates human preference using an LLM judge.

The reranking process has three steps:

```text
Validation
   ↓
Numerical scoring
   ↓
Pairwise tournament
```

First, invalid candidates are removed or pushed to the bottom. Then valid candidates are scored by an LLM judge. If enabled, humor classifiers provide an additional auxiliary score. Finally, the best candidates are compared in a small pairwise tournament, and the winner becomes the final joke for that input.

More details are available in [`RERANKING.md`](RERANKING.md).

## Refinement

The refinement step is used after a valid submission has already been created.

It does not regenerate the whole submission. Instead, it targets jokes that may be weak, too long, repetitive, or stylistically risky.

Example:

```powershell
python .\mwahaha_task_a_en.py refine `
  --input .\data\input\task-a-en.tsv `
  --incumbent-output .\submission\task-a-en.tsv `
  --output .\submission\task-a-en.refined.tsv `
  --backend openai `
  --base-url http://localhost:1234/v1 `
  --model mistralai/ministral-3-14b-reasoning `
  --variants-per-style 2 `
  --rerank-top-k 4 `
  --diagnostics-dir .\diagnostics `
  --refine-diagnostics-dir .\refine_diagnostics `
  --report .\refine_report.json `
  --resume
```

The refinement step is useful for outputs with:

- repeated openings such as `I tried`, `I asked`, or `I told`;
- frequent use of `turns out`;
- excessive length;
- unnecessary quotation marks;
- low diagnostic scores;
- invalid or weakly related text.

A new candidate replaces the existing joke only if it wins a conservative pairwise comparison.

## Output Format

The final submission must be a TSV file with this format:

```text
id	text
en_2001	...
en_2002	...
```

The file must be named:

```text
task-a-en.tsv
```

The final ZIP archive must contain only this file:

```text
task-a-en.tsv
```

To create the ZIP:

```powershell
Compress-Archive -Path .\submission\task-a-en.tsv -DestinationPath .\submission.zip -Force
```

## File Structure

```text
.
├── mwahaha_task_a_en.py              # Main script for generation, validation, evaluation, refinement, and ranking
├── requirements.txt                  # Python dependencies
├── README.md                         # Main project guide
├── RERANKING.md                      # Technical explanation of validation, scoring, and reranking
├── REPORT_ENSEMBLE.md                # Report about the ensemble run
├── DIAGNOSTICS_COMPARISON.md         # Comparison between single-model and ensemble runs
├── refine_report.json                # Report generated by the refinement step
├── data/
│   ├── input/
│   │   └── task-a-en.tsv             # Task A English input file
│   └── baseline/
│       └── task-a-en-baseline.tsv    # Baseline file for local analysis
├── submission/
│   ├── task-a-en.tsv                 # Final submission file
│   ├── task-a-en.refined.tsv         # Refined submission, if generated
│   └── task-a-en.ensemble.tsv        # Ensemble submission, if generated
├── candidate_pools/
│   └── ensemble_v1/                  # Candidate pools generated by different models
├── diagnostics/
│   └── ...                           # Diagnostics for the main run
├── diagnostics_ensemble/
│   └── ...                           # Diagnostics for the ensemble run
├── diagnostics_single_ministral/
│   └── ...                           # Diagnostics for the single-model run
├── refine_diagnostics/
│   └── ...                           # Diagnostics for refinement
└── submission.zip                    # Final archive for CodaBench
```

The `.venv/` folder is not included in this structure because it is a local virtual environment and should not be versioned.

## Main Files

### `mwahaha_task_a_en.py`

Main script of the project. It contains all commands used to generate, validate, rank, evaluate, and refine the submission.

### `RERANKING.md`

Explains how candidates are validated, scored, sorted, and compared through pairwise reranking.

### `REPORT_ENSEMBLE.md`

Contains the experimental report for the ensemble run, including model win shares, candidate validity, output style metrics, and refinement suggestions.

### `DIAGNOSTICS_COMPARISON.md`

Compares the single-model Ministral run with the multi-model ensemble. It highlights the trade-off between compactness and judged quality.

## Models

The project uses local instruction-tuned LLMs.

The ensemble configuration supports:

- Qwen3-14B
- Gemma 12B QAT
- Ministral 3 14B Reasoning

The single-model configuration uses **Ministral 3 14B Reasoning** to generate all candidates. This is the 14B model used in our experiments, selected because it provided the strongest contribution inside the ensemble.

Optional humor classifiers can be used as an auxiliary signal, but they do not decide the final output alone.

## Notes

This project is designed to be local and reproducible. Generated files such as candidate pools, diagnostics, refinement outputs, and submissions can become large, so they should usually be excluded from version control unless they are needed for reporting.

Recommended files to keep in the repository:

```text
README.md
RERANKING.md
REPORT_ENSEMBLE.md
DIAGNOSTICS_COMPARISON.md
mwahaha_task_a_en.py
requirements.txt
data/
```

Recommended files or folders to ignore:

```text
.venv/
__pycache__/
candidate_pools/
diagnostics/
diagnostics_ensemble/
diagnostics_single_ministral/
refine_diagnostics/
submission.zip
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- SemEval 2026 Task 1: MWAHAHA organizers.
- Hugging Face for open-source models and humor classifiers.
- LM Studio, llama.cpp, and Ollama for local LLM inference.
- The open-source LLM community.
