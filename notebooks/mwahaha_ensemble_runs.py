# %% [markdown]
# # MWAHAHA Task A EN - Ensemble Runs
#
# ## 0. Configuration
#
# Set project paths, LM Studio connection details, model aliases, and generation/ranking constants.

# %%
from pathlib import Path
import json
import re
import subprocess
from collections import Counter
from tqdm.auto import tqdm

ROOT = Path(r"E:/LLM")
INPUT = ROOT / "data" / "input" / "task-a-en.tsv"
POOL_DIR = ROOT / "candidate_pools" / "ensemble_v1"
ENSEMBLE_OUTPUT = ROOT / "submission" / "task-a-en.ensemble.tsv"
ENSEMBLE_DIAG = ROOT / "diagnostics_ensemble"
BASE_URL = "http://localhost:1234/v1"
PYTHON = "python"
SCRIPT = ROOT / "mwahaha_task_a_en.py"

MODELS = [
    {"alias": "qwen3-14b", "model": "qwen/qwen3-14b"},
    {"alias": "gemma-12b-qat", "model": "google/gemma-4-12b-qat"},
    {"alias": "ministral-3-14b-reasoning", "model": "mistralai/ministral-3-14b-reasoning"},
]

VARIANTS_PER_ENTRY = 3
TIMEOUT = 300
RERANK_TOP_K = 6

# %% [markdown]
# ## 0. Helpers And Input Count
#
# Define progress-aware command helpers, count the input rows, and reuse existing outputs when `--resume` is enabled.

# %%
def count_input_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as f:
        return max(0, sum(1 for _ in f) - 1)


def run_with_progress(cmd, total=None, label="run", verbose=False):
    print(" ".join(map(str, cmd)))
    pattern = re.compile(r"^\[(\d+)/(\d+)\]\s*(.*)$")
    proc = subprocess.Popen(
        [str(x) for x in cmd],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    bar = tqdm(total=total, desc=label, dynamic_ncols=True)
    recent_lines = []

    for line in proc.stdout:
        line = line.rstrip()
        recent_lines.append(line)
        recent_lines = recent_lines[-30:]

        match = pattern.match(line)
        if match:
            current = int(match.group(1))
            if total is None and bar.total is None:
                bar.total = int(match.group(2))
            if current > bar.n:
                bar.update(current - bar.n)
            detail = match.group(3).strip()
            if detail:
                bar.set_postfix_str(detail[:80])
            if verbose:
                tqdm.write(line)
        elif verbose:
            tqdm.write(line)

    code = proc.wait()
    bar.close()

    if code != 0:
        print("\nLast process output lines:")
        for line in recent_lines:
            print(line)
        raise RuntimeError(f"Command failed with exit code {code}")

    return recent_lines


def pool_line_count(pool_dir=POOL_DIR):
    total = 0
    by_file = {}
    if not pool_dir.exists():
        return total, by_file
    for path in sorted(pool_dir.glob("*.jsonl")):
        count = sum(1 for line in path.open("r", encoding="utf-8-sig") if line.strip())
        by_file[path.name] = count
        total += count
    return total, by_file


TOTAL_ROWS = count_input_rows(INPUT)
TOTAL_ROWS

# %% [markdown]
# ## 1. Generate Qwen Candidates
#
# Load Qwen in LM Studio, then write or resume `qwen3-14b.jsonl` with three candidates per input row.

# %%
m = MODELS[0]
cmd = [
    PYTHON, SCRIPT, "generate-candidates",
    "--input", INPUT,
    "--pool-dir", POOL_DIR,
    "--backend", "openai",
    "--base-url", BASE_URL,
    "--model", m["model"],
    "--model-alias", m["alias"],
    "--variants-per-entry", VARIANTS_PER_ENTRY,
    "--timeout", TIMEOUT,
    "--resume",
]
run_with_progress(cmd, total=TOTAL_ROWS, label=f"generate {m['alias']}")
pool_line_count()

# %% [markdown]
# ## 2. Generate Gemma Candidates
#
# Load Gemma in LM Studio, then write or resume `gemma-12b-qat.jsonl`; update `MODELS[1]["model"]` if LM Studio exposes a different name.

# %%
m = MODELS[1]
cmd = [
    PYTHON, SCRIPT, "generate-candidates",
    "--input", INPUT,
    "--pool-dir", POOL_DIR,
    "--backend", "openai",
    "--base-url", BASE_URL,
    "--model", m["model"],
    "--model-alias", m["alias"],
    "--variants-per-entry", VARIANTS_PER_ENTRY,
    "--timeout", TIMEOUT,
    "--resume",
]
run_with_progress(cmd, total=TOTAL_ROWS, label=f"generate {m['alias']}")
pool_line_count()

# %% [markdown]
# ## 3. Generate Ministral Candidates
#
# Load Ministral in LM Studio, then write or resume `ministral-3-14b-reasoning.jsonl`; update `MODELS[2]["model"]` if needed.

# %%
m = MODELS[2]
cmd = [
    PYTHON, SCRIPT, "generate-candidates",
    "--input", INPUT,
    "--pool-dir", POOL_DIR,
    "--backend", "openai",
    "--base-url", BASE_URL,
    "--model", m["model"],
    "--model-alias", m["alias"],
    "--variants-per-entry", VARIANTS_PER_ENTRY,
    "--timeout", TIMEOUT,
    "--resume",
]
run_with_progress(cmd, total=TOTAL_ROWS, label=f"generate {m['alias']}")
pool_line_count()

# %% [markdown]
# ## 4. Rank The Ensemble With Qwen
#
# Reload Qwen in LM Studio, then merge all candidate pools, score them, run pairwise reranking, and write the ensemble TSV plus diagnostics.

# %%
cmd = [
    PYTHON, SCRIPT, "rank-candidates",
    "--input", INPUT,
    "--pool-dir", POOL_DIR,
    "--output", ENSEMBLE_OUTPUT,
    "--backend", "openai",
    "--base-url", BASE_URL,
    "--model", MODELS[0]["model"],
    "--rerank-top-k", RERANK_TOP_K,
    "--timeout", TIMEOUT,
    "--diagnostics-dir", ENSEMBLE_DIAG,
    "--humor-model", "Humor-Research/humor-detection-comb-23",
    "--humor-model", "mohameddhiab/humor-no-humor",
    "--humor-weight", "0.20",
    "--humor-device", "-1",
    "--resume",
]
run_with_progress(cmd, total=TOTAL_ROWS, label="rank candidates")

# %% [markdown]
# ## 5. Validation
#
# Validate the ensemble TSV against the Task A input file before using it as a submission candidate.

# %%
cmd = [PYTHON, SCRIPT, "validate", "--input", INPUT, "--output", ENSEMBLE_OUTPUT]
run_with_progress(cmd, label="validate")

# %% [markdown]
# ## 6. Candidate Pool And Winner Summary By Model
#
# Count candidate-pool records and summarize how many final winners came from each source model.

# %%
total, by_file = pool_line_count()
print("pool records:", total)
print(by_file)

winners = Counter()
if ENSEMBLE_DIAG.exists():
    for path in ENSEMBLE_DIAG.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        winner = data.get("winner", "")
        for cand in data.get("candidates", []):
            if cand.get("text") == winner:
                winners[cand.get("source_model") or "unknown"] += 1
                break
winners
