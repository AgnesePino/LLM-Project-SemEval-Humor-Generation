# %% [markdown]
# # MWAHAHA Task A EN - Ensemble Runs
#
# Notebook operativo per generare candidate pool sequenziali con Qwen, Gemma e Ministral, poi fare ranking finale con Qwen judge.
#
# Usa questo notebook quando puoi caricare un solo modello alla volta in LM Studio.

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
# ## 1. Generazione candidati con Qwen
#
# Carica Qwen in LM Studio prima di eseguire questa cella.

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
# ## 2. Generazione candidati con Gemma
#
# Carica Gemma in LM Studio prima di eseguire questa cella. Aggiorna `MODELS[1]["model"]` se il nome esposto da LM Studio ? diverso.

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
# ## 3. Generazione candidati con Ministral
#
# Carica Ministral in LM Studio prima di eseguire questa cella. Aggiorna `MODELS[2]["model"]` se necessario.

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
# ## 4. Ranking finale con Qwen judge
#
# Ricarica Qwen in LM Studio prima di eseguire questa cella.

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
# ## 5. Validazione

# %%
cmd = [PYTHON, SCRIPT, "validate", "--input", INPUT, "--output", ENSEMBLE_OUTPUT]
run_with_progress(cmd, label="validate")

# %% [markdown]
# ## 6. Riepilogo candidate pool e winner per modello

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
