#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
REQUIREMENTS_FILE="requirements.txt"

if [[ "${1:-}" == "--dev" ]]; then
    REQUIREMENTS_FILE="requirements-dev.txt"
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--dev]" >&2
    exit 2
fi

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r "$REQUIREMENTS_FILE"

echo "Local environment ready. Activate it with: source .venv/bin/activate"
