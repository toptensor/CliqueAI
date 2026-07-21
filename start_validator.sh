#!/usr/bin/env bash
set -e

PROCESS_NAME="validator-CliqueAI"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ROOT="$PROJECT_ROOT/.venvs/validator"
VENV_DIR="$VENV_ROOT/$(date -u +%Y%m%d%H%M%S)-$$"
VENV_PYTHON="$VENV_DIR/bin/python"

cd "$PROJECT_ROOT"
mkdir -p "$VENV_ROOT"
python3.12 -m venv "$VENV_DIR"
"$VENV_PYTHON" -m pip install -r requirements.txt
"$VENV_PYTHON" -m pip install -e . --no-deps

pm2 delete "$PROCESS_NAME" >/dev/null 2>&1 || true
pm2 start "$VENV_PYTHON" --name "$PROCESS_NAME" --cwd "$PROJECT_ROOT" --interpreter none -- \
    -m CliqueAI.validator "$@"

find "$VENV_ROOT" -mindepth 1 -maxdepth 1 ! -path "$VENV_DIR" -exec rm -rf -- {} +
