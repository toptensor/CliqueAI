#!/bin/bash

VALIDATOR_NAME=validator-CliqueAI
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for arg in "$@"; do
    VALIDATOR_ARGS="$VALIDATOR_ARGS $arg"
done

cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install -e .

if pm2 list | grep -q "$VALIDATOR_NAME"; then
    pm2 delete "$VALIDATOR_NAME" 2>/dev/null || true
fi

pm2 start python3 --name "$VALIDATOR_NAME" -- \
    -m CliqueAI.validator \
    $VALIDATOR_ARGS
