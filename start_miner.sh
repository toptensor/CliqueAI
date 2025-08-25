#!/bin/bash

MINER_NAME=miner-CliqueAI
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for arg in "$@"; do
    MINER_ARGS="$MINER_ARGS $arg"
done

cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install -e .

if pm2 list | grep -q "$MINER_NAME"; then
    pm2 delete "$MINER_NAME" 2>/dev/null || true
fi

pm2 start python3 --name "$MINER_NAME" -- \
    -m CliqueAI.miner \
    $MINER_ARGS
