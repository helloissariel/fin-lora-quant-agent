#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
ADAPTER=${ADAPTER:-checkpoints/finlora-qwen2.5-1.5b}
${PY:-python} demo/app.py --adapter "$ADAPTER" --server-port "${PORT:-7860}"
