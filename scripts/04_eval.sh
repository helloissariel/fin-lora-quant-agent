#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

ADAPTER=${ADAPTER:-checkpoints/finlora-qwen2.5-1.5b}

echo "=== Baseline (no adapter) ==="
${PY:-python} train/eval_model.py --output eval/results_baseline.json

echo
echo "=== With FinLoRA adapter ==="
${PY:-python} train/eval_model.py --adapter "$ADAPTER" --output eval/results_lora.json

${PY:-python} - <<'PY'
import json
b = json.load(open("eval/results_baseline.json"))["metrics"]
l = json.load(open("eval/results_lora.json"))["metrics"]
print(f"\n[summary] N={b['n']}")
print(f"  baseline accuracy = {b['accuracy']:.4f}")
print(f"  lora     accuracy = {l['accuracy']:.4f}")
print(f"  delta             = {l['accuracy']-b['accuracy']:+.4f}")
PY
