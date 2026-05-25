#!/usr/bin/env bash
# 训练 LoRA。A6000 / batch 8 / 3 epoch / ~30min
set -e
cd "$(dirname "$0")/.."

OUT=${OUT:-checkpoints/finlora-qwen2.5-1.5b}

${PY:-python} train/train_lora.py \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --train-data data/processed/train.jsonl \
    --eval-data  data/processed/eval.jsonl \
    --output-dir "$OUT" \
    --epochs 3 --batch-size 8 --lr 2e-4 \
    "$@"

${PY:-python} train/plot_curves.py --log "$OUT/log_history.json" --out assets/training_curves.png
echo "[train] curves -> assets/training_curves.png"
