#!/usr/bin/env bash
# 32B QLoRA 训练完成后的全部自动化:
#   1) 等待 train_qlora.py 结束
#   2) 画曲线
#   3) eval base 4bit  vs  eval QLoRA 4bit
#   4) 汇总 + commit + push
#
# 使用: nohup bash scripts/07_post_train_32b.sh > logs/post_train_32b.log 2>&1 < /dev/null & disown
set -e
cd "$(dirname "$0")/.."

PY=${PY:-/opt/conda/envs/mind/bin/python}
ADAPTER=${ADAPTER:-checkpoints/finlora-qwen2.5-32b-qlora}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-32B-Instruct}
LOG=logs/train_qlora.log
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export TOKENIZERS_PARALLELISM=false

echo "[post32b] $(date) waiting for QLoRA training to finish..."
while pgrep -f "train_qlora.py" > /dev/null; do
    sleep 10
done

if ! grep -q "saved adapter" "$LOG" 2>/dev/null; then
    echo "[post32b] FAIL: training process gone but no 'saved adapter' marker"
    tail -50 "$LOG"
    exit 1
fi
echo "[post32b] $(date) training finished. proceeding."

# 1) 训练曲线
echo "[post32b] === generating curves ==="
$PY train/plot_curves.py --log "$ADAPTER/log_history.json" --out assets/training_curves_32b.png || true

# 2) eval base (4bit)
echo "[post32b] === eval base (4bit) ==="
$PY train/eval_model.py \
    --base-model "$BASE_MODEL" --load-in-4bit \
    --output eval/results_32b_baseline.json

# 3) eval QLoRA
echo "[post32b] === eval QLoRA ==="
$PY train/eval_model.py \
    --base-model "$BASE_MODEL" --load-in-4bit \
    --adapter "$ADAPTER" \
    --output eval/results_32b_qlora.json

# 4) 汇总
$PY - <<'PY'
import json
b15 = json.load(open("eval/results_baseline.json"))["metrics"]
l15 = json.load(open("eval/results_lora.json"))["metrics"]
b32 = json.load(open("eval/results_32b_baseline.json"))["metrics"]
l32 = json.load(open("eval/results_32b_qlora.json"))["metrics"]

def fmt(m):
    r = m["report"]
    return (f"acc={m['accuracy']:.4f}  macroF1={r['macro avg']['f1-score']:.4f}  "
            f"POS={r['POSITIVE']['f1-score']:.3f}  NEG={r['NEGATIVE']['f1-score']:.3f}  "
            f"NEU={r['NEUTRAL']['f1-score']:.3f}")

print("\n[post32b] === scaling summary (N=200) ===")
print(f"  1.5B base       : {fmt(b15)}")
print(f"  1.5B + LoRA     : {fmt(l15)}")
print(f"  32B  base (4bit): {fmt(b32)}")
print(f"  32B  + QLoRA    : {fmt(l32)}")

with open("eval/SUMMARY.md", "w") as f:
    f.write("# Eval Summary — Scaling Comparison\n\n")
    f.write("| Model | Trainable | Acc | Macro-F1 | POS F1 | NEG F1 | NEU F1 |\n")
    f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
    for name, trainable, m in [
        ("Qwen2.5-1.5B (base)",          "—",                  b15),
        ("Qwen2.5-1.5B + LoRA",          "4.36M (0.28%)",      l15),
        ("Qwen2.5-32B (base, 4bit NF4)", "—",                  b32),
        ("Qwen2.5-32B + **QLoRA**",      "~30M (~0.09%)",      l32),
    ]:
        r = m['report']
        f.write(f"| {name} | {trainable} | {m['accuracy']:.4f} | "
                f"{r['macro avg']['f1-score']:.4f} | "
                f"{r['POSITIVE']['f1-score']:.3f} | "
                f"{r['NEGATIVE']['f1-score']:.3f} | "
                f"{r['NEUTRAL']['f1-score']:.3f} |\n")
    f.write(f"\n*N={b15['n']} samples · deterministic decoding · 4bit eval uses NF4 + double quant + bf16 compute.*\n")
PY

# 5) 提交 + 推送
echo "[post32b] === committing and pushing ==="
git add assets/training_curves_32b.png \
        eval/results_32b_baseline.json eval/results_32b_qlora.json eval/SUMMARY.md \
        "$ADAPTER/log_history.json" "$ADAPTER/train_metrics.json" 2>/dev/null || true
git commit -m "scale-up: 32B QLoRA (NF4 4bit + paged AdamW) eval + curves

- 1.5B LoRA -> 32B QLoRA comparison
- eval base 4bit and QLoRA on identical N=200 holdout
- assets/training_curves_32b.png from log_history.json" || echo "[post32b] nothing to commit"

git push 2>&1 | tail -5 || echo "[post32b] push failed"

echo "[post32b] $(date) DONE"
