#!/usr/bin/env bash
# 训练完成后的全部自动化:
#   1. 等待 train_lora.py 进程结束
#   2. 生成训练曲线
#   3. 跑 base / lora 评测
#   4. 提交 + 推送
#
# 使用: nohup bash scripts/06_post_train.sh > logs/post_train.log 2>&1 < /dev/null & disown
set -e
cd "$(dirname "$0")/.."

PY=${PY:-/opt/conda/envs/mind/bin/python}
ADAPTER=${ADAPTER:-checkpoints/finlora-qwen2.5-1.5b}
LOG=logs/train.log
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export TOKENIZERS_PARALLELISM=false

echo "[post] $(date) waiting for training to finish..."
while pgrep -f "train_lora.py" > /dev/null; do
    sleep 10
done

if ! grep -q "saved LoRA adapter" "$LOG"; then
    echo "[post] FAIL: training process gone but no 'saved LoRA adapter' marker"
    tail -50 "$LOG"
    exit 1
fi
echo "[post] $(date) training finished. proceeding."

# 1) 训练曲线
echo "[post] === generating training curves ==="
$PY train/plot_curves.py --log "$ADAPTER/log_history.json" --out assets/training_curves.png

# 2) 评测
echo "[post] === eval baseline ==="
$PY train/eval_model.py --output eval/results_baseline.json

echo "[post] === eval LoRA ==="
$PY train/eval_model.py --adapter "$ADAPTER" --output eval/results_lora.json

# 3) 汇总
$PY - <<'PY'
import json
b = json.load(open("eval/results_baseline.json"))["metrics"]
l = json.load(open("eval/results_lora.json"))["metrics"]
print(f"\n[post] N={b['n']}")
print(f"  baseline accuracy = {b['accuracy']:.4f}")
print(f"  lora     accuracy = {l['accuracy']:.4f}")
print(f"  delta             = {l['accuracy']-b['accuracy']:+.4f}")

# 写入一个简版 markdown 供 README 引用
with open("eval/SUMMARY.md", "w") as f:
    f.write(f"# Eval Summary\n\n")
    f.write(f"| Metric | Base Qwen2.5-1.5B | + FinLoRA | Δ |\n")
    f.write(f"| --- | --- | --- | --- |\n")
    f.write(f"| Accuracy | {b['accuracy']:.4f} | {l['accuracy']:.4f} | {l['accuracy']-b['accuracy']:+.4f} |\n")
    if 'report' in b and 'report' in l:
        for k in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
            try:
                bf = b['report'][k]['f1-score']; lf = l['report'][k]['f1-score']
                f.write(f"| F1 ({k}) | {bf:.3f} | {lf:.3f} | {lf-bf:+.3f} |\n")
            except KeyError:
                pass
    f.write(f"\n*N={b['n']} samples, deterministic decoding.*\n")
PY

# 4) 提交并推送
echo "[post] === committing and pushing ==="
git add assets/training_curves.png \
        eval/results_baseline.json eval/results_lora.json eval/SUMMARY.md \
        "$ADAPTER/log_history.json" "$ADAPTER/train_metrics.json" 2>/dev/null || true
git commit -m "results: real training curves + base vs FinLoRA eval

- assets/training_curves.png from log_history.json
- eval/results_*.json with accuracy / F1 / confusion matrix
- eval/SUMMARY.md tabulated diff" || echo "[post] nothing to commit"

git push -u origin main || echo "[post] push failed — repo may not exist on GitHub yet"

echo "[post] $(date) DONE"
