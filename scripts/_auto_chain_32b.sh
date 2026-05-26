#!/usr/bin/env bash
# 等下载完 → 启动训练 → 启动 post-train。整条链都 nohup，独立于 SSH。
set -e
cd /root/wallclock/FinLoRA-Agent

# 1) 等下载结束 (grep DONE 或者进程退出且文件齐全)
echo "[chain] $(date) waiting for 32B download..."
while pgrep -f "snapshot_download.*Qwen2.5-32B" > /dev/null; do
    sleep 30
done
if ! grep -q "DONE in" logs/download_32b.log 2>/dev/null; then
    echo "[chain] FAIL: download exited without DONE marker"
    tail -20 logs/download_32b.log
    exit 1
fi
echo "[chain] $(date) download complete."

# 2) 启动 QLoRA 训练
echo "[chain] $(date) launching QLoRA training..."
source /opt/conda/etc/profile.d/conda.sh && conda activate mind
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
nohup python -u train/train_qlora.py --output-dir checkpoints/finlora-qwen2.5-32b-qlora > logs/train_qlora.log 2>&1 < /dev/null &
disown
TRAIN_PID=$!
echo "[chain] training PID=$TRAIN_PID, log=logs/train_qlora.log"

# 3) 启动 post-train 链 (它自己会等 train_qlora.py 结束)
sleep 5
nohup bash scripts/07_post_train_32b.sh > logs/post_train_32b.log 2>&1 < /dev/null &
disown
POST_PID=$!
echo "[chain] post-train PID=$POST_PID, log=logs/post_train_32b.log"
echo "[chain] $(date) DONE arming chain"
