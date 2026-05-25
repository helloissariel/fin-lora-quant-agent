#!/usr/bin/env bash
# 安装依赖。
#
# 本服务器环境备注 (172.23.166.123):
#   base env 的 torch 2.12 是 cu130 编译，与 cu128 驱动不兼容。
#   请用 mind env (torch 2.6+cu124, cuda True):
#       conda activate mind
#   或直接用解释器路径:
#       /opt/conda/envs/mind/bin/python
set -e
cd "$(dirname "$0")/.."
PY=${PY:-/opt/conda/envs/mind/bin/python}
$PY -m pip install -r requirements.txt
$PY -c 'import torch; print("[install] torch", torch.__version__, "cuda", torch.cuda.is_available())'
