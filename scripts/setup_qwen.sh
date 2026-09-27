#!/usr/bin/env bash
# 为 Qwen3-ASR 准备独立环境 .venv-qwen（Linux / macOS）
#
#   用法（在工程根目录）：
#     bash scripts/setup_qwen.sh
#     TORCH_INDEX=cpu bash scripts/setup_qwen.sh
#     DOWNLOAD_MODELS=1 bash scripts/setup_qwen.sh
#
# 为什么要单独一个环境：Qwen3-ASR 需要 torch + transformers（装完约 3GB），
# 而主工程刻意保持轻量（faster-whisper 走 CTranslate2，不需要 torch）。
# 混进同一个环境容易把主工程的依赖搞坏，所以单开 .venv-qwen，由子进程桥接调用。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "工程目录：$ROOT"

VENV="$ROOT/.venv-qwen"
PY="$VENV/bin/python"

# TORCH_INDEX 用 PyTorch 官方索引的目录名：cu126 / cu124 / cu121 / cpu
TORCH_INDEX="${TORCH_INDEX:-cu126}"
MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

if [ ! -x "$PY" ]; then
    echo "[1/4] 创建独立环境 .venv-qwen ..."
    python3 -m venv "$VENV"
else
    echo "[1/4] 已存在 .venv-qwen，跳过创建"
fi

"$PY" -m pip install --upgrade pip

if [ "$TORCH_INDEX" = "none" ]; then
    echo "[2/4] 按参数要求跳过 torch 安装"
else
    echo "[2/4] 安装 torch（$TORCH_INDEX，约 2.5GB，耐心等）..."
    # 直接 pip install torch 在 PyPI 上默认拿到的是 +cpu 构建，
    # 要 CUDA 版必须走 PyTorch 官方索引。
    "$PY" -m pip install torch --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
fi

echo "[3/4] 安装 qwen-asr 及其依赖 ..."
"$PY" -m pip install -i "$MIRROR" --timeout 60 --retries 5 -r requirements-qwen.txt

echo "[4/4] 环境自检 ..."
"$PY" -c "import torch, transformers, qwen_asr; print('  torch', torch.__version__, '| cuda 可用', torch.cuda.is_available(), '| transformers', transformers.__version__)"

if [ "${DOWNLOAD_MODELS:-0}" = "1" ]; then
    echo "[i] 下载模型（约 6GB，支持断点续传）..."
    MAIN_PY="$ROOT/.venv/bin/python"
    [ -x "$MAIN_PY" ] || MAIN_PY="python3"
    "$MAIN_PY" scripts/download_model.py --repo Qwen/Qwen3-ASR-1.7B --source modelscope --out models/Qwen3-ASR-1.7B
    "$MAIN_PY" scripts/download_model.py --repo Qwen/Qwen3-ForcedAligner-0.6B --source modelscope --out models/Qwen3-ForcedAligner-0.6B
fi

echo ""
echo "完成。用法："
echo "  .venv/bin/python -m video2context --engine qwen3-asr --qwen-setup"
if [ "${DOWNLOAD_MODELS:-0}" != "1" ]; then
    echo ""
    echo "还没下载模型的话，先跑（约 6GB）："
    echo "  .venv/bin/python scripts/download_model.py --repo Qwen/Qwen3-ASR-1.7B --source modelscope --out models/Qwen3-ASR-1.7B"
    echo "  .venv/bin/python scripts/download_model.py --repo Qwen/Qwen3-ForcedAligner-0.6B --source modelscope --out models/Qwen3-ForcedAligner-0.6B"
fi
