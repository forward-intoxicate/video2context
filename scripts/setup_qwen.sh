#!/usr/bin/env bash
# 部署 Qwen3-ASR 识别引擎（Linux / macOS）
#
# 单独跑这个脚本 = 完整部署「只用 Qwen3-ASR，不装 Whisper」的环境：
#   * 主环境 .venv        → 只装公共依赖（ffmpeg），保持无 torch
#   * 独立环境 .venv-qwen → torch + transformers + qwen-asr
#   * 两个模型            → 识别模型 + **强制对齐模型**（默认一起下，约 6GB）
#
#   bash scripts/setup_qwen.sh
#   TORCH_INDEX=cpu bash scripts/setup_qwen.sh
#   SKIP_MODELS=1 bash scripts/setup_qwen.sh        # 先不下载模型
#
# 为什么要单独一个环境：Qwen3-ASR 需要 torch + transformers（装完约 3GB），
# 而主工程刻意保持轻量（Whisper 引擎走 CTranslate2，不需要 torch）。
# 混进同一个环境容易把依赖搞坏，所以单开 .venv-qwen，由子进程桥接调用。
#
# 为什么要连对齐模型一起装：Qwen3-ASR 不输出时间戳，字幕时间是本工程造的。
# 没有对齐模型时只能按"说话区间"估算，而**只装 Qwen 的机器没有 faster-whisper，
# 也就没有 VAD**，会掉到最粗的一档（在整条音轨上平均分配，长静音视频字幕会明显错位）。
# 装上它就直接精确到词。详见 docs/troubleshooting.md 7.4。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "工程目录：$ROOT"

# TORCH_INDEX 用 PyTorch 官方索引的目录名：cu126 / cu124 / cu121 / cpu
TORCH_INDEX="${TORCH_INDEX:-cu126}"
MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
# 模型根目录：默认工程内的 models/，可用 V2C_MODELS_DIR 指到别处（代码与权重分开放）
MODELS_DIR="${V2C_MODELS_DIR:-$ROOT/models}"

#: 建 venv 时如果自带的 ensurepip 不可用（conda 的 python 常见），
#: 退回"先建空环境、再用外部 pip 灌进去"。
create_venv() {
    local target="$1" python_bin="$2"
    if "$python_bin" -m venv "$target" 2>/dev/null; then
        return 0
    fi
    echo "      venv 自带的 ensurepip 不可用 → 改用 --without-pip + 外部 pip"
    rm -rf "$target"
    "$python_bin" -m venv --without-pip "$target"
    "$python_bin" -m pip --python "$target/bin/python" install --upgrade pip
}

# ------------------------------------------------- 主环境（只装公共依赖，不含 Whisper）
PYTHON_BIN="${PYTHON:-python3}"
MAIN_PY="$ROOT/.venv/bin/python"
if [ ! -x "$MAIN_PY" ]; then
    echo "[1/5] 创建主环境 .venv（只装公共依赖，不含 Whisper）..."
    create_venv "$ROOT/.venv" "$PYTHON_BIN"
else
    echo "[1/5] 已存在主环境 .venv"
fi
"$MAIN_PY" -m pip install --upgrade pip
"$MAIN_PY" -m pip install -i "$MIRROR" --timeout 60 --retries 5 -r requirements-base.txt

# ---------------------------------------------------------------- Qwen 独立环境
VENV="$ROOT/.venv-qwen"
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
    echo "[2/5] 创建独立环境 .venv-qwen ..."
    create_venv "$VENV" "$PYTHON_BIN"
else
    echo "[2/5] 已存在 .venv-qwen，跳过创建"
fi

"$PY" -m pip install --upgrade pip

if [ "$TORCH_INDEX" = "none" ]; then
    echo "[3/5] 按参数要求跳过 torch 安装（请自行确保环境里已有 torch）"
else
    echo "[3/5] 安装 torch（$TORCH_INDEX，约 2.5GB，耐心等）..."
    # 直接 pip install torch 在 PyPI 上默认拿到的是 +cpu 构建，
    # 要 CUDA 版必须走 PyTorch 官方索引。
    "$PY" -m pip install torch --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
fi

echo "[4/5] 安装 qwen-asr 及其依赖 ..."
"$PY" -m pip install -i "$MIRROR" --timeout 60 --retries 5 -r requirements-qwen.txt

echo "[5/5] 环境自检 ..."
"$PY" -c "import torch, transformers, qwen_asr; print('  torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"
"$MAIN_PY" -m video2context --engine qwen3-asr --qwen-setup

if [ "${SKIP_MODELS:-0}" = "1" ]; then
    echo "[i] 按 SKIP_MODELS 跳过模型下载（第一次运行会自动下）"
else
    # 对齐模型不是可选项：没有它，字幕时间会掉到最粗的一档（见文件开头说明）。
    echo "[i] 下载识别模型（约 4GB，支持断点续传，中断后重跑即可接着下）..."
    "$MAIN_PY" scripts/download_model.py --repo Qwen/Qwen3-ASR-1.7B --source modelscope --out "$MODELS_DIR/Qwen3-ASR-1.7B"
    echo "[i] 下载强制对齐模型（约 1.8GB，字幕精确到词就靠它）..."
    "$MAIN_PY" scripts/download_model.py --repo Qwen/Qwen3-ForcedAligner-0.6B --source modelscope --out "$MODELS_DIR/Qwen3-ForcedAligner-0.6B"
fi

echo ""
echo "完成。用法："
echo "  .venv/bin/python -m video2context 我的视频.mp4"
if [ "${SKIP_MODELS:-0}" = "1" ]; then
    echo ""
    echo "记得补下模型（约 6GB；第一次运行也会自动下，但走 HuggingFace、国内会慢）："
    echo "  .venv/bin/python scripts/download_model.py --repo Qwen/Qwen3-ASR-1.7B --source modelscope --out "$MODELS_DIR/Qwen3-ASR-1.7B""
    echo "  .venv/bin/python scripts/download_model.py --repo Qwen/Qwen3-ForcedAligner-0.6B --source modelscope --out "$MODELS_DIR/Qwen3-ForcedAligner-0.6B""
fi
