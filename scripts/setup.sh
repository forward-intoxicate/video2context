#!/usr/bin/env bash
# 一键部署 video2context（Linux / macOS）
#
#   bash scripts/setup.sh                          # 部署 Qwen3-ASR（默认，推荐）
#   SKIP_MODELS=1 bash scripts/setup.sh            # 只建环境，不下模型
#   bash scripts/setup.sh whisper                  # 只部署 Whisper（轻量，不需要 torch）
#   bash scripts/setup.sh both                     # 两个引擎都装
#   WEB=1 bash scripts/setup.sh                    # 额外装网页界面
#   TORCH_INDEX=cpu bash scripts/setup.sh          # 没有 N 卡时
#
# 默认会把模型一起下好（Qwen 约 6GB / Whisper 约 3GB），可断点续传、中断后重跑接着下。
# 为什么默认就下：模型反正第一次运行要用，提前下好可以走国内镜像（快得多）；
# 而且 Qwen 的**强制对齐模型**决定字幕时间精度 —— 少了它字幕时间会掉到最粗的一档
# （详见 docs/troubleshooting.md 7.4）。
#
# 引擎怎么选、各设备怎么部署，见 docs/deploy.md。
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
echo "工程目录：$root"

python_bin="${PYTHON:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "找不到 python3，请先安装 Python 3.9+（建议 3.11/3.12）" >&2
    exit 1
fi

engine="${1:-${ENGINE:-qwen}}"
case "$engine" in
    qwen | whisper | both) ;;
    *) echo "参数错误：引擎只能是 qwen / whisper / both，收到 '$engine'" >&2; exit 2 ;;
esac

torch_index="${TORCH_INDEX:-cu126}"
mirror="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
skip_models="${SKIP_MODELS:-0}"

want_whisper=0
want_qwen=0
# 注意写法：不能写成 `[ a ] || [ b ] && x=1` —— bash 里那等价于 `([a] || [b]) && x=1`，
# 两个判断都不成立时整条语句返回非零，在 set -e 下会直接把脚本干掉。
if [ "$engine" = "whisper" ] || [ "$engine" = "both" ]; then
    want_whisper=1
fi
if [ "$engine" = "qwen" ] || [ "$engine" = "both" ]; then
    want_qwen=1
fi

# ---------------------------------------------------------------- 主环境
if [ ! -x ".venv/bin/python" ]; then
    echo "[1/4] 创建主虚拟环境 .venv ..."
    "$python_bin" -m venv .venv
else
    echo "[1/4] 已存在主环境 .venv，跳过创建"
fi

venv_py="$root/.venv/bin/python"
"$venv_py" -m pip install --upgrade pip

echo "[2/4] 安装公共依赖（ffmpeg 等）..."
"$venv_py" -m pip install -i "$mirror" --timeout 60 --retries 5 -r requirements-base.txt

if [ "$want_whisper" = "1" ]; then
    echo "[3/4] 安装 Whisper 引擎（faster-whisper）..."
    "$venv_py" -m pip install -i "$mirror" --timeout 60 --retries 5 -r requirements.txt
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "      检测到 NVIDIA 显卡。Linux 上 ctranslate2 还需要系统级 CUDA 12 与 cuDNN 9："
        echo "        sudo apt install libcublas-12-* libcudnn9-cuda-12   # Ubuntu 示例"
    else
        echo "      未检测到 NVIDIA 显卡，Whisper 将用 CPU 推理（慢但可用）"
    fi
else
    echo "[3/4] 跳过 Whisper 引擎（只部署 Qwen3-ASR，主环境保持无 torch）"
fi

if [ "${WEB:-0}" = "1" ]; then
    echo "      安装网页界面依赖（gradio，失败不影响命令行使用）..."
    "$venv_py" -m pip install -i "$mirror" --timeout 60 --retries 5 -r requirements-web.txt \
        || echo "  [跳过] 网页依赖安装失败"
fi

# ---------------------------------------------------------------- Qwen 独立环境
if [ "$want_qwen" = "1" ]; then
    echo "[4/4] 部署 Qwen3-ASR 独立环境 ..."
    # 模型下载交给 setup_qwen.sh（识别模型 + 强制对齐模型，两者默认都下）
    TORCH_INDEX="$torch_index" PIP_MIRROR="$mirror" SKIP_MODELS="$skip_models" \
        bash "$root/scripts/setup_qwen.sh"
else
    echo "[4/4] 跳过 Qwen3-ASR"
fi

# ---------------------------------------------------------------- Whisper 模型
if [ "$want_whisper" = "1" ] && [ "$skip_models" != "1" ] \
    && [ ! -f "models/faster-whisper-large-v3/model.bin" ]; then
    echo "[i] 下载 Whisper large-v3（约 3GB，支持断点续传）..."
    "$venv_py" scripts/download_model.py large-v3 --source modelscope --out models/faster-whisper-large-v3
fi

# ---------------------------------------------------------------- 自检
echo ""
echo "环境自检："
"$venv_py" -m video2context doctor

cat <<'EOF'

完成。用法示例：
  ./.venv/bin/python -m video2context 视频.mp4
  ./.venv/bin/python -m video2context webui
EOF

if [ "$skip_models" = "1" ]; then
    echo ""
    echo "注意：这次跳过了模型下载。第一次运行会自动下，但走 HuggingFace、国内会慢。"
    echo "建议现在补下（可断点续传）："
    if [ "$want_qwen" = "1" ]; then
        echo "  .venv/bin/python scripts/download_model.py --repo Qwen/Qwen3-ASR-1.7B --source modelscope --out models/Qwen3-ASR-1.7B"
        echo "  .venv/bin/python scripts/download_model.py --repo Qwen/Qwen3-ForcedAligner-0.6B --source modelscope --out models/Qwen3-ForcedAligner-0.6B"
    fi
    if [ "$want_whisper" = "1" ]; then
        echo "  .venv/bin/python scripts/download_model.py large-v3 --source modelscope --out models/faster-whisper-large-v3"
    fi
fi
