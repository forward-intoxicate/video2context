#!/usr/bin/env bash
# 一键准备运行环境（macOS / Linux）
#   用法： bash scripts/setup.sh
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
echo "工程目录：$root"

python_bin="${PYTHON:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "找不到 python3，请先安装 Python 3.9+（建议 3.11/3.12）" >&2
    exit 1
fi

if [ ! -d .venv ]; then
    echo "[1/4] 创建虚拟环境 .venv ..."
    "$python_bin" -m venv .venv
else
    echo "[1/4] 已存在 .venv，跳过创建"
fi

venv_py="$root/.venv/bin/python"
"$venv_py" -m pip install --upgrade pip

echo "[2/4] 安装核心依赖 ..."
"$venv_py" -m pip install -r requirements.txt

echo "[3/4] 安装网页界面依赖（可选，失败不影响命令行使用）..."
"$venv_py" -m pip install -r requirements-web.txt || echo "  [跳过] 网页依赖安装失败"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[i] 检测到 NVIDIA 显卡。Linux 上请确认已安装 CUDA 12 运行库与 cuDNN 9："
    echo "    Ubuntu: sudo apt install libcublas-12-* libcudnn9-cuda-12   （或使用 conda 安装）"
fi

echo "[4/4] 环境自检 ..."
"$venv_py" -m video2context doctor

cat <<'EOF'

完成。用法示例：
  ./.venv/bin/python -m video2context 视频.mp4
  ./.venv/bin/python -m video2context webui
EOF
