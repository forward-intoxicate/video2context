#!/usr/bin/env bash
# 启动本地网页界面（macOS / Linux）
#   用法： bash scripts/run_web.sh [端口] [设备]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

port="${1:-7860}"
device="${2:-auto}"

py="$root/.venv/bin/python"
if [ ! -x "$py" ]; then
    py="python3"
fi

exec "$py" -m video2context webui --port "$port" --device "$device"
