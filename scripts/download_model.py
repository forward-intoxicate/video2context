"""把 faster-whisper 模型下载到本地目录（分块 + 断点续传 + 多源支持）。

为什么需要它：``huggingface_hub`` 默认的下载方式在网络不稳时容易长时间挂起，
部分网络下还会走 Xet 存储直接卡死。这个脚本用小分块 + 失败重试 + 断点续传的方式
把模型文件拉到指定目录，之后用 ``--model-dir`` 使用，完全离线。

支持两个源：
    * ``hf``          HuggingFace / hf-mirror（默认）
    * ``modelscope``  魔搭社区，国内直连通常更快

用法：
    python scripts/download_model.py large-v3
    python scripts/download_model.py large-v3 --source modelscope
    python scripts/download_model.py large-v3 --endpoint https://hf-mirror.com
    python scripts/download_model.py --repo Systran/faster-whisper-medium --out models/medium

下载完成后：
    python -m video2context 视频.mp4 --model-dir models/faster-whisper-large-v3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HF_ENDPOINT = "https://hf-mirror.com"
MODELSCOPE_ENDPOINT = "https://modelscope.cn"

CHUNK_SIZE = 8 * 1024 * 1024  # 8MB / 块
MAX_RETRIES = 6

#: 仓库接口不可用时的兜底文件清单（faster-whisper 模型的标准文件）
FALLBACK_FILES: tuple[str, ...] = (
    "config.json",
    "model.bin",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocabulary.json",
)

MODEL_NAMES: tuple[str, ...] = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "distil-small.en",
    "distil-medium.en",
    "distil-large-v3",
    "distil-large-v3.5",
)

#: 模型名 → HuggingFace 仓库。优先用 faster-whisper 自带的映射表，保证与库完全一致；
#: 万一库没装或改了私有常量，就退回下面这份显式表（内容与 1.2.x 一致）。
_FALLBACK_HF_REPOS: dict[str, str] = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
}

try:  # 与库自带映射保持一致，避免上游更新后本脚本对不上
    from faster_whisper.utils import _MODELS as HF_REPOS  # type: ignore
except Exception:  # pragma: no cover - 未安装 faster-whisper 时走兜底表
    HF_REPOS = _FALLBACK_HF_REPOS

#: 模型名 → 魔搭（ModelScope）仓库。只收录实测存在的，其余请用 --repo 指定。
MODELSCOPE_REPOS: dict[str, str] = {
    **{
        name: f"pengzhendong/faster-whisper-{name}"
        for name in MODEL_NAMES
        if not name.startswith("distil")
    },
    "distil-small.en": "pengzhendong/faster-distil-whisper-small.en",
}


def _open(url: str, *, headers: dict[str, str] | None = None, timeout: float = 60):
    merged = {"User-Agent": "video2context-model-downloader/0.1"}
    merged.update(headers or {})
    return urllib.request.urlopen(urllib.request.Request(url, headers=merged), timeout=timeout)


def file_url(source: str, endpoint: str, repo: str, name: str) -> str:
    """拼出单个文件的下载地址。"""
    if source == "modelscope":
        query = urllib.parse.urlencode({"Revision": "master", "FilePath": name})
        return f"{endpoint}/api/v1/models/{repo}/repo?{query}"
    return f"{endpoint}/{repo}/resolve/main/{name}"


def list_files(source: str, endpoint: str, repo: str) -> list[dict]:
    """列出仓库里需要下载的文件；接口不可用时退回内置清单。"""
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            if source == "modelscope":
                query = urllib.parse.urlencode({"Revision": "master", "Recursive": "true"})
                url = f"{endpoint}/api/v1/models/{repo}/repo/files?{query}"
                with _open(url, timeout=30) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                entries = (payload.get("Data") or {}).get("Files") or []
                files = [
                    {"name": entry.get("Path") or entry.get("Name"), "size": entry.get("Size")}
                    for entry in entries
                    if (entry.get("Path") or entry.get("Name"))
                ]
            else:
                url = f"{endpoint}/api/models/{repo}"
                with _open(url, timeout=30) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                files = [
                    {"name": sib.get("rfilename"), "size": sib.get("size")}
                    for sib in payload.get("siblings", [])
                    if sib.get("rfilename")
                ]

            files = [
                f
                for f in files
                if not f["name"].startswith(".")
                and f["name"].lower() != "readme.md"
                and "/" not in f["name"]
            ]
            if files:
                return files
        except Exception as exc:
            last_error = exc
            time.sleep(2 * attempt)

    print(f"  [提示] 无法获取仓库文件列表（{last_error}），使用内置清单继续。")
    return [{"name": name, "size": None} for name in FALLBACK_FILES]


def remote_size(url: str) -> int | None:
    try:
        with _open(url, headers={"Range": "bytes=0-0"}) as response:
            content_range = response.headers.get("Content-Range", "")
            if "/" in content_range:
                return int(content_range.rsplit("/", 1)[1])
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def download_file(url: str, dest: Path, size: int | None, label: str) -> None:
    """分块下载，支持从已下载部分续传。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and size is not None and dest.stat().st_size == size:
        print(f"  [跳过] {label}（已存在，{size / 1e6:.1f} MB）")
        return

    while True:
        done = part.stat().st_size if part.exists() else 0
        if size is not None and done >= size:
            break

        end = done + CHUNK_SIZE - 1
        if size is not None:
            end = min(end, size - 1)

        for attempt in range(1, MAX_RETRIES + 1):
            done = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={done}-{end}"}
            try:
                with _open(url, headers=headers, timeout=90) as response:
                    chunk = response.read()
                if not chunk:
                    raise RuntimeError("服务端返回空数据")
                with open(part, "ab") as handle:
                    handle.write(chunk)
                break
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(
                        f"{label} 下到 {done / 1e6:.1f} MB 后连续 {MAX_RETRIES} 次失败：{exc}"
                    ) from exc
                time.sleep(min(2 ** attempt, 15))

        current = part.stat().st_size if part.exists() else 0
        if size:
            sys.stdout.write(
                f"\r  {label}: {current / 1e6:7.1f}/{size / 1e6:.1f} MB ({current / size * 100:5.1f}%)"
            )
        else:
            sys.stdout.write(f"\r  {label}: {current / 1e6:7.1f} MB")
        sys.stdout.flush()
        if size is None:
            break

    sys.stdout.write("\n")
    if part.exists():
        part.replace(dest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="下载 faster-whisper 模型到本地目录（分块 + 断点续传）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model", nargs="?", help=f"模型名，可选：{', '.join(MODEL_NAMES)}")
    parser.add_argument("--source", choices=["hf", "modelscope"], default="hf", help="下载源（默认 hf）")
    parser.add_argument("--repo", default=None, help="直接指定仓库名，如 Systran/faster-whisper-large-v3")
    parser.add_argument("--out", default=None, help="输出目录（默认 models/faster-whisper-<模型名>）")
    parser.add_argument("--endpoint", default=None, help="自定义镜像地址（默认 hf-mirror / modelscope 官方）")
    args = parser.parse_args(argv)

    repos = MODELSCOPE_REPOS if args.source == "modelscope" else HF_REPOS
    repo = args.repo or repos.get(args.model or "", "")
    if not repo:
        parser.error(
            f"未知模型 {args.model!r}。可用：{', '.join(repos)}；或用 --repo 直接指定仓库。"
        )

    endpoint = (args.endpoint or (MODELSCOPE_ENDPOINT if args.source == "modelscope" else HF_ENDPOINT)).rstrip("/")
    name = args.model or repo.split("/")[-1]
    out_dir = Path(args.out or f"models/faster-whisper-{name}").resolve()

    print(f"来源   : {args.source}  ({endpoint})")
    print(f"仓库   : {repo}")
    print(f"输出   : {out_dir}")

    files = list_files(args.source, endpoint, repo)
    if not files:
        print("没有发现可下载的文件，仓库名或镜像可能有误。", file=sys.stderr)
        return 1

    total = 0
    for entry in files:
        url = file_url(args.source, endpoint, repo, entry["name"])
        size = entry.get("size") or remote_size(url)
        download_file(url, out_dir / entry["name"], size, entry["name"])
        if size:
            total += size

    print(f"\n完成，共 {total / 1e6:.1f} MB。现在可以这样使用：")
    print(f'  python -m video2context 视频.mp4 --model-dir "{out_dir}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
