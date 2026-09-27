"""ffmpeg 定位、媒体信息探测与音轨提取。

ffmpeg 查找顺序：
    1. 环境变量 ``FFMPEG_BIN``（最高优先级，便于指定自带版本）
    2. 系统 PATH 中的 ``ffmpeg``
    3. ``imageio-ffmpeg`` 附带的静态 ffmpeg（pip 安装即得，免手工配置）
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# 语音识别模型统一要求的输入规格
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")
_FORMAT_RE = re.compile(r"Input #0,\s*([^,]+(?:,[^,]+)*?),\s*from\s")
_VIDEO_RE = re.compile(r"Stream #\d+:\d+.*?:\s*Video:")
_AUDIO_RE = re.compile(r"Stream #\d+:\d+.*?:\s*Audio:")

_INSTALL_HINT = (
    "未找到 ffmpeg。任选一种方式解决：\n"
    "  1) pip install imageio-ffmpeg        # 本工程默认依赖，自带静态 ffmpeg，免配置\n"
    "  2) winget install Gyan.FFmpeg        # 或 conda install -c conda-forge ffmpeg\n"
    "  3) 设置环境变量 FFMPEG_BIN=<ffmpeg.exe 完整路径>"
)


def _missing_ffmpeg_message() -> str:
    """拼出"找不到 ffmpeg"的完整提示。

    最常见的真实原因其实是**跑错了 Python**（敲 `python` 跑到了 conda base / 系统
    Python，而依赖装在工程 `.venv` 里）。所以把环境诊断放在最前面 ——
    否则用户会照着"pip install imageio-ffmpeg"去装，把 base 环境也搞乱。
    """
    from .config import interpreter_hint

    hint = interpreter_hint()
    if hint:
        return "未找到 ffmpeg。" + hint + "\n确实想在当前环境里用的话：\n" + _INSTALL_HINT.split("\n", 1)[1]
    return _INSTALL_HINT


class FFmpegError(RuntimeError):
    """ffmpeg 调用失败或媒体信息无法解析。"""


class FFmpegNotFoundError(FFmpegError):
    """系统内找不到可用的 ffmpeg 可执行文件。"""


@dataclass
class MediaInfo:
    """媒体文件的基本信息。"""

    path: Path
    duration: float
    has_video: bool
    has_audio: bool
    format_name: str = ""
    raw: str = ""

    @property
    def duration_hms(self) -> str:
        return format_hms(self.duration)


_ffmpeg_path: Optional[str] = None


def find_ffmpeg(refresh: bool = False) -> str:
    """返回可用的 ffmpeg 可执行文件路径，结果会被缓存。"""
    global _ffmpeg_path
    if _ffmpeg_path and not refresh and Path(_ffmpeg_path).is_file():
        return _ffmpeg_path

    candidates: list[str] = []

    env_value = os.environ.get("FFMPEG_BIN", "").strip().strip('"')
    if env_value:
        candidates.append(env_value)

    on_path = shutil.which("ffmpeg")
    if on_path:
        candidates.append(on_path)

    try:  # 可选依赖：pip install imageio-ffmpeg
        import imageio_ffmpeg  # type: ignore

        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:  # pragma: no cover - 未安装时静默跳过
        pass

    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            _ffmpeg_path = str(Path(candidate).resolve())
            return _ffmpeg_path

    raise FFmpegNotFoundError(_missing_ffmpeg_message())


def _run(args: Sequence[str], timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError(_missing_ffmpeg_message()) from exc
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"ffmpeg 执行超时（>{timeout}s）：{' '.join(map(str, args))}") from exc


def probe_media(path: str | os.PathLike[str]) -> MediaInfo:
    """读取媒体时长、是否含视频/音频轨等信息。

    只用 ``ffmpeg -i``，不依赖 ffprobe，因此自带静态 ffmpeg 也能工作。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"输入文件不存在：{p}")
    if not p.is_file():
        raise FFmpegError(f"输入不是文件：{p}")

    ffmpeg = find_ffmpeg()
    proc = _run([ffmpeg, "-hide_banner", "-i", str(p)])
    text = f"{proc.stderr or ''}\n{proc.stdout or ''}"

    match = _DURATION_RE.search(text)
    if not match:
        raise FFmpegError(
            f"无法解析媒体时长，文件可能损坏或不是音视频格式：{p}\n"
            f"ffmpeg 输出：\n{text.strip()[:800]}"
        )
    hours, minutes, seconds = match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    format_match = _FORMAT_RE.search(text)
    return MediaInfo(
        path=p,
        duration=duration,
        has_video=bool(_VIDEO_RE.search(text)),
        has_audio=bool(_AUDIO_RE.search(text)),
        format_name=(format_match.group(1).strip() if format_match else ""),
        raw=text,
    )


def extract_audio(
    source: str | os.PathLike[str],
    dest_wav: str | os.PathLike[str],
    *,
    start: Optional[float] = None,
    duration: Optional[float] = None,
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = TARGET_CHANNELS,
    overwrite: bool = True,
    timeout: Optional[float] = None,
) -> Path:
    """把 ``source`` 的第一条音轨抽成 PCM wav（默认 16kHz 单声道）。"""
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"输入文件不存在：{src}")

    dest = Path(dest_wav)
    dest.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = find_ffmpeg()
    args: list[str] = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
    if overwrite:
        args.append("-y")
    if start is not None:
        args += ["-ss", f"{float(start):.3f}"]
    args += ["-i", str(src)]
    if duration is not None:
        args += ["-t", f"{float(duration):.3f}"]
    args += [
        "-vn",
        "-sn",
        "-dn",
        "-map",
        "0:a:0",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(dest),
    ]

    proc = _run(args, timeout=timeout)
    if proc.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        raise FFmpegError(
            f"提取音频失败（ffmpeg 退出码 {proc.returncode}）：{src}\n"
            f"{(proc.stderr or '').strip()[:800]}"
        )
    return dest


def ffmpeg_version() -> str:
    """返回 ffmpeg 版本首行，用于诊断输出。"""
    proc = _run([find_ffmpeg(), "-hide_banner", "-version"])
    first_line = (proc.stdout or proc.stderr or "").strip().splitlines()
    return first_line[0] if first_line else "unknown"


def format_hms(seconds: float) -> str:
    """秒 → ``HH:MM:SS``。"""
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def format_hms_ms(seconds: float) -> str:
    """秒 → ``HH:MM:SS.mmm``。"""
    seconds = max(0.0, float(seconds))
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
