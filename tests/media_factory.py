"""测试用的媒体素材工厂：用 ffmpeg 现造视频，不依赖任何外部文件。"""

from __future__ import annotations

import subprocess
from pathlib import Path

from video2context.ffmpeg_tools import find_ffmpeg


def build_tone_wav(path: Path, *, seconds: float = 2.0, frequency: int = 440) -> Path:
    """生成一段正弦波 wav（44.1kHz 立体声，用于验证重采样）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"sine=frequency={frequency}:duration={seconds}",
            "-ac", "2", "-ar", "44100", str(path),
        ],
        check=True,
    )
    return path


def build_video(path: Path, *, seconds: float = 2.0, with_audio: bool = True) -> Path:
    """生成带（或不带）音轨的 mp4 测试视频。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"color=c=navy:s=320x240:r=10:duration={seconds}",
    ]
    if with_audio:
        args += [
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-shortest", "-c:v", "mpeg4", "-c:a", "aac",
        ]
    else:
        args += ["-c:v", "mpeg4", "-an"]
    args.append(str(path))
    subprocess.run(args, check=True)
    return path
