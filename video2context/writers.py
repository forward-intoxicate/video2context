"""结果写出：JSON（主产物）、纯文本、SRT / VTT 字幕。"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .ffmpeg_tools import format_hms_ms

SUPPORTED_FORMATS: tuple[str, ...] = ("json", "txt", "srt", "vtt")


def _srt_timestamp(seconds: float) -> str:
    return format_hms_ms(seconds).replace(".", ",")


def segments_to_srt(segments: Sequence[Mapping[str, Any]]) -> str:
    """把分段列表转成 SRT 字幕文本。"""
    blocks: list[str] = []
    for index, seg in enumerate(segments, start=1):
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        blocks.append(
            f"{index}\n"
            f"{_srt_timestamp(float(seg.get('start', 0.0)))} --> {_srt_timestamp(float(seg.get('end', 0.0)))}\n"
            f"{text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def segments_to_vtt(segments: Sequence[Mapping[str, Any]]) -> str:
    """把分段列表转成 WebVTT（HTML5 <track> 直接用）。"""
    cues: list[str] = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        cues.append(
            f"{format_hms_ms(float(seg.get('start', 0.0)))} --> "
            f"{format_hms_ms(float(seg.get('end', 0.0)))}\n{text}"
        )
    body = "\n\n".join(cues)
    return f"WEBVTT\n\n{body}\n" if body else "WEBVTT\n"


def segments_to_txt(segments: Sequence[Mapping[str, Any]], *, with_timestamps: bool = False) -> str:
    """把分段列表转成纯文本；可选是否带时间戳前缀。"""
    lines: list[str] = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        if with_timestamps:
            lines.append(f"[{format_hms_ms(float(seg.get('start', 0.0)))}] {text}")
        else:
            lines.append(text)
    return "\n".join(lines) + ("\n" if lines else "")


def write_outputs(
    payload: Mapping[str, Any],
    out_base: str | Path,
    formats: Iterable[str] = ("json",),
) -> dict[str, Path]:
    """按需写出结果文件，返回 ``{格式: 路径}``。"""
    base = Path(out_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    segments = list(payload.get("segments", []))

    written: dict[str, Path] = {}
    for fmt in formats:
        key = fmt.lower().lstrip(".")
        if key == "json":
            path = base.with_suffix(".json")
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        elif key == "txt":
            path = base.with_suffix(".txt")
            path.write_text(segments_to_txt(segments), encoding="utf-8")
        elif key == "srt":
            path = base.with_suffix(".srt")
            path.write_text(segments_to_srt(segments), encoding="utf-8")
        elif key == "vtt":
            path = base.with_suffix(".vtt")
            path.write_text(segments_to_vtt(segments), encoding="utf-8")
        else:
            raise ValueError(f"不支持的输出格式：{fmt}（可选：{', '.join(SUPPORTED_FORMATS)}）")
        written[key] = path
    return written


def parse_formats(value: str | Iterable[str]) -> tuple[str, ...]:
    """解析 ``--formats json,txt`` 之类的输入。"""
    if isinstance(value, str):
        items = [item.strip().lower() for item in value.replace(";", ",").split(",")]
    else:
        items = [str(item).strip().lower() for item in value]
    formats = tuple(dict.fromkeys(item.lstrip(".") for item in items if item))
    if not formats:
        return ("json",)
    unknown = [f for f in formats if f not in SUPPORTED_FORMATS]
    if unknown:
        raise ValueError(f"不支持的输出格式：{', '.join(unknown)}（可选：{', '.join(SUPPORTED_FORMATS)}）")
    return formats
