"""完整流水线：媒体探测 → 抽取音频 → 语音识别 → 写出结果。"""

from __future__ import annotations

import datetime as dt
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from . import __version__
from .ffmpeg_tools import (
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    extract_audio,
    format_hms,
    probe_media,
)
from .transcriber import DEFAULT_MODEL, get_transcriber
from .writers import SUPPORTED_FORMATS, write_outputs

ProgressCallback = Callable[[float, float, str], None]
LogCallback = Callable[[str], None]

_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass
class TranscribeOptions:
    """一次转写任务的全部可调参数。"""

    model: str = DEFAULT_MODEL
    device: str = "auto"
    compute_type: Optional[str] = None
    language: Optional[str] = None  # None = 自动检测
    task: str = "transcribe"  # transcribe | translate
    vad_filter: bool = True
    beam_size: int = 5
    batch_size: int = 0  # >1 且 GPU 时启用批量推理
    initial_prompt: Optional[str] = None
    condition_on_previous_text: bool = False
    temperature: Optional[float | list[float]] = None
    word_timestamps: bool = False
    formats: tuple[str, ...] = ("json",)
    output_dir: Path = Path("output")
    name: Optional[str] = None
    overwrite: bool = False
    keep_audio: bool = False
    hf_endpoint: Optional[str] = None
    model_dir: Optional[str] = None
    local_files_only: bool = False
    cpu_threads: int = 0
    start: Optional[float] = None
    duration: Optional[float] = None


@dataclass
class PipelineResult:
    """流水线产物。"""

    payload: dict[str, Any]
    outputs: dict[str, Path]
    audio_path: Optional[Path] = None
    elapsed_seconds: float = 0.0

    @property
    def json_path(self) -> Optional[Path]:
        return self.outputs.get("json")

    @property
    def text(self) -> str:
        return self.payload.get("text", "")

    @property
    def segments(self) -> list[dict[str, Any]]:
        return self.payload.get("segments", [])

    def summary(self) -> str:
        asr = self.payload.get("asr", {})
        return (
            f"语言={asr.get('language')}（置信度 {asr.get('language_probability')}）"
            f" 分段={asr.get('segments_count')}"
            f" 音频时长={format_hms(asr.get('duration') or 0)}"
            f" 处理耗时={asr.get('elapsed_seconds')}s"
            f" 实时率={asr.get('realtime_factor')}"
        )


def sanitize_name(name: str) -> str:
    """把文件名里非法或易出问题的字符替换掉。"""
    cleaned = _UNSAFE_CHARS.sub("_", str(name)).strip().strip(".")
    return cleaned or "transcript"


def _unique_base(output_dir: Path, stem: str) -> Path:
    """避免覆盖已有结果：report.json 存在则用 report-1.json。"""
    base = output_dir / stem
    if not any(base.with_suffix(f".{ext}").exists() for ext in SUPPORTED_FORMATS):
        return base
    index = 1
    while True:
        candidate = output_dir / f"{stem}-{index}"
        if not any(candidate.with_suffix(f".{ext}").exists() for ext in SUPPORTED_FORMATS):
            return candidate
        index += 1


def process(
    source: str | Path,
    options: Optional[TranscribeOptions] = None,
    *,
    progress_callback: Optional[ProgressCallback] = None,
    log_callback: Optional[LogCallback] = None,
) -> PipelineResult:
    """把一个视频/音频文件转成结构化文字结果。"""
    opts = options or TranscribeOptions()
    log: LogCallback = log_callback or (lambda _msg: None)

    src = Path(source).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"输入文件不存在：{src}")
    if not src.is_file():
        raise ValueError(f"输入不是文件：{src}")

    started = time.perf_counter()

    # 1) 媒体探测 --------------------------------------------------------
    log(f"[1/3] 探测媒体信息：{src.name}")
    media = probe_media(src)
    if not media.has_audio:
        raise RuntimeError(f"该文件不含音频轨道，无法转写：{src}")
    log(
        f"      时长 {media.duration_hms}（{media.duration:.1f}s）"
        f"｜格式 {media.format_name or '未知'}"
        f"｜视频轨 {'有' if media.has_video else '无'}"
    )

    # 2) 抽取音频 --------------------------------------------------------
    out_dir = Path(opts.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = sanitize_name(opts.name or src.stem)
    base = (out_dir / stem) if (opts.name or opts.overwrite) else _unique_base(out_dir, stem)

    temp_dir: Optional[Path] = None
    if opts.keep_audio:
        audio_path = out_dir / f"{base.name}.wav"
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="v2c-"))
        audio_path = temp_dir / "audio.wav"

    try:
        log(f"[2/3] 提取音频 → {TARGET_SAMPLE_RATE}Hz 单声道 wav")
        extract_audio(src, audio_path, start=opts.start, duration=opts.duration)
        size_mb = audio_path.stat().st_size / 1_048_576
        log(f"      音频就绪：{audio_path.name}（{size_mb:.1f} MB）")

        # 3) 语音识别 ----------------------------------------------------
        log(f"[3/3] 语音识别（模型 {opts.model}，任务 {opts.task}）")
        transcriber = get_transcriber(
            model=opts.model_dir or opts.model,
            device=opts.device,
            compute_type=opts.compute_type,
            local_files_only=opts.local_files_only,
            cpu_threads=opts.cpu_threads,
            hf_endpoint=opts.hf_endpoint,
            log_callback=log,
        )
        result = transcriber.transcribe(
            audio_path,
            language=opts.language,
            task=opts.task,
            vad_filter=opts.vad_filter,
            beam_size=opts.beam_size,
            batch_size=opts.batch_size,
            initial_prompt=opts.initial_prompt,
            condition_on_previous_text=opts.condition_on_previous_text,
            temperature=opts.temperature,
            word_timestamps=opts.word_timestamps,
            progress_callback=progress_callback,
            log_callback=log,
        )

        # 4) 组装并写出结果 ----------------------------------------------
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "generator": {"name": "video2context", "version": __version__},
            "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "source": {
                "path": str(src.resolve()),
                "filename": src.name,
                "size_bytes": src.stat().st_size,
                "duration": round(media.duration, 3),
                "duration_hms": media.duration_hms,
                "format": media.format_name,
                "has_video": media.has_video,
                "has_audio": media.has_audio,
                "clip": (
                    {"start": opts.start, "duration": opts.duration}
                    if (opts.start is not None or opts.duration is not None)
                    else None
                ),
            },
            "audio": {
                "path": str(audio_path.resolve()) if opts.keep_audio else None,
                "sample_rate": TARGET_SAMPLE_RATE,
                "channels": TARGET_CHANNELS,
                "codec": "pcm_s16le",
                "kept": bool(opts.keep_audio),
            },
            "asr": result.to_dict(),
            "text": result.text,
            "segments": [segment.to_dict() for segment in result.segments],
        }
        if payload["source"]["clip"] is None:
            payload["source"].pop("clip")

        log(f"      写出结果：{', '.join(opts.formats)}")
        outputs = write_outputs(payload, base, opts.formats)

        return PipelineResult(
            payload=payload,
            outputs=outputs,
            audio_path=audio_path if opts.keep_audio else None,
            elapsed_seconds=time.perf_counter() - started,
        )
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
