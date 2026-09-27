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
from .config import llm_settings
from .ffmpeg_tools import (
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    extract_audio,
    format_hms,
    probe_media,
)
from .glossary import Glossary, build_glossary_from_scan, symbol_hit_score
from .llm import LLMClient
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
    # ---- 领域词表（两遍解码）----
    glossary: Optional[Glossary] = None
    auto_glossary: bool = False
    scan_duration: float = 90.0
    scan_model: Optional[str] = None
    glossary_out: Optional[Path] = None
    verify_glossary: bool = True
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None


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
        text = (
            f"语言={asr.get('language')}（置信度 {asr.get('language_probability')}）"
            f" 分段={asr.get('segments_count')}"
            f" 音频时长={format_hms(asr.get('duration') or 0)}"
            f" 处理耗时={asr.get('elapsed_seconds')}s"
            f" 实时率={asr.get('realtime_factor')}"
        )
        glossary = self.payload.get("glossary")
        if glossary:
            symbols = len(glossary.get("symbols") or [])
            text += f" 词表={symbols}个符号（来源 {glossary.get('source')}）"
            verification = glossary.get("verification") or {}
            if verification:
                text += f" 命中={verification.get('expected_hits')}"
                if verification.get("fallback_used"):
                    text += "（已回退到无词表版本）"
        return text


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


def _step(index: int, total: int, text: str, log: LogCallback) -> None:
    log(f"[{index}/{total}] {text}")


def scan_and_build_glossary(
    source: str | Path,
    options: Optional[TranscribeOptions] = None,
    *,
    log_callback: Optional[LogCallback] = None,
) -> Glossary:
    """只跑第一遍：采样粗转写 → 大模型推断词表（不出正式结果）。

    对应命令行的 ``--dry-run-glossary``：先把词表拿出来给人看一眼，
    确认没问题再跑第二遍，避免错误的词表把整片结果带偏。
    """
    opts = options or TranscribeOptions()
    log: LogCallback = log_callback or (lambda _msg: None)

    src = Path(source).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"输入文件不存在：{src}")

    media = probe_media(src)
    if not media.has_audio:
        raise RuntimeError(f"该文件不含音频轨道，无法转写：{src}")

    scan_seconds = min(opts.scan_duration, max(media.duration, 1.0))
    temp_dir = Path(tempfile.mkdtemp(prefix="v2c-scan-"))
    try:
        scan_audio = temp_dir / "scan.wav"
        log(f"[1/2] 采样前 {scan_seconds:.0f} 秒音频用于粗转写")
        extract_audio(src, scan_audio, start=opts.start, duration=scan_seconds)

        transcriber = get_transcriber(
            model=opts.model_dir or opts.scan_model or opts.model,
            device=opts.device,
            compute_type=opts.compute_type,
            local_files_only=opts.local_files_only,
            cpu_threads=opts.cpu_threads,
            hf_endpoint=opts.hf_endpoint,
            log_callback=log,
        )
        log("[2/2] 第一遍粗转写（无提示词、不带上文条件）")
        scan = transcriber.transcribe(
            scan_audio,
            language=opts.language,
            task=opts.task,
            vad_filter=opts.vad_filter,
            beam_size=opts.beam_size,
            batch_size=opts.batch_size,
            condition_on_previous_text=False,
            log_callback=log,
        )

        client = LLMClient(llm_settings(base_url=opts.llm_base_url, model=opts.llm_model))
        log(f"      粗转写 {len(scan.text)} 字 → 交给大模型（{client.settings.describe()}）")
        glossary = build_glossary_from_scan(
            scan.text,
            client=client,
            scan_meta={
                "duration": round(scan_seconds, 1),
                "model": opts.scan_model or opts.model,
                "chars": len(scan.text),
            },
        )
        log("      推断出词表：\n" + "\n".join("        " + line for line in glossary.describe().splitlines()))
        if opts.glossary_out:
            _write_glossary(glossary, Path(opts.glossary_out))
            log(f"      词表已写出：{opts.glossary_out}")
        return glossary
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _write_glossary(glossary: Glossary, path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(glossary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


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
    total_steps = 4 if (opts.auto_glossary and opts.glossary is None) else 3

    # 1) 媒体探测 --------------------------------------------------------
    log(f"[1/{total_steps}] 探测媒体信息：{src.name}")
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
        log(f"[2/{total_steps}] 提取音频 → {TARGET_SAMPLE_RATE}Hz 单声道 wav")
        extract_audio(src, audio_path, start=opts.start, duration=opts.duration)
        size_mb = audio_path.stat().st_size / 1_048_576
        log(f"      音频就绪：{audio_path.name}（{size_mb:.1f} MB）")

        transcriber = get_transcriber(
            model=opts.model_dir or opts.model,
            device=opts.device,
            compute_type=opts.compute_type,
            local_files_only=opts.local_files_only,
            cpu_threads=opts.cpu_threads,
            hf_endpoint=opts.hf_endpoint,
            log_callback=log,
        )

        # 3) 领域词表（可选，两遍解码的第一遍）------------------------------
        glossary = opts.glossary
        verification: Optional[dict[str, Any]] = None
        if opts.auto_glossary and glossary is None:
            scan_seconds = min(opts.scan_duration, max(media.duration, 1.0))
            log(f"[3/{total_steps}] 推断领域词表（采样前 {scan_seconds:.0f} 秒做粗转写）")
            scan_temp = Path(tempfile.mkdtemp(prefix="v2c-scan-"))
            try:
                scan_audio = scan_temp / "scan.wav"
                extract_audio(src, scan_audio, start=opts.start, duration=scan_seconds)
                scan_result = transcriber.transcribe(
                    scan_audio,
                    language=opts.language,
                    task=opts.task,
                    vad_filter=opts.vad_filter,
                    beam_size=opts.beam_size,
                    batch_size=opts.batch_size,
                    condition_on_previous_text=False,
                    log_callback=log,
                )
                client = LLMClient(llm_settings(base_url=opts.llm_base_url, model=opts.llm_model))
                log(f"      粗转写 {len(scan_result.text)} 字 → 大模型（{client.settings.describe()}）")
                glossary = build_glossary_from_scan(
                    scan_result.text,
                    client=client,
                    scan_meta={
                        "duration": round(scan_seconds, 1),
                        "model": opts.scan_model or opts.model,
                        "chars": len(scan_result.text),
                    },
                )
                log("      " + glossary.describe().replace("\n", "\n      "))
            finally:
                shutil.rmtree(scan_temp, ignore_errors=True)

        prompt_parts = []
        if glossary is not None and glossary.to_prompt():
            prompt_parts.append(glossary.to_prompt())
        if opts.initial_prompt:
            prompt_parts.append(opts.initial_prompt.strip())
        effective_prompt = " ".join(prompt_parts) or None

        # 4) 正式识别 ------------------------------------------------------
        log(f"[{total_steps}/{total_steps}] 语音识别（模型 {opts.model}，任务 {opts.task}）")
        result = transcriber.transcribe(
            audio_path,
            language=opts.language,
            task=opts.task,
            vad_filter=opts.vad_filter,
            beam_size=opts.beam_size,
            batch_size=opts.batch_size,
            initial_prompt=effective_prompt,
            condition_on_previous_text=opts.condition_on_previous_text,
            temperature=opts.temperature,
            word_timestamps=opts.word_timestamps,
            progress_callback=progress_callback,
            log_callback=log,
        )

        # 4.1) 验证词表偏置是否真的生效，失效就回退（实测会静默失效）---------
        if glossary is not None and opts.verify_glossary and glossary.symbols:
            score = symbol_hit_score(result.text, glossary)
            fallback_used = False
            if score.bias_failed:
                log(
                    "      ⚠ 词表偏置似乎失效（期望符号命中 0 次，"
                    f"已知错误写法出现 {score.wrong_hits} 次）→ 回退到无词表结果"
                )
                fallback = transcriber.transcribe(
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
                if fallback.text.strip():
                    result = fallback
                    fallback_used = True
            verification = {**score.to_dict(), "fallback_used": fallback_used}
            if fallback_used:
                log("      词表校验：错误写法仍在、正确符号一个未命中 → 已回退到无词表版本")
            elif score.partially_failed:
                log(
                    f"      词表校验：{len(score.hit_probes)} 个符号已修正，但仍有错误写法残留"
                    f"（{'、'.join(score.unresolved)}）—— 保留当前结果"
                )
            else:
                log(f"      词表校验：未发现残留错误写法，命中 {len(score.hit_probes)} 个符号")
        elif glossary is not None:
            log("      已跳过词表校验")

        if glossary is not None and opts.glossary_out:
            _write_glossary(glossary, Path(opts.glossary_out))
            log(f"      词表已写出：{opts.glossary_out}")

        # 5) 组装并写出结果 ------------------------------------------------
        payload: dict[str, Any] = {
            "schema_version": "1.1",
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
        if glossary is not None:
            glossary_data = glossary.to_dict()
            if verification:
                glossary_data["verification"] = verification
            payload["glossary"] = glossary_data

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
