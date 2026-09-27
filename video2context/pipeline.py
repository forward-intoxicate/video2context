"""完整流水线：媒体探测 → 抽取音频 → 语音识别 → 写出结果。"""

from __future__ import annotations

import datetime as dt
import importlib.util
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

from . import __version__
from .config import DEFAULT_ENGINE, configured_engine, llm_settings
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

#: 识别引擎名（命令行 --engine 的可选值）
ENGINE_FASTER_WHISPER = "faster-whisper"
ENGINE_QWEN3_ASR = "qwen3-asr"
ENGINE_CHOICES: tuple[str, ...] = (ENGINE_FASTER_WHISPER, ENGINE_QWEN3_ASR)

#: 用户可能写的别名 → 规范引擎名
ENGINE_ALIASES: dict[str, str] = {
    "whisper": ENGINE_FASTER_WHISPER,
    "faster_whisper": ENGINE_FASTER_WHISPER,
    "fw": ENGINE_FASTER_WHISPER,
    "qwen": ENGINE_QWEN3_ASR,
    "qwen3": ENGINE_QWEN3_ASR,
    "qwen-asr": ENGINE_QWEN3_ASR,
    "qwen3_asr": ENGINE_QWEN3_ASR,
}

#: 都没指定时的引擎优先顺序（第一个「这台机器上可用」的胜出）
ENGINE_PREFERENCE: tuple[str, ...] = (ENGINE_QWEN3_ASR, ENGINE_FASTER_WHISPER)

_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def normalize_engine(value: Optional[str]) -> str:
    """把引擎名（含别名、大小写）归一化；空值按首选引擎处理。"""
    cleaned = (value or "").strip().lower().replace(" ", "")
    if not cleaned:
        return DEFAULT_ENGINE
    return ENGINE_ALIASES.get(cleaned, cleaned)


def is_qwen_engine(value: Optional[str]) -> bool:
    return normalize_engine(value) == ENGINE_QWEN3_ASR


def available_engines() -> list[str]:
    """这台机器上**现在就能用**的引擎。

    * Qwen3-ASR：只要独立环境（``.venv-qwen`` 或 ``V2C_QWEN_PYTHON``）存在就算可用 ——
      模型没下载的话首次运行会自己去下（与 Whisper 的行为一致）；
    * faster-whisper：看包装没装（用 ``find_spec`` 探测，不真的 import，省几百毫秒）。

    这是「本机部署了哪个引擎」的判定依据：``doctor`` 用它报告环境，
    默认引擎的选择也用它。
    """
    from .qwen_engine import find_qwen_python

    found: list[str] = []
    if find_qwen_python() is not None:
        found.append(ENGINE_QWEN3_ASR)
    try:
        if importlib.util.find_spec("faster_whisper") is not None:
            found.append(ENGINE_FASTER_WHISPER)
    except (ImportError, ValueError):  # pragma: no cover - 环境异常时当作没装
        pass
    return found


def resolve_engine(
    value: Optional[str] = None,
    *,
    detect: bool = True,
    available: Optional[list[str]] = None,
) -> str:
    """决定本次用哪个引擎。

    优先级：**显式参数 > `V2C_ENGINE`（环境变量或 .env）> 本机可用的引擎 > 首选默认**。

    第三档是"按部署情况自适应"：用户只装了 Whisper 就用 Whisper，只装了 Qwen 就用 Qwen，
    两个都装了按 :data:`ENGINE_PREFERENCE`（Qwen 优先）。
    这样无论走哪条部署路线，**默认都能直接跑起来**，不需要用户去记自己装了什么。

    两个都没装时返回首选引擎（Qwen），随后 preflight 会给出对应的安装指引 ——
    比抛一个"找不到模块"更有用。

    ``available`` 只是给测试注入用的（真实调用走 :func:`available_engines`）。
    """
    if (value or "").strip():
        return normalize_engine(value)

    # 注意：这里必须先判空再归一化 —— normalize_engine("") 会返回首选引擎（非空），
    # 直接 `if normalize_engine(configured_engine()):` 会让自动探测永远轮不到。
    configured = configured_engine()
    if configured.strip():
        return normalize_engine(configured)

    if detect:
        installed = available_engines() if available is None else list(available)
        for candidate in ENGINE_PREFERENCE:
            if candidate in installed:
                return candidate

    return DEFAULT_ENGINE


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
    # ---- 识别引擎 ----
    #: 空串 = 未指定，按 V2C_ENGINE / 默认解析（见 resolve_engine）
    engine: str = ""
    qwen_aligner: Optional[str] = None  # 强制对齐模型（None=自动探测，"off"=关闭）
    qwen_python: Optional[str] = None  # .venv-qwen 解释器（None=自动探测）
    qwen_low_mem: str = "auto"  # auto | on | off
    qwen_max_new_tokens: int = 4096
    qwen_max_batch_size: int = 8


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


def build_transcriber(opts: TranscribeOptions, log: LogCallback):
    """按 ``opts.engine`` 造出识别器（faster-whisper 或 Qwen3-ASR）。

    两个引擎的接口一致（``.transcribe(...)`` 返回 :class:`TranscriptionResult`），
    所以流水线后面的逻辑不用区分引擎。

    Qwen3-ASR 走独立环境 + 子进程，见 :mod:`video2context.qwen_engine`。
    """
    engine = resolve_engine(opts.engine)
    if engine == ENGINE_QWEN3_ASR:
        from .qwen_engine import get_qwen_engine, resolve_qwen_model

        # --model 的默认值是 Whisper 的 large-v3；用 Qwen 引擎时它显然不是用户的本意，
        # 当成"没指定"处理，交给 Qwen 的默认模型（本地 models/Qwen3-ASR-1.7B 或 HF 仓库名）。
        requested = opts.model
        if requested in (None, "", DEFAULT_MODEL):
            requested = None
            if opts.model == DEFAULT_MODEL:
                log(f"提示：--engine {ENGINE_QWEN3_ASR} 下未指定 --model，自动选用 Qwen 默认模型")
        return get_qwen_engine(
            model=resolve_qwen_model(requested),
            aligner=opts.qwen_aligner,
            python=opts.qwen_python,
            device=opts.device,
            low_mem=opts.qwen_low_mem,
            max_new_tokens=opts.qwen_max_new_tokens,
            max_batch_size=opts.qwen_max_batch_size,
            log_callback=log,
        )

    return get_transcriber(
        model=opts.model_dir or opts.model,
        device=opts.device,
        compute_type=opts.compute_type,
        local_files_only=opts.local_files_only,
        cpu_threads=opts.cpu_threads,
        hf_endpoint=opts.hf_endpoint,
        log_callback=log,
    )


def glossary_prompt_for(opts: TranscribeOptions, glossary: Optional[Glossary]) -> Optional[str]:
    """把词表 + 用户提示词拼成当前引擎需要的"提示"。

    * faster-whisper：极短的 ``initial_prompt``（实测 30~80 字最稳）；
    * Qwen3-ASR：``context`` 会作为 **system message** 注入。实测**指令式**的短词表
      会让模型复述词表而不转写，所以这里只用"描述视频里有什么"的完整句子。

    注意：进一步实测发现 Qwen 的 ``context`` 对结果**没有可观测影响**
    （换成完全无关的内容输出逐字相同，见 docs/models.md 第 6.4 节）。
    这里仍然照常拼装，是因为它无害、词表本身有复核价值，
    且上游一旦让它生效本工程不用改代码。
    """
    parts: list[str] = []
    if glossary is not None:
        if is_qwen_engine(resolve_engine(opts.engine)):
            context = glossary.to_context()
            if context:
                parts.append(context)
        elif glossary.to_prompt():
            parts.append(glossary.to_prompt())
    if opts.initial_prompt:
        parts.append(opts.initial_prompt.strip())
    return " ".join(part for part in parts if part) or None


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

        scan_opts = opts
        if opts.scan_model and opts.scan_model != opts.model:
            # 第一遍可以换个更小的模型，省时省显存
            scan_opts = replace(opts, model=opts.scan_model)
        transcriber = build_transcriber(scan_opts, log)
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

        transcriber = build_transcriber(opts, log)

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

        effective_prompt = glossary_prompt_for(opts, glossary)
        engine = resolve_engine(opts.engine)
        if effective_prompt:
            log(f"      识别提示（{len(effective_prompt)} 字）：{effective_prompt}")
            if is_qwen_engine(engine):
                # 实测结论，别让用户以为它有用（见 docs/models.md「context 实测」）
                log(
                    "      注意：Qwen3-ASR 的 context 在本工程实测中对结果**没有可观测影响**"
                    "（换成完全无关的内容，输出逐字相同）。词表仍会写进结果 JSON 备查，"
                    "但不要指望它修同音词。"
                )

        # 4) 正式识别 ------------------------------------------------------
        active_model = getattr(transcriber, "model", opts.model)
        log(f"[{total_steps}/{total_steps}] 语音识别（引擎 {engine}，模型 {active_model}，任务 {opts.task}）")
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

        # 4.1) 验证词表偏置是否真的生效，失效就回退（实测 faster-whisper 会静默失效）---
        if glossary is not None and opts.verify_glossary and glossary.symbols:
            score = symbol_hit_score(result.text, glossary)
            fallback_used = False
            # Qwen3-ASR 的 context 是 system message，不是"窗口级提示"，不存在
            # faster-whisper 那种"同一提示措辞一变就静默失效"的行为；而且重跑一次
            # 要重新加载模型（约 10s），代价不小，所以只报数不回退。
            if score.bias_failed and not is_qwen_engine(engine):
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
            elif score.bias_failed:
                log(
                    f"      ⚠ 结果里仍有 {score.wrong_hits} 处已知错误写法"
                    f"（{'、'.join(score.unresolved)}），且正确符号一个未命中 —— "
                    "Qwen 引擎不会自动回退，可换 --engine faster-whisper 对照，"
                    "或调整 --initial-prompt 的写法"
                )
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
