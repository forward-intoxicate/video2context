"""命令行入口：``transcribe`` / ``webui`` / ``doctor``。"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from . import __version__
from .config import llm_settings
from .ffmpeg_tools import format_hms, format_hms_ms
from .glossary import Glossary
from .pipeline import (
    ENGINE_CHOICES,
    TranscribeOptions,
    is_qwen_engine,
    process,
    resolve_engine,
    scan_and_build_glossary,
)
from .transcriber import DEFAULT_MODEL, RECOMMENDED_MODELS
from .writers import SUPPORTED_FORMATS, parse_formats

EPILOG = """\
示例：
  # 最简：中文视频转文字，输出 output/xxx.json
  python -m video2context 我的视频.mp4

  # 英文视频，指定语言更快更准，并同时输出 srt 字幕
  python -m video2context talk.mp4 --language en --formats json,srt

  # 批量处理一个目录下的所有 mp4
  python -m video2context .\\videos\\*.mp4 --output-dir .\\output

  # 国内网络首次下载模型走镜像，并保留抽出的音频
  python -m video2context a.mp4 --hf-mirror --keep-audio

  # 只转写前 10 分钟（配音/长视频试跑）
  python -m video2context long.mp4 --duration 600

  # 用 Qwen3-ASR 引擎（中文同音词、数学符号更准；需要先跑 scripts/setup_qwen.ps1）
  python -m video2context 课程.mp4 --engine qwen3-asr --language zh -f json,srt

  # 先自检 Qwen3-ASR 环境（解释器 / 模型 / 对齐模型是否就绪）
  python -m video2context --engine qwen3-asr --qwen-setup

  # 打开本地网页界面
  python -m video2context webui

  # 环境自检（ffmpeg / 显卡 / 依赖是否就绪）
  python -m video2context doctor
"""


def _safe_console() -> None:
    """控制台编码自适配。

    * 输出被重定向（管道/文件）时强制 UTF-8，避免 GBK 环境下中文变乱码；
    * 直接在终端里运行时保持系统编码，只把无法表示的字符替换掉，避免抛异常。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream.isatty():
                stream.reconfigure(errors="replace")  # type: ignore[union-attr]
            else:
                stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass


class ProgressPrinter:
    """在终端用一行进度条显示识别进度。"""

    def __init__(self, enabled: bool = True, stream=None, width: int = 24) -> None:
        self.enabled = enabled
        self.stream = stream or sys.stderr
        self.width = width
        self._last_draw = 0.0
        self._active = False

    def __call__(self, processed: float, total: float, text: str = "") -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        finished = total > 0 and processed >= total - 0.05
        if self._active and not finished and (now - self._last_draw) < 0.2:
            return
        self._last_draw = now
        ratio = (processed / total) if total > 0 else 0.0
        ratio = min(max(ratio, 0.0), 1.0)
        filled = int(self.width * ratio)
        bar = "=" * filled + "-" * (self.width - filled)
        self.stream.write(
            f"\r  [{bar}] {ratio * 100:5.1f}%  {format_hms(processed)}/{format_hms(total)}"
        )
        self.stream.flush()
        self._active = True

    def close(self) -> None:
        if self._active:
            self.stream.write("\n")
            self.stream.flush()
            self._active = False


def _normalize_language(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip().lower()
    if cleaned in {"", "auto", "自动", "自动检测", "none"}:
        return None
    return cleaned


def _parse_temperature(value: Optional[str]) -> Optional[float | list[float]]:
    """``"0"`` → 0.0；``"0,0.2,0.4"`` → [0.0, 0.2, 0.4]；未指定 → None（用库默认回退序列）。"""
    if value is None or str(value).strip() == "":
        return None
    parts = [p.strip() for p in str(value).replace(";", ",").split(",") if p.strip()]
    temperatures = [float(p) for p in parts]
    return temperatures[0] if len(temperatures) == 1 else temperatures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video2context",
        description="视频/音频 → 文字：本地 ffmpeg 抽音轨 + faster-whisper 识别，支持中文/英文。",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-V", "--version", action="version", version=f"video2context {__version__}")
    sub = parser.add_subparsers(dest="command")

    # ---------------------------------------------------------- transcribe
    t = sub.add_parser(
        "transcribe",
        aliases=["t", "run"],
        help="转写视频/音频（默认命令）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    t.add_argument("inputs", nargs="*", help="输入文件，可一次给多个")

    out = t.add_argument_group("输出")
    out.add_argument("-o", "--output-dir", default="output", help="输出目录（默认 output）")
    out.add_argument("-n", "--name", default=None, help="输出文件名（不含扩展名，默认取输入文件名）")
    out.add_argument(
        "-f",
        "--formats",
        default="json",
        help=f"输出格式，逗号分隔，可选 {','.join(SUPPORTED_FORMATS)}（默认 json）",
    )
    out.add_argument("--overwrite", action="store_true", help="允许覆盖同名结果")
    out.add_argument("--keep-audio", action="store_true", help="保留抽出的 16k wav")
    out.add_argument("--stream", action="store_true", help="逐句实时打印识别结果")
    out.add_argument("--print-text", action="store_true", help="结束时把全文打印到标准输出")
    out.add_argument("-q", "--quiet", action="store_true", help="只输出结果路径，不显示进度")

    asr = t.add_argument_group("识别")
    asr.add_argument(
        "--engine",
        default=None,
        help=(
            "识别引擎：faster-whisper（默认，轻量、无需 torch、CPU 也能跑）"
            "或 qwen3-asr（中文同音词/数学符号明显更准，英文 WER 更低，但需要独立环境与约 4GB 显存）。"
            "不指定时读环境变量/`.env` 里的 V2C_ENGINE，没有就用 faster-whisper"
        ),
    )
    asr.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=(
            f"模型名或本地模型目录。faster-whisper 默认 {DEFAULT_MODEL}"
            f"（常用：{', '.join(RECOMMENDED_MODELS)}）；"
            "qwen3-asr 默认自动找 models/Qwen3-ASR-1.7B，找不到则用仓库名 Qwen/Qwen3-ASR-1.7B"
        ),
    )
    asr.add_argument(
        "-l",
        "--language",
        default="auto",
        help="语言：auto/zh/en/ja…（默认 auto 自动检测，明确指定可提速并减少幻觉）",
    )
    asr.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="transcribe=原语言转写；translate=翻译成英文",
    )
    asr.add_argument("--device", default="auto", help="auto/cuda/cpu（默认 auto）")
    asr.add_argument(
        "--compute-type",
        default=None,
        help="精度：float16/int8_float16/int8/float32（默认 GPU 用 float16，CPU 用 int8）",
    )
    asr.add_argument("--beam-size", type=int, default=5, help="束搜索宽度（默认 5，越大越准越慢）")
    asr.add_argument(
        "--batch-size",
        type=int,
        default=-1,
        help="批量推理大小；-1=自动（GPU 用 8，CPU 用 0），0=逐段模式",
    )
    asr.add_argument("--no-vad", action="store_true", help="关闭 VAD 静音过滤")
    asr.add_argument("--initial-prompt", default=None, help="提示词，可喂入专有名词/术语提高命中率")
    asr.add_argument(
        "--temperature",
        default=None,
        help="采样温度，单个值或逗号分隔的回退序列（默认用 Whisper 自带的 0.0,0.2,…,1.0 回退）",
    )
    asr.add_argument(
        "--condition-on-previous-text",
        action="store_true",
        help="把上一段结果作为上下文（连贯性更好，但长音频易陷入重复/幻觉）",
    )
    asr.add_argument("--word-timestamps", action="store_true", help="输出词级时间戳（JSON 里更详细）")
    asr.add_argument("--cpu-threads", type=int, default=0, help="CPU 线程数，0=自动")
    asr.add_argument("--model-dir", default=None, help="本地模型目录（离线使用）")
    asr.add_argument("--local-files-only", action="store_true", help="只用本地缓存，不联网下载")
    asr.add_argument("--hf-endpoint", default=None, help="HuggingFace 镜像地址，如 https://hf-mirror.com")
    asr.add_argument("--hf-mirror", action="store_true", help="等价于 --hf-endpoint https://hf-mirror.com")

    clip = t.add_argument_group("裁剪")
    clip.add_argument("--start", type=float, default=None, help="从第几秒开始（默认 0）")
    clip.add_argument("--duration", type=float, default=None, help="只处理多少秒")

    qw = t.add_argument_group("Qwen3-ASR（--engine qwen3-asr 时生效）")
    qw.add_argument(
        "--qwen-aligner",
        default=None,
        help=(
            "强制对齐模型目录（默认自动找 models/Qwen3-ForcedAligner-0.6B）。"
            "它决定 SRT/VTT 的时间戳是否精确；写 off 可关掉（省 1.8GB 磁盘，"
            "但字幕时间只能按说话区间估算）"
        ),
    )
    qw.add_argument(
        "--qwen-python",
        default=None,
        help="装了 qwen_asr 的解释器路径（默认自动找工程内 .venv-qwen，或读环境变量 V2C_QWEN_PYTHON）",
    )
    qw.add_argument(
        "--qwen-low-mem",
        choices=["auto", "on", "off"],
        default="auto",
        help="auto=常规加载失败再退低内存模式；on=总是用低内存加载（8GB 内存机器推荐）；off=不重试",
    )
    qw.add_argument(
        "--qwen-max-new-tokens",
        type=int,
        default=4096,
        help="单段最多生成多少 token（默认 4096；调小可省显存，调太小会截断长音频）",
    )
    qw.add_argument(
        "--qwen-batch-size",
        type=int,
        default=8,
        help="Qwen 内部一次并行推理多少段（默认 8；显存紧张时调小，最小 1）",
    )
    qw.add_argument(
        "--qwen-setup",
        action="store_true",
        help="先打印 Qwen3-ASR 的环境自检（解释器 / 模型 / 对齐模型），不转写",
    )

    gl = t.add_argument_group("领域词表（可选，用于修同音词/专有名词）")
    gl.add_argument(
        "--glossary",
        default=None,
        help="手写词表文件：一行一个；`错形 -> 正确` 表示还原（如 `右F4 -> u(x)`），单独一行视为术语",
    )
    gl.add_argument(
        "--auto-glossary",
        action="store_true",
        help="自动两遍解码：先粗转写采样片段 → 交给大模型推断词表 → 再正式转写",
    )
    gl.add_argument("--scan-duration", type=float, default=90.0, help="第一遍采样秒数（默认 90）")
    gl.add_argument(
        "--scan-model",
        default=None,
        help="第一遍用的模型（默认与正式相同 —— 同模型只加载一次，省内存）",
    )
    gl.add_argument("--glossary-out", default=None, help="把词表写到文件，便于复用与人工复核")
    gl.add_argument(
        "--dry-run-glossary",
        action="store_true",
        help="只生成词表并打印，不做正式转写（推荐先跑这个确认词表）",
    )
    gl.add_argument(
        "--no-verify-glossary",
        action="store_true",
        help="不校验偏置是否生效（默认会校验：期望符号命中 0 次就自动回退到无词表结果）",
    )
    gl.add_argument("--llm-base-url", default=None, help="大模型服务地址（默认读 V2C_LLM_BASE_URL）")
    gl.add_argument("--llm-model", default=None, help="大模型名（默认读 V2C_LLM_MODEL）")
    # 故意不提供 --llm-api-key：命令行参数会进进程列表与命令历史

    # --------------------------------------------------------------- webui
    w = sub.add_parser("webui", aliases=["web", "ui"], help="启动本地网页界面")
    w.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1，仅本机可访问）")
    w.add_argument("--port", type=int, default=7860, help="端口（默认 7860）")
    w.add_argument("--share", action="store_true", help="生成公网临时链接（Gradio 官方隧道）")
    w.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    w.add_argument("--model", default=DEFAULT_MODEL, help=f"默认模型（默认 {DEFAULT_MODEL}）")
    w.add_argument(
        "--model-dir",
        default=None,
        help="本地模型目录（配合 scripts/download_model.py 使用，会作为界面里的默认模型）",
    )
    w.add_argument("--device", default="auto", help="auto/cuda/cpu")
    w.add_argument(
        "--engine",
        default=None,
        help=f"界面里默认选中的识别引擎（{', '.join(ENGINE_CHOICES)}）；不指定时读环境变量 V2C_ENGINE",
    )
    w.add_argument("-o", "--output-dir", default="output", help="输出目录（默认 output）")

    # -------------------------------------------------------------- doctor
    sub.add_parser("doctor", help="环境自检：ffmpeg / 显卡 / 依赖 / 模型缓存")

    return parser


def _cmd_transcribe(args: argparse.Namespace) -> int:
    try:
        formats = parse_formats(args.formats)
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2

    if args.name and len(args.inputs) > 1:
        print("参数错误：--name 只能用于单个输入文件（多个输入会自动按各自文件名命名）", file=sys.stderr)
        return 2

    try:
        temperature = _parse_temperature(args.temperature)
    except ValueError:
        print(f"参数错误：--temperature 需要数字或逗号分隔的数字，收到 {args.temperature!r}", file=sys.stderr)
        return 2

    # ---- 引擎校验 -------------------------------------------------------
    engine = resolve_engine(args.engine)
    if engine not in ENGINE_CHOICES:
        source = "--engine" if args.engine else "环境变量 V2C_ENGINE"
        print(
            f"参数错误：不认识的引擎 {engine!r}（来自 {source}），可选：{', '.join(ENGINE_CHOICES)}",
            file=sys.stderr,
        )
        return 2

    if is_qwen_engine(engine):
        if args.task != "transcribe":
            print(
                "参数错误：Qwen3-ASR 只做原语言转写，不支持 --task translate。\n"
                "需要翻译成英文请用：--engine faster-whisper --task translate",
                file=sys.stderr,
            )
            return 2
        if args.qwen_setup:
            return _cmd_qwen_setup(args)
    elif args.qwen_setup:
        print("参数错误：--qwen-setup 需要配合 --engine qwen3-asr 使用", file=sys.stderr)
        return 2

    if not args.inputs:
        print("参数错误：至少要给一个输入文件（或用 --engine qwen3-asr --qwen-setup 只做环境自检）", file=sys.stderr)
        return 2

    # ---- 词表相关校验 ---------------------------------------------------
    if is_qwen_engine(engine) and (args.auto_glossary or args.dry_run_glossary):
        print(
            "提示：实测 Qwen3-ASR 的 context 对识别结果没有可观测影响"
            "（见 docs/models.md），--auto-glossary 目前收益有限。\n"
            "      词表仍会照常生成并写进结果 JSON，方便人工复核；"
            "想省掉这次额外开销可以去掉 --auto-glossary。",
            file=sys.stderr,
        )

    glossary: Optional[Glossary] = None
    if args.glossary:
        try:
            glossary = Glossary.from_file(args.glossary)
        except FileNotFoundError:
            print(f"参数错误：词表文件不存在：{args.glossary}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"参数错误：读取词表失败：{exc}", file=sys.stderr)
            return 2
        if not glossary.symbols and not glossary.terms:
            print(f"参数错误：词表文件是空的：{args.glossary}", file=sys.stderr)
            return 2

    want_scan = bool(args.auto_glossary or args.dry_run_glossary) and glossary is None
    if want_scan:
        settings = llm_settings(base_url=args.llm_base_url, model=args.llm_model)
        if not settings.configured:
            print(
                "错误：自动推断词表需要大模型密钥，但当前未配置。\n"
                "请在工程根目录建一个 .env 文件（已在 .gitignore 中，不会被提交）：\n"
                "    V2C_LLM_API_KEY=sk-你的密钥\n"
                "    V2C_LLM_BASE_URL=https://api.deepseek.com\n"
                "    V2C_LLM_MODEL=deepseek-chat\n"
                "或设置环境变量 V2C_LLM_API_KEY。也可以用 --glossary 提供手写词表，完全离线。",
                file=sys.stderr,
            )
            return 2

    options = TranscribeOptions(
        engine=engine,
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=_normalize_language(args.language),
        task=args.task,
        vad_filter=not args.no_vad,
        beam_size=args.beam_size,
        batch_size=args.batch_size,
        initial_prompt=args.initial_prompt,
        condition_on_previous_text=args.condition_on_previous_text,
        temperature=temperature,
        word_timestamps=args.word_timestamps,
        formats=formats,
        output_dir=Path(args.output_dir),
        name=args.name,
        overwrite=args.overwrite,
        keep_audio=args.keep_audio,
        hf_endpoint=("https://hf-mirror.com" if args.hf_mirror else args.hf_endpoint),
        model_dir=args.model_dir,
        local_files_only=args.local_files_only,
        cpu_threads=args.cpu_threads,
        start=args.start,
        duration=args.duration,
        glossary=glossary,
        auto_glossary=bool(args.auto_glossary or args.dry_run_glossary),
        scan_duration=args.scan_duration,
        scan_model=args.scan_model,
        glossary_out=Path(args.glossary_out) if args.glossary_out else None,
        verify_glossary=not args.no_verify_glossary,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        qwen_aligner=args.qwen_aligner,
        qwen_python=args.qwen_python,
        qwen_low_mem=args.qwen_low_mem,
        qwen_max_new_tokens=args.qwen_max_new_tokens,
        qwen_max_batch_size=args.qwen_batch_size,
    )

    quiet = args.quiet
    printer = ProgressPrinter(enabled=not quiet)
    log = (lambda _msg: None) if quiet else (lambda msg: print(f"  {msg}", file=sys.stderr))
    stream = (lambda line: print(line, flush=True)) if args.stream else None

    # ---- 只生成词表（推荐先跑这一步，人工确认后再正式转写）--------------
    if args.dry_run_glossary:
        for item in args.inputs:
            try:
                built = scan_and_build_glossary(item, options, log_callback=log)
            except Exception as exc:
                print(f"[失败] {item}：{exc}", file=sys.stderr)
                return 1
            print(built.describe())
            if options.glossary_out:
                print(f"已写出：{options.glossary_out}", file=sys.stderr)
        return 0

    def on_progress(processed: float, total: float, text: str) -> None:
        printer(processed, total, text)
        if stream and text:
            stream(f"[{format_hms_ms(processed)}] {text}")

    failures = 0
    multiple = len(args.inputs) > 1
    for index, item in enumerate(args.inputs, start=1):
        if multiple and not quiet:
            print(f"\n=== [{index}/{len(args.inputs)}] {item} ===", file=sys.stderr)
        try:
            result = process(
                item,
                options,
                progress_callback=on_progress,
                log_callback=log,
            )
        except KeyboardInterrupt:
            printer.close()
            print("\n已中断。", file=sys.stderr)
            return 130
        except Exception as exc:
            printer.close()
            failures += 1
            print(f"[失败] {item}：{exc}", file=sys.stderr)
            continue

        printer.close()
        if not quiet:
            print(f"  {result.summary()}", file=sys.stderr)
            for fmt, path in result.outputs.items():
                print(f"  -> {fmt.upper():4s} {path}", file=sys.stderr)
        else:
            for path in result.outputs.values():
                print(path)

        if args.print_text and result.text:
            print(result.text)

    return 1 if failures else 0


def _cmd_qwen_setup(args: argparse.Namespace) -> int:
    """打印 Qwen3-ASR 的环境自检（不转写），用来定位"跑不起来"的原因。"""
    from .qwen_engine import QwenAsrEngine, resolve_aligner, resolve_qwen_model

    engine = QwenAsrEngine(
        model=resolve_qwen_model(args.model if args.model != DEFAULT_MODEL else None),
        aligner=args.qwen_aligner,
        python=args.qwen_python,
        device=args.device,
        low_mem=args.qwen_low_mem,
        max_new_tokens=args.qwen_max_new_tokens,
        max_batch_size=args.qwen_batch_size,
    )
    print(f"video2context {__version__} · Qwen3-ASR 环境自检")
    print(f"  独立环境 python : {engine.python or '未找到（见下方提示）'}")
    print(f"  工人脚本        : {engine.worker}（存在={engine.worker.is_file()}）")
    print(f"  识别模型        : {engine.model}")

    model_path = Path(engine.model)
    if model_path.is_dir():
        print(f"  模型来源        : 本地目录（config.json={ (model_path / 'config.json').is_file() }）")
    else:
        print("  模型来源        : 仓库名 —— 首次运行会联网下载（约 4GB）")

    aligner = resolve_aligner(args.qwen_aligner)
    print(f"  对齐模型        : {aligner or '已关闭（字幕时间只能估算）'}")
    if aligner and Path(aligner).is_dir():
        size_mb = sum(f.stat().st_size for f in Path(aligner).rglob("*") if f.is_file()) / 1_048_576
        print(f"  对齐模型体积    : {size_mb:.0f} MB")
    print(f"  低内存模式      : {args.qwen_low_mem}")
    print(f"  单段最多 token  : {args.qwen_max_new_tokens}")
    print(f"  内部批大小      : {args.qwen_batch_size}")

    try:
        engine.preflight()
    except Exception as exc:
        print(f"\n[不可用] {exc}", file=sys.stderr)
        return 1
    print("\n环境就绪。示例：")
    print("  python -m video2context 视频.mp4 --engine qwen3-asr --language zh -f json,srt")
    return 0


def _cmd_doctor(_args: argparse.Namespace) -> int:
    import platform

    print(f"video2context {__version__} 环境自检")
    print(f"  Python            : {sys.version.split()[0]}  ({sys.executable})")
    print(f"  平台              : {platform.platform()}")

    try:
        from .ffmpeg_tools import ffmpeg_version, find_ffmpeg

        print(f"  ffmpeg 路径        : {find_ffmpeg()}")
        print(f"  ffmpeg 版本        : {ffmpeg_version()}")
    except Exception as exc:
        print(f"  ffmpeg            : 不可用 → {exc}")

    for module_name in ("faster_whisper", "ctranslate2", "imageio_ffmpeg", "gradio"):
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "已安装")
            print(f"  {module_name:<18}: {version}")
        except Exception as exc:
            print(f"  {module_name:<18}: 未安装（{type(exc).__name__}）")

    try:
        from .transcriber import cuda_device_count, register_cuda_dll_dirs

        dll_dirs = register_cuda_dll_dirs()
        print(f"  CUDA 设备数        : {cuda_device_count()}")
        if dll_dirs:
            print(f"  CUDA 运行库目录    : {', '.join(dll_dirs)}")
    except Exception as exc:
        print(f"  CUDA 检测          : 失败（{type(exc).__name__}: {exc}）")

    from .config import find_env_file

    env_file = find_env_file()
    print(f"  配置文件 .env      : {env_file if env_file else '未找到（不影响命令行基础功能）'}")
    print(f"  大模型（词表推断） : {llm_settings().describe()}")
    print(f"  默认识别引擎       : {resolve_engine()}（可用 --engine 覆盖，或在 .env 写 V2C_ENGINE）")

    import os

    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    print(f"  模型缓存目录       : {hf_home}")
    return 0


def _cmd_webui(args: argparse.Namespace) -> int:
    try:
        from .webui import launch
    except ImportError as exc:
        print(f"缺少 gradio 依赖：{exc}\n请执行：pip install gradio", file=sys.stderr)
        return 1
    launch(
        host=args.host,
        port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        default_model=args.model_dir or args.model,
        default_device=args.device,
        output_dir=args.output_dir,
        default_engine=resolve_engine(args.engine),
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    _safe_console()
    raw = list(sys.argv[1:] if argv is None else argv)

    known = {"transcribe", "t", "run", "webui", "web", "ui", "doctor", "-h", "--help", "-V", "--version"}
    if raw and raw[0] not in known:
        raw.insert(0, "transcribe")  # 支持 `video2context xxx.mp4` 直接转写
    if not raw:
        raw = ["transcribe", "-h"]

    parser = build_parser()
    args = parser.parse_args(raw)

    if args.command in {"transcribe", "t", "run"}:
        return _cmd_transcribe(args)
    if args.command in {"webui", "web", "ui"}:
        return _cmd_webui(args)
    if args.command == "doctor":
        return _cmd_doctor(args)

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
