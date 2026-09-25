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
from .ffmpeg_tools import format_hms, format_hms_ms
from .pipeline import TranscribeOptions, process
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
    t.add_argument("inputs", nargs="+", help="输入文件，可一次给多个")

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
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"模型名或本地模型目录（默认 {DEFAULT_MODEL}；常用：{', '.join(RECOMMENDED_MODELS)}）",
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

    options = TranscribeOptions(
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
    )

    quiet = args.quiet
    printer = ProgressPrinter(enabled=not quiet)
    log = (lambda _msg: None) if quiet else (lambda msg: print(f"  {msg}", file=sys.stderr))
    stream = (lambda line: print(line, flush=True)) if args.stream else None

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
