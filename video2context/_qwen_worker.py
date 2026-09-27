"""Qwen3-ASR 子进程工人：在独立环境 ``.venv-qwen`` 里执行真正的识别。

为什么需要这个文件
------------------
Qwen3-ASR 依赖 ``torch`` + ``transformers``（装完约 3GB），而主工程刻意保持轻量：
faster-whisper + CTranslate2 走的是 ONNX/CTranslate2 路线，**不需要 torch**。
两者塞进同一个环境会互相牵制版本（例如 transformers 对 torch 版本很敏感），
所以这里用「子进程桥接」：

* 主进程（``.venv``）负责探测媒体、ffmpeg 抽音频、切句、写 JSON/SRT/VTT；
* 本脚本在 ``.venv-qwen`` 里被 ``subprocess`` 拉起，只做「读 wav → 出文字和时间戳 → 写 JSON」。

因此本文件**故意不 import 主工程任何模块**（那会连带导入 faster-whisper），
只使用标准库 + ``torch`` / ``transformers`` / ``qwen_asr``。

协议
----
命令行参数见 ``build_parser()``；日志一律走 **stderr**，结果写成 ``--out`` 指定的 JSON：

.. code-block:: json

    {
      "language": "Chinese",
      "text": "……",
      "time_stamps": [{"text": "今", "start": 0.12, "end": 0.34}, ...],
      "duration": 72.6, "load_seconds": 7.1, "elapsed_seconds": 8.5
    }

``time_stamps`` 只有传了 ``--aligner`` 才非空（强制对齐模型不仅体积大，
也是 SRT/VTT 精确分段的唯一来源）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any, Optional

#: 本工程推荐的默认值（与主工程 cli 保持一致）
DEFAULT_MAX_NEW_TOKENS = 4096


def log(message: str) -> None:
    """工人类日志统一走 stderr，stdout 留给结果（便于将来做管道）。"""
    print(f"[qwen] {message}", file=sys.stderr, flush=True)


def _drop_script_dir_from_path() -> None:
    """把本脚本所在目录（``video2context/``）从 sys.path 里摘掉。

    以文件路径运行脚本时 Python 会把脚本目录放到 ``sys.path[0]``，
    那里有 ``qwen_engine.py`` 之类的工程内部模块，会**遮住** site-packages 里的
    同名包，导致 ``import qwen_asr`` 拿错东西。工人只用标准库和第三方包，
    所以整段摘掉最省事。
    """
    here = str(Path(__file__).resolve().parent)
    sys.path[:] = [entry for entry in sys.path if str(Path(entry or ".").resolve()) != here]


def wav_duration_seconds(path: str | os.PathLike[str]) -> float:
    """用标准库读 wav 时长，避免为了一个数字再依赖 soundfile/librosa。"""
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate() or 1
            return frames / float(rate)
    except Exception:
        return 0.0


def _disable_allocator_warmup() -> bool:
    """跳过 transformers 加载前的显存预热。

    transformers 4.5x 会在加载模型前一次性 ``cudaMalloc`` 一大块（1.7B 约 3.2GB）
    来"预热"分配器。官方注释说这只是加载速度优化，但在 8GB 卡上会直接 OOM，
    所以默认跳过（可用 ``--no-skip-warmup`` 恢复）。
    """
    try:
        import transformers.modeling_utils as modeling_utils

        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None  # type: ignore[assignment]
        return True
    except Exception as exc:  # pragma: no cover - 版本差异时静默降级
        log(f"跳过显存预热失败（忽略）：{exc}")
        return False


def load_low_mem(model_dir: str, device: str = "cuda:0"):
    """低内存加载：分片直接读进显存，避开 transformers 把整个分片读进 CPU 内存。

    transformers 默认加载会把**整个 safetensors 分片**先读进 CPU 内存
    （1.7B 的第一个分片有 4GB），在提交内存（commit）吃紧的机器上会直接
    ``OSError: 页面文件太小``。这里改成：

        meta 设备建空模型 → 分片直接 ``load_file(device=...)`` 进显存 → assign 覆盖权重

    这样 CPU 侧几乎不占内存。代价是绕过了 ``from_pretrained`` 的常规流程，
    所以只在 ``--low-mem on`` 或自动模式下常规加载失败后使用。
    """
    import torch
    from qwen_asr import Qwen3ASRModel  # noqa: F401  (导入即注册 AutoModel)
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoProcessor

    directory = Path(model_dir)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModel.from_config(config)

    index_file = directory / "model.safetensors.index.json"
    if index_file.exists():
        weight_map = json.loads(index_file.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = ["model.safetensors"]

    missing: list[str] = []
    for shard in shards:
        state = load_file(str(directory / shard), device=device)
        result = model.load_state_dict(state, strict=False, assign=True)
        missing.extend(result.missing_keys)
        del state
    model.tie_weights()

    # 仍在 meta 上的张量（通常是 rotary 的 inv_freq 之类）需要实体化
    meta_buffers = [(name, buf) for name, buf in model.named_buffers() if buf.device.type == "meta"]
    for name, buf in meta_buffers:
        parts = name.split(".")
        target: Any = model
        for part in parts[:-1]:
            target = getattr(target, part) if not part.isdigit() else target[int(part)]
        setattr(target, parts[-1], torch.ones(tuple(buf.shape), dtype=torch.float32, device=device))

    model.eval()
    if meta_buffers:
        log(f"实体化了 {len(meta_buffers)} 个 meta buffer（如 {meta_buffers[0][0]}）")
    real_missing = [key for key in missing if "inv_freq" not in key]
    if real_missing:
        log(f"警告：权重缺失 {len(real_missing)} 个键：{real_missing[:3]}")

    processor = AutoProcessor.from_pretrained(model_dir, fix_mistral_regex=True)
    return model, processor


def build_model(args: argparse.Namespace, torch, Qwen3ASRModel):
    """按参数加载 ASR 模型（+ 可选的对齐模型），返回 Qwen3ASRModel。"""
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": device,
        "max_inference_batch_size": args.max_batch_size,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.aligner:
        kwargs["forced_aligner"] = args.aligner
        kwargs["forced_aligner_kwargs"] = {"dtype": torch.bfloat16, "device_map": device}

    if args.low_mem == "on":
        log("使用低内存加载（分片直接进显存）")
        model, processor = load_low_mem(args.model, device=device)
        forced_aligner = None
        if args.aligner:
            from qwen_asr import Qwen3ForcedAligner

            log(f"加载对齐模型 {args.aligner} …")
            forced_aligner = Qwen3ForcedAligner.from_pretrained(
                args.aligner,
                dtype=torch.bfloat16,
                device_map=device,
            )
        return Qwen3ASRModel(
            model=model,
            processor=processor,
            max_new_tokens=args.max_new_tokens,
            max_inference_batch_size=args.max_batch_size,
            forced_aligner=forced_aligner,
        )

    try:
        return Qwen3ASRModel.from_pretrained(args.model, **kwargs)
    except Exception as exc:
        if args.low_mem == "off" or not _looks_like_memory_error(exc):
            raise
        log(f"常规加载失败（{type(exc).__name__}: {exc}）→ 自动改用低内存加载重试")
        return build_model(_force_low_mem(args), torch, Qwen3ASRModel)


def _force_low_mem(args: argparse.Namespace) -> argparse.Namespace:
    clone = argparse.Namespace(**vars(args))
    clone.low_mem = "on"
    return clone


def _looks_like_memory_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in ("out of memory", "outofmemory", "页面文件太小", "paging file",
                       "insufficient memory", "memoryerror", "cannot allocate memory")
    )


def serialize_time_stamps(stamps: Any) -> list[dict[str, Any]]:
    """把 ForcedAlignResult 拍平成 ``[{"text", "start", "end"}]``。

    对齐结果在不同小版本里可能是 dataclass、dict 或 tuple，这里逐种容错，
    只要求最终能拿到「文本 + 起止秒」。
    """
    items = getattr(stamps, "items", stamps)
    if items is None:
        return []
    result: list[dict[str, Any]] = []
    for item in items:
        text = start = end = None
        if isinstance(item, dict):
            text = item.get("text")
            start = item.get("start_time", item.get("start"))
            end = item.get("end_time", item.get("end"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = item[0], item[1]
            text = item[2] if len(item) > 2 else ""
        else:
            text = getattr(item, "text", "")
            start = getattr(item, "start_time", getattr(item, "start", None))
            end = getattr(item, "end_time", getattr(item, "end", None))
        try:
            start_f, end_f = float(start), float(end)
        except (TypeError, ValueError):
            continue
        result.append({"text": str(text or ""), "start": round(start_f, 3), "end": round(end_f, 3)})
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-ASR 子进程工人（在 .venv-qwen 里运行）")
    parser.add_argument("--wav", required=True, help="16kHz 单声道 wav")
    parser.add_argument("--model", required=True, help="Qwen3-ASR 模型目录或仓库名")
    parser.add_argument("--aligner", default=None, help="Qwen3-ForcedAligner 目录，给出才返回时间戳")
    parser.add_argument("--language", default=None, help="规范语言名（Chinese/English/…），留空自动检测")
    parser.add_argument("--context", default=None, help="上下文（作为 system message 注入，用于偏置术语）")
    parser.add_argument("--out", required=True, help="结果 JSON 输出路径")
    parser.add_argument("--device", default="auto", help="auto/cuda:0/cpu")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument(
        "--low-mem",
        choices=["auto", "on", "off"],
        default="auto",
        help="auto=常规加载失败再退低内存；on=直接用低内存加载；off=不重试",
    )
    parser.add_argument("--no-skip-warmup", action="store_true", help="不跳过 transformers 的显存预热")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    _drop_script_dir_from_path()
    args = build_parser().parse_args(argv)

    # 小显存机器的两个必要处理（都是安全的）：
    # 1) expandable_segments 缓解显存碎片；2) 跳过 transformers 的分配器预热。
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if not args.no_skip_warmup and _disable_allocator_warmup():
        log("已跳过 transformers 的显存预热（小显存机器必需）")

    wav = Path(args.wav)
    if not wav.is_file():
        log(f"音频不存在：{wav}")
        return 2

    import torch
    from qwen_asr import Qwen3ASRModel

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log(f"torch {torch.__version__} | device={device} | cuda 可用={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        log(f"显存 空闲 {free / 2**30:.1f}GB / 共 {total / 2**30:.1f}GB")

    log(f"加载模型 {args.model} …")
    started = time.perf_counter()
    model = build_model(args, torch, Qwen3ASRModel)
    load_seconds = time.perf_counter() - started
    log(f"模型就绪，加载耗时 {load_seconds:.1f}s")

    call_kwargs: dict[str, Any] = {"audio": str(wav), "language": args.language or None}
    if args.context:
        call_kwargs["context"] = args.context
    if args.aligner:
        call_kwargs["return_time_stamps"] = True

    log(f"识别中（language={args.language or '自动检测'}{'，带 context' if args.context else ''}）…")
    started = time.perf_counter()
    results = model.transcribe(**call_kwargs)
    elapsed = time.perf_counter() - started

    result = results[0]
    text = (result.text or "").strip()
    stamps = serialize_time_stamps(getattr(result, "time_stamps", None)) if args.aligner else []

    payload = {
        "engine": "qwen3-asr",
        "model": str(args.model),
        "aligner": args.aligner,
        "language": getattr(result, "language", None),
        "text": text,
        "time_stamps": stamps,
        "duration": round(wav_duration_seconds(wav), 3),
        "load_seconds": round(load_seconds, 3),
        "elapsed_seconds": round(elapsed, 3),
        "device": device,
        "dtype": "bfloat16",
        "context": args.context or None,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    log(f"识别完成：{len(text)} 字，时间戳 {len(stamps)} 个，耗时 {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
