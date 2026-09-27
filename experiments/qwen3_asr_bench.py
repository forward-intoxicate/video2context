"""Qwen3-ASR 在本工程样例上的实测脚本（与 faster-whisper 对照）。

在 **独立环境** `.venv-qwen` 里运行：

    .\.venv-qwen\Scripts\python experiments\qwen3_asr_bench.py samples\formula.mp4 `
        --model models\Qwen3-ASR-1.7B

带时间戳（需要额外下载强制对齐模型，约 1.8GB）：

    .\.venv-qwen\Scripts\python experiments\qwen3_asr_bench.py samples\formula.mp4 `
        --model models\Qwen3-ASR-1.7B --aligner models\Qwen3-ForcedAligner-0.6B

与现有 faster-whisper 结果对照：

    ... --compare output\formula.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_ffmpeg() -> str:
    """优先用环境变量，其次 PATH，最后退回主工程 venv 里 imageio-ffmpeg 自带的那个。"""
    for candidate in (os.environ.get("FFMPEG_BIN"), shutil.which("ffmpeg")):
        if candidate and Path(candidate).is_file():
            return candidate
    bundled = sorted((PROJECT_ROOT / ".venv" / "Lib" / "site-packages" / "imageio_ffmpeg" / "binaries").glob("ffmpeg*"))
    if bundled:
        return str(bundled[-1])
    raise SystemExit("找不到 ffmpeg：请设置环境变量 FFMPEG_BIN 指向 ffmpeg 可执行文件")


def extract_audio(source: Path, dest: Path, *, duration: float | None = None) -> Path:
    """抽成 16kHz 单声道 wav（与主工程流水线一致）。"""
    args = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-nostdin"]
    if duration:
        args += ["-t", str(duration)]
    args += ["-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest)]
    subprocess.run(args, check=True)
    return dest


def load_low_mem(model_dir: str, device: str = "cuda:0"):
    """低内存加载（8GB 显存 + 16GB 内存这类机器的必需手段）。

    transformers 的默认加载会把**整个分片**先读进 CPU 内存（1.7B 的分片有 4GB），
    在提交内存（commit）吃紧的机器上会直接 `OSError: 页面文件太小`。
    这里改成：meta 设备建空模型 → 分片直接读进显存 → assign 覆盖权重，
    CPU 侧几乎不占内存。
    """
    import json

    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoProcessor

    from qwen_asr import Qwen3ASRModel  # noqa: F401  (导入即注册 AutoModel)

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
        state = load_file(str(directory / shard), device=device)  # 直接进显存
        result = model.load_state_dict(state, strict=False, assign=True)
        missing.extend(result.missing_keys)
        del state
    model.tie_weights()

    # 仍在 meta 上的张量（通常是 rotary 的 inv_freq 之类）需要实体化
    meta_buffers = [(n, b) for n, b in model.named_buffers() if b.device.type == "meta"]
    for name, buf in meta_buffers:
        parts = name.split(".")
        target = model
        for part in parts[:-1]:
            target = getattr(target, part) if not part.isdigit() else target[int(part)]
        shape = tuple(buf.shape)
        setattr(target, parts[-1], torch.ones(shape, dtype=torch.float32, device=device))

    model.eval()
    if meta_buffers:
        print(f"[i] 实体化了 {len(meta_buffers)} 个 meta buffer（如 {meta_buffers[0][0]}）")
    real_missing = [k for k in missing if "inv_freq" not in k]
    print(f"[i] 权重加载完成，缺失键 {len(real_missing)} 个" + (f"：{real_missing[:3]}" if real_missing else ""))

    processor = AutoProcessor.from_pretrained(model_dir, fix_mistral_regex=True)
    return model, processor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen3-ASR 实测")
    parser.add_argument("input", help="视频或音频文件")
    parser.add_argument("--model", default="models/Qwen3-ASR-1.7B", help="模型目录或仓库名")
    parser.add_argument("--aligner", default=None, help="Qwen3-ForcedAligner 目录（用于时间戳）")
    parser.add_argument("--language", default=None, help="Chinese / English / 留空自动检测")
    parser.add_argument("--duration", type=float, default=None, help="只处理前 N 秒（试跑用）")
    parser.add_argument("--compare", default=None, help="现有 faster-whisper 的 JSON，做对照输出")
    parser.add_argument("--out", default=None, help="把结果写成 JSON")
    parser.add_argument(
        "--low-mem",
        action="store_true",
        help="低内存加载：分片直接进显存，避开 transformers 把整个分片读进 CPU 内存",
    )
    args = parser.parse_args(argv)

    source = Path(args.input)
    if not source.exists():
        raise SystemExit(f"输入不存在：{source}")

    import torch
    from qwen_asr import Qwen3ASRModel

    # 小显存机器的两个必要处理（都是安全的）：
    # 1) transformers 4.5x 会在加载前"预热"显存分配器 —— 一次性 cudaMalloc 一大块
    #    （1.7B 模型要申请 3.2GB）。官方注释说它只是加载速度优化，
    #    但在 8GB 卡上会直接 OOM，所以这里跳过。
    # 2) expandable_segments 缓解碎片。
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        import transformers.modeling_utils as _modeling_utils

        _modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None  # type: ignore[assignment]
        print("[i] 已跳过 transformers 的显存预热（小显存机器必需）")
    except Exception as exc:  # pragma: no cover
        print(f"[i] 跳过显存预热失败（忽略）：{exc}")

    print("=" * 72)
    print(f"torch {torch.__version__} | cuda 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"显存: 空闲 {free / 2**30:.1f}GB / 共 {total / 2**30:.1f}GB")
    print("=" * 72)

    kwargs: dict = {
        "dtype": torch.bfloat16,
        "device_map": "cuda:0" if torch.cuda.is_available() else "cpu",
        "max_inference_batch_size": 8,
        "max_new_tokens": 256,
    }
    if args.aligner:
        kwargs["forced_aligner"] = args.aligner
        kwargs["forced_aligner_kwargs"] = dict(
            dtype=torch.bfloat16, device_map=kwargs["device_map"]
        )

    print(f"[1/3] 加载模型 {args.model} …")
    t0 = time.perf_counter()
    if args.low_mem:
        from qwen_asr import Qwen3ASRModel

        model, processor = load_low_mem(str(args.model), device=kwargs["device_map"])
        model = Qwen3ASRModel(
            model=model,
            processor=processor,
            max_new_tokens=kwargs["max_new_tokens"],
            max_inference_batch_size=kwargs["max_inference_batch_size"],
        )
    else:
        model = Qwen3ASRModel.from_pretrained(args.model, **kwargs)
    print(f"      加载耗时 {time.perf_counter() - t0:.1f}s")

    with tempfile.TemporaryDirectory() as tmp:
        wav = extract_audio(source, Path(tmp) / "audio.wav", duration=args.duration)
        size_mb = wav.stat().st_size / 1e6
        print(f"[2/3] 音频就绪：16kHz 单声道 wav（{size_mb:.1f} MB）")

        print(f"[3/3] 识别中（language={args.language or '自动检测'}）…")
        t0 = time.perf_counter()
        call_kwargs: dict = {"audio": str(wav), "language": args.language}
        if args.aligner:
            call_kwargs["return_time_stamps"] = True
        results = model.transcribe(**call_kwargs)
        elapsed = time.perf_counter() - t0

    result = results[0]
    text = (result.text or "").strip()
    print("\n" + "=" * 72)
    print(f"语言: {result.language} | 识别耗时 {elapsed:.2f}s")
    print("=" * 72)
    print(text)
    if args.aligner and getattr(result, "time_stamps", None):
        stamps = result.time_stamps[0]
        print(f"\n前 10 个时间戳: {stamps[:10]}")

    payload = {
        "engine": "qwen3-asr",
        "model": str(args.model),
        "aligner": args.aligner,
        "language": result.language,
        "elapsed_seconds": round(elapsed, 3),
        "text": text,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写出：{args.out}")

    if args.compare:
        whisper = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        print("\n" + "=" * 72)
        print("与 faster-whisper 对照")
        print("=" * 72)
        print(f"faster-whisper（{whisper.get('asr', {}).get('model')}）:")
        print("  " + whisper.get("text", "")[:400])
        print("\nQwen3-ASR:")
        print("  " + text[:400])

        def stats(label: str, value: str) -> None:
            print(
                f"{label:14} 右={value.count('右'):2d} 位={value.count('位'):2d} "
                f"导={value.count('导'):2d} u={value.count('u'):2d} v={value.count('v'):2d}"
            )

        print()
        stats("faster-whisper", whisper.get("text", ""))
        stats("Qwen3-ASR", text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
