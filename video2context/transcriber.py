"""faster-whisper 封装层。

职责：
    * 延迟导入（``--help``、``doctor`` 不需要加载模型依赖）
    * 自动选择设备：cuda/float16 → cuda/int8_float16 → cpu/int8
    * Windows 下把 pip 安装的 ``nvidia-*-cu12`` 运行库注册进 DLL 搜索路径
    * 把生成器式的识别结果收集成结构化对象，并回调进度
"""

from __future__ import annotations

import contextlib
import inspect
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

DEFAULT_MODEL = "large-v3"

#: 这些精度只适用于 GPU，CPU 上加载会失败
CPU_UNSUPPORTED_COMPUTE_TYPES = frozenset(
    {"float16", "int8_float16", "bfloat16", "int8_bfloat16"}
)

#: faster-whisper 官方支持的全部模型名（与 faster_whisper.utils._MODELS 对齐）
MODEL_CHOICES: tuple[str, ...] = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "distil-small.en",
    "distil-medium.en",
    "distil-large-v3",
    "distil-large-v3.5",
)

#: 命令行帮助里只列最常用的几个，避免刷屏
RECOMMENDED_MODELS: tuple[str, ...] = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "large-v3-turbo",
    "distil-large-v3",
)

#: 语言下拉选项（None 表示自动检测）
LANGUAGE_CHOICES: tuple[tuple[str, Optional[str]], ...] = (
    ("自动检测", None),
    ("中文 zh", "zh"),
    ("英文 en", "en"),
    ("日文 ja", "ja"),
    ("韩文 ko", "ko"),
    ("粤语 yue", "yue"),
    ("法文 fr", "fr"),
    ("德文 de", "de"),
    ("西班牙文 es", "es"),
    ("俄文 ru", "ru"),
)

ProgressCallback = Callable[[float, float, str], None]
LogCallback = Callable[[str], None]


@dataclass
class Segment:
    """一句话（一个识别片段）。"""

    id: int
    start: float
    end: float
    text: str
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    temperature: Optional[float] = None
    compression_ratio: Optional[float] = None
    words: Optional[list[dict[str, Any]]] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["words"] is None:
            data.pop("words")
        return data


@dataclass
class TranscriptionResult:
    """一次完整识别任务的结果与统计信息。"""

    segments: list[Segment] = field(default_factory=list)
    text: str = ""
    language: Optional[str] = None
    language_probability: Optional[float] = None
    duration: float = 0.0
    duration_after_vad: float = 0.0
    task: str = "transcribe"
    model: str = DEFAULT_MODEL
    device: str = "cpu"
    compute_type: str = "int8"
    batch_size: int = 0
    vad_filter: bool = True
    beam_size: int = 5
    initial_prompt: Optional[str] = None
    condition_on_previous_text: bool = False
    temperature: Optional[float | list[float]] = None
    elapsed_seconds: float = 0.0
    model_load_seconds: float = 0.0
    #: 产生这份结果的引擎（faster-whisper / qwen3-asr），写进 JSON 便于溯源
    engine: str = "faster-whisper"
    #: 引擎私有信息（Qwen 的对齐模型、时间戳来源等），会平铺进 to_dict()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def realtime_factor(self) -> float:
        """处理时长 / 音频时长，越小越快。"""
        if self.duration <= 0:
            return 0.0
        return self.elapsed_seconds / self.duration

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "engine": self.engine,
            "model": self.model,
            "device": self.device,
            "compute_type": self.compute_type,
            "task": self.task,
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": round(self.duration, 3),
            "duration_after_vad": round(self.duration_after_vad, 3),
            "vad_filter": self.vad_filter,
            "beam_size": self.beam_size,
            "batch_size": self.batch_size,
            "temperature": self.temperature,
            "condition_on_previous_text": self.condition_on_previous_text,
            "initial_prompt": self.initial_prompt,
            "segments_count": len(self.segments),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "model_load_seconds": round(self.model_load_seconds, 3),
            "realtime_factor": round(self.realtime_factor, 4),
        }
        data.update(self.extra)
        return data


def cuda_device_count() -> int:
    """CTranslate2 可见的 CUDA 设备数；不可用时返回 0。"""
    try:
        import ctranslate2  # type: ignore

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def free_gpu_memory_mb() -> Optional[int]:
    """查询第一块显卡的剩余显存（MB）；查不到返回 None。

    用 nvidia-smi 而不是额外的 Python 依赖，失败时静默降级。
    """
    for args in (
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
    ):
        try:
            proc = subprocess.run(
                args, capture_output=True, text=True, timeout=10, encoding="utf-8", errors="replace"
            )
            first_line = (proc.stdout or "").strip().splitlines()
            if proc.returncode == 0 and first_line:
                return int(float(first_line[0].strip()))
        except Exception:
            continue
    return None


def register_cuda_dll_dirs() -> list[str]:
    """Windows：把 pip 安装的 nvidia cuBLAS / cuDNN 目录加入 DLL 搜索路径。

    ``pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`` 会把 DLL 放在
    ``site-packages/nvidia/{cublas,cudnn}/bin``，ctranslate2 默认找不到它们。
    """
    added: list[str] = []
    if os.name != "nt":
        return added

    roots: list[str] = []
    try:
        import site

        roots.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        import sysconfig

        paths = sysconfig.get_paths()
        roots.extend(p for p in (paths.get("purelib"), paths.get("platlib")) if p)
    except Exception:
        pass

    for root in dict.fromkeys(r for r in roots if r):
        for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
            dll_dir = Path(root) / sub
            if not dll_dir.is_dir():
                continue
            with contextlib.suppress(Exception):  # 注册失败还有下面的 PATH 兜底
                os.add_dll_directory(str(dll_dir))  # type: ignore[attr-defined]
            os.environ["PATH"] = f"{dll_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            added.append(str(dll_dir))
    return added


class Transcriber:
    """faster-whisper 模型的薄封装，模型只在首次使用时加载。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str = "auto",
        compute_type: Optional[str] = None,
        download_root: Optional[str | os.PathLike[str]] = None,
        local_files_only: bool = False,
        cpu_threads: int = 0,
        num_workers: int = 1,
        hf_endpoint: Optional[str] = None,
        log_callback: Optional[LogCallback] = None,
    ) -> None:
        self.model = str(model)
        self.device = (device or "auto").lower()
        self.compute_type = compute_type
        self.download_root = str(download_root) if download_root else None
        self.local_files_only = local_files_only
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers
        self.hf_endpoint = hf_endpoint
        self.log = log_callback or (lambda _msg: None)

        self._model = None
        self.resolved_device: str = "?"
        self.resolved_compute_type: str = "?"
        self.model_load_seconds: float = 0.0

    # ------------------------------------------------------------------ 加载

    def _cpu_compute_type(self) -> str:
        """CPU 支持的精度与 GPU 不同：float16 / int8_float16 等在 CPU 上不可用。"""
        if self.compute_type and self.compute_type not in CPU_UNSUPPORTED_COMPUTE_TYPES:
            return self.compute_type
        if self.compute_type:
            self.log(
                f"提示：compute_type={self.compute_type} 在 CPU 上不受支持，"
                "已自动改用 int8（CPU 支持：int8 / int8_float32 / float32 / int16）"
            )
        return "int8"

    def _device_attempts(self) -> list[tuple[str, str]]:
        """按优先级给出 (device, compute_type) 尝试列表。"""
        wanted = self.device
        cuda_types = [self.compute_type] if self.compute_type else ["float16", "int8_float16"]

        if wanted == "cpu":
            return [("cpu", self._cpu_compute_type())]
        if wanted.startswith("cuda"):
            return [(wanted, ct) for ct in cuda_types]
        # auto：有 CUDA 就先试 GPU，失败再退 CPU（注意 CPU 要用 CPU 支持的精度）
        attempts: list[tuple[str, str]] = []
        if cuda_device_count() > 0:
            attempts += [("cuda", ct) for ct in cuda_types]
        attempts.append(("cpu", self._cpu_compute_type()))
        return attempts

    def load(self) -> Transcriber:
        """加载模型（幂等）。"""
        if self._model is not None:
            return self

        if self.hf_endpoint:
            os.environ["HF_ENDPOINT"] = self.hf_endpoint

        # huggingface_hub 新版默认走 Xet 存储，部分网络（尤其国内）会长时间无响应地挂起；
        # 默认退回普通 HTTP 下载，稳定且可断点续传。需要重新开启可设 HF_HUB_DISABLE_XET=0。
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        dll_dirs = register_cuda_dll_dirs()
        if dll_dirs:
            self.log(f"已注册 CUDA 运行库目录：{', '.join(dll_dirs)}")

        try:
            from faster_whisper import WhisperModel  # 延迟导入
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "缺少 faster-whisper 依赖，请先执行：pip install -r requirements.txt"
            ) from exc

        errors: list[str] = []
        for attempt_round in range(2):
            for device, compute_type in self._device_attempts():
                try:
                    self.log(f"加载模型 {self.model}（device={device}, compute_type={compute_type}）…")
                    started = time.perf_counter()
                    self._model = WhisperModel(
                        self.model,
                        device=device,
                        compute_type=compute_type,
                        download_root=self.download_root,
                        local_files_only=self.local_files_only,
                        cpu_threads=self.cpu_threads,
                        num_workers=self.num_workers,
                    )
                    self.model_load_seconds = time.perf_counter() - started
                    self.resolved_device = device
                    self.resolved_compute_type = compute_type
                    self.log(
                        f"模型就绪：device={device}, compute_type={compute_type}, "
                        f"耗时 {self.model_load_seconds:.1f}s"
                    )
                    return self
                except Exception as exc:
                    errors.append(
                        f"  - device={device}, compute_type={compute_type} → {type(exc).__name__}: {exc}"
                    )

            # 首次失败且用户没指定镜像时，自动改用 hf-mirror 再试一轮（国内网络常见）
            if attempt_round == 0 and not os.environ.get("HF_ENDPOINT") and not self.local_files_only:
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                self.log("直连 HuggingFace 失败，自动改用镜像 https://hf-mirror.com 重试…")

        raise RuntimeError(
            "模型加载失败，已尝试以下配置：\n"
            + "\n".join(errors)
            + "\n\n排查建议：\n"
            "  * 首次使用需联网下载模型（large-v3 约 3GB）；国内网络可加 --hf-mirror 使用镜像\n"
            "  * 显卡不可用时可加 --device cpu --compute-type int8\n"
            "  * 模型名写错或本地路径不存在"
        )

    # ------------------------------------------------------------------ 识别

    @staticmethod
    def _invoke(func: Callable[..., Any], audio_path: str, **kwargs: Any):
        """按目标函数签名过滤参数，兼容不同版本的 faster-whisper。"""
        accepted = set(inspect.signature(func).parameters)
        clean = {k: v for k, v in kwargs.items() if k in accepted and v is not None}
        return func(audio_path, **clean)

    def _fit_batch_to_free_memory(self, batch_size: int, log: LogCallback) -> int:
        """按当前剩余显存收缩批量大小，尽量避免推理中途 OOM。"""
        free_mb = free_gpu_memory_mb()
        if free_mb is None:
            return batch_size

        # 经验值：固定开销/碎片约 700MB，每一路批量（30 秒窗口）约 700MB。
        # 实测 large-v3 在剩余 4.0GB 时开 7 路会 OOM、4 路稳定，所以这里留足余量。
        affordable = int(max(free_mb - 700, 0) // 700)
        if affordable >= batch_size:
            return batch_size
        if affordable <= 1:
            log(f"剩余显存约 {free_mb}MB，批量推理放不下 → 改用逐段模式")
            return 0
        log(f"剩余显存约 {free_mb}MB → 批量大小 {batch_size} 收缩为 {affordable}")
        return affordable

    def transcribe(
        self,
        audio_path: str | os.PathLike[str],
        *,
        language: Optional[str] = None,
        task: str = "transcribe",
        vad_filter: bool = True,
        beam_size: int = 5,
        batch_size: int = -1,
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: bool = False,
        temperature: Optional[float | list[float]] = None,
        word_timestamps: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        log_callback: Optional[LogCallback] = None,
    ) -> TranscriptionResult:
        """识别音频，返回结构化结果。"""
        self.load()
        log = log_callback or self.log
        audio = str(audio_path)

        options: dict[str, Any] = {
            "language": language,
            "task": task,
            "beam_size": beam_size,
            "vad_filter": vad_filter,
            "vad_parameters": {"min_silence_duration_ms": 500, "speech_pad_ms": 200} if vad_filter else None,
            "initial_prompt": initial_prompt,
            "condition_on_previous_text": condition_on_previous_text,
            "temperature": temperature,
            "word_timestamps": word_timestamps,
        }

        started = time.perf_counter()
        if batch_size is None or batch_size < 0:  # -1 = 自动：GPU 批量、CPU 逐段
            batch_size = 8 if self.resolved_device.startswith("cuda") else 0
        use_batch = batch_size > 1 and self.resolved_device.startswith("cuda")
        effective_batch = batch_size if use_batch else 0
        if effective_batch:
            effective_batch = self._fit_batch_to_free_memory(effective_batch, log)

        try:
            segments, info = self._collect(audio, options, effective_batch, log, progress_callback)
        except Exception as exc:
            memory_issue = _is_out_of_memory(exc)
            context_broken = _is_cuda_context_broken(exc)
            if not (memory_issue or context_broken):
                raise
            if context_broken:
                # 上下文已失效，同进程内重试没有意义，直接给出可执行的建议
                raise _memory_error_help(exc) from exc
            if not effective_batch:
                raise _memory_error_help(exc) from exc

            log(f"显存不足（{exc}），自动改用逐段模式重试…")
            effective_batch = 0
            try:
                segments, info = self._collect(audio, options, 0, log, progress_callback)
            except Exception as retry_exc:
                raise _memory_error_help(retry_exc) from retry_exc

        elapsed = time.perf_counter() - started
        total = float(getattr(info, "duration_after_vad", 0.0) or getattr(info, "duration", 0.0) or 0.0)
        full_text = "".join(s.text for s in segments).strip()

        return TranscriptionResult(
            segments=segments,
            text=full_text,
            language=getattr(info, "language", None),
            language_probability=_opt_float(getattr(info, "language_probability", None)),
            duration=float(getattr(info, "duration", 0.0) or 0.0),
            duration_after_vad=total,
            task=task,
            model=self.model,
            device=self.resolved_device,
            compute_type=self.resolved_compute_type,
            batch_size=effective_batch,
            vad_filter=vad_filter,
            beam_size=beam_size,
            initial_prompt=initial_prompt,
            condition_on_previous_text=condition_on_previous_text,
            temperature=temperature,
            elapsed_seconds=elapsed,
            model_load_seconds=self.model_load_seconds,
        )

    def _collect(
        self,
        audio: str,
        options: dict[str, Any],
        batch_size: int,
        log: LogCallback,
        progress_callback: Optional[ProgressCallback],
    ) -> tuple[list[Segment], Any]:
        """跑一次识别并把生成器收集成列表（会消费进度回调）。"""
        if batch_size > 1:
            try:
                from faster_whisper import BatchedInferencePipeline  # type: ignore

                pipeline = BatchedInferencePipeline(model=self._model)
                log(f"使用批量推理（batch_size={batch_size}）")
                segment_iter, info = self._invoke(
                    pipeline.transcribe, audio, batch_size=batch_size, **options
                )
            except Exception as exc:
                log(f"批量模式不可用（{type(exc).__name__}: {exc}），回退逐段模式")
                segment_iter, info = self._invoke(self._model.transcribe, audio, **options)
        else:
            segment_iter, info = self._invoke(self._model.transcribe, audio, **options)

        total = float(getattr(info, "duration_after_vad", 0.0) or getattr(info, "duration", 0.0) or 0.0)
        segments: list[Segment] = []
        for raw in segment_iter:  # 生成器：边识别边产出，可实时回调
            text = (raw.text or "").strip()
            segment = Segment(
                id=int(getattr(raw, "id", len(segments)) or len(segments)),
                start=float(raw.start),
                end=float(raw.end),
                text=text,
                avg_logprob=_opt_float(getattr(raw, "avg_logprob", None)),
                no_speech_prob=_opt_float(getattr(raw, "no_speech_prob", None)),
                temperature=_opt_float(getattr(raw, "temperature", None)),
                compression_ratio=_opt_float(getattr(raw, "compression_ratio", None)),
                words=_words_to_dicts(getattr(raw, "words", None)),
            )
            segments.append(segment)
            if progress_callback:
                progress_callback(segment.end, total, text)
        return segments, info


def _is_out_of_memory(exc: BaseException) -> bool:
    """显存/内存不足类错误（CUDA OOM、mkl_malloc 失败等）。"""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in ("out of memory", "outofmemory", "mkl_malloc", "insufficient memory", "memoryerror")
    )


def _is_cuda_context_broken(exc: BaseException) -> bool:
    """CUDA 上下文已失效（通常紧跟一次 OOM 之后，同进程内无法再用 GPU）。"""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in ("invalid device ordinal", "cudainvaliddevice", "invalid device function")
    )


def _memory_error_help(exc: BaseException) -> RuntimeError:
    """把显存/内存相关错误包装成可直接照做的建议。"""
    return RuntimeError(
        f"显存或系统内存不足（{type(exc).__name__}: {exc}）。\n"
        "注意：**系统内存不足也会报同样的错**（模型要先读进内存再上卡），\n"
        "先看任务管理器里「可用内存」还剩多少，再按下列方式重跑：\n"
        "  * --batch-size 0 --compute-type int8_float16   （最省资源，推荐先试）\n"
        "  * --device cpu --compute-type int8             （走内存，慢但稳）\n"
        "  * --model medium                               （换更小的模型）\n"
        "  * 关闭其他吃内存/显存的程序（浏览器多标签、网盘、QQ、游戏等）\n"
        "详见 docs/troubleshooting.md「显存或内存不足」一节。"
    )


def _opt_float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _words_to_dicts(words: Any) -> Optional[list[dict[str, Any]]]:
    if not words:
        return None
    result: list[dict[str, Any]] = []
    for word in words:
        result.append(
            {
                "start": round(float(getattr(word, "start", 0.0)), 3),
                "end": round(float(getattr(word, "end", 0.0)), 3),
                "word": getattr(word, "word", ""),
                "probability": _opt_float(getattr(word, "probability", None)),
            }
        )
    return result


# --------------------------------------------------------------------- 模型缓存

#: 只缓存"最近一次"的模型：批量处理多个文件、网页端连续转写时省掉重复加载，
#: 切换模型时旧模型会被释放，不会把显存堆满。
_cached_transcriber: Optional[Transcriber] = None
_cached_key: Optional[tuple] = None


def _cache_key(**kwargs: Any) -> tuple:
    return tuple(sorted((k, str(v)) for k, v in kwargs.items()))


def get_transcriber(**kwargs: Any) -> Transcriber:
    """按配置复用已加载的模型，避免批量任务反复加载（每次加载要好几秒到几十秒）。"""
    global _cached_transcriber, _cached_key

    log_callback = kwargs.pop("log_callback", None)
    key = _cache_key(**kwargs)
    if _cached_transcriber is not None and key == _cached_key:
        if log_callback is not None:  # 复用模型时换上本次的回调
            _cached_transcriber.log = log_callback
        return _cached_transcriber

    transcriber = Transcriber(log_callback=log_callback, **kwargs)
    _cached_transcriber = transcriber
    _cached_key = key
    return transcriber


def release_cached_transcriber() -> None:
    """释放缓存的模型（显存紧张或任务结束时调用）。"""
    global _cached_transcriber, _cached_key
    _cached_transcriber = None
    _cached_key = None

