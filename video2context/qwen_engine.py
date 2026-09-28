"""Qwen3-ASR 引擎：主进程侧的「子进程桥接」+ 时间戳切句。

为什么用子进程
--------------
Qwen3-ASR 需要 ``torch`` + ``transformers``（约 3GB），而主工程刻意保持轻量
（faster-whisper 走 CTranslate2，不需要 torch）。装进同一个环境会互相牵制版本，
所以主进程不 import 任何 torch 相关库，而是把活交给独立环境 ``.venv-qwen`` 里的
:mod:`video2context._qwen_worker`，用 JSON 交换结果。

本模块负责三件事：

1. **找到独立环境与模型**（命令行参数 → 环境变量 → 工程内默认路径）；
2. **拉起工人进程**并把工人的日志实时转发到主进程日志；
3. **把结果变成分段**：Qwen3-ASR 本身只输出一整段文字，
   所以 SRT/VTT 需要的句子边界由这里生成 —— 有强制对齐模型时按时间戳切，
   没有时退化成「按标点切句 + 按说话区间比例分配时间」。

实测结论（见 docs/models.md）：中文 1.7B 在样例上 u(x)/v(x) 全对、无同音字错误，
英文 WER 0.00%（faster-whisper large-v3 为 2.97%）。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable, Optional

from .transcriber import Segment, TranscriptionResult

__all__ = [
    "QwenAsrEngine",
    "QwenAsrError",
    "DEFAULT_QWEN_REPO",
    "DEFAULT_ALIGNER_REPO",
    "find_qwen_python",
    "find_local_model",
    "normalize_language_name",
    "group_time_stamps",
    "split_text_into_sentences",
]

LogCallback = Callable[[str], None]
ProgressCallback = Callable[[float, float, str], None]

#: 没在本地找到模型时回退到 HuggingFace / ModelScope 仓库名
DEFAULT_QWEN_REPO = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_ALIGNER_REPO = "Qwen/Qwen3-ForcedAligner-0.6B"

#: 默认的本地模型**子目录名**（根目录默认是工程内的 ``models/``，
#: 可用 ``V2C_MODELS_DIR`` 改到别处 —— 服务器上常把代码和权重分开放）
LOCAL_QWEN_DIR = "Qwen3-ASR-1.7B"
LOCAL_QWEN_SMALL_DIR = "Qwen3-ASR-0.6B"
LOCAL_ALIGNER_DIR = "Qwen3-ForcedAligner-0.6B"

#: 句末标点：出现在某个 token 之后就断句
SENTENCE_END_CHARS = "。！？!?…；;"
#: 次级断点：句子已经够长时遇到它也可以断
CLAUSE_CHARS = "，,、：:"

#: 关掉对齐模型的写法（命令行 --qwen-aligner off）
DISABLED_VALUES = {"", "off", "none", "no", "false", "0", "无"}

#: ISO 码 / 常见写法 → Qwen3-ASR 要求的规范语言名
LANGUAGE_ALIASES: dict[str, str] = {
    "zh": "Chinese", "zho": "Chinese", "chi": "Chinese", "cmn": "Chinese",
    "zh-cn": "Chinese", "zh-tw": "Chinese", "chinese": "Chinese", "中文": "Chinese", "汉语": "Chinese",
    "en": "English", "eng": "English", "english": "English", "英文": "English",
    "yue": "Cantonese", "cantonese": "Cantonese", "粤语": "Cantonese",
    "ja": "Japanese", "jp": "Japanese", "jpn": "Japanese", "japanese": "Japanese", "日文": "Japanese",
    "ko": "Korean", "kor": "Korean", "korean": "Korean", "韩文": "Korean",
    "ar": "Arabic", "ara": "Arabic", "arabic": "Arabic",
    "de": "German", "deu": "German", "ger": "German", "german": "German",
    "fr": "French", "fra": "French", "fre": "French", "french": "French",
    "es": "Spanish", "spa": "Spanish", "spanish": "Spanish",
    "pt": "Portuguese", "por": "Portuguese", "portuguese": "Portuguese",
    "id": "Indonesian", "ind": "Indonesian", "indonesian": "Indonesian",
    "it": "Italian", "ita": "Italian", "italian": "Italian",
    "ru": "Russian", "rus": "Russian", "russian": "Russian",
    "th": "Thai", "tha": "Thai", "thai": "Thai",
    "vi": "Vietnamese", "vie": "Vietnamese", "vietnamese": "Vietnamese",
    "tr": "Turkish", "tur": "Turkish", "turkish": "Turkish",
    "hi": "Hindi", "hin": "Hindi", "hindi": "Hindi",
    "ms": "Malay", "msa": "Malay", "malay": "Malay",
    "nl": "Dutch", "nld": "Dutch", "dut": "Dutch", "dutch": "Dutch",
    "sv": "Swedish", "swe": "Swedish", "swedish": "Swedish",
    "da": "Danish", "dan": "Danish", "danish": "Danish",
    "fi": "Finnish", "fin": "Finnish", "finnish": "Finnish",
    "pl": "Polish", "pol": "Polish", "polish": "Polish",
    "cs": "Czech", "ces": "Czech", "cze": "Czech", "czech": "Czech",
    "fil": "Filipino", "tl": "Filipino", "filipino": "Filipino",
    "fa": "Persian", "fas": "Persian", "per": "Persian", "persian": "Persian",
    "el": "Greek", "ell": "Greek", "gre": "Greek", "greek": "Greek",
    "ro": "Romanian", "ron": "Romanian", "rum": "Romanian", "romanian": "Romanian",
    "hu": "Hungarian", "hun": "Hungarian", "hungarian": "Hungarian",
    "mk": "Macedonian", "mkd": "Macedonian", "macedonian": "Macedonian",
}

class QwenAsrError(RuntimeError):
    """Qwen3-ASR 相关的可读错误（环境缺失、模型缺失、识别失败）。"""


# --------------------------------------------------------------------- 环境


def _project_root() -> Path:
    """工程根目录（``video2context/`` 的上一层）。"""
    return Path(__file__).resolve().parent.parent


def find_qwen_python(explicit: Optional[str] = None) -> Optional[Path]:
    """找到装了 ``qwen_asr`` 的解释器。

    * **明确指定**了路径（命令行 ``--qwen-python`` 或环境变量 ``V2C_QWEN_PYTHON``）
      就以它为准：不存在时返回 ``None``，不再偷偷回退到别处 —— 否则用户会以为
      自己指定的解释器生效了，实际跑的是另一个，很难排查。
    * 没指定才按工程约定自动探测：``.venv-qwen``（Windows 的 ``Scripts`` 与
      POSIX 的 ``bin`` 都试）。
    """
    requested = explicit or os.environ.get("V2C_QWEN_PYTHON")
    if requested:
        candidate = Path(requested).expanduser()
        return candidate if candidate.is_file() else None

    root = _project_root()
    for relative in ("Scripts/python.exe", "bin/python", "bin/python3"):
        candidate = root / ".venv-qwen" / relative
        if candidate.is_file():
            return candidate
    return None


def find_local_model(*candidates: Path) -> Optional[str]:
    """返回第一个存在的本地模型目录（含 config.json），否则 None。"""
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return str(candidate)
    return None


def _models_root() -> Path:
    """模型根目录（默认工程内 ``models/``，可用 V2C_MODELS_DIR 改）。"""
    from .config import models_root

    return models_root()


def resolve_qwen_model(value: Optional[str], *, small: bool = False) -> str:
    """把用户给的模型参数解析成可直接加载的路径或仓库名。"""
    if value:
        return value
    root = _models_root()
    order = [root / LOCAL_QWEN_DIR, root / LOCAL_QWEN_SMALL_DIR] if not small else [
        root / LOCAL_QWEN_SMALL_DIR,
        root / LOCAL_QWEN_DIR,
    ]
    found = find_local_model(*order)
    return found or (DEFAULT_QWEN_REPO if not small else "Qwen/Qwen3-ASR-0.6B")


def resolve_aligner(value: Optional[str]) -> Optional[str]:
    """解析强制对齐模型；显式关掉（off/none/无）时返回 None。"""
    if value is not None and value.strip().lower() in DISABLED_VALUES:
        return None
    if value:
        return value
    found = find_local_model(_models_root() / LOCAL_ALIGNER_DIR)
    return found or DEFAULT_ALIGNER_REPO


def normalize_language_name(value: Optional[str]) -> Optional[str]:
    """把 ``zh`` / ``en`` / ``Chinese`` 统一成 Qwen3-ASR 要求的规范名。"""
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned or cleaned.lower() in {"auto", "自动", "自动检测", "none"}:
        return None
    return LANGUAGE_ALIASES.get(cleaned.lower(), cleaned)


# ----------------------------------------------------------------- 时间戳切句


def _is_kept_char(char: str) -> bool:
    """复刻 Qwen3-ForcedAligner 的取舍：只保留字母、数字和撇号。

    对齐模型返回的 token 是**去掉标点**的（``is_kept_char`` 决定去留，注意撇号
    是保留的，所以 ``we're`` 是一个完整 token）。靠这个函数才能把 token 映射回
    原文里的位置，从而知道标点 —— 也就是句子边界 —— 落在哪儿。
    """
    if char == "'":
        return True
    category = unicodedata.category(char)
    return category.startswith("L") or category.startswith("N")


def _map_tokens_to_text(text: str, stamps: list[dict[str, Any]]) -> Optional[list[tuple[int, int]]]:
    """把每个时间戳 token 映射回原文的下标区间 ``(start, end)``。

    原文先按"对齐模型会保留的字符"压成一条参考串，然后逐个 token 在参考串里
    顺序查找（允许前后几字符的容差，方便自我纠偏）。这样即使某个 token 与
    参考串略有出入，也不会像"按长度硬推"那样一路错到底。

    匹配上的字符数不足九成说明这套映射不成立，返回 None 让调用方走退化方案。
    """
    kept_index: list[int] = []
    kept_chars: list[str] = []
    for index, char in enumerate(text):
        if _is_kept_char(char):
            kept_index.append(index)
            kept_chars.append(char)
    reference = "".join(kept_chars)
    if not reference:
        return None

    spans: list[tuple[int, int]] = []
    cursor = 0
    matched = 0
    for stamp in stamps:
        token = str(stamp.get("text") or "")
        if not token:
            continue
        window = len(token) + 8
        position = reference.find(token, cursor, min(cursor + window, len(reference)))
        if position < 0:
            position = cursor  # 找不到就按长度硬推，后面的 token 还有机会自我纠偏
        end = min(position + len(token), len(kept_index))
        if end <= position:
            continue
        spans.append((kept_index[position], kept_index[end - 1]))
        matched += len(token)
        cursor = end

    if not spans or matched < len(reference) * 0.9:
        return None
    return spans


def _is_sentence_end(text: str, index: int) -> bool:
    """``index`` 是某个 token 最后一个字符在原文里的下标，判断它之后是否断句。"""
    if index + 1 >= len(text):
        return True
    following = text[index + 1]
    if following in SENTENCE_END_CHARS or following == "\n":
        return True
    if following == ".":
        # 英文句点：只在后面是空白/结尾时算句末（避免 3.14、u.s. 之类被切开）
        after = text[index + 2] if index + 2 < len(text) else ""
        return after == "" or after.isspace() or after == "\n"
    return False


def group_time_stamps(
    text: str,
    stamps: list[dict[str, Any]],
    *,
    max_chars: int = 80,
    max_duration: float = 12.0,
    min_duration: float = 1.0,
    max_gap: float = 1.2,
) -> list[Segment]:
    """按时间戳把整段文字切成一句一段的 :class:`Segment`。

    下列**任一**条件成立就断开（合起来看就是一份"字幕断句规则表"）：

    * **句末标点** —— 最自然，字幕尽量按句切；
    * **明显停顿**（间隔 ≥ ``max_gap`` 秒）—— 说话人换气/换话题；
    * **次级标点**（逗号、顿号）—— 只在片段已经超过 ``max_chars`` 时用；
    * **硬上限** —— 超过 ``max_duration`` 秒或 ``2*max_chars`` 字，强行断开。

    文本直接取原文切片，所以标点符号会原样保留在字幕里。
    """
    if not stamps:
        return []

    spans = _map_tokens_to_text(text, stamps)
    if spans is None:
        return []

    segments: list[Segment] = []
    buffer: list[int] = []  # 当前片段在 stamps 里的下标

    def flush() -> None:
        if not buffer:
            return
        first, last = buffer[0], buffer[-1]
        start_index = spans[first][0]
        end_index = spans[last][1] + 1
        # 时间戳 token 是去掉标点的，所以要把紧跟其后的标点（。，！？等）补回切片，
        # 否则字幕会丢掉所有句读。
        while end_index < len(text) and not _is_kept_char(text[end_index]) and text[end_index] != "\n":
            end_index += 1
        chunk = text[start_index:end_index].strip()
        if chunk:
            segments.append(
                Segment(
                    id=len(segments),
                    start=round(float(stamps[first]["start"]), 3),
                    end=round(float(stamps[last]["end"]), 3),
                    text=chunk,
                )
            )
        buffer.clear()

    for index, stamp in enumerate(stamps):
        buffer.append(index)
        text_end = spans[index][1]
        next_char = text[text_end + 1] if text_end + 1 < len(text) else ""
        span = float(stamp["end"]) - float(stamps[buffer[0]]["start"])
        visible = text_end - spans[buffer[0]][0] + 1
        gap = 0.0
        if index + 1 < len(stamps):
            gap = float(stamps[index + 1]["start"]) - float(stamp["end"])

        # 四条规则分开命名只是为了可读；任意一条成立都做同样的 flush，所以顺序无关。
        at_sentence_end = _is_sentence_end(text, text_end) and span >= min_duration
        at_pause = gap >= max_gap and (span >= min_duration or visible >= 10)
        at_clause = visible >= max_chars and next_char in CLAUSE_CHARS
        over_limit = span >= max_duration or visible >= max_chars * 2

        if at_sentence_end or at_pause or at_clause or over_limit:
            flush()

    flush()
    return segments


# ---------------------------------------------------------- 无时间戳时的退化


_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?…；;])")
#: 单条字幕的软上限：超过就先在逗号处断，再在空格处断，最后硬切
_SOFT_CHARS = 80
#: 硬上限：到这儿不管有没有标点都要断，免得整段变成一条巨型字幕
_HARD_CHARS = 120


def split_text_into_sentences(text: str) -> list[str]:
    """按句末标点切句；过长的句子再按逗号 → 空格 → 字符依次软切。

    Qwen3-ASR 输出的是**一整段**文字，SRT/VTT 需要句子。中文一般标点齐全，
    英文长句或没有标点的输出就得靠长度兜底。
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    sentences: list[str] = []
    for piece in _SENTENCE_SPLIT.split(cleaned):
        piece = piece.strip()
        if piece:
            sentences.extend(_wrap_long_piece(piece))
    return sentences


def _wrap_long_piece(piece: str) -> list[str]:
    """把一条过长的句子拆成若干不超过硬上限的片段。"""
    if len(piece) <= _HARD_CHARS:
        return [piece]

    # 第一轮：在逗号/顿号/分号后面断
    chunks: list[str] = []
    buffer = ""
    for unit in re.split(r"(?<=[,，、;；])", piece):
        if buffer and len(buffer) + len(unit) > _HARD_CHARS:
            chunks.append(buffer)
            buffer = unit
        else:
            buffer += unit
    if buffer:
        chunks.append(buffer)

    # 第二轮：仍然过长的（英文无标点）按空格断；中文没空格则硬切
    result: list[str] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if len(chunk) <= _HARD_CHARS:
            result.append(chunk)
            continue
        buffer = ""
        for word in chunk.split(" "):
            if buffer and len(buffer) + 1 + len(word) > _SOFT_CHARS:
                result.append(buffer)
                buffer = word
            else:
                buffer = f"{buffer} {word}".strip()
        if buffer:
            result.append(buffer)

    # 第三轮：完全不换行的长串，按字符硬切
    final: list[str] = []
    for chunk in result:
        while len(chunk) > _HARD_CHARS:
            final.append(chunk[:_HARD_CHARS])
            chunk = chunk[_HARD_CHARS:]
        if chunk:
            final.append(chunk)
    return final


def _speech_regions(audio_path: Path, log: LogCallback) -> list[tuple[float, float]]:
    """用 faster-whisper 自带的 Silero VAD 找说话区间（不可用则返回空表）。

    没有强制对齐模型时，靠这些区间把句子按字数比例放到"真的有人在说"的地方，
    总比平均铺满整条时间轴靠谱。

    这个能力**依赖 faster-whisper**（VAD 与音频解码都在它里面）。只部署 Qwen 的机器
    没有它，于是退化成"按总时长平均分配" —— 能用，但时间更粗。所以这里把原因说清楚，
    并指向真正的解法（装对齐模型），而不是丢一句 `ModuleNotFoundError` 让人以为坏了。
    """
    try:
        from faster_whisper.audio import decode_audio
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError:
        log(
            "      未安装 faster-whisper，拿不到说话区间（只部署 Qwen 时属正常）"
            "→ 句子时间按总时长平均分配；"
            "想要精确字幕时间请装对齐模型 Qwen3-ForcedAligner-0.6B，见 docs/deploy.md"
        )
        return []

    try:
        import numpy as np

        audio = decode_audio(str(audio_path), sampling_rate=16000)
        stamps = get_speech_timestamps(
            audio,
            VadOptions(min_silence_duration_ms=500, speech_pad_ms=200),
            sampling_rate=16000,
        )
        regions = [
            (float(item["start"]) / 16000.0, float(item["end"]) / 16000.0) for item in stamps
        ]
        if regions:
            log(f"      VAD 找到 {len(regions)} 个说话区间（用于无对齐模型时分配时间）")
        del audio, np  # 尽早释放
        return regions
    except Exception as exc:
        log(f"      VAD 不可用（{type(exc).__name__}: {exc}），句子时间将按总时长平均分配")
        return []


def distribute_sentences(
    sentences: list[str],
    regions: list[tuple[float, float]],
    duration: float,
) -> list[Segment]:
    """把句子按字数比例分配到说话区间（没有区间就铺满整条时间轴）。

    做法是在"说话时间轴"上分配：把所有说话区间首尾相接拼成一条长度为
    ``总说话时长`` 的时间轴，按字数比例在这条轴上切段，再把每段的起止点
    映射回绝对时间。这样跨静音时，后一句会自然跳到下一个说话区间的起点，
    而不是把静音也算进时长里。

    这是**没有强制对齐模型时的近似**：句子边界来自标点，时间只是估算，
    精确到帧的字幕请加 ``--qwen-aligner``。
    """
    if not sentences:
        return []
    if not regions:
        if duration <= 0:
            return [Segment(id=i, start=0.0, end=0.0, text=s) for i, s in enumerate(sentences)]
        regions = [(0.0, duration)]

    def to_absolute(offset: float, *, at_end: bool = False) -> float:
        """把"说话时间轴"上的位置映射成绝对秒数。

        正好落在两个区间的接缝上时，``at_end=False``（句子起点）映射到**下一个
        区间的开头**，``at_end=True``（句子终点）映射到**当前区间的结尾** ——
        这样跨静音时前一句收在静音前、后一句从静音后开始，中间留白是对的。
        """
        remaining = offset
        for start, end in regions:
            length = end - start
            if remaining < length:
                return start + remaining
            if at_end and remaining == length:
                return end
            remaining -= length
        return regions[-1][1]

    total_speech = sum(end - start for start, end in regions)
    total_chars = sum(len(s) for s in sentences) or 1

    segments: list[Segment] = []
    cursor = 0.0
    for sentence in sentences:
        start = to_absolute(cursor)
        cursor += total_speech * len(sentence) / total_chars
        end = to_absolute(cursor, at_end=True)
        segments.append(
            Segment(
                id=len(segments),
                start=round(start, 3),
                end=round(max(end, start), 3),
                text=sentence,
            )
        )
    return segments


# --------------------------------------------------------------------- 引擎


class QwenAsrEngine:
    """与 :class:`video2context.transcriber.Transcriber` 同接口的 Qwen3-ASR 引擎。

    真正的识别发生在 ``.venv-qwen`` 的子进程里；这里只做参数整理、进程管理和结果转换。
    """

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        aligner: Optional[str] = None,
        python: Optional[str] = None,
        device: str = "auto",
        context: Optional[str] = None,
        low_mem: str = "auto",
        max_new_tokens: int = 4096,
        max_batch_size: int = 8,
        log_callback: Optional[LogCallback] = None,
        **_ignored: Any,  # 兼容 faster-whisper 的 compute_type / beam_size 等参数
    ) -> None:
        self.model = resolve_qwen_model(model)
        self.aligner = resolve_aligner(aligner)
        self.requested_python = python or os.environ.get("V2C_QWEN_PYTHON")
        self.python = find_qwen_python(python)
        self.device = device or "auto"
        self.context = context
        self.low_mem = low_mem
        self.max_new_tokens = int(max_new_tokens)
        self.max_batch_size = int(max_batch_size)
        self.worker = Path(__file__).with_name("_qwen_worker.py")
        self.log = log_callback or (lambda _msg: None)
        self.model_load_seconds: float = 0.0
        self.resolved_device = self.device

    # ---------------------------------------------------------------- 校验

    def preflight(self) -> None:
        """尽早给出可执行的报错（环境没装、模型没下、任务不支持）。"""
        if self.python is None:
            requested = self.requested_python or os.environ.get("V2C_QWEN_PYTHON")
            if requested:
                raise QwenAsrError(
                    f"指定的 Qwen3-ASR 解释器不存在：{requested}\n"
                    "请检查 --qwen-python / 环境变量 V2C_QWEN_PYTHON 指向的路径，"
                    "或去掉它改用默认的 .venv-qwen。"
                )
            raise QwenAsrError(
                "找不到 Qwen3-ASR 的独立环境（.venv-qwen）。\n"
                "Qwen3-ASR 需要 torch，主环境刻意不装，所以它单开一个环境：\n"
                "    powershell -ExecutionPolicy Bypass -File scripts/setup_qwen.ps1\n"
                "（或手动：python -m venv .venv-qwen 后 pip install -r requirements-qwen.txt）\n"
                "也可以把已有解释器的路径写进环境变量 V2C_QWEN_PYTHON。"
            )
        if not self.worker.is_file():
            raise QwenAsrError(f"缺少工人脚本：{self.worker}")

    # ---------------------------------------------------------------- 识别

    def transcribe(
        self,
        audio_path: str | os.PathLike[str],
        *,
        language: Optional[str] = None,
        task: str = "transcribe",
        vad_filter: bool = True,
        beam_size: int = 5,
        batch_size: int = 0,
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: bool = False,
        temperature: Optional[float | list[float]] = None,
        word_timestamps: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        log_callback: Optional[LogCallback] = None,
        aligner: Optional[str] = None,
        context: Optional[str] = None,
    ) -> TranscriptionResult:
        """识别音频。除 ``language`` 外的 faster-whisper 参数在这里都是空操作。"""
        del vad_filter, beam_size, batch_size, condition_on_previous_text, temperature, word_timestamps
        self.preflight()
        log = log_callback or self.log

        if task and task != "transcribe":
            raise QwenAsrError(
                f"Qwen3-ASR 不支持 task={task}（它只做原语言转写，不做翻译）。\n"
                "需要翻译成英文请改用：--engine faster-whisper --task translate"
            )

        audio = Path(audio_path)
        wanted_aligner = resolve_aligner(aligner) if aligner is not None else self.aligner
        effective_context = context if context is not None else self.context

        with tempfile.TemporaryDirectory(prefix="v2c-qwen-") as temp_dir:
            out_json = Path(temp_dir) / "result.json"
            command = self._build_command(
                audio,
                out_json,
                language=normalize_language_name(language),
                context=effective_context,
                aligner=wanted_aligner,
            )
            log(f"      调用 Qwen3-ASR 子进程：{self.python}")
            _log_aligner_status(wanted_aligner, log)
            payload = self._run(command, log)

        return self._to_result(
            payload,
            audio_path=audio,
            aligner=wanted_aligner,
            context=effective_context,
            progress_callback=progress_callback,
            log=log,
        )

    # ------------------------------------------------------------ 内部实现

    def _build_command(
        self,
        audio: Path,
        out_json: Path,
        *,
        language: Optional[str],
        context: Optional[str],
        aligner: Optional[str],
    ) -> list[str]:
        assert self.python is not None
        command = [
            str(self.python),
            str(self.worker),
            "--wav",
            str(audio),
            "--model",
            str(self.model),
            "--out",
            str(out_json),
            "--device",
            self.device,
            "--low-mem",
            self.low_mem,
            "--max-new-tokens",
            str(self.max_new_tokens),
            "--max-batch-size",
            str(self.max_batch_size),
        ]
        if aligner:
            command += ["--aligner", str(aligner)]
        if language:
            command += ["--language", language]
        if context:
            command += ["--context", context]
        return command

    def _run(self, command: list[str], log: LogCallback) -> dict[str, Any]:
        """跑工人进程，实时转发它的 stderr 日志，读回结果 JSON。"""
        env = dict(os.environ)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("HF_HUB_DISABLE_XET", "1")

        started = time.perf_counter()
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(_project_root()),
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            raise QwenAsrError(f"无法启动 Qwen3-ASR 子进程（{self.python}）：{exc}") from exc

        assert process.stderr is not None
        tail: list[str] = []
        try:
            for line in process.stderr:
                stripped = line.rstrip()
                if not stripped:
                    continue
                tail.append(stripped)
                del tail[:-40]
                log(f"      {stripped}")
            process.wait()
        except KeyboardInterrupt:  # 用户中断：别留下孤儿进程占着显存
            process.kill()
            process.wait()
            raise

        stdout = (process.stdout.read() if process.stdout else "") or ""
        if process.returncode != 0:
            detail = "\n".join(tail[-20:]) or stdout[-2000:] or "(无输出)"
            raise QwenAsrError(self._explain_failure(process.returncode, detail))

        out_json = Path(command[command.index("--out") + 1])
        if not out_json.is_file():
            raise QwenAsrError(f"Qwen3-ASR 子进程没有写出结果文件\n{stdout[-1000:]}")
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        self.model_load_seconds = float(payload.get("load_seconds") or 0.0)
        log(f"      子进程总耗时 {time.perf_counter() - started:.1f}s")
        return payload

    @staticmethod
    def _explain_failure(returncode: int, detail: str) -> str:
        """把子进程的原始报错翻译成能照做的建议。"""
        lowered = detail.lower()
        hints: list[str] = []
        if "页面文件太小" in detail or "paging file" in lowered or "out of memory" in lowered:
            hints.append(
                "显存或系统内存不足。Qwen3-ASR-1.7B 加载需要约 13GB 提交内存，可尝试：\n"
                "  * 关闭浏览器/网盘等占内存的程序后重跑\n"
                "  * --qwen-low-mem on（分片直接进显存，CPU 侧几乎不占内存）\n"
                "  * 换小模型：--model models/Qwen3-ASR-0.6B"
            )
        if "no module named" in lowered:
            hints.append(
                "独立环境缺依赖，重新执行：\n"
                "  powershell -ExecutionPolicy Bypass -File scripts/setup_qwen.ps1"
            )
        if "0xc0000005" in lowered or "access violation" in lowered:
            hints.append("进程发生访问冲突，通常是内存不足的连带后果；重启机器后重试最有效。")
        message = f"Qwen3-ASR 子进程失败（退出码 {returncode}）：\n{detail}"
        if hints:
            message += "\n\n建议：\n  * " + "\n  * ".join(hints)
        return message

    def _to_result(
        self,
        payload: dict[str, Any],
        *,
        audio_path: Path,
        aligner: Optional[str],
        context: Optional[str],
        progress_callback: Optional[ProgressCallback],
        log: LogCallback,
    ) -> TranscriptionResult:
        text = str(payload.get("text") or "").strip()
        duration = float(payload.get("duration") or 0.0)
        stamps = payload.get("time_stamps") or []

        # 字幕时间有三档质量，这里记录**实际走了哪一档**并写进 JSON。
        # 注意不能用"有没有拿到时间戳"来判定：对齐时间戳存在但映射失败时会退回估算，
        # 那时标签必须是估算而不是 forced-aligner。
        segments: list[Segment] = []
        source = "none"
        if stamps:
            segments = group_time_stamps(text, stamps)
            if segments:
                source = "forced-aligner"
                log(f"      按时间戳切出 {len(segments)} 段（强制对齐，时间精确到词）")
            else:
                log("      ⚠ 对齐时间戳与文本对不上（token 映射失败），退回按标点切句估算时间")
        if not segments and text:
            sentences = split_text_into_sentences(text)
            regions = _speech_regions(audio_path, log)
            segments = distribute_sentences(sentences, regions, duration)
            if regions:
                source = "vad-proportional"
                log(
                    f"      无对齐模型：按标点切成 {len(segments)} 句，"
                    f"时间按 {len(regions)} 个说话区间比例分配（不精确）"
                )
            else:
                source = "even-spread"
                log(
                    f"      无对齐模型、也没有说话区间：按标点切成 {len(segments)} 句，"
                    "时间在整条音轨上平均分配（最粗的一档，字幕会铺到没人说话的地方）"
                )
        elif not text:
            log("      ⚠ 识别结果为空（音频可能是纯音乐或静音）")

        if progress_callback:
            progress_callback(duration, duration, segments[-1].text if segments else text)

        return TranscriptionResult(
            segments=segments,
            text=text,
            language=payload.get("language"),
            language_probability=None,
            duration=duration,
            duration_after_vad=sum(max(0.0, s.end - s.start) for s in segments),
            model=str(payload.get("model") or self.model),
            device=str(payload.get("device") or self.device),
            compute_type=str(payload.get("dtype") or "bfloat16"),
            # 这几个字段是 faster-whisper 的概念，Qwen 引擎里没有对应物，
            # 写进 JSON 时保持"空"，免得看结果的人以为它真的生效了。
            vad_filter=False,
            beam_size=0,
            batch_size=int(self.max_batch_size),
            elapsed_seconds=float(payload.get("elapsed_seconds") or 0.0),
            model_load_seconds=float(payload.get("load_seconds") or 0.0),
            initial_prompt=context,
            engine="qwen3-asr",
            extra={
                "aligner": aligner,
                "time_stamps": len(stamps),
                # 字幕时间实际来自哪一档，见 _to_result 里的注释：
                # forced-aligner（词级精确）/ vad-proportional（按说话区间估算）
                # / even-spread（整条音轨平均分，最粗）/ none（没切出分段）
                "timestamp_source": source,
                "low_mem": self.low_mem,
                "python": str(self.python) if self.python else None,
                "context_applied": bool(context),
                "context_effect": "none-observed",
            },
        )


def _log_aligner_status(aligner: Optional[str], log: LogCallback) -> None:
    """把对齐模型的状态说清楚。

    它是**字幕时间精度的分界线**，不是可有可无的附件：

    * 关掉了（``--qwen-aligner off``）→ 时间只能估算；
    * 本地没有 → 首次运行会从 HuggingFace 拉 1.8GB，国内很慢（建议改用下载器走魔搭）；
    * 就绪 → 时间精确到词。

    这三种情况用户都该在日志里一眼看到，而不是等字幕出来发现时间不对再回头查。

    注意"本地没有"这条**不区分**用户给的是仓库名还是写错的路径 ——
    ``Qwen/Qwen3-ForcedAligner-0.6B`` 和 ``models/xxx`` 结构上一模一样，猜不出来。
    所以措辞对两种情况都成立：按这个名字加载，本地没有就去网上拿。
    """
    if aligner is None:
        log("      提示：对齐模型已关闭，字幕时间只能按说话区间估算（去掉 --qwen-aligner off 可恢复）")
    elif Path(aligner).is_dir():
        log(f"      强制对齐模型：{aligner}（字幕时间精确到词）")
    else:
        log(
            f"      强制对齐模型：{aligner} —— 本地没有，首次运行会去 HuggingFace 拉约 1.8GB。\n"
            "        国内建议先预下载（走魔搭快得多，也可断点续传）：\n"
            "        python scripts/download_model.py --repo Qwen/Qwen3-ForcedAligner-0.6B "
            "--source modelscope --out models/Qwen3-ForcedAligner-0.6B"
        )


# --------------------------------------------------------------------- 缓存

_cached_engine: Optional[QwenAsrEngine] = None
_cached_key: Optional[tuple] = None


def get_qwen_engine(log_callback: Optional[LogCallback] = None, **kwargs: Any) -> QwenAsrEngine:
    """按配置复用引擎对象（复用只是省掉参数解析，模型仍由子进程自己加载）。"""
    global _cached_engine, _cached_key
    key = tuple(sorted((k, str(v)) for k, v in kwargs.items()))
    if _cached_engine is not None and key == _cached_key:
        if log_callback is not None:
            _cached_engine.log = log_callback
        return _cached_engine
    engine = QwenAsrEngine(log_callback=log_callback, **kwargs)
    _cached_engine = engine
    _cached_key = key
    return engine


def release_cached_engine() -> None:
    global _cached_engine, _cached_key
    _cached_engine = None
    _cached_key = None


if __name__ == "__main__":  # pragma: no cover - 手测用
    engine = QwenAsrEngine()
    print(f"python : {engine.python}")
    print(f"model  : {engine.model}")
    print(f"aligner: {engine.aligner}")
    print(f"worker : {engine.worker} (存在={engine.worker.is_file()})", file=sys.stderr)
