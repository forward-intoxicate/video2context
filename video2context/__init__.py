"""video2context · 视频 → 音频 → 文字 的本地转换工程。

流水线：
    视频/音频文件 --ffmpeg--> 16kHz 单声道 wav --faster-whisper--> 分段文字 + JSON
"""

__version__ = "0.1.0"

# 作为库使用：``from video2context import process, TranscribeOptions``
from .glossary import Glossary, SymbolRepair, symbol_hit_score  # noqa: E402
from .pipeline import (  # noqa: E402
    PipelineResult,
    TranscribeOptions,
    process,
    scan_and_build_glossary,
)

__all__ = [
    "__version__",
    "process",
    "scan_and_build_glossary",
    "TranscribeOptions",
    "PipelineResult",
    "Glossary",
    "SymbolRepair",
    "symbol_hit_score",
]
