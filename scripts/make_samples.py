"""生成 `samples/` 里的示例视频（**完全离线**，不下载任何第三方素材）。

为什么要有这个脚本
------------------
示例视频是「打开就能跑」的第一步，但它不能随便从网上拿 —— 课程录像、播客都有版权。
所以这里的做法是：**用本机自带的语音合成念我们自己写的稿子**，再和纯色背景合成视频。
生成出来的素材没有任何第三方内容，脚本本身也留在仓库里，谁都能复现或重新生成。

语音合成按平台自动挑一个可用的（都不需要联网）：

* Windows —— 系统自带的 SAPI（`System.Speech`），中文用 Microsoft Huihui、英文用 Microsoft Zira；
* macOS —— `say` 命令（中文 Tingting / 英文 Samantha）；
* Linux —— `espeak-ng`（音质一般，但能用）。

用法
----
    python scripts/make_samples.py                  # 生成全部示例
    python scripts/make_samples.py --only demo_en_ml
    python scripts/make_samples.py --keep-wav       # 保留中间 wav，便于排查
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = PROJECT_ROOT / "samples"

#: 视频画面尺寸与帧率（静态画面，用很低的帧率就够了）
VIDEO_SIZE = "960x540"
VIDEO_FPS = 10
#: 深色背景 / 浅色文字，录屏和截图都清楚
BACKGROUND = "0x111827"
FOREGROUND = "0xE5E7EB"


@dataclass
class Sample:
    """一个示例视频的完整定义。"""

    id: str
    title: str
    #: 逐句稿子。分开写是为了在句间插入停顿 —— 这样 VAD 与字幕切分才有东西可切。
    lines: list[str]
    #: 各平台的语音名（空串表示用该平台默认语音）
    voices: dict[str, str] = field(default_factory=dict)
    #: 句间停顿时长（秒）
    gap: float = 0.5
    #: 画面上每行最多放多少个字符（中英文宽度差一倍，所以中日韩用小的）
    wrap: int = 46
    #: 字体候选（按顺序取第一个存在的）
    fonts: tuple[str, ...] = ()
    use_font: bool = True


#: 中文字体候选（跨平台）—— 中文稿和标题都靠它，缺了就会显示成"豆腐块"
CJK_FONTS: tuple[str, ...] = (
    "C:/Windows/Fonts/msyh.ttc",  # Windows 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",
    "/System/Library/Fonts/PingFang.ttc",  # macOS
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
)

#: 拉丁字体候选 —— 纯英文稿用它更好看（但不能画中文）
LATIN_FONTS: tuple[str, ...] = (
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)

SAMPLES: tuple[Sample, ...] = (
    Sample(
        id="demo_zh_math",
        title="中文 · 导数运算法则（数学课风格，含 u/v 这类符号）",
        voices={
            "win32": "Microsoft Huihui Desktop",
            "darwin": "Tingting",
            "linux": "cmn",
        },
        fonts=CJK_FONTS,
        wrap=24,
        lines=[
            "好，我们来看导数的运算法则。",
            "如果 u 和 v 这两个函数在 x 零处都可导，",
            "那么 u 加 v 的导数，就等于 u 先求导，再加上 v 求导。",
            "注意这个公式的前提，是两个函数都已经可导，才能这样拆开。",
            "对于多个函数相加的情况，比如说 u 一 加 u 二，一直加到 u n，",
            "同样可以逐个求导，再把结果相加。",
            "这是一个非常常用的公式，大家一定要记牢。",
        ],
    ),
    Sample(
        id="demo_en_ml",
        title="英文 · 梯度下降（术语密集，适合对比两个引擎）",
        voices={
            "win32": "Microsoft Zira Desktop",
            "darwin": "Samantha",
            "linux": "en-us",
        },
        fonts=LATIN_FONTS + CJK_FONTS,
        wrap=58,
        lines=[
            "Today we're going to talk about how machine learning models are actually trained.",
            "The core idea is gradient descent, which minimizes a loss function by repeatedly "
            "taking small steps downhill.",
            "At each step we compute the gradient with respect to every parameter,",
            "then update the weights using a small multiplier called the learning rate.",
            "If the learning rate is too large, training becomes unstable and the loss diverges.",
            "If it is too small, convergence takes forever.",
            "That is why modern systems often rely on adaptive optimizers such as Adam, "
            "RMSProp, or Adagrad.",
        ],
    ),
    Sample(
        id="demo_zh_daily",
        title="中文 · 日常口语（短句多、停顿多，适合看 VAD 与字幕分段）",
        voices={
            "win32": "Microsoft Huihui Desktop",
            "darwin": "Tingting",
            "linux": "cmn",
        },
        fonts=CJK_FONTS,
        wrap=24,
        gap=0.7,
        lines=[
            "嗯，这个周末我打算去哪儿呢。",
            "要不就去公园走走吧，天气看起来还不错。",
            "对了，顺便去超市买点东西，家里的牛奶快没了。",
            "还有鸡蛋，也要买一盒。",
            "嗯，让我想想还有什么。",
            "算了，先这样吧，到时候再说。",
        ],
    ),
)


# --------------------------------------------------------------------- 工具


def find_ffmpeg() -> str:
    """和主工程一样：环境变量 → PATH → imageio-ffmpeg 自带的静态版。"""
    import os

    for candidate in (os.environ.get("FFMPEG_BIN"), shutil.which("ffmpeg")):
        if candidate and Path(candidate).is_file():
            return candidate
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"找不到 ffmpeg（{exc}）。请设置环境变量 FFMPEG_BIN。") from exc


def run(args: list[str]) -> None:
    """跑一条命令，失败时把 stderr 原样抛出来，便于定位。"""
    proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise SystemExit(f"命令失败（{proc.returncode}）：\n  {' '.join(args)}\n{proc.stderr[-2000:]}")


def pick_font(sample: Sample) -> str | None:
    """挑一个系统里存在的字体（用于正文稿）；都没有就不画字（视频仍可用）。"""
    if not sample.use_font:
        return None
    for candidate in sample.fonts:
        if Path(candidate).is_file():
            return candidate
    return None


def pick_cjk_font() -> str | None:
    """挑一个能画中文的字体 —— 标题是中文，用拉丁字体会渲染成豆腐块。"""
    for candidate in CJK_FONTS:
        if Path(candidate).is_file():
            return candidate
    return None


def wrap_text(lines: list[str], width: int) -> list[str]:
    """把一个句子按显示宽度折行（返回行列表，不插入空行）。"""
    wrapped: list[str] = []
    for line in lines:
        for piece in _wrap_one(line, width):
            wrapped.append(piece)
    return wrapped


def _wrap_one(line: str, width: int) -> list[str]:
    """中文按字切、英文按词切（英文单词不能拆开）。"""
    result: list[str] = []
    current = ""
    for word in _words(line):
        if current and _display_width(current + word) > width:
            result.append(current.rstrip())
            current = word.lstrip()
        else:
            current += word
    if current.strip():
        result.append(current.rstrip())
    return result or [""]


def _words(line: str) -> list[str]:
    """中文按字切、英文按词切（英文单词不能拆开）。"""
    words: list[str] = []
    buffer = ""
    for char in line:
        if char.isascii() and (char.isalnum() or char in "'-"):
            buffer += char
            continue
        if buffer:
            words.append(buffer)
            buffer = ""
        words.append(char)
    if buffer:
        words.append(buffer)
    return words


def _display_width(text: str) -> int:
    """粗略显示宽度：CJK 算 2 格，其余算 1 格。"""
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)


def _is_mostly_cjk(text: str) -> bool:
    letters = [ch for ch in text if not ch.isspace()]
    if not letters:
        return False
    cjk = sum(1 for ch in letters if ord(ch) > 0x2E7F)
    return cjk / len(letters) > 0.2


def wav_duration(path: Path) -> float:
    """读 wav 时长（SAPI 输出的是 16kHz 单声道 PCM，标准库就够）。"""
    import wave

    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate() or 1)


def fit_text(lines: list[str], size: tuple[int, int], margin: int = 48) -> tuple[list[str], int]:
    """挑一个能让**最长的一句**在画面上放下的字号。

    画面一次只显示当前这一句（见 :func:`build_video`），所以只要最长那句放得下就行。
    """
    width, height = size
    usable_w = width - 2 * margin
    usable_h = height - 2 * margin
    cjk = _is_mostly_cjk("".join(lines))
    char_ratio = 1.0 if cjk else 0.55  # 一个字符占多少倍字号宽

    best: tuple[list[str], int] = (list(lines), 12)
    for fontsize in range(34, 11, -1):
        wrap = max(12, int(usable_w / (fontsize * char_ratio)))
        wrapped = wrap_text(lines, wrap)
        # 每句单独算行数，取最多的那句
        per_line = [len(_wrap_one(line, wrap)) for line in lines]
        tallest = max(per_line) if per_line else 1
        if tallest * fontsize * 1.45 <= usable_h:
            return wrapped, fontsize
        best = (wrapped, fontsize)
    return best


# --------------------------------------------------------------------- 语音合成


def tts(text: str, out_wav: Path, sample: Sample, tmp: Path) -> None:
    """把一段文字合成为 16kHz 单声道 wav。"""
    if sys.platform == "win32":
        _tts_windows(text, out_wav, sample.voices.get("win32", ""), tmp)
    elif sys.platform == "darwin":
        _tts_macos(text, out_wav, sample.voices.get("darwin", ""))
    else:
        _tts_espeak(text, out_wav, sample.voices.get("linux", ""))


def _tts_windows(text: str, out_wav: Path, voice: str, tmp: Path) -> None:
    """Windows 自带 SAPI。文本走临时文件传，避免命令行编码把中文搞坏。"""
    text_file = tmp / "line.txt"
    text_file.write_text(text, encoding="utf-8")
    script = tmp / "speak.ps1"
    script.write_text(
        "param([string]$TextPath, [string]$OutPath, [string]$VoiceName)\n"
        "Add-Type -AssemblyName System.Speech\n"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
        "if ($VoiceName) { $s.SelectVoice($VoiceName) }\n"
        "$s.Rate = 0\n"
        "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(\n"
        "    16000,\n"
        "    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,\n"
        "    [System.Speech.AudioFormat.AudioChannel]::Mono)\n"
        "$s.SetOutputToWaveFile($OutPath, $fmt)\n"
        "$s.Speak([System.IO.File]::ReadAllText($TextPath, [System.Text.UTF8Encoding]::new($false)))\n"
        "$s.Dispose()\n",
        encoding="utf-8-sig",  # Windows PowerShell 5.1 需要 BOM，否则中文脚本会乱码
    )
    run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-TextPath",
            str(text_file),
            "-OutPath",
            str(out_wav),
            "-VoiceName",
            voice,
        ]
    )


def _tts_macos(text: str, out_wav: Path, voice: str) -> None:
    aiff = out_wav.with_suffix(".aiff")
    args = ["say", "-o", str(aiff)]
    if voice:
        args += ["-v", voice]
    args.append(text)
    run(args)
    run([find_ffmpeg(), "-y", "-loglevel", "error", "-i", str(aiff), "-ac", "1", "-ar", "16000", str(out_wav)])
    aiff.unlink(missing_ok=True)


def _tts_espeak(text: str, out_wav: Path, voice: str) -> None:
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe:
        raise SystemExit(
            "Linux 上需要 espeak-ng 才能生成示例音频：sudo apt install espeak-ng\n"
            "（示例视频已经提交在 samples/ 里，不生成也能直接用。）"
        )
    args = [exe, "-w", str(out_wav), "-s", "150"]
    if voice:
        args += ["-v", voice]
    args.append(text)
    run(args)


# --------------------------------------------------------------------- 合成


def join_with_gaps(wavs: list[Path], gap: float, out_wav: Path) -> list[tuple[float, float]]:
    """把逐句 wav 用静音拼起来 —— 有停顿才像真实讲课，VAD 也才有东西可切。

    返回每句在成品音轨里的 ``(起点, 终点)``，用来做「说到哪句就显示哪句」。
    """
    ffmpeg = find_ffmpeg()
    args = [ffmpeg, "-y", "-loglevel", "error"]
    for wav in wavs:
        args += ["-i", str(wav)]
    if gap > 0 and len(wavs) > 1:
        for _ in range(len(wavs) - 1):
            args += ["-f", "lavfi", "-t", str(gap), "-i", "anullsrc=r=16000:cl=mono"]

    # 输入顺序：wav0, wav1, ..., silence, silence, ...
    order: list[int] = []
    for index in range(len(wavs)):
        order.append(index)
        if index < len(wavs) - 1 and gap > 0:
            order.append(len(wavs) + index)

    labels = "".join(f"[{i}:a]" for i in order)
    graph = f"{labels}concat=n={len(order)}:v=0:a=1[out]"
    args += ["-filter_complex", graph, "-map", "[out]", "-ac", "1", "-ar", "16000", str(out_wav)]
    run(args)

    timings: list[tuple[float, float]] = []
    cursor = 0.0
    for index, wav in enumerate(wavs):
        length = wav_duration(wav)
        timings.append((round(cursor, 3), round(cursor + length, 3)))
        cursor += length + (gap if index < len(wavs) - 1 else 0.0)
    return timings


def _escape_filter_path(path: str) -> str:
    """ffmpeg 滤镜里的路径：反斜杠换成正斜杠，盘符的冒号要转义。"""
    return path.replace("\\", "/").replace(":", "\\:")


def build_video(
    audio: Path,
    out_mp4: Path,
    sample: Sample,
    timings: list[tuple[float, float]],
    work_dir: Path,
) -> None:
    """纯色背景 + 当前句文字 + 音轨 → mp4。

    一句一个 ``drawtext``，用 ``enable='between(t,起点,终点)'`` 控制显示时机 ——
    所以画面上的字和正在说的话是对上的，边听边看就能发现识别错在哪。
    """
    ffmpeg = find_ffmpeg()
    font = pick_font(sample)
    cjk_font = pick_cjk_font()
    _, fontsize = fit_text(sample.lines, (960, 540))
    filters: list[str] = []

    # 给背景一个**明确时长**。用无限长的 color 源 + -shortest 会多出几秒尾巴
    # （实测多 2～3 秒静音黑屏，容器 Duration 比音频实际长度长），
    # 所以这里直接按音轨长度截断。
    total = timings[-1][1] if timings else 0.0

    # 顶部常驻标题。标题里一定有中文，所以**必须**用能画中文的字体，
    # 否则在英文样例上（正文用 Arial）标题会整行变成豆腐块。
    title_font = cjk_font or font
    if title_font:
        title_file = work_dir / "title.txt"
        if cjk_font:
            title_text = f"{sample.title}\n（机器合成语音 · 文字稿见 {sample.id}.transcript.txt）"
        else:  # 系统里一个中文字体都没有：退回纯 ASCII，免得画出一堆方框
            title_text = f"{sample.id} · machine-generated speech\nTranscript: {sample.id}.transcript.txt"
        title_file.write_text(title_text, encoding="utf-8")
        filters.append(
            "drawtext="
            f"fontfile='{_escape_filter_path(title_font)}'"
            f":textfile='{_escape_filter_path(str(title_file.resolve()))}'"
            ":expansion=none"
            f":fontcolor={FOREGROUND}"
            ":fontsize=18"
            ":line_spacing=6"
            ":x=28:y=24"
            ":alpha=0.75"
        )

    if font:
        # 逐句字幕：说到哪句显示哪句
        for index, (line, (start, end)) in enumerate(zip(sample.lines, timings)):
            text_file = work_dir / f"sentence_{index:02d}.txt"
            text_file.write_text("\n".join(_wrap_one(line, _wrap_width_for(fontsize, line))), encoding="utf-8")
            filters.append(
                "drawtext="
                f"fontfile='{_escape_filter_path(font)}'"
                f":textfile='{_escape_filter_path(str(text_file.resolve()))}'"
                ":expansion=none"
                f":fontcolor={FOREGROUND}"
                f":fontsize={fontsize}"
                ":line_spacing=14"
                ":x=(w-text_w)/2"
                ":y=(h-text_h)/2"
                f":enable='between(t,{start},{end})'"
            )

    color_source = f"color=c={BACKGROUND}:s={VIDEO_SIZE}:r={VIDEO_FPS}"
    if total > 0:
        color_source += f":d={total:.3f}"
    args = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        color_source,
        "-i",
        str(audio),
    ]
    if filters:
        args += ["-vf", ",".join(filters)]
    args += [
        "-shortest",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "32",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]
    run(args)


def _wrap_width_for(fontsize: int, line: str) -> int:
    """让 ``_wrap_one`` 用和 :func:`fit_text` 一致的字宽估算。"""
    ratio = 1.0 if _is_mostly_cjk(line) else 0.55
    return max(12, int((960 - 96) / (fontsize * ratio)))


def make_sample(sample: Sample, out_dir: Path, keep_wav: bool) -> Path:
    print(f"[{sample.id}] {sample.title}")
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = out_dir / f".{sample.id}.parts"
    wav_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="v2c-demo-") as tmp_name:
        tmp = Path(tmp_name)

        parts: list[Path] = []
        for index, line in enumerate(sample.lines):
            wav = wav_dir / f"{index:02d}.wav"
            tts(line, wav, sample, tmp)
            parts.append(wav)
            print(f"      合成 {index + 1}/{len(sample.lines)}：{line[:28]}…")

        joined = out_dir / f".{sample.id}.wav"
        timings = join_with_gaps(parts, sample.gap, joined)

        mp4 = out_dir / f"{sample.id}.mp4"
        build_video(joined, mp4, sample, timings, wav_dir)

    # 逐句的文字稿：方便把识别结果对着稿子看，英文还能直接喂给 wer_report.py
    transcript = out_dir / f"{sample.id}.transcript.txt"
    transcript.write_text("\n".join(sample.lines) + "\n", encoding="utf-8")

    if keep_wav:
        joined = out_dir / f".{sample.id}.wav"
        if joined.exists():
            shutil.move(str(joined), out_dir / f"{sample.id}.wav")
    else:
        (out_dir / f".{sample.id}.wav").unlink(missing_ok=True)
        shutil.rmtree(wav_dir, ignore_errors=True)

    size_mb = mp4.stat().st_size / 1_048_576
    print(f"      → {mp4.name}（{size_mb:.2f} MB，{timings[-1][1]:.1f}s）+ {transcript.name}")
    return mp4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 samples/ 里的示例视频（离线）")
    parser.add_argument("--only", action="append", default=None, help="只生成指定 id（可重复）")
    parser.add_argument("--out-dir", default=str(SAMPLES_DIR), help="输出目录（默认 samples/）")
    parser.add_argument("--keep-wav", action="store_true", help="保留中间 wav")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    wanted = [s for s in SAMPLES if not args.only or s.id in args.only]
    if not wanted:
        raise SystemExit(f"--only 没匹配到任何示例，可选：{', '.join(s.id for s in SAMPLES)}")

    print(f"输出目录：{out_dir}")
    print(f"语音合成后端：{_backend_name()}\n")
    for sample in wanted:
        make_sample(sample, out_dir, args.keep_wav)

    print("\n完成。跑一下看看：")
    print(f"  python -m video2context {out_dir / (wanted[0].id + '.mp4')}")
    return 0


def _backend_name() -> str:
    if sys.platform == "win32":
        return "Windows SAPI（System.Speech）"
    if sys.platform == "darwin":
        return "macOS say"
    return "espeak-ng"


if __name__ == "__main__":
    raise SystemExit(main())
