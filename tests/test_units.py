"""离线单元测试：不需要联网、不需要模型，直接跑：

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video2context.cli import _normalize_language, _parse_temperature  # noqa: E402
from video2context.config import interpreter_hint, project_python  # noqa: E402
from video2context.ffmpeg_tools import (  # noqa: E402
    extract_audio,
    find_ffmpeg,
    format_hms,
    format_hms_ms,
    probe_media,
)
from video2context.pipeline import sanitize_name  # noqa: E402
from video2context.transcriber import (  # noqa: E402
    _is_cuda_context_broken,
    _is_out_of_memory,
    _memory_error_help,
)
from video2context.writers import (  # noqa: E402
    parse_formats,
    segments_to_srt,
    segments_to_txt,
    segments_to_vtt,
)

SEGMENTS = [
    {"id": 0, "start": 0.0, "end": 2.5, "text": "你好，世界。"},
    {"id": 1, "start": 2.5, "end": 3661.25, "text": "Hello world."},
]


class TestFormatting(unittest.TestCase):
    def test_format_hms(self) -> None:
        self.assertEqual(format_hms(0), "00:00:00")
        self.assertEqual(format_hms(61.4), "00:01:01")
        self.assertEqual(format_hms(3661), "01:01:01")

    def test_format_hms_ms(self) -> None:
        self.assertEqual(format_hms_ms(0), "00:00:00.000")
        self.assertEqual(format_hms_ms(1.2345), "00:00:01.234")
        self.assertEqual(format_hms_ms(3661.5), "01:01:01.500")


class TestWriters(unittest.TestCase):
    def test_parse_formats(self) -> None:
        self.assertEqual(parse_formats("json,txt"), ("json", "txt"))
        self.assertEqual(parse_formats("JSON ; srt"), ("json", "srt"))
        self.assertEqual(parse_formats(""), ("json",))
        self.assertEqual(parse_formats(["json", "json"]), ("json",))
        self.assertEqual(parse_formats("json,vtt"), ("json", "vtt"))

    def test_parse_formats_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            parse_formats("json,docx")

    def test_srt_output(self) -> None:
        srt = segments_to_srt(SEGMENTS)
        self.assertIn("1\n00:00:00,000 --> 00:00:02,500\n你好，世界。", srt)
        self.assertIn("2\n00:00:02,500 --> 01:01:01,250\nHello world.", srt)

    def test_vtt_output(self) -> None:
        vtt = segments_to_vtt(SEGMENTS)
        self.assertTrue(vtt.startswith("WEBVTT\n\n"))
        self.assertIn("00:00:00.000 --> 00:00:02.500\n你好，世界。", vtt)
        self.assertEqual(segments_to_vtt([]), "WEBVTT\n")

    def test_txt_output(self) -> None:
        self.assertEqual(segments_to_txt(SEGMENTS), "你好，世界。\nHello world.\n")
        self.assertTrue(segments_to_txt(SEGMENTS, with_timestamps=True).startswith("[00:00:00.000] 你好"))


class TestCliHelpers(unittest.TestCase):
    def test_normalize_language(self) -> None:
        self.assertIsNone(_normalize_language("auto"))
        self.assertIsNone(_normalize_language("自动检测"))
        self.assertIsNone(_normalize_language("  "))
        self.assertIsNone(_normalize_language(None))
        self.assertEqual(_normalize_language("ZH"), "zh")

    def test_parse_temperature(self) -> None:
        self.assertIsNone(_parse_temperature(None))
        self.assertIsNone(_parse_temperature(""))
        self.assertEqual(_parse_temperature("0"), 0.0)
        self.assertEqual(_parse_temperature("0,0.2,0.4"), [0.0, 0.2, 0.4])
        self.assertEqual(_parse_temperature("0; 0.5"), [0.0, 0.5])
        with self.assertRaises(ValueError):
            _parse_temperature("热")


class TestDeviceSelection(unittest.TestCase):
    """CPU 不支持 float16 / int8_float16 —— 自动降级时不能沿用它。"""

    def _attempts(self, device: str, compute_type: str):
        from video2context.transcriber import Transcriber

        return Transcriber("x", device=device, compute_type=compute_type)._device_attempts()

    def test_cpu_downgrades_gpu_only_precision(self) -> None:
        self.assertEqual(self._attempts("cpu", "int8_float16"), [("cpu", "int8")])
        self.assertEqual(self._attempts("cpu", "float16"), [("cpu", "int8")])

    def test_cpu_keeps_supported_precision(self) -> None:
        self.assertEqual(self._attempts("cpu", "int8"), [("cpu", "int8")])
        self.assertEqual(self._attempts("cpu", "float32"), [("cpu", "float32")])

    def test_gpu_keeps_requested_precision(self) -> None:
        self.assertEqual(self._attempts("cuda", "int8_float16"), [("cuda", "int8_float16")])

    def test_auto_ends_with_cpu_safe_precision(self) -> None:
        attempts = self._attempts("auto", "int8_float16")
        self.assertEqual(attempts[-1], ("cpu", "int8"))


class TestOomDetection(unittest.TestCase):
    def test_detects_memory_errors(self) -> None:
        self.assertTrue(_is_out_of_memory(RuntimeError("CUDA failed with error out of memory")))
        self.assertTrue(_is_out_of_memory(RuntimeError("mkl_malloc: failed to allocate memory")))
        self.assertFalse(_is_out_of_memory(RuntimeError("model not found")))
        self.assertFalse(_is_out_of_memory(RuntimeError("cudaErrorInvalidDevice")))

    def test_detects_broken_cuda_context(self) -> None:
        self.assertTrue(_is_cuda_context_broken(RuntimeError("parallel_for failed: cudaErrorInvalidDevice: invalid device ordinal")))
        self.assertFalse(_is_cuda_context_broken(RuntimeError("CUDA failed with error out of memory")))

    def test_memory_help_mentions_fixes(self) -> None:
        message = str(_memory_error_help(RuntimeError("CUDA failed with error out of memory")))
        self.assertIn("--batch-size 0", message)
        self.assertIn("--device cpu", message)


class TestHelpers(unittest.TestCase):
    def test_sanitize_name(self) -> None:
        self.assertEqual(sanitize_name('a<b>:"/\\|?*b'), "a_b" + "_" * 8 + "b")
        self.assertEqual(sanitize_name("  ...  "), "transcript")


class TestFFmpegPipeline(unittest.TestCase):
    """用 ffmpeg 现场合成素材，验证媒体探测与音轨提取。"""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.ffmpeg = find_ffmpeg()
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest(f"未找到 ffmpeg：{exc}") from exc
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)

        cls.wav = cls.tmp / "tone.wav"
        subprocess.run(
            [cls.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
             "-i", "sine=frequency=440:duration=2", "-ac", "2", "-ar", "44100", str(cls.wav)],
            check=True,
        )

        cls.video = cls.tmp / "clip.mp4"
        subprocess.run(
            [cls.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "color=c=navy:s=320x240:r=10:duration=2",
             "-i", str(cls.wav), "-shortest", "-c:v", "mpeg4", "-c:a", "aac", str(cls.video)],
            check=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_probe_video(self) -> None:
        info = probe_media(self.video)
        self.assertTrue(info.has_video)
        self.assertTrue(info.has_audio)
        self.assertAlmostEqual(info.duration, 2.0, delta=0.4)

    def test_extract_audio(self) -> None:
        out = self.tmp / "extracted.wav"
        extract_audio(self.video, out)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 1000)

        info = probe_media(out)
        self.assertTrue(info.has_audio)
        self.assertFalse(info.has_video)  # -vn 已去视频
        self.assertAlmostEqual(info.duration, 2.0, delta=0.4)

    def test_extract_audio_clip(self) -> None:
        out = self.tmp / "clip01.wav"
        extract_audio(self.video, out, start=0.5, duration=0.5)
        info = probe_media(out)
        self.assertAlmostEqual(info.duration, 0.5, delta=0.3)

    def test_probe_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            probe_media(self.tmp / "nope.mp4")


class InterpreterHintTest(unittest.TestCase):
    """跑错 Python 是新手最容易踩的坑，提示必须准确且只在真的跑错时出现。"""

    def test_quiet_when_running_in_the_project_venv(self) -> None:
        venv = project_python()
        if venv is None:
            self.skipTest("本机没有 .venv（CI 就是这样），由注入用例覆盖")
        self.assertEqual(interpreter_hint(venv_python=venv, current=venv), "")

    def test_hint_names_both_interpreters(self) -> None:
        venv = Path("/proj/.venv/bin/python")
        other = Path("/usr/bin/python3")
        text = interpreter_hint(venv_python=venv, current=other)
        self.assertIn(str(venv), text)
        self.assertIn(str(other), text)
        # 要给出可照做的两条路：直接用 venv 的 python / 先激活
        self.assertIn("-m video2context", text)
        self.assertIn("Activate.ps1", text)

    def test_no_hint_when_project_has_no_venv(self) -> None:
        # 没建过虚拟环境就谈不上"用错了"，不该误导用户。
        # 注意 venv_python=None 的含义是"自动探测"，要模拟"没有 .venv"得 patch 探测函数。
        with mock.patch("video2context.config.project_python", return_value=None):
            self.assertEqual(interpreter_hint(current=Path("/usr/bin/python3")), "")

    def test_project_python_is_an_existing_file(self) -> None:
        found = project_python()
        if found is not None:
            self.assertTrue(found.is_file())


class MissingFfmpegMessageTest(unittest.TestCase):
    """「未找到 ffmpeg」的提示要先说环境问题（如果是环境问题），再给安装方式。"""

    def test_plain_message_when_environment_is_fine(self) -> None:
        from video2context import config, ffmpeg_tools

        with mock.patch.object(config, "interpreter_hint", return_value=""):
            text = ffmpeg_tools._missing_ffmpeg_message()
        self.assertTrue(text.startswith("未找到 ffmpeg。任选一种方式解决："))
        self.assertNotIn("你现在用的不是工程的虚拟环境", text)

    def test_environment_hint_comes_before_install_options(self) -> None:
        from video2context import config, ffmpeg_tools

        with mock.patch.object(config, "interpreter_hint", return_value="\n⚠ 你现在用的不是工程的虚拟环境：\n"):
            text = ffmpeg_tools._missing_ffmpeg_message()
        self.assertLess(
            text.index("你现在用的不是工程的虚拟环境"),
            text.index("pip install imageio-ffmpeg"),
            "环境诊断要排在安装建议前面 —— 否则用户会照着去装，把 base 环境也搞乱",
        )


class DoctorInterpreterWarningTest(unittest.TestCase):
    """doctor 是大家检查环境的第一站，所以"用错 Python"必须在这里就点出来。"""

    def _run_doctor(self) -> str:
        import argparse
        import contextlib
        import io

        from video2context import cli

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            cli._cmd_doctor(argparse.Namespace())
        return buffer.getvalue()

    def test_warns_when_using_the_wrong_python(self) -> None:
        from video2context import config

        with mock.patch.object(config, "interpreter_hint", return_value="\n⚠ 你现在用的不是工程的虚拟环境：\n"):
            text = self._run_doctor()
        self.assertIn("你现在用的不是工程的虚拟环境", text)

    def test_ffmpeg_line_does_not_repeat_the_long_hint(self) -> None:
        from video2context import config, ffmpeg_tools

        hint = "\n⚠ 你现在用的不是工程的虚拟环境：\n"
        not_found = ffmpeg_tools.FFmpegNotFoundError("未找到 ffmpeg。")
        with (
            mock.patch.object(config, "interpreter_hint", return_value=hint),
            mock.patch.object(ffmpeg_tools, "find_ffmpeg", side_effect=not_found),
        ):
            text = self._run_doctor()
        self.assertIn("原因见上方", text)  # 同一条长提示不该刷两遍
        self.assertEqual(text.count("你现在用的不是工程的虚拟环境"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
