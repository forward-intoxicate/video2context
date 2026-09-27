"""流水线端到端测试：用假识别器替换模型，验证整条链路与输出格式。

不需要下载模型、不需要显卡，只用 ffmpeg 现造素材。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 便于 import media_factory

from media_factory import build_video  # noqa: E402

from video2context.ffmpeg_tools import find_ffmpeg  # noqa: E402
from video2context.glossary import Glossary, SymbolRepair  # noqa: E402
from video2context.pipeline import TranscribeOptions, process  # noqa: E402
from video2context.transcriber import Segment, TranscriptionResult  # noqa: E402

DEFAULT_TEXT = "你好，世界。Hello world."


class StubTranscriber:
    """伪装成 Transcriber：接口一致，但不加载任何模型。

    ``texts`` 可以指定第 1、2、3… 次调用分别返回什么文本，用于测试回退逻辑。
    """

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.audio_path: Path | None = None
        self.transcribe_kwargs: dict = {}
        self.calls: list[dict] = []
        self.texts: list[str] = []

    def _text_for_call(self, index: int) -> str:
        return self.texts[min(index, len(self.texts) - 1)]

    def transcribe(self, audio_path, **kwargs) -> TranscriptionResult:
        self.audio_path = Path(audio_path)
        self.transcribe_kwargs = kwargs
        self.calls.append(dict(kwargs))

        if self.texts:  # 自定义文本：每次调用返回一整段
            text = self._text_for_call(len(self.calls) - 1)
            segments = [Segment(id=0, start=0.0, end=2.0, text=text, avg_logprob=-0.1, no_speech_prob=0.01)]
        else:  # 默认：两段中英混合
            text = DEFAULT_TEXT
            segments = [
                Segment(id=0, start=0.0, end=1.0, text="你好，世界。", avg_logprob=-0.1, no_speech_prob=0.01),
                Segment(id=1, start=1.0, end=2.0, text="Hello world.", avg_logprob=-0.2, no_speech_prob=0.02),
            ]

        callback = kwargs.get("progress_callback")
        if callback:
            callback(1.0, 2.0, segments[0].text)
            callback(2.0, 2.0, segments[-1].text)

        return TranscriptionResult(
            segments=segments,
            text=text,
            language="zh",
            language_probability=0.99,
            duration=2.0,
            duration_after_vad=2.0,
            task=kwargs.get("task", "transcribe"),
            model=str(self.init_kwargs.get("model", "stub")),
            device="cpu",
            compute_type="int8",
            batch_size=0,
            vad_filter=bool(kwargs.get("vad_filter", True)),
            beam_size=int(kwargs.get("beam_size", 5)),
            initial_prompt=kwargs.get("initial_prompt"),
            condition_on_previous_text=bool(kwargs.get("condition_on_previous_text", False)),
            temperature=kwargs.get("temperature"),
            elapsed_seconds=0.5,
            model_load_seconds=0.1,
        )


class PipelineTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            find_ffmpeg()
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest(f"未找到 ffmpeg：{exc}") from exc
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.video = build_video(cls.tmp / "clip.mp4", seconds=2.0, with_audio=True)
        cls.silent = build_video(cls.tmp / "silent.mp4", seconds=1.0, with_audio=False)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def setUp(self) -> None:
        self.out = self.tmp / self._testMethodName
        self.stub = StubTranscriber()

        def _factory(**kwargs):
            self.stub.init_kwargs = kwargs  # 记录 pipeline 传给识别器的构造参数
            return self.stub

        patcher = mock.patch("video2context.pipeline.get_transcriber", side_effect=_factory)
        self.addCleanup(patcher.stop)
        patcher.start()

    # ------------------------------------------------------------------ 正常路径

    def test_payload_schema(self) -> None:
        result = process(self.video, TranscribeOptions(formats=("json",), output_dir=self.out))

        payload = json.loads(result.json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], "1.1")
        self.assertEqual(payload["generator"]["name"], "video2context")
        self.assertNotIn("glossary", payload)  # 没用词表时不应出现该字段

        source = payload["source"]
        self.assertEqual(source["filename"], "clip.mp4")
        self.assertTrue(source["has_video"])
        self.assertTrue(source["has_audio"])
        self.assertAlmostEqual(source["duration"], 2.0, delta=0.4)
        self.assertTrue(source["format"])

        asr = payload["asr"]
        self.assertEqual(asr["engine"], "faster-whisper")
        self.assertEqual(asr["language"], "zh")
        self.assertEqual(asr["segments_count"], 2)
        self.assertEqual(asr["device"], "cpu")
        self.assertIn("realtime_factor", asr)
        self.assertIn("condition_on_previous_text", asr)

        self.assertEqual(payload["text"], "你好，世界。Hello world.")
        self.assertEqual(len(payload["segments"]), 2)
        self.assertEqual(payload["segments"][0]["text"], "你好，世界。")

    def test_default_output_is_json_only(self) -> None:
        result = process(self.video, TranscribeOptions(output_dir=self.out))
        self.assertEqual(set(result.outputs), {"json"})
        self.assertTrue(result.json_path.exists())

    def test_all_output_formats(self) -> None:
        result = process(
            self.video, TranscribeOptions(formats=("json", "txt", "srt", "vtt"), output_dir=self.out)
        )
        self.assertEqual(set(result.outputs), {"json", "txt", "srt", "vtt"})

        srt = result.outputs["srt"].read_text(encoding="utf-8")
        self.assertIn("1\n00:00:00,000 --> 00:00:01,000\n你好，世界。", srt)
        vtt = result.outputs["vtt"].read_text(encoding="utf-8")
        self.assertTrue(vtt.startswith("WEBVTT"))
        self.assertIn("00:00:01.000 --> 00:00:02.000\nHello world.", vtt)
        txt = result.outputs["txt"].read_text(encoding="utf-8")
        self.assertEqual(txt, "你好，世界。\nHello world.\n")

    def test_progress_callback_is_invoked(self) -> None:
        seen: list[tuple[float, float]] = []
        process(
            self.video,
            TranscribeOptions(output_dir=self.out),
            progress_callback=lambda processed, total, text: seen.append((processed, total)),
        )
        self.assertEqual(seen[-1], (2.0, 2.0))

    def test_options_are_forwarded(self) -> None:
        process(
            self.video,
            TranscribeOptions(
                model="tiny",
                language="en",
                task="translate",
                beam_size=3,
                initial_prompt="专有名词",
                temperature=[0.0, 0.2],
                output_dir=self.out,
            ),
        )
        kwargs = self.stub.transcribe_kwargs
        self.assertEqual(kwargs["language"], "en")
        self.assertEqual(kwargs["task"], "translate")
        self.assertEqual(kwargs["beam_size"], 3)
        self.assertEqual(kwargs["initial_prompt"], "专有名词")
        self.assertEqual(kwargs["temperature"], [0.0, 0.2])
        self.assertEqual(self.stub.init_kwargs["model"], "tiny")

    # ------------------------------------------------------------------ 命名与落盘

    def test_name_and_overwrite(self) -> None:
        first = process(self.video, TranscribeOptions(output_dir=self.out, name="report"))
        self.assertEqual(first.json_path.name, "report.json")

        second = process(self.video, TranscribeOptions(output_dir=self.out, name="report"))
        self.assertEqual(second.json_path.name, "report.json")  # 显式命名时覆盖

        third = process(self.video, TranscribeOptions(output_dir=self.out))
        fourth = process(self.video, TranscribeOptions(output_dir=self.out))
        self.assertEqual(third.json_path.name, "clip.json")
        self.assertEqual(fourth.json_path.name, "clip-1.json")  # 自动避让

    def test_keep_audio(self) -> None:
        result = process(self.video, TranscribeOptions(output_dir=self.out, keep_audio=True))
        self.assertIsNotNone(result.audio_path)
        self.assertTrue(result.audio_path.exists())
        payload = json.loads(result.json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["audio"]["sample_rate"], 16000)
        self.assertEqual(payload["audio"]["channels"], 1)
        self.assertTrue(payload["audio"]["kept"])
        self.assertTrue(payload["audio"]["path"])

    def test_temp_audio_is_removed_by_default(self) -> None:
        result = process(self.video, TranscribeOptions(output_dir=self.out))
        self.assertIsNone(result.audio_path)
        self.assertFalse(self.stub.audio_path.exists())  # 临时 wav 已清理
        payload = json.loads(result.json_path.read_text(encoding="utf-8"))
        self.assertFalse(payload["audio"]["kept"])
        self.assertIsNone(payload["audio"]["path"])

    # ------------------------------------------------------------------ 错误路径

    def test_missing_input_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            process(self.tmp / "nope.mp4", TranscribeOptions(output_dir=self.out))

    def test_video_without_audio_raises(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            process(self.silent, TranscribeOptions(output_dir=self.out))
        self.assertIn("音频", str(ctx.exception))

    def test_directory_as_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            process(self.tmp, TranscribeOptions(output_dir=self.out))

    # ------------------------------------------------------------------ 领域词表

    def test_glossary_is_applied_to_prompt(self) -> None:
        glossary = Glossary(
            domain="数学分析，导数运算",
            symbols=[SymbolRepair("右F4", "u(x)"), SymbolRepair("位F4", "v(x)")],
        )
        self.stub.texts = ["u(x) 和 v(x) 都可导"]
        result = process(self.video, TranscribeOptions(glossary=glossary, output_dir=self.out))

        payload = json.loads(result.json_path.read_text(encoding="utf-8"))
        self.assertIn("glossary", payload)
        self.assertEqual(payload["glossary"]["symbols"][0], {"from": "右F4", "to": "u(x)"})
        self.assertIn("u(x)", self.stub.transcribe_kwargs["initial_prompt"])
        self.assertNotIn("右F4", self.stub.transcribe_kwargs["initial_prompt"])
        # 成功的偏置不应触发回退
        self.assertFalse(payload["glossary"]["verification"]["fallback_used"])
        self.assertEqual(payload["glossary"]["verification"]["expected_hits"], 2)
        self.assertEqual(len(self.stub.calls), 1)

    def test_glossary_bias_failure_falls_back(self) -> None:
        glossary = Glossary(symbols=[SymbolRepair("右F4", "u(x)")])
        self.stub.texts = ["右F4 如果在 F0 处可倒", "回退之后的普通文本"]
        result = process(self.video, TranscribeOptions(glossary=glossary, output_dir=self.out))

        payload = json.loads(result.json_path.read_text(encoding="utf-8"))
        verification = payload["glossary"]["verification"]
        self.assertTrue(verification["fallback_used"])
        self.assertEqual(verification["expected_hits"], 0)
        self.assertGreater(verification["wrong_hits"], 0)
        self.assertEqual(payload["text"], "回退之后的普通文本")
        self.assertEqual(len(self.stub.calls), 2)  # 一次带词表 + 一次回退

    def test_auto_glossary_runs_two_passes(self) -> None:
        built = Glossary(
            domain="数学分析",
            symbols=[SymbolRepair("右F4", "u(x)")],
            scan={"duration": 1.0, "model": "stub", "chars": 4},
        )
        self.stub.texts = ["粗转写文本", "u(x) 加 v(x)"]
        with mock.patch("video2context.pipeline.build_glossary_from_scan", return_value=built) as builder:
            result = process(
                self.video,
                TranscribeOptions(auto_glossary=True, scan_duration=1.0, output_dir=self.out),
            )

        self.assertTrue(builder.called)
        self.assertEqual(builder.call_args[0][0], "粗转写文本")  # 第一遍的粗转写被交给 LLM
        self.assertEqual(len(self.stub.calls), 2)  # 粗转写 + 正式转写
        self.assertIsNone(self.stub.calls[0].get("initial_prompt"))  # 第一遍不带提示词
        self.assertIn("u(x)", self.stub.calls[1]["initial_prompt"])  # 第二遍带词表

        payload = json.loads(result.json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["glossary"]["source"], "llm")
        self.assertEqual(payload["glossary"]["scan"]["chars"], 4)

    def test_glossary_out_file_written(self) -> None:
        glossary = Glossary(domain="数学分析", symbols=[SymbolRepair("右F4", "u(x)")])
        target = self.out / "terms.json"
        self.stub.texts = ["u(x) 可导"]
        process(
            self.video,
            TranscribeOptions(glossary=glossary, glossary_out=target, output_dir=self.out),
        )
        saved = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(saved["prompt"], "数学分析。符号：u(x)。")


if __name__ == "__main__":
    unittest.main(verbosity=2)
