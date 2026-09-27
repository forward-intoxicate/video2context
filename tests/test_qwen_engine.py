"""Qwen3-ASR 引擎的离线单元测试。

全部不需要显卡、网络或 ``.venv-qwen``：只测参数归一化、时间戳切句、
无时间戳时的退化分配，以及引擎选择与词表提示的分支。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video2context.config import ENGINE_ENV_VAR, configured_engine  # noqa: E402
from video2context.glossary import Glossary, SymbolRepair  # noqa: E402
from video2context.pipeline import (  # noqa: E402
    ENGINE_FASTER_WHISPER,
    ENGINE_QWEN3_ASR,
    TranscribeOptions,
    build_transcriber,
    glossary_prompt_for,
    is_qwen_engine,
    normalize_engine,
    resolve_engine,
)
from video2context.qwen_engine import (  # noqa: E402
    _map_tokens_to_text,
    distribute_sentences,
    find_qwen_python,
    group_time_stamps,
    normalize_language_name,
    resolve_aligner,
    split_text_into_sentences,
)

#: 一段"英文"样例：token 是去掉标点的（对齐模型的真实行为）
EN_TEXT = "Hello world. This is, a test!"
EN_STAMPS = [
    {"text": "Hello", "start": 0.0, "end": 0.5},
    {"text": "world", "start": 0.5, "end": 1.0},
    {"text": "This", "start": 1.2, "end": 1.5},
    {"text": "is", "start": 1.5, "end": 1.7},
    {"text": "a", "start": 1.7, "end": 1.8},
    {"text": "test", "start": 1.8, "end": 2.2},
]

#: 中文样例（同样是去标点的 token）
ZH_TEXT = "今天我们讲导数。先看定义，再看例子。"
ZH_STAMPS = [
    {"text": "今", "start": 0.0, "end": 0.2},
    {"text": "天", "start": 0.2, "end": 0.4},
    {"text": "我", "start": 0.4, "end": 0.6},
    {"text": "们", "start": 0.6, "end": 0.8},
    {"text": "讲", "start": 0.8, "end": 1.0},
    {"text": "导", "start": 1.0, "end": 1.2},
    {"text": "数", "start": 1.2, "end": 1.4},
    {"text": "先", "start": 1.8, "end": 2.0},
    {"text": "看", "start": 2.0, "end": 2.2},
    {"text": "定", "start": 2.2, "end": 2.4},
    {"text": "义", "start": 2.4, "end": 2.6},
    {"text": "再", "start": 2.7, "end": 2.9},
    {"text": "看", "start": 2.9, "end": 3.1},
    {"text": "例", "start": 3.1, "end": 3.3},
    {"text": "子", "start": 3.3, "end": 3.5},
]


class LanguageNameTest(unittest.TestCase):
    def test_iso_codes_map_to_canonical_names(self) -> None:
        self.assertEqual(normalize_language_name("zh"), "Chinese")
        self.assertEqual(normalize_language_name("EN"), "English")
        self.assertEqual(normalize_language_name("yue"), "Cantonese")
        self.assertEqual(normalize_language_name("ja"), "Japanese")

    def test_auto_and_empty_become_none(self) -> None:
        for value in (None, "", "auto", "自动", "自动检测", "none"):
            self.assertIsNone(normalize_language_name(value))

    def test_unknown_value_passes_through(self) -> None:
        # 交给上游去报错，比在这里悄悄改成自动检测更好排查
        self.assertEqual(normalize_language_name("Klingon"), "Klingon")


class AlignerResolveTest(unittest.TestCase):
    def test_disabled_values_turn_it_off(self) -> None:
        for value in ("off", "none", "no", "false", "0", "无", "OFF"):
            self.assertIsNone(resolve_aligner(value))

    def test_explicit_path_wins(self) -> None:
        self.assertEqual(resolve_aligner("D:/models/aligner"), "D:/models/aligner")

    def test_autodetect_returns_something(self) -> None:
        # 本地没有就退回仓库名；两种都算"探测成功"
        self.assertTrue(resolve_aligner(None))


class MapTokensTest(unittest.TestCase):
    def test_tokens_map_back_to_original_positions(self) -> None:
        spans = _map_tokens_to_text(EN_TEXT, EN_STAMPS)
        self.assertIsNotNone(spans)
        assert spans is not None
        self.assertEqual(len(spans), len(EN_STAMPS))
        # "Hello" 落在原文 0..4
        self.assertEqual(spans[0], (0, 4))
        # 每个 token 切片回来应当就是它自己
        for stamp, (start, end) in zip(EN_STAMPS, spans):
            self.assertEqual(EN_TEXT[start : end + 1], stamp["text"])

    def test_returns_none_when_nothing_matches(self) -> None:
        stamps = [{"text": "完全不相干", "start": 0.0, "end": 1.0}]
        self.assertIsNone(_map_tokens_to_text(EN_TEXT, stamps))


class GroupTimeStampsTest(unittest.TestCase):
    def test_english_splits_on_period_and_keeps_punctuation(self) -> None:
        segments = group_time_stamps(EN_TEXT, EN_STAMPS)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].text, "Hello world.")
        self.assertEqual(segments[1].text, "This is, a test!")
        self.assertAlmostEqual(segments[0].start, 0.0)
        self.assertAlmostEqual(segments[0].end, 1.0)

    def test_chinese_splits_on_full_stop(self) -> None:
        segments = group_time_stamps(ZH_TEXT, ZH_STAMPS)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].text, "今天我们讲导数。")
        self.assertEqual(segments[1].text, "先看定义，再看例子。")

    def test_segments_reconstruct_the_full_text(self) -> None:
        segments = group_time_stamps(ZH_TEXT, ZH_STAMPS)
        joined = "".join(s.text for s in segments)
        self.assertEqual(joined, ZH_TEXT)

    def test_no_stamps_means_no_segments(self) -> None:
        self.assertEqual(group_time_stamps(ZH_TEXT, []), [])


class SentenceSplitTest(unittest.TestCase):
    def test_splits_on_sentence_punctuation(self) -> None:
        self.assertEqual(
            split_text_into_sentences("你好。世界！真的吗？"),
            ["你好。", "世界！", "真的吗？"],
        )

    def test_empty_text(self) -> None:
        self.assertEqual(split_text_into_sentences(""), [])
        self.assertEqual(split_text_into_sentences("   "), [])

    def test_long_unpunctuated_text_is_split(self) -> None:
        text = "word " * 60
        sentences = split_text_into_sentences(text)
        self.assertGreater(len(sentences), 1)


class DistributeSentencesTest(unittest.TestCase):
    def test_proportional_inside_one_region(self) -> None:
        segments = distribute_sentences(["AAAA", "BB"], [(0.0, 6.0)], 6.0)
        self.assertEqual(len(segments), 2)
        self.assertAlmostEqual(segments[0].start, 0.0)
        self.assertAlmostEqual(segments[0].end, 4.0)
        self.assertAlmostEqual(segments[1].start, 4.0)
        self.assertAlmostEqual(segments[1].end, 6.0)

    def test_skips_silence_between_regions(self) -> None:
        segments = distribute_sentences(["AAAA", "BBBB"], [(0.0, 2.0), (8.0, 10.0)], 10.0)
        self.assertAlmostEqual(segments[0].start, 0.0)
        self.assertAlmostEqual(segments[0].end, 2.0)
        # 第二句从第二个说话区间开始，而不是紧接着 2.0 秒
        self.assertAlmostEqual(segments[1].start, 8.0)

    def test_without_regions_covers_whole_timeline(self) -> None:
        segments = distribute_sentences(["AAAA", "BBBB"], [], 10.0)
        self.assertAlmostEqual(segments[0].start, 0.0)
        self.assertAlmostEqual(segments[-1].end, 10.0)

    def test_empty_sentences(self) -> None:
        self.assertEqual(distribute_sentences([], [(0.0, 1.0)], 1.0), [])


class EngineSelectionTest(unittest.TestCase):
    def test_aliases(self) -> None:
        self.assertEqual(normalize_engine(None), ENGINE_FASTER_WHISPER)
        self.assertEqual(normalize_engine("whisper"), ENGINE_FASTER_WHISPER)
        self.assertEqual(normalize_engine("Qwen3-ASR"), ENGINE_QWEN3_ASR)
        self.assertEqual(normalize_engine("qwen"), ENGINE_QWEN3_ASR)
        self.assertTrue(is_qwen_engine("qwen3-asr"))
        self.assertFalse(is_qwen_engine("faster-whisper"))

    def test_build_transcriber_returns_qwen_engine(self) -> None:
        from video2context.qwen_engine import QwenAsrEngine

        opts = TranscribeOptions(engine=ENGINE_QWEN3_ASR, model="models/Qwen3-ASR-1.7B")
        engine = build_transcriber(opts, lambda _msg: None)
        self.assertIsInstance(engine, QwenAsrEngine)
        self.assertEqual(engine.model, "models/Qwen3-ASR-1.7B")

    def test_whisper_default_model_is_ignored_for_qwen(self) -> None:
        from video2context.transcriber import DEFAULT_MODEL

        opts = TranscribeOptions(engine=ENGINE_QWEN3_ASR, model=DEFAULT_MODEL)
        engine = build_transcriber(opts, lambda _msg: None)
        # 不该把 "large-v3" 当成 Qwen 模型
        self.assertNotEqual(engine.model, DEFAULT_MODEL)


class ResolveEngineTest(unittest.TestCase):
    """默认识别引擎：显式参数 > V2C_ENGINE > faster-whisper。"""

    def setUp(self) -> None:
        self._saved = {key: os.environ.get(key) for key in (ENGINE_ENV_VAR, "V2C_ENV_FILE")}
        # 指向一个不存在的文件，避免开发机上真实的 .env 干扰断言
        os.environ["V2C_ENV_FILE"] = str(Path(__file__).resolve().parent / "_no_such_env_file")
        os.environ.pop(ENGINE_ENV_VAR, None)

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_defaults_to_faster_whisper(self) -> None:
        self.assertEqual(resolve_engine(), ENGINE_FASTER_WHISPER)
        self.assertEqual(resolve_engine(""), ENGINE_FASTER_WHISPER)
        self.assertEqual(configured_engine(), "")

    def test_env_var_supplies_the_default(self) -> None:
        os.environ[ENGINE_ENV_VAR] = "qwen3-asr"
        self.assertEqual(configured_engine(), "qwen3-asr")
        self.assertEqual(resolve_engine(), ENGINE_QWEN3_ASR)

    def test_explicit_argument_beats_env_var(self) -> None:
        os.environ[ENGINE_ENV_VAR] = "qwen3-asr"
        self.assertEqual(resolve_engine("faster-whisper"), ENGINE_FASTER_WHISPER)
        self.assertEqual(resolve_engine("whisper"), ENGINE_FASTER_WHISPER)

    def test_env_var_accepts_aliases(self) -> None:
        os.environ[ENGINE_ENV_VAR] = "qwen"
        self.assertEqual(resolve_engine(), ENGINE_QWEN3_ASR)


class GlossaryPromptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.glossary = Glossary(
            domain="导数的运算法则",
            symbols=[SymbolRepair(wrong="右F4", right="u(x)"), SymbolRepair(wrong="位F4", right="v(x)")],
            terms=["可导"],
        )

    def test_whisper_gets_short_prompt(self) -> None:
        prompt = glossary_prompt_for(TranscribeOptions(engine=ENGINE_FASTER_WHISPER), self.glossary)
        assert prompt is not None
        self.assertIn("u(x)", prompt)
        self.assertLessEqual(len(prompt), 80)

    def test_qwen_gets_descriptive_context_not_instruction(self) -> None:
        context = glossary_prompt_for(TranscribeOptions(engine=ENGINE_QWEN3_ASR), self.glossary)
        assert context is not None
        self.assertIn("u(x)", context)
        # 不能用祈使句（实测会让模型复述 context 而不转写）
        for forbidden in ("请", "必须", "注意识别"):
            self.assertNotIn(forbidden, context)

    def test_initial_prompt_is_appended_for_both(self) -> None:
        opts = TranscribeOptions(engine=ENGINE_QWEN3_ASR, initial_prompt="这是一节数学课")
        context = glossary_prompt_for(opts, None)
        self.assertEqual(context, "这是一节数学课")

    def test_no_glossary_and_no_prompt(self) -> None:
        self.assertIsNone(glossary_prompt_for(TranscribeOptions(), None))


class QwenPythonTest(unittest.TestCase):
    def test_explicit_missing_path_is_not_silently_ignored(self) -> None:
        # 明确指定了就用它，不存在就报 None（而不是偷偷回退到 .venv-qwen）
        self.assertIsNone(find_qwen_python("Z:/definitely/not/here/python.exe"))

    def test_explicit_existing_path_wins(self) -> None:
        found = find_qwen_python(sys.executable)
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found, Path(sys.executable))

    def test_autodetect_returns_path_or_none(self) -> None:
        saved = __import__("os").environ.pop("V2C_QWEN_PYTHON", None)
        try:
            found = find_qwen_python()
        finally:
            if saved is not None:
                __import__("os").environ["V2C_QWEN_PYTHON"] = saved
        if found is not None:
            self.assertTrue(Path(found).is_file())


if __name__ == "__main__":
    unittest.main()
