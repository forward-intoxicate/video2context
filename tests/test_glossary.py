"""词表与配置模块的离线单元测试（不联网、不需要密钥）。"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video2context.config import (  # noqa: E402
    LLMSettings,
    load_env_file,
    mask_secret,
    parse_env_text,
    redact,
)
from video2context.glossary import (  # noqa: E402
    Glossary,
    SymbolRepair,
    parse_llm_json,
    symbol_hit_score,
)

# 故意用可识别的假密钥（含 example 关键字，会被 scripts/check_secrets.py 放行）
KEY = "sk-example-000000000000abcd"


class TestParseEnvText(unittest.TestCase):
    def test_basic(self) -> None:
        values = parse_env_text(
            "\n".join(
                [
                    "# 注释行",
                    "",
                    "V2C_LLM_API_KEY=sk-abc123",
                    'V2C_LLM_BASE_URL="https://api.deepseek.com"',
                    "export V2C_LLM_MODEL=deepseek-chat",
                    "V2C_LLM_TIMEOUT=30  # 行尾注释",
                    "坏行没有等号",
                ]
            )
        )
        self.assertEqual(values["V2C_LLM_API_KEY"], "sk-abc123")
        self.assertEqual(values["V2C_LLM_BASE_URL"], "https://api.deepseek.com")
        self.assertEqual(values["V2C_LLM_MODEL"], "deepseek-chat")
        self.assertEqual(values["V2C_LLM_TIMEOUT"], "30")
        self.assertNotIn("坏行没有等号", values)

    def test_tolerates_bom(self) -> None:
        # PowerShell / 记事本 写出的 UTF-8 会带 BOM
        values = parse_env_text("\ufeffV2C_LLM_API_KEY=sk-example-1234\n")
        self.assertEqual(values["V2C_LLM_API_KEY"], "sk-example-1234")

    def test_load_env_file_does_not_override_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("V2C_TEST_VALUE=from_file\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"V2C_TEST_VALUE": "from_env"}, clear=False):
                load_env_file(env_path)
                self.assertEqual(os.environ["V2C_TEST_VALUE"], "from_env")
                load_env_file(env_path, override=True)
                self.assertEqual(os.environ["V2C_TEST_VALUE"], "from_file")
            os.environ.pop("V2C_TEST_VALUE", None)


class TestSecretMasking(unittest.TestCase):
    def test_mask_secret(self) -> None:
        self.assertEqual(mask_secret(KEY), "sk-***abcd")
        self.assertEqual(mask_secret(""), "（未配置）")
        self.assertEqual(mask_secret(None), "（未配置）")
        self.assertEqual(mask_secret("short"), "***")
        self.assertNotIn(KEY[3:-4], mask_secret(KEY))

    def test_redact(self) -> None:
        text = f"Authorization: Bearer {KEY} 失败"
        cleaned = redact(text)
        self.assertNotIn(KEY, cleaned)
        self.assertIn("***", cleaned)

    def test_settings_describe_hides_key(self) -> None:
        settings = LLMSettings(api_key=KEY, model="deepseek-chat", source=".env")
        described = settings.describe()
        self.assertIn("sk-***abcd", described)
        self.assertNotIn(KEY, described)
        self.assertFalse(LLMSettings().configured)


class TestParseLlmJson(unittest.TestCase):
    def test_variants(self) -> None:
        self.assertEqual(parse_llm_json('{"domain": "数学"}')["domain"], "数学")
        fenced = '好的，结果如下：\n```json\n{"symbols": [["右F4", "u(x)"]]}\n```\n以上。'
        self.assertEqual(parse_llm_json(fenced)["symbols"][0][1], "u(x)")
        self.assertEqual(parse_llm_json('前言 {"domain": "x"} 后语')["domain"], "x")
        self.assertEqual(parse_llm_json("完全不是 JSON"), {})
        self.assertEqual(parse_llm_json(""), {})
        self.assertEqual(parse_llm_json("[1, 2, 3]"), {})


class TestGlossaryFromLlm(unittest.TestCase):
    def test_from_llm_json(self) -> None:
        data = {
            "domain": "数学分析，导数运算",
            "symbols": [["右F4", "u(x)"], ["位F4", "v(x)"], ["F0", "x0"], "可导"],
        }
        glossary = Glossary.from_llm_json(data)
        self.assertEqual(len(glossary.symbols), 4)
        self.assertEqual(glossary.symbols[0].to_dict(), {"from": "右F4", "to": "u(x)"})
        self.assertEqual(glossary.symbols[3].right, "可导")  # 容错：只给了正确写法

    def test_skips_identity_pairs(self) -> None:
        glossary = Glossary.from_llm_json({"symbols": [["u", "u"], ["", "x0"], ["右", ""]]})
        self.assertEqual(len(glossary.symbols), 0)

    def test_prompt_is_minimal_and_capped(self) -> None:
        glossary = Glossary(
            domain="数学分析，导数运算",
            symbols=[SymbolRepair("a", "u(x)"), SymbolRepair("b", "v(x)")],
        )
        prompt = glossary.to_prompt()
        self.assertEqual(prompt, "数学分析，导数运算。符号：u(x)、v(x)。")
        self.assertNotIn("右F4", prompt)  # 错误写法绝不能进 prompt

        many = Glossary(symbols=[SymbolRepair(f"w{i}", f"s{i}(x)") for i in range(40)])
        self.assertLessEqual(len(many.to_prompt()), 80)

    def test_empty_glossary_gives_empty_prompt(self) -> None:
        self.assertEqual(Glossary().to_prompt(), "")


class TestGlossaryFromFile(unittest.TestCase):
    def test_parse_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "terms.txt"
            path.write_text(
                "# 注释\n右F4 -> u(x)\n位F4=>v(x)\n可导\n求导\n\n",
                encoding="utf-8",
            )
            glossary = Glossary.from_file(path)
        self.assertEqual(glossary.source, "file")
        self.assertEqual([s.right for s in glossary.symbols], ["u(x)", "v(x)"])
        self.assertEqual(glossary.terms, ["可导", "求导"])
        self.assertEqual(glossary.to_prompt(), "符号：u(x)、v(x)。术语：可导、求导。")


class TestSymbolHitScore(unittest.TestCase):
    def setUp(self) -> None:
        self.glossary = Glossary(
            symbols=[
                SymbolRepair("右F4", "u(x)"),
                SymbolRepair("位F4", "v(x)"),
                SymbolRepair("F0", "x0"),
                SymbolRepair("求到", "求导"),
            ]
        )

    def test_detects_success(self) -> None:
        score = symbol_hit_score("u x 和 v x 如果在 x0 处可导", self.glossary)
        self.assertEqual(score.expected_hits, 3)
        self.assertEqual(score.wrong_hits, 0)
        self.assertEqual(sorted(score.hit_probes), ["u", "v", "x0"])
        self.assertEqual(score.unresolved, [])
        self.assertFalse(score.bias_failed)
        self.assertFalse(score.partially_failed)

    def test_detects_partial_failure(self) -> None:
        # u 被修好，但"位F4"这个错误写法还在 → 部分失效：不清空结果，只提醒
        score = symbol_hit_score("u x 和位F4如果在 x0 处可导", self.glossary)
        self.assertEqual(score.hit_probes, ["u", "x0"])
        self.assertEqual(score.unresolved, ["位F4→v(x)"])
        self.assertTrue(score.partially_failed)
        self.assertFalse(score.bias_failed)  # 不算完全失效，不该回退

    def test_missing_word_is_not_a_failure(self) -> None:
        # 音频里压根没出现"求到/求导" → 不能算偏置失败
        score = symbol_hit_score("u x 和 v x 在 x0 处", self.glossary)
        self.assertIn("求导", score.missing_probes)
        self.assertFalse(score.bias_failed)
        self.assertFalse(score.partially_failed)

    def test_detects_failure(self) -> None:
        score = symbol_hit_score("右F4和位F4如果在F0处可倒", self.glossary)
        self.assertEqual(score.expected_hits, 0)
        self.assertEqual(score.wrong_hits, 3)
        self.assertEqual(score.hit_probes, [])
        self.assertEqual(set(score.unresolved), {"右F4→u(x)", "位F4→v(x)", "F0→x0"})
        self.assertTrue(score.bias_failed)

    def test_chinese_probe(self) -> None:
        score = symbol_hit_score("先相加再求导等于先求导再相加", self.glossary)
        self.assertGreaterEqual(score.expected_hits, 2)

    def test_empty_inputs(self) -> None:
        self.assertEqual(symbol_hit_score("", self.glossary).expected_hits, 0)
        self.assertFalse(symbol_hit_score("", self.glossary).bias_failed)  # 空文本不判失败
        self.assertEqual(symbol_hit_score("随便", Glossary()).expected_hits, 0)
        self.assertFalse(symbol_hit_score("随便", Glossary()).bias_failed)  # 无符号也不判失败


if __name__ == "__main__":
    unittest.main(verbosity=2)
