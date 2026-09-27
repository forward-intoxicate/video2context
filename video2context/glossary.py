"""领域词表：从粗转写里还原被听错的符号/术语，再拼成 Whisper 的提示词。

这个模块的设计**完全基于实测结论**（见 docs/glossary.md）：

* LLM 能可靠地把「右F4」还原成「u(x)」——**前提是只让它输出还原结果**；
* 一旦让它输出"术语列表"，它会把错别字（右F4、位F4）当成术语抄进去，
  拼进 prompt 反而**比不加提示词更差**。所以这里只保留 ``symbols`` 字段。
* prompt 必须**极短**（几十字），只放还原后的符号，不放长词表。

另外提供 :func:`symbol_hit_score`：判定提示词偏置**是否真的生效**。
实测 Whisper 的提示词偏置存在"静默失效"（同样的正确符号，措辞一变就完全无效），
所以必须靠这个评分来决定是否回退。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .config import redact
from .llm import LLMClient, LLMError, LLMReply

__all__ = [
    "SymbolRepair",
    "Glossary",
    "GlossaryScore",
    "build_glossary_from_scan",
    "parse_llm_json",
    "symbol_hit_score",
    "SCAN_PROMPT_TEMPLATE",
]

#: 喂给 LLM 的粗转写最多截取多少字符（控制 token 消耗）
MAX_SCAN_CHARS = 1500
#: 拼给 Whisper 的 prompt 最多多少字符（实测 30~80 字区间稳定有效）
MAX_PROMPT_CHARS = 80
#: prompt 里最多放几个符号
MAX_SYMBOLS = 15

_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z0-9]*")


@dataclass
class SymbolRepair:
    """一条「听错的写法 → 正确写法」。"""

    wrong: str
    right: str

    def to_dict(self) -> dict[str, str]:
        return {"from": self.wrong, "to": self.right}


@dataclass
class GlossaryScore:
    """提示词偏置的效果评分。

    判断依据是**证据**而不是"猜词有没有出现"：只有当真结果里**仍然出现已知错误写法**
    时，才说明这个词的偏置没生效（如果某个词压根没在音频里出现，不算失败）。
    """

    expected_hits: int = 0
    wrong_hits: int = 0
    probes: list[str] = field(default_factory=list)
    hit_probes: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)  # 形如 "右F4→u(x)"，错误写法仍在结果里

    @property
    def missing_probes(self) -> list[str]:
        """一个都没命中的期望符号（可能只是音频里没出现这个词）。"""
        return [probe for probe in self.probes if probe not in self.hit_probes]

    @property
    def coverage(self) -> float:
        """命中的符号占全部期望符号的比例。"""
        return len(self.hit_probes) / len(self.probes) if self.probes else 1.0

    @property
    def bias_failed(self) -> bool:
        """期望符号一个都没出现、却仍有错误写法残留 → 偏置完全失效（应当回退）。"""
        return not self.hit_probes and self.wrong_hits > 0

    @property
    def partially_failed(self) -> bool:
        """部分修好、部分仍有错误写法残留 → 保留结果但提醒用户。"""
        return bool(self.hit_probes) and self.wrong_hits > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_hits": self.expected_hits,
            "wrong_hits": self.wrong_hits,
            "probes": self.probes,
            "hit_probes": self.hit_probes,
            "missing_probes": self.missing_probes,
            "unresolved": self.unresolved,
            "coverage": round(self.coverage, 3),
        }


@dataclass
class Glossary:
    """一次转写使用的领域词表。"""

    domain: str = ""
    symbols: list[SymbolRepair] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    source: str = "llm"  # llm | file
    scan: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ 拼装

    def to_prompt(self, *, max_chars: int = MAX_PROMPT_CHARS) -> str:
        """拼成 Whisper 的 initial_prompt：极短、只含还原后的符号。"""
        parts: list[str] = []
        if self.domain:
            parts.append(self.domain.rstrip("。.") + "。")

        repaired = [s.right for s in self.symbols if s.right][:MAX_SYMBOLS]
        if repaired:
            parts.append("符号：" + "、".join(dict.fromkeys(repaired)) + "。")

        if self.terms:
            parts.append("术语：" + "、".join(dict.fromkeys(self.terms))[: max_chars // 2] + "。")

        prompt = "".join(parts)
        if len(prompt) > max_chars:
            prompt = prompt[: max_chars - 1].rstrip("、。") + "。"
        return prompt

    # ------------------------------------------------------------------ 序列化

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "source": self.source,
            "domain": self.domain,
            "symbols": [s.to_dict() for s in self.symbols],
            "terms": list(self.terms),
            "prompt": self.to_prompt(),
        }
        if self.scan:
            data["scan"] = self.scan
        if self.llm:
            data["llm"] = self.llm
        return data

    def describe(self) -> str:
        lines = [f"来源: {self.source}"]
        if self.domain:
            lines.append(f"领域: {self.domain}")
        if self.symbols:
            lines.append("符号还原:")
            for item in self.symbols:
                lines.append(f"  {item.wrong} → {item.right}")
        if self.terms:
            lines.append("术语: " + "、".join(self.terms))
        lines.append(f"拼成 prompt（{len(self.to_prompt())} 字）: {self.to_prompt()}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ 构造

    @classmethod
    def from_llm_json(cls, data: dict[str, Any], *, scan: Optional[dict[str, Any]] = None,
                      llm: Optional[dict[str, Any]] = None) -> Glossary:
        symbols: list[SymbolRepair] = []
        for item in data.get("symbols") or []:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                wrong, right = str(item[0]).strip(), str(item[1]).strip()
                if wrong and right and wrong != right:
                    symbols.append(SymbolRepair(wrong=wrong, right=right))
            elif isinstance(item, str) and item.strip():
                # 容错：模型只给了正确写法
                symbols.append(SymbolRepair(wrong="", right=item.strip()))
        terms = [str(t).strip() for t in (data.get("terms") or []) if str(t).strip()]
        return cls(
            domain=str(data.get("domain") or "").strip(),
            symbols=symbols,
            terms=terms,
            source="llm",
            scan=scan or {},
            llm=llm or {},
        )

    @classmethod
    def from_file(cls, path: str | Path) -> Glossary:
        """读手写词表：``错形 -> 正确`` 表示还原，单独一行则视为术语。"""
        text = Path(path).read_text(encoding="utf-8")
        symbols: list[SymbolRepair] = []
        terms: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            match = re.split(r"\s*(?:->|=>|→)\s*", line, maxsplit=1)
            if len(match) == 2 and match[0].strip() and match[1].strip():
                symbols.append(SymbolRepair(wrong=match[0].strip(), right=match[1].strip()))
            else:
                terms.append(line)
        return cls(symbols=symbols, terms=terms, source="file")


# ---------------------------------------------------------------------- LLM


SCAN_PROMPT_TEMPLATE = """下面是一段中文语音识别的粗转写结果，含同音字错误，也可能把拉丁字母或数学符号听成了汉字。

请只输出 JSON，不要任何解释：
1. domain：一句话判断这是什么领域、在讲什么（20 字以内）
2. symbols：逐句检查哪些汉字其实是被听错的拉丁字母或数学符号，
   输出 [["粗转写里的原词", "还原后的正确写法"], ...]；
   正确写法用数学形式（例如 u(x)、v(x)、x0、u+v、u1(x)），没有就输出空数组。

粗转写：
\"\"\"
{rough}
\"\"\"

输出格式：{{"domain": "...", "symbols": [["右F4", "u(x)"]]}}"""


def parse_llm_json(text: str) -> dict[str, Any]:
    """从模型回复里抠出 JSON（容忍 ```json 代码块和前后废话）。"""
    if not text:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        candidate = text[start : end + 1] if start >= 0 and end > start else ""
    if not candidate:
        return {}
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def build_glossary_from_scan(
    scan_text: str,
    *,
    client: Optional[LLMClient] = None,
    scan_meta: Optional[dict[str, Any]] = None,
    max_chars: int = MAX_SCAN_CHARS,
) -> Glossary:
    """把粗转写丢给大模型，得到还原后的词表。"""
    client = client or LLMClient()
    rough = (scan_text or "").strip()[:max_chars]
    if not rough:
        raise LLMError("粗转写为空，无法推断词表")

    reply: LLMReply = client.chat(
        SCAN_PROMPT_TEMPLATE.format(rough=rough),
        system="你是语音识别后处理助手，只输出 JSON。",
        temperature=0.0,
        max_tokens=800,
    )
    parsed = parse_llm_json(reply.text)
    if not parsed:
        raise LLMError(f"大模型没有返回可解析的 JSON：{redact(reply.text)[:200]}")

    return Glossary.from_llm_json(
        parsed,
        scan=dict(scan_meta or {}, text=rough[:400]),
        llm=reply.usage_dict(),
    )


# ---------------------------------------------------------------------- 评分


def _probes(symbols: Sequence[SymbolRepair]) -> list[str]:
    """从还原结果里提取"探针"：用于判断输出里是否真的出现了期望写法。"""
    probes: list[str] = []
    for item in symbols:
        right = (item.right or "").strip()
        if not right:
            continue
        latin = _LATIN_RUN.search(right)
        if latin:
            probes.append(latin.group(0))
        elif len(right) <= 4:
            probes.append(right)
    return list(dict.fromkeys(probes))


def symbol_hit_score(text: str, glossary: Glossary) -> GlossaryScore:
    """统计期望符号与已知错误写法在结果里各出现几次（逐符号）。"""
    if not text:  # 空文本无从判断，不能算作偏置失效
        return GlossaryScore()
    probes = _probes(glossary.symbols)
    hit_probes = [probe for probe in probes if probe in text]
    expected = sum(text.count(probe) for probe in probes)
    unresolved: list[str] = []
    wrong_hits = 0
    for item in glossary.symbols:
        if item.wrong and item.wrong in text:
            wrong_hits += text.count(item.wrong)
            unresolved.append(f"{item.wrong}→{item.right}")
    return GlossaryScore(
        expected_hits=expected,
        wrong_hits=wrong_hits,
        probes=probes,
        hit_probes=hit_probes,
        unresolved=unresolved,
    )
