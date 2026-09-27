"""极小的大模型客户端（OpenAI 兼容协议，零新增依赖）。

只用到标准库 ``urllib``：DeepSeek / 通义 / OpenAI / 本地 Ollama、vLLM
都提供 ``/chat/completions`` 这一套接口，换服务商只需要改 ``V2C_LLM_BASE_URL``。

密钥处理：只从 :mod:`video2context.config` 读取（环境变量或 .env），
报错信息与日志里一律做脱敏，**不会把密钥写进任何输出**。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import LLMSettings, llm_settings, redact

__all__ = ["LLMError", "LLMReply", "LLMClient"]


class LLMError(RuntimeError):
    """调用大模型失败（网络、鉴权、返回格式等）。"""


@dataclass
class LLMReply:
    """一次对话调用的结果与用量。"""

    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def usage_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


class LLMClient:
    """按需创建、可复用的最简客户端。"""

    def __init__(self, settings: Optional[LLMSettings] = None) -> None:
        self.settings = settings or llm_settings()

    @property
    def configured(self) -> bool:
        return self.settings.configured

    def chat(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> LLMReply:
        """发一条消息，返回模型回复。失败抛 :class:`LLMError`。"""
        if not self.configured:
            raise LLMError(
                "未配置大模型密钥。请设置环境变量 V2C_LLM_API_KEY，"
                "或把 V2C_LLM_API_KEY=sk-xxx 写进工程根目录的 .env 文件"
                "（该文件已在 .gitignore 中）。"
            )

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.settings.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.api_key}",
            },
            method="POST",
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.settings.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = redact(exc.read().decode("utf-8", errors="replace"))[:400]
            if exc.code in (401, 403):
                raise LLMError(
                    f"大模型鉴权失败（HTTP {exc.code}）：{body}\n"
                    f"当前密钥 {self.settings.masked_key}（来源：{self.settings.source}）。\n"
                    "请检查工程根目录 .env 里的 V2C_LLM_API_KEY 是否有效 ——"
                    "常见原因是密钥已过期/被吊销，或复制时带了空格、引号。"
                ) from exc
            if exc.code == 402 or "insufficient" in body.lower():
                raise LLMError(f"大模型账户余额不足（HTTP {exc.code}）：{body}") from exc
            if exc.code == 429:
                raise LLMError(f"大模型触发限流（HTTP 429），稍后重试：{body}") from exc
            raise LLMError(f"大模型返回 HTTP {exc.code}：{body}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(
                f"连不上大模型服务（{self.settings.base_url}）：{redact(str(exc.reason))}"
            ) from exc
        except Exception as exc:  # pragma: no cover - 兜底
            raise LLMError(f"调用大模型失败：{redact(str(exc))}") from exc

        elapsed = time.perf_counter() - started
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"大模型返回格式异常：{redact(json.dumps(data)[:300])}") from exc

        usage = data.get("usage") or {}
        return LLMReply(
            text=text or "",
            model=str(data.get("model") or self.settings.model),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            elapsed_seconds=elapsed,
            raw=data,
        )
