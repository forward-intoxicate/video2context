"""配置与密钥管理。

设计原则：**密钥只从环境变量或 .env 文件读取，永不进入代码、日志和输出。**

读取优先级（高 → 低）：
    1. 进程环境变量（例如你在终端里 ``$env:V2C_LLM_API_KEY = "..."``）
    2. ``V2C_ENV_FILE`` 指向的文件
    3. 从当前目录向上找到的第一个 ``.env``
    4. 工程根目录的 ``.env``

``.env`` 格式（与 dotenv 基本一致）：

    # 注释
    V2C_LLM_API_KEY=sk-xxxx
    V2C_LLM_BASE_URL="https://api.deepseek.com"   # 引号可选
    export V2C_LLM_MODEL=deepseek-chat            # 兼容 export 前缀
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ENV_FILE_VAR = "V2C_ENV_FILE"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT = 120.0

#: 首选的默认识别引擎（用户没指定、也没配 V2C_ENGINE 时）。
#: 实际生效值还要看这台机器上装了哪个引擎，见 ``pipeline.resolve_engine()``。
#: 引擎名的归一化放在 pipeline 里做，这里只放常量，避免 config ←→ pipeline 循环依赖。
DEFAULT_ENGINE = "qwen3-asr"
ENGINE_ENV_VAR = "V2C_ENGINE"

#: 模型根目录。默认是工程内的 ``models/``；设了 ``V2C_MODELS_DIR`` 就用它 ——
#: 服务器上常把代码与权重分开放（权重放数据盘、或多个工程共用一份）。
MODELS_DIR_ENV_VAR = "V2C_MODELS_DIR"

#: 需要在日志/输出里打码的密钥模式
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.]{16,}"),
)


def mask_secret(value: Optional[str]) -> str:
    """把密钥变成可安全展示的形式：``sk-30c9…e8f9`` → ``sk-***e8f9``。"""
    if not value:
        return "（未配置）"
    if len(value) <= 8:
        return "***"
    return f"{value[:3]}***{value[-4:]}"


def redact(text: str) -> str:
    """把文本里出现的任何密钥替换成 ``***``（用于日志、报错、JSON）。"""
    if not text:
        return text
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("***", text)
    return text


def parse_env_text(text: str) -> dict[str, str]:
    """解析 .env 文本内容。"""
    values: dict[str, str] = {}
    text = text.lstrip("\ufeff")  # 容忍记事本/PowerShell 写入的 BOM
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        # 去掉行尾注释（仅当 # 前有空白时）
        if " #" in value:
            value = value.split(" #", 1)[0].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def find_env_file(start: Optional[Path] = None) -> Optional[Path]:
    """从 ``start``（默认当前目录）向上查找 ``.env``，找不到再退回工程根目录。"""
    explicit = os.environ.get(ENV_FILE_VAR)
    if explicit:
        candidate = Path(explicit).expanduser()
        return candidate if candidate.is_file() else None

    current = Path(start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate

    fallback = PROJECT_ROOT / ".env"
    return fallback if fallback.is_file() else None


def load_env_file(path: Optional[Path] = None, *, override: bool = False) -> dict[str, str]:
    """加载 .env 并写入 ``os.environ``（默认不覆盖已存在的环境变量）。"""
    env_path = path or find_env_file()
    if not env_path or not env_path.is_file():
        return {}
    try:
        values = parse_env_text(env_path.read_text(encoding="utf-8"))
    except OSError:
        return {}
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return values


def configured_engine(load_env: bool = True) -> str:
    """用户配置的默认识别引擎（``V2C_ENGINE`` 环境变量或 ``.env``）。

    返回**原始字符串**（未归一化），未配置时返回空串 —— 归一化与合法性校验在
    :func:`video2context.pipeline.resolve_engine` 里做，那边才认识引擎名。
    """
    if load_env:
        env_file = find_env_file()
        if env_file:
            load_env_file(env_file)
    return os.environ.get(ENGINE_ENV_VAR, "").strip()


def models_root(load_env: bool = True) -> Path:
    """模型根目录。

    默认是工程内的 ``models/``；设了 ``V2C_MODELS_DIR``（环境变量或 ``.env``）就用它。
    这样"代码放一处、权重放另一处"不用每次在命令行上传路径 —— 服务器上很常见：
    权重放数据盘、或者多个工程共用一份，省得重复下 6GB。
    """
    if load_env:
        env_file = find_env_file()
        if env_file:
            load_env_file(env_file)
    configured = os.environ.get(MODELS_DIR_ENV_VAR, "").strip()
    if configured:
        return Path(configured).expanduser()
    return PROJECT_ROOT / "models"


# --------------------------------------------------------- 解释器环境诊断


def project_python() -> Optional[Path]:
    """工程内虚拟环境（``.venv``）的解释器路径；没建过则返回 None。"""
    for relative in ("Scripts/python.exe", "bin/python", "bin/python3"):
        candidate = PROJECT_ROOT / ".venv" / relative
        if candidate.is_file():
            return candidate
    return None


def interpreter_hint(
    *,
    venv_python: Optional[Path] = None,
    current: Optional[Path] = None,
) -> str:
    """当前解释器**不是**工程的虚拟环境时，给出一段可照做的提示；否则返回空串。

    这是最容易踩的坑：工程依赖（自带 ffmpeg 的 imageio-ffmpeg、faster-whisper 等）
    都装在 ``.venv`` 里，但很多人习惯直接敲 ``python``，于是跑到了 conda base 或系统
    Python 上，然后看到「未找到 ffmpeg」这种和真实原因毫不相干的报错。

    ``venv_python`` / ``current`` 只是给测试注入用的。
    """
    import sys

    venv = venv_python if venv_python is not None else project_python()
    if venv is None:  # 没建过虚拟环境：谈不上"用错了"
        return ""
    running = current if current is not None else Path(sys.executable)
    try:
        if Path(running).resolve() == Path(venv).resolve():
            return ""
    except OSError:  # pragma: no cover - 路径异常时按"用错了"处理
        pass

    return (
        "\n⚠ 你现在用的不是工程的虚拟环境：\n"
        f"    当前 Python ：{running}\n"
        f"    工程环境    ：{venv}\n"
        "  依赖（含自带 ffmpeg 的 imageio-ffmpeg）都装在工程环境里。\n"
        "  改用工程环境跑（Windows），例如：\n"
        f"    {venv} -m video2context samples\\demo_zh_math.mp4\n"
        "  或先激活它，之后 python 就指向工程环境：\n"
        "    .\\.venv\\Scripts\\Activate.ps1\n"
    )


@dataclass
class LLMSettings:
    """调用大模型所需的配置。``api_key`` 为空表示未配置。"""

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = ""
    timeout: float = DEFAULT_TIMEOUT
    source: str = "未配置"  # 说明密钥来自哪里，便于排查
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @property
    def masked_key(self) -> str:
        return mask_secret(self.api_key)

    def describe(self) -> str:
        if not self.configured:
            return "未配置（设置 V2C_LLM_API_KEY 或写进 .env 后可用）"
        return f"已配置（{self.source}，{self.masked_key}，模型 {self.model}）"


def llm_settings(
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    load_env: bool = True,
) -> LLMSettings:
    """汇总 LLM 配置：显式参数 > 环境变量 > .env 文件。"""
    env_file = None
    if load_env:
        env_file = find_env_file()
        if env_file:
            load_env_file(env_file)

    def pick(name: str, default: str) -> str:
        return os.environ.get(name, "").strip() or default

    if env_file:
        source = f".env（{env_file}）"
    elif os.environ.get("V2C_LLM_API_KEY"):
        source = "环境变量"
    else:
        source = "未配置"

    return LLMSettings(
        base_url=(base_url or pick("V2C_LLM_BASE_URL", DEFAULT_BASE_URL)).rstrip("/"),
        model=model or pick("V2C_LLM_MODEL", DEFAULT_MODEL),
        api_key=pick("V2C_LLM_API_KEY", ""),
        timeout=timeout if timeout is not None else float(pick("V2C_LLM_TIMEOUT", str(DEFAULT_TIMEOUT))),
        source=source,
    )
