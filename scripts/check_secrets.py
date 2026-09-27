"""提交前防泄露扫描：检查即将提交的内容里有没有密钥。

用法：
    python scripts/check_secrets.py              # 扫描待提交文件（git diff --cached）
    python scripts/check_secrets.py --all        # 扫描工作区所有受版本控制的文件
    python scripts/check_secrets.py 路径...       # 扫描指定文件

退出码 1 表示发现疑似密钥。CI 里会对整个仓库跑一遍。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("OpenAI/DeepSeek 风格密钥", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}")),
    ("GitHub Personal Access Token", re.compile(r"\bghp_[A-Za-z0-9]{20,}")),
    ("GitHub 细粒度令牌", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}")),
    ("AWS Access Key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Google API Key", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}")),
    ("Slack Token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("私钥文件头", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
]

#: 明显的占位符/示例，不算泄露
ALLOWLIST = re.compile(
    r"(sk-xxx|sk-你的|sk-abc123|在这里填|your[-_]?key|example|placeholder|xxx+)", re.IGNORECASE
)

TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".yml", ".yaml", ".json",
    ".ps1", ".sh", ".bat", ".env", ".example", ".gitignore", ".gitattributes",
}
SKIP_DIRS = {".git", ".venv", ".condaenv", "models", "output", ".tmp", "__pycache__"}


def git_files(*args: str) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []
    return [ROOT / line.strip() for line in result.stdout.splitlines() if line.strip()]


def iter_committable_files() -> list[Path]:
    """列出"有可能被提交"的文件：已跟踪 + 未跟踪但未被忽略。

    直接问 git 要这个列表，天然排除 .env / models / .venv 等被 gitignore 的内容，
    避免对自己明知道不会提交的文件误报。
    """
    tracked = git_files("ls-files", "--cached", "--others", "--exclude-standard")
    if tracked:
        return tracked
    # 没有 git 时的兜底：按目录名过滤
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


def scan_file(path: Path) -> list[tuple[str, int, str]]:
    """返回 [(规则名, 行号, 行内容)]。``.env`` 只检查是否被 git 跟踪，不当内容泄露。"""
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {".env", ".gitignore"}:
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    findings: list[tuple[str, int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if ALLOWLIST.search(line):
            continue
        for name, pattern in PATTERNS:
            if pattern.search(line):
                findings.append((name, lineno, line.strip()[:120]))
                break
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="扫描疑似密钥")
    parser.add_argument("paths", nargs="*", help="指定文件；留空则按模式选择")
    parser.add_argument("--all", action="store_true", help="扫描所有可能被提交的文件（而不是仅暂存区）")
    args = parser.parse_args(argv)

    if args.paths:
        files = [Path(p) for p in args.paths]
    elif args.all:
        files = iter_committable_files()
    else:
        files = git_files("diff", "--cached", "--name-only", "--diff-filter=ACM")
        if not files:
            files = git_files("ls-files")

    problems: list[str] = []
    for path in files:
        rel = path.relative_to(ROOT) if path.is_absolute() and ROOT in path.parents else path
        for name, lineno, line in scan_file(path):
            problems.append(f"  {rel}:{lineno}  [{name}]  {line}")

    if problems:
        print("⚠️  发现疑似密钥，请勿提交：")
        print("\n".join(problems))
        print("\n处理方式：把密钥移到 .env（已在 .gitignore 中），或使用环境变量注入。")
        return 1

    print(f"检查了 {len(files)} 个文件，未发现疑似密钥。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
