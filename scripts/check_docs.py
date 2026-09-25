"""校验 Markdown 文档：相对链接是否存在、目录锚点是否指向真实标题。"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    ROOT / "README.md",
    ROOT / "CHANGELOG.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "THIRD_PARTY_NOTICES.md",
    ROOT / "docs" / "architecture.md",
    ROOT / "docs" / "models.md",
    ROOT / "docs" / "troubleshooting.md",
    ROOT / "docs" / "development.md",
    ROOT / "samples" / "README.md",
]

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)


def slugify(text: str) -> str:
    """近似 GitHub 的标题锚点算法。"""
    text = text.strip().lower()
    out = []
    for ch in text:
        if ch.isalnum() or ch in "-_ " or unicodedata.category(ch).startswith("L"):
            out.append(ch)
        # 其余标点（. , : / （ ） 等）直接丢弃
    slug = "".join(out).replace(" ", "-")
    return slug


def anchors_of(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    seen: dict[str, int] = {}
    anchors = set()
    for _level, title in HEADING_RE.findall(text):
        title = re.sub(r"`", "", title)
        slug = slugify(title)
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(slug if count == 0 else f"{slug}-{count}")
    return anchors


def main() -> int:
    problems: list[str] = []
    anchor_cache: dict[Path, set[str]] = {}

    for doc in DOCS:
        if not doc.exists():
            problems.append(f"[缺文件] {doc.relative_to(ROOT)}")
            continue
        text = doc.read_text(encoding="utf-8")
        for target in LINK_RE.findall(text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path_part, _, anchor = target.partition("#")
            if path_part:
                resolved = (doc.parent / path_part).resolve()
                if not resolved.exists():
                    problems.append(f"[链接失效] {doc.relative_to(ROOT)} → {target}")
                    continue
            else:
                resolved = doc
            if anchor:
                if resolved not in anchor_cache:
                    anchor_cache[resolved] = anchors_of(resolved)
                if anchor not in anchor_cache[resolved]:
                    problems.append(f"[锚点失效] {doc.relative_to(ROOT)} → {target}")

    if problems:
        print(f"发现 {len(problems)} 个问题：")
        for item in problems:
            print("  " + item)
        return 1

    print(f"检查了 {len(DOCS)} 个文档：所有相对链接与锚点均有效。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
