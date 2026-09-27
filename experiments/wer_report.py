"""英文识别质量对比：对标准答案算 WER，并列出逐词差异。

用法：
    python experiments/wer_report.py 标准答案.txt 候选1.txt 候选2.txt

WER = (替换 + 删除 + 插入) / 标准答案词数。
归一化规则：转小写、去掉标点（保留词内撇号与连字符）、按空白切词。
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

_PUNCT = re.compile(r"[^\w\s'\-]+", re.UNICODE)


def normalize(text: str) -> list[str]:
    """把文本归一化成词序列，便于比较。"""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("\u2019", "'")
    text = _PUNCT.sub(" ", text)
    return text.split()


def edit_ops(ref: list[str], hyp: list[str]) -> list[tuple[str, str, str]]:
    """Levenshtein 回溯，返回 (操作, 参考词, 候选词) 列表。"""
    n, m = len(ref), len(hyp)
    # dp[i][j] = ref[:i] 与 hyp[:j] 的编辑距离
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    ops: list[tuple[str, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] != hyp[j - 1]:
                ops.append(("替换", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("删除", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("插入", "", hyp[j - 1]))
            j -= 1
    return list(reversed(ops))


def score(ref_words: list[str], hyp_words: list[str]) -> tuple[float, list[tuple[str, str, str]]]:
    ops = edit_ops(ref_words, hyp_words)
    errors = sum(1 for op, _, _ in ops if op != "正确")
    return (errors / len(ref_words) if ref_words else 0.0), ops


def main() -> int:
    parser = argparse.ArgumentParser(description="英文 WER 对比")
    parser.add_argument("reference", help="标准答案文本文件")
    parser.add_argument("hypotheses", nargs="+", help="候选文本文件（可多个）")
    parser.add_argument("--show", type=int, default=30, help="每个候选最多列几个差异")
    args = parser.parse_args()

    ref_path = Path(args.reference)
    ref_words = normalize(ref_path.read_text(encoding="utf-8"))
    print(f"标准答案：{ref_path}  词数 = {len(ref_words)}\n")

    rows: list[tuple[str, float, int, int, int]] = []
    for hyp_name in args.hypotheses:
        hyp_path = Path(hyp_name)
        hyp_words = normalize(hyp_path.read_text(encoding="utf-8"))
        wer, ops = score(ref_words, hyp_words)
        subs = [o for o in ops if o[0] == "替换"]
        dels = [o for o in ops if o[0] == "删除"]
        ins = [o for o in ops if o[0] == "插入"]
        errors = len(subs) + len(dels) + len(ins)
        rows.append((hyp_path.stem, wer, errors, len(dels), len(ins)))
        print(f"=== {hyp_path.name} ===")
        print(f"WER = {wer * 100:.2f}%   错误 {errors} 处（替换 {len(subs)} / 删除 {len(dels)} / 插入 {len(ins)}）")
        shown = 0
        for op, r, h in ops:
            if op == "正确":
                continue
            if shown >= args.show:
                print(f"  … 其余 {errors - shown} 处省略")
                break
            if op == "替换":
                print(f"  [替换] {r}  →  {h}")
            elif op == "删除":
                print(f"  [删除] 漏掉 {r}")
            else:
                print(f"  [插入] 多出 {h}")
            shown += 1
        print()

    if len(rows) > 1:
        print("=== 汇总（按 WER 升序）===")
        print(f"{'候选':<28}{'WER':>9}{'错误':>7}{'漏词':>7}{'多词':>7}")
        for name, wer, errors, dels, ins in sorted(rows, key=lambda r: r[1]):
            print(f"{name:<28}{wer * 100:>8.2f}%{errors:>7}{dels:>7}{ins:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
