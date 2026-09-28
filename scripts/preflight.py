"""部署前体检：看看这台机器够不够跑，以及该用哪条安装命令（**只看不装**）。

为什么单独写一个（而不是用 ``doctor``）：

* ``doctor`` 是**装完之后**的自检 —— 需要依赖已就位才能跑；
* 这个脚本只用标准库，**在装任何东西之前**就能跑，专门用来回答三个问题：
  1. 这台机器（尤其是云 GPU 容器）够不够？
  2. 该用哪个 CUDA 版本的 torch？
  3. 要不要新建环境、工程该放哪个盘？

用法：

    python scripts/preflight.py            # 体检 + 给出建议命令
    python scripts/preflight.py --json     # 输出 JSON（给自动化用）

在服务器上可以先克隆仓库、用它自带的 python 直接跑（不需要先装依赖）：

    git clone <repo> && cd video2context
    python3 scripts/preflight.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import shutil
import subprocess
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 各条路线大致的磁盘占用（GB）：环境 + 模型 + 中间文件余量
DISK_NEED = {"qwen": 12, "whisper": 6, "both": 16}

#: Qwen3-ASR-1.7B 加载时需要的提交内存（GB）——实测约 13GB（分片要先读进 CPU 内存）
COMMIT_MEMORY_GB = 14
#: Qwen 的显存需求（GB）
QWEN_VRAM_GB = 5
#: faster-whisper large-v3 的显存需求（GB），int8 时可以低到 2
WHISPER_VRAM_GB = 5

#: PyTorch 官方索引里我们支持的几个（从新到旧）
TORCH_INDEXES = ("cu126", "cu124", "cu121")


@dataclass
class Report:
    """体检结果。``ok`` / ``warn`` / ``bad`` 三档，用于最后汇总。"""

    system: str = ""
    machine: str = ""
    python: str = ""
    python_ok: bool = False
    cpu_count: int = 0
    memory_total_gb: float = 0.0
    memory_available_gb: float = 0.0
    gpu_names: list[str] = field(default_factory=list)
    gpu_vram_gb: float = 0.0
    cuda_version: Optional[str] = None
    recommended_torch: Optional[str] = None
    disks: list[dict[str, Any]] = field(default_factory=list)
    project_dir: str = ""
    project_free_gb: float = 0.0
    home_free_gb: float = 0.0
    has_git: bool = False
    notes: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------- 采集


def _run(args: list[str], timeout: float = 15.0) -> str:
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace"
        )
        return (proc.stdout or "").strip() if proc.returncode == 0 else ""
    except Exception:
        return ""


def _memory_gb() -> tuple[float, float]:
    """返回 (总内存, 可用内存)，单位 GB；取不到就返回 (0, 0)。"""
    # Linux：/proc/meminfo 最准，而且 MemAvailable 已经考虑了缓存
    try:
        info: dict[str, float] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            number = rest.strip().split()[0]
            info[key.strip()] = float(number) / 1024 / 1024  # kB → GB
        return info.get("MemTotal", 0.0), info.get("MemAvailable", info.get("MemFree", 0.0))
    except Exception:
        pass
    # Windows
    try:
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _Status()
        status.dwLength = ctypes.sizeof(_Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return status.ullTotalPhys / 2**30, status.ullAvailPhys / 2**30
    except Exception:
        pass
    # 其它 Unix
    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
        available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
        return float(total), float(available)
    except Exception:
        return 0.0, 0.0


def _gpu_info() -> tuple[list[str], float, Optional[str]]:
    """用 nvidia-smi 取 (显卡名列表, 总显存GB, 驱动支持的 CUDA 版本)。"""
    if not shutil.which("nvidia-smi"):
        return [], 0.0, None

    names: list[str] = []
    vram = 0.0
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        names.append(parts[0])
        if len(parts) > 1:
            with contextlib.suppress(ValueError):
                vram += float(parts[1]) / 1024  # MiB → GB

    # nvidia-smi 抬头里的 "CUDA Version: 12.2" 才是驱动支持的版本
    header = _run(["nvidia-smi"])
    cuda = None
    for line in header.splitlines():
        if "CUDA Version" in line:
            tail = line.split("CUDA Version", 1)[1]
            cuda = tail.strip().lstrip(":").strip().split()[0].strip()
            break
    return names, vram, cuda


def _recommend_torch_index(cuda: Optional[str]) -> Optional[str]:
    """按驱动支持的 CUDA 版本，挑一个 PyTorch 官方索引。

    CUDA 向后兼容：驱动支持 12.4 就能跑 cu121 的 wheel，反之不行。
    所以取"不高于驱动上限"里最新的那个。
    """
    if not cuda:
        return None
    try:
        supported = tuple(int(x) for x in cuda.split(".")[:2])
    except ValueError:
        return None
    for index in TORCH_INDEXES:
        need = (int(index[2:4]), int(index[4:6]))
        if supported >= need:
            return index
    return None


def _disk_of(path: Path) -> Optional[dict[str, Any]]:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return None
    return {
        "path": str(path),
        "total_gb": round(usage.total / 2**30, 1),
        "free_gb": round(usage.free / 2**30, 1),
    }


def collect() -> Report:
    report = Report()
    report.system = f"{platform.system()} {platform.release()}"
    report.machine = platform.machine()
    report.python = f"{sys.version.split()[0]} ({sys.executable})"
    report.python_ok = sys.version_info >= (3, 9)
    # 容器里 os.cpu_count() 可能返回 None（文档允许），别让它把后续算炸
    report.cpu_count = os.cpu_count() or 0
    report.memory_total_gb, report.memory_available_gb = _memory_gb()
    report.gpu_names, report.gpu_vram_gb, report.cuda_version = _gpu_info()
    report.recommended_torch = _recommend_torch_index(report.cuda_version)
    report.has_git = bool(shutil.which("git"))
    report.project_dir = str(PROJECT_ROOT)

    # 磁盘：工程目录 / 用户目录 / 常见的数据盘挂载点（AutoDL 等云容器）
    seen: set[str] = set()
    for path in (PROJECT_ROOT, Path.home(), Path("/root/autodl-tmp"), Path("/data"), Path("/mnt")):
        if not path.exists():
            continue
        info = _disk_of(path)
        if info and info["path"] not in seen:
            seen.add(info["path"])
            report.disks.append(info)

    project_disk = _disk_of(PROJECT_ROOT)
    report.project_free_gb = project_disk["free_gb"] if project_disk else 0.0
    home_disk = _disk_of(Path.home())
    report.home_free_gb = home_disk["free_gb"] if home_disk else 0.0

    if report.gpu_names:
        report.notes.append(f"GPU 显存 {report.gpu_vram_gb:.0f}GB，CUDA 驱动上限 {report.cuda_version or '未知'}")
    else:
        report.notes.append("没检测到 NVIDIA 显卡 → 走 Whisper 路线（CPU 也能跑）；Qwen 在 CPU 上比实时还慢")

    return report


# --------------------------------------------------------------------- 判断


def evaluate(report: Report, engine: str) -> tuple[list[str], list[str]]:
    """按选定路线判断能不能跑，返回 (致命问题, 提醒)。"""
    problems: list[str] = []
    warnings: list[str] = []

    if not report.python_ok:
        problems.append(f"Python 版本太低（{report.python}），需要 3.9+")
    if not report.has_git:
        warnings.append("没找到 git —— 如果是用压缩包传的代码，这条可以忽略")

    need_disk = DISK_NEED.get(engine, DISK_NEED["qwen"])
    if report.project_free_gb and report.project_free_gb < need_disk:
        problems.append(
            f"工程所在磁盘只剩 {report.project_free_gb:.1f}GB，装 {engine} 路线大约要 {need_disk}GB。"
            f"云容器常见做法是把工程放到数据盘（如 /root/autodl-tmp）再装。"
        )
    if report.home_free_gb and report.home_free_gb < 8:
        warnings.append(
            f"用户目录只剩 {report.home_free_gb:.1f}GB —— 模型缓存默认写这里（~/.cache/huggingface），"
            "建议设 HF_HOME 到数据盘"
        )

    if engine in {"qwen", "both"}:
        if not report.gpu_names:
            warnings.append("没检测到 NVIDIA 显卡：Qwen 在 CPU 上只有 0.5× 实时（比实时还慢），建议改走 Whisper 路线")
        elif report.gpu_vram_gb < QWEN_VRAM_GB:
            problems.append(f"显存只有 {report.gpu_vram_gb:.0f}GB，Qwen3-ASR-1.7B 大约要 {QWEN_VRAM_GB}GB")
        if report.recommended_torch is None and report.gpu_names:
            warnings.append("没识别出驱动支持的 CUDA 版本，安装时请手动指定 -Torch / TORCH_INDEX")

    if report.memory_total_gb and report.memory_total_gb < COMMIT_MEMORY_GB:
        if engine in {"qwen", "both"}:
            problems.append(
                f"内存只有 {report.memory_total_gb:.0f}GB —— Qwen 加载模型要约 {COMMIT_MEMORY_GB}GB 提交内存，"
                "需要加 --qwen-low-mem on"
            )
        else:
            warnings.append(f"内存 {report.memory_total_gb:.0f}GB，Whisper 够用（约 8GB 起）")

    if engine == "whisper" and report.gpu_names and report.recommended_torch is None:
        warnings.append("Whisper 引擎不需要 torch，但需要 CUDA 运行库（Linux 上是系统级 cuBLAS/cuDNN）")

    return problems, warnings


# --------------------------------------------------------------------- 输出


def _align(label: str, width: int = 13) -> str:
    """按显示宽度补齐（中文一个字占两格），让冒号对齐。"""
    shown = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in label)
    return label + " " * max(width - shown, 1)


def print_report(report: Report, engine: str, problems: list[str], warnings: list[str]) -> None:
    def row(label: str, value: str) -> None:
        print(f"  {_align(label)}: {value}")

    print("video2context 部署前体检（只看不装）\n")
    row("系统", f"{report.system}  {report.machine}")
    row("Python", report.python)
    row("CPU", f"{report.cpu_count} 核" if report.cpu_count else "读不到（不影响安装）")
    if report.memory_total_gb:
        row("内存", f"总 {report.memory_total_gb:.0f}GB / 可用 {report.memory_available_gb:.0f}GB")
    else:
        row("内存", "读不到（不影响安装）")

    if report.gpu_names:
        row("GPU", f"{'、'.join(report.gpu_names)}  显存共 {report.gpu_vram_gb:.0f}GB")
        row("CUDA", f"驱动上限 {report.cuda_version or '未知'} → 推荐 torch 索引 {report.recommended_torch or '（需手动指定）'}")
    else:
        row("GPU", "未检测到 NVIDIA 显卡")

    print("\n  磁盘：")
    for disk in report.disks:
        mark = "  ← 工程在这里" if disk["path"] == report.project_dir else ""
        print(f"    {disk['path']:<28} 可用 {disk['free_gb']:>7.1f}GB / 共 {disk['total_gb']:.0f}GB{mark}")

    print("\n  结论：")
    if problems:
        for item in problems:
            print(f"    ✗ {item}")
    for item in warnings:
        print(f"    ! {item}")
    if not problems and not warnings:
        print("    ✓ 满足要求，可以直接按下面装")
    elif not problems:
        print("    ✓ 可以安装（上面的提醒看一眼即可）")

    print("\n  建议的安装命令（Linux / macOS）：")
    for line in suggested_commands(report, engine):
        print(f"    {line}")


def suggested_commands(report: Report, engine: str) -> list[str]:
    """给出这台机器上可以直接照抄的安装命令（**Linux / macOS 的 bash 写法**）。

    注意别把 Windows 那套参数混进来：``-Torch cu126`` 是 setup.ps1 的 PowerShell 参数，
    bash 版走的是环境变量 ``TORCH_INDEX``。
    """
    torch = report.recommended_torch or "cu121"
    lines: list[str] = []

    needs_data_disk = bool(report.project_free_gb and report.project_free_gb < DISK_NEED.get(engine, 12))
    if needs_data_disk:
        lines += [
            "# 工程所在磁盘空间不够，先挪到数据盘（挂载点按实际情况改）",
            "mkdir -p /root/autodl-tmp && cd /root/autodl-tmp",
            "",
        ]

    lines += [
        "# 1) 克隆（已有代码就跳过）",
        "git clone https://github.com/forward-intoxicate/video2context.git",
        "cd video2context",
        "",
        "# 2) 让 HuggingFace 缓存别写进系统盘（云容器上系统盘很小，这步很重要）",
        "export HF_HOME=/root/autodl-tmp/hf-cache",
        "",
    ]

    if engine == "whisper":
        lines += [
            "# 3) 一键装（Whisper 路线，不需要 torch，约 4GB）",
            "bash scripts/setup.sh whisper",
        ]
    else:
        suffix = " both" if engine == "both" else ""
        lines += [
            f"# 3) 一键装（约 9GB：环境 3GB + 模型 6GB；torch 用 {torch}）",
            f"TORCH_INDEX={torch} bash scripts/setup.sh{suffix}",
        ]

    lines += [
        "",
        "# 4) 确认环境（两个引擎都装了就看默认是哪个）",
        "./.venv/bin/python -m video2context doctor",
        "",
        "# 5) 跑仓库自带的样例",
        "./.venv/bin/python -m video2context samples/demo_zh_math.mp4 --language zh -f json,srt",
    ]
    return lines


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="部署前体检：这台机器够不够跑 video2context")
    parser.add_argument(
        "--engine",
        choices=["qwen", "whisper", "both"],
        default="qwen",
        help="打算走的路线（默认 qwen，即默认路线）",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON，便于自动化")
    args = parser.parse_args(argv)

    report = collect()
    problems, warnings = evaluate(report, args.engine)

    if args.json:
        payload = asdict(report)
        payload["engine"] = args.engine
        payload["problems"] = problems
        payload["warnings"] = warnings
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_report(report, args.engine, problems, warnings)

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
