# 部署 Qwen3-ASR 识别引擎（Windows PowerShell）
#
# 单独跑这个脚本 = 完整部署「只用 Qwen3-ASR，不装 Whisper」的环境：
#   * 主环境 .venv        → 只装公共依赖（ffmpeg），保持无 torch
#   * 独立环境 .venv-qwen → torch + transformers + qwen-asr
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_qwen.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\setup_qwen.ps1 -Torch cpu
#   powershell -ExecutionPolicy Bypass -File scripts\setup_qwen.ps1 -DownloadModels
#
# 为什么要单独一个环境：Qwen3-ASR 需要 torch + transformers（装完约 3GB），
# 而主工程刻意保持轻量（Whisper 引擎走 CTranslate2，不需要 torch）。
# 混进同一个环境容易把依赖搞坏，所以单开 .venv-qwen，由子进程桥接调用。
param(
    [ValidateSet("cu126", "cu124", "cu121", "cpu", "none")]
    [string]$Torch = "cu126",
    [switch]$DownloadModels,
    [string]$Mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
Write-Host "工程目录：$root" -ForegroundColor Cyan

# ------------------------------------------------- 主环境（只装公共依赖，不含 Whisper）
$mainPy = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $mainPy)) {
    Write-Host "[1/5] 创建主环境 .venv（只装公共依赖，不含 Whisper）..." -ForegroundColor Cyan
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "创建 .venv 失败。若是 Anaconda 的 ensurepip 被裁剪，改用：" -ForegroundColor Yellow
        Write-Host "  python -m venv --without-pip .venv"
        Write-Host "  python -m pip --python .\.venv\Scripts\python.exe install --upgrade pip"
        exit 1
    }
    python -m pip --python $mainPy install --upgrade pip
} else {
    Write-Host "[1/5] 已存在主环境 .venv" -ForegroundColor DarkGray
}
& $mainPy -m pip install -i $Mirror --timeout 60 --retries 5 -r requirements-base.txt

# ---------------------------------------------------------------- Qwen 独立环境
$venv = Join-Path $root ".venv-qwen"
$py = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "[2/5] 创建独立环境 .venv-qwen ..." -ForegroundColor Cyan
    # Anaconda 自带的 python -m venv 常常无法直接引导 pip（ensurepip 被裁剪），
    # 所以先建一个不带 pip 的空环境，再用主解释器的 pip 灌进去。
    python -m venv --without-pip $venv
    python -m pip --python $py install --upgrade pip
} else {
    Write-Host "[2/5] 已存在 .venv-qwen，跳过创建" -ForegroundColor DarkGray
}

& $py -m pip install --upgrade pip

if ($Torch -eq "none") {
    Write-Host "[3/5] 按参数要求跳过 torch 安装（请自行确保环境里已有 torch）" -ForegroundColor Yellow
} else {
    Write-Host "[3/5] 安装 torch（$Torch，约 2.5GB，耐心等）..." -ForegroundColor Cyan
    if ($Torch -eq "cpu") {
        # CPU 版：走 PyTorch 官方 CPU 索引，避免误装 CUDA 版
        & $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    } else {
        # 注意：直接 pip install torch 在 PyPI 上拿到的是 +cpu 构建（没有 CUDA），
        # 必须用 PyTorch 官方索引才能装上带 CUDA 的 wheel。
        & $py -m pip install torch --index-url "https://download.pytorch.org/whl/$Torch"
    }
}

Write-Host "[4/5] 安装 qwen-asr 及其依赖 ..." -ForegroundColor Cyan
& $py -m pip install -i $Mirror --timeout 60 --retries 5 -r requirements-qwen.txt

Write-Host "[5/5] 环境自检 ..." -ForegroundColor Cyan
# 这行刻意用纯 ASCII：子进程 python 的 stdout 走控制台代码页（中文 Windows 上是 GBK），
# 里面出现中文会显示成乱码；而 Write-Host 的中文有 BOM 保护，不受影响。
& $py -c "import torch, transformers, qwen_asr; print('  torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"
& $mainPy -m video2context --engine qwen3-asr --qwen-setup

if ($DownloadModels) {
    Write-Host "[i] 下载模型（约 6GB，支持断点续传）..." -ForegroundColor Cyan
    # download_model.py 只用标准库，主环境哪怕只装了 imageio-ffmpeg 也能跑
    & $mainPy scripts\download_model.py --repo Qwen/Qwen3-ASR-1.7B --source modelscope --out models\Qwen3-ASR-1.7B
    & $mainPy scripts\download_model.py --repo Qwen/Qwen3-ForcedAligner-0.6B --source modelscope --out models\Qwen3-ForcedAligner-0.6B
}

Write-Host ""
Write-Host "完成。用法：" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python -m video2context 我的视频.mp4"
if (-not $DownloadModels) {
    Write-Host ""
    Write-Host "还没下载模型的话（约 6GB；第一次运行也会自动下）：" -ForegroundColor Yellow
    Write-Host "  .\.venv\Scripts\python scripts\download_model.py --repo Qwen/Qwen3-ASR-1.7B --source modelscope --out models\Qwen3-ASR-1.7B"
    Write-Host "  .\.venv\Scripts\python scripts\download_model.py --repo Qwen/Qwen3-ForcedAligner-0.6B --source modelscope --out models\Qwen3-ForcedAligner-0.6B"
}
