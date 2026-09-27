# 为 Qwen3-ASR 准备独立环境 .venv-qwen（Windows PowerShell）
#
#   用法（在工程根目录）：
#     powershell -ExecutionPolicy Bypass -File scripts\setup_qwen.ps1
#     powershell -ExecutionPolicy Bypass -File scripts\setup_qwen.ps1 -Torch cpu
#     powershell -ExecutionPolicy Bypass -File scripts\setup_qwen.ps1 -DownloadModels
#
# 为什么要单独一个环境：Qwen3-ASR 需要 torch + transformers（装完约 3GB），
# 而主工程刻意保持轻量（faster-whisper 走 CTranslate2，不需要 torch）。
# 混进同一个环境容易把主工程的依赖搞坏，所以单开 .venv-qwen，由子进程桥接调用。
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

$venv = Join-Path $root ".venv-qwen"
$py = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "[1/4] 创建独立环境 .venv-qwen ..." -ForegroundColor Cyan
    # Anaconda 自带的 python -m venv 常常无法直接引导 pip（ensurepip 被裁剪），
    # 所以先建一个不带 pip 的空环境，再用主解释器的 pip 灌进去。
    python -m venv --without-pip $venv
    python -m pip --python $py install --upgrade pip
} else {
    Write-Host "[1/4] 已存在 .venv-qwen，跳过创建" -ForegroundColor DarkGray
}

& $py -m pip install --upgrade pip

if ($Torch -eq "none") {
    Write-Host "[2/4] 按参数要求跳过 torch 安装（请自行确保环境里已有 torch）" -ForegroundColor Yellow
} else {
    Write-Host "[2/4] 安装 torch（$Torch，约 2.5GB，耐心等）..." -ForegroundColor Cyan
    if ($Torch -eq "cpu") {
        # CPU 版：走 PyPI 默认 wheel 即可，但必须显式指定索引避免装到 CUDA 版
        & $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    } else {
        # 注意：直接 pip install torch 在 PyPI 上拿到的是 +cpu 构建（没有 CUDA），
        # 必须用 PyTorch 官方索引才能装上带 CUDA 的 wheel。
        & $py -m pip install torch --index-url "https://download.pytorch.org/whl/$Torch"
    }
}

Write-Host "[3/4] 安装 qwen-asr 及其依赖 ..." -ForegroundColor Cyan
& $py -m pip install -i $Mirror --timeout 60 --retries 5 -r requirements-qwen.txt

Write-Host "[4/4] 环境自检 ..." -ForegroundColor Cyan
# 这行刻意用纯 ASCII：子进程 python 的 stdout 走控制台代码页（中文 Windows 上是 GBK），
# 里面出现中文会显示成乱码；而 Write-Host 的中文有 BOM 保护，不受影响。
& $py -c "import torch, transformers, qwen_asr; print('  torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"

if ($DownloadModels) {
    Write-Host "[i] 下载模型（约 6GB，支持断点续传）..." -ForegroundColor Cyan
    $mainPy = Join-Path $root ".venv\Scripts\python.exe"
    if (-not (Test-Path $mainPy)) { $mainPy = "python" }
    & $mainPy scripts\download_model.py --repo Qwen/Qwen3-ASR-1.7B --source modelscope --out models\Qwen3-ASR-1.7B
    & $mainPy scripts\download_model.py --repo Qwen/Qwen3-ForcedAligner-0.6B --source modelscope --out models\Qwen3-ForcedAligner-0.6B
}

Write-Host ""
Write-Host "完成。用法：" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python -m video2context --engine qwen3-asr --qwen-setup"
if (-not $DownloadModels) {
    Write-Host ""
    Write-Host "还没下载模型的话，先跑（约 6GB）：" -ForegroundColor Yellow
    Write-Host "  .\.venv\Scripts\python scripts\download_model.py --repo Qwen/Qwen3-ASR-1.7B --source modelscope --out models\Qwen3-ASR-1.7B"
    Write-Host "  .\.venv\Scripts\python scripts\download_model.py --repo Qwen/Qwen3-ForcedAligner-0.6B --source modelscope --out models\Qwen3-ForcedAligner-0.6B"
}
