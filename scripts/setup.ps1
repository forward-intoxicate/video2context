# 一键准备运行环境（Windows PowerShell）
#   用法： powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
Write-Host "工程目录：$root" -ForegroundColor Cyan

if (-not (Test-Path ".venv")) {
    Write-Host "[1/4] 创建虚拟环境 .venv ..." -ForegroundColor Cyan
    python -m venv .venv
} else {
    Write-Host "[1/4] 已存在 .venv，跳过创建" -ForegroundColor DarkGray
}

$py = Join-Path $root ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip

Write-Host "[2/4] 安装核心依赖 ..." -ForegroundColor Cyan
& $py -m pip install -r requirements.txt

Write-Host "[3/4] 安装网页界面依赖（可选）..." -ForegroundColor Cyan
try {
    & $py -m pip install -r requirements-web.txt
} catch {
    Write-Host "  [跳过] 网页依赖安装失败，命令行功能不受影响" -ForegroundColor Yellow
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "[i] 检测到 NVIDIA 显卡，安装 CUDA 运行库（约 1.5GB，仅首次）..." -ForegroundColor Cyan
    & $py -m pip install -r requirements-gpu-win.txt
} else {
    Write-Host "[i] 未检测到 NVIDIA 显卡，将使用 CPU 推理（速度较慢但可用）" -ForegroundColor Yellow
}

Write-Host "[4/4] 环境自检 ..." -ForegroundColor Cyan
& $py -m video2context doctor

Write-Host ""
Write-Host "完成。用法示例：" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python -m video2context 我的视频.mp4"
Write-Host "  .\.venv\Scripts\python -m video2context webui"
