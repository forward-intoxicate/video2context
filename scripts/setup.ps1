# 一键部署 video2context（Windows PowerShell）
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1                     # 部署 Qwen3-ASR（默认，推荐）
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -SkipModels         # 只建环境，不下模型
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Engine whisper     # 只部署 Whisper（轻量，不需要 torch）
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Engine both        # 两个引擎都装
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Web                # 额外装网页界面
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Torch cpu          # 没有 N 卡时
#
# 默认会把模型一起下好（Qwen 约 6GB / Whisper 约 3GB），可断点续传、中断后重跑接着下。
# 为什么默认就下：模型反正第一次运行要用，提前下好可以走国内镜像（快得多）；
# 而且 Qwen 的**强制对齐模型**决定字幕时间精度 —— 少了它字幕时间会掉到最粗的一档
# （详见 docs/troubleshooting.md 7.4）。
#
# 引擎怎么选、各设备怎么部署，见 docs/deploy.md。
param(
    [ValidateSet("qwen", "whisper", "both")]
    [string]$Engine = "qwen",
    [ValidateSet("cu126", "cu124", "cu121", "cpu", "none")]
    [string]$Torch = "cu126",
    [switch]$SkipModels,
    [switch]$Web,
    [string]$Mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
Write-Host "工程目录：$root" -ForegroundColor Cyan

$wantWhisper = @("whisper", "both") -contains $Engine
$wantQwen = @("qwen", "both") -contains $Engine

# ---------------------------------------------------------------- 主环境
if (-not (Test-Path ".venv")) {
    Write-Host "[1/4] 创建主虚拟环境 .venv ..." -ForegroundColor Cyan
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "创建 .venv 失败。若是 Anaconda 的 ensurepip 被裁剪，改用：" -ForegroundColor Yellow
        Write-Host "  python -m venv --without-pip .venv"
        Write-Host "  python -m pip --python .\.venv\Scripts\python.exe install --upgrade pip"
        exit 1
    }
} else {
    Write-Host "[1/4] 已存在主环境 .venv，跳过创建" -ForegroundColor DarkGray
}

$py = Join-Path $root ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip

Write-Host "[2/4] 安装公共依赖（ffmpeg 等）..." -ForegroundColor Cyan
& $py -m pip install -i $Mirror --timeout 60 --retries 5 -r requirements-base.txt

if ($wantWhisper) {
    Write-Host "[3/4] 安装 Whisper 引擎（faster-whisper）..." -ForegroundColor Cyan
    & $py -m pip install -i $Mirror --timeout 60 --retries 5 -r requirements.txt

    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        Write-Host "      检测到 NVIDIA 显卡，安装 CUDA 运行库（约 1.5GB，仅首次）..." -ForegroundColor Cyan
        & $py -m pip install -r requirements-gpu-win.txt
    } else {
        Write-Host "      未检测到 NVIDIA 显卡，Whisper 将用 CPU 推理（慢但可用）" -ForegroundColor Yellow
    }
} else {
    Write-Host "[3/4] 跳过 Whisper 引擎（只部署 Qwen3-ASR，主环境保持无 torch）" -ForegroundColor DarkGray
}

if ($Web) {
    Write-Host "      安装网页界面依赖（gradio）..." -ForegroundColor Cyan
    & $py -m pip install -i $Mirror --timeout 60 --retries 5 -r requirements-web.txt
}

# ---------------------------------------------------------------- Qwen 独立环境
if ($wantQwen) {
    Write-Host "[4/4] 部署 Qwen3-ASR 独立环境 ..." -ForegroundColor Cyan
    $qwenArgs = @("-Torch", $Torch, "-Mirror", $Mirror)
    if ($SkipModels) { $qwenArgs += "-SkipModels" }
    # 模型下载交给 setup_qwen.ps1（识别模型 + 强制对齐模型，两者默认都下）
    & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "setup_qwen.ps1") @qwenArgs
} else {
    Write-Host "[4/4] 跳过 Qwen3-ASR" -ForegroundColor DarkGray
}

# ---------------------------------------------------------------- Whisper 模型
$whisperModel = Join-Path $root "models\faster-whisper-large-v3"
if ($wantWhisper -and -not $SkipModels -and -not (Test-Path (Join-Path $whisperModel "model.bin"))) {
    Write-Host "[i] 下载 Whisper large-v3（约 3GB，支持断点续传）..." -ForegroundColor Cyan
    & $py scripts\download_model.py large-v3 --source modelscope --out $whisperModel
}

# ---------------------------------------------------------------- 自检
Write-Host ""
Write-Host "环境自检：" -ForegroundColor Cyan
& $py -m video2context doctor

Write-Host ""
Write-Host "完成。用法：" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python -m video2context 我的视频.mp4"
Write-Host "  .\.venv\Scripts\python -m video2context webui"
if ($SkipModels) {
    Write-Host ""
    Write-Host "注意：这次跳过了模型下载。第一次运行会自动下，但走 HuggingFace、国内会慢。" -ForegroundColor Yellow
    Write-Host "建议现在补下（可断点续传）：" -ForegroundColor Yellow
    if ($wantQwen) {
        Write-Host "  .\.venv\Scripts\python scripts\download_model.py --repo Qwen/Qwen3-ASR-1.7B --source modelscope --out models\Qwen3-ASR-1.7B"
        Write-Host "  .\.venv\Scripts\python scripts\download_model.py --repo Qwen/Qwen3-ForcedAligner-0.6B --source modelscope --out models\Qwen3-ForcedAligner-0.6B"
    }
    if ($wantWhisper) {
        Write-Host "  .\.venv\Scripts\python scripts\download_model.py large-v3 --source modelscope --out models\faster-whisper-large-v3"
    }
}
