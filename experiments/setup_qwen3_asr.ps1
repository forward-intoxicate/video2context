# 为 Qwen3-ASR 准备独立环境（不污染主工程 .venv）
#   用法： powershell -ExecutionPolicy Bypass -File experiments\setup_qwen3_asr.ps1
#
# 为什么单独一个环境：qwen-asr 把 transformers / accelerate 等版本钉死了，
# 混进主工程容易把 faster-whisper 的依赖搞坏。
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$venv = Join-Path $root ".venv-qwen"
$py = Join-Path $venv "Scripts\python.exe"
$mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"

if (-not (Test-Path $py)) {
    Write-Host "[1/3] 创建独立环境 .venv-qwen ..." -ForegroundColor Cyan
    # Anaconda 的 python -m venv 常常自带的 ensurepip 不可用，先建空环境再灌 pip
    python -m venv --without-pip $venv
    python -m pip --python $py install --upgrade pip
} else {
    Write-Host "[1/3] 已存在 .venv-qwen" -ForegroundColor DarkGray
}

Write-Host "[2/3] 安装 qwen-asr（会拉 torch，约 2.5GB，耐心等）..." -ForegroundColor Cyan
& $py -m pip install -i $mirror --timeout 60 --retries 5 -U qwen-asr

Write-Host "[3/3] 校验 ..." -ForegroundColor Cyan
& $py -c "import torch, transformers, qwen_asr; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"

Write-Host ""
Write-Host "完成。下一步下载模型：" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python scripts\download_model.py --repo Qwen/Qwen3-ASR-1.7B --source modelscope --out models\Qwen3-ASR-1.7B"
