# 部署 Qwen3-ASR 识别引擎（Windows PowerShell）
#
# 单独跑这个脚本 = 完整部署「只用 Qwen3-ASR，不装 Whisper」的环境：
#   * 主环境 .venv        → 只装公共依赖（ffmpeg），保持无 torch
#   * 独立环境 .venv-qwen → torch + transformers + qwen-asr
#   * 两个模型            → 识别模型 + **强制对齐模型**（默认一起下，约 6GB）
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_qwen.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\setup_qwen.ps1 -Torch cpu
#   powershell -ExecutionPolicy Bypass -File scripts\setup_qwen.ps1 -SkipModels   # 先不下载模型
#
# 为什么单独一个环境：Qwen3-ASR 需要 torch + transformers（装完约 3GB），
# 而主工程刻意保持轻量（Whisper 引擎走 CTranslate2，不需要 torch）。
# 混进同一个环境容易把依赖搞坏，所以单开 .venv-qwen，由子进程桥接调用。
#
# 为什么要连对齐模型一起装：Qwen3-ASR 不输出时间戳，字幕时间是本工程造的。
# 没有对齐模型时只能按"说话区间"估算，而**只装 Qwen 的机器没有 faster-whisper，
# 也就没有 VAD**，会掉到最粗的一档（在整条音轨上平均分配，长静音视频字幕会明显错位）。
# 装上它就直接精确到词。详见 docs/troubleshooting.md 7.4。
param(
    [ValidateSet("", "cu128", "cu126", "cu124", "cu121", "cpu", "none")]
    [string]$Torch = "",
    [switch]$SkipModels,
    [string]$Mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
Write-Host "工程目录：$root" -ForegroundColor Cyan

# 显卡算力 → 该用哪个 torch 索引（与 setup_qwen.sh 里的 detect_torch_index 同一套逻辑）。
#
# 为什么不能写死：Blackwell（RTX 50 系 / RTX PRO 6000 / B 系列）算力是 12.0，
# 它的 kernel **只在 cu128 及以上的 wheel 里**。cu126 装上去能 import、
# torch.cuda.is_available() 也返回 True，但一跑算子就炸：
#   CUDA error: no kernel image is available for execution on the device
function Get-TorchIndex {
    $cc = ""
    $cuda = ""
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        try { $cc = (nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>$null | Select-Object -First 1) } catch { $cc = "" }
        try {
            $header = (nvidia-smi 2>$null | Out-String)
            if ($header -match "CUDA Version:\s*([0-9]+\.[0-9]+)") { $cuda = $Matches[1] }
        } catch { $cuda = "" }
    }
    $major = 0
    $parsed = $false
    if ($cc) { $parsed = [int]::TryParse((($cc -split "\.")[0]).Trim(), [ref]$major) }
    if ($parsed -and $major -ge 12) { return "cu128" }
    switch -Regex ($cuda) {
        "^12\.(6|7|8|9)$|^13\." { return "cu126" }
        "^12\.[45]$" { return "cu124" }
        "^12\.[0-3]$" { return "cu121" }
        default { return "cu126" }
    }
}

if (-not $Torch) {
    $Torch = Get-TorchIndex
    Write-Host "[i] 按显卡自动选择 torch 索引：$Torch" -ForegroundColor DarkGray
}

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
        $torchBase = if ($env:TORCH_BASE_URL) { $env:TORCH_BASE_URL } else { "https://download.pytorch.org/whl" }
        & $py -m pip install torch --index-url "$torchBase/$Torch"
    }
}

Write-Host "[4/5] 安装 qwen-asr 及其依赖 ..." -ForegroundColor Cyan
& $py -m pip install -i $Mirror --timeout 60 --retries 5 -r requirements-qwen.txt

Write-Host "[5/5] 环境自检 ..." -ForegroundColor Cyan
# 这行刻意用纯 ASCII：子进程 python 的 stdout 走控制台代码页（中文 Windows 上是 GBK），
# 里面出现中文会显示成乱码；而 Write-Host 的中文有 BOM 保护，不受影响。
& $py -c "import torch, transformers, qwen_asr; print('  torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"
# 只打印 is_available 是不够的：装错 arch 时它照样是 True，要真算一次才暴露。
# 这一步就是冲着 "no kernel image is available for execution on the device" 来的。
$hasCuda = $false
& $py -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -eq 0) { $hasCuda = $true }
if ($hasCuda) {
    & $py -c "import torch; a=torch.randn(8,8,device='cuda'); print('  GPU smoke test OK:', round((a@a).sum().item(),3))"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "" 
        Write-Host "  [x] GPU 能识别但算不了 —— 几乎肯定是 torch 的 CUDA 版本和显卡架构不匹配。" -ForegroundColor Red
        Write-Host "      当前 torch 编译进去的架构：" -ForegroundColor Red
        & $py -c "import torch; print('   ', torch.cuda.get_arch_list())"
        Write-Host "      按 docs/troubleshooting.md「no kernel image」一节换 wheel，再重跑本脚本。" -ForegroundColor Red
        exit 1
    }
}
& $mainPy -m video2context --engine qwen3-asr --qwen-setup

if ($SkipModels) {
    Write-Host "[i] 按 -SkipModels 跳过模型下载（第一次运行会自动下）" -ForegroundColor Yellow
} else {
    # 对齐模型不是可选项：没有它，字幕时间会掉到最粗的一档（见文件开头说明）。
    Write-Host "[i] 下载识别模型（约 4GB，支持断点续传，中断后重跑即可接着下）..." -ForegroundColor Cyan
    & $mainPy scripts\download_model.py --repo Qwen/Qwen3-ASR-1.7B --source modelscope --out "$modelsDir\Qwen3-ASR-1.7B"
    Write-Host "[i] 下载强制对齐模型（约 1.8GB，字幕精确到词就靠它）..." -ForegroundColor Cyan
    & $mainPy scripts\download_model.py --repo Qwen/Qwen3-ForcedAligner-0.6B --source modelscope --out "$modelsDir\Qwen3-ForcedAligner-0.6B"
}

Write-Host ""
Write-Host "完成。用法：" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python -m video2context 我的视频.mp4"
if ($SkipModels) {
    Write-Host ""
    Write-Host "记得补下模型（约 6GB；第一次运行也会自动下，但走 HuggingFace、国内会慢）：" -ForegroundColor Yellow
    Write-Host "  .\.venv\Scripts\python scripts\download_model.py --repo Qwen/Qwen3-ASR-1.7B --source modelscope --out $modelsDir\Qwen3-ASR-1.7B"
    Write-Host "  .\.venv\Scripts\python scripts\download_model.py --repo Qwen/Qwen3-ForcedAligner-0.6B --source modelscope --out $modelsDir\Qwen3-ForcedAligner-0.6B"
}
