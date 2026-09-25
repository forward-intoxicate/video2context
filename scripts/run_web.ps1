# 启动本地网页界面
#   用法： powershell -ExecutionPolicy Bypass -File scripts\run_web.ps1 [-Port 7860]
param(
    [int]$Port = 7860,
    [string]$Device = "auto",
    [string]$Model = "large-v3",
    [string]$HfEndpoint = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

if ($HfEndpoint) { $env:HF_ENDPOINT = $HfEndpoint }

& $py -m video2context webui --port $Port --device $Device --model $Model
