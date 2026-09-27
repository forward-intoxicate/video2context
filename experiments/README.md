# experiments · 引擎对比实验

这里放**不影响主工程**的探索性实验：试新模型、量数据、决定要不要正式集成。
主工程的依赖保持轻量（只有 faster-whisper + imageio-ffmpeg），
重依赖的实验各自独立环境。

## Qwen3-ASR 实验（`feat/qwen3-asr` 分支）

### 为什么单独一个环境

`qwen-asr` 把 `transformers==4.57.6`、`accelerate==1.12.0` 等版本**钉死**，还会拉 torch；
混进主工程 `.venv` 很可能把 faster-whisper 的依赖搞坏。所以：

* 主工程 `.venv`：保持 **torch-free**（这是它的优点：装包小、启动快）
* `.venv-qwen`：只给 Qwen3-ASR 用

### 准备

```powershell
# 1) 建独立环境并装 qwen-asr（含 torch）
powershell -ExecutionPolicy Bypass -File experiments\setup_qwen3_asr.ps1

# 2) torch 默认装到的是 CPU 版，必须换 CUDA 版（否则 1.7B 跑不动）
.\.venv-qwen\Scripts\python -m pip install `
    --index-url https://download.pytorch.org/whl/cu126 `
    "torch==2.14.0+cu126"

# 3) 从魔搭下模型（国内快；复用主工程的断点续传下载器）
.\.venv\Scripts\python scripts\download_model.py --repo Qwen/Qwen3-ASR-1.7B `
    --source modelscope --out models\Qwen3-ASR-1.7B
```

### 跑

```powershell
# 只转写（不带时间戳）
.\.venv-qwen\Scripts\python experiments\qwen3_asr_bench.py samples\formula.mp4 `
    --model models\Qwen3-ASR-1.7B --compare output\formula.json

# 带时间戳（还需 1.8GB 的强制对齐模型）
.\.venv-qwen\Scripts\python experiments\qwen3_asr_bench.py samples\formula.mp4 `
    --model models\Qwen3-ASR-1.7B --aligner models\Qwen3-ForcedAligner-0.6B
```

### 已知的坑（实测踩到）

| 坑 | 说明 |
|---|---|
| **PyPI 的 torch 是 CPU 版** | 默认装出来是 `2.14.0+cpu`，`cuda.is_available()` 为 False；必须从 `download.pytorch.org/whl/cu126` 装 |
| **内存要求高** | 1.7B 权重 4.7GB（bf16），加载时需要在内存里暂存分片；可用内存低于 5GB 有失败风险 |
| **Windows 没有 vLLM** | 官方推荐用 vLLM 后端，但 vLLM 基本只支持 Linux，Windows 只能用 transformers 后端 |
| **flash-attn 难装** | 官方建议装，但 Windows 要源码编译（需 MSVC + CUDA toolkit），可跳过 |
| **时间戳要额外模型** | 不带 `Qwen3-ForcedAligner-0.6B` 就只有整段文本，出不了 SRT/VTT |

### 实验结果

见本目录下的 `qwen3_asr_result.md`（实验完成后填写）。
