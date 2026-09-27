# experiments · 引擎对比实验

这里放**不影响主工程**的探索性实验：试新模型、量数据、决定要不要正式集成。
主工程的依赖保持轻量（只有 faster-whisper + imageio-ffmpeg），
重依赖的实验各自独立环境。

> **Qwen3-ASR 已经从实验转正**：主工程现在支持 `--engine qwen3-asr`
> （见 [README](../README.md#识别引擎怎么选) 与 [docs/models.md](../docs/models.md#6-qwen3-asr默认引擎)），
> 一键安装脚本是 `scripts\setup_qwen.ps1`。
> 本目录保留的是**当时的实验脚本与原始数据**，用于复现结论；
> 日常使用不需要到这里来。

## Qwen3-ASR 实验（`feat/qwen3-asr` 分支）

### 为什么单独一个环境

`qwen-asr` 把 `transformers==4.57.6`、`accelerate==1.12.0` 等版本**钉死**，还会拉 torch；
混进主工程 `.venv` 很可能把 faster-whisper 的依赖搞坏。所以：

* 主工程 `.venv`：保持 **torch-free**（这是它的优点：装包小、启动快）
* `.venv-qwen`：只给 Qwen3-ASR 用

### 准备

```powershell
# 建独立环境并装好依赖（含 CUDA 版 torch，脚本会自动处理版本）
powershell -ExecutionPolicy Bypass -File scripts\setup_qwen.ps1

# 从魔搭下模型（国内快；复用主工程的断点续传下载器）
.\.venv\Scripts\python scripts\download_model.py --repo Qwen/Qwen3-ASR-1.7B `
    --source modelscope --out models\Qwen3-ASR-1.7B
.\.venv\Scripts\python scripts\download_model.py --repo Qwen/Qwen3-ForcedAligner-0.6B `
    --source modelscope --out models\Qwen3-ForcedAligner-0.6B
```

### 跑

```powershell
# 只转写（不带时间戳）
.\.venv-qwen\Scripts\python experiments\qwen3_asr_bench.py samples\formula.mp4 `
    --model models\Qwen3-ASR-1.7B --compare output\formula.json

# 带时间戳
.\.venv-qwen\Scripts\python experiments\qwen3_asr_bench.py samples\formula.mp4 `
    --model models\Qwen3-ASR-1.7B --aligner models\Qwen3-ForcedAligner-0.6B

# 英文 WER 对比（标准答案 vs 各引擎输出）
.\.venv\Scripts\python experiments\wer_report.py 标准答案.txt 候选1.txt 候选2.txt
```

### 已知的坑（实测踩到）

| 坑 | 说明 |
|---|---|
| **PyPI 的 torch 是 CPU 版** | 默认装出来是 `2.14.0+cpu`，`cuda.is_available()` 为 False；必须从 `download.pytorch.org/whl/cu126` 装 |
| **内存要求高** | 1.7B 的第一个分片有 4GB，加载时整个读进内存；提交内存低于 ~13GB 会报 `页面文件太小`。用 `--qwen-low-mem on` 或主工程的 `--qwen-low-mem on` 绕开 |
| **worker 目录会遮住同名包** | 用脚本路径启动时 `sys.path[0]` 是脚本目录，工程内的模块会遮住 site-packages 同名包。`_qwen_worker.py` 已显式摘除自身目录 |
| **Windows 没有 vLLM** | 官方推荐用 vLLM 后端，但 vLLM 基本只支持 Linux，Windows 只能用 transformers 后端 |
| **flash-attn 难装** | 官方建议装，但 Windows 要源码编译（需 MSVC + CUDA toolkit），可跳过 |
| **时间戳要额外模型** | 不带 `Qwen3-ForcedAligner-0.6B` 就只有整段文本，出不了精确的 SRT/VTT |
| **`context` 无效** | 实测传什么都对结果没影响（含完全无关的内容）。见 [docs/models.md](../docs/models.md#64-实测context词表偏置没有效果) |

### 实验结果

* 中文数学课 73s：`u(x)` / `v(x)` / `导数` 全对且标点完整（large-v3 为右/位/倒 ×13 处、无标点）
* 英文 42s（101 词）：WER **0.00%**（large-v3 为 2.97%）
* 速度：1.7B + 对齐，73s 中文实测 12.5× 实时（推理 5.8s + 加载 6.6s）

原始数据与完整对照见 `qwen3_asr_result.md`。
