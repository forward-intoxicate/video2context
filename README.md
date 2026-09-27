# video2context · 视频转文字

> 把视频丢进去，本地抽出音轨、识别成文字，输出**带时间戳的 JSON / SRT / VTT / TXT**。
> 默认用 **Qwen3-ASR**（中文同音词、数学符号、英文都更准），也可以换成轻量的 faster-whisper。
> 全程在本机运行、不需要任何 API Key。

![CI](https://github.com/forward-intoxicate/video2context/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)

```
视频.mp4 ──ffmpeg──▶ 16kHz 单声道 wav ──▶ 识别引擎 ──▶ JSON / SRT / VTT / TXT
                                            ├─ qwen3-asr（默认，效果最好，需要 torch + 显卡）
                                            └─ faster-whisper（轻量，无需 torch，CPU 也能跑）
```

**两个识别引擎**（`--engine` 切换，输出格式完全一致，**可以只装其中一个**）：

| 引擎 | 依赖 | 中文同音词 | 英文 WER | 适合 |
|---|---|---|---|---|
| `qwen3-asr`（默认） | 独立环境 + torch，显存 ≥ 6GB | **好**（样例上 0 处同音词错误） | **0.00%** | 中文课程/讲座、术语多的内容 |
| `faster-whisper` | 轻量，**不需要 torch**，CPU 就能跑 | 一般（`u(x)`→「右F4」） | 4.35% | 没有显卡、想省 3GB、要翻译成英文 |

程序会**按你实际装了的引擎来定默认值**：只装了 Whisper 就用 Whisper，只装了 Qwen 就用 Qwen，
两个都装则优先 Qwen。所以按任意一条路线部署完，直接敲命令就能跑。

（质量数据来自实测，详见 [docs/models.md](docs/models.md#6-qwen3-asr默认引擎)。）

|  | 说明 |
|---|---|
| **输入** | 视频（mp4/mkv/mov/avi/flv…）或纯音频（mp3/wav/m4a/flac…） |
| **语言** | 中文、英文自动识别；也可强制指定（更快、更准） |
| **输出** | JSON（分段+时间戳+语种+参数）、SRT、VTT、TXT |
| **硬件** | NVIDIA 显卡（Qwen 需要；Whisper 可选，纯 CPU 也能跑） |
| **联网** | 仅首次下载模型需要；之后可完全离线 |

---

## 目录

- [命令怎么敲（先看这个）](#命令怎么敲先看这个)
- [快速开始：从 0 到跑通](#快速开始从-0-到跑通)
- [选部署路线（换引擎 / 没显卡 / macOS）](#选部署路线换引擎--没显卡--macos)
- [这个工程适合谁](#这个工程适合谁)
- [安装详解](#安装详解)
- [怎么用](#怎么用)
- [识别引擎怎么选](#识别引擎怎么选)
- [领域词表（修同音词/专有名词）](#领域词表修同音词专有名词)
- [输出文件说明](#输出文件说明)
- [模型怎么选](#模型怎么选)
- [实测性能](#实测性能)
- [工作原理](#工作原理)
- [常见问题](#常见问题)
- [项目结构](#项目结构)
- [开发 / 测试](#开发--测试)
- [文档索引](#文档索引)
- [许可证](#许可证)

---

## 命令怎么敲（先看这个）

工程依赖装在专用的虚拟环境 `.venv` 里。**命令有两种写法，任选一种，效果完全一样**：

| | Windows（PowerShell） | macOS / Linux |
|---|---|---|
| **写法 A**<br>直接用 venv 里的 python（不用激活，推荐） | `.\.venv\Scripts\python -m video2context 视频.mp4` | `./.venv/bin/python -m video2context 视频.mp4` |
| **写法 B**<br>先激活，之后 `python` 就指向它 | `.\.venv\Scripts\Activate.ps1`<br>然后 `python -m video2context 视频.mp4` | `source .venv/bin/activate`<br>然后 `python -m video2context 视频.mp4` |

激活成功后提示符会从 `(base)` 之类变成 **`(.venv)`**；退出用 `deactivate`。

> ⚠️ **直接敲 `python`（既没激活、也没写 venv 路径）= 跑到了系统或 conda 的 Python 上**，
> 会报「未找到 ffmpeg」——因为依赖（含自带 ffmpeg 的 imageio-ffmpeg）都在 `.venv` 里。
> 这是本工程最常见的坑，详见 [docs/troubleshooting.md 1.5](docs/troubleshooting.md#15-敲-python-报未找到-ffmpeg缺少-faster-whisper-多半是跑错了-python)。

**本文档后面为了简短，示例统一写成 `python -m video2context ...`。**
看到这种命令时，请按上面的写法 A 或写法 B 替换 —— 例如：

```powershell
# 文档里写的是：      python -m video2context samples\demo_zh_math.mp4
# Windows 实际要敲：  .\.venv\Scripts\python -m video2context samples\demo_zh_math.mp4
```

> 先跑一次 `python -m video2context doctor`（同样按上面的写法）能确认环境是否就绪；
> 如果你用错了 Python，它会直接把这件事指出来。

---

## 快速开始：从 0 到跑通

> 下面每一步都给了 **Windows** 与 **Linux / macOS** 两版命令，照着顺序敲即可。
> 全程只需要 **Python 3.9+** 和 **git** —— ffmpeg 不用单独装（依赖里自带静态版本）。

### 第 0 步 · 确认前置条件

| 检查 | Windows（PowerShell） | Linux / macOS |
|---|---|---|
| Python | `python --version` | `python3 --version` |
| git | `git --version` | `git --version` |

需要 Python ≥ 3.9（**3.11 / 3.12 / 3.13 最稳**；网页界面需要 3.10+）。
Linux 上如果 `python3 -m venv` 报 `ensurepip is not available`，先 `sudo apt install python3-venv`。

### 第 1 步 · 克隆仓库

```bash
git clone https://github.com/forward-intoxicate/video2context.git
cd video2context
```

两个平台命令相同。

### 第 2 步 · 一键装环境 + 下模型

```powershell
# Windows（默认部署 Qwen3-ASR，并把模型一起下好）
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```
```bash
# Linux / macOS
bash scripts/setup.sh
```

这个脚本会自动做完 5 件事：

1. 建主环境 `.venv`（只装公共依赖：自带 ffmpeg 的 `imageio-ffmpeg`）；
2. 建独立环境 `.venv-qwen`（torch + transformers + qwen-asr，约 3GB）；
3. 装对应 CUDA 版本的 torch（PyPI 上是 CPU 版，必须走 PyTorch 官方索引）；
4. 从魔搭镜像下载**两个**模型：识别模型 4GB + 强制对齐模型 1.8GB（可断点续传）；
5. 跑一次自检并打印结果。

**总体积约 9GB，时间基本都花在下载上**（国内走镜像一般十几分钟）。

**如果你的机器不一样**：

```powershell
# Windows：没有 N 卡 / 想省 3GB torch → 改走 Whisper 路线（约 4GB，纯 CPU 也能跑）
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Engine whisper

# Windows：只想先建环境，模型稍后再下
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -SkipModels
```
```bash
# Linux / macOS 同理
bash scripts/setup.sh whisper            # 只装 Whisper
SKIP_MODELS=1 bash scripts/setup.sh      # 只建环境，不下模型
TORCH_INDEX=cpu bash scripts/setup.sh    # macOS / 无 N 卡：装 CPU 版 torch
```

### 第 3 步 ·（可选）单独补下模型

第 2 步已经下好了，**正常情况下跳过这一步**。
只有在用了 `-SkipModels`、或者下载中断想补下时才需要 —— 下载器可反复执行，每次接着上次的进度继续：

```powershell
# Windows
.\.venv\Scripts\python scripts\download_model.py --repo Qwen/Qwen3-ASR-1.7B --source modelscope --out models\Qwen3-ASR-1.7B
.\.venv\Scripts\python scripts\download_model.py --repo Qwen/Qwen3-ForcedAligner-0.6B --source modelscope --out models\Qwen3-ForcedAligner-0.6B
```
```bash
# Linux / macOS
./.venv/bin/python scripts/download_model.py --repo Qwen/Qwen3-ASR-1.7B --source modelscope --out models/Qwen3-ASR-1.7B
./.venv/bin/python scripts/download_model.py --repo Qwen/Qwen3-ForcedAligner-0.6B --source modelscope --out models/Qwen3-ForcedAligner-0.6B
```

> Whisper 路线换成：`scripts\download_model.py large-v3 --source modelscope --out models\faster-whisper-large-v3`

### 第 4 步 · 确认环境就绪

```powershell
# Windows
.\.venv\Scripts\python -m video2context doctor
```
```bash
# Linux / macOS
./.venv/bin/python -m video2context doctor
```

关键几行应该长这样：

```
  ── 识别引擎 ──────────────────────────────
  qwen3-asr            : 就绪
  faster-whisper       : 未部署 → pip install -r requirements.txt
  Qwen 解释器          : E:\...\video2context\.venv-qwen\Scripts\python.exe
  Qwen 识别模型        : E:\...\video2context\models\Qwen3-ASR-1.7B
  Qwen 对齐模型        : E:\...\models\Qwen3-ForcedAligner-0.6B（字幕时间精确到词）
  实际默认引擎         : qwen3-asr（本机只部署了 qwen3-asr；可用 --engine 覆盖…）
  CUDA 设备数          : 1
```

**判断标准**：引擎显示「就绪」、两个模型路径指向 `models\` 下的**本地目录**（不是仓库名）、
`CUDA 设备数` ≥ 1。哪一项不对就见 [docs/troubleshooting.md](docs/troubleshooting.md)。

### 第 5 步 · 用仓库自带的参考视频试跑

不用先找自己的素材 —— 仓库里带了示例视频：

```powershell
# Windows
.\.venv\Scripts\python -m video2context samples\demo_zh_math.mp4 --language zh -f json,txt,srt,vtt
```
```bash
# Linux / macOS
./.venv/bin/python -m video2context samples/demo_zh_math.mp4 --language zh -f json,txt,srt,vtt
```

42 秒的中文数学课样例。期望看到（首次运行会含模型加载的几秒）：

```
  [1/3] 探测媒体信息：demo_zh_math.mp4
        时长 00:00:42（41.8s）｜格式 mov,mp4,m4a,3gp,3g2,mj2｜视频轨 有
  [2/3] 提取音频 → 16000Hz 单声道 wav
        音频就绪：audio.wav（1.3 MB）
  提示：--engine qwen3-asr 下未指定 --model，自动选用 Qwen 默认模型
  [3/3] 语音识别（引擎 qwen3-asr，模型 ...\models\Qwen3-ASR-1.7B，任务 transcribe）
        强制对齐模型：...\models\Qwen3-ForcedAligner-0.6B（字幕时间精确到词）
        按时间戳切出 7 段（强制对齐，时间精确到词）
  语言=Chinese 分段=7 音频时长=00:00:42 处理耗时=5.5s 实时率=0.1326
  -> JSON output\demo_zh_math.json
  -> TXT  output\demo_zh_math.txt
  -> SRT  output\demo_zh_math.srt
  -> VTT  output\demo_zh_math.vtt
```

**看到最后那 4 行 `-> ...` 就说明整条链路通了。**
耗时随机器不同（GPU 上约 5～9 秒；CPU 上会明显更慢），
关键是 **`分段=7`** —— 它正好对应样例稿子的 7 句，说明切句和时间戳都对上了。

仓库里另外还有 3 个样例（英文技术内容、中文日常口语、6 分半真人朗读），
清单和更多示例命令见 **[samples/README.md](samples/README.md)**。

想对比两个引擎的英文准确率（需要路线 C，两个引擎都装了）：

```powershell
# Windows
.\.venv\Scripts\python -m video2context samples\demo_en_ml.mp4 --language en -f json,txt -n en-qwen
.\.venv\Scripts\python -m video2context samples\demo_en_ml.mp4 --language en -f json,txt -n en-whisper `
    --engine faster-whisper --model-dir models\faster-whisper-large-v3
.\.venv\Scripts\python experiments\wer_report.py samples\demo_en_ml.transcript.txt output\en-qwen.txt output\en-whisper.txt
```
```bash
# Linux / macOS
./.venv/bin/python -m video2context samples/demo_en_ml.mp4 --language en -f json,txt -n en-qwen
./.venv/bin/python -m video2context samples/demo_en_ml.mp4 --language en -f json,txt -n en-whisper \
    --engine faster-whisper --model-dir models/faster-whisper-large-v3
./.venv/bin/python experiments/wer_report.py samples/demo_en_ml.transcript.txt output/en-qwen.txt output/en-whisper.txt
```

实测 WER：Qwen3-ASR **0.00%**，faster-whisper large-v3 **4.35%**。

### 第 6 步 · 对一下答案

```powershell
# Windows
type output\demo_zh_math.txt
```
```bash
# Linux / macOS
cat output/demo_zh_math.txt
```

再打开 `samples/demo_zh_math.transcript.txt`（逐句稿子）对照。
识别结果应该和稿子**几乎逐字一致**，差异只会在标点上（逗号取舍、`？` 写成 `。`），
**不该有错字**——原因见 [samples/README.md](samples/README.md#识别结果和稿子对不上先看差在哪一类)。

### 第 7 步 · 换成自己的视频

```powershell
# Windows
.\.venv\Scripts\python -m video2context "D:\videos\我的视频.mp4" --language zh -f json,srt
```
```bash
# Linux / macOS
./.venv/bin/python -m video2context ~/videos/my_video.mp4 --language zh -f json,srt
```

结果落在 `output/`（`我的视频.json` / `.srt`）。想用网页界面：

```powershell
# Windows
.\.venv\Scripts\python -m video2context webui
```
```bash
# Linux / macOS
./.venv/bin/python -m video2context webui
```

浏览器打开 <http://127.0.0.1:7860>，把视频拖进去即可。

### 卡在哪一步了？

| 现象 | 原因 / 去哪看 |
|---|---|
| `未找到 ffmpeg` | 用错了 Python（没激活 `.venv`）→ [命令怎么敲](#命令怎么敲先看这个)、[troubleshooting 1.5](docs/troubleshooting.md#15-敲-python-报未找到-ffmpeg缺少-faster-whisper-多半是跑错了-python) |
| `找不到 Qwen3-ASR 的独立环境（.venv-qwen）` | 第 2 步没跑或失败了 → 重跑安装脚本、[troubleshooting 7.1](docs/troubleshooting.md#71-找不到-qwen3-asr-的独立环境venv-qwen) |
| `qwen3-asr : 未部署` | 同上 |
| `页面文件太小` / `CUDA out of memory` | 内存/显存不够 → [troubleshooting 7.2](docs/troubleshooting.md#72-页面文件太小无法完成操作--cuda-out-of-memory--0xc0000005) |
| 模型下载卡住 / 很慢 | 用自带下载器走魔搭 → [troubleshooting 3.1](docs/troubleshooting.md#31-下载卡住进度条不动) |
| 字幕时间明显不对 | 看 JSON 里的 `asr.timestamp_source` → [troubleshooting 7.4](docs/troubleshooting.md#74-字幕时间戳不准--所有段落挤在一起) |
| 其它 | [docs/troubleshooting.md](docs/troubleshooting.md) 全文 |

---

## 选部署路线（换引擎 / 没显卡 / macOS）

上面走的是**默认路线**（Qwen3-ASR）。想换引擎或设备不同，看这张表；
完整的手把手步骤在 **[docs/deploy.md](docs/deploy.md)**
（含无显卡、纯 CPU、macOS、离线内网、国内加速、磁盘占用与卸载）：

| 路线 | 适合 | 磁盘 | Windows | macOS / Linux |
|---|---|---|---|---|
| **A. 只装 Qwen3-ASR**（默认） | 有 N 卡，要最好的中文效果 | ~9GB | `scripts\setup.ps1` | `bash scripts/setup.sh` |
| **B. 只装 Whisper** | 没显卡 / 想省 3GB torch | ~4GB | `scripts\setup.ps1 -Engine whisper` | `bash scripts/setup.sh whisper` |
| **C. 两个都装** | 想 A/B 对比 | ~13GB | `scripts\setup.ps1 -Engine both` | `bash scripts/setup.sh both` |

**模型默认一起下好**（走魔搭镜像，可断点续传、中断后重跑接着下）；
只想先建环境就加 `-SkipModels`（Windows）/ `SKIP_MODELS=1`（macOS、Linux）。

走 Qwen 路线时脚本会连**强制对齐模型**（1.8GB）一起装 —— 它不是可选项：
Qwen 不输出时间戳，缺了它字幕时间会掉到最粗的一档（详见 [docs/deploy.md](docs/deploy.md#为什么对齐模型是默认装的不是可选项)）。

**硬件门槛**：

| | 只装 Qwen | 只装 Whisper |
|---|---|---|
| 显卡 | **NVIDIA，显存 ≥ 6GB**（推荐 8GB） | 可选；没有就用 CPU |
| 内存 | **16GB 起**（加载模型要约 13GB 提交内存） | 8GB 够 |
| 速度 | GPU 约 12× 实时；**CPU 只有 0.5×（比实时慢）** | GPU 20～24×；CPU（small）约 6× |

> **不需要单独安装 ffmpeg**，依赖里自带的静态版本开箱即用。
> Qwen 是**默认引擎**，但要先跑安装脚本准备独立环境；没准备的话程序会明确告诉你去跑哪条命令。

---

## 这个工程适合谁

**适合**：想把会议录像、课程视频、采访、播客、Vlog 转成文字或字幕；
想给视频做内容检索；不想把视频传到第三方服务；网络环境一般（国内）也想顺利装上。

**不合适**：需要实时（边说边出字）——本工程是文件级批处理；
需要专业级说话人分离/精确标点（可自行接入，见[扩展方向](CONTRIBUTING.md)）。

**不用懂 AI**：装好之后就是一条命令的事。默认参数已经调到"中文长视频要精度"的档位。

---

## 安装详解

> 完整版（各设备、无显卡、macOS、离线内网、国内加速、卸载清理）见 **[docs/deploy.md](docs/deploy.md)**。
> 下面是速查。

依赖文件按"公共 / 引擎 / 附加"分层，互不冲突：

| 文件 | 内容 | 谁需要 |
|---|---|---|
| `requirements-base.txt` | `imageio-ffmpeg`（抽取音轨） | **所有人**（装哪个引擎都要） |
| `requirements.txt` | base + `faster-whisper` | 装 Whisper 引擎的人 |
| `requirements-qwen.txt` | `qwen-asr` + torch 等 | 装 Qwen 引擎的人（装在**独立**的 `.venv-qwen`） |
| `requirements-web.txt` | + Gradio 网页界面 | 想要图形界面的人 |
| `requirements-gpu-win.txt` | + cuBLAS / cuDNN 运行库 | Windows + N 卡 + **Whisper 引擎** |

### 只装 Whisper（轻量、不需要 torch）

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -r requirements.txt          # base + faster-whisper
.\.venv\Scripts\python -m pip install -r requirements-gpu-win.txt  # Windows + N 卡才需要
.\.venv\Scripts\python -m video2context doctor
```

**没有显卡（纯 CPU）**也走这条路，用更小的模型和 int8 精度：

```powershell
.\.venv\Scripts\python -m video2context 视频.mp4 --model small --device cpu --compute-type int8
```

**Windows + N 卡**：`requirements-gpu-win.txt` 做两件事 —— 装 cuBLAS 12 + cuDNN 9 的 DLL，
并被本工程**自动注册进 DLL 搜索路径**（这是 Windows 上 faster-whisper 报
`cudnn64_9.dll not found` 的根因）。已装 CUDA Toolkit 12.x + cuDNN 9 的机器可以跳过。

**Linux**：ctranslate2 需要系统级 CUDA 12 与 cuDNN 9：

```bash
# Ubuntu 示例（或改用 conda install -c conda-forge cudnn cuda-toolkit）
sudo apt install libcublas-12-* libcudnn9-cuda-12
```

### 只装 Qwen3-ASR（默认路线，不含 Whisper）

Qwen3-ASR 需要 `torch` + `transformers`（约 3GB），而主环境刻意保持 torch-free
（Whisper 走 CTranslate2，不需要 torch）。所以它装在**另一个虚拟环境 `.venv-qwen`** 里，
由子进程桥接调用 —— 详见 [docs/architecture.md](docs/architecture.md)「Qwen3-ASR 为什么走子进程」。

```powershell
# Windows：一次把主环境（只装 ffmpeg）+ .venv-qwen 都建好，模型也一起下
powershell -ExecutionPolicy Bypass -File scripts\setup_qwen.ps1

# 自检：解释器 / 模型 / 对齐模型是否就绪
.\.venv\Scripts\python -m video2context --engine qwen3-asr --qwen-setup
```

```bash
# macOS / Linux
bash scripts/setup_qwen.sh
```

> **对齐模型默认就装**（1.8GB）。它是字幕时间精度的分界线：少了它，只装 Qwen 的机器
> 会掉到"在整条音轨上平均分配"那一档。真要省这 1.8GB 用 `--qwen-aligner off`。
> 只想先建环境、不下模型：加 `-SkipModels`（Windows）或 `SKIP_MODELS=1`。
>
> 只想换掉 torch 的 CUDA 版本：`-Torch cu124` / `-Torch cpu`；
> 已经有别的环境装了 `qwen-asr`：设 `V2C_QWEN_PYTHON=<解释器路径>`，不用新建。

### 国内网络加速

```powershell
# pip 走清华源（安装脚本已默认带上）
.\.venv\Scripts\python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 模型走魔搭（比 HuggingFace 快很多，支持断点续传）
.\.venv\Scripts\python scripts\download_model.py large-v3 --source modelscope
```

模型也可以单独下（用主环境跑下载器即可）：

```powershell
.\.venv\Scripts\python scripts\download_model.py --repo Qwen/Qwen3-ASR-1.7B --source modelscope --out models\Qwen3-ASR-1.7B
.\.venv\Scripts\python scripts\download_model.py --repo Qwen/Qwen3-ForcedAligner-0.6B --source modelscope --out models\Qwen3-ForcedAligner-0.6B
```

> `Qwen3-ForcedAligner-0.6B`（1.8GB）只负责**给你精确的字幕时间戳**。
> 不装也能用（`--qwen-aligner off`），但 SRT 的时间只能按说话区间估算。

### 手动下载模型（网络不稳时的正解）

`huggingface_hub` 默认下载在弱网下会长时间挂起。本工程自带一个**分块 + 断点续传**下载器，
支持 HuggingFace 与魔搭（ModelScope，国内通常快 5 倍以上）：

```powershell
# 下到本地目录（可反复执行，每次都接着上次进度继续）
.\.venv\Scripts\python scripts\download_model.py large-v3 --source modelscope

# 之后完全离线使用
.\.venv\Scripts\python -m video2context 视频.mp4 --model-dir models\faster-whisper-large-v3
```

---

## 怎么用

### 命令行

> 下面的示例写成 `python -m video2context ...` 是为了简短。
> **实际要按[「命令怎么敲」](#命令怎么敲先看这个)替换**：
> Windows 用 `.\.venv\Scripts\python -m video2context ...`，
> macOS / Linux 用 `./.venv/bin/python -m video2context ...`；
> 或者先激活虚拟环境（Windows `.\.venv\Scripts\Activate.ps1`，macOS/Linux `source .venv/bin/activate`）之后照抄即可。

```powershell
# 最简（transcribe 是默认命令，可省略）
python -m video2context 视频.mp4

# 英文视频：指定语种更快更准，同时出字幕和文本
python -m video2context talk.mp4 --language en -f json,srt,txt

# 批量：一个目录下所有 mp4（模型只加载一次）
python -m video2context .\videos\*.mp4 -o .\output

# 长视频先试跑前 10 分钟
python -m video2context long.mp4 --duration 600

# 逐句实时打印（像字幕一样滚出来）
python -m video2context a.mp4 --stream

# 专有名词多的视频：喂提示词，人名地名准确率明显提升（Whisper 引擎）
python -m video2context 讲座.mp4 --initial-prompt "本次讲座涉及：卷积神经网络、反向传播、吴恩达。"

# 只想要字幕
python -m video2context a.mp4 -f srt

# 只装了 Whisper 想手动指定引擎，或想切回 Whisper
python -m video2context a.mp4 --engine faster-whisper

# 没显卡 / 显存不够（Whisper 引擎）
python -m video2context a.mp4 --engine faster-whisper --device cpu --model small --compute-type int8

# 环境自检：看本机装了什么引擎、默认用哪个
python -m video2context doctor

# Qwen 那一侧单独自检
python -m video2context --engine qwen3-asr --qwen-setup
```

### 默认引擎是怎么定的

优先级 **`--engine` > 环境变量/`.env` 的 `V2C_ENGINE` > 本机装了的引擎（Qwen 优先）**。
第三档让你不用记自己的部署路线：装了哪个就用哪个。想固定成某个引擎就写 `.env`：

```ini
V2C_ENGINE=faster-whisper
```

运行时会打印实际用的引擎，结果 JSON 里也有 `asr.engine` 字段可以核对。

### 完整参数表

| 参数 | 说明 | 默认 |
|---|---|---|
| `-o, --output-dir` | 输出目录 | `output` |
| `-n, --name` | 输出文件名（不含扩展名，仅单输入可用） | 取输入文件名 |
| `-f, --formats` | 输出格式：`json,txt,srt,vtt` | `json` |
| `--overwrite` | 允许覆盖同名结果（默认自动改名 `xxx-1.json`） | 关 |
| `--keep-audio` | 保留抽出的 16kHz wav | 关 |
| `--stream` | 逐句实时打印 | 关 |
| `--print-text` | 结束时把全文打到标准输出 | 关 |
| `-q, --quiet` | 只输出结果路径 | 关 |
| `--engine` | 识别引擎：`faster-whisper` / `qwen3-asr`（不指定时按 `V2C_ENGINE` → 本机部署情况自动定） | 按部署情况，优先 `qwen3-asr` |
| `-m, --model` | 模型名或本地模型目录（两个引擎通用） | 按引擎自动 |
| `-l, --language` | `auto` / `zh` / `en` / `ja` … | `auto` |
| `--task` | `transcribe` 保持原语言 / `translate` 翻成英文 | `transcribe` |
| `--device` | `auto` / `cuda` / `cpu` | `auto` |
| `--compute-type` | `float16` / `int8_float16` / `int8` / `float32` | 自动 |
| `--beam-size` | 束搜索宽度，越大越准越慢 | `5` |
| `--batch-size` | `-1` 自动（GPU 用 8）、`0` 逐段 | `-1` |
| `--no-vad` | 关闭静音过滤（默认开启，长静音视频很省时间） | 关 |
| `--initial-prompt` | 提示词，喂专有名词/术语 | 无 |
| `--temperature` | 采样温度或回退序列，如 `0,0.2,0.4` | Whisper 默认回退 |
| `--condition-on-previous-text` | 以上文为条件（更连贯，长音频易重复） | 关 |
| `--word-timestamps` | 词级时间戳（JSON 里多 `words` 字段） | 关 |
| `--start` / `--duration` | 只处理某一段（秒） | 全程 |
| `--glossary` | 手写词表文件（`错形 -> 正确` 或纯术语） | 无 |
| `--auto-glossary` | 自动两遍解码：粗转写 → 大模型推断词表 → 正式转写 | 关 |
| `--dry-run-glossary` | 只生成词表并打印，不做正式转写 | 关 |
| `--scan-duration` | 第一遍采样秒数 | `90` |
| `--scan-model` | 第一遍用的模型（默认与正式相同） | 同正式 |
| `--glossary-out` | 把词表写到文件（便于复用与复核） | 无 |
| `--no-verify-glossary` | 不校验偏置是否生效（默认校验，失效则回退） | 关 |
| `--llm-base-url` / `--llm-model` | 大模型服务地址 / 模型名 | 读环境变量 |
| `--model-dir` | 本地模型目录（配合下载器，完全离线） | 无 |
| `--local-files-only` | 只用本地缓存，不联网 | 关 |
| `--hf-endpoint` / `--hf-mirror` | 模型下载源 / 用 hf-mirror | 官方源 |
| `--cpu-threads` | CPU 线程数，0=自动 | `0` |
| `--qwen-aligner` | 强制对齐模型目录，决定字幕时间戳精度（`off` 关闭） | 自动探测 |
| `--qwen-python` | `.venv-qwen` 解释器路径 | 自动探测 |
| `--qwen-low-mem` | `auto`/`on`/`off`：低内存加载（8GB 内存机器用 `on`） | `auto` |
| `--qwen-max-new-tokens` | 单段最多生成多少 token | `4096` |
| `--qwen-batch-size` | Qwen 内部并行段数（显存紧张调小） | `8` |
| `--qwen-setup` | 只做 Qwen 环境自检，不转写 | 关 |

### 网页界面

```powershell
python -m video2context webui                          # http://127.0.0.1:7860，自动开浏览器
python -m video2context webui --port 8000 --no-browser  # 换端口 / 不自动打开
python -m video2context webui --model-dir models\faster-whisper-large-v3   # 用本地模型
python -m video2context webui --share                  # 生成公网临时链接
```

页面左侧上传视频、选**识别引擎**/模型/语种/设备，右侧实时进度 + 全文 + 分段表格 + JSON 下载。
引擎下拉框的默认选中项同样由部署情况决定（Qwen 优先）。
默认只监听 `127.0.0.1`（仅本机可访问）；局域网访问加 `--host 0.0.0.0`。

```powershell
# 强制让界面默认选中 Whisper（或者用 --engine qwen3-asr 反过来）
python -m video2context webui --engine faster-whisper
```

> 界面里选 Qwen3-ASR 但没装独立环境时，会直接弹出可照做的报错，不会静默失败。
> Qwen 引擎下「翻译成英文」不可用，界面会提前拦住并提示改用 faster-whisper。

---

## 识别引擎怎么选

同一个流水线，`--engine` 换识别器，**输出格式、命令行参数、SRT/VTT 全部一致**，所以可以随时 A/B 对比。

| | `qwen3-asr`（默认） | `faster-whisper` |
|---|---|---|
| 模型 | Qwen3-ASR-1.7B（Apache-2.0） | Whisper large-v3 的 CTranslate2 版（MIT） |
| 依赖 | 独立环境 `.venv-qwen`（torch + transformers，约 3GB） | `requirements.txt`，**不需要 torch** |
| 磁盘 | 4GB（+1.8GB 对齐模型） | 3.1GB |
| 显存 | ~4GB（加载需约 13GB **提交内存**） | ~4GB（int8 可低到 2GB） |
| CPU | 0.5× 实时（比实时慢，不推荐） | 可用（`small` 约 6 倍实时） |
| 中文标点 | 更完整 | 模型自带，尚可 |
| 中文同音词 | **样例上 0 处** | `u(x)`→「右F4」这类错误多 |
| 英文 WER | **0.00%** | 4.35% |
| 翻译成英文 | 不支持（只做原语言转写） | 支持（`--task translate`） |

**建议**：

- **有 N 卡就走默认的 `qwen3-asr`**（`scripts\setup.ps1` 默认装的就是它），效果明显更好。
- **没有显卡、或者要翻译成英文** → 用 `faster-whisper` 那条路线。
- 两个引擎都装的话可以跑一遍对比：它们读同一份 16kHz wav，结果 JSON 里的
  `asr.engine` 字段能直接区分。

```powershell
# 同一条视频，两种引擎各跑一遍（输出名不同，方便 diff）
python -m video2context 课程.mp4 -f json,srt -n 课程-qwen
python -m video2context 课程.mp4 --engine faster-whisper -f json,srt -n 课程-whisper
```

> 引擎的默认值**由部署情况决定**：只装了 Whisper 就用 Whisper，只装了 Qwen 就用 Qwen，
> 两个都装则优先 Qwen。所以走路线 B（只装 Whisper）的用户不用每次都加 `--engine`。
> 想强制固定，见上一节「默认引擎是怎么定的」。

### Qwen 引擎的两个实测细节

**1. 字幕时间戳来自「强制对齐模型」**

Qwen3-ASR 本身只吐**一整段文字**，没有分段。本工程的做法是：

- 装了 `Qwen3-ForcedAligner-0.6B` → 拿到**词级时间戳**，再把 token 映射回原文下标，
  按标点/停顿切句 → SRT 时间精确到词（实测切出的分段能 100% 还原原文，标点不丢）；
- 没装对齐模型 → 按标点切句，再用 faster-whisper 自带的 Silero VAD 找出说话区间，
  按字数比例把句子分配到区间里。**能用，但时间只是估算**，JSON 里
  `asr.timestamp_source` 会写成 `vad-proportional`；
- 连说话区间也拿不到（**只装 Qwen 的机器没有 faster-whisper，也就没有 VAD**）→
  退化成"在整条音轨上平均分配"，这时标签是 `even-spread`（最粗的一档），
  日志会说明原因。所以走 Qwen-only 路线时，**建议把对齐模型一起装上**。

三档质量在 JSON 里用 `asr.timestamp_source` 如实标明：
`forced-aligner`（精确到词）/ `vad-proportional`（估算）/ `even-spread`（最粗）/ `none`。

**2. `context`（词表偏置）实测无效**

Qwen3-ASR 支持把一段文本作为 `context` 注入 system message，官方定位是"上下文偏置"。
但我在本机实测：**换成完全无关的内容（"这段音频讲的是量子力学，公式是 E=mc²"），
输出与不传 context 逐字相同**；中文样例上传词表也一字未变。

我核对过链路 —— 渲染出的 prompt 里 system 消息是正确写入的，所以不是本工程的接线问题，
是这一版模型/实现确实没吃这个信息。因此：

- `--glossary` / `--auto-glossary` 在 Qwen 引擎下**仍会照常生成词表并写进 JSON**（供人工复核、复用）；
- 但程序会明确提示"不要指望它修同音词"；
- 想修同音词，请在 faster-whisper 引擎上用词表（见下一节），或者做后处理改写。

> 完整实验记录与命令见 [docs/models.md](docs/models.md)。

---

## 领域词表（修同音词/专有名词）

中文识别最难的错误是**同音词**：`u(x)` 被听成「右F4」、`导数` 被听成「倒数」、`再求导` 被听成「在求导」。
喂一段词表就能修掉，实测在 `samples/formula.mp4` 上**「右/位」错误从 13 处降到 0 处**。

### 方式一：手写词表（完全离线、零额外开销）

```powershell
# terms.txt 内容：
#   右F4 -> u(x)       「错形 -> 正确」表示还原
#   位F4 -> v(x)
#   可导
python -m video2context 视频.mp4 --glossary terms.txt
```

### 方式二：自动两遍解码

先配好大模型密钥（**只把一小段转写文本发给大模型，音频不出本机**）：

```powershell
copy .env.example .env          # macOS/Linux: cp .env.example .env
# 编辑 .env，填入 V2C_LLM_API_KEY=sk-xxx
python -m video2context doctor  # 应显示「大模型（词表推断）: 已配置（.env，sk-***abcd）」
```

```powershell
# 推荐：先生成词表看一眼，确认后再正式跑（第二步完全离线、不再调用大模型）
python -m video2context 视频.mp4 --dry-run-glossary --glossary-out terms.json
python -m video2context 视频.mp4 --glossary terms.json

# 或者一条命令跑完
python -m video2context 视频.mp4 --auto-glossary
```

额外开销：采样 90 秒做第一遍（约 3~5 秒 GPU）+ 一次大模型调用（约 0.0003 元）。

### ⚠️ 为什么内置「验证 + 回退」

Whisper 的提示词偏置会**静默失效**：实测同样包含正确符号 `u(x)、v(x)、x0` 的 prompt，
**措辞一变就可能完全无效**（且确定性复现，跑 3 次结果一字不差）。
我试过用"填充词""长度""截断""裸字母"四个假设解释，**全部被实验否定**。

所以程序会在带词表转写后统计「期望符号命中次数」，命中 0 就自动回退到无词表版本，
并把两版的评分写进 JSON。完整实验数据（含 9 个 prompt 变体的对照表）见 **[docs/glossary.md](docs/glossary.md)**。

### 密钥安全

* 密钥只从 `V2C_LLM_API_KEY` 环境变量或工程根目录的 `.env` 读取（`.env` 已在 `.gitignore` 中）
* **没有命令行参数可以传密钥** —— 命令行参数会进进程列表与命令历史
* 日志、报错、JSON 里的密钥一律打码成 `sk-***abcd`
* 提交前可跑 `python scripts/check_secrets.py`（CI 里也会自动跑）

---

## 输出文件说明

`output/视频名.json` 是主产物：

```json
{
  "schema_version": "1.0",
  "generator": { "name": "video2context", "version": "0.1.0" },
  "created_at": "2025-09-25T00:36:32+08:00",
  "source": {
    "path": "D:\\videos\\demo.mp4", "filename": "demo.mp4", "size_bytes": 12345678,
    "duration": 390.5, "duration_hms": "00:06:30",
    "format": "mov,mp4,m4a,3gp,3g2,mj2", "has_video": true, "has_audio": true
  },
  "audio": {
    "path": null, "sample_rate": 16000, "channels": 1, "codec": "pcm_s16le", "kept": false
  },
  "asr": {
    "engine": "faster-whisper", "model": "large-v3",
    "device": "cuda", "compute_type": "float16", "task": "transcribe",
    "language": "zh", "language_probability": 0.994,
    "duration": 390.5, "duration_after_vad": 353.6,
    "segments_count": 17, "elapsed_seconds": 19.1, "model_load_seconds": 5.2,
    "realtime_factor": 0.049,
    "vad_filter": true, "beam_size": 5, "batch_size": 4,
    "temperature": null, "condition_on_previous_text": false, "initial_prompt": null
  },
  "text": "整篇文字稿……",
  "segments": [
    {
      "id": 0, "start": 0.02, "end": 31.37, "text": "……",
      "avg_logprob": -0.21, "no_speech_prob": 0.01,
      "temperature": 0.0, "compression_ratio": 1.4
    }
  ]
}
```

| 字段 | 含义 / 用途 |
|---|---|
| `asr.language` / `language_probability` | 自动检测出的语种及置信度 |
| `asr.realtime_factor` | 处理耗时 ÷ 音频时长，**小于 1 表示比实时快** |
| `asr.temperature` 等 | 实际生效的解码参数，便于复现结果 |
| `segments[].avg_logprob` | 该段平均对数概率，越接近 0 越自信（可用于筛查低质量段） |
| `segments[].no_speech_prob` | 该段"不是语音"的概率（可用于剔除幻觉） |
| `audio.kept` / `audio.path` | 是否保留了抽出的音频及路径 |

另可导出：`txt`（纯文本）、`srt`（播放器/剪辑软件字幕）、`vtt`（HTML5 `<track>`，网页播放器直接用）。

---

## 模型怎么选

底层用的是 **Whisper**（OpenAI 开源，MIT）的 **CTranslate2 加速版**，由 faster-whisper 调用。
模型名直接对应 faster-whisper 的官方命名：

| 模型 | 体积 | 显存占用 | 中文 | 英文 | 建议 |
|---|---|---|---|---|---|
| `tiny` / `base` | 75MB / 145MB | ~1GB | ✗ | 一般 | 只用来验证流程 |
| `small` | 484MB | ~2GB | 偏弱 | 尚可 | **没显卡时的首选** |
| `medium` | 1.5GB | ~5GB | 良好 | 良好 | 显卡较弱时用 |
| `large-v3` | **3.1GB** | ~5GB | **最好** | **最好** | **默认，推荐** |
| `large-v3-turbo` | 1.6GB | ~4GB | 好 | 好 | 要速度选它，约 4 倍速 |
| `distil-large-v3` | 1.5GB | ~4GB | ✗（别用） | 最好 | 纯英文专用 |
| `*.en` 系列 | — | — | ✗ | 略优于多语版 | 只处理英文时可用 |

**一句话选型**：

- **有显卡** → 直接默认 `large-v3`；想要更快就 `large-v3-turbo`
- **没显卡** → `small`（`--device cpu --compute-type int8`）；能等就用 `medium`
- 明确知道语言时**一定加** `--language zh` 或 `--language en`：省掉检测、减少小语种幻觉

更多细节（参数量、量化精度含义、如何换成 FunASR/SenseVoice）见 **[docs/models.md](docs/models.md)**。

### Qwen3-ASR（`--engine qwen3-asr`）

| 模型 | 体积 | 说明 |
|---|---|---|
| `Qwen3-ASR-1.7B` | 4.0GB | **推荐**。中文同音词/符号识别明显更好，加载约 7s，8GB 显存可跑 |
| `Qwen3-ASR-0.6B` | 1.6GB | 省显存，但也更容易错（样例上 `v(x)` 正确、`u(x)` 仍错 9 处） |
| `Qwen3-ForcedAligner-0.6B` | 1.8GB | 不是识别模型，只提供**词级时间戳**；不装则字幕时间只能估算 |

```powershell
python -m video2context 课程.mp4 --engine qwen3-asr -m models\Qwen3-ASR-1.7B --language zh
python -m video2context 课程.mp4 --engine qwen3-asr -m models\Qwen3-ASR-0.6B   # 省显存
python -m video2context 课程.mp4 --engine qwen3-asr --qwen-aligner off          # 不要精确字幕
```

---

## 实测性能

测试机：**i9-13900HX（32 线程）+ RTX 4060 Laptop 8GB + 15.7GB 内存**。

### Qwen3-ASR（默认引擎）

素材都用仓库里自带的样例，命令见[快速开始](#快速开始从-0-到跑通)，可以自己复现：

| 素材 | 设备 | 推理耗时 | 相对实时 |
|---|---|---|---|
| `samples/demo_zh_math.mp4`（41.8s 中文） | GPU bf16 + 对齐模型 | 5.5s | **7.6×** |
| `samples/demo_en_ml.mp4`（40.4s 英文） | GPU bf16 + 对齐模型 | 6.2s | **6.5×** |
| `samples/demo_zh_daily.mp4`（29.3s 中文） | GPU bf16 + 对齐模型 | 4.3s | **6.9×** |
| 12 秒英文片段 | **CPU**（32 线程） | 23.9s | **0.5×**（比实时还慢） |

> 另外一次 73 秒中文长素材（未提交仓库）跑到过 **12.5×** —— 音频越长越接近稳态。
> **模型加载另算**，本地模型每次 6.6～10.9s；音频越长这部分越无所谓。
> GPU 上 1 小时视频约 5～10 分钟。CPU 上**比实时还慢**，所以没有 N 卡请走 Whisper 路线。

**识别质量对比**（同一段音频、逐字核对，素材都在仓库里）：

| 素材 | `Qwen3-ASR-1.7B`（默认） | `faster-whisper` large-v3 |
|---|---|---|
| 英文 `demo_en_ml.mp4`（92 词，**WER**） | **0.00%** | **4.35%**（`Adam`→`Atom`、`RMSProp`→`RMS Prop`、`Adagrad`→`Atigrid`） |
| 中文 `demo_zh_math.mp4`（41.8s） | 「求导/可导」**全对**，标点完整 | 「求导」写成「**球倒**」，标点全变逗号 |
| 中文 `samples/formula.mp4`（73s，未提交） | `u(x)`、`v(x)`、`导数` **全对**，标点完整 | `u(x)`→「右」×4、`v(x)`→「位」×4、`倒/求倒/求到` 混乱 ×5，**无标点** |

> 中文那段的遗留小瑕疵：讲师口中的「x 零」「u1 x」会被写成 `x零`、`u1x`。
> 这是**把读法直接转写**的结果（讲师确实念作"x 零"），不是听错；
> 要变成 `x0`、`u1(x)` 得靠后处理改写，词表偏置改不动它（见上一节）。

### faster-whisper

输入 `samples/chinese_speech_sample.mp4`（6 分 30 秒 = 390 秒中文真人语音，VAD 后约 354 秒）。

| 配置 | 推理耗时 | 相对实时 | 说明 |
|---|---|---|---|
| `large-v3` + GPU float16 + 批量（默认，实际批量 4～7） | 16.2～19.1s | **20～24×** | 1 小时视频 ≈ 2.5～3 分钟 |
| `large-v3` + GPU float16 + 批量（73s 素材） | 2.45s | **29.5×** | 稳态吞吐 |
| `small` + CPU int8 | 64.2s | **6.1×** | 没显卡时的推荐配置 |
| `tiny` + CPU int8 | 11.2s | **35×** | 快，但中文质量差，仅用于试流程 |

> 批量大小由**剩余显存**自动决定（日志里会打印收缩过程），所以同一台机器上耗时会有波动。
> Whisper 固定按 30 秒窗口计算，音频越长越接近稳态吞吐。GPU 上约 **20～30 倍实时**。
> 短视频显得"慢"主要是模型加载（本地模型约 5s）和首次 CUDA 预热。
> CPU 数据强依赖核数：上表是 32 线程的成绩，4 核笔记本大致要慢 3～5 倍
> （`small` 约 1～2 倍实时，仍可用）。

复现全部实验（脚本与原始输出）：

```powershell
.\.venv-qwen\Scripts\python experiments\qwen3_asr_bench.py samples\formula.mp4 --model models\Qwen3-ASR-1.7B --aligner models\Qwen3-ForcedAligner-0.6B --language Chinese
.\.venv\Scripts\python experiments\wer_report.py 标准答案.txt 候选.txt     # 英文 WER 对比
```

---

## 工作原理

四步流水线，每一步都能单独替换：

```
①  媒体探测              ②  抽取音频                    ③  语音识别                  ④  写结果
ffmpeg -i 读元数据  →   ffmpeg -vn -ac 1 -ar 16000  →   识别引擎（二选一）        →   JSON / TXT / SRT / VTT
（时长/有无音轨/格式）    -c:a pcm_s16le 得 16k 单声道    faster-whisper / Qwen3-ASR   （带时间戳与参数快照）
```

| 模块 | 文件 | 职责 |
|---|---|---|
| 媒体工具层 | `video2context/ffmpeg_tools.py` | 定位 ffmpeg（环境变量 → PATH → 自带静态版）、探测媒体信息、抽取音轨 |
| 识别层（Whisper） | `video2context/transcriber.py` | 模型加载与设备选择、进度回调、显存预检、降级重试、模型缓存 |
| 识别层（Qwen） | `video2context/qwen_engine.py` | 环境/模型探测、子进程调度、**时间戳切句**、无对齐模型时的退化分配 |
| 子进程工人 | `video2context/_qwen_worker.py` | 在 `.venv-qwen` 里跑真正的 Qwen 推理，只依赖标准库 + torch |
| 流水线层 | `video2context/pipeline.py` | 串起四步、按 `--engine` 选识别器、组装结构化结果 |
| 输出层 | `video2context/writers.py` | JSON / TXT / SRT / VTT 序列化 |
| 交互层 | `video2context/cli.py`、`video2context/webui.py` | 命令行与网页界面 |

**用到的模型**（都在首次运行时自动下载）：

| 用途 | 模型 | 说明 |
|---|---|---|
| 语音识别（默认） | **Whisper large-v3**（OpenAI，MIT）的 CTranslate2 转换版 | 1.55B 参数，128 维 mel，多语种；CT2 做算子融合与量化 |
| 语音识别（可选） | **Qwen3-ASR-1.7B**（阿里，Apache-2.0） | 中文同音词与符号识别更强；走独立环境 + 子进程 |
| 字幕时间戳（可选） | **Qwen3-ForcedAligner-0.6B**（Apache-2.0） | 给 Qwen 的文字配词级时间戳；不装则按说话区间估算 |
| 人声检测（VAD） | **Silero VAD v6**（`silero_vad_v6.onnx`，1.2MB） | 随 faster-whisper 附带，用于切掉静音、避免幻觉 |
| 语种识别 | Whisper / Qwen 自带 | 解码首段时输出语种（Qwen 返回中文名如 `Chinese`） |

**几个关键设计**（详细论证见 [docs/architecture.md](docs/architecture.md)）：

- **为什么要转 16kHz 单声道**：Whisper 的输入就是 16kHz mel 频谱，提前用 ffmpeg 归一化可以避免重复重采样，也让 VAD 的判定稳定。
- **为什么默认开 VAD 且把静音阈值调成 500ms**：库默认 2000ms 偏保守；调小后字幕切分更细、长静音视频更快，同时仍能抑制静音段的幻觉。
- **为什么默认 `condition_on_previous_text=False`**：长音频里把上文当条件容易陷入"重复同一句"的循环；关掉更稳，代价是跨段连贯性略降（可用 `--condition-on-previous-text` 打开）。
- **为什么默认保留 Whisper 的温度回退序列**：当某段解码质量不达标（压缩比/对数概率异常）时，回退到更高温度重解码，比硬扛着输出乱码好。
- **设备选择策略**：`cuda/float16` → `cuda/int8_float16` → `cpu/int8` 依次尝试；推理前还会查一次剩余显存并自动收缩批量大小。
- **Qwen3-ASR 为什么走子进程**：它需要 torch，而主工程刻意不装 torch（faster-whisper 走 CTranslate2）。
  两个环境分开后，主环境永远轻量，Qwen 升级 torch/transformers 也不会波及主流程；
  代价是每次转写要多付一次模型加载（约 7～11s），音频越长越无所谓。

---

## 常见问题

**Q：没有显卡能用吗？**
用 **Whisper 引擎**可以（默认引擎 Qwen 在 CPU 上比实时还慢）：
`scripts\setup.ps1 -Engine whisper`，然后 `--model small --compute-type int8`。
CPU 上不要用 `large-v3`。

**Q：我只想用 Qwen，不想装 Whisper，可以吗？**
可以，这正是默认部署路线。`scripts\setup.ps1`（不带参数）只装公共依赖 + `.venv-qwen`，
**完全不装 faster-whisper**。详见 [docs/deploy.md](docs/deploy.md)。

**Q：我只装了 Whisper，为什么没让我指定引擎？**
因为默认引擎会按部署情况自适应：只装了 Whisper 就用 Whisper。跑 `doctor` 能看到
本机部署了哪些引擎、默认用哪个。

**Q：必须装 ffmpeg 吗？**
不用。`imageio-ffmpeg` 自带静态 ffmpeg。想用系统 ffmpeg 就设环境变量 `FFMPEG_BIN=<ffmpeg.exe 路径>`。

**Q：模型下载卡住 / 很慢？**
1) 加 `--hf-mirror`；2) 用自带下载器 `python scripts\download_model.py large-v3 --source modelscope`（断点续传，可反复跑）；
3) 代码已默认关闭 HuggingFace 的 Xet 存储（弱网下会挂起），需要时设 `HF_HUB_DISABLE_XET=0` 打开。
详见 [docs/troubleshooting.md](docs/troubleshooting.md)。

**Q：报错找不到 `cudnn64_9.dll` / `cublas64_12.dll`？**
Windows 执行 `pip install -r requirements-gpu-win.txt`，本工程会自动把 DLL 目录注册进搜索路径。

**Q：显存不足（CUDA out of memory）？**
- **Qwen 引擎**：优先加 `--qwen-low-mem on`，再不行换 `models\Qwen3-ASR-0.6B`。
  注意 Qwen 加载还需要约 13GB **系统提交内存**，光有显存不够。
- **Whisper 引擎**：程序会先自动收缩批量、再自动降级重试；仍失败会给出可直接照做的命令。
  手动优先 `--batch-size 0 --compute-type int8_float16`，再不行 `--device cpu`。
- 共同的坑：**系统内存不足会报同样的错**（模型先读进内存再上卡）。如果任务管理器里可用内存
  只有 4GB 左右，先关掉浏览器多标签/网盘/QQ 之类的常驻程序，也别同时跑两个任务。

**Q：识别结果有重复句子 / 幻觉？**
保持 VAD 开启（默认）、明确 `--language`、必要时加 `--initial-prompt`。
极端情况可把 `--beam-size` 提到 8～10。

**Q：专有名词、人名地名总是错（同音词）？**
用**领域词表**：手写 `--glossary terms.txt`，或自动 `--auto-glossary`（先 `--dry-run-glossary` 确认词表）。
实测能把「右F4 → u(x)」这类错误从 13 处降到 0 处。详见 [领域词表](#领域词表修同音词专有名词) 与 [docs/glossary.md](docs/glossary.md)。
⚠️ 注意提示词偏置会**静默失效**，所以程序内置了验证与自动回退。

**Q：中文标点不理想？**
Whisper 自带标点是模型行为，不是后处理。若对中文标点要求高，可以换 `--engine qwen3-asr`
（标点更完整），或换成 FunASR/SenseVoice，见 [docs/models.md](docs/models.md)。

**Q：`--engine qwen3-asr` 报「找不到独立环境 .venv-qwen」？**
说明还没建 Qwen 的独立环境（主环境刻意不装 torch）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_qwen.ps1
.\.venv\Scripts\python -m video2context --engine qwen3-asr --qwen-setup   # 确认就绪
```

已经有别的环境装了 `qwen-asr`，就把解释器路径写进环境变量 `V2C_QWEN_PYTHON`。

**Q：Qwen 引擎报「页面文件太小 / out of memory」？**
1.7B 模型加载要读 4GB 分片，**需要约 13GB 提交内存**（不只是显存）。按顺序试：
`--qwen-low-mem on`（分片直接进显存，CPU 侧几乎不占内存）→ 关掉浏览器/网盘 →
换 `models\Qwen3-ASR-0.6B`。详见 [docs/troubleshooting.md](docs/troubleshooting.md)。

**Q：Qwen 引擎的字幕时间戳不准？**
检查 JSON 里的 `asr.timestamp_source`，它标明字幕时间用的是哪一档：
`forced-aligner` = 精确到词；`vad-proportional` = 按说话区间估算；
`even-spread` = 整条音轨平均分（只装 Qwen、没装 Whisper 也没有对齐模型时会出现，最粗）。
除第一档外都建议补下对齐模型（1.8GB），补完直接跳到精确档：
`python scripts\download_model.py --repo Qwen/Qwen3-ForcedAligner-0.6B --source modelscope --out models\Qwen3-ForcedAligner-0.6B`。
详见 [docs/troubleshooting.md 7.4](docs/troubleshooting.md#74-字幕时间戳不准--所有段落挤在一起)。

**Q：网页界面能远程/局域网访问吗？**
`--host 0.0.0.0 --port 7860`，然后用本机 IP 访问。公网临时分享用 `--share`（走 Gradio 官方隧道）。
⚠️ 开放到公网前请自行加认证，页面本身没有权限控制。

**Q：能处理 2 小时的视频吗？**
能。建议加 `--duration 600` 先试跑；长视频的内存占用与视频长度无关（音频是流式读取的），
磁盘上会临时产生一个 16kHz wav（1 小时约 115MB，可用 `--keep-audio` 保留）。

**Q：结果里时间戳对不上画面？**
时间戳相对**原始视频**计算，`--start` 裁剪时会记录在 `source.clip` 里。若源文件本身有可变帧率（VFR），建议先转成固定帧率再处理。

---

## 项目结构

```
video2context/
├─ video2context/            # 主包
│  ├─ __main__.py            # python -m video2context 入口
│  ├─ cli.py                 # 命令行：transcribe / webui / doctor
│  ├─ pipeline.py            # 主流程：探测 → 抽音频 → 识别 → 写结果（含引擎选择）
│  ├─ transcriber.py         # faster-whisper 封装：设备选择、显存预检、降级、缓存
│  ├─ qwen_engine.py         # Qwen3-ASR 引擎：环境探测、子进程调度、时间戳切句
│  ├─ _qwen_worker.py        # 在 .venv-qwen 里跑的工人（只依赖标准库 + torch）
│  ├─ glossary.py            # 领域词表：符号还原、prompt/context 拼装、偏置评分
│  ├─ llm.py                 # 极小的大模型客户端（OpenAI 兼容，零新依赖）
│  ├─ config.py              # .env / 环境变量读取、密钥脱敏
│  ├─ ffmpeg_tools.py        # ffmpeg 定位、媒体探测、音轨提取
│  ├─ writers.py             # JSON / TXT / SRT / VTT 输出
│  └─ webui.py               # Gradio 网页界面
├─ docs/                     # 详细文档（部署、架构、模型、排查、开发）
├─ experiments/              # Qwen3-ASR 实测脚本与原始结果（不进流水线）
├─ samples/                  # 可直接跑的示例视频（合成 + 公共领域真人录音）+ 逐句稿子
├─ scripts/
│  ├─ setup.ps1 / setup.sh   # 一键部署，-Engine qwen|whisper|both（默认 qwen）
│  ├─ setup_qwen.ps1 / .sh   # 只部署 Qwen（含主环境 + .venv-qwen），可单独使用
│  ├─ run_web.ps1 / run_web.sh
│  ├─ download_model.py      # 模型分块断点续传下载器（HF / 魔搭双源，纯标准库）
│  ├─ make_samples.py        # 离线生成 samples/ 里的示例视频（本机 TTS + ffmpeg）
│  ├─ check_docs.py          # 文档链接与锚点校验
│  └─ check_secrets.py       # 提交前防泄露扫描
├─ tests/                    # 离线单元测试（不需要模型和显卡）
├─ .env.example              # 密钥与默认识别引擎的配置模板（.env 本身已 gitignore）
├─ .github/workflows/ci.yml  # 持续集成
├─ requirements-base.txt     # 公共依赖（ffmpeg），任何引擎都要
├─ requirements.txt          # + Whisper 引擎（faster-whisper）
├─ requirements-qwen.txt     # + Qwen3-ASR（装在独立的 .venv-qwen）
├─ requirements-web.txt      # + 网页界面
├─ requirements-gpu-win.txt  # + Windows NVIDIA CUDA 运行库（Whisper 引擎用）
├─ requirements-dev.txt      # + pytest / ruff
├─ pyproject.toml
├─ LICENSE / THIRD_PARTY_NOTICES.md
├─ CHANGELOG.md / CONTRIBUTING.md
└─ README.md
```

---

## 开发 / 测试

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v    # 109 项测试，离线、不需要模型和显卡、不需要密钥
ruff check video2context tests scripts      # 代码检查
python scripts/check_docs.py                # 文档链接与锚点
python scripts/check_secrets.py --all       # 防泄露扫描
```

测试用 ffmpeg 现场合成素材，并用替身对象替换识别器与大模型，因此**在 CI 里无需下载模型、无需 API Key**。
详见 [CONTRIBUTING.md](CONTRIBUTING.md) 与 [docs/development.md](docs/development.md)。

---

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/deploy.md](docs/deploy.md) | **从 0 部署**：三条路线（只装 Qwen / 只装 Whisper / 都装）、各设备与平台、模型下载、离线内网、国内加速、磁盘占用与卸载 |
| [docs/glossary.md](docs/glossary.md) | 领域词表：三种用法、两遍解码实现、**9 个 prompt 变体的实验数据**、验证回退机制、密钥安全 |
| [docs/architecture.md](docs/architecture.md) | 每个模块怎么实现、ffmpeg 具体命令、VAD/解码参数、显存与批量、错误处理与降级链路 |
| [docs/models.md](docs/models.md) | 两个引擎的模型详解（Qwen3-ASR 三个模型 + Whisper 家族 + CTranslate2 量化）、选型表、实测数据 |
| [docs/troubleshooting.md](docs/troubleshooting.md) | 按症状排查：安装、ffmpeg、模型下载、显卡、识别质量、性能 |
| [docs/development.md](docs/development.md) | 代码结构、测试策略、如何加参数/加格式/换引擎、发布检查清单 |

---

## 路线图

- [ ] 中英互译与要点摘要（基于已有的结构化 JSON 接 LLM）
- [ ] 说话人分离（pyannote.audio），给 segment 补 `speaker`
- [ ] WebVTT 样式与双语字幕输出
- [ ] Docker 镜像（CPU / GPU 两版）
- [ ] 批量目录监听与增量处理

---

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

第三方依赖、模型权重与测试素材的来源和许可情况见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)；
`samples/` 里的中文测试音频来自 [LibriVox](https://librivox.org/)（公共领域）。

如果这个工程帮到了你，欢迎点个 Star ⭐
