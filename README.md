# video2context · 视频转文字

> 把视频丢进去，本地抽出音轨、识别成文字，输出**带时间戳的 JSON / SRT / VTT / TXT**。
> 默认全程在本机运行、不需要任何 API Key；**有显卡能用，没有显卡也能用**。
> 可选接入大模型做「领域词表」，修掉同音词与专有名词错误（不上传音频，只发一小段转写文本）。

![CI](https://github.com/forward-intoxicate/video2context/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)

```
视频.mp4 ──ffmpeg──▶ 16kHz 单声道 wav ──▶ 识别引擎 ──▶ JSON / SRT / VTT / TXT
                                            ├─ faster-whisper（默认，轻量、无需 torch）
                                            └─ Qwen3-ASR（中文同音词/数学符号更准）
```

**两个识别引擎**（`--engine` 切换，输出格式完全一致）：

| 引擎 | 依赖 | 中文同音词 | 英文 WER | 适合 |
|---|---|---|---|---|
| `faster-whisper`（默认） | 轻量，**不需要 torch**，CPU 就能跑 | 一般（`u(x)`→「右F4」） | 2.97% | 通用、无显卡、想省事 |
| `qwen3-asr` | 独立环境 + torch，约 4GB 显存 | **好**（样例上 0 处同音词错误） | **0.00%** | 中文课程/讲座、术语多的内容 |

（数据来源见[实测性能](#实测性能)，实验脚本与原始结果在 `experiments/`。）

|  | 说明 |
|---|---|
| **输入** | 视频（mp4/mkv/mov/avi/flv…）或纯音频（mp3/wav/m4a/flac…） |
| **语言** | 中文、英文自动识别；也可强制指定（更快、更准） |
| **输出** | JSON（分段+时间戳+语种+参数）、SRT、VTT、TXT |
| **硬件** | NVIDIA 显卡（推荐）或纯 CPU，程序自动选择 |
| **联网** | 仅首次下载模型需要；之后可完全离线（自动词表除外） |

---

## 目录

- [这个工程适合谁](#这个工程适合谁)
- [快速开始](#快速开始)
- [安装详解（有显卡 / 没有显卡）](#安装详解有显卡--没有显卡)
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

## 这个工程适合谁

**适合**：想把会议录像、课程视频、采访、播客、Vlog 转成文字或字幕；
想给视频做内容检索；不想把视频传到第三方服务；网络环境一般（国内）也想顺利装上。

**不合适**：需要实时（边说边出字）——本工程是文件级批处理；
需要专业级说话人分离/精确标点（可自行接入，见[扩展方向](CONTRIBUTING.md)）。

**不用懂 AI**：装好之后就是一条命令的事。默认参数已经调到"中文长视频要精度"的档位。

---

## 快速开始

### 0. 你需要什么

| | 最低 | 推荐 |
|---|---|---|
| 系统 | Windows 10/11、macOS、Linux | 同左 |
| Python | 3.9（网页界面需 3.10+） | 3.11 / 3.12 / 3.13 |
| 内存 | 8 GB | 16 GB |
| 显卡 | **不需要**，纯 CPU 可跑 | NVIDIA，显存 ≥ 6GB（8GB 可跑 large-v3） |
| 磁盘 | 2 GB（小模型） | 6 GB（large-v3 + 中间文件） |

> **不需要单独安装 ffmpeg**：依赖 `imageio-ffmpeg` 自带的静态 ffmpeg，装完即可用。

### 1. 装环境（Windows）

```powershell
git clone https://github.com/forward-intoxicate/video2context.git
cd video2context

# 一键脚本：自动判断有没有 N 卡、装依赖、跑自检
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

### 也可以在 macOS / Linux 上装

```bash
git clone https://github.com/forward-intoxicate/video2context.git
cd video2context
bash scripts/setup.sh
```

### 2. 转写

```powershell
# Windows
.\.venv\Scripts\python -m video2context "D:\videos\我的视频.mp4"
```
```bash
# macOS / Linux
./.venv/bin/python -m video2context ~/videos/my_video.mp4
```

结果落在 `output/我的视频.json`（如需字幕加 `-f json,srt`）。

### 3. 或者用网页界面（拖拽即用）

```powershell
.\.venv\Scripts\python -m video2context webui      # 浏览器打开 http://127.0.0.1:7860
```

### 4. 想先拿现成的素材试跑？

仓库里带了一段 6 分 30 秒的中文真人语音测试视频（公共领域）：

```powershell
.\.venv\Scripts\python -m video2context samples\chinese_speech_sample.mp4 -f json,srt,txt -o output
```

---

## 安装详解（有显卡 / 没有显卡）

三种依赖按需组合，互不冲突：

| 文件 | 内容 | 谁需要 |
|---|---|---|
| `requirements.txt` | faster-whisper + imageio-ffmpeg | **所有人**（纯 CPU 也靠它） |
| `requirements-web.txt` | + Gradio 网页界面 | 想要图形界面的人 |
| `requirements-gpu-win.txt` | + cuBLAS / cuDNN 运行库 | **Windows + NVIDIA 显卡** |
| `requirements-qwen.txt` | + Qwen3-ASR（装在**独立**的 `.venv-qwen`） | 想要 Qwen 引擎的人（可选） |

### 没有显卡（纯 CPU）

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -r requirements.txt          # 命令行
.\.venv\Scripts\python -m pip install -r requirements-web.txt      # 可选：网页界面
.\.venv\Scripts\python -m video2context doctor                      # 确认环境就绪
```

CPU 上建议用小一点的模型和 int8 精度（见[模型怎么选](#模型怎么选)）：

```powershell
.\.venv\Scripts\python -m video2context 视频.mp4 --model small --device cpu --compute-type int8
```

### 有显卡（Windows + NVIDIA）

照上面装完核心依赖后，再装 CUDA 运行库：

```powershell
.\.venv\Scripts\python -m pip install -r requirements-gpu-win.txt
```

它做两件事：装 cuBLAS 12 + cuDNN 9 的 DLL，并被本工程**自动注册进 DLL 搜索路径**
（这是 Windows 上 faster-whisper 报 `cudnn64_9.dll not found` 的根因）。
已装 CUDA Toolkit 12.x + cuDNN 9 的机器可以跳过。

### 有显卡（Linux）

Linux 上 ctranslate2 需要系统级 CUDA 12 运行库与 cuDNN 9：

```bash
# Ubuntu 示例（或改用 conda install -c conda-forge cudnn cuda-toolkit）
sudo apt install libcublas-12-* libcudnn9-cuda-12
```

### 国内网络加速

```powershell
# pip 走清华源
.\.venv\Scripts\python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 下载模型走镜像（或直接看下面的"手动下载模型"）
.\.venv\Scripts\python -m video2context 视频.mp4 --hf-mirror
```

### 可选：装 Qwen3-ASR 引擎

Qwen3-ASR 需要 `torch` + `transformers`（约 3GB），而主环境刻意保持 torch-free。
所以它装在**另一个虚拟环境 `.venv-qwen`** 里，由子进程桥接调用 —— 详见
[docs/architecture.md](docs/architecture.md)「Qwen3-ASR 为什么走子进程」。

```powershell
# Windows：装独立环境（默认 cu126；没显卡用 -Torch cpu）
powershell -ExecutionPolicy Bypass -File scripts\setup_qwen.ps1

# 顺带把模型也下了（约 6GB，可断点续传；不加这个开关就按提示手动下）
powershell -ExecutionPolicy Bypass -File scripts\setup_qwen.ps1 -DownloadModels

# 自检：解释器 / 模型 / 对齐模型是否就绪
.\.venv\Scripts\python -m video2context --engine qwen3-asr --qwen-setup
```

```bash
# macOS / Linux
bash scripts/setup_qwen.sh
DOWNLOAD_MODELS=1 bash scripts/setup_qwen.sh     # 连模型一起下
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

> 下面的命令假设已经激活虚拟环境：
> Windows `.\.venv\Scripts\Activate.ps1`，macOS/Linux `source .venv/bin/activate`。
> 不想激活就把 `python` 换成 `.\.venv\Scripts\python`（Windows）或 `./.venv/bin/python`。

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

# 专有名词多的视频：喂提示词，人名地名准确率明显提升
python -m video2context 讲座.mp4 --initial-prompt "本次讲座涉及：卷积神经网络、反向传播、吴恩达。"

# 只想要字幕
python -m video2context a.mp4 -f srt

# 没显卡 / 显存不够
python -m video2context a.mp4 --device cpu --model small --compute-type int8

# 换 Qwen3-ASR 引擎（中文同音词、数学符号更准；需先装 .venv-qwen）
python -m video2context 课程.mp4 --engine qwen3-asr --language zh -f json,srt

# Qwen 引擎的环境自检
python -m video2context --engine qwen3-asr --qwen-setup

# 环境自检
python -m video2context doctor
```

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
| `--engine` | 识别引擎：`faster-whisper` / `qwen3-asr` | `faster-whisper` |
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

页面左侧上传视频、选模型/语种/设备，右侧实时进度 + 全文 + 分段表格 + JSON 下载。
默认只监听 `127.0.0.1`（仅本机可访问）；局域网访问加 `--host 0.0.0.0`。

---

## 识别引擎怎么选

同一个流水线，`--engine` 换识别器，**输出格式、命令行参数、SRT/VTT 全部一致**，所以可以随时 A/B 对比。

| | `faster-whisper`（默认） | `qwen3-asr` |
|---|---|---|
| 模型 | Whisper large-v3 的 CTranslate2 版（MIT） | Qwen3-ASR-1.7B（Apache-2.0） |
| 依赖 | `requirements.txt`，**不需要 torch** | 独立环境 `.venv-qwen`（torch + transformers，约 3GB） |
| 磁盘 | 3.1GB | 4GB（+1.8GB 对齐模型） |
| 显存 | ~4GB（int8 可低到 2GB） | ~4GB（加载需约 13GB **提交内存**） |
| CPU | 可用（`small` 约 6 倍实时） | 很慢，不推荐 |
| 中文标点 | 模型自带，尚可 | 更完整 |
| 中文同音词 | `u(x)`→「右F4」这类错误多 | **样例上 0 处** |
| 英文 WER | 2.97% | **0.00%** |
| 翻译成英文 | 支持（`--task translate`） | 不支持（只做原语言转写） |

**建议**：

- **先用默认的 `faster-whisper`**。它装起来最省事，没有显卡也能跑，够用。
- 内容里有**数学符号、公式、专业术语、大量英文人名**（课程录像、技术讲座），再上 `--engine qwen3-asr`。
- 两个引擎都跑一遍对比：它们读同一份 16kHz wav，结果 JSON 里的 `asr.engine` 字段能直接区分。

```powershell
# 同一条视频，两种引擎各跑一遍（输出名不同，方便 diff）
python -m video2context 课程.mp4 -f json,srt -n 课程-whisper
python -m video2context 课程.mp4 --engine qwen3-asr -f json,srt -n 课程-qwen
```

### Qwen 引擎的两个实测细节

**1. 字幕时间戳来自「强制对齐模型」**

Qwen3-ASR 本身只吐**一整段文字**，没有分段。本工程的做法是：

- 装了 `Qwen3-ForcedAligner-0.6B` → 拿到**词级时间戳**，再把 token 映射回原文下标，
  按标点/停顿切句 → SRT 时间精确到词（实测切出的分段能 100% 还原原文，标点不丢）；
- 没装对齐模型 → 按标点切句，再用 faster-whisper 自带的 Silero VAD 找出说话区间，
  按字数比例把句子分配到区间里。**能用，但时间只是估算**，JSON 里
  `asr.timestamp_source` 会明确写成 `vad-proportional` 而不是 `forced-aligner`。

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

测试机：**i9-13900HX（32 线程）+ RTX 4060 Laptop 8GB**，输入 `samples/chinese_speech_sample.mp4`
（6 分 30 秒 = 390 秒中文真人语音，VAD 后约 354 秒）。

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

### Qwen3-ASR 实测

同一台机器（RTX 4060 Laptop 8GB）：

| 素材 | 配置 | 加载 | 推理 | 相对实时 |
|---|---|---|---|---|
| `samples/formula.mp4`（73s 中文数学课） | 1.7B bf16 + 对齐模型 | 6.6s | 5.8s | **12.5×** |
| 同上 | 1.7B bf16 + 对齐模型 + 词表 context | 6.6s | 7.3s | 10× |
| 英文测试素材（42s） | 1.7B bf16 + 对齐模型 | 10.9s | 8.8s | 4.8× |

**识别质量对比**（同一段音频，逐字核对）：

| 素材 | `faster-whisper` large-v3 | `Qwen3-ASR-1.7B` |
|---|---|---|
| 中文数学课 73s | `u(x)`→「右」×4、`v(x)`→「位」×4、`倒/求倒/求到` 混乱 ×5，**无标点** | `u(x)`、`v(x)`、`导数` **全对**，标点完整 |
| 英文 42s（101 词，WER） | **2.97%**（`Adam`→`Atom`、`RMSProp`→`RMS Prop`） | **0.00%** |

> 中文那段的遗留小瑕疵：讲师口中的「x 零」「u1 x」会被写成 `x零`、`u1x`。
> 这是**把读法直接转写**的结果（讲师确实念作"x 零"），不是听错；
> 要变成 `x0`、`u1(x)` 得靠后处理改写，词表偏置改不动它（见上一节）。

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
能，程序会自动退回 CPU。只是慢一些，建议 `--model small --compute-type int8`。CPU 上不要用 `large-v3`。

**Q：必须装 ffmpeg 吗？**
不用。`imageio-ffmpeg` 自带静态 ffmpeg。想用系统 ffmpeg 就设环境变量 `FFMPEG_BIN=<ffmpeg.exe 路径>`。

**Q：模型下载卡住 / 很慢？**
1) 加 `--hf-mirror`；2) 用自带下载器 `python scripts\download_model.py large-v3 --source modelscope`（断点续传，可反复跑）；
3) 代码已默认关闭 HuggingFace 的 Xet 存储（弱网下会挂起），需要时设 `HF_HUB_DISABLE_XET=0` 打开。
详见 [docs/troubleshooting.md](docs/troubleshooting.md)。

**Q：报错找不到 `cudnn64_9.dll` / `cublas64_12.dll`？**
Windows 执行 `pip install -r requirements-gpu-win.txt`，本工程会自动把 DLL 目录注册进搜索路径。

**Q：显存不足（CUDA out of memory）？**
程序会先自动收缩批量、再自动降级重试；仍失败会给出可直接照做的命令。
手动优先 `--batch-size 0 --compute-type int8_float16`，再不行 `--device cpu`。
⚠️ 注意**系统内存不足会报同样的错**（模型先读进内存再上卡）：如果任务管理器里可用内存只有 4GB 左右，
先关掉浏览器多标签/网盘/QQ 之类的常驻程序。也别同时跑两个任务（网页界面一个、命令行一个）。

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
检查 JSON 里的 `asr.timestamp_source`：`forced-aligner` = 精确，`vad-proportional` = 估算。
是后者就说明没装对齐模型，补下：
`python scripts\download_model.py --repo Qwen/Qwen3-ForcedAligner-0.6B --source modelscope --out models\Qwen3-ForcedAligner-0.6B`。

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
├─ docs/                     # 详细文档（架构、模型、排查、开发）
├─ experiments/              # Qwen3-ASR 实测脚本与原始结果（不进流水线）
├─ samples/                  # 公共领域中文测试视频 + 参考输出
├─ scripts/
│  ├─ setup.ps1 / setup.sh   # 一键建主环境（Windows / macOS+Linux）
│  ├─ setup_qwen.ps1 / .sh   # 一键建 Qwen 独立环境 .venv-qwen（可选）
│  ├─ run_web.ps1 / run_web.sh
│  ├─ download_model.py      # 模型分块断点续传下载器（HF / 魔搭双源）
│  ├─ check_docs.py          # 文档链接与锚点校验
│  └─ check_secrets.py       # 提交前防泄露扫描
├─ tests/                    # 离线单元测试（不需要模型和显卡）
├─ .env.example              # 大模型密钥配置模板（.env 本身已 gitignore）
├─ .github/workflows/ci.yml  # 持续集成
├─ requirements.txt          # 核心依赖（CPU 也能跑）
├─ requirements-web.txt      # + 网页界面
├─ requirements-gpu-win.txt  # + Windows NVIDIA CUDA 运行库
├─ requirements-qwen.txt     # + Qwen3-ASR（装在独立的 .venv-qwen）
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
python -m unittest discover -s tests -v    # 83 项测试，离线、不需要模型和显卡、不需要密钥
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
| [docs/glossary.md](docs/glossary.md) | 领域词表：三种用法、两遍解码实现、**9 个 prompt 变体的实验数据**、验证回退机制、密钥安全 |
| [docs/architecture.md](docs/architecture.md) | 每个模块怎么实现、ffmpeg 具体命令、VAD/解码参数、显存与批量、错误处理与降级链路 |
| [docs/models.md](docs/models.md) | Whisper 家族与 CTranslate2 量化详解、VAD 模型、选型表、如何换 FunASR/SenseVoice |
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
