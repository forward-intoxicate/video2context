# video2context · 视频转文字

> 把视频丢进去，本地抽出音轨、识别成文字，输出**带时间戳的 JSON / SRT / VTT / TXT**。
> 全部在本机运行，不需要 API Key，**有显卡能用，没有显卡也能用**。

![CI](https://github.com/forward-intoxicate/video2context/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)

```
视频.mp4 ──ffmpeg──▶ 16kHz 单声道 wav ──Silero VAD 切分──▶ faster-whisper ──▶ JSON / SRT / VTT / TXT
```

|  | 说明 |
|---|---|
| **输入** | 视频（mp4/mkv/mov/avi/flv…）或纯音频（mp3/wav/m4a/flac…） |
| **语言** | 中文、英文自动识别；也可强制指定（更快、更准） |
| **输出** | JSON（分段+时间戳+语种+参数）、SRT、VTT、TXT |
| **硬件** | NVIDIA 显卡（推荐）或纯 CPU，程序自动选择 |
| **联网** | 仅首次下载模型需要；之后可完全离线 |

---

## 目录

- [这个工程适合谁](#这个工程适合谁)
- [快速开始](#快速开始)
- [安装详解（有显卡 / 没有显卡）](#安装详解有显卡--没有显卡)
- [怎么用](#怎么用)
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
| `-m, --model` | 模型名或本地模型目录 | `large-v3` |
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
| `--model-dir` | 本地模型目录（配合下载器，完全离线） | 无 |
| `--local-files-only` | 只用本地缓存，不联网 | 关 |
| `--hf-endpoint` / `--hf-mirror` | 模型下载源 / 用 hf-mirror | 官方源 |
| `--cpu-threads` | CPU 线程数，0=自动 | `0` |

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

---

## 工作原理

四步流水线，每一步都能单独替换：

```
①  媒体探测              ②  抽取音频                    ③  语音识别                  ④  写结果
ffmpeg -i 读元数据  →   ffmpeg -vn -ac 1 -ar 16000  →   Silero VAD 切分语音段   →   JSON / TXT / SRT / VTT
（时长/有无音轨/格式）    -c:a pcm_s16le 得 16k 单声道    faster-whisper 逐段识别      （带时间戳与参数快照）
```

| 模块 | 文件 | 职责 |
|---|---|---|
| 媒体工具层 | `video2context/ffmpeg_tools.py` | 定位 ffmpeg（环境变量 → PATH → 自带静态版）、探测媒体信息、抽取音轨 |
| 识别层 | `video2context/transcriber.py` | 模型加载与设备选择、进度回调、显存预检、降级重试、模型缓存 |
| 流水线层 | `video2context/pipeline.py` | 串起四步，组装结构化结果 |
| 输出层 | `video2context/writers.py` | JSON / TXT / SRT / VTT 序列化 |
| 交互层 | `video2context/cli.py`、`video2context/webui.py` | 命令行与网页界面 |

**用到的模型**（都在首次运行时自动下载）：

| 用途 | 模型 | 说明 |
|---|---|---|
| 语音识别 | **Whisper large-v3**（OpenAI，MIT）的 CTranslate2 转换版 | 1.55B 参数，128 维 mel，多语种；CT2 做算子融合与量化 |
| 人声检测（VAD） | **Silero VAD v6**（`silero_vad_v6.onnx`，1.2MB） | 随 faster-whisper 附带，用于切掉静音、避免幻觉 |
| 语种识别 | Whisper 自带 | 解码首段时输出语种概率分布 |

**几个关键设计**（详细论证见 [docs/architecture.md](docs/architecture.md)）：

- **为什么要转 16kHz 单声道**：Whisper 的输入就是 16kHz mel 频谱，提前用 ffmpeg 归一化可以避免重复重采样，也让 VAD 的判定稳定。
- **为什么默认开 VAD 且把静音阈值调成 500ms**：库默认 2000ms 偏保守；调小后字幕切分更细、长静音视频更快，同时仍能抑制静音段的幻觉。
- **为什么默认 `condition_on_previous_text=False`**：长音频里把上文当条件容易陷入"重复同一句"的循环；关掉更稳，代价是跨段连贯性略降（可用 `--condition-on-previous-text` 打开）。
- **为什么默认保留 Whisper 的温度回退序列**：当某段解码质量不达标（压缩比/对数概率异常）时，回退到更高温度重解码，比硬扛着输出乱码好。
- **设备选择策略**：`cuda/float16` → `cuda/int8_float16` → `cpu/int8` 依次尝试；推理前还会查一次剩余显存并自动收缩批量大小。

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

**Q：中文标点不理想？**
Whisper 自带标点是模型行为，不是后处理。若对中文标点要求高，可换成 FunASR/SenseVoice，见 [docs/models.md](docs/models.md)。

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
│  ├─ pipeline.py            # 主流程：探测 → 抽音频 → 识别 → 写结果
│  ├─ transcriber.py         # faster-whisper 封装：设备选择、显存预检、降级、缓存
│  ├─ ffmpeg_tools.py        # ffmpeg 定位、媒体探测、音轨提取
│  ├─ writers.py             # JSON / TXT / SRT / VTT 输出
│  └─ webui.py               # Gradio 网页界面
├─ docs/                     # 详细文档（架构、模型、排查、开发）
├─ samples/                  # 公共领域中文测试视频 + 参考输出
├─ scripts/
│  ├─ setup.ps1 / setup.sh   # 一键建环境（Windows / macOS+Linux）
│  ├─ run_web.ps1 / run_web.sh
│  └─ download_model.py      # 模型分块断点续传下载器（HF / 魔搭双源）
├─ tests/                    # 离线单元测试（不需要模型和显卡）
├─ .github/workflows/ci.yml  # 持续集成
├─ requirements.txt          # 核心依赖（CPU 也能跑）
├─ requirements-web.txt      # + 网页界面
├─ requirements-gpu-win.txt  # + Windows NVIDIA CUDA 运行库
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
python -m unittest discover -s tests -v    # 28 项测试，离线、不需要模型和显卡
ruff check video2context tests scripts      # 代码检查
```

测试用 ffmpeg 现场合成素材，并用替身对象替换识别器，因此**在 CI 里无需下载模型**。
详见 [CONTRIBUTING.md](CONTRIBUTING.md) 与 [docs/development.md](docs/development.md)。

---

## 文档索引

| 文档 | 内容 |
|---|---|
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
