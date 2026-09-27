# 从 0 部署与使用

本文是**部署手册**：从一台干净机器到能跑出第一条字幕，逐步说明不同设备该走哪条路。
想先了解工程是干什么的，看 [README](../README.md)；想改代码，看 [development.md](development.md)。

---

## 0. 先选路线

工程带**两个识别引擎**，都通过 `--engine` 切换，输出格式完全一致：

| 引擎 | 一句话 | 需要装什么 |
|---|---|---|
| **`qwen3-asr`**（默认，推荐） | 中文同音词/数学符号明显更准、英文 WER 更低 | 独立环境 `.venv-qwen`（torch，约 3GB）+ 模型 4GB |
| `faster-whisper` | 轻量、不需要 torch、CPU 也能跑 | 主环境里一个 pip 包 + 模型 3.1GB |

**两者互不依赖**：可以只装 Qwen、完全不装 Whisper。按下面三条路线挑一条：

| 路线 | 适合谁 | 磁盘 | 命令 |
|---|---|---|---|
| **A. 只装 Qwen3-ASR** | 有 NVIDIA 显卡，要最好的中文效果 | ~9GB | `setup.ps1`（默认） / `setup.sh`（默认） |
| **B. 只装 Whisper** | 没有显卡、或者想省 3GB 的 torch | ~4GB | `setup.ps1 -Engine whisper` / `setup.sh whisper` |
| **C. 两个都装** | 想 A/B 对比，或不确定 | ~13GB | `setup.ps1 -Engine both` / `setup.sh both` |

> 装哪个引擎，默认就用哪个 —— 程序会自己探测（见[第 6 节](#6-引擎选择与切换)）。
> 所以路线 B 的用户不用记着加 `--engine faster-whisper`。

### 硬件要求

| | 只装 Qwen | 只装 Whisper |
|---|---|---|
| 系统 | Windows 10/11、Linux、macOS | 同左 |
| Python | **3.9 ~ 3.13** | 同左（网页界面需 3.10+） |
| 显卡 | **NVIDIA，显存 ≥ 6GB**（推荐 8GB） | 可选，没有就用 CPU |
| 系统内存 | **16GB 起**（加载模型时要约 13GB 提交内存） | 8GB 够 |
| 磁盘 | ~9GB（环境 3GB + 模型 6GB） | ~4GB（依赖 0.9GB + 模型 3.1GB） |

**关于内存**（只装 Qwen 时最容易踩的坑）：`transformers` 加载权重时会把 4GB 的分片
整个读进 CPU 内存再搬上显卡，所以峰值提交内存约 13GB。16GB 内存的机器上，
如果浏览器开着几十个标签，就会报 `页面文件太小` 或直接崩。
两个解法：关掉占内存的程序，或者加 `--qwen-low-mem on`（分片直接进显存，CPU 侧几乎不占内存）。

**没有 NVIDIA 显卡能用 Qwen 吗？** 能，但**比实时还慢**：实测 12 秒音频要跑 23.9 秒
（约 0.5 倍实时，1 小时视频要等 2 小时），只适合偶尔跑一小段。没有显卡请走路线 B。

---

## 1. 通用前置

```bash
git clone https://github.com/forward-intoxicate/video2context.git
cd video2context
```

只需要 **Python 3.9+** 和 **git**。

> **不需要单独安装 ffmpeg**：依赖里的 `imageio-ffmpeg` 自带静态 ffmpeg 可执行文件。
> 想用系统 ffmpeg 的话，设环境变量 `FFMPEG_BIN=<ffmpeg 完整路径>` 即可。

**Linux 上如果 `python3 -m venv` 报 `ensurepip is not available`**，
先补系统包再重跑安装脚本（Debian / Ubuntu，发行版不同名字略有差异）：

```bash
sudo apt install python3-venv python3-pip
```

**Anaconda 用户**若遇到 venv 建不起来（`ensurepip` 被裁剪），
安装脚本已内置绕法（先 `--without-pip` 建空环境，再用主解释器的 pip 灌进去）；
手工操作就是：

```bash
python -m venv --without-pip .venv
python -m pip --python .venv/bin/python install --upgrade pip
```

---

## 2. 路线 A：只部署 Qwen3-ASR（默认，推荐）

这一条路**不安装 Whisper**：主环境只有一个 ffmpeg 包，识别全部跑在独立的 `.venv-qwen` 里。

脚本会**默认把两个模型都下好**（识别模型 4GB + 强制对齐模型 1.8GB，共约 6GB）：

| 模型 | 体积 | 作用 | 能不能不下 |
|---|---|---|---|
| `Qwen3-ASR-1.7B` | 4.0GB | 识别主力 | 不能 |
| `Qwen3-ForcedAligner-0.6B` | 1.8GB | 字幕时间**精确到词** | 可以，但**不建议**（见下方说明） |

### Windows（有 NVIDIA 显卡）

```powershell
# 默认就是 qwen；-Torch cu126 对应 CUDA 12.x 驱动
# 模型默认一起下（约 6GB，可断点续传，中途 Ctrl+C 下次接着下）
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

# 想先只建环境、不下模型（第一次运行会自动下，但走 HuggingFace 会慢）
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -SkipModels
```

### Windows（没有 N 卡 / 只想先跑通流程）

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Torch cpu
```

### macOS / Linux

```bash
bash scripts/setup.sh                      # 默认 qwen + cu126，模型一起下
TORCH_INDEX=cpu bash scripts/setup.sh      # macOS 一律用 cpu
SKIP_MODELS=1 bash scripts/setup.sh        # 只建环境，不下模型
```

> **macOS 说明**：Apple Silicon 上 PyTorch 的 MPS 后端本工程还没有接（走的是 `device_map`），
> 目前按 CPU 跑。能用，但慢，建议只在短音频上用。

### 为什么对齐模型是默认装的，不是可选项

Qwen3-ASR **不输出时间戳**，字幕时间是本工程造的，有三档质量：

| 档位 | 条件 | 精度 |
|---|---|---|
| `forced-aligner` | 装了强制对齐模型 | **精确到词** |
| `vad-proportional` | 没装，但能拿到 Silero VAD 的说话区间 | 估算 |
| `even-spread` | 没装，**且**没有 VAD（只装 Qwen 的机器就是这样） | 最粗，字幕会铺到没人说话的地方 |

只装 Qwen 时没有 `faster-whisper`，也就没有 VAD，所以少了对齐模型就会直接掉到最粗的一档。
1.8GB 换"字幕能不能用"，这笔账很划算 —— 所以安装脚本默认帮你装上。
真要省这 1.8GB 也可以：`--qwen-aligner off`（结果 JSON 里的 `asr.timestamp_source`
会如实写成 `even-spread`）。

### 这个脚本做了什么

1. 建主环境 `.venv`，只装 `requirements-base.txt`（= `imageio-ffmpeg`）；
2. 建独立环境 `.venv-qwen`，用 PyTorch 官方索引装对应 CUDA 版本的 torch；
3. 装 `requirements-qwen.txt`（`qwen-asr` + `transformers==4.57.6` 等）；
4. 跑一次自检，打印解释器 / 模型 / 对齐模型是否就绪；
5. 下载识别模型与强制对齐模型（走魔搭，可断点续传）。

也可以只跑 Qwen 那部分（效果完全一样，脚本自己会把主环境补上）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_qwen.ps1
```
```bash
bash scripts/setup_qwen.sh
```

---

## 3. 路线 B：只部署 Whisper

不装 torch，磁盘占用最小，没有显卡也能跑。模型（Whisper large-v3，约 3GB）同样默认一起下。

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Engine whisper
```
```bash
# macOS / Linux
bash scripts/setup.sh whisper
```

手工装也行：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -r requirements.txt          # base + faster-whisper
.\.venv\Scripts\python -m pip install -r requirements-gpu-win.txt  # Windows + N 卡才需要
.\.venv\Scripts\python -m video2context doctor
```

Linux 上 ctranslate2 还需要系统级 CUDA 12 与 cuDNN 9：

```bash
sudo apt install libcublas-12-* libcudnn9-cuda-12    # Ubuntu 示例
```

---

## 4. 路线 C：两个都装

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Engine both
```
```bash
bash scripts/setup.sh both
```

两个引擎都可用时，默认走 **Qwen**（效果更好），随时用 `--engine faster-whisper` 换回来。
适合想在同一条视频上对比两种结果的情况：

```powershell
python -m video2context 课程.mp4 -f json,srt -n 课程-whisper --engine faster-whisper
python -m video2context 课程.mp4 -f json,srt -n 课程-qwen
```

---

## 5. 模型怎么下（三条路都要）

模型**不在仓库里**。安装脚本默认会下好（走魔搭，国内快），所以正常流程里不用管这一节。
需要单独下（离线准备、补下、或者用了 `-SkipModels`）时，用自带的**分块 + 断点续传**下载器，
它支持 HuggingFace 与魔搭（ModelScope，国内通常快 5 倍以上），可以反复跑到下完为止：

```powershell
# Qwen3-ASR（路线 A / C）—— 两个都要下，第二个决定字幕时间精度
.\.venv\Scripts\python scripts\download_model.py --repo Qwen/Qwen3-ASR-1.7B --source modelscope --out models\Qwen3-ASR-1.7B
.\.venv\Scripts\python scripts\download_model.py --repo Qwen/Qwen3-ForcedAligner-0.6B --source modelscope --out models\Qwen3-ForcedAligner-0.6B

# Whisper（路线 B / C）
.\.venv\Scripts\python scripts\download_model.py large-v3 --source modelscope --out models\faster-whisper-large-v3
```

| 模型 | 体积 | 作用 | 能不能不下 |
|---|---|---|---|
| `Qwen3-ASR-1.7B` | 4.0GB | 识别主力 | 不能（不想下这么大的话用 `models\Qwen3-ASR-0.6B`，1.6GB） |
| `Qwen3-ForcedAligner-0.6B` | 1.8GB | 给字幕**精确到词**的时间戳 | 技术上能（`--qwen-aligner off`），但**不建议**，见路线 A 的说明 |
| `faster-whisper-large-v3` | 3.1GB | Whisper 引擎 | 不能 |

`download_model.py` 只用标准库，所以主环境只装了 `imageio-ffmpeg` 也能跑它。

---

## 6. 引擎选择与切换

优先级：**`--engine` 参数 > 环境变量 `V2C_ENGINE`（或 `.env`）> 本机部署了哪个 > 首选 Qwen**。

第三档是关键：**装了什么就用什么**。只装了 Whisper 的机器默认就是 Whisper，
只装了 Qwen 的机器默认就是 Qwen —— 不需要用户记住自己的部署路线。

```powershell
python -m video2context doctor                              # 看本机部署了哪些引擎、默认用哪个
python -m video2context a.mp4 --engine faster-whisper       # 这一次用 Whisper
```

想把默认固定成某个引擎，在 `.env` 里写一行（`.env` 已在 `.gitignore` 中）：

```ini
V2C_ENGINE=faster-whisper
```

---

## 7. 怎么用

### 命令怎么敲（先看这个）

工程依赖装在专用虚拟环境 `.venv` 里，**命令有两种写法，任选一种，效果完全一样**：

| | Windows（PowerShell） | macOS / Linux |
|---|---|---|
| **写法 A**<br>直接用 venv 里的 python（不用激活，推荐） | `.\.venv\Scripts\python -m video2context 视频.mp4` | `./.venv/bin/python -m video2context 视频.mp4` |
| **写法 B**<br>先激活，之后 `python` 就指向它 | `.\.venv\Scripts\Activate.ps1`<br>然后 `python -m video2context 视频.mp4` | `source .venv/bin/activate`<br>然后 `python -m video2context 视频.mp4` |

激活成功后提示符会从 `(base)` 之类变成 **`(.venv)`**；退出用 `deactivate`。

> ⚠️ **直接敲 `python`（既没激活、也没写 venv 路径）= 跑到了系统或 conda 的 Python 上**，
> 会报「未找到 ffmpeg」。这是本工程最常见的坑 ——
> 详见 [troubleshooting.md 1.5](troubleshooting.md#15-敲-python-报未找到-ffmpeg缺少-faster-whisper-多半是跑错了-python)。

**本节后面的示例为了简短统一写成 `python -m video2context ...`，
请按上面的写法 A / B 替换。** 先跑一次自检可以确认环境对不对：

```powershell
.\.venv\Scripts\python -m video2context doctor
```

### 先拿仓库自带的样例试一条

装完别急着找自己的视频 —— 仓库里带了 4 个可以直接跑的样例（详见 [samples/README.md](../samples/README.md)）：

```powershell
# 42 秒中文数学课样例，出全套产物
python -m video2context samples\demo_zh_math.mp4 --language zh -f json,txt,srt,vtt

# 40 秒英文样例（术语密集），想对比两个引擎就用这个
python -m video2context samples\demo_en_ml.mp4 --language en -f json,txt
```

每个 `demo_*` 都配了逐句稿子 `demo_*.transcript.txt`，可以对着看识别得对不对；
跑通了再换自己的视频。

### 最常用

```powershell
# 最简：转写一条视频，出 JSON
python -m video2context 课程.mp4

# 出字幕（SRT / VTT）和纯文本
python -m video2context 课程.mp4 -f json,srt,vtt,txt

# 明确指定语言：更快更准，减少小语种幻觉
python -m video2context 课程.mp4 --language zh

# 批量：一个目录下所有 mp4（模型只加载一次）
python -m video2context .\videos\*.mp4 -o .\output

# 长视频先试跑前 10 分钟
python -m video2context long.mp4 --duration 600

# 逐句实时打印
python -m video2context a.mp4 --stream
```

结果默认落在 `output\视频名.json`（同名的加 `-1` 避让，或加 `--overwrite` 覆盖）。

### 网页界面（拖拽即用）

```powershell
python -m video2context webui          # 浏览器打开 http://127.0.0.1:7860
```

页面左侧上传视频、选**识别引擎**/模型/语种，右侧是实时进度 + 全文 + 分段表格 + JSON 下载。
默认只监听本机；局域网访问加 `--host 0.0.0.0`（注意页面本身没有鉴权）。

### 在 Python 里调用

```python
from video2context import process, TranscribeOptions

result = process("课程.mp4", TranscribeOptions(language="zh", formats=("json", "srt")))
print(result.text)
print(result.outputs["srt"])
print(result.payload["asr"]["engine"], result.payload["asr"]["timestamp_source"])
```

> `timestamp_source` 标明字幕时间来自哪一档：`forced-aligner`（词级精确）/
> `vad-proportional`（按说话区间估算）/ `even-spread`（整条音轨平均分，最粗）/
> `none`。**只装 Qwen 的机器**没有 faster-whisper 也就没有 VAD，
> 不装对齐模型时会落到 `even-spread` —— 所以这条路线建议把对齐模型一起装上。

`process()` 是无状态函数（模型由内部缓存管理），可以直接放进任务队列。
多进程并发时每个进程会各自加载一份模型，显存是叠加的 —— 别一下子开好几个。

### 环境自检

```powershell
python -m video2context doctor                       # ffmpeg / 依赖 / 显卡 / 部署了哪些引擎
python -m video2context --engine qwen3-asr --qwen-setup   # 只看 Qwen 那一侧
```

---

## 8. 各设备上的实测速度

测试机：**i9-13900HX（32 线程）+ RTX 4060 Laptop 8GB + 15.7GB 内存**。

| 引擎 / 模型 | 设备 | 素材 | 推理耗时 | 相对实时 |
|---|---|---|---|---|
| Qwen3-ASR-1.7B | GPU bf16 | `demo_zh_math.mp4`（41.8s 中文） | 5.5s | **7.6×** |
| Qwen3-ASR-1.7B | GPU bf16 | `demo_en_ml.mp4`（40.4s 英文） | 6.2s | **6.5×** |
| Qwen3-ASR-1.7B | GPU bf16 | `demo_zh_daily.mp4`（29.3s 中文） | 4.3s | **6.9×** |
| Qwen3-ASR-1.7B | **CPU**（32 线程） | 英文 12s 片段 | 23.9s | **0.5×**（比实时慢一倍） |
| faster-whisper large-v3 | GPU float16 + 批量 | 中文 390s | 16～19s | **20～24×** |
| faster-whisper small | CPU int8 | 中文 390s | 64s | 6.1× |

> `demo_*` 三个素材都在仓库里，命令见[第 7 节](#7-怎么用)，可以自己复现。
> 「相对实时」= 处理 1 秒音频要多久的倒数。7.6× 表示 1 小时视频约 8 分钟；
> 0.5× 表示 1 小时视频要 2 小时。
> Qwen 还要额外加一次模型加载（GPU 上 6.6～10.9s），音频越长越无所谓。
> CPU 数据强依赖核数，上表是 32 线程的成绩；4 核笔记本要再慢 3～5 倍。
> 完整的质量对比（中文同音词、英文 WER）见 [models.md](models.md#6-qwen3-asr默认引擎)。

---

## 9. 离线部署（内网机器 / 不能联网）

在**有网**的机器上把环境和模型都准备好，然后整目录拷过去：

```powershell
# 有网机器：装依赖 + 下模型（都在工程目录内）
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Engine both

# 额外把 pip 包也导出成离线 wheels（可选，目标机器不能连 pip 源时用）
.\.venv\Scripts\python -m pip download -r requirements.txt -r requirements-web.txt -d wheels
.\.venv-qwen\Scripts\python -m pip download -r requirements-qwen.txt -d wheels-qwen
```

把整个工程目录（含 `.venv`、`.venv-qwen`、`models`、`wheels*`）拷到目标机器，然后：

```powershell
# 目标机器：路径变了要重建虚拟环境（venv 里写死了绝对路径）
python -m venv .venv
.\.venv\Scripts\python -m pip install --no-index --find-links wheels -r requirements.txt
python -m venv --without-pip .venv-qwen
python -m pip --python .\.venv-qwen\Scripts\python.exe install --no-index --find-links wheels-qwen -r requirements-qwen.txt

# 让程序只用本地缓存、不联网
$env:HF_HUB_OFFLINE = "1"
.\.venv\Scripts\python -m video2context 视频.mp4
```

**只想搬模型**（依赖能连网）时更简单：把 `models\` 整个目录拷过去即可，
程序会优先用工程内的本地模型目录。

---

## 10. 国内网络加速

```powershell
# pip 走清华源（安装脚本已经默认带上了 -i 参数）
.\.venv\Scripts\python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 模型走魔搭（比 HuggingFace 快很多，支持断点续传）
.\.venv\Scripts\python scripts\download_model.py large-v3 --source modelscope
```

代码里默认关掉了 HuggingFace 的 Xet 存储（弱网下会长时间挂起）；
需要时可以设 `HF_HUB_DISABLE_XET=0` 打开。

---

## 11. 磁盘占用与卸载

| 内容 | 体积 | 怎么删 |
|---|---|---|
| 主环境 `.venv` | 0.9GB（路线 B 含 Whisper）/ 0.05GB（只装 Qwen） | 直接删目录 |
| Qwen 环境 `.venv-qwen` | 约 3GB | 直接删目录 |
| Qwen 模型 | 4.0GB + 对齐模型 1.8GB | 删 `models\` 下对应目录 |
| Whisper 模型 | 3.1GB | 删 `models\faster-whisper-large-v3` |
| HuggingFace 缓存 | 视情况 | `%USERPROFILE%\.cache\huggingface`（可用 `HF_HOME` 改位置） |
| 输出与临时文件 | 视情况 | `output\`；转写时的临时 wav 会自动清理 |

用 `--model-dir` 指定的本地模型目录可以随时删，程序不会自动往那里写东西。

---

## 12. 部署出问题？

先跑自检，它会把"缺哪一环"直接打出来：

```powershell
python -m video2context doctor
```

| 症状 | 去哪看 |
|---|---|
| `找不到 Qwen3-ASR 的独立环境（.venv-qwen）` | [troubleshooting 7.1](troubleshooting.md#71-找不到-qwen3-asr-的独立环境venv-qwen) |
| `页面文件太小` / `CUDA out of memory` | [troubleshooting 7.2](troubleshooting.md#72-页面文件太小无法完成操作--cuda-out-of-memory--0xc0000005) |
| Qwen 跑得很慢 / 自检显示 `cuda False` | [troubleshooting 7.3](troubleshooting.md#73-环境自检显示-设备cpu--速度特别慢)（多半是 torch 装成了 CPU 版） |
| 字幕时间戳不准 | [troubleshooting 7.4](troubleshooting.md#74-字幕时间戳不准--所有段落挤在一起) |
| 模型下载卡住 / 很慢 | [troubleshooting 3.1](troubleshooting.md#31-下载卡住进度条不动) |
| 网页界面打不开 | [troubleshooting 6.3](troubleshooting.md#63-网页界面打不开--端口被占用) |
| 其它（安装、ffmpeg、显卡、识别质量） | [troubleshooting.md](troubleshooting.md) 全文 |
