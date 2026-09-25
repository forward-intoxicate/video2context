# 疑难排查手册

按**症状**查找。每条都给出：快速判断方法 → 解决办法 → 为什么会这样。
如果这里没覆盖，提 Issue 时请附上 `python -m video2context doctor` 的输出。

---

## 0. 先跑自检

```powershell
python -m video2context doctor
```

正常输出长这样：

```
video2context 0.1.0 环境自检
  Python            : 3.13.9  (E:\...\.venv\Scripts\python.exe)
  平台              : Windows-11-10.0.26200-SP0
  ffmpeg 路径        : ...\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe
  ffmpeg 版本        : ffmpeg version 7.1-essentials_build-www.gyan.dev ...
  faster_whisper    : 1.2.1
  ctranslate2       : 4.8.2
  imageio_ffmpeg    : 0.6.0
  gradio            : 6.28.0
  CUDA 设备数        : 1
  CUDA 运行库目录    : ...\site-packages\nvidia\cublas\bin, ...\nvidia\cudnn\bin
  模型缓存目录       : C:\Users\<你>\.cache\huggingface
```

**`CUDA 设备数` 是 0 或 `CUDA 检测失败`** → 说明只能走 CPU（能跑，就是慢）。
**缺哪个包** → 按提示 `pip install`。

---

## 1. 安装阶段

### 1.1 `python -m venv .venv` 之后没有 pip / `ensurepip` 报权限错误

某些 Anaconda 发行版、以及受限的临时目录环境会出现。三种走法：

```powershell
# 走法 A：让 venv 先不带 pip，再用基础环境的 pip 装进去
python -m venv --without-pip .venv
python -m pip --python .venv\Scripts\python.exe install --upgrade pip

# 走法 B：直接用 conda 建环境（推荐 Anaconda 用户）
conda create -n v2c python=3.11 -y
conda activate v2c
pip install -r requirements.txt

# 走法 C：不建虚拟环境，直接装到当前环境（不推荐，但最快）
pip install -r requirements.txt
```

Linux/macOS 上如果报 `ensurepip is not available`，先装 `python3-venv`：
`sudo apt install python3-venv`。

### 1.2 `pip install` 卡住不动 / 极慢

国内网络默认走 pypi.org，可能超时。换镜像：

```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 1.3 提示 `No matching distribution found for gradio`

Gradio 6.x 要求 **Python ≥ 3.10**。
要么升级 Python，要么只装核心依赖用命令行（`requirements.txt` 不含 gradio）。

### 1.4 `ImportError: DLL load failed` / `No module named 'ctranslate2'`

- `ctranslate2` 装失败通常是 Python 版本太新或平台没有对应 wheel：
  用 3.11～3.13 的官方 CPython（不是某些精简版发行版）；
- Windows 上如果装了 `nvidia-cudnn-cu12` 却仍报 DLL 缺失，见 [2.2](#22-报错-cudnn64_9dll--cublas64_12dll-找不到)。

---

## 2. ffmpeg 与显卡

### 2.1 `未找到 ffmpeg` / `提取音频失败`

按顺序检查：

1. `pip show imageio-ffmpeg` 是否已安装（它是默认的 ffmpeg 来源）；
2. 想用系统 ffmpeg：设环境变量 `FFMPEG_BIN=D:\ffmpeg\bin\ffmpeg.exe`；
3. 验证：`python -c "from video2context.ffmpeg_tools import ffmpeg_version; print(ffmpeg_version())"`。

### 2.2 报错 `cudnn64_9.dll` / `cublas64_12.dll` 找不到

**原因**：ctranslate2 的 Windows 版本不打包 CUDA 运行库，需要 cuBLAS 12 + cuDNN 9。

**解决**：

```powershell
pip install -r requirements-gpu-win.txt      # 约 1.5GB
```

本工程在加载模型前会自动把 `site-packages\nvidia\{cublas,cudnn}\bin` 注册进 DLL 搜索路径
（`os.add_dll_directory` + 追加 `PATH`），装完直接可用，不需要手动改环境变量。

**Linux**：用发行版或 conda 装 CUDA 12 + cuDNN 9：

```bash
sudo apt install libcublas-12-* libcudnn9-cuda-12
# 或
conda install -c conda-forge cudnn=9 cuda-toolkit=12
```

### 2.3 `CUDA 设备数` 显示 1，但模型加载仍然失败

说明驱动能看见卡，但运行库不全或版本不匹配。看报错文本：

- `cudnn` / `cublas` 相关 → 见 2.2；
- `CUDA driver version is insufficient` → 显卡驱动太旧，升级 NVIDIA 驱动（CUDA 12 需要驱动 ≥ 525，建议 550+）；
- `no kernel image is available` → 显卡算力过低（如 Kepler 老卡），只能用 CPU。

程序在 `--device auto` 下会自动退到 CPU 继续跑，日志里能看到 `device=cpu, compute_type=int8`。

### 2.4 显存或内存不足（CUDA out of memory / MemoryError）

> **先分清是哪种不足**：模型要先读进**系统内存**再上显卡，所以系统内存不够时
> 报的也常常是 `CUDA failed with error out of memory` 或 `MemoryError`。
> 打开任务管理器看「可用内存」，只有 4GB 上下就很可能是内存问题，而不是显存。

**程序已经做了两层保护**：

1. **推理前预检**剩余显存并自动收缩批量大小，日志形如
   `剩余显存约 3976MB → 批量大小 8 收缩为 4`；放不下就直接改用逐段模式；
2. **推理中 OOM** 会自动降级重试；若 CUDA 上下文已损坏（`invalid device ordinal`），
   会直接抛出带命令建议的错误，不再无意义重试。

手动解决优先级：

```powershell
# 1) 最省资源：关批量 + 半精度 int8
python -m video2context a.mp4 --batch-size 0 --compute-type int8_float16

# 2) 还不行：走 CPU（慢但一定成功）
python -m video2context a.mp4 --device cpu --compute-type int8 --model small

# 3) 换小模型
python -m video2context a.mp4 --model medium
```

**最容易被忽略的两个原因**：

- **同时跑了两个任务**（网页界面开着一个 `large-v3`，又用命令行跑一个）：显存和内存都是成倍占用，先关掉一个；
- **系统内存本身就不够**：`large-v3` 大约需要 4GB 以上空闲内存。浏览器多标签、网盘、QQ
  这类常驻程序很容易吃掉几个 GB，关掉就能过。

各模型的大致资源需求：`small` 约 2GB 显存、`medium`/`large-v3` 约 5GB 显存
（`int8_float16` 可再省一半）。

### 2.5 我到底有没有在跑 GPU？

看日志里这行：`模型就绪：device=cuda, compute_type=float16` →
`cuda` 就是 GPU；`device=cpu` 就是 CPU。JSON 的 `asr.device` 字段同样记录了实际设备。

---

## 3. 模型下载

### 3.1 下载卡住、进度条不动

**原因**：新版 `huggingface_hub` 默认走 **Xet** 存储（`cas-server.xethub.hf.co`），
部分网络下会**无响应地挂起**（不是超时，所以不会自动报错）。

**本工程已默认关闭 Xet**（`HF_HUB_DISABLE_XET=1`），所以正常安装的版本不会踩到。
如果你手动设置过 `HF_HUB_DISABLE_XET=0` 想打开，卡了就先删掉这个变量。

### 3.2 下载很慢

```powershell
# 方案 1：走镜像
python -m video2context a.mp4 --hf-mirror

# 方案 2（推荐）：用自带的分块断点续传下载器，国内实测快 5 倍以上
python scripts\download_model.py large-v3 --source modelscope
python -m video2context a.mp4 --model-dir models\faster-whisper-large-v3
```

下载器特性：8MB 分块、每块失败自动重试、可**反复执行续传**、
支持 `--source hf` 与 `--source modelscope` 两个源、`--repo` 可指定任意仓库。

### 3.3 想彻底离线

```powershell
# 先在有网的机器上下好
python scripts\download_model.py large-v3 --source modelscope --out models\faster-whisper-large-v3
# 把 models 目录整体拷到离线机器，然后
python -m video2context a.mp4 --model-dir models\faster-whisper-large-v3 --local-files-only
```

`--local-files-only` 会禁止任何联网尝试。

### 3.4 模型缓存占了多少磁盘 / 能删吗

默认在 `%USERPROFILE%\.cache\huggingface`（Linux/macOS 是 `~/.cache/huggingface`）。
可以整目录删除，下次会重新下载；用 `--model-dir` 的本地模型目录也可随时删。

---

## 4. 识别质量

### 4.1 出现重复句子、莫名其妙的句子（幻觉）

Whisper 的典型问题，多发生在**静音、纯音乐、强背景噪声**片段。按顺序试：

1. 保持 VAD 开启（默认就是开的，除非你加了 `--no-vad`）；
2. 明确指定语种：`--language zh` / `--language en`（不指定时模型可能"听成"别的语言）；
3. 打开上文条件反而更容易循环，确保**没有**加 `--condition-on-previous-text`；
4. 提高束搜索：`--beam-size 8`；
5. 后处理：JSON 里 `segments[].no_speech_prob` 高、`avg_logprob` 低（如 < -1.0）的段可直接丢弃。

### 4.2 专有名词、人名地名总是错

这是最值得投入的一项优化——**用 `--initial-prompt`**：

```powershell
python -m video2context 讲座.mp4 --initial-prompt "涉及概念：卷积神经网络、反向传播、Transformer。人名：吴恩达。"
```

实测（见 [samples/README.md](../samples/README.md)）：一段人名地名的录音，加提示词后
错误写法出现次数从每次 1～5 次降到 **0**，正确写法从 0 次升到 9 次。

### 4.3 中文标点不好 / 输出繁体

- `tiny`/`base` 模型会输出**繁体**且错字多 —— 这不是 bug，换 `large-v3` 即可，实测输出简体；
- `large-v3` 的标点是模型行为，句子之间偶尔缺句号属正常；
- 对中文标点要求高 → 换 FunASR/SenseVoice（带独立标点模型），见 [models.md](models.md)。

### 4.4 时间戳偏移 / 对不上画面

- 时间戳是相对**输入文件**计算的；用 `--start` 裁剪时原始起点记录在 JSON 的 `source.clip` 里；
- 源文件若是**可变帧率（VFR）**（手机录屏、部分下载视频常见），时间轴可能漂移，
  建议先转成固定帧率：`ffmpeg -i in.mp4 -vsync cfr -r 30 out.mp4`；
- 想校正字幕：`--word-timestamps` 能拿到词级时间戳，便于二次对齐。

### 4.5 英文视频识别出来是别的语言

`--language en` 明确指定。自动检测在开头几秒是音乐/静音时容易判错。

---

## 5. 性能

### 5.1 太慢了

| 情况 | 处理 |
|---|---|
| 在 CPU 上跑 `large-v3` | 换 `--model small --compute-type int8`，或上显卡 |
| GPU 没被用上 | 看日志 `device=` 那行；`device=cpu` 说明退回了 CPU，见第 2 节 |
| 首次运行特别慢 | 正常：要下载模型 + CUDA 预热。第二次就快了 |
| 长视频想先看效果 | `--duration 600` 只跑前 10 分钟 |
| 想更快 | `--model large-v3-turbo`（约 4 倍速）、`--beam-size 1`、GPU 上加 `--batch-size 16` |

### 5.2 CPU 上 `--batch-size` 没效果

批量推理只在 CUDA 上启用（CPU 上批量并不会更快）。`--batch-size -1`（默认）会自动判断。

### 5.3 磁盘占用

转写过程中会在临时目录生成 16kHz 单声道 wav（1 小时约 115MB），结束后自动删除；
`--keep-audio` 会保留到输出目录。

---

## 6. 使用与集成

### 6.1 输出文件被改名成 `xxx-1.json`

同名结果已存在时自动避让，避免覆盖。想覆盖用 `--overwrite`，想指定名字用 `--name`。

### 6.2 批量处理时每个文件都重新加载模型

不会。`get_transcriber()` 会缓存最近一次的模型（批量循环里只加载一次）。
但**跨进程**不共享：如果你写脚本对每个文件都起一个 `python -m video2context`，
那就是每次都加载。批量请用一次调用传多个文件，或直接 `from video2context import process`。

### 6.3 网页界面打不开 / 端口被占用

```powershell
python -m video2context webui --port 8000        # 换端口
python -m video2context webui --host 0.0.0.0     # 允许局域网访问（注意无鉴权）
```

`--share` 生成的公网链接走 Gradio 官方隧道，页面**没有权限控制**，切勿用于敏感视频。

### 6.4 作为库调用

```python
from video2context import process, TranscribeOptions

result = process(
    "video.mp4",
    TranscribeOptions(model="large-v3", language="zh", formats=("json", "srt")),
    progress_callback=lambda done, total, text: print(f"{done:.0f}/{total:.0f}s"),
)
print(result.text)
print(result.outputs["srt"])
```

### 6.5 想接进自己的流水线 / 队列

`process()` 是无状态纯函数（模型除外，由内部缓存管理），可以直接在任务队列里调用。
多进程并发时请注意：每个进程都会各自加载一份模型，显存是叠加的。
