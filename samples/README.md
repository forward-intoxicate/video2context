# 测试样例

仓库里有 4 个可以直接跑的样例，覆盖不同场景：

| 文件 | 时长 | 语言 | 内容 | 适合看什么 |
|---|---|---|---|---|
| `demo_zh_math.mp4` | 42s | 中文 | 导数运算法则（数学课风格，含 `u`/`v` 这类符号） | **默认引擎的强项**：标点、术语、中英混排 |
| `demo_en_ml.mp4` | 40s | 英文 | 梯度下降（`Adam`、`RMSProp` 等术语密集） | 英文识别、**两个引擎的 WER 对比** |
| `demo_zh_daily.mp4` | 29s | 中文 | 日常口语（短句多、停顿多） | VAD 与字幕分段 |
| `chinese_speech_sample.mp4` | 6:30 | 中文 | 真人朗读鲁迅《鸭的喜剧》 | 长音频、真实人声、专有名词与提示词效果 |

前三个是**机器合成语音**（见下文「怎么生成的」），第四个是**公共领域真人录音**。
想快速验证流程用前三个（几十秒就出结果），想看真实效果用第四个。

---

## 跑一下看看

下面的命令为了简短写成 `python -m video2context ...`。
**实际要换成工程虚拟环境里的 python**，两种写法任选一种：

| | Windows（PowerShell） | macOS / Linux |
|---|---|---|
| **写法 A**<br>直接用 venv 里的 python（不用激活，推荐） | `.\.venv\Scripts\python -m video2context 视频.mp4` | `./.venv/bin/python -m video2context 视频.mp4` |
| **写法 B**<br>先激活，之后 `python` 就指向它 | `.\.venv\Scripts\Activate.ps1`<br>然后 `python -m video2context 视频.mp4` | `source .venv/bin/activate`<br>然后 `python -m video2context 视频.mp4` |

⚠️ 直接敲 `python`（既没激活、也没写 venv 路径）= 跑到了系统或 conda 的 Python 上，
会报「未找到 ffmpeg」。详见
[docs/troubleshooting.md 1.5](../docs/troubleshooting.md#15-敲-python-报未找到-ffmpeg缺少-faster-whisper-多半是跑错了-python)。

举例 —— 第一条命令的两种真实写法：

```powershell
# 写法 A（Windows，推荐）
.\.venv\Scripts\python -m video2context samples\demo_zh_math.mp4

# 写法 B：先激活一次，之后下面的 python 都成立
.\.venv\Scripts\Activate.ps1
python -m video2context samples\demo_zh_math.mp4
```

```powershell
# ① 最简：一条命令出文字（默认引擎，自动选 Qwen 或 Whisper）
python -m video2context samples\demo_zh_math.mp4

# ② 出全套产物：JSON + 纯文本 + SRT + VTT
python -m video2context samples\demo_zh_math.mp4 --language zh -f json,txt,srt,vtt

# ③ 日常口语样例：主要看分段和字幕时间
python -m video2context samples\demo_zh_daily.mp4 --language zh -f json,srt

# ④ 只跑前 10 秒，调参数时省时间；--stream 会把每句实时打出来
python -m video2context samples\demo_zh_math.mp4 --duration 10 --stream

# ⑤ 英文样例：两个引擎各跑一遍，然后用 WER 工具比准确率
python -m video2context samples\demo_en_ml.mp4 --language en -f json,txt -n en-qwen
python -m video2context samples\demo_en_ml.mp4 --language en -f json,txt -n en-whisper `
    --engine faster-whisper --model-dir models\faster-whisper-large-v3
python experiments\wer_report.py samples\demo_en_ml.transcript.txt output\en-qwen.txt output\en-whisper.txt

# ⑥ 打开网页界面，把 mp4 拖进去
python -m video2context webui
```

**结果在哪**：默认落在 `output\样例名.json`（加了 `-n` 的按你给的名字）。
SRT / VTT 可以直接丢进播放器或剪辑软件；JSON 里有分段、时间戳、语种和实际生效的参数。

**怎么对答案**：每个 `demo_*.mp4` 都配了 `demo_*.transcript.txt`，里面是逐句稿子 ——
和识别结果对着看就知道哪里错了。英文那份还能直接喂给 `experiments/wer_report.py` 算 WER。

### 识别结果和稿子对不上？先看差在哪一类

实测下来，**默认引擎（`qwen3-asr`）的差异基本都是标点级的，没有词错**。
本机跑三个样例的真实对照：

| 样例 | 差异 |
|---|---|
| `demo_zh_math` | 3 处：把「导数，就等于」的逗号省了、「前提，是」省了、「u 一 加 u 二」的空格并了。**字全对** |
| `demo_en_ml` | 1 处：`At each step we compute` 被断成两句（多一个逗号并切成两段）。**词全对，WER 0.00%** |
| `demo_zh_daily` | 4 处：句末标点在 `。` `？` `，` 之间跳（「…去哪儿呢。」→「…去哪儿呢？」）。**字全对** |

所以看到 `？` 被写成 `。`、逗号多一个少一个，**不是坏了** —— 标点是模型自己"顺手"生成的，
不是后处理规则，本来就是概率性的。真正该盯的是**同音字/专有名词错**，那才是质量问题。

**换成 `--engine faster-whisper` 跑同一个中文样例，就能看到真正的质量差异**：

| 稿子 | `qwen3-asr`（默认） | `faster-whisper` large-v3 |
|---|---|---|
| 同样可以逐个**求导**，再把结果相加。 | 同样可以逐个**求导**，再把结果相加。 | 同样可以逐个**球倒**，再把结果相加。 |

「球倒」就是同音词错误 —— 这正是默认引擎要解决的问题。复现（需要装了 Whisper 引擎）：

```powershell
.\.venv\Scripts\python -m video2context samples\demo_zh_math.mp4 --language zh -f txt `
    --engine faster-whisper --model-dir models\faster-whisper-large-v3
```

英文样例上两个引擎的差距更量化：**WER 0.00% vs 4.35%**。

> 这一批样例是**机器合成语音**，发音标准、没有口音和噪声，所以词错本来就少 ——
> 它适合验证流程和命令，不适合评估真实效果。真人素材上差得更明显
> （`u(x)`→「右F4」这类同音错）见 [docs/models.md](../docs/models.md#6-qwen3-asr默认引擎)。

---

## 三个合成样例详解

### `demo_zh_math.mp4` · 中文数学课

稿子（也就是画面上一句句显示的文字）：

```
好，我们来看导数的运算法则。
如果 u 和 v 这两个函数在 x 零处都可导，
那么 u 加 v 的导数，就等于 u 先求导，再加上 v 求导。
注意这个公式的前提，是两个函数都已经可导，才能这样拆开。
对于多个函数相加的情况，比如说 u 一 加 u 二，一直加到 u n，
同样可以逐个求导，再把结果相加。
这是一个非常常用的公式，大家一定要记牢。
```

本机实测（RTX 4060 Laptop 8GB，Qwen3-ASR-1.7B + 强制对齐模型）：

| 指标 | 结果 |
|---|---|
| 时长 | 41.8s |
| 分段 | **7**（正好对应稿子的 7 句） |
| 推理耗时 | 5.5s（约 7.6 倍实时） |
| 字幕时间来源 | `forced-aligner`（精确到词） |
| 识别结果 | 与稿子几乎逐字一致，标点齐全 |

实测字幕（和画面上的句子是对齐的）：

```
1
00:00:00,080 --> 00:00:03,200
好，我们来看导数的运算法则。

2
00:00:04,560 --> 00:00:08,480
如果 u 和 v 这两个函数在 x 零处都可导，
```

> 为什么这个样例值得看：中英混排（`u`、`v`、`x` 这些拉丁字母夹在中文里）
> 是中文 ASR 最容易翻车的地方。想体会差别，用 `--engine faster-whisper` 再跑一遍对比。

### `demo_en_ml.mp4` · 英文梯度下降

时长 40.4s，稿子见 `demo_en_ml.transcript.txt`（92 个词），含 `gradient descent`、`Adam`、
`RMSProp`、`Adagrad` 这些专有术语。

| 引擎 | 本机实测 WER |
|---|---|
| `qwen3-asr`（默认） | **0.00%**（分段 8 段，推理 6.2s） |
| `faster-whisper` large-v3 | **4.35%**（`Adam`→`Atom`、`RMSProp`→`RMS Prop`、`Adagrad`→`Atigrid`） |

用「跑一下看看」里的第 ⑤ 组命令可以自己复现这张表。

### `demo_zh_daily.mp4` · 中文日常口语

时长 29.3s，短句多、停顿多，专门用来看 **VAD 切分**和字幕分句：6 句话 → 6 段
（推理 4.3s）。把语音换成 `--no-vad` 跑一次，能直观看到静音过滤对分段的影响。

---

## 怎么生成的（`scripts/make_samples.py`）

示例视频不能随便从网上拿 —— 课程录像、播客都有版权。所以这三个的做法是：

**用本机自带的语音合成念我们自己写的稿子，再和纯色背景合成视频。**

* 全程**离线**，不下载任何第三方素材；
* 语音：Windows 用系统自带的 SAPI（中文 Microsoft Huihui / 英文 Microsoft Zira），
  macOS 用 `say`，Linux 用 `espeak-ng`；
* 画面：每句一个 `drawtext`，用时间戳控制显示时机 —— **说到哪句就显示哪句**，
  边听边看就能发现识别错在哪；
* 逐句合成再按 0.5～0.7 秒停顿拼起来，这样 VAD 和字幕切分才有东西可切。

脚本就在仓库里，随时可以重新生成或改稿子：

```powershell
python scripts\make_samples.py                       # 全部重新生成
python scripts\make_samples.py --only demo_en_ml     # 只生成一个
python scripts\make_samples.py --keep-wav            # 保留中间 wav，便于排查
```

改稿子就编辑脚本里的 `SAMPLES`；改完重跑，`.transcript.txt` 会跟着更新。

> ⚠️ 合成语音是用来**试流程**的，不能替代真实效果评估：真人讲课有口音、语速变化、
> 背景噪声，而同音词错误恰恰来自这些。真实素材上的对比数据见
> [docs/models.md](../docs/models.md#6-qwen3-asr默认引擎)。

---

## `chinese_speech_sample.mp4` · 真人中文长音频

一段 **6 分 30 秒的真实中文人声**测试视频，用来验证/演示本工程的「视频 → 文字」全流程。
不是机器合成音，是真人朗读，所以能真实反映识别效果。

### 出处与许可

| 项 | 内容 |
|---|---|
| 内容 | 鲁迅《呐喊》中的《鸭的喜剧》（1922 年作） |
| 朗读 | Jing Li（李晶），普通话 |
| 来源 | [LibriVox · 呐喊 / Call to Arms](https://archive.org/details/call_to_arms_jl_librivox)（archive.org） |
| 原始音频 | `calltoarms_15_lu_64kb.mp3`（6:30，22.05kHz 单声道 64kbps，3.1MB） |
| 录音年份 | 2012 |
| 许可 | **CC Public Domain Mark 1.0（公共领域）** — 鲁迅原文与 LibriVox 录音均为公共领域，可自由使用 |
| 视频画面 | 由该条目封面图 + 音频合成（仅为方便当作「视频」测试，画面内容无实际意义） |

### 目录内容

| 文件 | 说明 |
|---|---|
| `chinese_speech_sample.mp4` | 测试视频：640×360，6:30，3.4MB，H.264 + 原 MP3 音轨 |
| `chinese_speech_sample.json` | 参考输出（large-v3 跑出来的结构化结果，含 17 段与时间戳） |
| `chinese_speech_sample.txt` | 参考纯文本 |
| `chinese_speech_sample.srt` | 参考字幕（播放器 / 剪辑软件） |
| `chinese_speech_sample.vtt` | 参考字幕（HTML5 `<track>`，网页播放器直接用） |

### 怎么用

```powershell
# 命令行（用本地模型，离线）
.\.venv\Scripts\python -m video2context samples\chinese_speech_sample.mp4 `
    --model-dir models\faster-whisper-large-v3 -f json,srt,txt -o output

# 网页界面：启动后把 mp4 拖进去即可
.\.venv\Scripts\python -m video2context webui --model-dir models\faster-whisper-large-v3

# 单元测试里也会用到 ffmpeg 现造素材，不依赖这个文件
.\.venv\Scripts\python -m unittest discover -s tests -v
```

### 参考结果（本机实测）

环境：RTX 4060 Laptop 8GB，`large-v3` + `float16`，VAD 开启，批量推理（批量大小由剩余显存自动定）。

| 指标 | 结果 |
|---|---|
| 媒体时长 | 390.5s（6:30） |
| VAD 后有效语音 | 353.6s |
| 语言检测 | `zh`，置信度 0.994 |
| 分段数 | 17 |
| 推理耗时 | 16.2～19.1s（**约 20～24 倍实时**，随批量大小波动） |
| 模型加载 | 约 5s（本地模型目录） |

识别整体可用，主要误差集中在**专有名词**上（人名、地名），这是 Whisper 的典型表现。
`--initial-prompt` 能明显改善，实测对比：

```powershell
--initial-prompt "鲁迅呐喊中的鸭的喜剧。人物：爱罗先珂、仲密、仲密夫人。词语：蝌蚪、蛤蟆、小鸭、荷花池、缅甸。"
```

| 词条 | 不加提示词 | 加提示词 |
|---|---|---|
| 爱罗先珂（人名，正确写法） | 0 | **9** |
| 鸭的喜剧（篇名） | 0 | **2** |
| 缅甸（地名） | 0 | **1** |
| 仲密（人名） | 2 | **8** |
| 错误写法「爱罗仙科」 | 1 | **0** |
| 错误写法「埃罗先科」 | 1 | **0** |
| 错误写法「艾罗仙科」 | 1 | **0** |
| 错误写法「鸭的戏剧」 | 2 | **0** |

第一段对比：

```
不加提示词：鲁迅小说集《呐喊》,鸭的戏剧,此次LibriVox录音由公众所有,俄国的盲诗人艾罗仙科军带了…
加提示词　：鲁迅小说集《呐喊》。鸭的喜剧。此次LibriVox录音由公众所有。俄国的盲诗人爱罗先科军带了…
```

结论：**领域词汇/人名地名多的视频，务必用 `--initial-prompt` 喂进去。**

## 其它可选中文素材（同样公共领域）

如果这段不够，可以在 archive.org 上换别的 LibriVox 中文条目，例如：

- `digongan_2406_librivox`（狄公案，2024 年录制，音质更好，每节约 12 分钟）
- `cowherd_and_weaver_girl_2007_librivox`（牛郎织女，每节 7～11 分钟）
- `changshengdian_2502_librivox`（长生殿）

下载任意一节的方式（把标识符和文件名换成上面对应的即可）：

```powershell
curl.exe -L -o test.mp3 "https://archive.org/download/call_to_arms_jl_librivox/calltoarms_15_lu_64kb.mp3"
.\.venv\Scripts\python -m video2context test.mp3 -f json,txt -o output
```
