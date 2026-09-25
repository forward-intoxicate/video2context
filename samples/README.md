# 测试样例 · 中文语音视频

`chinese_speech_sample.mp4` —— 一段 **6 分 30 秒的真实中文人声**测试视频，用来验证/演示本工程的
「视频 → 文字」全流程。不是机器合成音，是真人朗读，所以能真实反映识别效果。

## 出处与许可

| 项 | 内容 |
|---|---|
| 内容 | 鲁迅《呐喊》中的《鸭的喜剧》（1922 年作） |
| 朗读 | Jing Li（李晶），普通话 |
| 来源 | [LibriVox · 呐喊 / Call to Arms](https://archive.org/details/call_to_arms_jl_librivox)（archive.org） |
| 原始音频 | `calltoarms_15_lu_64kb.mp3`（6:30，22.05kHz 单声道 64kbps，3.1MB） |
| 录音年份 | 2012 |
| 许可 | **CC Public Domain Mark 1.0（公共领域）** — 鲁迅原文与 LibriVox 录音均为公共领域，可自由使用 |
| 视频画面 | 由该条目封面图 + 音频合成（仅为方便当作「视频」测试，画面内容无实际意义） |

## 目录内容

| 文件 | 说明 |
|---|---|
| `chinese_speech_sample.mp4` | 测试视频：640×360，6:30，3.4MB，H.264 + 原 MP3 音轨 |
| `chinese_speech_sample.json` | 参考输出（large-v3 跑出来的结构化结果，含 17 段与时间戳） |
| `chinese_speech_sample.txt` | 参考纯文本 |
| `chinese_speech_sample.srt` | 参考字幕（播放器 / 剪辑软件） |
| `chinese_speech_sample.vtt` | 参考字幕（HTML5 `<track>`，网页播放器直接用） |

## 怎么用

```powershell
# 命令行（用本地模型，离线）
.\.venv\Scripts\python -m video2context samples\chinese_speech_sample.mp4 `
    --model-dir models\faster-whisper-large-v3 -f json,srt,txt -o output

# 网页界面：启动后把 mp4 拖进去即可
.\.venv\Scripts\python -m video2context webui --model-dir models\faster-whisper-large-v3

# 单元测试里也会用到 ffmpeg 现造素材，不依赖这个文件
.\.venv\Scripts\python -m unittest discover -s tests -v
```

## 参考结果（本机实测）

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
