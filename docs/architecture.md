# 架构与实现细节

本文说明「视频 → 文字」这条流水线**每一步具体做了什么、为什么这么做**。
只想用的话看 [README](../README.md) 就够了。

## 1. 总览

```
                      ┌──────────────────────────────────────────────────────────┐
                      │                  pipeline.process()                      │
                      └──────────────────────────────────────────────────────────┘
  视频/音频文件
       │
       │ ① 探测            ffmpeg -hide_banner -i <input>        → MediaInfo(时长, 是否有音轨/视频轨, 容器格式)
       ▼
   ┌─────────┐
   │ ffmpeg  │ ② 抽音轨     ffmpeg -vn -sn -dn -map 0:a:0 -ac 1 -ar 16000 -c:a pcm_s16le audio.wav
   └─────────┘
       │
       ▼  16kHz 单声道 PCM wav（临时文件，除非 --keep-audio）
   ┌───────────────────┐
   │ 识别引擎（二选一） │ ③ 识别            --engine faster-whisper（默认） / qwen3-asr
   │  ├ Silero VAD v6  │    先按静音切成语音段（vad_filter=True）
   │  ├ 编码器         │    每段归一化到窗口做 mel 频谱 → 编码
   │  └ 解码器         │    束搜索解码 → 文本 + 时间戳（+ 温度回退）
   └───────────────────┘
       │
       ▼  生成器：边识别边产出 segment，可实时回调进度
   ┌─────────┐
   │ writers │ ④ 写结果    JSON（主产物）/ TXT / SRT / VTT
   └─────────┘
```

模块与文件对应关系：

| 阶段 | 模块 | 关键函数 |
|---|---|---|
| ① 探测 | `ffmpeg_tools.py` | `probe_media()` |
| ② 抽音轨 | `ffmpeg_tools.py` | `find_ffmpeg()`、`extract_audio()` |
| ③ 识别（Whisper） | `transcriber.py` | `Transcriber.load()`、`Transcriber.transcribe()`、`Transcriber._collect()` |
| ③ 识别（Qwen） | `qwen_engine.py` + `_qwen_worker.py` | `QwenAsrEngine.transcribe()`、`group_time_stamps()`、`distribute_sentences()` |
| 引擎选择 | `pipeline.py` | `build_transcriber()`、`normalize_engine()` |
| 流程编排 | `pipeline.py` | `process()` |
| ④ 输出 | `writers.py` | `write_outputs()`、`segments_to_srt()`、`segments_to_vtt()` |
| 交互 | `cli.py` / `webui.py` | `main()` / `build_demo()` |
| 领域词表（可选） | `glossary.py` / `llm.py` / `config.py` | `build_glossary_from_scan()`、`symbol_hit_score()` —— 详见 [glossary.md](glossary.md) |

## 2. ① 媒体探测

不依赖 `ffprobe`（自带的静态 ffmpeg 未必附带它），直接解析 `ffmpeg -i` 的 stderr：

```python
proc = _run([ffmpeg, "-hide_banner", "-i", str(path)])
text = proc.stderr + proc.stdout
duration = _DURATION_RE.search(text)      # Duration: 00:06:30.53
has_video = bool(_VIDEO_RE.search(text))  # Stream #0:0: Video:
has_audio = bool(_AUDIO_RE.search(text))  # Stream #0:1: Audio:
format_name = _FORMAT_RE.search(text)     # Input #0, mov,mp4,m4a,..., from '...'
```

注意 `ffmpeg -i 文件`（不给输出）**一定会返回非 0 退出码**，这是正常行为，所以这里只看 stderr 内容，不看退出码。

**没有音轨的视频会在这一步就被拦下**，报错是"该文件不含音频轨道，无法转写"，而不是让后续步骤抛出难懂的 ffmpeg 错误。

## 3. ② 抽取音轨

```python
ffmpeg -hide_banner -loglevel error -nostdin -y \
       [-ss <start>] -i <input> [-t <duration>] \
       -vn -sn -dn -map 0:a:0 \
       -ac 1 -ar 16000 -c:a pcm_s16le -f wav <output.wav>
```

逐项说明：

| 参数 | 作用 |
|---|---|
| `-nostdin` | 禁止 ffmpeg 读取标准输入，避免在后台任务里挂死 |
| `-vn -sn -dn` | 丢弃视频/字幕/数据流 |
| `-map 0:a:0` | **只取第一条音轨**（多音轨视频里明确选一条，避免 ffmpeg 猜） |
| `-ac 1 -ar 16000` | 单声道 + 16kHz —— Whisper 的标准输入规格 |
| `-c:a pcm_s16le` | 16 位小端 PCM，无损、无解码歧义 |
| `-ss/-t` | 裁剪，配合 `--start` / `--duration` |

**为什么要自己转 16kHz 单声道**：Whisper 内部要把音频转成 16kHz 的 80/128 维 mel 频谱。
如果直接喂 48kHz 立体声，faster-whisper 会用 PyAV 在解码时重采样——结果一样，但
（1）VAD 与解码器共用同一份已是目标格式的数据，判定更稳定；
（2）音频解码与重采样交给 ffmpeg（最快的那条路）而不是 Python；
（3）临时文件可保留（`--keep-audio`）用于复查。

**ffmpeg 从哪来**（`find_ffmpeg()` 的查找顺序）：

1. 环境变量 `FFMPEG_BIN`（想指定自己的版本时用）
2. 系统 `PATH` 里的 `ffmpeg`
3. `imageio-ffmpeg` 附带的静态二进制 —— 这是**默认路径**，所以用户不需要手动装 ffmpeg

## 4. ③ 语音识别

### 4.1 设备与精度选择

`Transcriber._device_attempts()` 给出尝试顺序，`load()` 逐个尝试直到成功：

| 顺序 | device | compute_type | 场景 |
|---|---|---|---|
| 1 | `cuda` | `float16` | 有 N 卡，默认 |
| 2 | `cuda` | `int8_float16` | 显存紧张 |
| 3 | `cpu` | `int8` | 无显卡或显卡不可用 |

- `--device cuda/cpu` 时不会自动跨设备降级（用户显式指定就尊重用户），只在该设备上换精度重试；
- `--device auto`（默认）才允许"GPU 失败退 CPU"；
- Windows 上加载前会调用 `register_cuda_dll_dirs()`，把
  `site-packages\nvidia\{cublas,cudnn}\bin` 注册进 DLL 搜索路径（`os.add_dll_directory` + 追加 `PATH`），
  这是 pip 安装的 CUDA 运行库能被 ctranslate2 找到的关键。

### 4.2 显存预检（避免中途 OOM）

模型加载完成后、推理开始前，用 `nvidia-smi --query-gpu=memory.free` 查一次剩余显存，
按经验公式收缩批量大小：

```
可用批量 ≈ (剩余显存MB − 700MB 固定开销) ÷ 700MB（每路 30s 窗口的激活值）
```

例如日志里的 `剩余显存约 3976MB → 批量大小 8 收缩为 4`。
剩余显存连一路都放不下时直接改用逐段模式（`batch_size=0`）。

每路 700MB 这个系数是实测调出来的：早期用 450MB 估算时，在剩余 3.9GB 的机器上会给出 7 路，
结果批量推理 OOM；改成 700MB 后同一环境下自动选 4 路，稳定通过。

**注意主机内存**：模型要先读进内存再上卡，批量推理也会在主机侧分配缓冲。
实测在“可用物理内存仅 4.3GB”的机器上，即便显存充裕也会失败（`MemoryError`）。
这种情况预检无法预知，所以错误处理会明确提示用户去看任务管理器的可用内存（见 4.6）。

### 4.3 VAD 切分

```python
vad_parameters = {"min_silence_duration_ms": 500, "speech_pad_ms": 200}
```

对比库默认值（`min_silence_duration_ms=2000, speech_pad_ms=400`）我们**调小**了：

- 优点：字幕分段更细、长静音视频更快、静音段的幻觉更少；
- 代价：切得更碎时上下文变短，个别地方断句不如库默认自然。

VAD 模型是随 faster-whisper 分发的 **Silero VAD v6**（`silero_vad_v6.onnx`，约 1.2MB，ONNX Runtime 推理），
完全离线、不需要额外下载。

### 4.4 解码参数

传给 `WhisperModel.transcribe()` 的参数（`_invoke()` 会按目标函数签名过滤，以兼容不同版本的 faster-whisper）：

| 参数 | 我们传的值 | 说明 |
|---|---|---|
| `language` | `None`（自动）/ `zh` / `en` | 自动检测时由模型输出语种概率 |
| `task` | `transcribe` / `translate` | `translate` 会把结果转成英文 |
| `beam_size` | 默认 5 | 束搜索宽度 |
| `vad_filter` | 默认 `True` | 见上 |
| `initial_prompt` | 默认 `None` | 领域词汇提示，对专有名词命中率提升明显 |
| `condition_on_previous_text` | 默认 `False` | 长音频防重复/幻觉 |
| `temperature` | 默认 `None` | 即使用库默认回退序列 `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` |
| `word_timestamps` | 默认 `False` | 打开后 JSON 每段多一个 `words` 数组 |

关于温度回退：Whisper 解码后会用压缩比、平均对数概率、`no_speech_prob` 做质量检查；
不达标就用更高温度重解码。把它设成单个 `0.0`（等于关掉回退）会更快，但遇到难音频时
更容易输出重复句子或乱码——所以默认保留回退。

### 4.5 批量推理

GPU 上默认走 `BatchedInferencePipeline`（faster-whisper ≥1.1）：

- 先用 VAD 把音频切成 ≤30 秒的语音块，再把多块**打包成一个 batch** 送进编码器；
- GPU 利用率显著提升，本机实测 24×～30× 实时，逐段模式约 5×～8×；
- 显存放不下时自动降级（见 4.2/4.6）；
- 实现上通过 `_invoke()` 过滤参数：批量管线不支持的参数不会硬塞进去。

### 4.6 错误处理与降级链路

```
load():   cuda/float16 ──失败──▶ cuda/int8_float16 ──失败──▶ cpu/int8 ──失败──▶ 抛错（附排查建议）
             │                                                      
             └─ 模型文件下载失败时：直连 ──失败──▶ 自动换 hf-mirror 重试一轮

transcribe(): 批量推理 ──OOM──▶ 逐段模式 ──失败──▶ 抛错（附可执行的命令建议）
                  │
                  └─ CUDA 上下文已失效（invalid device ordinal）时直接给建议，不做无意义重试
```

进度回调（`progress_callback`）在 `_collect()` 里边识别边触发，因此长音频能实时看到进度；
CLI 用它画进度条和 `--stream` 逐句输出，Web UI 用它更新 Gradio 进度。

### 4.7 模型缓存

`get_transcriber()` 维护**单槽**缓存（按模型名/设备/精度等参数做 key）：
批量处理多个文件、网页端连续上传时不会重复加载模型（加载一次要 5～30 秒）。
换模型时旧实例被替换、显存释放，避免堆满。

### 4.8 Qwen3-ASR 引擎（`--engine qwen3-asr`）

#### 为什么走子进程

Qwen3-ASR 需要 `torch + transformers`，而主工程**刻意不装 torch**：
faster-whisper 走 CTranslate2/ONNX 路线，两者塞进同一个环境会互相牵制版本
（`transformers` 对 `torch` 版本很敏感）。所以拆成两个环境：

```
主环境 .venv                          独立环境 .venv-qwen
├─ faster-whisper / ctranslate2       ├─ torch（按显卡选 cu126 / cpu）
├─ imageio-ffmpeg                     ├─ transformers==4.57.6
├─ gradio（可选）                      └─ qwen-asr==0.0.6
└─ qwen_engine.py ──subprocess──────▶ _qwen_worker.py ──▶ result.json
   （探测/调度/切句）                    （只做推理，不 import 主工程）
```

收益：主环境永远轻量（没显卡也能装、CI 里不用下 3GB torch）；
代价：每次转写多付一次模型加载（实测 6.6～10.9s），音频越长越无所谓。

#### 数据契约

`_qwen_worker.py` 只认命令行参数，结果写 JSON，**日志全部走 stderr**
（主进程逐行转发到自己的日志，所以用户看到的是一份连续的日志）：

```json
{
  "language": "Chinese", "text": "整段文字……",
  "time_stamps": [{"text": "今", "start": 0.12, "end": 0.34}, ...],
  "duration": 72.538, "load_seconds": 6.638, "elapsed_seconds": 5.8
}
```

工人里有两处针对小显存机器的处理（都是安全的）：
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解碎片；
跳过 `transformers` 加载前的显存预热（1.7B 会一次性申请 3.2GB，8GB 卡直接 OOM）。
另外 `--low-mem` 会走 meta 设备 + `safetensors.load_file(device="cuda:0")`，
让权重分片**直接进显存**，绕开"整片读进 CPU 内存"导致的 `页面文件太小`。

#### Qwen 只输出一整段文字，字幕分段是本工程做的

这是接入 Qwen 最费事的地方：它没有 faster-whisper 那种天然的分段与时间戳。
`qwen_engine.py` 提供两条路径：

| 条件 | 时间戳来源 | 算法 |
|---|---|---|
| 装了 `Qwen3-ForcedAligner-0.6B` | `forced-aligner` | 见下 |
| 没装 | `vad-proportional` | 按标点切句 → Silero VAD 找说话区间 → 按字数比例分配 |

**精确路径（有对齐模型）**——难点在于对齐模型返回的 token **是去掉标点的**
（`is_kept_char()` 只保留字母、数字、撇号），所以不能直接拼：

1. 把原文按"会保留的字符"压成一条参考串，同时记下每个字符在原文里的下标；
2. 每个 token 在参考串里顺序查找（带 8 字符容差，可自我纠偏），映射回原文下标区间；
3. 命中率不足 90% 就判定这套映射不成立，退回退化路径；
4. 遍历 token 累积成段，遇到**句末标点 / 明显停顿（≥1.2s）/ 超长（>80 字或 >12s）**
   就断开；切片时把紧随其后的标点补回来（否则字幕会丢掉所有句读）。

实测这段算法在 42 秒英文素材上切出 7 段，`''.join(段文字)` 与原文
**归一化后长度完全相等**（525/525 字符），标点一个不丢。

### 4.9 引擎选择与词表提示的差异

`pipeline.build_transcriber()` 按 `opts.engine` 造识别器，两个引擎的
`.transcribe()` 签名与返回值一致（都返回 `TranscriptionResult`），所以
流水线后面（写文件、进度条、JSON 组装）完全不用区分引擎。

`--model` 的默认值是 Whisper 的 `large-v3`；切到 Qwen 引擎时它显然不是用户本意，
会被当作"未指定"，改用 Qwen 的默认模型（本地 `models/Qwen3-ASR-1.7B`，否则仓库名）。

**词表提示的写法必须区分引擎**（`glossary_prompt_for()`）：

| 引擎 | 用哪个字段 | 写法 |
|---|---|---|
| faster-whisper | `initial_prompt` | 极短、只放符号：`符号：u(x)、v(x)。`（实测 30～80 字最稳） |
| qwen3-asr | `context`（system message） | **描述性完整句**：`视频里写出的公式和符号有u(x)、v(x)、x0。` |

Qwen 这一栏不能用祈使句（「请准确识别以下术语…」）：实测那样写模型会**复述 context
而不转写音频**。改成描述"视频里有什么"之后能正常转写 —— 但进一步实测发现
context 对结果**没有可观测影响**，详见 [models.md 第 6.4 节](models.md#64-实测context词表偏置没有效果)。

另外，Qwen 引擎下**不做**"偏置失效就回退重跑"（那是 faster-whisper 特有的静默失效问题），
只统计评分写进 JSON —— 重跑一次要多付 10 秒的模型加载，不值得。

## 5. ④ 输出

`write_outputs()` 按 `--formats` 写文件，文件名冲突时自动改名（`xxx.json` → `xxx-1.json`），
除非显式传了 `--name` 或 `--overwrite`。JSON 结构见 [README](../README.md#输出文件说明)。

几个实现细节：

- JSON 用 `ensure_ascii=False` 写，中文直出，便于人读和下游处理；
- 时间戳统一 `HH:MM:SS.mmm`（SRT 用逗号版本），由 `format_hms_ms()` 单一实现，避免各格式不一致；
- `segments[].words` 只在 `--word-timestamps` 时出现，其余情况字段被剔除，保持文件干净。

## 6. 交互层

### CLI（`cli.py`）

- 子命令 `transcribe` / `webui` / `doctor`，并做了命令别名与"省略 transcribe"的兼容：
  `python -m video2context 视频.mp4` 等价于 `python -m video2context transcribe 视频.mp4`；
- 输出编码自适配：**重定向时强制 UTF-8**（避免 GBK 环境写日志乱码），终端里保持系统编码；
- 批量模式返回码：只要有文件失败就返回 1，便于脚本判断；
- 模型在批量循环外复用（见 4.7）。

### Web UI（`webui.py`）

- Gradio `Blocks`，`queue(default_concurrency_limit=1)` —— 串行处理，避免并发抢显存；
- 识别引擎做成下拉框（`ENGINE_LABELS`），标签里直接写清取舍，`--engine` 决定默认选中项；
  切到 Qwen 时若模型下拉框仍是 Whisper 的默认名，会被当成"未指定"交给内部自动选；
  选 Qwen + 翻译任务会在点击时被拦下并提示怎么改（而不是等模型加载完才报错）；
- 无对齐模型时会用 `asr.timestamp_source` 判断并在状态栏挂一条"字幕时间是估算的"警告；
- 兼容 Gradio 5.x/6.x：`theme` 在 6.0 移到了 `launch()`，`show_copy_button` 换成了 `buttons`，
  代码用 `inspect.signature` 判断后择优传参；
- `GRADIO_TEMP_DIR` 默认指向工程内 `output/.gradio`，避免受限系统临时目录带来的问题；
- 上传文件走 Gradio 自己的缓存目录，转写结果落到 `--output-dir`。

## 7. 设计取舍记录

| 决策 | 原因 | 代价 |
|---|---|---|
| 自带静态 ffmpeg（不强制用户装） | 90% 的安装失败都出在 ffmpeg | wheel 体积约 30MB |
| 先转成 wav 再识别 | 格式统一、VAD 稳定、可保留复查 | 磁盘临时文件（1 小时约 115MB） |
| 延迟导入 faster-whisper | `--help` / `doctor` 秒开，不必先装齐重依赖 | 多一层函数内 import |
| 单文件一个 `process()` 调用 | 易测试、易并发、无全局状态 | 批量时需外部的模型缓存 |
| JSON 作为主产物 | 下游（字幕/检索/翻译）都从它二次加工 | 比纯文本略大 |
| 保留温度回退序列 | 难音频不会输出垃圾 | 极端情况下更慢 |
| Qwen3-ASR 走独立环境 + 子进程 | torch 与 CTranslate2 不互相牵制；主环境永远轻量、CI 不用装 3GB | 每次转写多一次模型加载（6.6～10.9s）；多一层 JSON 契约 |
| Qwen 的时间戳靠强制对齐模型 | 模型本身不分段，对齐模型能给出词级时间 | 多 1.8GB 磁盘；不装就得接受估算时间（JSON 里会标明） |
| 两个引擎共用 `TranscriptionResult` | 上层（写文件/进度/JSON）零改动，可随时 A/B | 接口里有些字段对 Qwen 无意义（置空并在 JSON 里标明） |

## 8. 想改的话从哪下手

| 想做的事 | 改哪里 | 注意 |
|---|---|---|
| 换识别引擎（FunASR/SenseVoice） | `transcriber.py` 的 `Transcriber.transcribe()` | 保持返回 `TranscriptionResult`（内含 `Segment` 列表）即可，上层不用动 |
| 给 Qwen 加新的后处理/切句规则 | `qwen_engine.py` 的 `group_time_stamps()` | 有 `tests/test_qwen_engine.py` 的合成用例兜底 |
| 加导出格式 | `writers.py` + `SUPPORTED_FORMATS` | 记得补 `tests/test_units.py` 用例 |
| 加命令行参数 | `cli.py` 的 `build_parser()` | 同时更新 README 参数表与本文档 |
| 改 VAD 策略 | `transcriber.py` 的 `vad_parameters` | 改完用 `samples/` 里的样例对比分段数 |
| 接入翻译/摘要 | `pipeline.py` 组装 `payload` 之后 | 建议写进 JSON 的新顶层字段，保持 schema_version 递增 |
