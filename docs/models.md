# 用了哪些模型，怎么选

本工程一共用到 **两类模型**，都是首次运行时自动下载、之后可完全离线。

| 用途 | 模型 | 体积 | 来源 | 许可证 |
|---|---|---|---|---|
| 语音识别 | **Whisper**（默认 `large-v3`）的 CTranslate2 转换版 | 75MB ～ 3.1GB | OpenAI 训练 → Systran 等转 CT2 格式 | MIT |
| 人声检测（VAD） | **Silero VAD v6** | 1.2MB | 随 faster-whisper 包分发（ONNX） | MIT |
| 语种识别 | Whisper 内置 | — | 编码器输出的语种 token 概率 | MIT |

推理后端是 **CTranslate2**（MIT）：一个为 Transformer 推理优化的 C++ 引擎，
负责算子融合、量化与批量调度。`faster-whisper` 是它的 Python 封装（MIT）。

---

## 1. Whisper 是什么

Whisper 是 OpenAI 2022 年开源的语音识别模型，特点是**多语种 + 强鲁棒性**（带口音、带噪声、
专业术语都能扛），并且同时输出**时间戳**和**语种**。原始权重是 PyTorch 格式，
社区用 CTranslate2 转成推理友好格式后，速度提升 3～4 倍、显存减半，这就是本工程用的版本。

模型默认从 HuggingFace 的 `Systran/faster-whisper-*` 仓库下载（与 faster-whisper 官方映射一致）。
`large-v3-turbo` 用的是 `mobiuslabsgmbh/faster-whisper-large-v3-turbo`。

### 各版本对照

| 模型 | 参数量 | CT2 体积 | 显存（float16） | 中文 | 英文 | 速度 |
|---|---|---|---|---|---|---|
| `tiny` | 39M | 75MB | ~1GB | ✗ 常出繁体/错字 | 一般 | 极快 |
| `base` | 74M | 145MB | ~1GB | ✗ | 一般 | 极快 |
| `small` | 244M | 484MB | ~2GB | 勉强 | 尚可 | 快 |
| `medium` | 769M | 1.5GB | ~5GB | 良好 | 良好 | 中 |
| `large-v2` | 1550M | 3.0GB | ~5GB | 很好 | 很好 | 慢 |
| **`large-v3`** | **1550M** | **3.1GB** | **~5GB** | **最好** | **最好** | 慢 |
| `large-v3-turbo` | 809M | 1.6GB | ~4GB | 好 | 好 | 约 large-v3 的 4 倍 |
| `distil-large-v3` | 756M | 1.5GB | ~4GB | ✗（不可用） | 接近 large-v3 | 快 |

> `large-v3` 相对 `large-v2` 的主要变化：mel 特征从 80 维升到 **128 维**，
> 训练数据扩充到 **500 万小时**（含更多中文/粤语），中文简繁与标点表现明显更好。
> 我们实测 `large-v3` 输出的就是**简体中文**，而 `tiny` 会输出繁体并出现错字。

### `*.en` 与 `distil-*` 怎么用

- `tiny.en` / `base.en` / `small.en` / `medium.en`：只在英文上训练，**英文略强但完全不会中文**。
- `distil-large-v3`：蒸馏版，**只支持英文**。中文视频千万别选（会输出英文/乱码）。

---

## 2. CTranslate2 的"精度"（compute_type）是什么意思

同一个模型可以用不同数值精度推理，直接影响**显存占用**和**速度**：

| compute_type | 适用设备 | 说明 |
|---|---|---|
| `float16` | **GPU 默认** | 半精度，速度最快、精度几乎无损 |
| `int8_float16` | GPU | 权重 8 位、计算 16 位，显存约减半，质量略降 |
| `int8` | **CPU 默认** | CPU 上最快，质量略降 |
| `int8_float32` | CPU | int8 权重 + float32 计算 |
| `float32` | 都行 | 最准最慢，一般不需要 |

程序默认：GPU 用 `float16`、CPU 用 `int8`（`--compute-type` 可覆盖）。
注意 **CPU 不支持 `int8_float16`**（那是 GPU 专用组合），指定了会加载失败。

---

## 3. VAD：为什么需要它，用的是什么

VAD（Voice Activity Detection）负责找出"哪里有人在说话"。

**用的是什么**：`silero_vad_v6.onnx`（Silero VAD 第 6 版，约 1.2MB），
随 faster-whisper 一起安装，由 ONNX Runtime 在本机推理，**不需要联网、不需要显卡**。

**为什么需要**：

1. **速度**：长视频里大量静音/音乐被整段跳过，不送进模型；
2. **质量**：Whisper 在纯静音/纯音乐片段上会产生"幻觉"（凭空编出一句话、或反复重复上一句），
   切掉这些片段能极大减少这类问题；
3. **字幕**：天然得到"一句一段"的切分，时间戳更贴合字幕需求。

**参数**（`transcriber.py`）：

```python
vad_parameters = {"min_silence_duration_ms": 500, "speech_pad_ms": 200}
```

- `min_silence_duration_ms`：静音超过多久算"一段结束"。库默认 2000ms 偏保守，
  我们调成 500ms → 分段更细、更适合字幕。
- `speech_pad_ms`：在语音段两端各补多少毫秒，避免把首尾音节切掉。库默认 400ms，我们调成 200ms。

想关掉用 `--no-vad`（不推荐，除非源音频没有静音且你想保留完整上下文）。

---

## 4. 怎么选（按你的硬件）

> 本节讲的是 **Whisper 引擎**的选型（`--engine faster-whisper`）。
> 默认引擎是 Qwen3-ASR，见[第 6 节](#6-qwen3-asr默认引擎)。

### 有 NVIDIA 显卡

| 显存 | 推荐 | 命令 |
|---|---|---|
| ≥ 8GB | `large-v3` | 默认即可 |
| 6～8GB | `large-v3`（自动收缩批量）或 `large-v3-turbo` | 默认，必要时 `--compute-type int8_float16` |
| 4～6GB | `large-v3-turbo` / `medium` | `--model large-v3-turbo` |
| 显存不足 | `medium` + `int8_float16` | `--model medium --compute-type int8_float16` |

### 没有显卡（纯 CPU）

| CPU | 推荐 | 说明 |
|---|---|---|
| 8 核以上现代 CPU | `small` + `int8` | 速度与质量平衡点，**推荐默认** |
| 想更准、能等 | `medium` + `int8` | 约慢 2～3 倍 |
| 只是试试 | `tiny` | 秒出结果，但中文质量差 |
| 长视频 | `small` + `--model small` | 别用 `large-v3`，CPU 上会非常慢 |

```powershell
python -m video2context 视频.mp4 --device cpu --model small --compute-type int8
```

CPU 线程数用 `--cpu-threads`（默认 0 = 自动，通常等于物理核心数）。

### 明确知道语言时一定要指定

```powershell
--language zh    # 或 en
```

好处：跳过语种检测、减少"听着像别的语言就切过去"的幻觉、速度略快。
只有你不确定视频是什么语言时才用默认的 `auto`。

---

## 5. 实测数据

测试机：**i9-13900HX（32 线程）+ RTX 4060 Laptop 8GB**，
素材：`samples/chinese_speech_sample.mp4`（6 分 30 秒中文真人语音，VAD 后约 354 秒，音频全长 390 秒）。

| 模型 | 设备 / 精度 | 推理耗时 | 相对实时 |
|---|---|---|---|
| `large-v3` | GPU float16 + 批量（实际 4～7 路） | 16.2～19.1s | **20～24×** |
| `tiny` | CPU int8 + 逐段 | 11.2s | **35×** |
| `small` | CPU int8 + 逐段 | 64.2s | **6.1×** |
| `medium` | CPU int8 + 逐段 | 未实测（按参数量约为 `small` 的 2～3 倍耗时） | ~2～3× |

GPU 那一行的波动来自**批量大小是按剩余显存自动定的**（程序会打印
`剩余显存约 3976MB → 批量大小 8 收缩为 4`）。想稳定复现可显式指定 `--batch-size`。

关于 CPU 数据的两点提醒：

1. **强依赖核心数**。上表是 32 线程的成绩；4 核笔记本大致慢 3～5 倍，
   `small` 会落到 1～2 倍实时，仍然可用，`medium` 就偏慢了。
2. **CPU 上批量推理没有意义**（`--batch-size` 只对 GPU 生效），走的是逐段模式。

> Whisper 按 30 秒窗口计算，**音频越长越接近稳态吞吐**；短视频看起来"慢"，
> 主要是模型加载（本地模型约 5s）与首次 CUDA 预热（约 1～2s）。

---

## 6. Qwen3-ASR（默认引擎）

`qwen3-asr` 是**默认引擎**：阿里通义开源的 **Qwen3-ASR**（Apache-2.0）。
它跟 Whisper 是并列的两个引擎 —— 没装它就自动用 Whisper，只装了它就用它，
两个都装则优先它（见 [architecture.md 4.9](architecture.md#49-引擎选择与词表提示的差异)）。

### 6.1 三个模型分别干什么

| 模型 | 体积 | 作用 | 能不能不装 |
|---|---|---|---|
| `Qwen3-ASR-1.7B` | 4.0GB | 语音识别主力 | 不能（或者换 0.6B） |
| `Qwen3-ASR-0.6B` | 1.6GB | 省显存的降级选项 | 可以（默认用 1.7B） |
| `Qwen3-ForcedAligner-0.6B` | 1.8GB | 只给文字配**词级时间戳** | **可以**，`--qwen-aligner off` |

三者都是纯粹的权重文件，本地加载、不联网、不上传音频。

### 6.2 实测：质量对比

素材一：`samples/formula.mp4`（73 秒中文数学课，讲导数运算法则）。

| | `faster-whisper` large-v3 | `Qwen3-ASR-1.7B` |
|---|---|---|
| `u(x)` | 写成「右」4 次 | ✅ 全对 |
| `v(x)` | 写成「位」4 次 | ✅ 全对 |
| `导数` | 混成「倒数/求倒/求到」5 次 | ✅ 全对 |
| 标点 | **一个都没有** | ✅ 完整（`，` `。` `、`） |
| 遗留问题 | — | 讲师念的「x 零」「u1 x」→ 写成 `x零`、`u1x` |

素材二：英文技术内容。仓库里的 `samples/demo_en_ml.mp4`（40.4s，92 词，有逐字稿子）
就是按这个内容合成的，可以直接复现：

| | WER | 错在哪 |
|---|---|---|
| `faster-whisper` large-v3 | **4.35%** | `Adam`→`Atom`、`RMSProp`→`RMS Prop`、`Adagrad`→`Atigrid` |
| `Qwen3-ASR-1.7B` | **0.00%** | 无 |

> 更早一次用的是另一段 42 秒 TTS（101 词，未提交仓库），large-v3 是 2.97%。
> 两段素材结论一致，仓库里那份可复现，所以正文统一用它。
> 复现命令见 [samples/README.md](../samples/README.md)。

中文侧同样有可复现的对比：`samples/demo_zh_math.mp4`（41.8s）上
large-v3 把「求导」写成「**球倒**」，Qwen 全对。

> `x零` / `u1x` 不是听错：讲师**确实**念作"x 零"、"u 一 x"。
> 要变成 `x0`、`u1(x)` 属于**后处理改写**，不是 ASR 能力问题。
> 这一点很重要 —— 它解释了为什么下面第 6.4 节的词表偏置帮不上忙。

### 6.3 实测：速度与资源

测试机同上（i9-13900HX + RTX 4060 Laptop 8GB）。

| 素材 | 加载 | 推理 | 相对实时 |
|---|---|---|---|
| `samples/demo_zh_math.mp4`（41.8s 中文） | — | 5.5s | **7.6×** |
| `samples/demo_en_ml.mp4`（40.4s 英文） | — | 6.2s | **6.5×** |
| `samples/demo_zh_daily.mp4`（29.3s 中文） | — | 4.3s | **6.9×** |
| 中文 73s（`formula.mp4`，未提交；含词表 context） | 6.6s | 5.8～7.3s | 10～12.5× |
| 12s 英文片段（**CPU**，32 线程） | 1.4s | 23.9s | **0.5×** |

模型加载每次 6.6～10.9s（本地模型），上表里略去以免混淆；音频越长这部分越无所谓。

资源上的两个硬约束：

* **提交内存（commit）要约 13GB**：`transformers` 会先把 4GB 的权重分片整个读进
  CPU 内存再搬到显存，所以 16GB 内存的机器在浏览器多开时会报 `页面文件太小`。
  绕法是 `--qwen-low-mem on`（meta 设备建空模型 → 分片直接读进显存）。
* **显存约 4GB**：加载前 `transformers` 还会一次性预热分配约 3.2GB，
  在 8GB 卡上会直接 OOM，所以工人脚本**默认跳过这个预热**
  （`modeling_utils.caching_allocator_warmup`，官方注释说它只是加载加速，
  跳过不影响结果）。

### 6.4 实测：`context`（词表偏置）**没有效果**

`Qwen3ASRModel.transcribe(context=...)` 会把文本作为 **system message** 注入。
官方定位是"上下文偏置"，理论上能把术语/专有名词喂给模型。

我按三种写法各测了一遍，**输出逐字相同**（先是 188 字完全一致，后来英文 625 字也完全一致）：

| 传入的 context | 结果 |
|---|---|
| 不传 | 基准 |
| 「视频里写出的公式和符号有u(x)、v(x)、x0、u1(x)。讲到的术语包括导数运算法则。」 | 与基准**逐字相同** |
| 「这段视频讲解导数的运算法则。板书内容是：函数u(x)和v(x)在点x0处可导，则u+v的导数等于…」 | 与基准**逐字相同** |
| 「这段音频讲的是量子力学，视频里写出的公式是E=mc^2。」（**故意完全无关**） | 与基准**逐字相同** |

排除了接线错误的可能：直接渲染 `processor.apply_chat_template`，
确认 system 消息确实进了 prompt：

```
<|im_start|>system
这段音频讲的是量子力学。<|im_end|>
<|im_start|>user
<|audio_start|><|audio_pad|><|audio_end|><|im_end|>
<|im_start|>assistant
```

**结论**：在 `qwen-asr==0.0.6` + `Qwen3-ASR-1.7B` 这一版里，`context` 对识别结果
没有可观测影响。因此本工程的行为是：

* `--glossary` / `--auto-glossary` 照常工作、照常把词表写进结果 JSON（供人工复核与复用）；
* 但命令行会明确提示"不要指望它修同音词"；
* 结果 JSON 里 `asr.context_applied` 记录是否传了 context，
  `asr.context_effect: "none-observed"` 记录这个实测结论；
* 上游将来若让它生效，本工程不需要改代码就能受益。

### 6.5 怎么选

| 你的情况 | 选 |
|---|---|
| 中文课程/讲座，有公式、符号、术语 | `--engine qwen3-asr` |
| 英文技术内容，人名/产品名多 | `--engine qwen3-asr` |
| 没有 NVIDIA 显卡 | `faster-whisper`（Qwen 在 CPU 上太慢） |
| 内存只有 8～16GB 且不想折腾 | `faster-whisper` |
| 需要翻译成英文 | `faster-whisper --task translate`（Qwen 不支持） |
| 想省事、通用场景 | `faster-whisper`（默认） |

---

## 7. 换成 FunASR / SenseVoice（中文标点更好）

Whisper 的中文标点是模型自己"顺手"生成的，专业名词多或口语重的视频标点会不理想。
阿里达摩院的 **FunASR**（含 **SenseVoice-Small**、**Paraformer-large**）在中文上通常表现更好，
而且自带标点恢复与热词功能。

**为什么本工程默认不用**：FunASR 依赖 ModelScope 生态，在部分环境下安装更麻烦
（Python 3.13 上轮子不全），而 Whisper 的跨平台稳定性更好。

**怎么换**（只改一个文件，上层完全不用动）：

```python
# video2context/transcriber.py
class FunAsrTranscriber:
    def load(self): ...
    def transcribe(self, audio_path, *, language=None, task="transcribe", vad_filter=True,
                   beam_size=5, batch_size=0, initial_prompt=None,
                   condition_on_previous_text=False, temperature=None,
                   word_timestamps=False, progress_callback=None, log_callback=None):
        # 1) 调用 AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc")
        # 2) 把结果转成 TranscriptionResult(segments=[Segment(id, start, end, text), ...])
        return TranscriptionResult(...)
```

只要返回 `TranscriptionResult`（内含 `Segment` 列表），`pipeline.py`、`cli.py`、`webui.py`
以及所有输出格式都不用改。记得同步更新 `get_transcriber()` 的分发逻辑。

热词功能可以映射到 `--initial-prompt`，保持命令行接口不变。
