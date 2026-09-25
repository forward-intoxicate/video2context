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

## 6. 换成 FunASR / SenseVoice（中文标点更好）

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
