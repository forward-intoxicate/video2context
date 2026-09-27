# 第三方组件与素材说明

本工程自身代码采用 [MIT 许可证](LICENSE)。下面是它依赖的第三方组件、模型权重与测试素材的
来源和许可情况，供分发和商用前核对。

## 1. 运行时依赖（pip 安装）

| 组件 | 用途 | 许可证 |
|---|---|---|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Whisper 推理封装 | MIT |
| [CTranslate2](https://github.com/OpenNMT/CTranslate2) | Transformer 推理引擎 | MIT |
| [PyAV](https://github.com/PyAV-Org/PyAV) | 音频解码 | BSD-3-Clause |
| [onnxruntime](https://github.com/microsoft/onnxruntime) | 运行 Silero VAD | MIT |
| [tokenizers](https://github.com/huggingface/tokenizers) | 文本分词 | Apache-2.0 |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | 模型下载 | Apache-2.0 |
| [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) | 提供静态 ffmpeg 可执行文件 | BSD-2-Clause（其打包的 ffmpeg 为 LGPL/GPL，见下） |
| [Gradio](https://github.com/gradio-app/gradio) | 网页界面（可选） | Apache-2.0 |
| [NumPy](https://github.com/numpy/numpy) | 数值计算（间接依赖） | BSD-3-Clause |

### 可选引擎 Qwen3-ASR 的依赖（装在独立的 `.venv-qwen`，不装也能用）

| 组件 | 用途 | 许可证 |
|---|---|---|
| [qwen-asr](https://pypi.org/project/qwen-asr/) | Qwen3-ASR 官方推理封装 | Apache-2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) | 深度学习运行时 | BSD-3-Clause |
| [Transformers](https://github.com/huggingface/transformers) | 模型加载与生成 | Apache-2.0 |

平时不装这一套；只有使用 `--engine qwen3-asr` 时才需要，
详见 [README](README.md#可选装-qwen3-asr-引擎)。

## 2. ffmpeg

`imageio-ffmpeg` 分发的 ffmpeg 静态构建来自 [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)，
按 GPL 授权分发。本工程只是以子进程方式调用它做音视频解码，**不链接其代码**。
如果你对许可证有严格要求，可以改为调用系统自带的 ffmpeg（设置环境变量 `FFMPEG_BIN`）。

## 3. 模型权重（首次运行时自动下载，不在本仓库内）

| 模型 | 来源 | 许可证 |
|---|---|---|
| Whisper（默认 `large-v3` 等） | OpenAI 训练并开源 | MIT |
| faster-whisper 的 CTranslate2 转换版 | HuggingFace `Systran/faster-whisper-*` 等仓库 | MIT（转换版沿用原模型许可） |
| Silero VAD v6 | 随 faster-whisper 包分发（`silero_vad_v6.onnx`） | MIT |
| [Qwen3-ASR-1.7B / 0.6B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)（可选引擎） | 阿里通义千问团队 | **Apache-2.0** |
| [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)（可选，字幕时间戳） | 阿里通义千问团队 | **Apache-2.0** |

使用模型前请自行确认其许可证与你的使用场景相符（尤其是商用场景）。

## 4. 测试素材

`samples/chinese_speech_sample.mp4` 及其参考输出：

- 内容：鲁迅《呐喊》中的《鸭的喜剧》（1922 年作，原文已进入公共领域）
- 朗读：[LibriVox](https://librivox.org/) 录音，朗读者 Jing Li
- 来源：<https://archive.org/details/call_to_arms_jl_librivox>
- 许可证：**CC Public Domain Mark 1.0（公共领域）**
- 处理：由该条目的封面图与音频合成为视频，仅为方便测试；署名信息见 `samples/README.md`
