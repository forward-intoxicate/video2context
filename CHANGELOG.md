# 更新日志

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

### 新增：识别引擎 `--engine`（faster-whisper / Qwen3-ASR）

- `--engine qwen3-asr`：接入阿里通义 **Qwen3-ASR-1.7B**（Apache-2.0）作为第二个识别引擎。
  实测中文数学课样例上 `u(x)` / `v(x)` / `导数` **全对且标点完整**（large-v3 分别为右/位/倒 ×13 处且无标点）；
  英文 42 秒素材 **WER 0.00%**（large-v3 为 2.97%）
- **独立环境 + 子进程桥接**：Qwen 需要 torch，主工程刻意保持 torch-free，
  因此拆成 `.venv-qwen`，由 `_qwen_worker.py` 子进程执行、JSON 交换结果。
  主环境不变，CI 也不需要装 3GB torch
- **字幕分段自研**：Qwen 只输出一整段文字。装了 `Qwen3-ForcedAligner-0.6B` 时按词级时间戳
  映射回原文下标切句（实测切分能 100% 还原原文、标点不丢）；没装则按标点切句 +
  Silero VAD 说话区间按字数比例分配，并在 JSON 里标明 `timestamp_source` 为 `vad-proportional`
- 面向小显存/小内存机器的处理：跳过 transformers 的显存预热、`expandable_segments`、
  `--qwen-low-mem`（分片直接进显存，绕开"整片读进 CPU 内存"导致的 `页面文件太小`）、
  失败自动降级重试、把原始报错翻译成可照做的建议
- `--qwen-setup`：只做环境自检（解释器 / 模型 / 对齐模型），不转写
- 网页界面新增「识别引擎」下拉框（`webui --engine` 决定默认选中项）；选 Qwen + 翻译任务会提前拦下并提示，
  没有对齐模型时状态栏会标注"字幕时间是估算的"
- 新增 `V2C_ENGINE` 配置项（环境变量或 `.env`）：在自己的机器上把 Qwen 设成默认引擎，
  优先级为 `--engine` > `V2C_ENGINE` > `faster-whisper`。
  出厂默认仍保留 faster-whisper —— 新克隆的仓库只有主环境（torch-free），
  默认成 Qwen 会让"装完就能用"直接失败；`doctor` 会显示当前默认引擎
- 新增模块/脚本：`qwen_engine.py`、`_qwen_worker.py`、`scripts/setup_qwen.ps1`、`scripts/setup_qwen.sh`、
  `requirements-qwen.txt`、`experiments/`（实测脚本与原始数据，含英文 WER 报告工具）
- `TranscriptionResult` 增加 `engine` 与 `extra` 字段；JSON 的 `asr.engine` 标明结果出自哪个引擎

### 新增：领域词表（修同音词/专有名词）

- `--glossary`：手写词表（`错形 -> 正确` 或纯术语），完全离线
- `--auto-glossary`：自动两遍解码 —— 采样 90 秒粗转写 → 大模型推断符号还原 → 带词表正式转写
- `--dry-run-glossary`：只生成词表供人工复核（推荐流程）
- **验证 + 回退**：带词表结果里"正确符号一个未命中且仍有错误写法残留"即判定偏置失效，自动回退到无词表版本；
  另报告逐符号覆盖率 `coverage`（比总命中次数更能反映真实效果）
- 修复：`--device` 自动降级到 CPU 时误用 GPU 专用精度（`float16`/`int8_float16`）导致加载失败
- 新增模块：`glossary.py`（词表与评分）、`llm.py`（OpenAI 兼容客户端，零新依赖）、`config.py`（.env 与密钥脱敏）
- JSON 增加可选 `glossary` 段（词表、来源、大模型用量、校验结果）；`schema_version` 升到 `1.1`

### 已知限制（实测记录，不是 bug）

- Qwen3-ASR 的 `context`（词表偏置）在 `qwen-asr==0.0.6` + 1.7B 上**对结果没有可观测影响**：
  换成完全无关的内容，输出逐字相同。链路本身是通的（渲染出的 prompt 里 system 消息确实存在）。
  因此词表在 Qwen 引擎下仍会生成并写进 JSON，但程序会提示"不要指望它修同音词"；
  同音词修正仍请用 faster-whisper 引擎，或做后处理改写。详见 `docs/models.md` 第 6.4 节
- Qwen 引擎不支持 `--task translate`（只做原语言转写），会给出明确报错

### 安全

- `.env` / `.env.example`：密钥只从环境变量或 `.env` 读取，**不提供任何命令行参数传密钥**
- `scripts/check_secrets.py`：提交前防泄露扫描（已接入 CI）；开发过程中它就抓到过一次真实密钥误入测试文件
- 日志、报错、JSON 里的密钥统一打码；`doctor` 只显示 `sk-***abcd`

### 文档

- 新增 `docs/glossary.md`：含 9 个 prompt 变体的实验数据（含"四个假设全被否定"的记录）
- `README.md` 新增「识别引擎怎么选」章节（两引擎对照表、Qwen 的实测质量/速度数据、context 实测结论）
- `docs/models.md` 新增第 6 节：Qwen3-ASR 三个模型的职责、实测数据、`context` 无效的四组对照实验
- `docs/architecture.md` 新增 4.8/4.9：子进程桥接的数据契约、时间戳→下标映射算法、引擎选择与提示差异
- `docs/troubleshooting.md` 新增第 7 节：Qwen 引擎的 7 类症状排查（环境、提交内存、torch CPU 版、时间戳、词表、翻译、加载耗时）

### 测试

- 测试数 28 → 87：新增词表/配置单测与两遍解码的流水线集成测试（桩接大模型，不联网）；
  新增 `tests/test_qwen_engine.py`（29 项）：语种归一化、时间戳映射、切句、比例分配、引擎选择、解释器探测
## [0.1.0] - 2025-09-25

首个可用版本。

### 功能

- 视频/音频 → 文字的完整流水线：ffmpeg 抽音轨 → VAD 切分 → faster-whisper 识别 → 结构化输出
- 中英文自动识别语种，也可强制指定
- 输出格式：JSON（主产物，含分段与时间戳）、TXT、SRT、VTT
- 命令行：`transcribe`（支持批量、裁剪、实时逐句打印）、`webui`、`doctor`
- 本地网页界面（Gradio），可拖拽上传、查看分段表格、下载 JSON
- 设备自动选择：CUDA（float16）→ CPU（int8），加载失败自动降级
- Windows 下自动注册 pip 版 CUDA 运行库（cuBLAS/cuDNN）目录
- 推理前预检剩余显存并收缩批量大小；显存不足时自动降级重试并给出可执行建议
- 模型缓存：批量任务与网页端连续转写不重复加载模型
- 模型下载：自动回退 hf-mirror；默认关闭 huggingface_hub 的 Xet 存储（弱网下会挂起）
- 附赠模型分块断点续传下载器（支持 HuggingFace 与魔搭 ModelScope 双源）
- 中文测试样例（公共领域真人朗读）+ 参考输出
- 文档：`README.md`（含"有显卡 / 无显卡"两条安装路径）、`docs/architecture.md`（流水线实现细节）、
  `docs/models.md`（模型选型与量化说明）、`docs/troubleshooting.md`（按症状排查）、`docs/development.md`
- 28 项离线单元测试（不需要模型和网络）
- GitHub Actions CI：4 组平台/Python 版本矩阵 + ruff 检查 + 文档链接校验 + PowerShell 脚本编码检查
- 自带 `scripts/check_docs.py`：校验文档相对链接与锚点
- `THIRD_PARTY_NOTICES.md`：第三方依赖、模型权重与测试素材的许可说明

[0.1.0]: https://github.com/forward-intoxicate/video2context/releases/tag/v0.1.0
