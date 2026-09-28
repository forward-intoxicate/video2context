# 更新日志

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

### 新增：README 的「快速开始」改成 7 步走通全流程（每步都给 Windows / Linux 两版）

原来的「快速开始」是散的（0. 你需要什么 / 1. 装环境 / 2. 转写 / 3. 网页界面 / 4. 样例试跑），
缺「下模型」「确认环境就绪」这两步，也没做到每一步都给两个平台的命令。

- README 新增 **「快速开始：从 0 到跑通」**，放在「命令怎么敲」之后：
  第 0 步确认前置 → 第 1 步 clone → 第 2 步一键装环境 + 下模型 → 第 3 步（可选）单独补下模型
  → 第 4 步 doctor 确认就绪 → 第 5 步拿仓库样例试跑 → 第 6 步对答案 → 第 7 步换成自己的视频
- **每一步都给了 Windows 与 Linux/macOS 两版命令**；关键步骤还给出「期望看到什么」的
  真实输出片段，让人能自己判断有没有成功
- 附「卡在哪一步了？」对照表，把常见失败直接映射到 troubleshooting 的对应小节
- 原「从 0 部署」表改名为「选部署路线（换引擎 / 没显卡 / macOS）」，定位为默认路线之外的备选

### 修复：示例视频的容器时长比音频长 2～3 秒（生成脚本）

`scripts/make_samples.py` 用无限长的 `color` 源 + `-shortest` 合成，
实测末尾会多出 2～3 秒静音黑屏（容器 Duration 45.10s，音频实际只有 41.85s）。
改成按音轨长度显式截断 `color=...:d=<时长>`，现在容器与音频完全一致。

### 实测数据校正：文档里的 WER / 时长改用仓库自带样例

之前正文里的英文 WER 用的是早期一段临时 TTS（101 词，未提交仓库，2.97%），别人复现不了。
现在统一改成仓库自带的 `samples/demo_en_ml.mp4`（92 词）实测：

- `qwen3-asr` **0.00%** vs `faster-whisper` large-v3 **4.35%**
  （`Adam`→`Atom`、`RMSProp`→`RMS Prop`、`Adagrad`→`Atigrid`）
- 中文侧也换成可复现的 `samples/demo_zh_math.mp4`：large-v3 把「求导」写成「**球倒**」，Qwen 全对
- 三个 `demo_*` 的时长 / 耗时全部按实测更新（41.8s / 40.4s / 29.3s）

### 文档：「命令怎么敲」提到最前面，保证第一次打开仓库就能跑起来

用户按示例命令跑，报「未找到 ffmpeg」——因为文档里的示例为了简短统一写成
`python -m video2context ...`，而依赖装在 `.venv` 里，直接敲 `python` 会跑到
系统 / conda 的 Python 上。原先只在「怎么用」开头有一句"假设已激活虚拟环境"，
位置太靠后、也太容易被跳过（全文 75 处裸 `python`，只有 16 处带 venv 前缀）。

- README **新增「命令怎么敲（先看这个）」章节**，放在目录之后、所有其他内容之前：
  两种写法的 Windows / macOS / Linux 对照表 + "文档后面写成 `python` 时请自行替换"
  的明确说明 + 一个替换示例
- 「快速开始 → 拿现成的样例试跑」里的命令改成**显式的 venv 路径**（这是新用户
  最先复制粘贴的一段）
- 同一张对照表同步到 `docs/deploy.md`（新增「命令怎么敲」小节）与 `samples/README.md`
  （替换原来那句较弱的警示），并各配一个"第一条命令的两种真实写法"的例子
- **运行时兜底**：`doctor` 检测到当前解释器不是工程 `.venv` 时，
  在自检输出最前面直接点出来（不再让人从"ffmpeg 不可用"去猜原因）；
  此时 ffmpeg 那一行只说"原因见上方 ⚠"，避免同一段长提示刷两遍

### 修复：「未找到 ffmpeg」的提示没说真正的原因

用户按示例命令跑，报「未找到 ffmpeg」——真实原因是他用的是 conda base，
而依赖装在工程的 `.venv` 里。原来的提示只列了"装 ffmpeg 的三种方式"，
照着做只会把 base 环境也搞乱。

- 新增 `config.interpreter_hint()`：当前解释器不是工程 `.venv` 时，
  给出一段可照做的提示（打印两个 Python 路径 + 两条改法）。
  没建过 `.venv` 时返回空串，不误导
- `ffmpeg_tools` 的报错改为：**环境诊断排在安装建议前面**；
  确实在正确环境里仍找不到 ffmpeg 时，才只给安装方式
- `transcriber` 的「缺少 faster-whisper」也用同一套诊断
- 文档：`docs/troubleshooting.md` 新增 1.5 节（怎么从提示符认出用错了 Python、
  两条改法、为什么不要"在 base 里补装"）；samples/README.md 的命令集合顶部加了警示

### 新增：仓库自带可直接跑的示例视频与示例命令

- **新增 3 个离线合成的示例视频**（`samples/`），覆盖不同场景，几十秒就能出结果：

  | 样例 | 时长 | 用途 |
  |---|---|---|
  | `demo_zh_math.mp4` | 42s | 中文数学课，含 `u`/`v` 中英混排符号 —— 默认引擎的强项 |
  | `demo_en_ml.mp4` | 92 词 | 英文梯度下降，术语密集 —— 两个引擎的 WER 对比 |
  | `demo_zh_daily.mp4` | 29s | 中文日常口语，停顿多 —— 看 VAD 与字幕分段 |

- **`scripts/make_samples.py`**：生成这些样例的脚本，**全程离线** ——
  用本机自带语音合成（Windows SAPI / macOS `say` / Linux `espeak-ng`）念我们自己写的稿子，
  再与纯色背景合成。不引入任何第三方素材，避免版权问题（这也是 `samples/formula.mp4`
  一直 gitignore 的原因）。逐句合成后按停顿拼接，VAD 才有东西可切。
- 画面上**说到哪句显示哪句**（逐句 `drawtext` + `enable=between(...)`），
  边听边看就能发现识别错在哪；每句还配 `demo_*.transcript.txt` 逐句稿子，
  可直接喂给 `experiments/wer_report.py` 算 WER
- `samples/README.md` 重写：4 个样例的对照表 + **可直接复制粘贴的命令集合**（含
  两引擎对比跑 WER 的完整流程）；README 与 docs/deploy.md 也各加了一段「先拿样例试跑」
- 本机实测：中文样例 7 句 → 识别出 7 段，与稿子几乎逐字一致；
  英文样例 WER 0.00%（Qwen3-ASR-1.7B）

### 变更：Qwen3-ASR 成为默认引擎，并支持「只部署 Qwen、不装 Whisper」

- **默认引擎改为 `qwen3-asr`**。两个都装了就用 Qwen；
  **只装了一个就用那个**（按部署情况自适应），所以走"只装 Whisper"路线的用户
  不需要每次加 `--engine`。解析优先级：`--engine` > `V2C_ENGINE` > 本机可用的引擎 > 首选 Qwen
- **依赖分层**，两个引擎可以只装一个：
  `requirements-base.txt`（公共：ffmpeg）→ `requirements.txt`（+ Whisper）／
  `requirements-qwen.txt`（+ Qwen，装在独立的 `.venv-qwen`）。
  `pyproject.toml` 的 `dependencies` 只留公共依赖，Whisper 变成 extra：`pip install "video2context[whisper]"`
- **安装脚本支持选引擎**：`setup.ps1 -Engine qwen|whisper|both`（默认 qwen）、
  `setup.sh qwen|whisper|both`；**模型默认一起下**（`-SkipModels` / `SKIP_MODELS=1` 才是跳过）。
  `setup_qwen.ps1` / `setup_qwen.sh` 现在自身就是完整的 Qwen-only 部署（连主环境一起备好）
- **`doctor` 重写**：报告本机部署了哪些引擎、Qwen 解释器与模型状态、**实际会用哪个引擎**；
  标签按东亚字宽对齐；只装了 Qwen 时用 `nvidia-smi` 兜底探测显卡
- **强制对齐模型改为默认安装**（Qwen 路线，1.8GB）。它是字幕时间精度的分界线：
  Qwen3-ASR 不输出时间戳，字幕时间由本工程生成，没有对齐模型时只能估算，
  而只装 Qwen 的机器又没有 VAD，会掉到最粗的一档（在整条音轨上平均分配）。
  运行时会打印对齐模型的三态（就绪 / 本地没有 / 已关闭），不再等到字幕出来才发现时间不对
- 只装 Qwen 时没有 faster-whisper，也就没有 Silero VAD，无对齐模型的退化路径会
  **明确说明原因**并指向"装对齐模型"这个正解，而不是抛 `ModuleNotFoundError`
- **新增 `docs/deploy.md`**：从 0 部署手册 —— 三条路线、各设备/平台、无显卡与纯 CPU、
  模型下载、离线内网、国内加速、磁盘占用与卸载、按症状索引
- 实测补充：Qwen3-ASR-1.7B 在 **CPU 上是 0.5× 实时**（12 秒音频跑 23.9 秒，比实时还慢），
  因此没有 N 卡不推荐 Qwen 路线

### 修复

- `resolve_engine()` 里 `normalize_engine("")` 会返回首选引擎（非空），
  导致"按部署情况自适应"那一档永远轮不到 —— 由新增的单测抓到
- 流水线测试原先只打补丁 `get_transcriber`，默认引擎改成自适应后会**真的去起 Qwen 子进程**、
  测试直接卡死；改为打补丁到统一的引擎工厂 `build_transcriber`，测试从此与"本机装了哪个引擎"无关
- `scripts/setup.sh` 里 `[ a ] || [ b ] && x=1` 的写法在 `set -e` 下会让脚本提前退出，
  改成显式 `if`；CI 增加 `bash -n` 与 `shellcheck -S error`

### 新增：识别引擎 `--engine`（faster-whisper / Qwen3-ASR）

- `--engine qwen3-asr`：接入阿里通义 **Qwen3-ASR-1.7B**（Apache-2.0）作为第二个识别引擎。
  实测中文数学课样例上 `u(x)` / `v(x)` / `导数` **全对且标点完整**（large-v3 分别为右/位/倒 ×13 处且无标点）；
  英文素材 **WER 0.00%**（large-v3 为 2.97%）
  —— 后续用仓库自带的 `samples/demo_en_ml.mp4` 复测，结论一致（0.00% vs 4.35%），
  文档正文统一改用可复现的那份数据，见「实测性能」
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

- 测试数 28 → 114：新增词表/配置单测与两遍解码的流水线集成测试（桩接大模型，不联网）；
  新增 `tests/test_qwen_engine.py`（51 项）：语种归一化、时间戳映射、切句、比例分配、
  引擎选择与默认引擎解析、解释器探测
  另新增 `docs/deploy.md` 的链接与锚点校验（`check_docs.py` 现在覆盖 11 个文档）

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
