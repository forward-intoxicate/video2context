# 更新日志

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

### 新增：领域词表（修同音词/专有名词）

- `--glossary`：手写词表（`错形 -> 正确` 或纯术语），完全离线
- `--auto-glossary`：自动两遍解码 —— 采样 90 秒粗转写 → 大模型推断符号还原 → 带词表正式转写
- `--dry-run-glossary`：只生成词表供人工复核（推荐流程）
- **验证 + 回退**：带词表结果里"正确符号一个未命中且仍有错误写法残留"即判定偏置失效，自动回退到无词表版本；
  另报告逐符号覆盖率 `coverage`（比总命中次数更能反映真实效果）
- 修复：`--device` 自动降级到 CPU 时误用 GPU 专用精度（`float16`/`int8_float16`）导致加载失败
- 新增模块：`glossary.py`（词表与评分）、`llm.py`（OpenAI 兼容客户端，零新依赖）、`config.py`（.env 与密钥脱敏）
- JSON 增加可选 `glossary` 段（词表、来源、大模型用量、校验结果）；`schema_version` 升到 `1.1`

### 安全

- `.env` / `.env.example`：密钥只从环境变量或 `.env` 读取，**不提供任何命令行参数传密钥**
- `scripts/check_secrets.py`：提交前防泄露扫描（已接入 CI）；开发过程中它就抓到过一次真实密钥误入测试文件
- 日志、报错、JSON 里的密钥统一打码；`doctor` 只显示 `sk-***abcd`

### 文档

- 新增 `docs/glossary.md`：含 9 个 prompt 变体的实验数据（含"四个假设全被否定"的记录）

### 测试

- 测试数 28 → 54：新增词表/配置单测与两遍解码的流水线集成测试（桩接大模型，不联网）

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
