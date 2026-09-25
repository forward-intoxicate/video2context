# 更新日志

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

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
