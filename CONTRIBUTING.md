# 参与贡献

欢迎提 Issue 和 PR。这个工程的目标是：**把「视频转文字」做成任何人都能在自己机器上跑起来的东西**。

## 开发环境

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt   # 含 pytest / ruff
```

需要本地网页界面时再装 `pip install -r requirements-web.txt`。

## 提交前自检

```bash
python -m unittest discover -s tests -v   # 单元测试（离线，不需要模型/显卡）
ruff check video2context tests scripts    # 代码风格
ruff format --check video2context tests   # 格式（可选）
python scripts/check_docs.py              # 校验文档里的相对链接与锚点是否有效
```

测试设计原则：**不联网、不下载模型、不依赖显卡**。需要媒体素材时用
`tests/media_factory.py` 里的 ffmpeg 现场合成，需要识别结果时用替身对象
（见 `tests/test_pipeline.py` 的 `StubTranscriber`）。

## 代码约定

- 面向用户的文案、注释、文档一律用中文；标识符用英文
- 新增依赖要慎重，能不加就不加（当前第三方依赖只有 faster-whisper + imageio-ffmpeg）
- 新增命令行参数必须同时更新：`README.md` 参数表、`docs/` 对应说明
- 提交信息建议用 `feat: / fix: / docs: / test: / chore:` 前缀

## 想扩展的方向

| 方向 | 改哪里 |
|---|---|
| 换中文更强的识别引擎（FunASR / SenseVoice） | `video2context/transcriber.py`，保持 `Transcriber.transcribe()` 的输入输出契约 |
| 新增导出格式（如 ASS 字幕、Markdown） | `video2context/writers.py`，记得加到 `SUPPORTED_FORMATS` 并补测试 |
| 中英互译 / 内容摘要 | 在 `pipeline.py` 里对 `payload["segments"]` 后处理，或新增独立模块 |
| 说话人分离 | 接 `pyannote.audio`，给每个 segment 补 `speaker` 字段 |
| 新的模型下载源 | `scripts/download_model.py` 的 `file_url()` / `list_files()` |

## 报告问题

请附上 `python -m video2context doctor` 的输出、完整报错信息，以及
（如果方便）出问题的视频的基本信息：时长、音轨采样率、是中文还是英文。
