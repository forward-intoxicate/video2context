# 开发指南

面向想改代码的人。使用说明见 [README](../README.md)，实现原理见 [architecture.md](architecture.md)。

## 1. 环境

```bash
git clone https://github.com/forward-intoxicate/video2context.git
cd video2context

python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1

pip install -r requirements-dev.txt  # 核心依赖 + pytest + ruff
pip install -r requirements-web.txt  # 改网页界面时才需要
```

以可编辑模式安装（可选，装完可以直接用 `video2context` 命令）：

```bash
pip install -e .
video2context doctor
```

## 2. 测试策略

```bash
python -m unittest discover -s tests -v   # 28 项，约 2 秒
pytest                                      # 同样的用例（pyproject 已配好）
```

设计原则：**不联网、不下载模型、不依赖显卡、不依赖外部素材**。这样 CI 才跑得起来、跑得快。

| 测试文件 | 覆盖内容 | 依赖 |
|---|---|---|
| `tests/test_units.py` | 时间戳格式化、SRT/VTT/TXT 序列化、参数解析、OOM 判定、ffmpeg 探测与抽音轨 | ffmpeg（自动跳过） |
| `tests/test_pipeline.py` | 完整流水线：JSON schema、四种输出格式、命名避让、临时音频清理、错误路径 | ffmpeg |
| `tests/media_factory.py` | 用 ffmpeg lavfi 现场合成测试视频/音频 | ffmpeg |

两个关键手法：

1. **现场造素材**：`tests/media_factory.py` 用 `lavfi` 生成正弦波与色块，不往仓库塞二进制文件。
2. **替身识别器**：`tests/test_pipeline.py` 的 `StubTranscriber` 接口与真实 `Transcriber` 一致，
   但直接返回固定分段。因此可以在没有 GPU、没有模型的情况下验证整条链路：

```python
patcher = mock.patch("video2context.pipeline.get_transcriber", side_effect=_factory)
```

> 注意打补丁的位置是 `video2context.pipeline.get_transcriber`（导入到使用处的名字），
> 不是 `video2context.transcriber.get_transcriber`。

## 3. 代码结构

```
video2context/
├─ ffmpeg_tools.py   媒体层：定位 ffmpeg、探测、抽音轨（无第三方依赖，纯 subprocess）
├─ transcriber.py    识别层：模型加载、设备选择、显存预检、降级、缓存
├─ pipeline.py       编排层：把上面两层串成"输入文件 → 结构化结果"
├─ writers.py        输出层：JSON / TXT / SRT / VTT
├─ cli.py            交互层：命令行
└─ webui.py          交互层：Gradio
```

依赖方向是单向的：`cli/webui → pipeline → {ffmpeg_tools, transcriber, writers}`。
**不要让下层反过来 import 上层**，也不要让 `transcriber.py` 依赖 `pipeline.py`。

### 数据契约

```python
Segment(id, start, end, text, avg_logprob, no_speech_prob, temperature, compression_ratio, words)
TranscriptionResult(segments, text, language, language_probability, duration, duration_after_vad,
                    task, model, device, compute_type, batch_size, vad_filter, beam_size,
                    initial_prompt, condition_on_previous_text, temperature,
                    elapsed_seconds, model_load_seconds)
```

`TranscriptionResult.to_dict()` 直接变成 JSON 里的 `asr` 段。
换识别引擎时，只要保证这两个类型不变，上层全部无需修改。

## 4. 常见改动怎么做

### 4.1 加一个命令行参数

1. `cli.py` 的 `build_parser()` 加 `add_argument`；
2. `pipeline.TranscribeOptions` 加字段（如果它要往下传）；
3. `pipeline.process()` 里把值传给 `transcriber.transcribe()`；
4. `webui.py` 如需暴露再加控件；
5. 更新 `README.md` 的参数表、必要时更新 `docs/`。

### 4.2 加一种输出格式

1. `writers.py`：写 `segments_to_xxx()`，并把格式名加进 `SUPPORTED_FORMATS`；
2. `write_outputs()` 里加分支；
3. `tests/test_units.py` 加断言，`tests/test_pipeline.py::test_all_output_formats` 加进去；
4. 更新 README 的参数表与输出说明。

### 4.3 换识别引擎

参照 [docs/models.md](models.md) 的「换成 FunASR / SenseVoice」一节：实现一个类，方法签名与
`Transcriber.transcribe()` 一致，返回 `TranscriptionResult`；
然后在 `get_transcriber()` 里按 `--model` 或新参数分发。

### 4.4 调整 VAD / 解码默认值

`transcriber.py` 里改完，**务必用 `samples/chinese_speech_sample.mp4` 对比一次**：

```bash
python -m video2context samples/chinese_speech_sample.mp4 --model-dir models/faster-whisper-large-v3 -o /tmp/after --overwrite
# 关注：segments 数量、总耗时、文本是否正确
```

## 5. 提交前检查清单

```bash
python -m unittest discover -s tests -v     # 必须全绿
ruff check video2context tests scripts      # 必须 0 警告
python scripts/check_docs.py                # 文档链接与锚点必须全部有效
python -m video2context doctor              # 环境自检正常
```

再确认一遍：

- [ ] 新增/修改的参数已同步到 `README.md` 参数表
- [ ] 涉及行为变化时更新了 `CHANGELOG.md`
- [ ] 新功能带了测试；没带的在 PR 描述里说明原因
- [ ] 面向用户的报错信息是**中文**且**可执行**（告诉用户下一步做什么）

## 6. 发布流程

1. 更新 `pyproject.toml` 与 `video2context/__init__.py` 里的版本号；
2. 在 `CHANGELOG.md` 顶部加一节（Keep a Changelog 风格）；
3. 打 tag：`git tag -a v0.1.0 -m "v0.1.0" && git push origin v0.1.0`；
4. 想发 PyPI：`python -m build && twine upload dist/*`（需要先在 PyPI 配好 token）。

## 7. CI

`.github/workflows/ci.yml` 做两件事：

- **test**：矩阵 `ubuntu(3.9, 3.13) / windows(3.11) / macos(3.11)`，
  装 `requirements.txt`（不含 gradio 与 CUDA 库）后跑全部单元测试 + `doctor`；
- **lint**：`ruff check`。

CI 里**不需要**显卡、模型或 API Key —— 这正是测试设计成"离线替身"的收益。

## 8. 代码风格

- `ruff` 配置见 `pyproject.toml`（`line-length = 110`，`target-version = py39`）；
- 保留 `Optional[X]` 而不是 `X | None`：Gradio 会在**运行时**读取函数注解，
  3.9 上求值 `X | None` 会抛 `TypeError`（因此 `UP045` 被显式忽略）；
- 面向用户的字符串、注释、文档用中文；标识符用英文；
- 报错信息尽量给出"下一步怎么做"，例如显存不足时直接给出可用命令。
