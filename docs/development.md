# 开发指南

面向想改代码的人。使用说明见 [README](../README.md)，实现原理见 [architecture.md](architecture.md)。

## 1. 环境

```bash
git clone https://github.com/forward-intoxicate/video2context.git
cd video2context

python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1

pip install -r requirements-dev.txt  # 公共依赖 + Whisper 引擎 + pytest + ruff
pip install -r requirements-web.txt  # 改网页界面时才需要
```

依赖是**分层**的，改代码前先搞清楚自己动的是哪一层：

| 文件 | 内容 | 谁依赖它 |
|---|---|---|
| `requirements-base.txt` | `imageio-ffmpeg` | 所有代码路径（抽音轨） |
| `requirements.txt` | base + `faster-whisper` | **只有** Whisper 引擎 |
| `requirements-qwen.txt` | `qwen-asr` + torch 等（装在 `.venv-qwen`） | **只有** Qwen 引擎 |
| `requirements-web.txt` | + Gradio | 只有网页界面 |

所以主包**不能**在模块顶层 import `faster_whisper` 或任何 torch 相关的东西 ——
只会用 Qwen 的用户根本没装它们。`transcriber.py` 里的 `faster_whisper` 是延迟导入，
`qwen_engine._speech_regions()` 的 VAD 也是（拿不到就退化），这两处是范例。

改 Qwen 引擎时才需要独立环境（约 3GB）：

```bash
powershell -ExecutionPolicy Bypass -File scripts\setup_qwen.ps1   # Windows
bash scripts/setup_qwen.sh                                        # macOS / Linux
```

以可编辑模式安装（可选，装完可以直接用 `video2context` 命令）：

```bash
pip install -e .                     # 只装公共依赖
pip install -e ".[whisper]"          # 想同时装 Whisper 引擎
video2context doctor
```

## 2. 测试策略

```bash
python -m unittest discover -s tests -v   # 101 项，约 3 秒
pytest                                      # 同样的用例（pyproject 已配好）
```

设计原则：**不联网、不下载模型、不依赖显卡、不依赖外部素材**。这样 CI 才跑得起来、跑得快。

| 测试文件 | 覆盖内容 | 依赖 |
|---|---|---|
| `tests/test_units.py` | 时间戳格式化、SRT/VTT/TXT 序列化、参数解析、OOM 判定、ffmpeg 探测与抽音轨 | ffmpeg（自动跳过） |
| `tests/test_pipeline.py` | 完整流水线：JSON schema、四种输出格式、命名避让、临时音频清理、错误路径 | ffmpeg |
| `tests/test_glossary.py` | 词表解析、prompt/context 拼装、符号命中评分、回退判定 | 无 |
| `tests/test_qwen_engine.py` | Qwen 引擎：语种归一化、时间戳→下标映射、**切句**、无对齐模型时的比例分配、引擎选择与默认引擎解析、解释器探测 | 无 |
| `tests/media_factory.py` | 用 ffmpeg lavfi 现场合成测试视频/音频 | ffmpeg |

三个关键手法：

1. **现场造素材**：`tests/media_factory.py` 用 `lavfi` 生成正弦波与色块，不往仓库塞二进制文件。
2. **替身识别器**：`tests/test_pipeline.py` 的 `StubTranscriber` 接口与真实 `Transcriber` 一致，
   但直接返回固定分段。因此可以在没有 GPU、没有模型的情况下验证整条链路：

```python
patcher = mock.patch("video2context.pipeline.get_transcriber", side_effect=_factory)
```

3. **纯函数化 Qwen 的切句逻辑**：`group_time_stamps()` / `distribute_sentences()` 只吃
   `(文本, 时间戳列表)`，所以能用几行合成数据把边界情况测透，不需要加载 4GB 模型：

```python
segs = group_time_stamps("Hello world. This is, a test!", EN_STAMPS)   # token 是去标点的
assert [s.text for s in segs] == ["Hello world.", "This is, a test!"]
```

> 注意打补丁的位置是 `video2context.pipeline.build_transcriber`（**引擎工厂**），
> 不是某个具体引擎的 `get_transcriber` ——
> 默认引擎会按"本机装了哪个引擎"自适应，只替换 `get_transcriber` 的话，
> 在装了 `.venv-qwen` 的开发机上会真的去起 Qwen 子进程，测试直接卡死（踩过）。
> 同理，凡是断言"引擎行为"的用例都要**显式指定** `TranscribeOptions(engine=...)`。

## 3. 代码结构

```
video2context/
├─ ffmpeg_tools.py   媒体层：定位 ffmpeg、探测、抽音轨（无第三方依赖，纯 subprocess）
├─ transcriber.py    识别层 A：faster-whisper —— 模型加载、设备选择、显存预检、降级、缓存
├─ qwen_engine.py    识别层 B：Qwen3-ASR —— 环境探测、子进程调度、时间戳切句
├─ _qwen_worker.py   识别层 B 的子进程入口（在 .venv-qwen 里跑，只依赖标准库 + torch）
├─ glossary.py       词表层：符号还原、prompt/context 拼装、偏置评分
├─ llm.py            大模型客户端（OpenAI 兼容，urllib 实现，零新依赖）
├─ pipeline.py       编排层：把上面几层串成"输入文件 → 结构化结果"，并选择引擎
├─ writers.py        输出层：JSON / TXT / SRT / VTT
├─ cli.py            交互层：命令行
└─ webui.py          交互层：Gradio
```

依赖方向是单向的：`cli/webui → pipeline → {ffmpeg_tools, transcriber, qwen_engine, writers}`。
**不要让下层反过来 import 上层**，也不要让 `transcriber.py` 依赖 `pipeline.py`。

`_qwen_worker.py` 是个特例：它**刻意不 import 主工程任何模块**（那会连带导入
faster-whisper，而它跑在没装 faster-whisper 的 `.venv-qwen` 里）。它还额外做了一件事 ——
把脚本自身所在目录从 `sys.path` 里摘掉，否则 `video2context/` 下的模块会**遮住**
site-packages 里的同名包（这个坑真实踩过：曾经的 `qwen_asr.py` 遮住了上游 `qwen_asr` 包）。

### 数据契约

```python
Segment(id, start, end, text, avg_logprob, no_speech_prob, temperature, compression_ratio, words)
TranscriptionResult(segments, text, language, language_probability, duration, duration_after_vad,
                    task, model, device, compute_type, batch_size, vad_filter, beam_size,
                    initial_prompt, condition_on_previous_text, temperature,
                    elapsed_seconds, model_load_seconds, engine, extra)
```

`TranscriptionResult.to_dict()` 直接变成 JSON 里的 `asr` 段；`extra` 会被平铺进去，
留给各引擎放自己的元信息（Qwen 用它写 `aligner` / `timestamp_source` / `context_effect`）。

两个约定：

* `engine` 字段标明结果出自哪个引擎（`faster-whisper` / `qwen3-asr`），方便对比与溯源；
* 引擎**用不到**的字段不要硬编一个看似生效的值 —— 例如 Qwen 没有 VAD 与束搜索概念，
  `vad_filter` / `beam_size` 会被置成 `False` / `0`，而不是沿用 whisper 的默认值。

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

### 4.3 换识别引擎 / 加一个新引擎

**已经在做的范例**：`--engine qwen3-asr` 就是按这个套路加进来的，可以照抄：

1. 写一个类，方法签名与 `Transcriber.transcribe()` 一致，返回 `TranscriptionResult`；
   `__init__` 多出来的参数用 `**_ignored` 兜住，这样上层换引擎时不用改调用点；
2. 在 `pipeline.build_transcriber()` 里按 `opts.engine` 分发，并把它加进 `ENGINE_CHOICES` / `ENGINE_ALIASES`；
3. `cli.py` 加 `--engine` 与引擎私有参数，同步 `TranscribeOptions` 字段；
4. 如果新引擎**没有时间戳**，必须自己造分段（参考 `qwen_engine.group_time_stamps()` 与
   `distribute_sentences()`），否则 SRT/VTT 会退化成一条巨型字幕；
5. 补测试：切句这类纯函数直接测（`tests/test_qwen_engine.py`）；引擎分发用
   `build_transcriber()` 断言返回类型。

只换 Whisper 家族之外的实现（FunASR/SenseVoice）同样适用，见
[docs/models.md](models.md) 的「换成 FunASR / SenseVoice」一节。

### 4.4 调整 VAD / 解码默认值

`transcriber.py` 里改完，**务必用 `samples/chinese_speech_sample.mp4` 对比一次**：

```bash
python -m video2context samples/chinese_speech_sample.mp4 --model-dir models/faster-whisper-large-v3 -o /tmp/after --overwrite
# 关注：segments 数量、总耗时、文本是否正确
```

### 4.5 调 Qwen 的切句规则

`qwen_engine.group_time_stamps()` 的 `max_chars` / `max_duration` / `gap` 阈值决定字幕长短。
改完建议拿真实素材肉眼检查，并确认**拼接结果仍等于原文**：

```python
segs = group_time_stamps(text, stamps)
assert "".join(s.text for s in segs).replace(" ", "") == text.replace(" ", "")   # 不该丢字
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

`.github/workflows/ci.yml` 做四件事：

- **test**：矩阵 `ubuntu(3.9, 3.13) / windows(3.11) / macos(3.11)`，
  装 `requirements-dev.txt`（只有公共依赖 + Whisper，**不含 torch**）后
  跑全部单元测试 + `doctor`；
- **lint**：`ruff check`、`check_docs.py`、`check_secrets.py`；
- **PowerShell 编码校验**：`scripts/*.ps1` 必须以 **UTF-8 BOM** 开头
  （Windows PowerShell 5.1 会按 GBK 解析无 BOM 的 UTF-8，中文会被截断），并检查语法。
  **新增 `.ps1` 记得同时加进这个列表**；
- **shell 脚本校验**：`bash -n` + `shellcheck -S error`。
  部署脚本是很多人接触本工程的第一步，语法错误必须在这里拦住
  （完整跑一遍会下 3GB 的 torch，不适合放进 CI）。

CI 里**不需要**显卡、模型或 API Key —— 这正是测试设计成"离线替身"的收益。
也**不需要** `.venv-qwen`：Qwen 引擎的测试同样不碰 torch，切句与分配都是纯函数；
而凡是要断言引擎行为的用例都显式指定了引擎，不受"本机装了哪个"影响。

## 8. 代码风格

- `ruff` 配置见 `pyproject.toml`（`line-length = 110`，`target-version = py39`）；
- 保留 `Optional[X]` 而不是 `X | None`：Gradio 会在**运行时**读取函数注解，
  3.9 上求值 `X | None` 会抛 `TypeError`（因此 `UP045` 被显式忽略）；
- 面向用户的字符串、注释、文档用中文；标识符用英文；
- 报错信息尽量给出"下一步怎么做"，例如显存不足时直接给出可用命令。
