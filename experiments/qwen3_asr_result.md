# Qwen3-ASR 实测结果（`samples/formula.mp4`，72.6 秒中文导数课）

测试日期：2026-09-27｜机器：i9-13900HX + RTX 4060 Laptop 8GB + 15.7GB 内存

## 结论速览

| 模型 | 本机 | 下载 | 加载 | 推理 | 中文效果 |
|---|---|---|---|---|---|
| **Qwen3-ASR-1.7B** | ✅ 跑通 | 4.70GB | 4.8s | 8.53s（**8.5 倍实时**） | **几乎完美** |
| Qwen3-ASR-0.6B | ✅ 跑通 | 1.88GB | 2.4s | 8.34s（8.7 倍实时） | 标点好，`u` 仍错 |
| faster-whisper large-v3 | ✅ 跑通 | 3.1GB | 5.2s | 6.49s（11 倍实时） | 无标点、同音错字多 |

**1.7B 需要约 13GB 提交内存（commit）才能加载**，详见文末「加载失败的坑」。

## 三引擎逐项对照（同一段音频）

| 检查项 | Whisper large-v3 | Qwen3-ASR-0.6B | **Qwen3-ASR-1.7B** |
|---|---|---|---|
| u/v 写成拉丁字母 | 5 | 4 | **13** |
| 【错】`u`→「右」 | 4 | 9 | **0** |
| 【错】`v`→「位」 | 4 | 0 | **0** |
| 「导数/求导」正确 | 4 | 9 | **9** |
| 【错】「倒数/求倒/求到」 | 5 | 0 | **0** |
| 「再求导」正确 | 2 | 3 | 3 |
| 【错】「在求导」 | 0 | 0 | 0 |
| 句末标点（。！？） | **0** | 6 | 5 |
| 逗号 | **0** | 15 | 11 |

### 三家全文对照

**faster-whisper large-v3**（无标点、`u/v` 全错、把"导数"写成"倒数"）：

```
右F4和位F4如果在F0处可倒那么右加位的倒数等于右先求倒再加位求倒注意这个公式它的前提条件
是右位都已经说了可倒了才能把这个倒数把它带进去先相加再求倒等于先求导再相加另外一个就是
如果是多个表达式的时候比方说是U1X一直加到UNX有N个表达式也可以把它带进去U1先求导U2再求导
一直到UN再求导这个公式啊大家几乎不用花太多的时间因为
```

**Qwen3-ASR-0.6B**（标点完整、`v` 对、`u` 仍错）：

```
右f和v f，如果在f零处可导，那么，右加v的导数等于右先求导，再加v求导。注意这个公式啊…
```

**Qwen3-ASR-1.7B**（**`u(x)`/`v(x)` 全对，含括号**）：

```
u(x)和v(x)如果在x零处可导，那么u加v的导数等于u先求导再加v求导。注意这个公式啊，它的前提
条件是u、v啊都已经说了可导了，才能把这个导数把它带进去。先相加再求导，等于先求导再相加。
另外一个就是，如果是多个表达式的时候，比方说是u1x一直加到u nx有n个表达式，也可以把它带进去。
u1先求导，u2再求导，一直到u n再求导。这个公式啊，大家几乎不用花太多的时间，因为
```

**结论**：在中文长句 + 含拉丁字母符号的场景下，**1.7B 明显优于 Whisper large-v3**，
而且**完全不需要提示词或词表** —— 它自己就把 `u(x)`、`v(x)` 写对了（Whisper 写成了"右F4/位F4"）。
剩余小瑕疵：`x零`（应为 `x0`）、`u1x`（应为 `u1(x)`）。

## `context` 参数：真的能用，但给法有讲究

`Qwen3ASRModel.transcribe(audio, context, language, return_time_stamps)` —— **`context` 会作为
system message 注入**，这是真正的上下文偏置（比 Whisper 那个会静默失效的 prompt 可靠）。

但实测三种给法差异巨大：

| context 内容 | 结果 |
|---|---|
| `请准确识别以下术语与符号：u(x)、v(x)…`（指令式） | ❌ **模型原样复述 context，不识别音频**（耗时 3.4s vs 正常 8.5s） |
| `u(x) v(x) u'(x) v'(x)…`（纯术语列表） | ❌ **同样复述** |
| `2.2 求导法则 1) u(x) v(x) (u(x)+v(x))'=u'(x)+v'(x)…`（**完整句子/板书原文**） | ✅ 正常识别，且 **`u1x` → `u1(x)`**（括号修好了） |

**用法结论**：`context` 要喂**自然文本（句子级）**，例如板书原文、领域段落、对话历史；
**不要喂简短的术语列表** —— 那会被当成"要输出的内容"而直接复述。

## 加载失败的坑（为什么必须重启）

重启前 1.7B 三种加载方式全部失败：

| 尝试 | 结果 |
|---|---|
| 标准加载 | `CUDA out of memory`（transformers 的显存预热申请 3.2GB） |
| 跳过显存预热 | `OSError: 页面文件太小 (1455)` |
| 低内存加载（分片直进显存） | 进程被杀 → `0xC0000005` 访问违例 |

重启前后对比：

| 指标 | 重启前 | 重启后 |
|---|---|---|
| 提交内存余量 | 5.8 ~ 7.8 GB | **12.7 GB** |
| 结果 | ❌ 崩溃 | ✅ 加载 4.8s 成功 |

**根因**：transformers 加载 safetensors 时会把**整个分片读进 CPU 内存**（`safe_open` 不支持
`map_location`），1.7B 的第一个分片就有 **4.02GB**，加上 state_dict/模型/副本需要 ~8GB 以上提交内存。
重启前系统已提交 33.6GB（15.7GB 物理内存 + 24GB 页面文件已到上限），余量不足；
余量不足时不是抛干净的 MemoryError，而是 **mmap 访问直接段错误**。

顺带：`pip cache purge` 释放了 C 盘 4GB（0.5GB → 4.2GB），否则页面文件连现状都维持不住。

## 复现方式

```powershell
# 独立环境（不污染主工程 .venv）
powershell -ExecutionPolicy Bypass -File experiments\setup_qwen3_asr.ps1
.\.venv-qwen\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.14.0+cu126"
.\.venv\Scripts\python scripts\download_model.py --repo Qwen/Qwen3-ASR-1.7B --source modelscope --out models\Qwen3-ASR-1.7B

# 跑
.\.venv-qwen\Scripts\python experiments\qwen3_asr_bench.py samples\formula.mp4 --model models\Qwen3-ASR-1.7B
.\.venv-qwen\Scripts\python experiments\qwen3_asr_bench.py samples\formula.mp4 --model models\Qwen3-ASR-1.7B `
    --context "2.2 求导法则 (u(x)+v(x))'=u'(x)+v'(x)"
```

## 要不要集成进主工程？

**优点**：中文效果（标点、同音字、拉丁字母）明显更好，且不需要词表就能处理符号。
**代价**：

| 项 | 说明 |
|---|---|
| 独立环境 | torch ~3GB + qwen-asr 依赖，主工程要保持 torch-free 就得用子进程调用 |
| 模型体积 | 4.7GB（vs faster-whisper large-v3 的 3.1GB） |
| 内存门槛 | 加载需 ~13GB 提交内存；8GB 显存够用，但 16GB 内存机器必须先腾内存 |
| **时间戳** | **必须另下 `Qwen3-ForcedAligner-0.6B`（1.84GB）**，否则只有整段文本，出不了 SRT/VTT |
| Windows | 没有 vLLM，只能用 transformers 后端（本机实测 8.5 倍实时，够用） |
| 速度 | 8.5 倍实时 vs Whisper 的 11 倍实时（差距不大） |

**建议**：作为**可选第二引擎**集成（`--engine qwen3-asr`，子进程调用 `.venv-qwen`），
中文内容默认用它、英文仍用 Whisper；集成前需先下 ForcedAligner 以支持字幕输出。
