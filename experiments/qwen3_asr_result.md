# Qwen3-ASR 实测结果

测试日期：2026-09-27 ~ 2026-09-28｜机器：i9-13900HX + RTX 4060 Laptop 8GB + 15.7GB 内存

> **本文档记录的是决策依据，包含一次自我纠错**（见「`context` 参数」一节）：
> 早期实验认为 `context` 有效，后续做了更严格的 A/B 对照后**推翻了这个结论**。
> 保留整个过程，是为了让后来的人知道「看似有效的观察」可能是怎么骗过人的。

## 结论速览

| 模型 | 本机 | 下载 | 加载 | 推理 | 中文效果 |
|---|---|---|---|---|---|
| **Qwen3-ASR-1.7B** | ✅ 跑通 | 4.70GB | 4.8s | 8.53s（**8.5 倍实时**） | **几乎完美** |
| Qwen3-ASR-0.6B | ✅ 跑通 | 1.88GB | 2.4s | 8.34s（8.7 倍实时） | 标点好，`u` 仍错 |
| faster-whisper large-v3 | ✅ 跑通 | 3.1GB | 5.2s | 6.49s（11 倍实时） | 无标点、同音错字多 |

**1.7B 需要约 13GB 提交内存（commit）才能加载**，详见文末「加载失败的坑」。

集成进主工程后（带强制对齐模型、走流水线）实测：

| 素材 | 加载 | 推理 | 相对实时 |
|---|---|---|---|
| 中文 73s（1.7B + 对齐） | 6.6s | 5.8s | **12.5×** |
| 英文 42s（1.7B + 对齐） | 10.9s | 8.8s | 4.8× |

## 三引擎逐项对照（`samples/formula.mp4`，72.6 秒中文导数课）

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

## 英文质量（42 秒 TTS，101 词，有逐字标准答案）

用 `experiments/wer_report.py` 对同一段 16kHz wav 计算 WER（转小写、去标点后逐词比对）：

| 引擎 | WER | 错误 |
|---|---|---|
| **Qwen3-ASR-1.7B** | **0.00%** | 无 |
| faster-whisper large-v3 | 2.97% | `Adam`→`Atom`（并多切一个词）、`RMSProp`→`RMS Prop` |

两家都把 `Adam`、`RMSProp` 这类专有人名/产品名识别出来了，但 large-v3 在
连读处切错了词；1.7B 完全正确。**英文不再是 faster-whisper 的护城河。**

## 时间戳：靠 `Qwen3-ForcedAligner-0.6B`

Qwen3-ASR 本身**只输出一整段文字**，没有分段与时间戳。加了对齐模型后得到词级时间戳：

```
[{"text": "Today", "start": 0.16, "end": 0.56},
 {"text": "we're", "start": 0.56, "end": 0.72},
 {"text": "going", "start": 0.72, "end": 0.96}, ...]
```

（101 个 token 对应 101 个英文词。）

但 token 是**去掉标点的**（`is_kept_char()` 只保留字母、数字、撇号），所以不能直接拼成句子。
主工程 `qwen_engine.group_time_stamps()` 的做法是：把原文按"保留字符"压成参考串并记下
每个字符的原文下标 → 每个 token 在参考串里顺序查找（带 8 字符容差自我纠偏）→ 映射回原文下标 →
按句末标点/停顿/长度切段 → 切片时把标点补回来。

实测效果：42 秒英文切出 7 段，`"".join(段文字)` 与原文归一化后**都是 525 字符，标点一个不丢**。

## `context` 参数：**早期结论是错的**

### 早期实验（2026-09-27）

`Qwen3ASRModel.transcribe(audio, context, language, return_time_stamps)` 会把 `context`
作为 **system message** 注入。当时的对照表是：

| context 内容 | 当时观察到的结果 |
|---|---|
| `请准确识别以下术语与符号：u(x)、v(x)…`（指令式） | ❌ 模型原样复述 context，不识别音频（3.4s vs 正常 8.5s） |
| `u(x) v(x) u'(x) v'(x)…`（纯术语列表） | ❌ 同样复述 |
| `2.2 求导法则 1) u(x) v(x) (u(x)+v(x))'=u'(x)+v'(x)…`（完整句子/板书原文） | ✅ **看起来正常**，而且 **`u1x` → `u1(x)`** |

据此当时写下结论：「`context` 要喂自然文本（句子级），不要喂简短术语列表」。

### 复查实验（2026-09-28）：结论翻车

集成时做了更严格的 A/B（**固定其它所有变量，只改 context，逐字比输出**）：

| 传入的 context | 输出长度 | 与基准是否逐字相同 |
|---|---|---|
| 不传（基准） | 188 字 | — |
| 「视频里写出的公式和符号有u(x)、v(x)、x0、u1(x)。讲到的术语包括导数运算法则。」 | 188 字 | ✅ **完全相同** |
| 「这段视频讲解导数的运算法则。板书内容是：函数u(x)和v(x)在点x0处可导，则u+v的导数等于…」 | 188 字 | ✅ **完全相同** |
| 「这段音频讲的是量子力学，视频里写出的公式是E=mc^2。」（**故意完全无关**） | 188 字 | ✅ **完全相同** |

英文素材（625 字）同样做了三组对照，**全部逐字相同**，包括自动检测语言与强制指定语言两种情况。

也排除了"接线错误"：直接渲染 `processor.apply_chat_template`，确认 system 消息确实进了 prompt：

```
<|im_start|>system
这段音频讲的是量子力学。<|im_end|>
<|im_start|>user
<|audio_start|<|audio_pad|><|audio_end|><|im_end|>
<|im_start|>assistant
```

### 为什么会误判

三项原因叠在一起：

1. **没有固定变量**。早期那三行不是同一次实验的严格对照 —— "看起来正常"那一行
   用的可能已经是别轮的输出，缺少"不传 context"的同一时刻基准。
2. **把巧合当因果**。那一轮输出里 `u1x` 变成了 `u1(x)`，我就归因给 context；
   实际重复跑同一段音频，模型本身在不同轮次就可能给出 `u1x` 或 `u1(x)`。
3. **"复述 context" 是真的，"派上用场"是假的**。前两行（指令式、术语列表）的复述现象
   可稳定复现，所以当时以为整条链路是通的 —— 但通不代表有用，
   模型完全可以"读到了 context 却不用它"。

### 现在的结论

在 `qwen-asr==0.0.6` + `Qwen3-ASR-1.7B` 这一版里，**`context` 对识别结果没有可观测影响**。
主工程因此：照常生成词表、照常写进 JSON（供人工复核与复用），但在日志里明确提示
"不要指望它修同音词"，并在 JSON 里记录 `context_effect: "none-observed"`。
上游将来若让它生效，本工程不用改代码就能受益。

> **教训**：判断"某个输入有没有生效"，唯一可靠的办法是**逐字 diff 输出**。
> "结果看起来对" 什么都证明不了。

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

这两条都做成了代码里的默认行为：工人脚本**默认跳过显存预热**；
`--qwen-low-mem on` 走 meta 设备 + `safetensors.load_file(device="cuda:0")`，
让分片直接进显存，绕开"整片读进 CPU 内存"。

## 复现方式

```powershell
# 独立环境（不污染主工程 .venv）
powershell -ExecutionPolicy Bypass -File scripts\setup_qwen.ps1
.\.venv\Scripts\python scripts\download_model.py --repo Qwen/Qwen3-ASR-1.7B --source modelscope --out models\Qwen3-ASR-1.7B
.\.venv\Scripts\python scripts\download_model.py --repo Qwen/Qwen3-ForcedAligner-0.6B --source modelscope --out models\Qwen3-ForcedAligner-0.6B

# 实验脚本
.\.venv-qwen\Scripts\python experiments\qwen3_asr_bench.py samples\formula.mp4 --model models\Qwen3-ASR-1.7B
.\.venv-qwen\Scripts\python experiments\qwen3_asr_bench.py samples\formula.mp4 --model models\Qwen3-ASR-1.7B `
    --aligner models\Qwen3-ForcedAligner-0.6B
.\.venv\Scripts\python experiments\wer_report.py 标准答案.txt 候选.txt

# 复现"context 无效"：同一段音频跑两次，只改 context，比对输出
.\.venv-qwen\Scripts\python video2context\_qwen_worker.py --wav audio.wav --model models\Qwen3-ASR-1.7B --out a.json
.\.venv-qwen\Scripts\python video2context\_qwen_worker.py --wav audio.wav --model models\Qwen3-ASR-1.7B `
    --context "这段音频讲的是量子力学。" --out b.json
```

## 要不要集成进主工程？

**优点**：中文效果（标点、同音字、拉丁字母）明显更好，且不需要词表就能处理符号；英文 WER 也更低。
**代价**：

| 项 | 说明 |
|---|---|
| 独立环境 | torch ~3GB + qwen-asr 依赖，主工程要保持 torch-free 就得用子进程调用 |
| 模型体积 | 4.7GB（vs faster-whisper large-v3 的 3.1GB） |
| 内存门槛 | 加载需 ~13GB 提交内存；8GB 显存够用，但 16GB 内存机器必须先腾内存 |
| **时间戳** | **必须另下 `Qwen3-ForcedAligner-0.6B`（1.84GB）**，否则只有整段文本，出不了精确 SRT/VTT |
| Windows | 没有 vLLM，只能用 transformers 后端（本机实测 12.5 倍实时，够用） |
| 每次加载 | 子进程方案每次转写都要重新加载模型（6.6~10.9s） |

**结论：已作为可选第二引擎集成**（`--engine qwen3-asr`，子进程调用 `.venv-qwen`），
默认引擎仍是 faster-whisper；中文课程/术语密集内容建议切到 Qwen。
