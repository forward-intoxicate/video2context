# Qwen3-ASR 实测结果（`samples/formula.mp4`，72.6 秒中文导数课）

测试日期：2026-09-27｜机器：i9-13900HX + RTX 4060 Laptop 8GB + 15.7GB 内存

## 结论速览

| 模型 | 能否在本机跑 | 加载 | 推理 | 说明 |
|---|---|---|---|---|
| **Qwen3-ASR-0.6B** | ✅ **可以** | 2.4s | 8.34s（**8.7 倍实时**） | 标点完整、同音错字明显更少 |
| Qwen3-ASR-1.7B | ❌ **本机跑不动** | 崩溃 | — | 见下面「1.7B 为什么失败」 |

## 0.6B 的实际效果（与 faster-whisper large-v3 同段对照）

| 检查项 | Whisper large-v3 | **Qwen3-ASR-0.6B** | 谁更好 |
|---|---|---|---|
| 句末标点（。！？） | 0 | **6** | ✅ Qwen |
| 逗号 | 0 | **15** | ✅ Qwen |
| 同音错字「倒数/求倒/求到」 | 5 | **0** | ✅ Qwen |
| 「导数/求导」正确次数 | 4 | **9** | ✅ Qwen |
| `v` 被写成「位」 | 4 | **0** | ✅ Qwen |
| `u` 被写成「右」 | 4 | 9 | ❌ Whisper 略好 |
| 写成拉丁字母的 u/v | 5 | 4 | 平 |

**Whisper large-v3 输出**（无标点）：

```
右F4和位F4如果在F0处可倒那么右加位的倒数等于右先求倒再加位求倒注意这个公式它的前提条件
是右位都已经说了可倒了才能把这个倒数把它带进去先相加再求倒等于先求导再相加…
```

**Qwen3-ASR-0.6B 输出**（标点完整）：

```
右f和v f，如果在f零处可导，那么，右加v的导数等于右先求导，再加v求导。注意这个公式啊，
它的前提条件是右v啊都已经说了可导了，才能把这个导数把它带进去。先相加再求导，等于先求导
再相加。另外一个啊…比方说是右一x一直加到右n x，有n个表达式…
```

**差异要点**：

1. **标点是压倒性优势** —— Whisper 一个标点都没有，Qwen3 的断句基本可以直接当字幕用
2. **同音错字少很多** —— 「导数」全对（Whisper 写成「倒数/求倒」）、「再求导」正确
3. **`v` 认对了**（Whisper 写成「位」）—— 说明它保留拉丁字母的倾向更强
4. **`u` 仍然错**（写成「右」）—— **裸字母对谁都是难题**，这一点词表方案依然必要
5. 它更**逐字**（保留了「啊」这类口语填充词），Whisper 更倾向润色
6. 数字用中文（「右一」「右二」），Whisper 用阿拉伯数字（「U1」「U2」）

> 组合思路：**Qwen3-ASR 出标点和主干文本 + 词表修拉丁字母**，正好互补。

## 1.7B 为什么在本机失败

失败表现（三种都试过）：

| 尝试 | 结果 |
|---|---|
| 标准加载 | `CUDA out of memory`（transformers 的显存预热申请 3.2GB） |
| 跳过显存预热后 | `OSError: 页面文件太小 (os error 1455)` |
| 低内存加载（分片直接进显存） | 进程被系统杀掉 → 再试为 `0xC0000005 访问违例`（段错误） |

根因（实测数据）：

| 指标 | 数值 |
|---|---|
| 物理内存 | 15.7 GB（**可用仅 ~3GB**） |
| 已提交 / 提交上限 | 33.6 GB / 39.4 GB（**余量 5.8~7.8GB**） |
| 页面文件 | 已达 24GB **上限**，无法再增长 |
| 1.7B 权重 | 4.70 GB（分片一 4.02GB、分片二 0.46GB） |

两个叠加的硬约束：

1. **transformers 加载 safetensors 时会把整个分片读进 CPU 内存**（`safe_open` 不支持 `map_location`），
   单分片 4.02GB ×（state_dict + 模型 + 副本）→ 需要 ~8GB 以上提交内存，本机没有
2. **页面文件已到 24GB 上限**，提交内存无法增长；余量不足时不是抛干净的 MemoryError，
   而是 mmap/张量访问直接段错误

### 想跑 1.7B，按性价比排序

1. **重启机器**（最有效）—— 当前已提交 33.6GB，多半是长时间运行累积的；重启后提交通常降到 8~10GB，
   余量 25GB+，1.7B 大概率能直接跑起来
2. **关掉吃内存的程序**（浏览器多标签、VS Code、Zotero 等），腾出 5GB+ 提交内存
3. **把页面文件上限调大**（需要管理员）—— 现在 24GB 已到顶，且它在 C 盘（只剩 4GB）；
   改到 D 盘（61GB 空闲）并设上限 40GB，重启后生效
4. **换 0.6B** —— 现在就能跑，且标点质量已经明显优于 Whisper

## 复现方式

```powershell
# 环境（独立 venv，不污染主工程）
powershell -ExecutionPolicy Bypass -File experiments\setup_qwen3_asr.ps1
.\.venv-qwen\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.14.0+cu126"
.\.venv\Scripts\python scripts\download_model.py --repo Qwen/Qwen3-ASR-0.6B --source modelscope --out models\Qwen3-ASR-0.6B

# 跑
.\.venv-qwen\Scripts\python experiments\qwen3_asr_bench.py samples\formula.mp4 --model models\Qwen3-ASR-0.6B
```

## 踩坑记录（都已写进脚本注释）

| 坑 | 处理 |
|---|---|
| PyPI 装出来的是 `torch+cpu` | 必须从 `download.pytorch.org/whl/cu126` 装 CUDA 版 |
| transformers 的显存预热在小显存卡上直接 OOM | 脚本里 monkeypatch 掉（官方注释说它只是加载速度优化） |
| pip 缓存吃掉 C 盘 4GB，导致页面文件无法增长 | `pip cache purge`（C 盘 0.5GB → 4.2GB） |
| `--compare` 需要项目 JSON | 用主工程 `-f txt -o` 产出后再对照 |
