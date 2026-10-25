# torch.compile 与 CUDA Graph：从原理到本项目的实现

本文解释两种 decode 优化：

1. `torch.compile`：把 Python/PyTorch 的小算子融合成更少的 kernel；
2. CUDA Graph：把一整段 GPU 工作录制成图，replay 时只发一次 launch。

它们解决的是不同层次的问题。MicroInfer 目前把它们做成**互斥模式**：
`use_compile=True` 或 `use_cuda_graph=True`，但不能同时开启。

相关代码：

- `microinfer/runner.py`：编译入口、graph 捕获、bucket、静态 buffer、replay
- `microinfer/config/engine.py`：`use_compile` / `use_cuda_graph`
- `microinfer/llm.py`：参数透传
- [P1 优化记录](p1.md)：compile 的逐步结果
- [P2 CUDA Graph](p2.md)：graph 的设计与结果

---

## 0. 先看一张图

```text
eager decode:
  Python loop
   ├─ embedding
   ├─ 28 × (QKV GEMM, RMSNorm, RoPE, attention, O GEMM, FFN ...)
   └─ LM head
  每个小算子都要一次 CPU launch → GPU 执行

torch.compile:
  Dynamo 捕获 FX graph
   → Inductor 融合 pointwise/reduction
   → 一个 kernel 做原本 5–10 个小算子
   → 仍然每步由 Python 发起

CUDA Graph:
  第一次：把整段 forward 录制成 GPU graph
  之后：填充静态 buffer → graph.replay()
   → 整段 forward 只需一次 CPU 调用
```

两者不是替代关系：

| | torch.compile | CUDA Graph |
| --- | --- | --- |
| 主要消除 | kernel 数量、elementwise 开销 | CPU launch 次数、CPU/GPU 气泡 |
| 要求 | 可 trace 的 PyTorch 代码 | 固定 shape + 固定指针 + 无 host 同步 |
| 动态 batch | 可以，但会 guard/recompile | 必须分桶 + padding |
| 典型收益 | 融合算子 | 小 batch decode 的固定开销 |

---

## 1. torch.compile

### 1.1 它做了什么

`torch.compile(fn)` 不是简单的 JIT：

```text
Python bytecode
  → TorchDynamo 捕获成 FX graph（遇到不支持的代码就 graph break）
  → AOTAutograd 处理 autograd（推理时基本直通）
  → Inductor 生成 Triton / C++ kernel
  → 缓存编译结果，用 guards 判断能否复用
```

几个关键词：

- **Graph break**：Dynamo 遇到无法 trace 的 Python/第三方 op，会切断图，
  前后分别编译。break 太多，收益会被碎片化。
- **Guard**：Dynamo 对 Python 值、dtype、shape 等加 guard。guard 不匹配就
  recompile；动态 shape 会引入符号 shape 和更多 guard。
- **Inductor**：负责融合和 codegen。典型收益来自把
  `cast → add → mul → rsqrt → cast` 这种 elementwise 链合成一个 kernel。
- **动态 shape**：`dynamic=True` 接受变化的 batch/length，但会降低 specialization；
  固定 shape 更快，但更容易 recompile。

### 1.2 为什么本项目只 compile decode

```python
if self.use_compile and self.model is not None:
    rotary = getattr(self.model, "rotary", None)
    if rotary is not None:
        rotary.ensure(torch.device(self.device))
    self._compiled_forward = torch.compile(
        self.model.forward, dynamic=True, mode="default"
    )
```

来源：`microinfer/runner.py` 的 `ModelRunner.__init__`。

原因：

1. **prefill 形状变化大**：prompt 长度、chunk 边界、`cu_seqlens` 都是动态的，
   compile 容易 recompile，收益也低；
2. **decode 每步只处理 1 个 token/请求**：GEMM 小、elementwise 多，融合收益明显；
3. **decode 的 Python 控制流已经清理干净**：显式 `AttentionMetadata`、fast KV
   写入、共享 RoPE、fused norm，P1 为 compile 做了铺垫。

`_forward_for` 决定走哪条路径：

```python
def _forward_for(self, is_prefill):
    if self._compiled_forward is not None and not is_prefill:
        return self._compiled_forward
    return self._forward
```

prefill 始终走 eager `model.forward`。

### 1.3 哪些东西被融合了

典型 decode 层：

```text
QKV projection
RMSNorm(Q/K)
RoPE
attention
O projection
RMSNorm
SwiGLU
down projection
```

compile 主要融合的是**元素级/归约**部分：

- RMSNorm 的 `cast/add/square/mean/rsqrt/mul`；
- RoPE 的 `index_select/mul/add/cat`；
- SwiGLU 的 `silu/mul`；
- residual add、bias、cast。

GEMM 本身通常仍由 cuBLAS/Inductor 选择的 matmul kernel 执行；
这也是为什么项目还做了 QKV/FFN 打包：减少 GEMM **数量**，而 compile 减少
elementwise kernel **数量**。

### 1.4 为了 compile 清理热路径

compile 之前，P1 做了这些事：

| 问题 | 处理 |
| --- | --- |
| 每层读 `ContextVar` | 改成显式 `AttentionMetadata` |
| 每层 `.tolist()` / `torch.equal` | 校验移到 Runner / metadata 构建 |
| 每层重算 RoPE 复数频率 | 共享 `RotaryCache`，预先建表 |
| 手写 RMSNorm 多 op | `F.rms_norm` |
| prefill 逐请求 Python 循环 | prefill 不 compile |
| 40k 行 RoPE 表构造 | compile 前 `rotary.ensure(device)` |

特别说明两个坑：

#### 坑 1：不要在 capture/compile 路径里构造 RoPE 表

如果把 `RotaryCache._ensure()` 标成 `torch._dynamo.disable`，Dynamo 会把它当成
不透明函数，每层都产生一个 graph break。正确方式是：

```python
rotary.ensure(device)          # 在 compile 之前建表
compiled = torch.compile(model.forward)
```

这样 `apply()` 里只剩 `index_select` 和旋转算术。

#### 坑 2：Python list 比较会产生 data-dependent guard

P1 曾经在 `CachedAttention.forward` 里：

```python
if direct and metadata.q_lens_cpu != metadata.context_lens_cpu:
    ...
```

Dynamo 无法证明这个比较在编译期恒定，会 graph break。现在这个不变量由
`Runner.validate_step` 和 metadata 构建负责，热路径不再比较 Python list。

### 1.5 编译成本

冷进程实测：

- 模型加载：约 3.5 s；
- 第一次 generate：约 4.9 s（含 Dynamo/Inductor 编译）；
- 之后同样的 generate：约 0.03 s。

所以 benchmark 必须 warmup 目标 shape：

```bash
python -m bench.bench_offline ... --use-compile --warmup 2
```

第一次调用会编译；如果 batch shape 改变，可能触发新的 guard/recompile。
生产上建议对目标 batch bucket 做 warmup。

### 1.6 编译后的数字

RTX 4060 Laptop / Qwen3-0.6B bf16 / block_size=256：

| workload | eager tok/s | compile tok/s | TPOT eager → compile |
| --- | ---: | ---: | ---: |
| single-short | 69.5 | 89.2 | 14.3 → 11.0 ms |
| batch8-short | 497.5 | 576.0 | 14.7 → 12.2 ms |
| batch32-mixed | 875.3 | 928.3 | 16.0 → 14.0 ms |
| long-prefill | 52.5 | 56.5 | 14.4 → 12.9 ms |
| shared-prefix | 534.7 | 620.4 | 14.7 → 12.5 ms |

Profiling（single-short，8 个 decode token）：

| 指标 | eager | compile |
| --- | ---: | ---: |
| `cudaLaunchKernel`/token | 1,838 | 354 |
| `cudaStreamSynchronize`/token | 1.6 | 1.5 |

### 1.7 如何排查 compile 问题

```bash
# 看 graph break
TORCH_LOGS=graph_breaks python -m microinfer ... --use-compile

# 看 Dynamo 解释
python - <<'PY'
import torch
from torch._dynamo import explain
# explain(compiled_fn)(...)
PY

# 看 profiler
python -m bench.profile_runner ... --use-compile --export torch
```

常见 break 原因：

- `.item()` / `.tolist()` / `if tensor.any()`；
- 依赖 tensor 值的 Python 分支；
- 第三方 pybind/C++ op（FlashAttention 的 `fwd_kvcache`）；
- 动态 shape 的 Python int（例如把 `max_seqlen` 当普通参数）；
- 可变全局状态、ContextVar、日志/print、文件 IO。

本项目的 FlashAttention break 被接受为“opaque custom kernel”边界；
compile 仍然能融合 attention 之外的 norm/FFN/elementwise。

P3 的 Triton paged decode 是一个例外：它是普通 Python 函数 + Triton launch，
Dynamo 能完整追踪。`block_size=16` 时 decode 编译成单图（0 graph break），
而 `block_size=256` 的 FlashAttention 路径有 7 次 break。这也是小 block compile
在低/中 batch 上能超过 CUDA Graph 的原因，详见 [P3 文档](p3.md)。

---

## 2. CUDA Graph

### 2.1 它解决什么问题

eager/compile 的每一步仍然由 CPU 发起很多 kernel。decode 单步可能只有几毫秒，
但 CPU 要依次 launch 数百个 kernel；launch 本身和 CPU/GPU 同步会形成气泡。

CUDA Graph 的做法：

1. 在 capture 阶段，把一段时间内的 GPU 操作（kernel、memcpy、依赖关系）录下来；
2. 之后 replay 时，CPU 只调用一次 `graph.replay()`；
3. GPU 按录好的依赖执行，不再需要逐个 launch。

代价是：

- **所有张量指针固定**：输入/输出 buffer 不能在 capture 后重新分配；
- **shape 固定**：kernel grid 在 capture 时确定；
- **不能有 host 同步**：capture 期间 `.item()`、`.tolist()`、`cudaStreamSynchronize`
  都会失败；
- **不能有 CPU 依赖的控制流**：Python 分支在 capture 时就被确定了；
- **autograd 不参与**：本项目是 inference，天然适合。

### 2.2 为什么必须分桶

decode 的 batch 会变化：请求完成、新请求加入、抢占重计算。Graph 的 batch 是固定的，
所以需要：

```text
bucket = 1, 2, 4, 8, 16, 32, ..., max_num_seqs
```

实际 batch 向上取整到 bucket，例如 batch=3 使用 bucket=4。

每个 bucket 有独立的：

- `_DecodeBuffers`（input_ids/positions/slot/context/block table）
- logits 输出 buffer
- `CUDAGraph`

所有 bucket 共享一个 graph pool。

### 2.3 Padding 行怎么处理

padding 行也会执行完整的 forward。如果它们写真实请求的 KV，就会破坏缓存。
本项目额外分配一个 **pad block**：

```python
extra = 1 if self.use_cuda_graph else 0
self._pad_block_id = num_blocks if self.use_cuda_graph else None
cache = KVCache(..., num_blocks + extra, ...)
```

`BlockManager` 仍然只管理 `num_blocks`，pad block 永远不参与调度。

padding 行使用：

```text
input_ids = 0
positions = 0
context_lens = 1
slot_mapping = pad_block * block_size + row % block_size
block_tables[0] = pad_block
```

输出 logits 被丢弃。这样 padding 的写入完全隔离，真实请求的 KV 不受影响。

### 2.4 静态输入与 replay

每个 bucket 的 `_DecodeBuffers` 是固定地址的；每步 decode：

```text
1. 填充 pinned CPU staging
2. copy_(non_blocking=True) 到固定 device buffer
3. graph.replay()
4. 从静态 logits buffer 取 [:real_batch]
5. argmax / sample
```

`copy_` 和 `replay` 在同一个 stream 上排队，GPU 会先完成拷贝再执行图。

### 2.5 关键：slot 索引必须在图内重算

这是 CUDA Graph 实现里最容易错的地方。

`CachedAttention` 用：

```python
blocks, offsets = metadata.slot_indices(cache.block_size)
cache.write_fast(blocks, offsets, k, v)
```

`slot_indices()` 首次调用会计算：

```python
slots = metadata.slot_mapping_long
blocks = torch.div(slots, block_size, rounding_mode="floor")
offsets = slots - blocks * block_size
```

如果这些索引在 capture **之前**就算好并缓存，replay 时会一直使用 capture 时的旧
`blocks/offsets`，但 slot buffer 已经变了，KV 会写错位置。

本项目的做法：

1. warmup 使用 metadata A，让 allocator/kernel 预热；
2. capture 使用**全新的** metadata B，`_slot_block_size = None`；
3. `slot_mapping.long()`、div/mod 被捕获进图；
4. replay 时这些 kernel 读取新的 slot buffer，更新图内的 blocks/offsets；
5. 28 层共享同一份图内 blocks/offsets。

对应代码：

- `ModelRunner._capture_bucket`
- `AttentionMetadata.slot_indices`
- `CachedAttention.forward`

### 2.6 捕获流程

`ModelRunner.capture_cuda_graphs()`：

```python
for batch in buckets:
    self._capture_bucket(batch)
```

`_capture_bucket` 的核心：

```python
buffers = _DecodeBuffers(device, pin_memory=True)
buffers.ensure(batch, kv_cache.num_blocks)
self._fill_pad_buffers(buffers, batch)

# 1) warmup：side stream 上跑两次
warm_metadata = self._make_decode_metadata(buffers, batch)
stream = torch.cuda.Stream(device=device)
stream.wait_stream(torch.cuda.current_stream(device))
with torch.cuda.stream(stream):
    for _ in range(2):
        self._decode_forward(input_ids, positions, warm_metadata)

# 2) 用全新 metadata capture
capture_metadata = self._make_decode_metadata(buffers, batch)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, self._graph_pool):
    logits = self._decode_forward(input_ids, positions, capture_metadata)
```

要点：

- warmup 必须在 side stream；
- capture 前 `torch.cuda.synchronize()`；
- capture 的 metadata 不能复用 warmup 的 slot 索引缓存；
- 输出 logits 是图内分配的静态 tensor；
- 第一个 graph 的 pool 被后续 bucket 复用。

### 2.7 前置条件

Graph 捕获要求 decode 走 FlashAttention 或 Triton paged kernel：

- CUDA；
- FP16/BF16；
- FlashAttention 可用且 `block_size % 256 == 0`，或 Triton 可用且
  `block_size`、`head_dim` 为 2 的幂；
- `head_dim <= 256` 且 `head_dim % 8 == 0`。

不满足时 `capture_cuda_graphs()` 会给出明确错误，而不是静默退回 eager。

原因：reference decode 路径里有 `.tolist()`/host 同步，无法被 graph 捕获。
小 block 的 Triton kernel 与 graph 的配合见 [P3 文档](p3.md)。

### 2.8 性能数字

RTX 4060 Laptop / Qwen3-0.6B bf16 / block_size=256：

| workload | eager | compile | graph | graph vs compile |
| --- | ---: | ---: | ---: | ---: |
| single-short | 69.5 | 89.2 | **110.4** | +23.8% |
| batch8-short | 497.5 | 576.0 | **678.1** | +17.7% |
| batch32-mixed | 875.3 | **928.3** | 917.8 | -1.1% |
| long-prefill | 52.5 | 56.5 | **64.4** | +14.0% |
| shared-prefix | 534.7 | 620.4 | **691.6** | +11.5% |

Profiling（single-short，8 个 decode token）：

| 指标 | eager | compile | graph |
| --- | ---: | ---: | ---: |
| `cudaLaunchKernel`/token | 1,838 | 354 | **234** |
| `cudaStreamSynchronize`/token | 1.6 | 1.5 | 1.6 |

显存：

- KV：1,792 → 1,820 MB（pad block）；
- single-short 峰值：2,963 → 3,115 MB；
- long-prefill：3,050 → 3,202 MB。

增量约 150 MB：pad block + graph pool + 各 bucket 的 logits buffer。

### 2.9 正确性验证

`tests/integration/test_cuda_graph.py`：

- 同一模型权重下 graph vs eager；
- batch=3 padding 到 bucket=4；
- context length 不同；
- 输出 token 完全一致。

手工验证：

- prompt 长度 7/20/33，graph/eager 8 个贪心 token 一致；
- prompt 250 + 16 token，跨 256 block 边界，graph/eager 完全一致；
- `Hello` 的 token 与 HF Transformers 一致；
- `--debug-mode --use-cuda-graph` 通过集中校验。

### 2.10 常见失败模式

| 现象 | 原因 | 修复 |
| --- | --- | --- |
| capture 报 “operation not permitted when stream is capturing” | 图内有 host 同步或 CPU 分支 | 移除 `.tolist`/`.item`/`.any`/`if tensor` |
| replay 后 KV 错位 | slot 索引在 capture 前被缓存 | 用全新 metadata 捕获，让 div/mod 进图 |
| 输出 shape/地址错误 | capture 后重新分配 buffer | 固定 buffer，禁止 capture 后重建 |
| 第二次 replay 更慢/报错 | graph pool 未共享/输出被释放 | 保存 logits tensor，共享 `graph.pool()` |
| 某 batch 没走 graph | 没有对应 bucket | 增加 bucket 或 fallback eager |
| 图捕获后重新绑 KV cache | graph 持有旧指针 | `bind_kv_cache` 清空 graph，重新捕获 |
| 多 stream 使用错误 | replay 不在当前 stream | 保持 copy + replay 在同一 stream |

### 2.11 为什么当前不与 compile 同时开启

`torch.compile` 自己可能在内部使用 CUDA Graph（尤其 `mode="reduce-overhead"`），
而显式 `torch.cuda.graph` 需要控制 capture 边界、buffer 和 stream。两者叠加时：

- compile 的 graph break 会把 capture 切成多段；
- compile 可能重新分配输出/buffer；
- 内部 cudagraph 与外部 capture 可能冲突。

所以 P2 先保证一个稳定、可验证的显式 graph 路径：

```python
use_compile=True  XOR  use_cuda_graph=True
```

`EngineConfig` 和 `ModelRunner` 都会拒绝同时开启。

未来可行的组合方式：

1. 让 compile 只生成 graph-safe 的 fused forward；
2. 用 `torch._dynamo.disable` 把 FlashAttention 包成 opaque custom op；
3. 再用 `torch.cuda.graph` 捕获 compile 后的 forward；
4. 或者使用 `mode="reduce-overhead"`，但需要接受它内部的 graph 管理。

---

## 3. 两种优化怎么选

| 场景 | 建议 |
| --- | --- |
| 低/中 batch decode，CPU launch 是瓶颈 | CUDA Graph |
| 低/中 batch + 小 block Triton decode | compile（单图 + 融合） |
| 无法固定 shape，或需要动态长度 | torch.compile / eager |
| prefill 长度变化大 | eager（或专门 varlen kernel） |
| batch 很大、GPU 已计算受限 | 先做 kernel 融合/打包；compile 可能比 graph 更值 |
| 想同时获得融合 + launch 消除 | 组合 compile + graph（后续工作） |
| 需要快速排查数值/形状问题 | eager + debug_mode |

本项目当前推荐：

- 交互/低并发 + `block_size=16`：`--use-compile`（P3 的 Triton kernel 可完整追踪）；
- 交互/低并发 + `block_size=256`：`--use-cuda-graph`；
- 批处理/GPU 受限：`--use-compile`；
- 正确性/调试：默认 eager + `--debug-mode`。

---

## 4. 复现命令

```bash
# 正确性
pytest -q

# compile
python -m bench.bench_offline --model /path/to/Qwen3-0.6B \
  --engine microinfer --workloads single-short,batch8-short \
  --runs 3 --warmup 2 --block-size 256 --num-blocks 64 --use-compile

# CUDA Graph
python -m bench.bench_offline --model /path/to/Qwen3-0.6B \
  --engine microinfer --workloads single-short,batch8-short \
  --runs 3 --warmup 1 --block-size 256 --num-blocks 64 --use-cuda-graph

# 生成 trace
python -m bench.profile_runner --model /path/to/Qwen3-0.6B \
  --workload single-short --max-tokens 8 --use-compile --export torch
python -m bench.profile_runner --model /path/to/Qwen3-0.6B \
  --workload single-short --max-tokens 8 --use-cuda-graph --export torch

# 看 graph break
TORCH_LOGS=graph_breaks python -m microinfer --model /path/to/Qwen3-0.6B \
  --prompt 'Hello' --max-tokens 2 --greedy --block-size 256 \
  --num-blocks 64 --use-compile
```

---

## 5. 面试追问

1. **为什么 decode 是 CPU-bound？**
   kernel launch、Python 调度、host-device 同步。P0 profile 里 3,256 launches/token、
   281 次 stream sync；P1/P2 后分别降到 ~1,838/1.6 和 234/1.6。

2. **torch.compile 为什么不直接解决 launch 问题？**
   它减少 kernel 数，但每步仍有 Python 调用和多个 graph segment；CUDA Graph 把整段
   变成一个 replay。

3. **CUDA Graph 怎么处理动态 batch？**
   batch bucket + padding；padding 写独立 pad block；输出只取真实行。

4. **为什么 slot 索引必须在图内算？**
   图里 kernel 参数/缓冲区指针固定；如果索引在 capture 前算好，replay 时不会更新，
   KV 会写到旧位置。

5. **为什么 graph 不能有 `.tolist()`？**
   capture 期间禁止 host 同步；`.tolist()` 需要 D2H 拷贝和同步，会中断 capture。

6. **为什么 prefill 不做 graph？**
   长度可变、逐请求循环多，shape/grid 不稳定；decode 形状由 bucket 固定。

7. **graph 与 compile 为什么互斥？**
   compile 可能内部用 cudagraph、产生 graph break 和重新分配；显式 capture 需要
   自己控制 buffer/stream。当前先保证单一路径可验证。

8. **怎么保证数值正确？**
   graph vs eager token 对照（padding、变长 context、跨 block 边界）；
   HF parity；`--debug-mode` 校验。

9. **graph 的显存代价是什么？**
   pad block（约 28 MB/block）、graph pool、每个 bucket 的 logits 输出；总计约 150 MB。

10. **如果 batch32 graph 反而比 compile 慢？**
    GPU 接近计算受限，compile 的算子融合收益大于 graph 的 launch 消除；这是选择
    两种优化的边界，不是实现错误。

---

## 6. 代码索引

| 主题 | 位置 |
| --- | --- |
| compile 入口 | `microinfer/runner.py`：`ModelRunner.__init__` / `_forward_for` |
| compile 只用于 decode | `microinfer/runner.py`：`_forward_for` |
| graph 捕获 | `microinfer/runner.py`：`capture_cuda_graphs` / `_capture_bucket` |
| graph bucket | `microinfer/runner.py`：`_GraphBucket` / `_select_graph_bucket` |
| padding | `microinfer/runner.py`：`_fill_pad_buffers` / `_fill_graph_bucket` |
| replay | `microinfer/runner.py`：`_run_decode` |
| slot 索引 | `microinfer/layer/metadata.py`：`slot_indices` |
| KV 写入 | `microinfer/worker/kv_cache.py`：`write_fast` / `write_checked` |
| 融合算子 | `microinfer/layer/norm.py`、`microinfer/layer/transformer.py` |
| 配置 | `microinfer/config/engine.py` |
| P1 结果 | [p1.md](p1.md) |
| P2 结果 | [p2.md](p2.md) |
