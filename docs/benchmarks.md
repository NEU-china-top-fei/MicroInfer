# 基准测试

本页记录 MicroInfer 的离线推理基线、对比方法和 profiling 结果。所有数字都可复现；
原始 JSON 写入 `results/bench/`（已在 `.gitignore` 中忽略），提交的是环境、命令和汇总表。

> 这是**离线引擎**基准，不是 serving 基准。并发到达、排队、流式输出和 P99 端到端时延
> 会在服务化完成后单独测量。

## 1. 测量环境

测量日期：2026-10-22。所有数字来自同一台机器、同一次代码状态。

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8 GB |
| CPU / RAM | 16 threads / 14 GB |
| OS | Linux 7.0.0 |
| Python / PyTorch | 3.11 / 2.8.0+cu128 |
| CUDA | 12.8 |
| FlashAttention | 2.8.3.post1 |
| Transformers | 5.19.0 |
| 模型 | Qwen3-0.6B, bf16 |
| MicroInfer 配置 | `block_size=256`, `num_blocks=64`（KV 1.79 GB）, `max_num_seqs=64`, `max_num_batched_tokens=8192`, `max_model_len=4096` |

`block_size=16` 对照组使用 `num_blocks=1024`，KV 容量同样是 1.79 GB。

## 2. Workload

固定 seed=0；token ID 在 `[100, 20000)` 内随机，避免特殊 token 影响生成长度。
所有请求设置 `ignore_eos=True`，因此每个请求严格生成 `output_len` 个 token。

| 名称 | 请求数 | prompt 长度 | 生成长度 | 用途 |
| --- | ---: | ---: | ---: | --- |
| `single-short` | 1 | 128 | 32 | 单流 decode / launch 开销 |
| `batch8-short` | 8 | 128 | 32 | 典型 continuous batching |
| `batch32-mixed` | 32 | 64–384 | 32 | 混合长度与更大 batch |
| `shared-prefix` | 8 | 512 共享 + 32 独有 | 32 | 前缀缓存 |
| `long-prefill` | 1 | 2048 | 32 | 长 prefill 与长上下文 decode |

## 3. 指标定义

- **TTFT**：从提交请求到每个请求收到第一个 completion token。包含入队、调度、prefill
  和排队时间；是引擎侧指标，不含 tokenizer 编码和文本解码。
- **TPOT**：`(最后一个 token 时间 - 第一个 token 时间) / (输出 token 数 - 1)`。
- **ITL**：相邻 token 的到达间隔，用于观察 decode 抖动。
- **output tok/s**：所有请求生成的 token 总数 / 总墙钟时间。
- **peak MB**：`torch.cuda.max_memory_allocated`，包含权重、KV 池和 activation。

每轮先 warmup 目标 shape，再跑 3 次；吞吐报 3 次的中位数，TTFT/TPOT 池化所有请求后
取 p50/p99。`single-short` 只有 3 个样本，p99 实际等于最大值，不要过度解读。

## 4. 结果

单位：output tok/s；TTFT/TPOT 为 ms；peak 为 MB。最后一列是相对 HF 基线的吞吐倍率，
`<1` 表示比 HF 慢。默认 `--cache-mode cold`：每个 measured run 前清空前缀缓存；
`shared-prefix` 会先单独 prime 共享前缀，再测跨请求复用。

| workload | engine | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | peak MB | vs HF |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single-short | microinfer (bs16) | 33.4 | 30.7 | 31.3 | 29.9 | 2948 | 0.45x |
| single-short | microinfer (bs256) | 37.5 | 30.3 | 32.5 | 26.5 | 2948 | 0.50x |
| single-short | hf | 75.1 | 15.5 | 15.5 | 13.2 | 1163 | - |
| single-short | nano | 153.0 | 16.7 | 17.0 | 6.2 | 6581 | 2.04x |
| batch8-short | microinfer (bs16) | 122.0 | 98.9 | 99.9 | 64.5 | 2980 | 0.23x |
| batch8-short | microinfer (bs256) | 257.9 | 95.8 | 96.3 | 29.0 | 2980 | 0.49x |
| batch8-short | hf | 531.3 | 59.1 | 59.2 | 13.6 | 1291 | - |
| batch8-short | nano | 964.5 | 47.4 | 47.9 | 7.0 | 6605 | 1.82x |
| batch32-mixed | microinfer (bs16) | 155.0 | 925.1 | 925.2 | 183.3 | 3241 | 0.46x |
| batch32-mixed | microinfer (bs256) | 506.1 | 921.5 | 923.5 | 35.5 | 3241 | 1.52x |
| batch32-mixed | hf | 333.8 | 1051.2 | 1053.0 | 65.1 | 2769 | - |
| batch32-mixed | nano | 1391.8 | 365.9 | 371.7 | 11.8 | 6777 | 4.17x |
| shared-prefix | microinfer (bs16) | 122.0 | 77.1 | 77.2 | 65.2 | 2953 | 0.40x |
| shared-prefix | microinfer (bs256) | 278.1 | 39.6 | 39.8 | 28.4 | 2948 | 0.92x |
| shared-prefix | hf | 303.8 | 328.9 | 329.6 | 16.6 | 1735 | - |
| shared-prefix | nano | 930.6 | 19.3 | 20.3 | 8.3 | 6585 | 3.06x |
| long-prefill | microinfer (bs16) | 20.0 | 662.2 | 662.2 | 30.4 | 3653 | 0.35x |
| long-prefill | microinfer (bs256) | 21.7 | 659.4 | 659.4 | 26.3 | 3653 | 0.38x |
| long-prefill | hf | 56.9 | 152.2 | 153.8 | 13.2 | 1422 | - |
| long-prefill | nano | 93.0 | 124.4 | 125.5 | 7.1 | 6633 | 1.63x |

MicroInfer bs256 的缓存/容量指标：

| workload | KV peak | prefix hit tokens | prefix hit rate |
| --- | ---: | ---: | ---: |
| single-short | 1.6% | 0 | 0.0% |
| batch8-short | 12.5% | 0 | 0.0% |
| batch32-mixed | 78.1% | 0 | 0.0% |
| shared-prefix | 15.6% | 12,288 | 94.1% |
| long-prefill | 14.1% | 0 | 0.0% |

### 解读

1. **decode 的瓶颈是每步固定开销，不是 attention。**
   MicroInfer bs256 的单流吞吐只有 HF eager 的 0.50×、nano-vllm 的 0.24×；
   batch-8 分别是 0.49× 和 0.27×。TPOT 26.5 ms vs HF 13.2 / nano 6.2，
   而 TTFT 30.3 ms 与 HF 15.5 / nano 16.7 的差距相对小。差距集中在 decode。
2. **冷启动 prefill 是第二个瓶颈。**
   `long-prefill`（2048 prompt）的 TTFT 为 659 ms，而 HF 是 152 ms、nano 是 124 ms；
   `batch32-mixed` 的 TTFT 为 922 ms，nano 是 366 ms。profiling 显示 no-prefix prefill
   落到了 SDPA 的 math 后端：`bmm`、`add_`、`softmax`、`where`、`isneginf` 合计占
   大部分 GPU 时间（见第 5 节）。这是 P1/P3 的首要目标（varlen FlashAttention）。
3. **`block_size=16` 曾经会静默关闭 FlashAttention 分页路径。**
   P0 时实现要求 `block_size % 256 == 0`；默认 16 时回退到逐请求 Python 参考实现：
   单流 33.4 vs 37.5 tok/s、batch-8 122.0 vs 257.9、batch32 155.0 vs 506.1。
   P3 已用 Triton paged decode + gather/varlen prefill 去掉这个隐式约束：
   block16 graph 达到单流 112.5、batch8 676.9、batch32 914.3 tok/s，
   block16 compile 单流 126.9、batch8 710.7。设计、数值验证和与 FA 的
   对照见 [P3 文档](p3.md)。
4. **前缀缓存有效，且测到了真实的跨请求复用。**
   `shared-prefix` bs256 命中 12,288 tokens（8 × 512），命中率 94.1%；
   TTFT 39.6 ms，低于 `batch8-short` 的 95.8 ms；吞吐 278.1 tok/s 与 HF 基本持平
   （0.92×）。nano-vllm 仍有 3.06× 优势。
5. **batch32 上 MicroInfer 快于 HF，但不能当作通用结论。**
   HF 基线是静态左 padding batch，会把所有 prompt 补到 384 并一次性 prefill；
   TTFT 1.05 s 和显存 2.8 GB 都包含 padding 与静态 batch 的代价。真正公平的
   单流/batch-8 对比中，HF 仍然更快。nano-vllm 在 batch32 上是 MicroInfer 的 4.17×。
6. **显存占用可解释。** MicroInfer 固定 1.79 GB KV + ~1.2 GB 权重 ≈ 3.0 GB；
   batch32 的 KV peak 78.1%，activation 让峰值到 3.24 GB；long-prefill 的
   math-SDPA 中间矩阵让峰值到 3.65 GB。HF 为 1.16–2.77 GB；nano-vllm 使用
   `gpu_memory_utilization=0.9`，常驻约 6.6 GB。

### cold vs warm

`--cache-mode warm` 不做每轮清理，让重复 workload 的完整块留在前缀缓存中。两种口径
差别很大，说明为什么默认用 cold：

| workload | cold tok/s | warm tok/s | cold TTFT | warm TTFT |
| --- | ---: | ---: | ---: | ---: |
| single-short | 37.5 | 37.5 | 30.3 | 29.2 |
| batch8-short | 257.9 | 260.3 | 95.8 | 95.8 |
| batch32-mixed | 506.1 | 739.6 | 921.5 | 275.6 |
| shared-prefix | 278.1 | 275.0 | 39.6 | 34.8 |
| long-prefill | 21.7 | 37.0 | 659.4 | 33.6 |

warm 的 `long-prefill` 只 prefill 最后一个不足 256 token 的块，所以 TTFT 看起来很好；
这不是冷请求的真实成本。cold 才是默认口径；warm 可用于观察稳态前缀命中收益。

## 5. Profiling

命令：

```bash
python -m bench.profile_runner --model /path/to/Qwen3-0.6B \
  --workload single-short --block-size 256 --max-tokens 8
```

`single-short`、8 个 decode token 的 trace 摘要：

| 计数 | 总量 | 每 token |
| --- | ---: | ---: |
| `cudaLaunchKernel` | 26,047 | 3,256 |
| `cudaStreamSynchronize` | 2,248 | 281 |
| `aten::mm` | 1,576 | 197 |
| `aten::copy_` | 6,445 | 806 |
| `aten::_unique2` | 224 | 28 |
| `aten::nonzero` | 672 | 84 |
| `aten::mean` | 904 | 113 |

CUDA 时间占比（profiled run）：`aten::mm` 47.3%、`copy_` 10.9%、`_unique2` 4.7%，
elementwise/RMSNorm/RoPE 合计约 30%。CPU self time 中 `cudaLaunchKernel` 约 28%，
`cudaStreamSynchronize` 约 7.6%。

解释：

- 28 层 × 每层约 115 次 launch ≈ 3.2k launch/token。每层的 Q/K/V/O 四个 GEMM、
  FFN 三个 GEMM、两次 RMSNorm、两次 RoPE、KV 写入校验和 attention 元数据校验
  都各自产生 kernel。
- `_unique2` 每 token 28 次（每层一次），来自 `LayerKVCache.write` 的重复 slot 检查；
  `nonzero` 每 token 84 次（每层三次），来自 `torch.any` / mask 计算。
- `cudaStreamSynchronize` 每 token 281 次，来自 attention 和 cache 热路径上的
  `.tolist()`、`.item()`、`unique` 等 host 同步。
- 这些正是 P1 的目标：校验移出热路径、RoPE 预计算、RMSNorm/SwiGLU 融合、
  QKV/FFN 打包、`torch.compile`，最后用 CUDA Graph 消掉 launch overhead。
- 注意：profiler 会干扰每个 launch，上面的 throughput（21.5 tok/s）不能当作性能数字；
  只使用“每 token 的调用次数/占比”做热点定位。

### 冷启动 prefill（2048 tokens）

```bash
python -m bench.profile_runner --model /path/to/Qwen3-0.6B \
  --workload long-prefill --max-tokens 1 --cold
```

一次 2048-token prefill（1 个输出 token）的 profiled wall 为 665 ms，CUDA 时间 645 ms；
`cudaLaunchKernel` 只有 3,822 次，说明这一步是 **GPU-bound**，不是 launch-bound。
主要 kernel：

| activity | CUDA 时间 | 占比 | 调用数 |
| --- | ---: | ---: | ---: |
| `aten::bmm` | 146.5 ms | 22.7% | 56 |
| `aten::add_`（mask 加法） | 98.3 ms | 15.2% | 28 |
| `aten::copy_` | 84.8 ms | 13.1% | 880 |
| `aten::_softmax` | 65.8 ms | 10.2% | 28 |
| `aten::where` | 61.6 ms | 9.6% | 84 |
| `aten::isneginf` | 40.3 ms | 6.2% | 28 |
| FP32 `sgemm`（tn + nn） | 146.5 ms | 22.7% | 56 |

这些 kernel 说明 no-prefix prefill 的 `F.scaled_dot_product_attention` 落到了 math
后端：它物化了 `[H, 2048, 2048]` 的 score 矩阵，然后做 mask add、softmax、`where`、
`isneginf`，并且部分 GEMM 走 FP32。每层 2 个 `bmm` + 2 个 FP32 `sgemm`，28 层累积到
600 ms 量级。

P1/P3 的修复方向：

1. 无前缀 prefill 直接调用 `flash_attn_varlen_func`（packed `[T,H,D]`），或至少让 SDPA
   走 flash/mem-efficient 后端；
2. 去掉 `repeat_interleave`，用 GQA 的 KV 扩展布局或 varlen kernel；
3. 保证 q/k/v 在送入 attention 前是连续布局，避免 math 后端；
4. 预期 `long-prefill` 的 TTFT 从 659 ms 降到 ~100 ms 量级（HF 当前 152 ms）。

### 可视化输出

`profile_runner.py` 默认 `--export torch`，一次生成两种可直接打开的格式：

| 格式 | 文件 | 打开方式 |
| --- | --- | --- |
| Chrome / Perfetto | `results/bench/traces/<stem>.json` | `chrome://tracing` 或 https://ui.perfetto.dev |
| TensorBoard / PyTorch Profiler | `results/bench/traces/tensorboard/<stem>.<pid>.<ts>.pt.trace.json.gz` | `tensorboard --logdir results/bench/traces/tensorboard`，打开 PROFILER 页 |
| Nsight Systems | `results/bench/traces/<stem>.nsys-rep` | `nsys-ui <file>` 或 `nsys stats --report cuda_gpu_kern_sum <file>` |

TensorBoard 的 PyTorch Profiler 页由 `torch-tb-profiler` 提供（只装 `tensorboard`
会看到 `profile_redirect`，没有实际解析器）：

```bash
uv pip install tensorboard torch-tb-profiler
# 或者安装项目的 profile extra
python -m pip install -e '.[profile]'

tensorboard --logdir results/bench/traces/tensorboard
# 浏览器打开 http://localhost:6006 -> PROFILER
```

nsys 输出：

```bash
python -m bench.profile_runner --model /path/to/Qwen3-0.6B \
  --workload single-short --max-tokens 1 --export nsys

nsys-ui results/bench/traces/single-short_bs256.nsys-rep
# 或无 GUI 环境下看 kernel 汇总
nsys stats --report cuda_gpu_kern_sum \
  results/bench/traces/single-short_bs256.nsys-rep
```

`--export nsys` 会用子进程运行 workload，并传入
`--trace=cuda,nvtx --stats=false --resolve-symbols=false`。`--stats=false` 和
`--resolve-symbols=false` 用来避免离线环境下的符号下载卡住；NVTX range
`microinfer.load_model`、`microinfer.warmup`、`microinfer.generate` 会把时间线分段。
nsys 与 torch.profiler 不能在同一进程里同时使用 CUPTI，所以脚本自动切换为子进程。
nsys 的 trace 包含模型加载和 warmup，分析时请缩放到 `microinfer.generate` 这一段。

其他导出模式：`--export chrome` 只写 Chrome JSON，`--export tensorboard` 只写
TensorBoard trace，`--export run` 只跑 workload 不做 profiling。trace 较大，建议先用
`--max-tokens 1`（prefill）或 `--max-tokens 8`（decode）控制体积。

## 6. 复现

以下命令都在仓库根目录运行（`bench/` 不是必须安装的运行时包）。

```bash
# 安装（需要 HF tokenizer 与 transformers 基线；profile 可选）
python -m pip install -e '.[hf,test]'
python -m pip install -e '.[hf,test,profile]'  # TensorBoard trace 可视化

# 全部引擎；每个引擎在独立进程中运行
python -m bench.run_all --model /path/to/Qwen3-0.6B --runs 3 --warmup 1

# 只跑 MicroInfer 的单个 workload，重复 5 次
python -m bench.bench_offline --model /path/to/Qwen3-0.6B --engine microinfer \
  --workloads single-short --runs 5 --warmup 1 --block-size 256 --num-blocks 64

# 只看汇总表
python -m bench.report --input-dir results/bench

# 记录一次优化消融（追加到 results/bench/ablation.csv）
python -m bench.ablation --model /path/to/Qwen3-0.6B --label p1.1-metadata \
  --runs 3 --warmup 1 --block-size 256 --num-blocks 64

# decode CUDA Graph
python -m bench.bench_offline --model /path/to/Qwen3-0.6B \
  --engine microinfer --workloads single-short,batch8-short \
  --runs 3 --warmup 1 --block-size 256 --num-blocks 64 --use-cuda-graph

# 生成 Chrome + TensorBoard trace
python -m bench.profile_runner --model /path/to/Qwen3-0.6B \
  --workload single-short --max-tokens 8 --export torch

# 生成 Nsight Systems report
python -m bench.profile_runner --model /path/to/Qwen3-0.6B \
  --workload single-short --max-tokens 1 --export nsys
```

`run_all` 会依次运行 MicroInfer（`bs256`、`bs16`）、HF、nano-vllm；nano-vllm 不可用时
只打印失败，不影响其他结果。GPU 驱动偶发 abort 时 `--retries`（默认 1）会自动重跑。

结果 JSON 结构：`meta`（环境/版本/配置）、`workload`（长度/请求数）、`runs`（每次的
TTFT/TPOT/ITL/吞吐/显存）、`summary`（池化后的中位数与分位数）。

优化阶段的 debug/compile 开关和消融工作流见 [P1 优化工作流](p1.md)，
decode CUDA Graph 的设计与结果见 [P2](p2.md)，小 block 的 Triton decode 与
gather/varlen prefill 见 [P3](p3.md)，混合调度/异步服务的在线数据见
[P4](p4.md)；compile/graph 的原理与排查见
[torch.compile 与 CUDA Graph](compile-and-cuda-graph.md)。

### 在线并发（P4）

`bench/online.py` 不经过 HTTP，直接驱动 `AsyncEngine`，用于隔离调度和模型时间：

```bash
python -m bench.online --model /path/to/Qwen3-0.6B \
  --requests 64 --prompt-len 128 --output-len 16 \
  --max-num-seqs 64 --max-num-batched-tokens 4096 \
  --max-num-prefill-tokens 512 --block-size 16 --num-blocks 1024 \
  --use-cuda-graph
```

RTX 4060 Laptop / Qwen3-0.6B bf16：

| 场景 | output tok/s | TTFT p99 | ITL p99 |
| --- | ---: | ---: | ---: |
| 64 请求同时到达，混合 batch | 1060.8 | 684 ms | 339 ms |
| 64 请求同时到达，prefill-first | **1098.1** | **659 ms** | 348 ms |
| 16 个长 decode，晚到 8×2048 prefill，混合 | 424.5 | — | **104 ms** |
| 同上，prefill-first | **435.7** | — | 1098 ms |

关键结论：混合调度在晚到长 prefill 时把 decode 尾延迟从 1.1 s 压到约 0.1 s，
代价是约 3% 吞吐和 2.5% 晚到 TTFT。chunk=256 可把尾延迟降到 63 ms，但吞吐再降
约 7%。完整配置和 OpenAI 服务说明见 [P4](p4.md)。

## 7. 对比口径与注意事项

- **HF 基线**：`AutoModelForCausalLM` + SDPA，静态左 padding batch，手工 prefill/decode
  循环，`logits_to_keep=1`，贪心 argmax。没有 continuous batching 和 paged KV；
  batch32 的 padding 会放大 TTFT 和显存。单流/batch-8 是最公平的对比。
- **nano-vllm 基线**：`enforce_eager=False`，启用 `torch.compile` 和 CUDA Graph；
  参考实现不支持 greedy，使用 `temperature=0.6` 采样；`gpu_memory_utilization=0.9`。
  吞吐接近，但采样策略和显存配置不同。
- **MicroInfer 基线**：greedy、eager、无 compile/graph；KV 池固定 1.79 GB。
  P3 之后 `block_size=16` 走 Triton paged decode + gather/varlen prefill，
  `block_size=256` 走 FlashAttention 分页路径；两组的吞吐对照见 [P3](p3.md)。
- **缓存口径**：默认 `--cache-mode cold`，每轮 measured run 前清空前缀缓存；
  MicroInfer 和 nano-vllm 都按这个口径重置，HF 没有前缀缓存。`shared-prefix` 会先
  prime 共享前缀，因此仍能测到跨请求复用。`--cache-mode warm` 保留重复 workload 的
  前缀命中，用于观察稳态收益；两种口径不能混在同一张表里。
- **TTFT 是引擎侧**：不包含 tokenizer 编码和 detokenize；端到端文本生成的时延会更高。
- **单机单次结果**：不要和 vLLM 的公开吞吐直接比较；硬件、模型、workload 和采样参数
  不同。这里的价值是同一环境下的前后对比和消融。
- **先 warmup 再测**：nano-vllm 首次 batch-8 可能只有稳态的 30%，compile/graph capture
  和 kernel autotune 必须排除在计时之外。
