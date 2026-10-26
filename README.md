# MicroInfer

MicroInfer 是一个参考 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的轻量级
Qwen3 推理引擎，支持 CPU 与单卡 CUDA，适合学习和实验推理引擎的核心组件。
模型计算、调度和 KV 缓存由本项目实现，Transformers 用于 tokenizer 加载和独立数值验证。

## 功能

- 单进程连续批处理、decode 优先的混合 batch 与 chunked prefill
- 分页 KV 缓存、LRU 前缀复用、内存压力下的抢占重算
- Qwen3 GQA、Q/K RMSNorm、split-half RoPE 与 SwiGLU
- 温度采样、贪心生成、EOS / 长度停止与取消请求
- 本地 safetensors 权重加载，支持 tied embeddings
- CPU 参考实现与可选的 FlashAttention 后端
- 可复现离线 benchmark（TTFT、TPOT、吞吐、显存），含 HF 与 nano-vllm 基线
- 可选 decode CUDA Graph / decode-only `torch.compile`
- 任意 2 的幂 block size：大 block 走 FlashAttention 分页 kernel，小 block 走
  Triton paged decode + gather/varlen prefill
- AsyncEngine 流式输出、请求取消、背压与混合调度
- OpenAI 兼容 HTTP 服务（completions/chat/SSE/health/Prometheus metrics）

默认采用 eager 执行；性能实验可开启 decode-only compile 或 CUDA Graph。
暂不支持 Tensor Parallel、量化模型、滑动窗口注意力和 scaled RoPE。

## 安装

需要 Python 3.10+ 和 PyTorch 2.8+。先安装适合机器的 PyTorch CPU/CUDA 版本，再在仓库目录运行：

```bash
python -m pip install -e '.[hf]'
```

开发和测试环境：

```bash
python -m pip install -e '.[hf,test]'
python -m pytest -q
```

HTTP 服务需要额外依赖：

```bash
python -m pip install -e '.[hf,serve]'
```

FlashAttention 为可选依赖。未安装或输入布局不满足约束时，使用 PyTorch 后端。

## 快速开始

准备本地 dense Qwen3 模型目录，包含 `config.json`、tokenizer 文件和 safetensors 权重。
以下示例中的 `models/Qwen3-0.6B` 应替换为自己的模型路径；模型文件不纳入仓库。

```python
from microinfer import LLM, SampleParameter

llm = LLM.from_pretrained(
    "models/Qwen3-0.6B",
    device="cuda",
    dtype="bfloat16",
    block_size=16,
    num_blocks=64,
)
outputs = llm.generate(
    ["Hello, my name is", "The capital of France is"],
    SampleParameter(max_tokens=16, greedy=True),
    use_tqdm=False,
)
print(outputs)  # [{"text": ..., "token_ids": [...]}, ...]
```

CPU 运行使用 `device="cpu", dtype="float32"`。
也可通过命令行执行，重复 `--prompt` 可以构造一个批次：

```bash
python -m microinfer --model models/Qwen3-0.6B \
  --prompt 'Hello, my name is' --max-tokens 16 --greedy --device cuda
```

不下载模型的端到端示例：

```bash
python examples/tiny_engine.py
```

启动 OpenAI 兼容服务：

```bash
python -m microinfer.serve --model models/Qwen3-0.6B \
  --block-size 16 --num-blocks 1024 --use-cuda-graph

curl -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":16,"stream":true}'
```

`/health` 返回队列/KV 状态，`/metrics` 输出 Prometheus text。详细设计见
[P4 文档](docs/p4.md)。

## 接口约定

- `generate` 接受文本列表或 `list[list[int]]`，结果保持输入顺序。
- 采样参数可以是共享的 `SampleParameter`，也可以是逐请求列表。
- `max_tokens` 为生成上限，EOS 或上下文容量可能使请求提前结束。
- `temperature` 必须为有限正数；贪心模式使用 `greedy=True`。
- 输入按原始文本处理，不自动添加 chat template。对话可先由 tokenizer 的
  `apply_chat_template` 转为 token 列表，再传入引擎。
- 加载仅使用本地文件，不自动下载模型或执行远程代码。
- 已构造模型可通过 `LLM.from_model(model, tokenizer, ...)` 组装引擎。
- 同一个 LLM 实例不支持并发 `generate`。

单请求的 prompt 与 completion 总长度上限为
`min(engine.max_model_len, num_blocks * block_size)`。
增加 `num_blocks` 可以扩大 KV 池，也会增加内存或显存占用。

## 项目结构

```text
microinfer/
  config/       模型、调度与采样配置
  engine/       请求状态、LRU 块管理、混合调度、异步执行
  layer/        模型算子与 attention 后端
  model/        Qwen3 模型及权重加载
  serve/        OpenAI 兼容 FastAPI 服务与 metrics
  worker/       KV 张量存储
  runner.py     模型执行、metadata 与缓存绑定
  llm.py        同步生成接口
examples/       可运行示例
scripts/        独立参考验证工具
bench/          离线/在线基准、引擎适配与 profiling
tests/          单元、组件、集成测试与测试专用 adapter（tests/support/）
docs/           架构与基准文档
```

## 测试与参考验证

```bash
python -m pytest -q
python -m pytest tests/integration -q
python -m pytest -m cuda -q
python scripts/verify_checkpoint.py --model models/Qwen3-0.6B \
  --device cuda --output results/qwen3-verification.json
```

默认测试使用小张量和临时随机 checkpoint，不下载模型。没有可用 CUDA 时，GPU 测试会跳过。
独立参考验证覆盖 dense、chunked prefill、decode、前缀复用与贪心生成。
详见 [测试指南](tests/README.md) 和 [架构说明](docs/architecture.md)。

## 性能基线

`bench/` 提供可复现的离线基准，覆盖单流、batch、混合长度、共享前缀和长 prefill；
指标包括 TTFT、TPOT、输出吞吐和峰值显存，基线为 HF Transformers 与 nano-vllm。
以下命令在仓库根目录运行；每个引擎在独立进程中执行，先 warmup 目标 shape，
再取多次运行的中位数：

```bash
python -m bench.run_all --model /path/to/Qwen3-0.6B --runs 3 --warmup 1
python -m bench.report --input-dir results/bench
```

当前 RTX 4060 Laptop / Qwen3-0.6B、block_size=256、cold cache 的结果
（output tok/s）：

| workload | eager | decode compile | decode CUDA Graph | HF | nano-vllm |
| --- | ---: | ---: | ---: | ---: | ---: |
| single-short | 69.5 | 89.2 | **110.4** | 75.1 | 152.6 |
| batch8-short | 497.5 | 576.0 | **678.1** | 531.3 | 962.7 |
| long-prefill | 52.5 | 56.5 | **64.4** | 56.9 | 93.7 |

`block_size=16` 不再需要为了性能退回 256；小 block 路径使用 Triton paged decode +
gather/varlen prefill（`num_blocks=1024`，KV 容量相同）：

| workload | bs16 eager | bs16 compile | bs16 graph | bs256 graph |
| --- | ---: | ---: | ---: | ---: |
| single-short | 63.4 | **126.9** | 112.5 | 110.3 |
| batch8-short | 469.4 | **710.7** | 676.9 | 679.1 |
| batch32-mixed | 847.8 | 847.0 | 914.3 | **918.0** |
| shared-prefix | 482.1 | 619.1 | **699.3** | 690.2 |
| long-prefill | 50.0 | 59.7 | **65.0** | 64.1 |

- compile 和 graph 是互斥模式；block16 的 Triton kernel 可被 Dynamo 完整追踪，
  decode 编译成单图，因此单流/batch8 上 compile 更快；高 batch/长上下文 graph
  与 `block_size=256` + FlashAttention 持平。
- 4K context decode 的 TPOT：block16 12.6 ms vs block256 12.1 ms。

P4 在线并发（64 请求、128 prompt、16 output、block16 + CUDA Graph）：

| 场景 | output tok/s | TTFT p99 | ITL p99 |
| --- | ---: | ---: | ---: |
| 同时到达，混合 batch | 1060.8 | 684 ms | 339 ms |
| 同时到达，prefill-first | **1098.1** | **659 ms** | 348 ms |
| 晚到 8×2048 prefill，混合 | 424.5 | — | **104 ms** |
| 晚到 8×2048 prefill，prefill-first | **435.7** | — | 1098 ms |

混合调度在晚到长 prefill 下把 decode 的秒级停顿压到约 100 ms；代价是约 3%
吞吐和 2.5% 晚到 TTFT。完整数据、chunk 大小权衡和 HTTP 服务说明见
[P4 调度与服务化](docs/p4.md)。

### 当前版本 vs nano-vllm

同一台机器、同一份 Qwen3-0.6B、cold cache、3 次运行中位数。MicroInfer 使用
P4 代码 + `block_size=16`、`num_blocks=1024`、CUDA Graph、
`max_num_prefill_tokens=8192`；nano-vllm 使用 `bb823b3` 的默认 compile + CUDA Graph、
`block_size=256`、193 个 KV block。

| workload | MicroInfer tok/s | nano-vllm tok/s | nano / MicroInfer |
| --- | ---: | ---: | ---: |
| single-short | 116.3 | 152.6 | 1.31× |
| batch8-short | 695.3 | 962.7 | 1.38× |
| batch32-mixed | 933.7 | 1406.0 | 1.51× |
| shared-prefix | 723.8 | 938.9 | 1.30× |
| long-prefill | 66.6 | 93.7 | 1.41× |

| workload | MI TTFT | nano TTFT | MI TPOT | nano TPOT | MI peak | nano peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single-short | 17.4 ms | 17.3 ms | 8.3 ms | 6.2 ms | 3.09 GB | 6.58 GB |
| batch8-short | 65.8 ms | 46.9 ms | 9.7 ms | 7.1 ms | 3.13 GB | 6.61 GB |
| batch32-mixed | 666.7 ms | 362.7 ms | 13.9 ms | 11.7 ms | 3.41 GB | 6.78 GB |
| shared-prefix | 30.9 ms | 18.9 ms | 10.4 ms | 8.2 ms | 3.13 GB | 6.59 GB |
| long-prefill | 164.9 ms | 121.4 ms | 10.2 ms | 7.1 ms | 3.18 GB | 6.63 GB |

解读：

- P0 时 MicroInfer 单流约为 nano-vllm 的 1/4，现在差距缩小到 **1.30–1.51×**；
  batch32 和 long-prefill 仍是最明显的差距，主要来自 prefill TTFT 和 GPU 计算效率。
- 显存差异主要来自 **KV 池容量**，不是算子或 attention 的显存效率：nano-vllm 用
  90% GPU 自动分配 KV，193 个 256-token block ≈ **5.3 GiB KV**；MicroInfer 固定
  1024 个 16-token block ≈ **1.75 GiB KV**。权重都约 1.2 GB，activation/graph
  workspace 也在同一量级。batch32 只用了 MicroInfer KV 池的约 52%，所以这份额外
  容量对当前 workload 的吞吐没有直接帮助；把 `num_blocks` 调到约 3072 就能得到
  接近 nano 的显存占用。
- 两者采样口径不同：MicroInfer 是 greedy，nano 参考实现没有 greedy，使用
  `temperature=0.6`。因此这里比较的是吞吐，不是逐 token 输出一致性。
- 复现时把 `--nano-root` 指向 nano-vllm checkout；详细口径见
  [基准测试](docs/benchmarks.md)。

把 MicroInfer 改成与 nano 相同的 KV 容量（`block_size=256`、`num_blocks=193`，
KV 5.43 GiB）后，五组吞吐为 **112.3 / 694.7 / 922.0 / 703.1 / 65.2 tok/s**，
相对 nano 仍是 **1.34–1.52×**，与 1.79 GiB KV 池基本一致。说明在这些 workload
下显存容量不是瓶颈；更大的 KV 只在容量受限时有收益：64 个 512-token prompt 的
在线测试中，1024 block 触发 1057 次前缀淘汰、292.8 tok/s、TTFT p99 3341 ms；
3088 block 时淘汰为 0、311.3 tok/s、TTFT p99 2882 ms。

同容量下 MicroInfer 峰值约 6.7–7.05 GB，nano 约 6.58–6.78 GB；MicroInfer 需要
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 才能稳定跑 batch32，多出的
约 0.2–0.3 GB 来自 CUDA Graph pool 和 prefill 临时张量。

完整消融和 profiling 见 [P1 优化记录](docs/p1.md)、
[P2 CUDA Graph](docs/p2.md) 与 [P3 Attention 快路径](docs/p3.md)；
两种执行优化的原理、取舍和排查方法见
[torch.compile 与 CUDA Graph](docs/compile-and-cuda-graph.md)。

默认使用 cold cache；`--cache-mode warm` 用于观察重复 workload 的前缀缓存稳态收益。
profiling 默认输出 Chrome/Perfetto JSON + TensorBoard trace，`--export nsys` 输出
可直接用 Nsight Systems 打开的 `.nsys-rep`：

```bash
python -m bench.profile_runner --model /path/to/Qwen3-0.6B \
  --workload single-short --max-tokens 8 --export torch
python -m bench.profile_runner --model /path/to/Qwen3-0.6B \
  --workload single-short --max-tokens 1 --export nsys
```

完整结果、profiling 热点和可视化方式见 [docs/benchmarks.md](docs/benchmarks.md)；
P1 的 `debug_mode` / `use_compile` 开关和消融记录见 [docs/p1.md](docs/p1.md)，
decode CUDA Graph 的设计与结果见 [docs/p2.md](docs/p2.md)，小 block 的 Triton
decode 与 gather/varlen prefill 见 [docs/p3.md](docs/p3.md)，混合调度、
异步引擎与 OpenAI 服务见 [docs/p4.md](docs/p4.md)。

## 参考项目

设计参考 nano-vllm，调度策略包含对其实现的改编。相关来源与版权声明保留于
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
