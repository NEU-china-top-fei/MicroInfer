# 架构

MicroInfer 将请求管理、模型执行和 KV 张量存储分离。引擎采用单进程、同步 eager 执行，
同一模型实例的缓存不能并发写入。

## 生成流程

```text
LLM.generate
  → InputProcessor → Sequence → Scheduler.add
  → while not engine.is_finished():
      seqs, is_prefill = scheduler.schedule()
      token_ids = runner.run(seqs, is_prefill)
      scheduler.postprocess(seqs, token_ids, is_prefill)
      收集已完成序列
  → OutputProcessor → 按输入顺序返回 text / token_ids
```

`LLM.from_pretrained` 读取本地配置、safetensors 和 tokenizer。
`LLM.from_model` 接受已构造的 QwenModel 和 tokenizer，适合实验及模块化测试。

## 模块边界

| 模块 | 职责 |
| --- | --- |
| Sequence | token 历史、已计算长度、采样参数和请求状态 |
| BlockManager | 物理块分配、引用计数、前缀 hash 和淘汰 |
| Scheduler | 请求准入、prefill/decode 选择、预算、抢占和完成 |
| KVCache / LayerKVCache | 全局张量分配与单层零拷贝视图 |
| CachedAttention | 写入本层缓存，接收每步 `AttentionMetadata` |
| paged_attention_meta | 物理块读取、GQA 与带前缀偏移的 causal mask（生产入口） |
| QwenModel | embedding、decoder layers、final norm 和 LM head |
| ModelRunner | 准备模型输入、绑定缓存、执行与采样 |
| LLMEngine | schedule → run → postprocess 循环 |
| LLM / processors | 输入验证、tokenizer 边界、输出顺序和文本解码 |

## 调度与恢复

调度策略参考 [nano-vllm bb823b3](https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8)：
prefill 优先，prefill/decode 分轮执行。每批只有第一个 prefill 请求可以截成 chunk。
未完成 chunk 的采样结果由 postprocess 丢弃；完成判断覆盖全部已知 token，包括历史生成 token。

decode 缺块时抢占剩余 running 队尾，释放物理块并保留 token 历史，之后重新 prefill。
只有模型执行成功后才发布完整块 hash。执行失败或取消请求会释放缓存引用；generate 失败时
会清理本次调用中已提交的请求，允许后续再次调用。

Scheduler 限制单轮在途执行，校验 postprocess 与已调度工作的一致性。
前端先验证所有 prompts，再入队；入队失败会回滚已提交请求。
`step()` 返回 `(已完成请求列表, token计数)`，prefill 计数为正，decode 为负。

单请求长度上限为 `min(engine.max_model_len, num_blocks * block_size)`。
超长 prompt 在入队前报错；生成到容量上限按长度结束。
Sequence 的 block_size 按实例设置，多个引擎不需要修改进程级全局配置。

## KV 缓存与 AttentionMetadata

Runner 持有 `KVCache.keys/values: [layers, blocks, B, Hkv, D]`。
`cache.layer(i)` 返回本层 `[blocks, B, Hkv, D]` 张量视图，绑定过程不复制存储。
每层的 `Attention.attn: CachedAttention` 持有该视图；模型 forward 不传全局池或层号。

LayerKVCache 有两个写入入口：`write_checked` 支持 `-1` padding，并在写入前验证形状、
范围和重复 slot；`write_fast` 接收 `AttentionMetadata.slot_indices` 预计算好的
`blocks/offsets`，不做 host 校验。生产 decode 走 `write_fast`，
`debug_mode=True` 时走 `write_checked`。
BlockManager 管引用计数，释放块不清零张量。计算只读取有效块表与 context lengths 指定的数据。

Runner 每步构建一次 `AttentionMetadata`，包含 slot、位置、`cu_seqlens_q/k`、
context lengths、block tables 和直接 prefill 标记；`ModelRunner.run_model` 把它显式传给
`QwenModel` → `TransformerLayer` → `Attention` → `CachedAttention`。CPU 侧的
`cu_q_cpu` / `context_lens_cpu` 是懒加载并缓存的：decode flash 路径不会 `.tolist()`，
只有 prefill/参考路径在第一次访问时转换一次。

metadata 是每步的显式运行时参数，没有全局 Context：`prepare_prefill/decode`
返回 `(input_ids, positions, metadata)`，模型、attention 和 cache 接收同一个对象。
测试需要的 packed-batch/forward adapter 放在 `tests/support/`，不进入运行时包。
共享 KV 缓存不能并发写入。

先确定模型 dtype/device，再分配池；更换池时需要同步重建 BlockManager，不能保留旧前缀索引。
缓存是运行时数据，不属于模型权重，也不写入 state_dict。

## 模型与 attention 后端

Qwen3 使用 bias-free SwiGLU、逐 head Q/K RMSNorm、split-half RoPE 和 GQA。
`QwenModel` 创建一个 `RotaryCache`，28 层共享同一张 FP32 `cos/sin` 表；表在第一次
forward 时按 activation device 懒构建，按 position 索引，不在每层重新计算角度或复数。
RMSNorm 走 `F.rms_norm` fused kernel（内部 FP32 累加），并返回累加后的 residual；
Q/K norm 复用同一个 `FuseLayerNorm`。

`QwenModel.fuse_weights()` 可选地把每层 Q/K/V 合并成一次 GEMM、gate/up 合并成一次
GEMM，结果放在非持久 buffer 中。原始 projection Parameter 随后被替换为该 buffer
的视图，与 packed 权重共享 storage；因此 checkpoint 映射和 `state_dict` key 不变，
稳态显存也不会翻倍。`load_state_dict` 写入视图时会同步更新 packed buffer。
`--no-fuse-weights` 可关闭打包用于 A/B。fuse 应在最终 `.to(device/dtype)` 之后调用。

Attention.forward 支持 dense `[batch, length, hidden]` 与 packed `[tokens, hidden]`。
dense 路径不改动 KV 池，packed 路径通过 `AttentionMetadata` 读写本层缓存：

- Runner 每步构建一次 metadata，显式传给 Attention/CachedAttention。
- 无前缀 prefill 先尝试 `flash_attn_varlen_func`（packed 请求一次调用）；CPU/FP32
  或形状不满足时回退到 4D `[1,H,L,D]` + `enable_gqa=True` 的 SDPA，不再
  `repeat_interleave` 或落到 math 后端。
- 有前缀 prefill 的 context lengths 由 metadata 从 `cu_seqlens_k` 差分得到。
- decode 的 Q offsets 由 metadata 懒生成，不再每层 `arange`。
- `paged_attention_meta` 是唯一的生产入口：不做 host 校验、不 `.tolist()`，直接使用
  int32/连续 tensor。测试用的严格校验入口在 `tests/support/attention.py`。
- 分页参考后端的 FP16/BF16 输入在 FP32 中计算 QK、softmax、PV，输出转回输入 dtype。
- 满足设备、dtype、head dimension 和块大小约束时，可调用 FlashAttention 分页后端。

`compute_logits` 在 prefill 选择每请求最后一个 query，在 decode 保留所有行；
metadata 显式传入。数值测试的 all-query adapter 在 `tests/support/model.py`。

### 执行与可选编译

Runner 默认 eager 执行。`use_compile=True` 时只编译 decode 的 `model.forward`，
prefill 继续 eager：prefill 有可变长度逐请求循环，compile 收益低且容易 graph break；
decode 形状更稳定，compile 能把 elementwise/norm 等小 kernel 融合。编译前 Runner 会
调用 `RotaryCache.ensure()` 建好 cos/sin 表，避免 Dynamo 追踪 40k 行表构造。
`AttentionMetadata`、fast KV 写入和显式 metadata 是这一阶段能 compile 的前提。
FlashAttention 的 pybind op 仍是 opaque custom kernel 边界，compile 会在它周围分段。

decode 的 metadata 使用 `_DecodeBuffers`：pinned CPU staging + 复用 device buffer，
每个 step 原地填充 `input_ids/positions/slot_mapping/context_lens/block_tables`，
再 `copy_(non_blocking=True)`。这去掉了每步 `torch.tensor(list, device="cuda")`
的 H2D 同步。

`use_cuda_graph=True` 时，Runner 在 KV cache 绑定后按 `[1,2,4,8,16,...]` bucket
捕获 decode Graph：

- batch 向上取整到 bucket，padding 行写入单独的 pad block，不触碰真实请求的 KV；
- 每个 bucket 有固定的 `_DecodeBuffers` 和 logits 输出 buffer；
- capture 使用全新的 `AttentionMetadata`，使 `slot_indices()` 的
  `long/div/mod` kernel 进入图内，replay 时根据新 slot 重新计算；
- `graph.replay()` 后只取真实 batch 的 logits，再 argmax/sample。

CUDA Graph 只用于 decode，prefill 仍 eager；它与 `use_compile` 互斥。唯一的架构性
同步仍是最后 `tokens.tolist()`。`Sampler` 的 logits 有限性校验只在
`debug_mode=True` 时执行。详细设计与结果见 [P2 文档](p2.md)；两种优化的原理和排查方法见
[torch.compile 与 CUDA Graph](compile-and-cuda-graph.md)。

## 权重与验证

`load_model` 流式读取 safetensors，映射 Qwen3 参数名并支持 tied embeddings。
加载完成后 `QwenModel.fuse_weights()` 可把 QKV 和 gate/up 打包成共享 storage 的
buffer；`--no-fuse-weights` 可关闭。

`tests/integration/test_hf_parity.py` 使用独立 Transformers Qwen3 的随机小 checkpoint，
比较 dense、chunked、decode、共享前缀及完整贪心生成。
`scripts/verify_checkpoint.py` 顺序加载真实模型，避免同时驻留两个完整模型，验证 logits 与生成 token。
参考项目和改编代码的声明见 [第三方声明](../THIRD_PARTY_NOTICES.md)。

`bench/` 是推理主路径之外的离线性能工具：它驱动 Scheduler/LLMEngine 直接收集 TTFT、TPOT、
吞吐与显存，并提供 HF 与 nano-vllm 适配器。基线、profiling 热点和对比口径见
[基准测试](benchmarks.md)。
