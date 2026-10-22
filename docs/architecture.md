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
| CachedAttention | 写入本层缓存，将执行上下文适配为后端参数 |
| paged_attention | 物理块读取、GQA 与带前缀偏移的 causal mask |
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

## KV 缓存与 Context

Runner 持有 `KVCache.keys/values: [layers, blocks, B, Hkv, D]`。
`cache.layer(i)` 返回本层 `[blocks, B, Hkv, D]` 张量视图，绑定过程不复制存储。
每层的 `Attention.attn: CachedAttention` 持有该视图；模型 forward 不传全局池或层号。

LayerKVCache.write 支持 `-1` padding，并在写入前验证形状、范围和重复 slot。
BlockManager 管引用计数，释放块不清零张量。计算只读取有效块表与 context lengths 指定的数据。

Context 使用 ContextVar 保存本轮的 slot、位置、序列长度与块表。Runner 在 finally 中清理，
数值测试 adapter 使用 use_context 恢复外层状态。Context 隔离不等于共享缓存可并发写入。

先确定模型 dtype/device，再分配池；更换池时需要同步重建 BlockManager，不能保留旧前缀索引。
缓存是运行时数据，不属于模型权重，也不写入 state_dict。

## 模型与 attention 后端

Qwen3 使用 bias-free SwiGLU、逐 head Q/K RMSNorm、split-half RoPE 和 GQA。
`rope.apply` 同时保留相邻实虚数对布局供独立算子测试，模型默认采用 Qwen3 的 split-half 布局。
RMSNorm 在 FP32 中归一化，并返回累加后的 residual。

Attention.forward 支持 dense `[batch, length, hidden]` 与 packed `[tokens, hidden]`。
dense 路径不改动 KV 池，packed 路径通过 Context 读写本层缓存：

- 无前缀 prefill 直接对本轮 K/V 做逐请求 SDPA，同时写缓存。
- 有前缀 prefill 的 context lengths 取 `diff(cu_seqlens_k)`。
- decode 每请求一个 query，Q offsets 为 `arange(N + 1)`。
- 分页参考后端的 FP16/BF16 输入在 FP32 中计算 QK、softmax、PV，输出转回输入 dtype。
- 满足设备、dtype、head dimension 和块大小约束时，可调用 FlashAttention 分页后端。

`compute_logits` 在 prefill 选择每请求最后一个 query，在 decode 保留所有行。
`forward_batch(batch)` 为数值验证 adapter，返回全部 query 的 logits；调用前由 Runner 绑定测试池。

## 权重与验证

`load_model` 流式读取 safetensors，映射 Qwen3 参数名并支持 tied embeddings。
通用模块也支持 nano-vllm 风格的 packed_modules_mapping / weight_loader 回调。
`load_weights(mapping)` 是严格映射接口，先验证全部键和形状，再修改参数。

`tests/integration/test_hf_parity.py` 使用独立 Transformers Qwen3 的随机小 checkpoint，
比较 dense、chunked、decode、共享前缀及完整贪心生成。
`scripts/verify_checkpoint.py` 顺序加载真实模型，避免同时驻留两个完整模型，验证 logits 与生成 token。
参考项目和改编代码的声明见 [第三方声明](../THIRD_PARTY_NOTICES.md)。

`bench/` 是推理主路径之外的离线性能工具：它驱动 Scheduler/LLMEngine 直接收集 TTFT、TPOT、
吞吐与显存，并提供 HF 与 nano-vllm 适配器。基线、profiling 热点和对比口径见
[基准测试](benchmarks.md)。
