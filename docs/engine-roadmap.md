# 按 nano-vllm 完成 MicroInfer

参考项目是 **[GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)**。
本路线以本地 checkout 的 `bb823b3e06983d71485a8e1f23715ebd87d98ef8` 为基线，
查阅日期 2026-10-11。该版本包含 chunked prefill；不同教程中的旧版本可能不同。

目标先限定为单进程、单卡、Qwen3 文本生成、eager 执行，完成后再增加 nano-vllm
已有的 Tensor Parallel、torch.compile、CUDA Graph。当前交付仍是**接口骨架与验收测试**，
不是已经实现的推理引擎。你已完成的 Sequence / BlockManager 算法不改动。

## 主调用链

对照 [LLMEngine](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/llm_engine.py)：

```text
LLM.generate(prompts, sampling_params)
  → tokenize / Sequence / Scheduler.add
  → while not engine.is_finished():
      seqs, is_prefill = scheduler.schedule()
      token_ids = model_runner.run(seqs, is_prefill)
      scheduler.postprocess(seqs, token_ids, is_prefill)
      收集本轮已结束序列的 completion_token_ids
  → 按输入顺序返回 [{"text": ..., "token_ids": [...]}, ...]
```

这里没有 SchedulerOutput 消息层，也不混合 prefill/decode。
源码通过 `model_runner.call("run", ...)` 兼容多进程通信；MicroInfer 单进程阶段直接调用
`runner.run`。构造函数保留依赖注入，便于 CPU 单测，不要求初始化 GPU、进程组或下载模型。

## 分模块实现顺序

| 阶段 | nano-vllm 对照文件 | MicroInfer 实现位置 | 对应验收 |
| --- | --- | --- | --- |
| M00 已完成 | `engine/sequence.py`, `engine/block_manager.py` | `engine/sequence.py`, `engine/blockmanager.py` | `tests/engine/`，40 项 |
| M01 调度 | `engine/scheduler.py` | `engine/scheduler.py` | `test_scheduler.py`：prefill 优先、同质 batch、chunk、抢占、EOS |
| M02 输入与上下文 | `engine/model_runner.py`, `utils/context.py` | `runner.py:prepare_prefill/prepare_decode`, `utils/context.py` | `test_runner.py`, `test_context.py`：位置、slot、Q/K offsets、上下文清理 |
| M03 KV 张量 | `model_runner.allocate_kv_cache`, `layers/attention.py` | `worker/kv_cache.py`，由 Runner 分配并挂到各层 | `test_kv_cache.py`：实体存储、非连续块、padding、层隔离 |
| M04 注意力 | `layers/attention.py` | `layer/paged_attention.py`、现有 Attention | `test_paged_attention.py` 与 `tests/layers/test_attention.py`：写 KV、GQA、causal mask |
| M05 采样 | `layers/sampler.py` | `layer/sampler.py:forward` | `test_nano_sampler.py`：可调用模块、温度、分类分布 |
| M06 加载 | `utils/loader.py`, `layers/linear.py` | `model/loader.py:load_model` | `test_nano_loader.py`：本地 safetensors、普通参数、packed shard 回调 |
| M07 模型 | `models/qwen3.py`, `layers/*` | 现有 model/layer 文件 | `tests/layers/`, `tests/model/`, `test_nano_lm_head.py`, `test_model_cache.py`：基础数值、每请求一行 logits、cached forward 等价 |
| M08 Runner | `engine/model_runner.py:run/run_model` | `runner.py` | `test_runner.py`：每序列一个采样结果、顺序不变、上下文生命周期 |
| M09 引擎循环 | `engine/llm_engine.py:step` | `engine/llm_engine.py` | `test_engine.py`：schedule→run→postprocess、完成结果、正负 token 计数 |
| M10 外部接口 | `engine/llm_engine.py:generate`, `llm.py` | `llm.py`，输入输出 helper 可内联 | `test_llm.py`：共享/逐请求采样参数、dict 输出、输入顺序 |
| 最终数值验收 | 独立本地参考模型 | 转换后的本地权重与参考产物 | `test_offline_reference.py`：logits/下一 token 对齐 |

上表省略目录的测试文件均位于 `tests/roadmap/`。
建议从 **M01 Scheduler** 开始，再做 M02–M06，修好现有模型算子后连接 M07–M10。

## M01：必须对齐的调度行为

对照 [Scheduler 源码](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py)：

1. `add(seq)` 放入 waiting。`is_finished()` 判断 waiting 和 running 是否均为空。
2. `schedule() -> (list[Sequence], is_prefill)` 先尝试 waiting。选出任何 prefill 后就返回，
   即使预算还有余量，也不把 running decode 混进来。
3. prefill 受 `max_num_batched_tokens` 和 `max_num_seqs` 限制。**只有本轮第一个请求可以
   被截成 chunk**；后续请求剩余 token 放不下时留待下轮，不再拆分。
4. 不完整 chunk 仍在 waiting；本轮即将完成全部已知 token 的请求在 schedule 时移入 running。
5. 没有可调度的 prefill 时才执行 decode，每个请求一个 token，最多 `max_num_seqs` 个。
   该版本的 decode 不额外应用 prefill 的 token budget。
6. decode 缺块时从剩余 running 队尾弹出请求并 `preempt(seq)`；preempt 负责释放、
   标记 WAITING/is_prefill 并 appendleft 到 waiting，调用方负责先从 running 移除。
7. `postprocess(seqs, token_ids, is_prefill)` 按位置配对，先 hash 完整块，再推进
   num_cached_tokens、清零 num_scheduled_tokens。不完整 prefill 的采样 token **丢弃**；
   完成 prefill 或 decode 时才 append_token。EOS 或 max_tokens 触发完成并释放块。

`preempt` 接收 Sequence，不是 seq_id。`postprocess` 返回 None，不产生 RequestOutput 事件。
在引擎循环先检查 `is_finished()`；上游空 schedule 使用断言，不应把空轮次行为混入核心验收。

当前 BlockManager 与参考版本一样预留整个已知序列的块；chunked prefill 限制每轮计算量，
并不实现按 chunk 懒分配物理块。前缀复用始终保留最后一块重算。

## M02 / M08：Context 与 ModelRunner

对照 [ModelRunner](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py)
和 [Context](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/utils/context.py)。

`prepare_prefill(seqs)`：

- 输入切片是 `[num_cached_tokens : num_cached_tokens + num_scheduled_tokens]`；
- positions 是这些 token 在原序列中的绝对位置；
- `cu_seqlens_q` 累计本轮 query 长度，`cu_seqlens_k` 累计前缀加本轮后的 context 长度；
- 没有 cached prefix 时 Context.block_tables=None；有 prefix 时需要物理块表；
- slot 是 `block_table[position // B] * B + position % B`，类型 int32。

`prepare_decode(seqs)` 每序列只传 last_token、len(seq)-1 和一个物理 slot；Context 使用
context_lens 和 block_tables，不沿用 prefill 的 Q/K offsets。prepare 方法返回 `(input_ids, positions)`，
其余信息写入当前调用的 Context。单进程同步版本可用全局 Context；不要用于并发引擎。

`run_model` 走 `model(input_ids, positions)` 得 hidden，再调用 `model.compute_logits(hidden)`。
nano-vllm 的 LM head 在 prefill 用 `cu_seqlens_q[1:]-1` 选择每个请求的最后一行，decode 用所有行。
因此 logits 和 `run` 的输出都与 seqs **一一对应**，包含未完成 chunk；丢弃逻辑属于 Scheduler。
`test_nano_lm_head.py` 直接验收真实 QwenModel 的选行行为；现有 compute_logits 仍需补齐这一步。

`run(seqs, is_prefill) -> list[int]` 调用 `sampler(logits, temperatures)`，结束后 reset_context。
异常时也用 finally 清理是 MicroInfer 增强要求，上游这一版本不是 finally；该测试标为 extension。

## M03–M07：张量与数值

nano-vllm 在 Runner 中分配 `[2,layers,blocks,B,Hkv,D]` 的 KV 张量，并把每层视图赋给
attention 的 k_cache/v_cache。MicroInfer 的 KVCache helper 将 K/V 分开存为
`[layers,blocks,B,Hkv,D]`，便于独立验证，最后仍由 Runner 持有并向各层挂载视图。
**BlockManager 管所有权，KVCache 管张量；释放引用不等于立即清零 KV。**

PagedAttention 的 CPU 参考测试保留：已缓存 p 个 token，本轮 query 的第 j 行只能看
key `[0,p+j]`。跨请求隔离、GQA head 映射、-1 padding、非连续物理块必须先验证正确，
再接 nano-vllm 的 `store_kvcache` / FlashAttention prefill/decode 路径。

采样核心改为 `Sampler(nn.Module).forward(logits, temperatures)`。nano-vllm 使用正温度、
FP32 softmax 和 exponential-race categorical sampling。概率分布是验收目标，不要求不同实现
产生逐个相同的随机 token。上一版的 greedy 开关、probabilities/sample helper 保留为 extension，
不再作为 Runner 的核心依赖。

权重加载核心改为 `load_model(model, path)`：本地 safetensors → 参数 → weight_loader。
`packed_modules_mapping` 把 Q/K/V 或 gate/up 的源名称映射到目标参数和 shard_id；
普通参数用 default_weight_loader。测试用临时小权重文件，无需真实模型下载。
上一版 `load_weights(mapping)` 的原子性校验是增强功能，不是上游行为。

已有 norm、RoPE、FFN、模型测试继续使用。真实 Qwen 权重的 RoPE 布局和 projection 名称
必须与模型实现统一。`forward_batch(batch, kv_cache)` 及 `worker/batch.py` 保留为可选的
数值验证 adapter：它们方便比较所有 query 的 logits，不是 nano-vllm 主调用链必须增加的接口。
`test_model_cache.py` 验证 full/chunk/decode、独立/拼接请求以及共享前缀的自一致性；
独立预训练参考的 logits 对齐仍然是最终检查，不能由自一致性代替。

## M09 / M10：引擎返回值

`step()` 返回 `(outputs, num_tokens)`：

- outputs 只包含完成请求的 `(seq_id, completion_token_ids)`；
- prefill 计数是正的 `sum(seq.num_scheduled_tokens)`，必须在 postprocess 清零前取值；
- decode 计数是 `-len(seqs)`；
- Runner 抛错时不调用 postprocess，避免发布未完成 KV。

`generate(prompts, sampling_params, use_tqdm=True)` 接受统一参数或逐请求参数列表，
按输入顺序返回 `[{"text": ..., "token_ids": [...]}, ...]`。MicroInfer 的输入输出 helper
是单元测试边界；按 nano-vllm 简化代码时可合并进 LLMEngine，LLM 也可以像上游一样做薄继承。

## 验收分组与命令

`nano` 表示核心上游行为，`extension` 表示增强检查或独立辅助接口，`roadmap` 表示阶段仍待实现。
这些标记都不会自动 skip/xfail。`known_bug` 继续标记现有算子错误。

```bash
# 已完成的 Sequence / BlockManager：40 项
../.venv/bin/python -m pytest tests/engine -q

# 先完成 nano Scheduler
../.venv/bin/python -m pytest tests/roadmap/test_scheduler.py -x -vv

# nano 核心接口验收；暂不要求增强功能
../.venv/bin/python -m pytest tests/roadmap -m 'nano and not extension' -q

# 可复现地验证调度测试符合本地 nano-vllm 参考
../.venv/bin/python scripts/check_nano_scheduler.py
# 其他目录可用 --reference-root /path/to/nano-vllm

# 已完成部分的回归
../.venv/bin/python -m pytest -m 'not known_bug and not roadmap and not cuda' -q

# 所有阶段和可选增强
../.venv/bin/python -m pytest tests/roadmap -q --tb=short
```

`check_nano_scheduler.py` 仅适配 import/构造参数，用真正的 nano Scheduler 方法和
MicroInfer Sequence/BlockManager 运行 12 项核心调度测试，已全部通过。
它不会修改源码，也不能替代 MicroInfer 自身 Scheduler 的测试：后者当前仍是 TODO。

可选增强包括输入验证、取消请求、非法写入原子性、加载前完整校验、异常后上下文清理。
这些规则不应被误认为 nano-vllm 原本就具备。单请求超过整个 KV 池、重复提交、
并发调用等健壮性设计也应在核心链路完成后独立推进。

## 离线真实权重验收

保留 `test_offline_reference.py`。设置 `MICROINFER_REFERENCE_DIR`，提供：

- `microinfer_config.json`：最终补齐 rope_base 等字段的 MicroInfer 配置；
- `weights.pt`：已转换为 MicroInfer 参数名/布局的 state_dict；
- `case01.pt`：独立参考 input_ids `[1,L]`、logits `[1,L,V]`，可由现有 dump_reference.py 导出。

```bash
MICROINFER_REFERENCE_DIR=/path/to/converted-reference \
  ../.venv/bin/python -m pytest tests/roadmap/test_offline_reference.py -vv
```

这是复用已有参考产物的验证 adapter，不是 nano-vllm 原生的 safetensors 加载格式。
原生加载另由 `test_nano_loader.py` 验证。测试不下载权重；未指定产物则明确跳过，指定后缺失则失败。
