# MicroInfer 模块验证

测试针对模块应满足的接口与数学定义编写，不需要下载模型或 tokenizer。
当前项目仍在开发中：默认运行会真实报告已有实现缺陷，不会将其自动转换为
`xfail` 或跳过。`known_bug` 只是筛选标记；修复对应实现后应删除该标记。

完整引擎的后续拆分见 [开发路线](../docs/engine-roadmap.md)。`tests/roadmap/` 包含
参考 nano-vllm 拆分的 Scheduler、Context、KV 实体存储、paged attention、sampler、loader、cached model、
runner、engine 和输入输出接口的验收；`roadmap` 标记表示待实现阶段，不会自动跳过失败。

## 安装与运行

在 `MicroInfer/` 目录执行：

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

现有工作区虚拟环境也可以直接使用：

```bash
../.venv/bin/python -m pytest -q
```

PyTorch 请按机器的 CPU / CUDA 环境安装。FlashAttention 仅真实 GPU attention
和模型集成测试需要，CPU 组件测试不导入它。CUDA 不可用或 FlashAttention 未安装时，
相关 GPU 测试会显示明确的 skip 原因；安装了但二进制不兼容则会真实失败。

常用模块化开发命令：

```bash
# 纯元数据组件：不需要 torch、GPU 或模型权重
python -m pytest tests/engine tests/test_config.py -q

# 单独验证 BlockManager；筛选 hash 发布用例
python -m pytest tests/engine/test_blockmanager.py -q
python -m pytest tests/engine/test_blockmanager.py -k hash_publishes -x -vv

# 数学算子 / 模型接口；CPU 用例
python -m pytest tests/layers -m 'not cuda' -q
python -m pytest tests/model -m 'not cuda' -q

# 当前已实现功能的回归子集（不代表全项目通过）
python -m pytest -m 'not known_bug and not roadmap and not cuda' -q

# 缺陷清单 / GPU / 跨组件验证
python -m pytest -m 'known_bug and not cuda' --tb=short
python -m pytest -m cuda -q
python -m pytest -m integration -q

# 持续集成可读取的结果
python -m pytest --junitxml=/tmp/microinfer-tests.xml
```

如果仅开发 engine，只需安装 `pytest numpy xxhash`，然后指定上述 engine/config 路径。
不指定路径会收集算子测试，因而需要 PyTorch。

## 覆盖与约定

| 组件 | 验证内容 |
| --- | --- |
| Config / SampleParameter | 配置独立性、参数透传、温度边界、Attention 所需字段 |
| Sequence | token 所有权、块边界、追加 token、状态、prefill/decode worker 序列化 |
| Block | 分配后重置内容与引用计数 |
| BlockManager | 容量准入、free/used 分区、精确引用计数、共享释放、缓存重新激活、淘汰与 hash 碰撞防御、尾块重算、decode 跨块、EOS 释放、chunked prefill |
| RoPE | 逆频率公式、缓存长度不变性、旋转数值、形状、dtype、设备、不修改输入 |
| RMSNorm | 独立均方根公式、权重缩放、零输入、残差累积、fused/unfused、FP32/FP16/BF16 |
| FFN | 投影维度、Qwen bias 约束、真实 `nn.Module` 调用、SwiGLU 公式 |
| Attention | GQA 投影宽度、逐 head Q/K norm、GPU causal GQA 数值参考 |
| TransformerLayer | 位置参数传递、两次归一化与残差连接的组合 |
| QwenModel | 模块注册、embedding/final norm/logits、位置与最终残差、GPU 多层 prefill 因果性 |
| 导入边界 | engine 不加载 torch；CPU layer 不依赖 FlashAttention |

验证分层：

- BlockManager 是 CPU 元数据管理器。分配/释放单元测试通过显式 fixture 构造
  “完整块已计算”的缓存状态，独立验证所有权与缓存复用行为。
  标记为 `integration` 的 chunked prefill 测试走真实 `hash_blocks` 调用链。
- `num_scheduled_tokens` 表示本轮计划执行的 token 数；执行成功后发布 hash 时尚未递增
  `num_cached_tokens`。仅发布完整块，最后一块在新请求 prefill 时仍需重算。
- RMSNorm 的第二个返回值约定为累加后的 residual，供后续层继续累加。
  fused 算术测试直接执行编译包装器的原函数，不验证 `torch.compile` 性能或后端。
- RoPE 按当前 `apply` 文档的相邻实虚数对布局验证。它不同于 HF Qwen 的
  split-half 权重布局；这组测试不宣称已实现预训练权重兼容。
- Transformer 编排单测替换 attention/FFN 为确定性小模块，保留真实 norm；
  模型编排单测替换 decoder layer。GPU 集成用例使用真实所有层。
- Sequence 序列化按模型 worker 的精简传输协议验证，不要求恢复完整调度状态。
- 后续模块已有接口骨架与 `tests/roadmap/` 验收，算法仍待实现。
  引擎编排单测使用小型替身；真正模型与 KV 数值由独立的集成验收覆盖。

## 当前开发基线

当前 BlockManager 已完成并通过 30 项测试，Sequence 通过 10 项；
BlockManager 已解决的 `known_bug` 标记已移除。新增基础配置与快照契约另有 16 项测试。
排除 `known_bug`、`roadmap` 和 `cuda` 后，当前已完成回归子集为 73 项。
本次完整运行共 206 项：73 通过、20 跳过、113 失败；失败中 22 项属于既有缺陷，
91 项属于后续模块验收。接口骨架保留 TODO，不以修改断言掩盖未完成实现。

全量运行中的失败分为两类：既有算子/模型的已知缺陷，以及新接口骨架的待实现失败。
不要把 `roadmap` 中的 NotImplementedError 当成已经完成组件的回归错误；
也不要以该子集筛选命令通过来宣称完整引擎可用。

失败原因可以通过 `pytest --tb=short` 和各测试的 `known_bug(reason=...)` 定位：

| 组件 | 当前阻塞点 |
| --- | --- |
| BlockManager | 已通过当前 30 项回归，后续由 Scheduler 集成测试继续验证 |
| Config | 缺少 `rope_base` |
| RoPE | 频率公式、`shape(...)`、保留 head 维、BF16 转换 |
| RMSNorm | fused 构造参数类型、均方根公式、残差累积 |
| FFN | `foward` 拼写、gate 激活位置、Qwen projection bias |
| Attention | output projection 宽度、Q/K norm 维度、RoPE 和 FlashAttention 参数 |
| TransformerLayer | `foward` 拼写及残差传递 |
| QwenModel | `nn.modules` 基类、`super` 初始化、层数迭代、参数顺序与最终 norm 返回值 |

修复一个组件时，先运行该文件，再运行相关 integration 测试。参数化用例会分别
显示每个输入边界或 dtype 的失败，失败数量不等于独立缺陷数量。

## nano-vllm 参考修订

当前基线为 nano-vllm `bb823b3`，不是 vLLM V1。调度接口为
`schedule() -> (seqs, is_prefill)`、`postprocess(seqs, token_ids, is_prefill)`；
Runner 使用 `run`，引擎使用 `is_finished`，prefill/decode 分轮执行。
`nano` 标记选择核心对齐测试，`extension` 标记可选辅助接口与更强校验。
`SchedulerOutput` / `forward_batch` 只保留为数值验证 helper，不属于主引擎协议。

```bash
../.venv/bin/python scripts/check_nano_scheduler.py
../.venv/bin/python -m pytest tests/roadmap -m 'nano and not extension' -q
```

前一条命令验证测试与参考源码相容（12 项通过），后一条命令验收你自己的实现。
二者不能互相替代。完整模块映射与差异说明见开发路线。
