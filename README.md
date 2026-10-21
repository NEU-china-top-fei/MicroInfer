# MicroInfer

MicroInfer 是一个参考 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的轻量级
Qwen3 推理引擎，支持 CPU 与单卡 CUDA，适合学习和实验推理引擎的核心组件。
模型计算、调度和 KV 缓存由本项目实现，Transformers 用于 tokenizer 加载和独立数值验证。

## 功能

- 单进程连续批处理与 chunked prefill
- 分页 KV 缓存、前缀复用、内存压力下的抢占重算
- Qwen3 GQA、Q/K RMSNorm、split-half RoPE 与 SwiGLU
- 温度采样、贪心生成、EOS / 长度停止与取消请求
- 本地 safetensors 权重加载，支持 tied embeddings
- CPU 参考实现与可选的 FlashAttention 后端

默认采用 eager 执行。暂不支持 Tensor Parallel、CUDA Graph、量化模型、滑动窗口注意力和
scaled RoPE；本项目不以生产服务或吞吐性能为目标。

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
  engine/       请求状态、物理块管理、调度与执行循环
  layer/        模型算子与 attention 后端
  model/        Qwen3 模型及权重加载
  worker/       KV 张量存储与输入打包
  utils/        执行上下文
  runner.py     模型执行与缓存绑定
  llm.py        生成接口
examples/       可运行示例
scripts/        独立参考验证工具
tests/          单元、组件和集成测试
docs/           架构文档
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

## 参考项目

设计参考 nano-vllm，调度策略包含对其实现的改编。相关来源与版权声明保留于
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
