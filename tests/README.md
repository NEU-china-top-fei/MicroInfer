# 测试

安装测试依赖后运行：

```bash
python -m pip install -e '.[hf,test]'
python -m pytest -q
```

测试不会下载模型。CPU 测试不要求 FlashAttention；CUDA 不可用时 GPU 用例跳过。
Transformers 未安装时，独立参考测试会明确跳过；安装 test extra 可启用这部分测试。

## 测试组织

| 目录 | 覆盖 |
| --- | --- |
| engine/ | Sequence、物理块边界、引用计数、hash 发布与缓存复用 |
| layers/ | RMSNorm、RoPE、FFN、Attention 和残差编排 |
| model/ | Qwen3 模块注册、前向与 LM head |
| components/ | Scheduler、KV 存储、paged attention、Runner、加载与输入输出契约 |
| integration/ | 独立 Transformers 数值对照、真实引擎生成、失败恢复与容量边界 |

`tests/components/test_bench.py` 只覆盖 benchmark 的 workload 生成、指标插值和计时契约，
不加载模型。真实吞吐基线用 `bench/` 运行，见 [基准测试](../docs/benchmarks.md)。

```bash
python -m pytest tests/engine -q
python -m pytest tests/layers tests/model -q
python -m pytest tests/components/test_cache_binding.py -q
python -m pytest tests/integration -q
python -m pytest -m cuda -q
```

标记：`component` 选择组件测试，`integration` 选择联合测试，`nano` 选择参考版本核心行为，
`extension` 选择额外校验或辅助接口。标记本身不会把失败转为 skip / xfail。
分页 attention 测试覆盖 CPU/CUDA、FP32/FP16/BF16、causal mask、请求隔离和大 logits 溢出。

## 本地权重对照

```bash
python scripts/verify_checkpoint.py --model models/Qwen3-0.6B \
  --device cuda --output results/qwen3-verification.json
```

该工具将 MicroInfer 与 Transformers 的 FP32 logits、cached forward 和贪心生成比较，
输出可复现的误差记录。`results/` 是被忽略的本地产物目录，不应提交模型或测试输出。

## 参考调度器交叉检查

需要本地 nano-vllm checkout；建议使用参考版本 `bb823b3e06983d71485a8e1f23715ebd87d98ef8`：

```bash
python scripts/check_nano_scheduler.py --reference-root /path/to/nano-vllm
```

脚本仅适配 import 和构造参数，运行上游 Scheduler 的方法以验证测试契约。
它不替代 MicroInfer 自身的调度器测试。

## 可选离线参考产物

`components/test_offline_reference.py` 支持将 `MICROINFER_REFERENCE_DIR` 指向转换产物目录：

- `microinfer_config.json`：MicroInfer 模型配置。
- `weights.pt`：使用 MicroInfer 参数名的 state_dict。
- `case01.pt`：独立参考的 input_ids / logits。

未配置该目录时跳过，配置后文件缺失则失败。新的真实 checkpoint 验证推荐直接使用
verify_checkpoint.py，不需要转换权重。
