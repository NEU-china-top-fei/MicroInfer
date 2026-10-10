# MicroInfer

模块化验证测试位于 `tests/`，使用小尺寸数据和独立数学参考，无需下载模型。

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

当前实现仍有未完成部分，默认测试会如实报告失败。模块运行命令、接口约定、
GPU 条件和已知缺陷见 [测试开发指南](tests/README.md)。

从已完成的 BlockManager 继续构建推理引擎，见
[组件拆分、接口契约与 M01–M10 开发路线](docs/engine-roadmap.md)。
新增阶段验收位于 `tests/roadmap/`；对应骨架明确抛出 `NotImplementedError`，供逐模块实现。
