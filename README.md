# 边缘推理调度器（本地可复现）

用**本地设备探测 + 模拟运行时**承载多种模型变体，在硬件能力不匹配、并发资源
争抢、推理中途失败三种情形下给出**确定且可解释**的调度结论。

零外部依赖（运行时仅用 Python 标准库；Web 演示使用仓库已有的 FastAPI），
不需要真实加速卡、外部服务或凭据。

## 快速开始

```bash
uv sync                       # 安装依赖（含 pytest 开发依赖）
uv run pytest                 # 运行全部单测（20 个）
uv run pytest -s              # 同时打印每个用例的 输入/判定依据/归一化结果
uv run python examples/demo.py   # 命令行演示四类场景
uv run python main.py         # 可选：启动一个只读 HTTP 概览接口
```

## 四类测试场景

测试位于 `tests/`，日志统一打印「输入 / 判定依据 / 归一化结果」：

| 文件 | 场景 | 关键断言 |
| --- | --- | --- |
| `test_capability.py` | 能力缺失 | 缺精度与缺算子分别报告；拒绝后无残留占用 |
| `test_variant_matching.py` | 变体不兼容 | 默认严格模式不静默换变体；仅显式 `use_chain=True` 才按链降级 |
| `test_concurrency.py` | 并发争抢 | FIFO 串行、`queue_full` 与内存不足可区分、无重复占用、结果不串号 |
| `test_failures.py` | 推理中途失败 | 三类错误归一化可区分、确定顺序重试、无重复副作用、已释放资源不可再用 |
| `test_integration.py` | 端到端 | 混合并发负载下落点正确、设备状态最终一致 |

## 能力探测与变体匹配

探测固定返回三台设备（顺序确定）：真实 `local-cpu`（仅 fp32 + 基础算子）、
模拟 `sim-gpu`（fp16 + flash_attention，2048MB）、模拟 `sim-npu`
（int8 + rope 等，512MB 低显存）。变体要求的**全部精度**与**算子超集**
都满足才算兼容；内存占用在准入阶段判定。详见 `docs/design.md` 第 2–3 节。

## 降级顺序与错误分类

- 候选顺序：`首选变体(+显式后备链)` × `兼容设备(探测顺序)`；
- `unsupported_op / inference_timeout / device_unavailable` 归一化后释放占用
  并继续下一候选；`device_unavailable` 会令设备离线并在后续跳过；
- 全部失败聚合为 `fallback_exhausted`，`details.attempts` 保留每步可区分原因。

完整规则见 [`docs/design.md`](docs/design.md)。

## 以代码方式使用

```python
from edge_scheduler.devices import DevicePool, LocalProbe
from edge_scheduler.variants import ModelVariant, FallbackChain, VariantRegistry
from edge_scheduler.runtime import SimulatedRuntime, InferenceInput, FailureSpec
from edge_scheduler.scheduler import EdgeScheduler
from edge_scheduler.errors import ErrorKind

reg = VariantRegistry()
reg.register(ModelVariant("m", "int8", frozenset({"int8"}),
                          frozenset({"matmul", "rope"}), 400))
reg.register(ModelVariant("m", "fp16", frozenset({"fp16"}),
                          frozenset({"matmul"}), 1500))
reg.register_chain(FallbackChain("m", ("int8", "fp16")))

sch = EdgeScheduler(reg, DevicePool(LocalProbe().detect()), SimulatedRuntime())

# 严格模式：只尝试 int8；不兼容直接拒绝，不静默换变体
sch.infer("m", "int8", InferenceInput("r1", "payload"))

# 显式降级：首次超时后按链重试
sch.infer("m", "int8", InferenceInput("r2", "payload"),
          failure=FailureSpec(ErrorKind.INFERENCE_TIMEOUT, frozenset({1})),
          use_chain=True)
```

## 本地验证（无网络/无 GPU）

所有逻辑确定性运行：

```bash
uv run pytest -q          # 期望全部通过，耗时 < 5s
uv run pytest -s 2>&1 | grep -E "case=|成功|失败|拒绝"
```
