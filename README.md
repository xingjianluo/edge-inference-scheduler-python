# edge-inference-scheduler-python

本地可复现的边缘推理调度器：用**本地设备探测**与**模拟运行时**承载多种模型变体，
在硬件能力、并发争抢与推理失败场景下给出一致且可解释的调度结论。
不依赖任何外部服务、真实加速卡或凭据。

## 能力探测与变体匹配规则

- `probe_local_devices()` 始终探测到一个本机 CPU 设备（核数、可用内存来自 `os`）；
  可通过环境变量 `EDGE_SIM_DEVICES` 以 JSON 声明额外的模拟加速设备，便于复现异构场景：
  ```bash
  EDGE_SIM_DEVICES='[{"device_id":"npu0","kind":"npu","supported_quantizations":["int8"],
                      "supported_operators":["conv","matmul"],"total_memory_mb":256,
                      "max_concurrency":1}]' uv run python main.py
  ```
- 每个 `ModelVariant` 声明硬性约束：量化精度 `quantization`、算子集
  `required_operators`、内存占用 `memory_mb`。
- 匹配是**硬性**的：量化不在设备支持列表、所需算子缺失、或内存超过设备总量，
  该变体在此设备上即不可调度，`Device.mismatch_reasons()` 给出全部原因。
  不满足约束的变体只会被拒绝（`CAPABILITY_MISSING`），绝不静默换用精度不符的变体；
  跨模型的降级链会被拒绝为 `VARIANT_INCOMPATIBLE`。

## 并发与资源策略

- 每个设备有 `max_concurrency` 个并发槽位与 `total_memory_mb` 内存上限，
  `try_acquire`/`release` 必须配对，重复占用或重复释放会被检测为错误。
- 资源不足时按确定策略处理：先在 `queue_timeout_s` 内排队等待，
  超时仍无资源则拒绝为 `RESOURCE_EXHAUSTED`。
- 推理结果携带 `request_id`，只返回给对应请求，并发下互不污染。

## 降级顺序与错误分类

候选链 = 请求变体 + 其 `fallback_variant_ids` 声明的**同模型**变体（顺序即降级顺序）。
对每个候选变体，按 `device_id` 排序遍历满足约束的设备，逐次尝试；
运行失败归一化后继续下一个候选。所有失败归一化为可区分的 `ErrorCode`：

| 错误码 | 含义 |
| --- | --- |
| `CAPABILITY_MISSING` | 没有任何设备满足链上任一变体的硬性约束 |
| `VARIANT_INCOMPATIBLE` | 变体未注册 / 重复注册 / 降级链跨模型 |
| `RESOURCE_EXHAUSTED` | 排队超时仍无并发槽位或内存 |
| `OPERATOR_UNSUPPORTED` | 运行时不支持所需算子 |
| `INFERENCE_TIMEOUT` | 单次调用超时 |
| `DEVICE_UNAVAILABLE` | 设备中途不可用 |
| `RUNTIME_FAILURE` | 其他运行时失败 |

每次尝试都记录在 `Scheduler.decision_log`（变体、设备、结果、原因），
重试在释放资源后进行，不会产生重复副作用或复用已释放资源。

## 本地验证

```bash
uv sync                 # 安装依赖（仅标准库 + pytest）
uv run pytest           # 运行四类场景测试，日志打印输入/判定依据/归一化结果
uv run python main.py   # 本地演示：探测 -> 匹配 -> 调度 -> 决策日志
```

测试覆盖四类场景：
- `tests/test_capability.py`：能力缺失、算子缺失、变体不兼容（原因可区分）；
- `tests/test_concurrency.py`：并发争抢下的排队/拒绝、并发上限不被突破、
  结果不串请求、设备状态最终一致；
- `tests/test_failure.py`：超时/设备不可用/未知运行时失败的归一化、
  按声明顺序降级、失败后资源正确释放。
