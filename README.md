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

## 等待队列治理

等待阶段同样可控、可解释（实现见 `edge_scheduler/queueing.py`）：

- **容量与背压**：每个设备的等待队列有上限（`Device(max_queue_size=...)`，
  默认 64）。到顶后新请求**立即**被拒绝为 `QUEUE_FULL`，不会无限排队；
  这与"队列还有位置、只是槽位/内存一时腾不出来"的 `RESOURCE_EXHAUSTED`
  （排队超时后拒绝）是两个可单独断言的错误码。
- **取消与放弃**：仍在等待、尚未占用资源的请求可通过
  `Scheduler.cancel(request_id)` 取消（或以 `queue_timeout_s` 超时自行放弃）。
  被取消的请求以 `REQUEST_CANCELLED` 结束：彻底移出等待序列，之后资源空出来
  也不会再分配给它、不会再产出它的结果；移除不影响同队列其他请求的相对先后。
  已在运行的请求不受取消影响。
- **优先级与防饿死**：`submit(..., priority=N)` 携带优先级（整数，越大越优先）。
  资源释放时有效优先级最高的等待者先拿到资源，同一优先级严格 FIFO。
  等待时长达到 `aging_after_s`（`Device(aging_after_s=...)`，默认 1.0s）的
  请求会被自动提升到最高优先段，段内仍按入队先后 FIFO——因此任何请求的
  最长排队时间有界，低优先级不会被后来的高优先级无限插队。
- **与硬性匹配/降级链的兼容**：排队治理只改变"谁先拿到资源"，不改变
  "能不能用这个设备/这个变体"。量化、算子、内存等硬性约束在入队前判定，
  优先级再高也不会被调度到不满足约束的设备，也不会被静默换成链外变体；
  降级顺序仍是候选链声明顺序，`QUEUE_FULL` 与 `RESOURCE_EXHAUSTED` 一样
  会沿候选链继续尝试下一台设备/下一个变体，最后归一化为可区分的错误码。

## 降级顺序与错误分类

候选链 = 请求变体 + 其 `fallback_variant_ids` 声明的**同模型**变体（顺序即降级顺序）。
对每个候选变体，按 `device_id` 排序遍历满足约束的设备，逐次尝试；
运行失败归一化后继续下一个候选。所有失败归一化为可区分的 `ErrorCode`：

| 错误码 | 含义 |
| --- | --- |
| `CAPABILITY_MISSING` | 没有任何设备满足链上任一变体的硬性约束 |
| `VARIANT_INCOMPATIBLE` | 变体未注册 / 重复注册 / 降级链跨模型 |
| `RESOURCE_EXHAUSTED` | 队列有位置但排队超时仍无并发槽位或内存 |
| `QUEUE_FULL` | 等待队列已满，背压立即拒绝 |
| `OPERATOR_UNSUPPORTED` | 运行时不支持所需算子 |
| `INFERENCE_TIMEOUT` | 单次调用超时 |
| `DEVICE_UNAVAILABLE` | 设备中途不可用 |
| `REQUEST_CANCELLED` | 等待中的请求被取消，不再占用资源也不产出结果 |
| `RUNTIME_FAILURE` | 其他运行时失败 |

每次尝试都记录在 `Scheduler.decision_log`（变体、设备、结果、原因），
重试在释放资源后进行，不会产生重复副作用或复用已释放资源。

## 本地验证

```bash
uv sync                 # 安装依赖（仅标准库 + pytest）
uv run pytest           # 运行五类场景测试，日志打印输入/判定依据/归一化结果
uv run python main.py   # 本地演示：探测 -> 匹配 -> 调度 -> 决策日志
```

测试覆盖五类场景：
- `tests/test_capability.py`：能力缺失、算子缺失、变体不兼容（原因可区分）；
- `tests/test_concurrency.py`：并发争抢下的排队/拒绝、并发上限不被突破、
  结果不串请求、设备状态最终一致；
- `tests/test_failure.py`：超时/设备不可用/未知运行时失败的归一化、
  按声明顺序降级、失败后资源正确释放；
- `tests/test_queueing.py`：队列满的背压拒绝（与资源不足可区分）、
  取消后资源不被占用也不产出结果、优先级准入顺序与同优先级 FIFO、
  低优先级经老化不被饿死、优先级不绕过硬性约束。
