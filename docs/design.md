# 边缘推理调度器设计说明

本项目实现一套**本地可复现**的边缘推理调度能力：用本地设备探测 + 模拟运行时承载
多种模型变体，使「能力匹配 / 并发争抢 / 推理失败」三类决策确定且可解释。

- 不依赖外部服务、真实加速卡或任何凭据；
- 设备探测仅使用标准库（`platform` / `os.sysconf` / `multiprocessing`），
  模拟设备的能力画像为代码内常量，跨机器结果确定；
- 推理为纯 Python 模拟，不执行真实算子。

## 1. 模块结构

| 模块 | 职责 |
| --- | --- |
| `edge_scheduler/devices.py` | 本地探测 `LocalProbe`、设备能力 `DeviceCapability`、设备池 `DevicePool`（内存准入 / 释放 / 上下线） |
| `edge_scheduler/variants.py` | 模型变体 `ModelVariant`、后备链 `FallbackChain`、硬性匹配 `check_compatibility`、注册表 |
| `edge_scheduler/runtime.py` | 模拟运行时 `SimulatedRuntime`、输入/结果、失败注入 `FailureSpec` |
| `edge_scheduler/scheduler.py` | 调度器：FIFO 准入排队、确定顺序降级、错误聚合、inflight 去重 |
| `edge_scheduler/errors.py` | 归一化错误类型 `ErrorKind` 与 `SchedulerError` |
| `edge_scheduler/logging_utils.py` | 统一打印「输入 / 判定依据 / 归一化结果」 |

## 2. 能力探测规则

`LocalProbe.detect()` 始终按固定顺序返回三台设备：

1. `local-cpu`（真实本地设备）：
   - 精度：仅 `fp32`；算子：基础集 `{matmul, conv2d, relu, softmax, layernorm}`；
   - 内存：读取 `SC_PHYS_PAGES * SC_PAGE_SIZE`，受限环境回退到 1024MB。
2. `sim-gpu`（模拟）：精度 `{fp32, fp16}`，额外算子 `flash_attention`，2048MB。
3. `sim-npu`（模拟）：精度 `{fp32, int8}`，额外算子 `{rope, flash_attention,
   quantized_matmul}`，**512MB（低显存，用于复现内存争抢）**。

探测结果两次调用完全一致（确定性），不读取任何外部状态。

## 3. 变体匹配规则（硬性约束）

一个变体包含：

- `required_dtypes`：设备必须支持其中**全部**精度（通常只声明一种，如 `int8`）；
- `required_ops`：设备算子集必须是其**超集**；
- `memory_mb`：加载 + 推理所需内存，在调度准入阶段判定。

`check_compatibility(variant, device)` 的结果：

- 全部满足 → 返回 `None`（兼容）；
- 否则返回 `MismatchDetail`，把缺失项拆成 **`missing_dtypes`** 与 **`missing_ops`**
  两个互不混淆的字段，拒绝原因可区分，绝不静默换用变体。

匹配示例（注册表见测试 fixture）：

| 变体 | 要求 | 唯一可用设备 |
| --- | --- | --- |
| `int8` | 精度 `int8` + 算子 `rope` | `sim-npu` |
| `fp16` | 精度 `fp16` + 算子 `flash_attention` | `sim-gpu` |
| `fp32` | 精度 `fp32` + 基础算子 | 三台均可 |

**严格模式（默认）**：`infer(model, variant_id, ...)` 只尝试调用方指定的变体。
该变体没有任何兼容设备时，直接以 `fallback_exhausted` 拒绝，
`details.attempts` 里逐设备保留 `capability_missing` 与具体缺失项——
即使该模型注册了其它变体，也**不会偷偷换成精度/算子要求不同的变体**。

**显式降级模式**：只有传入 `use_chain=True` 时，才按注册的 `FallbackChain`
顺序追加候选变体（首选变体恒在最前）。

## 4. 并发与资源约束策略

- 每个设备一把 `RLock`，准入校验「在线 → 未重复占用 → 空闲内存足够」与
  占用登记在同一临界区完成，杜绝并发下重复占用。
- 每个设备一个**显式 FIFO 准入闸门**（`threading.Condition` + `deque`）：
  仅队首请求有资格尝试准入；队首因内存不足无法进入时，在条件变量上等待
  占用释放，释放后按「世代(epoch)」谓词唤醒，避免通知早于等待而丢失唤醒，
  也不会忙等。顺序由入队顺序决定，结果确定。
- **锁序约定**：准入检查 `pool.try_admit` 不在闸门临界区内调用；闸门等待者
  仅在条件变量锁内读取身份/世代。`_acquire` 与 `_release` 都遵循
  “先设备池锁、后闸门锁”的顺序，避免锁序反转导致死锁。
- 等待超过 `admission_timeout` → 归一化为 `insufficient_memory`（含
  `queued=True` 等依据）。
- 入队时队列长度已达 `queue_capacity` → 立即归一化为 `queue_full`，
  这是确定性的立即拒绝，**不**再尝试后续设备/变体，与「内存不足」可区分。
- 占用以 `request_id` 为凭证：重复准入、重复/越权释放均抛
  `duplicate_request`，保证「已释放资源不被再次使用」。
- 同一 `request_id` 的 `infer` 重入会被 inflight 集合拒绝（防结果串号）。
- 返回结果在返回前核对 `result.request_id`，归属不符直接报错。

## 5. 失败归一化与降级顺序

### 5.1 错误分类（`ErrorKind`）

| kind | 触发条件 |
| --- | --- |
| `capability_missing` | 设备缺精度/算子，无法承载变体 |
| `variant_incompatible` | 引用了未注册/非法的变体 |
| `insufficient_memory` | 当前空闲不足，或排队等待超时 |
| `queue_full` | 设备等待队列达到容量，立即拒绝 |
| `unsupported_op` | 运行时确认不支持所需算子 |
| `inference_timeout` | 单次调用预算不足/超时（超时在产生副作用**之前**判定） |
| `device_unavailable` | 设备执行前/执行期间离线 |
| `fallback_exhausted` | 所有候选路径失败，`details.attempts` 聚合每步原因 |
| `duplicate_request` | 重复准入 / 重复释放 / 请求并发重入 |

### 5.2 候选与重试顺序（确定）

```
for 变体 in (首选变体 + [显式 use_chain 时的后备链其余变体]):     # 链顺序
    for 设备 in 该变体的兼容设备:                                 # 探测顺序
        准入(FIFO) -> runtime.run
```

- 准入成功后执行；执行失败归一化为 `unsupported_op / inference_timeout /
  device_unavailable` 时，在 `finally` 中**先释放占用、唤醒下一排队者**，
  再进入下一条候选路径。
- `device_unavailable` 会把设备标记离线，后续候选自动跳过它。
- 失败尝试**不产生副作用**：模拟运行时在通过全部前置校验、真正开始执行时才
  记录一次副作用，且同一 `(request, variant, device)` 组合至多一次。
- 任一路径成功即返回；全部失败聚合为 `fallback_exhausted`，保留每一步的
  `variant_id / device_id / kind / reasons`，可解释、可复现。

## 6. 可解释性

调度器接受 `logger` 回调，测试中输出三类信息：

1. 输入：`request_id`、`payload`、降级模式、候选顺序；
2. 判定依据：每台设备缺失的精度/算子、排队排位/所需内存、每次失败的归一化类型；
3. 归一化结果：成功落点（变体@设备、attempt、副作用数）或聚合拒绝原因。

用 `uv run pytest -s` 可在日志中查看完整判定链路。

## 7. 本地验证

见 `README.md` 的「本地验证」一节。
