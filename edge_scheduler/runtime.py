"""模拟推理运行时：本地执行，支持确定性故障注入。

故障模型（全部归一化为 SchedulerError）：
- 设备算子集不含变体所需算子      -> OPERATOR_UNSUPPORTED
- 注入 "timeout" 或时延超过超时    -> INFERENCE_TIMEOUT
- 注入 "device_unavailable" 或
  设备被标记为下线                 -> DEVICE_UNAVAILABLE
故障通过 fault_plan 按 (device_id, variant_id) 注入，每次调用弹出
一个故障，保证测试可复现且不产生真实副作用。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .capabilities import DeviceCapability
from .errors import ErrorCode, SchedulerError
from .variants import ModelVariant


@dataclass(frozen=True)
class InferenceResult:
    request_id: str
    variant_id: str
    device_id: str
    output: str


class SimulatedRuntime:
    """本地模拟运行时。

    latency_s: 每次成功调用的模拟时延（秒），默认 0 保证测试快速。
    fault_plan: {(device_id, variant_id): [fault, ...]}，按调用顺序消费。
    """

    def __init__(self, latency_s: float = 0.0,
                 fault_plan: dict[tuple[str, str], list[str]] | None = None) -> None:
        self._latency_s = latency_s
        self._fault_plan: dict[tuple[str, str], list[str]] = {
            k: list(v) for k, v in (fault_plan or {}).items()
        }
        self._down_devices: set[str] = set()
        self.call_log: list[tuple[str, str, str]] = field(default_factory=list)  # type: ignore[assignment]
        self.call_log = []

    # ---- 测试/仿真控制接口 ----

    def mark_device_down(self, device_id: str) -> None:
        self._down_devices.add(device_id)

    def mark_device_up(self, device_id: str) -> None:
        self._down_devices.discard(device_id)

    # ---- 推理 ----

    def infer(
        self,
        device: DeviceCapability,
        variant: ModelVariant,
        request_id: str,
        payload: str,
        timeout_s: float,
    ) -> InferenceResult:
        self.call_log.append((request_id, device.device_id, variant.variant_id))

        missing = variant.required_operators - device.supported_operators
        if missing:
            raise SchedulerError(
                ErrorCode.OPERATOR_UNSUPPORTED,
                f"device {device.device_id} lacks operators {sorted(missing)} "
                f"for variant {variant.variant_id}",
                detail={"missing_operators": sorted(missing)},
            )

        if device.device_id in self._down_devices:
            raise SchedulerError(
                ErrorCode.DEVICE_UNAVAILABLE,
                f"device {device.device_id} is unavailable",
            )

        faults = self._fault_plan.get((device.device_id, variant.variant_id))
        if faults:
            fault = faults.pop(0)
            if fault == "timeout":
                raise SchedulerError(
                    ErrorCode.INFERENCE_TIMEOUT,
                    f"inference timed out on {device.device_id}",
                )
            if fault == "device_unavailable":
                raise SchedulerError(
                    ErrorCode.DEVICE_UNAVAILABLE,
                    f"device {device.device_id} failed mid-inference",
                )
            raise SchedulerError(ErrorCode.RUNTIME_FAILURE, f"injected fault: {fault}")

        if self._latency_s > timeout_s:
            raise SchedulerError(
                ErrorCode.INFERENCE_TIMEOUT,
                f"latency {self._latency_s}s exceeded timeout {timeout_s}s",
            )
        if self._latency_s > 0:
            time.sleep(self._latency_s)

        # 输出携带 request_id，保证结果只会返回给对应请求
        return InferenceResult(
            request_id=request_id,
            variant_id=variant.variant_id,
            device_id=device.device_id,
            output=f"output[{request_id}]:{payload}:{variant.variant_id}@{device.device_id}",
        )
