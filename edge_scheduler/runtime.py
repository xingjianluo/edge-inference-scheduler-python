"""模拟推理运行时：把算子不支持、超时、设备掉线归一化为可区分错误。

不执行真实推理：以纯 Python 模拟执行。副作用计数只在“真正开始执行”时发生，
且同一 (request_id, variant_id, device_id) 组合至多一次，供测试断言
“重试不产生重复副作用、已释放资源不被再用”。

结果中回带 request_id，调度器在返回前会核对，杜绝结果串号。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .devices import DeviceRuntimeState, DeviceState
from .errors import ErrorKind, SchedulerError
from .variants import ModelVariant


@dataclass(frozen=True)
class InferenceInput:
    request_id: str
    payload: str


@dataclass(frozen=True)
class InferenceResult:
    request_id: str
    model: str
    variant_id: str
    device_id: str
    output: str
    attempt: int
    side_effects: int


@dataclass(frozen=True)
class FailureSpec:
    """对某次执行注入的失败（测试用），按 attempt（从 1 开始）生效。"""

    kind: ErrorKind
    fail_attempts: frozenset[int] = frozenset({1})


class SimulatedRuntime:
    """在指定设备上执行变体的模拟运行时。"""

    def __init__(self) -> None:
        self._effects_lock = threading.Lock()
        # key -> 已产生的副作用次数（每个执行组合最多 1）
        self._effects: dict[tuple[str, str, str], int] = {}
        # request_id -> 副作用总数（跨设备/变体累计）
        self._effects_per_request: dict[str, int] = {}

    def run(
        self,
        variant: ModelVariant,
        device: DeviceRuntimeState,
        inp: InferenceInput,
        *,
        attempt: int,
        failure: FailureSpec | None = None,
        run_seconds: float = 0.0,
        timeout_seconds: float | None = None,
    ) -> InferenceResult:
        # 0) 入口先判定设备在线状态（模拟“设备中途不可用”）
        if device.state is DeviceState.OFFLINE:
            raise SchedulerError(
                ErrorKind.DEVICE_UNAVAILABLE,
                f"设备 {device.capability.device_id} 在执行前已离线",
                details={
                    "device_id": device.capability.device_id,
                    "request_id": inp.request_id,
                    "attempt": attempt,
                },
            )

        # 1) 算子支持是运行时硬性检查（即使匹配阶段已查过，执行时仍归一化）
        missing_ops = variant.required_ops - device.capability.ops
        if missing_ops:
            raise SchedulerError(
                ErrorKind.UNSUPPORTED_OP,
                (
                    f"设备 {device.capability.device_id} 运行时不支持算子: "
                    f"{sorted(missing_ops)}"
                ),
                details={
                    "device_id": device.capability.device_id,
                    "request_id": inp.request_id,
                    "missing_ops": sorted(missing_ops),
                    "attempt": attempt,
                },
            )

        # 2) 注入失败（超时/掉线/算子不支持），在产生副作用之前判定
        if failure is not None and attempt in failure.fail_attempts:
            raise self._injected_error(failure.kind, variant, device, inp, attempt)

        # 3) 超时判定：预算不足直接归一化为超时，不开始执行、不产生副作用
        if timeout_seconds is not None and run_seconds > timeout_seconds:
            raise SchedulerError(
                ErrorKind.INFERENCE_TIMEOUT,
                (
                    f"推理超时: 预计 {run_seconds:.3f}s 超过预算 "
                    f"{timeout_seconds:.3f}s"
                ),
                details={
                    "device_id": device.capability.device_id,
                    "request_id": inp.request_id,
                    "run_seconds": run_seconds,
                    "timeout_seconds": timeout_seconds,
                    "attempt": attempt,
                },
            )

        # 4) 真正开始执行：产生唯一副作用，然后模拟耗时
        key = (inp.request_id, variant.variant_id, device.capability.device_id)
        with self._effects_lock:
            if key not in self._effects:
                self._effects[key] = 1
                self._effects_per_request[inp.request_id] = (
                    self._effects_per_request.get(inp.request_id, 0) + 1
                )
            effects = self._effects_per_request[inp.request_id]

        if run_seconds > 0:
            time.sleep(run_seconds)

        # 5) 执行后再次确认设备未在运行期间掉线
        if device.state is DeviceState.OFFLINE:
            raise SchedulerError(
                ErrorKind.DEVICE_UNAVAILABLE,
                f"设备 {device.capability.device_id} 在执行期间离线",
                details={
                    "device_id": device.capability.device_id,
                    "request_id": inp.request_id,
                    "attempt": attempt,
                },
            )

        return InferenceResult(
            request_id=inp.request_id,
            model=variant.model,
            variant_id=variant.variant_id,
            device_id=device.capability.device_id,
            output=f"ok:{variant.key}:{inp.payload}",
            attempt=attempt,
            side_effects=effects,
        )

    def _injected_error(
        self,
        kind: ErrorKind,
        variant: ModelVariant,
        device: DeviceRuntimeState,
        inp: InferenceInput,
        attempt: int,
    ) -> SchedulerError:
        details = {
            "device_id": device.capability.device_id,
            "request_id": inp.request_id,
            "attempt": attempt,
            "injected": True,
        }
        if kind is ErrorKind.UNSUPPORTED_OP:
            return SchedulerError(
                kind, "注入失败：运行时不支持算子", details=details
            )
        if kind is ErrorKind.INFERENCE_TIMEOUT:
            return SchedulerError(kind, "注入失败：单次调用超时", details=details)
        if kind is ErrorKind.DEVICE_UNAVAILABLE:
            return SchedulerError(kind, "注入失败：设备中途不可用", details=details)
        return SchedulerError(kind, f"注入失败: {kind.value}", details=details)

    def side_effect_count(self, request_id: str) -> int:
        with self._effects_lock:
            return self._effects_per_request.get(request_id, 0)
