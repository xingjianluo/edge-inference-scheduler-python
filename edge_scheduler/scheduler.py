"""调度器：能力匹配、确定性排队/拒绝、失败重试与降级。

调度规则（全部确定性）：
1. 候选链 = [请求变体] + 其 fallback_variant_ids 声明的同模型变体，
   顺序即降级顺序；不会尝试链外变体，绝不静默换用。
2. 对每个候选变体，按 device_id 排序遍历满足硬性约束的设备：
   先尝试直接占用，资源不足则按队列规则排队（容量背压、优先级、
   防饿死，见 queueing.py），超时仍失败则记录 RESOURCE_EXHAUSTED
   并继续下一个候选；队列已满则记录 QUEUE_FULL 并继续下一个候选。
3. 等待中的请求可通过 cancel() 取消：取消是终态，不再尝试其他
   候选，直接以 REQUEST_CANCELLED 结束，且不会再占用任何资源。
4. 推理失败（算子不支持/超时/设备不可用）时释放资源后按同一顺序
   继续尝试；每次尝试都记录到 decision_log，保证结论可解释。
5. 所有候选都失败时抛出归一化错误：
   - 没有任何设备满足硬性约束      -> CAPABILITY_MISSING
   - 否则                          -> 最后一次失败（RESOURCE_EXHAUSTED /
                                      QUEUE_FULL / 运行时错误，可区分）
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .device import Device
from .errors import ErrorCode, SchedulerError
from .runtime import InferenceResult, SimulatedRuntime
from .variants import ModelVariant, VariantRegistry


@dataclass(frozen=True)
class Attempt:
    request_id: str
    variant_id: str
    device_id: str
    outcome: str  # "success" / ErrorCode.value
    reason: str


@dataclass
class Decision:
    request_id: str
    attempts: list[Attempt] = field(default_factory=list)
    result: InferenceResult | None = None
    error_code: str | None = None


class Scheduler:
    def __init__(
        self,
        devices: list[Device],
        registry: VariantRegistry,
        runtime: SimulatedRuntime,
        *,
        queue_timeout_s: float = 0.5,
        call_timeout_s: float = 5.0,
    ) -> None:
        # 按 device_id 排序，保证候选遍历顺序确定
        self._devices = sorted(devices, key=lambda d: d.capability.device_id)
        self._registry = registry
        self._runtime = runtime
        self._queue_timeout_s = queue_timeout_s
        self._call_timeout_s = call_timeout_s
        self._log_lock = threading.Lock()
        self.decision_log: list[Decision] = []

    # ---- 候选链 ----

    def _candidate_chain(self, variant: ModelVariant) -> list[ModelVariant]:
        chain = [variant]
        for fid in variant.fallback_variant_ids:
            fb = self._registry.get(fid)
            if fb.model_name != variant.model_name:
                # 拒绝跨模型降级：这属于变体不兼容，而不是静默换用
                raise SchedulerError(
                    ErrorCode.VARIANT_INCOMPATIBLE,
                    f"fallback {fid} belongs to model {fb.model_name}, "
                    f"expected {variant.model_name}",
                )
            chain.append(fb)
        return chain

    # ---- 取消 ----

    def cancel(self, request_id: str) -> bool:
        """取消一个仍在排队等待的请求。

        返回是否找到等待中的请求。被取消的请求会以 REQUEST_CANCELLED
        结束，不再占用资源、不再产出结果；已在运行的请求不受影响。
        """
        found = False
        for device in self._devices:
            if device.cancel_waiter(request_id):
                found = True
        return found

    # ---- 主入口 ----

    def submit(
        self,
        request_id: str,
        variant_id: str,
        payload: str,
        *,
        priority: int = 0,
    ) -> InferenceResult:
        variant = self._registry.get(variant_id)
        chain = self._candidate_chain(variant)
        decision = Decision(request_id=request_id)

        saw_capability_match = False
        last_error: SchedulerError | None = None

        for candidate in chain:
            eligible = [d for d in self._devices if d.fits(candidate)]
            if not eligible:
                reasons = {
                    d.capability.device_id: d.mismatch_reasons(candidate)
                    for d in self._devices
                }
                decision.attempts.append(Attempt(
                    request_id, candidate.variant_id, "*",
                    ErrorCode.CAPABILITY_MISSING.value,
                    f"no eligible device: {reasons}",
                ))
                continue
            saw_capability_match = True

            for device in eligible:
                dev_id = device.capability.device_id
                try:
                    acquired = device.acquire(
                        request_id, candidate.memory_mb,
                        self._queue_timeout_s, priority=priority,
                    )
                except SchedulerError as e:
                    if e.code is ErrorCode.REQUEST_CANCELLED:
                        # 取消是终态：不再尝试其他候选，直接结束
                        decision.attempts.append(Attempt(
                            request_id, candidate.variant_id, dev_id,
                            e.code.value, str(e),
                        ))
                        decision.error_code = e.code.value
                        with self._log_lock:
                            self.decision_log.append(decision)
                        raise
                    # QUEUE_FULL：背压拒绝，与资源不足区分记录
                    decision.attempts.append(Attempt(
                        request_id, candidate.variant_id, dev_id,
                        e.code.value, str(e),
                    ))
                    last_error = e
                    continue
                if not acquired:
                    decision.attempts.append(Attempt(
                        request_id, candidate.variant_id, dev_id,
                        ErrorCode.RESOURCE_EXHAUSTED.value,
                        "no free slot/memory within queue timeout",
                    ))
                    last_error = SchedulerError(
                        ErrorCode.RESOURCE_EXHAUSTED,
                        f"device {dev_id} exhausted for {candidate.variant_id}",
                    )
                    continue
                try:
                    result = self._runtime.infer(
                        device.capability, candidate, request_id,
                        payload, self._call_timeout_s,
                    )
                except SchedulerError as e:
                    decision.attempts.append(Attempt(
                        request_id, candidate.variant_id, dev_id,
                        e.code.value, str(e),
                    ))
                    last_error = e
                else:
                    decision.attempts.append(Attempt(
                        request_id, candidate.variant_id, dev_id, "success", "",
                    ))
                    decision.result = result
                    with self._log_lock:
                        self.decision_log.append(decision)
                    return result
                finally:
                    # 无论成功失败都释放，保证资源不被泄漏或重复使用
                    device.release(request_id, candidate.memory_mb)

        # 全部候选失败：归一化最终错误
        if not saw_capability_match:
            final = SchedulerError(
                ErrorCode.CAPABILITY_MISSING,
                f"no device satisfies hard constraints for any variant in "
                f"{[c.variant_id for c in chain]}",
            )
        else:
            assert last_error is not None
            final = last_error
        decision.error_code = final.code.value
        with self._log_lock:
            self.decision_log.append(decision)
        raise final
