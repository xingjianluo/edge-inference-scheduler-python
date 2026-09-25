"""设备实例：并发槽位与内存占用的确定性、线程安全管理。

约束：
- 同一设备上 active 请求数不超过 max_concurrency；
- 已占用内存 + 新请求内存不超过 total_memory_mb；
- acquire/release 必须配对，重复释放会被检测为错误，
  保证不会出现重复占用或状态互相污染。
"""

from __future__ import annotations

import threading

from .capabilities import DeviceCapability
from .errors import ErrorCode, SchedulerError
from .variants import ModelVariant


class Device:
    """运行中的设备：能力 + 实时资源占用。"""

    def __init__(self, capability: DeviceCapability) -> None:
        self.capability = capability
        # RLock：wait_acquire 在持有 cond 锁的情况下调用 try_acquire
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._used_memory_mb = 0
        self._active = 0
        self._holders: set[str] = set()  # request_id 集合，防止重复占用/重复释放

    # ---- 能力匹配（硬性约束） ----

    def mismatch_reasons(self, variant: ModelVariant) -> list[str]:
        """返回变体与该设备不匹配的全部原因；空列表表示可调度。"""
        cap = self.capability
        reasons: list[str] = []
        if variant.quantization not in cap.supported_quantizations:
            reasons.append(
                f"quantization '{variant.quantization}' not in "
                f"{sorted(cap.supported_quantizations)}"
            )
        missing = variant.required_operators - cap.supported_operators
        if missing:
            reasons.append(f"missing operators {sorted(missing)}")
        if variant.memory_mb > cap.total_memory_mb:
            reasons.append(
                f"memory {variant.memory_mb}MB exceeds total {cap.total_memory_mb}MB"
            )
        return reasons

    def fits(self, variant: ModelVariant) -> bool:
        return not self.mismatch_reasons(variant)

    # ---- 资源占用 ----

    def try_acquire(self, request_id: str, memory_mb: int) -> bool:
        """原子地尝试占用一个并发槽位与内存；成功返回 True。"""
        with self._cond:
            if request_id in self._holders:
                raise SchedulerError(
                    ErrorCode.RUNTIME_FAILURE,
                    f"request {request_id} already holds resources on "
                    f"{self.capability.device_id}",
                )
            if self._active >= self.capability.max_concurrency:
                return False
            if self._used_memory_mb + memory_mb > self.capability.total_memory_mb:
                return False
            self._active += 1
            self._used_memory_mb += memory_mb
            self._holders.add(request_id)
            return True

    def wait_acquire(self, request_id: str, memory_mb: int, timeout_s: float) -> bool:
        """在超时内等待资源可用（FIFO 由调度器的排队策略保证）。"""
        with self._cond:
            ok = self._cond.wait_for(
                lambda: self._can_fit(memory_mb), timeout=timeout_s
            )
            if not ok:
                return False
            return self.try_acquire(request_id, memory_mb)

    def _can_fit(self, memory_mb: int) -> bool:
        return (
            self._active < self.capability.max_concurrency
            and self._used_memory_mb + memory_mb <= self.capability.total_memory_mb
        )

    def release(self, request_id: str, memory_mb: int) -> None:
        """释放资源；未持有资源的释放是错误，防止已释放资源被再次使用。"""
        with self._cond:
            if request_id not in self._holders:
                raise SchedulerError(
                    ErrorCode.RUNTIME_FAILURE,
                    f"request {request_id} does not hold resources on "
                    f"{self.capability.device_id}",
                )
            self._holders.discard(request_id)
            self._active -= 1
            self._used_memory_mb -= memory_mb
            self._cond.notify_all()

    # ---- 状态观测（测试与日志用） ----

    @property
    def used_memory_mb(self) -> int:
        with self._lock:
            return self._used_memory_mb

    @property
    def active_count(self) -> int:
        with self._lock:
            return self._active
