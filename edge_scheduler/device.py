"""设备实例：并发槽位、内存占用与等待队列的确定性、线程安全管理。

约束：
- 同一设备上 active 请求数不超过 max_concurrency；
- 已占用内存 + 新请求内存不超过 total_memory_mb；
- acquire/release 必须配对，重复释放会被检测为错误，
  保证不会出现重复占用或状态互相污染。

排队治理（WaitQueue，规则详见 queueing.py）：
- 等待队列有容量上限，满员时 acquire 立即抛出 QUEUE_FULL（背压），
  与"排队超时仍无资源"的 RESOURCE_EXHAUSTED（返回 False）明确区分；
- 等待中的请求可被 cancel_waiter 取消：彻底移出等待序列，
  之后资源空出来也不会再分配给它；
- 准入按有效优先级（优先级 + 老化提升）排序，同优先级 FIFO，
  低优先级等待有界，不会被无限插队。
"""

from __future__ import annotations

import threading
import time

from .capabilities import DeviceCapability
from .errors import ErrorCode, SchedulerError
from .queueing import WaitQueue
from .variants import ModelVariant

#: 默认每设备等待队列容量与老化时间
DEFAULT_MAX_QUEUE_SIZE = 64
DEFAULT_AGING_AFTER_S = 1.0


class Device:
    """运行中的设备：能力 + 实时资源占用 + 等待队列。"""

    def __init__(
        self,
        capability: DeviceCapability,
        *,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
        aging_after_s: float = DEFAULT_AGING_AFTER_S,
        clock=time.monotonic,
    ) -> None:
        self.capability = capability
        # RLock：acquire 在持有 cond 锁的情况下调用 try_acquire
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._clock = clock
        self._used_memory_mb = 0
        self._active = 0
        self._holders: set[str] = set()  # request_id 集合，防止重复占用/重复释放
        self._queue = WaitQueue(max_queue_size, aging_after_s, clock=clock)

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

    def acquire(
        self,
        request_id: str,
        memory_mb: int,
        timeout_s: float,
        *,
        priority: int = 0,
    ) -> bool:
        """占用资源，必要时按队列规则排队等待。

        返回 True 表示已占用；返回 False 表示排队超时仍无资源
        （调用方归一化为 RESOURCE_EXHAUSTED）。
        队列已满抛出 QUEUE_FULL；等待中被取消抛出 REQUEST_CANCELLED。
        无论超时还是取消，请求都会被彻底移出等待队列。
        """
        with self._cond:
            # 快路径：没有等待者且资源够，直接占用，不进队列
            if len(self._queue) == 0 and self._can_fit(memory_mb):
                return self.try_acquire(request_id, memory_mb)

            waiter = self._queue.admit(request_id, memory_mb, priority)
            try:
                deadline = self._clock() + timeout_s
                while True:
                    if waiter.cancelled:
                        raise SchedulerError(
                            ErrorCode.REQUEST_CANCELLED,
                            f"request {request_id} cancelled while waiting on "
                            f"{self.capability.device_id}",
                        )
                    remaining = deadline - self._clock()
                    if remaining <= 0:
                        return False
                    if self._queue.is_turn(request_id) and self._can_fit(memory_mb):
                        return self.try_acquire(request_id, memory_mb)
                    self._cond.wait(remaining)
            finally:
                # 超时/取消/成功都彻底出队，且不打乱其他等待者的先后
                self._queue.remove(request_id)

    def cancel_waiter(self, request_id: str) -> bool:
        """取消一个仍在等待的请求；返回是否找到。已占用资源的请求不受影响。"""
        with self._cond:
            found = self._queue.cancel(request_id)
            if found:
                self._cond.notify_all()
            return found

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

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._queue)

    def pending_ids(self) -> list[str]:
        """按当前准入顺序返回等待中的请求（确定性快照）。"""
        with self._lock:
            return self._queue.pending_ids()
