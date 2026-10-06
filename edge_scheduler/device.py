"""设备实例：并发槽位与内存占用的确定性、线程安全管理。

约束：
- 同一设备上 active 请求数不超过 max_concurrency；
- 已占用内存 + 新请求内存不超过 total_memory_mb；
- acquire/release 必须配对，重复释放会被检测为错误，
  保证不会出现重复占用或状态互相污染。

等待队列治理（每台设备一条 FIFO+优先级队列）：
- 容量上限 max_queue_depth：到顶后新请求立即以 QUEUE_FULL 拒绝（背压），
  与"已入队但等资源超时"的 RESOURCE_EXHAUSTED 可区分；
- 准入顺序：有效优先级（基础优先级 + 被插队次数）高者先得，
  相同有效优先级按入队序号 FIFO；
- 防饿死：每当一个后来者被准入，所有排在它前面的等待者 skips + 1，
  即有效优先级被抬高 1，因此低优先级请求最多被插队
  （对方优先级 - 自己优先级）次后必然轮到自己，等待有确定上界；
- 取消/放弃：等待中可通过 cancel_event 取消，或超时自动放弃；
  取消/放弃的等待者被彻底移出队列，不会再占用资源、不会再产出结果，
  且不影响队列中其他等待者的相对先后。
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass

from .capabilities import DeviceCapability
from .errors import ErrorCode, SchedulerError
from .variants import ModelVariant

# 等待中检查取消事件的轮询间隔（秒）
_CANCEL_POLL_S = 0.01


@dataclass
class _Waiter:
    """队列中的一个等待者。有效优先级 = priority + skips。"""

    request_id: str
    priority: int
    seq: int  # 入队序号，同有效优先级时保证 FIFO
    skips: int = 0  # 被后来者插队的次数（防饿死老化）

    @property
    def effective_priority(self) -> int:
        return self.priority + self.skips


class Device:
    """运行中的设备：能力 + 实时资源占用 + 等待队列。"""

    def __init__(self, capability: DeviceCapability, *, max_queue_depth: int = 1024) -> None:
        if max_queue_depth < 0:
            raise ValueError("max_queue_depth must be >= 0")
        self.capability = capability
        self._max_queue_depth = max_queue_depth
        # RLock：wait_acquire 在持有 cond 锁的情况下调用 try_acquire
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._used_memory_mb = 0
        self._active = 0
        self._holders: set[str] = set()  # request_id 集合，防止重复占用/重复释放
        self._waiters: list[_Waiter] = []  # 等待队列（无序存储，准入时按规则选头）
        self._seq_counter = itertools.count()  # 单调入队序号

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
        """原子地尝试占用一个并发槽位与内存；成功返回 True。

        队列非空时直接占用会插队，因此一律返回 False，
        调用方应走 wait_acquire 排队，由队列决定准入顺序。
        """
        with self._cond:
            if request_id in self._holders:
                raise SchedulerError(
                    ErrorCode.RUNTIME_FAILURE,
                    f"request {request_id} already holds resources on "
                    f"{self.capability.device_id}",
                )
            if self._waiters:
                return False
            if not self._can_fit(memory_mb):
                return False
            self._active += 1
            self._used_memory_mb += memory_mb
            self._holders.add(request_id)
            return True

    def wait_acquire(
        self,
        request_id: str,
        memory_mb: int,
        timeout_s: float,
        *,
        priority: int = 0,
        cancel_event: threading.Event | None = None,
    ) -> str:
        """排队等待资源，返回 "acquired" / "timeout" / "cancelled"。

        - 队列已满（达到 max_queue_depth）时立即抛 QUEUE_FULL（背压）；
        - 准入规则：有效优先级（priority + skips）最高者得，并列时先入队先得；
        - 等待中被取消（cancel_event 置位）或超时自动放弃时，
          等待者被彻底移出队列，返回后绝不会再占用资源。
        """
        deadline = time.monotonic() + timeout_s
        with self._cond:
            if request_id in self._holders:
                raise SchedulerError(
                    ErrorCode.RUNTIME_FAILURE,
                    f"request {request_id} already holds resources on "
                    f"{self.capability.device_id}",
                )
            if len(self._waiters) >= self._max_queue_depth:
                raise SchedulerError(
                    ErrorCode.QUEUE_FULL,
                    f"device {self.capability.device_id} wait queue full "
                    f"({self._max_queue_depth} pending)",
                    detail={
                        "device_id": self.capability.device_id,
                        "max_queue_depth": self._max_queue_depth,
                    },
                )
            waiter = _Waiter(request_id, priority, next(self._seq_counter))
            self._waiters.append(waiter)
            try:
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        return "cancelled"
                    if self._is_head(waiter) and self._can_fit(memory_mb):
                        self._waiters.remove(waiter)
                        self._record_skips(waiter)
                        # 此时必定能占用：是队头且资源足够
                        self._active += 1
                        self._used_memory_mb += memory_mb
                        self._holders.add(request_id)
                        return "acquired"
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return "timeout"
                    self._cond.wait(timeout=min(remaining, _CANCEL_POLL_S))
            finally:
                # 无论以何种方式离开，都必须彻底出队，且不影响他人相对顺序
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                self._cond.notify_all()

    def _is_head(self, waiter: _Waiter) -> bool:
        """该等待者是否是当前应准入的队头（有效优先级最高，并列取先入队）。"""
        best = min(
            self._waiters,
            key=lambda w: (-w.effective_priority, w.seq),
        )
        return best is waiter

    def _record_skips(self, selected: _Waiter) -> None:
        """selected 被准入后，所有比它先入队却仍在等的等待者 skips + 1。

        这保证低优先级请求被插队的次数有确定上界：
        对方优先级 - 自己优先级 次之后，有效优先级追平，
        再凭更小的入队序号必然先被准入。
        """
        for w in self._waiters:
            if w.seq < selected.seq:
                w.skips += 1

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
        """当前挂在等待队列中的请求数。"""
        with self._lock:
            return len(self._waiters)
