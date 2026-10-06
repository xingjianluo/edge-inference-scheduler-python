"""等待队列治理：容量背压、取消/放弃、优先级准入与防饿死。

规则（全部确定性，调用方需持有设备锁）：
1. 容量：队列中等待者数量达到 max_size 时，admit 立即抛出 QUEUE_FULL，
   与"队列有位置但资源一时腾不出来"的 RESOURCE_EXHAUSTED 明确区分。
2. 取消：cancel 仅做标记，remove 彻底移除；移除不影响其他等待者的
   相对先后。被取消的等待者永远不会再被判定为队首。
3. 准入顺序：有效优先级高者优先；有效优先级相同按入队序号 FIFO。
4. 防饿死：等待时长达到 aging_after_s 的等待者获得 AGING_BOOST 提升，
   进入最高优先段；最高优先段内部仍按入队序号 FIFO。因此任何等待者
   的最长排队时间有界：aging_after_s 加上先于它老化的等待者被服务的
   时间，之后它必然成为队首，不会被无限插队。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .errors import ErrorCode, SchedulerError

#: 老化提升量：等待超过 aging_after_s 的请求被提升到最高优先段，
#: 保证低优先级请求不会被无限插队（防饿死）。
AGING_BOOST = 1_000_000


@dataclass
class Waiter:
    """一个排队中的请求。"""

    request_id: str
    memory_mb: int
    priority: int
    seq: int
    enqueued_at: float
    cancelled: bool = False


class WaitQueue:
    """单设备等待队列：容量上限 + 优先级排序 + 老化防饿死 + 取消。

    本类只做排队簿记，不持有锁；所有方法须由 Device 在持有其
    条件变量锁时调用，保证与资源状态判断的原子性。
    """

    def __init__(self, max_size: int, aging_after_s: float,
                 clock=time.monotonic) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._max_size = max_size
        self._aging_after_s = aging_after_s
        self._clock = clock
        self._waiters: list[Waiter] = []
        self._next_seq = 0

    # ---- 入队 / 背压 ----

    def admit(self, request_id: str, memory_mb: int, priority: int) -> Waiter:
        """入队；队列已满时抛出 QUEUE_FULL（背压拒绝）。"""
        if len(self._waiters) >= self._max_size:
            raise SchedulerError(
                ErrorCode.QUEUE_FULL,
                f"wait queue full ({self._max_size} pending), "
                f"rejecting {request_id}",
                detail={"max_queue_size": self._max_size},
            )
        waiter = Waiter(
            request_id=request_id,
            memory_mb=memory_mb,
            priority=priority,
            seq=self._next_seq,
            enqueued_at=self._clock(),
        )
        self._next_seq += 1
        self._waiters.append(waiter)
        return waiter

    # ---- 取消 / 移除 ----

    def cancel(self, request_id: str) -> bool:
        """标记取消；返回是否找到仍在等待的请求。

        被取消的等待者保留在队列中直到其等待循环醒来自行 remove，
        但 is_turn 永远不会再选中它，因此它不会再占用任何资源。
        """
        for w in self._waiters:
            if w.request_id == request_id and not w.cancelled:
                w.cancelled = True
                return True
        return False

    def remove(self, request_id: str) -> None:
        """从队列中彻底移除；其余等待者的相对顺序保持不变。"""
        self._waiters = [w for w in self._waiters if w.request_id != request_id]

    # ---- 准入判定 ----

    def _effective_priority(self, waiter: Waiter, now: float) -> int:
        if now - waiter.enqueued_at >= self._aging_after_s:
            return waiter.priority + AGING_BOOST
        return waiter.priority

    def is_turn(self, request_id: str) -> bool:
        """该请求是否是当前应获准占用资源的队首。

        排序键 = (有效优先级降序, 入队序号升序)；已取消的等待者
        不参与排序，保证取消后资源绝不会再分配给它。
        """
        now = self._clock()
        candidates = [w for w in self._waiters if not w.cancelled]
        if not candidates:
            return False
        head = min(
            candidates,
            key=lambda w: (-self._effective_priority(w, now), w.seq),
        )
        return head.request_id == request_id

    # ---- 观测（测试与日志用） ----

    def __len__(self) -> int:
        return len(self._waiters)

    def pending_ids(self) -> list[str]:
        """按当前准入顺序返回未取消的等待者（确定性快照）。"""
        now = self._clock()
        candidates = [w for w in self._waiters if not w.cancelled]
        ordered = sorted(
            candidates,
            key=lambda w: (-self._effective_priority(w, now), w.seq),
        )
        return [w.request_id for w in ordered]
