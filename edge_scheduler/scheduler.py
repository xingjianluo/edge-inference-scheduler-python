"""调度器：能力匹配 + 内存准入排队 + 确定顺序降级。

确定性策略：
- 候选顺序：后备链中的变体顺序 × 每个变体的兼容设备顺序（探测顺序）；
- 内存不足：在每设备的显式 FIFO 队列中等待 admission_timeout；
  队列长度超 queue_capacity 立即以 QUEUE_FULL 拒绝；
- 执行失败（算子不支持/超时/设备掉线）归一化后释放当前占用，
  按候选顺序继续下一条路径；全部失败聚合为 FALLBACK_EXHAUSTED，
  details 保留每一步可区分的原因，全程不静默换用。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from .devices import DevicePool, DeviceState
from .errors import ErrorKind, SchedulerError
from .runtime import InferenceInput, InferenceResult, SimulatedRuntime
from .variants import VariantRegistry

# 触发“换下一条后备路径”的归一化错误类型
_RETRYABLE_KINDS = frozenset(
    {
        ErrorKind.UNSUPPORTED_OP,
        ErrorKind.INFERENCE_TIMEOUT,
        ErrorKind.DEVICE_UNAVAILABLE,
    }
)


@dataclass
class SchedulerConfig:
    queue_capacity: int = 8          # 每个设备等待队列容量（不含正在执行的）
    admission_timeout: float = 1.0   # 排队等待准入的最长时间（秒）


@dataclass
class _Gate:
    """单设备 FIFO 准入闸门：严格按入队顺序获得准入资格。

    epoch 在每次释放后递增，作为条件变量的“世代”谓词，
    避免释放通知发生在等待者进入 wait 之前时丢失唤醒。
    """

    device_id: str
    cond: threading.Condition = field(
        default_factory=lambda: threading.Condition()
    )
    waiters: deque[threading.Event] = field(default_factory=deque)
    epoch: int = 0


class EdgeScheduler:
    def __init__(
        self,
        registry: VariantRegistry,
        pool: DevicePool,
        runtime: SimulatedRuntime,
        config: SchedulerConfig | None = None,
        logger: Callable[[str], None] | None = None,
    ):
        self.registry = registry
        self.pool = pool
        self.runtime = runtime
        self.config = config or SchedulerConfig()
        self._log = logger or (lambda _msg: None)
        self._gates: dict[str, _Gate] = {}
        self._gates_lock = threading.Lock()
        # 正在处理的请求：防止同一请求并发重入
        self._inflight_lock = threading.Lock()
        self._inflight: set[str] = set()

    # ------------------------------------------------------------------ gates
    def _gate(self, device_id: str) -> _Gate:
        with self._gates_lock:
            gate = self._gates.get(device_id)
            if gate is None:
                gate = _Gate(device_id=device_id)
                self._gates[device_id] = gate
            return gate

    def queue_depth(self, device_id: str) -> int:
        """返回某设备准入队列中正在等待的请求数（不含正在执行的）。"""
        gate = self._gate(device_id)
        with gate.cond:
            return len(gate.waiters)

    def _acquire(self, device_id: str, request_id: str, need_mb: int) -> None:
        """在设备 FIFO 队列中排队并准入；失败归一化。

        关键：pool.try_admit 绝不在 gate.cond 临界区内调用，
        与 _release（先 pool 后 gate）保持一致的锁序，杜绝死锁。
        """
        gate = self._gate(device_id)
        deadline = time.monotonic() + self.config.admission_timeout
        with gate.cond:
            if len(gate.waiters) >= self.config.queue_capacity:
                raise SchedulerError(
                    ErrorKind.QUEUE_FULL,
                    (
                        f"设备 {device_id} 等待队列已满 "
                        f"({len(gate.waiters)}/{self.config.queue_capacity})"
                    ),
                    details={
                        "device_id": device_id,
                        "request_id": request_id,
                        "queue_len": len(gate.waiters),
                    },
                )
            gate.waiters.append(request_id)
            self._log(
                f"request={request_id} 进入设备 {device_id} 准入队列, "
                f"排位={len(gate.waiters)}, 需要={need_mb}MB"
            )

        try:
            while True:
                # 1) 仅队首；锁内只读取身份与世代，不做准入
                with gate.cond:
                    is_head = bool(gate.waiters) and gate.waiters[0] == request_id
                    if not is_head:
                        self._wait_turn(gate, request_id, deadline, device_id, need_mb)
                        continue
                    epoch = gate.epoch

                # 2) 锁外准入（不与 pool 锁嵌套）
                try:
                    self.pool.try_admit(device_id, request_id, need_mb)
                except SchedulerError as exc:
                    if exc.kind is not ErrorKind.INSUFFICIENT_MEMORY:
                        # 离线/重复占用等：直接冒泡
                        raise
                    # 内存不足：等待“释放世代”推进后重试
                    with gate.cond:
                        self._wait_turn(gate, request_id, deadline, device_id, need_mb, epoch)
                    continue

                # 3) 准入成功，出队并唤醒后续等待者重新判定
                with gate.cond:
                    if gate.waiters and gate.waiters[0] == request_id:
                        gate.waiters.popleft()
                    gate.cond.notify_all()
                return
        except BaseException:
            with gate.cond:
                try:
                    gate.waiters.remove(request_id)
                except ValueError:
                    pass
                gate.cond.notify_all()
            raise

    def _wait_turn(
        self,
        gate: _Gate,
        request_id: str,
        deadline: float,
        device_id: str,
        need_mb: int,
        seen_epoch: int | None = None,
    ) -> None:
        """在 gate.cond 上等待：轮到自己或释放世代推进；超时归一化。"""
        def _predicate() -> bool:
            head = bool(gate.waiters) and gate.waiters[0] == request_id
            advanced = seen_epoch is not None and gate.epoch > seen_epoch
            return head or advanced

        remaining = deadline - time.monotonic()
        if remaining <= 0 or not gate.cond.wait_for(
            _predicate, timeout=max(remaining, 0)
        ):
            raise SchedulerError(
                ErrorKind.INSUFFICIENT_MEMORY,
                f"设备 {device_id} 排队等待内存超时: 需要 {need_mb}MB",
                details={
                    "device_id": device_id,
                    "request_id": request_id,
                    "need_mb": need_mb,
                    "queued": True,
                    "timeout_s": self.config.admission_timeout,
                },
            )

    def _release(self, device_id: str, request_id: str, used_mb: int) -> None:
        self.pool.release(device_id, request_id, used_mb)
        gate = self._gate(device_id)
        with gate.cond:
            gate.epoch += 1  # 世代推进：唤醒队首重新尝试准入
            gate.cond.notify_all()

    # ------------------------------------------------------------------ main
    def infer(
        self,
        model: str,
        preferred_variant_id: str,
        inp: InferenceInput,
        *,
        failure: object | None = None,
        timeout_seconds: float | None = None,
        run_seconds: float = 0.0,
        use_chain: bool = False,
    ) -> InferenceResult:
        # 防止同一 request_id 并发重入（结果串号/重复副作用的第一道闸）
        with self._inflight_lock:
            if inp.request_id in self._inflight:
                raise SchedulerError(
                    ErrorKind.DUPLICATE_REQUEST,
                    f"请求 {inp.request_id} 正在处理中，禁止重复调度",
                    details={"request_id": inp.request_id},
                )
            self._inflight.add(inp.request_id)

        attempts: list[dict] = []
        try:
            # use_chain 必须显式开启；默认严格模式，绝不静默换用其它变体
            candidates = self.registry.chain_for(
                model, preferred_variant_id, use_chain=use_chain
            )
            self._log(
                f"request={inp.request_id} 输入(payload={inp.payload!r}) "
                f"降级模式={'显式后备链' if use_chain else '严格(不换变体)'} "
                f"候选顺序={[v.key for v in candidates]}"
            )
            devices = [s.capability for s in self.pool.list_devices()]

            for variant in candidates:
                compatible, mismatches = self.registry.find_compatible(
                    variant, devices
                )
                if not compatible:
                    for m in mismatches:
                        rec = {
                            "variant_id": variant.variant_id,
                            "device_id": m.device_id,
                            "kind": ErrorKind.CAPABILITY_MISSING.value,
                            "reasons": list(m.reasons),
                        }
                        attempts.append(rec)
                        self._log(
                            f"request={inp.request_id} 拒绝 {m.device_id} 承载 "
                            f"{variant.key}: {m.reasons}"
                        )
                    continue

                for cap in compatible:
                    device = self.pool.get(cap.device_id)
                    if device.state is DeviceState.OFFLINE:
                        attempts.append(
                            {
                                "variant_id": variant.variant_id,
                                "device_id": cap.device_id,
                                "kind": ErrorKind.DEVICE_UNAVAILABLE.value,
                                "reasons": ["设备已离线，跳过"],
                            }
                        )
                        continue

                    attempt_no = len(attempts) + 1
                    try:
                        self._acquire(cap.device_id, inp.request_id, variant.memory_mb)
                    except SchedulerError as exc:
                        attempts.append(
                            {
                                "variant_id": variant.variant_id,
                                "device_id": cap.device_id,
                                "kind": exc.kind.value,
                                "reasons": [exc.reason],
                            }
                        )
                        self._log(
                            f"request={inp.request_id} 准入失败 "
                            f"{variant.key}@{cap.device_id}: "
                            f"{exc.kind.value} {exc.reason}"
                        )
                        if exc.kind is ErrorKind.QUEUE_FULL:
                            # 确定性策略：队列满是确定性立即拒绝，
                            # 不再尝试后续设备/变体（原因必须可区分）
                            raise
                        if exc.kind is ErrorKind.DEVICE_UNAVAILABLE:
                            # 设备离线（含排队等待期间被其它请求标记）：
                            # 根因必须以 device_unavailable 冒泡，可区分
                            raise
                        continue

                    try:
                        result = self.runtime.run(
                            variant,
                            self.pool.get(cap.device_id),
                            inp,
                            attempt=attempt_no,
                            failure=failure,
                            run_seconds=run_seconds,
                            timeout_seconds=timeout_seconds,
                        )
                    except SchedulerError as exc:
                        attempts.append(
                            {
                                "variant_id": variant.variant_id,
                                "device_id": cap.device_id,
                                "kind": exc.kind.value,
                                "reasons": [exc.reason],
                            }
                        )
                        self._log(
                            f"request={inp.request_id} 执行失败 "
                            f"{variant.key}@{cap.device_id} "
                            f"attempt={attempt_no}: {exc.kind.value} {exc.reason}"
                        )
                        if exc.kind is ErrorKind.DEVICE_UNAVAILABLE:
                            # 设备中途不可用：归一化标记，后续候选自动跳过
                            self.pool.mark_offline(cap.device_id)
                        if exc.kind not in _RETRYABLE_KINDS:
                            raise
                        continue
                    else:
                        # 结果归属核对：返回值必须属于当前请求
                        if result.request_id != inp.request_id:
                            raise SchedulerError(
                                ErrorKind.FALLBACK_EXHAUSTED,
                                "结果归属校验失败：结果不属于当前请求",
                                details={
                                    "expected": inp.request_id,
                                    "got": result.request_id,
                                },
                            )
                        self._log(
                            f"request={inp.request_id} 成功 "
                            f"{variant.key}@{cap.device_id} attempt={attempt_no} "
                            f"output={result.output!r} 副作用={result.side_effects}"
                        )
                        return result
                    finally:
                        # 无论成功失败都释放占用并唤醒下一排队者
                        self._release(cap.device_id, inp.request_id, variant.memory_mb)

            raise SchedulerError(
                ErrorKind.FALLBACK_EXHAUSTED,
                (
                    f"请求 {inp.request_id}: 后备链全部失败 "
                    f"({len(attempts)} 次尝试)"
                ),
                details={"request_id": inp.request_id, "attempts": attempts},
            )
        finally:
            with self._inflight_lock:
                self._inflight.discard(inp.request_id)
