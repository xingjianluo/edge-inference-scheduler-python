"""场景四：等待队列治理——背压、取消/放弃、优先级准入与防饿死。

所有用例通过 BlockingRuntime 显式控制每次推理的放行时机，
使准入顺序的断言完全确定、可复现；日志打印输入、判定依据与归一化结果。
"""

import logging
import threading
import time

import pytest

from edge_scheduler import (
    Device,
    DeviceCapability,
    ErrorCode,
    ModelVariant,
    Scheduler,
    SchedulerError,
    SimulatedRuntime,
    VariantRegistry,
)

log = logging.getLogger("scenario.queue")


class BlockingRuntime(SimulatedRuntime):
    """每次 infer 阻塞，直到测试按 request_id 显式放行；记录启动顺序。"""

    def __init__(self):
        super().__init__(latency_s=0.0)
        self._lock = threading.Lock()
        self.start_log: list[str] = []  # 请求进入推理的顺序（即准入顺序）
        self._events: dict[str, threading.Event] = {}

    def release(self, request_id: str) -> None:
        with self._lock:
            ev = self._events.setdefault(request_id, threading.Event())
        ev.set()

    def infer(self, device, variant, request_id, payload, timeout_s):
        with self._lock:
            self.start_log.append(request_id)
            ev = self._events.setdefault(request_id, threading.Event())
        ev.wait(timeout=5.0)  # 兜底，防止测试逻辑错误时挂死
        return super().infer(device, variant, request_id, payload, timeout_s)


def wait_until(pred, timeout_s=2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return False


def _build(*, max_queue_depth, queue_timeout_s):
    """单设备：内存仅够一个请求，并发槽位 1，排队行为完全确定。"""
    cap = DeviceCapability(
        device_id="cpu0", kind="cpu",
        supported_quantizations=frozenset({"fp32"}),
        supported_operators=frozenset({"conv"}),
        total_memory_mb=100, max_concurrency=1,
    )
    device = Device(cap, max_queue_depth=max_queue_depth)
    registry = VariantRegistry()
    registry.register(ModelVariant("m-fp32", "m", "fp32", frozenset({"conv"}),
                                   memory_mb=100))  # 一个请求即占满内存
    runtime = BlockingRuntime()
    sch = Scheduler([device], registry, runtime, queue_timeout_s=queue_timeout_s)
    return sch, device, runtime


def _submit_async(sch, results, errors, rid, *, priority=0, cancel_event=None):
    def work():
        try:
            results[rid] = sch.submit(
                rid, "m-fp32", f"payload-{rid}",
                priority=priority, cancel_event=cancel_event,
            )
        except SchedulerError as e:
            errors[rid] = e.code

    t = threading.Thread(target=work, daemon=True)
    t.start()
    return t


def log_decision_for(sch, request_id, normalized):
    """打印该请求的输入、每次尝试的判定依据与归一化结果。"""
    decision = next(d for d in sch.decision_log if d.request_id == request_id)
    log.info("input request_id=%s", request_id)
    for a in decision.attempts:
        log.info("  attempt variant=%s device=%s -> %s %s",
                 a.variant_id, a.device_id, a.outcome, a.reason)
    log.info("normalized result: %s", normalized)


def test_queue_full_backpressure_distinct_from_resource_exhausted():
    """队列容量 1：占满后新请求立即被 QUEUE_FULL 拒绝（背压），
    与"已入队但等资源超时"的 RESOURCE_EXHAUSTED 可区分。"""
    sch, device, runtime = _build(max_queue_depth=1, queue_timeout_s=2.0)
    results, errors = {}, {}

    t_block = _submit_async(sch, results, errors, "blocker")
    assert wait_until(lambda: runtime.start_log == ["blocker"])
    t_w1 = _submit_async(sch, results, errors, "w1")
    assert wait_until(lambda: device.pending_count == 1)

    # 队列已满：w2 必须立即被拒绝，而不是等到 queue_timeout
    t0 = time.monotonic()
    with pytest.raises(SchedulerError) as exc_info:
        sch.submit("w2", "m-fp32", "payload-w2")
    elapsed = time.monotonic() - t0
    assert exc_info.value.code is ErrorCode.QUEUE_FULL
    assert elapsed < 1.0, "QUEUE_FULL 必须是立即拒绝的背压"
    assert "w2" not in runtime.start_log  # 被拒绝的请求绝不进入推理

    # 放行后 w1 正常完成，设备状态最终一致
    runtime.release("blocker")
    assert wait_until(lambda: "w1" in runtime.start_log)
    runtime.release("w1")
    t_block.join()
    t_w1.join()
    assert set(results) == {"blocker", "w1"}
    assert device.pending_count == 0
    assert device.active_count == 0
    assert device.used_memory_mb == 0
    log_decision_for(sch, "w2", ErrorCode.QUEUE_FULL.value)
    log_decision_for(sch, "w1", "success")


def test_resource_exhausted_when_queued_but_timeout():
    """队列还有位置、只是资源一时腾不出来：入队后超时 -> RESOURCE_EXHAUSTED，
    与 QUEUE_FULL 是两种可单独断言的拒绝；放弃后等待者彻底出队。"""
    sch, device, runtime = _build(max_queue_depth=4, queue_timeout_s=0.05)
    results, errors = {}, {}

    t_block = _submit_async(sch, results, errors, "blocker")
    assert wait_until(lambda: runtime.start_log == ["blocker"])
    t_w1 = _submit_async(sch, results, errors, "w1")
    t_w1.join()  # 0.05s 后自动放弃

    assert errors["w1"] is ErrorCode.RESOURCE_EXHAUSTED
    assert errors["w1"] is not ErrorCode.QUEUE_FULL
    # 放弃后彻底出队：不占队列、不占用资源、之后也不会再冒出结果
    assert device.pending_count == 0
    runtime.release("blocker")
    t_block.join()
    time.sleep(0.1)
    assert "w1" not in runtime.start_log

    # 资源空出后新请求立即可调度，说明放弃者没有残留占用
    res = sch.submit("w2", "m-fp32", "payload-w2")
    assert res.request_id == "w2"
    assert device.active_count == 0
    assert device.used_memory_mb == 0
    log_decision_for(sch, "w1", ErrorCode.RESOURCE_EXHAUSTED.value)
    log_decision_for(sch, "w2", "success")


def test_cancelled_waiter_never_acquires_nor_produces_result():
    """等待中的请求被取消：彻底出队，资源空出也不被它占用、不产出结果，
    且不打乱同队列其他请求的先后。"""
    sch, device, runtime = _build(max_queue_depth=4, queue_timeout_s=5.0)
    results, errors = {}, {}
    cancel_w1 = threading.Event()

    t_block = _submit_async(sch, results, errors, "blocker")
    assert wait_until(lambda: runtime.start_log == ["blocker"])
    t_w1 = _submit_async(sch, results, errors, "w1", cancel_event=cancel_w1)
    assert wait_until(lambda: device.pending_count == 1)
    t_w2 = _submit_async(sch, results, errors, "w2")
    assert wait_until(lambda: device.pending_count == 2)

    cancel_w1.set()
    t_w1.join()
    assert errors["w1"] is ErrorCode.REQUEST_CANCELLED
    assert device.pending_count == 1  # w2 仍在队列中，相对顺序不受影响

    # 资源空出后：w2 被准入，w1 绝不出现
    runtime.release("blocker")
    assert wait_until(lambda: "w2" in runtime.start_log)
    runtime.release("w2")
    t_block.join()
    t_w2.join()
    assert "w1" not in runtime.start_log
    assert set(results) == {"blocker", "w2"}
    assert device.pending_count == 0
    assert device.active_count == 0
    assert device.used_memory_mb == 0
    log_decision_for(sch, "w1", ErrorCode.REQUEST_CANCELLED.value)
    log_decision_for(sch, "w2", "success")


def test_priority_admission_order_and_fifo_within_same_priority():
    """高优先级先准入；同优先级严格 FIFO。"""
    sch, device, runtime = _build(max_queue_depth=8, queue_timeout_s=5.0)
    results, errors = {}, {}

    t_block = _submit_async(sch, results, errors, "blocker")
    assert wait_until(lambda: runtime.start_log == ["blocker"])
    threads = [t_block]
    # 依次入队：低、高、低（逐个确认入队，保证队列顺序确定）
    threads.append(_submit_async(sch, results, errors, "low1", priority=0))
    assert wait_until(lambda: device.pending_count == 1)
    threads.append(_submit_async(sch, results, errors, "high1", priority=5))
    assert wait_until(lambda: device.pending_count == 2)
    threads.append(_submit_async(sch, results, errors, "low2", priority=0))
    assert wait_until(lambda: device.pending_count == 3)

    runtime.release("blocker")
    assert wait_until(lambda: "high1" in runtime.start_log)
    runtime.release("high1")
    assert wait_until(lambda: "low1" in runtime.start_log)
    runtime.release("low1")
    assert wait_until(lambda: "low2" in runtime.start_log)
    runtime.release("low2")
    for t in threads:
        t.join()

    # 准入顺序：高优先级插队，两个低优先级之间仍 FIFO
    assert runtime.start_log == ["blocker", "high1", "low1", "low2"]
    assert not errors
    assert device.pending_count == 0
    assert device.active_count == 0
    for rid in ["high1", "low1", "low2"]:
        log_decision_for(sch, rid, "success")


def test_low_priority_not_starved():
    """防饿死：低优先级最多被插队（优先级差）次，之后必然先于新的高优先级准入。"""
    sch, device, runtime = _build(max_queue_depth=8, queue_timeout_s=5.0)
    results, errors = {}, {}

    t_block = _submit_async(sch, results, errors, "blocker")
    assert wait_until(lambda: runtime.start_log == ["blocker"])
    threads = [t_block]
    threads.append(_submit_async(sch, results, errors, "low", priority=0))
    assert wait_until(lambda: device.pending_count == 1)
    threads.append(_submit_async(sch, results, errors, "high1", priority=1))
    assert wait_until(lambda: device.pending_count == 2)

    # 第一次资源空出：high1 插队成功，low 被插 1 次（= 优先级差），有效优先级追平
    runtime.release("blocker")
    assert wait_until(lambda: "high1" in runtime.start_log)
    # high1 运行期间再来一个同级高优先级
    threads.append(_submit_async(sch, results, errors, "high2", priority=1))
    assert wait_until(lambda: device.pending_count == 2)

    # 第二次资源空出：low 有效优先级已追平且入队更早，必须先于 high2 准入
    runtime.release("high1")
    assert wait_until(lambda: "low" in runtime.start_log)
    runtime.release("low")
    assert wait_until(lambda: "high2" in runtime.start_log)
    runtime.release("high2")
    for t in threads:
        t.join()

    assert runtime.start_log == ["blocker", "high1", "low", "high2"]
    assert not errors
    log.info("starvation bound verified: low admitted after exactly 1 skip "
             "(priority gap = 1)")
    for rid in ["low", "high1", "high2"]:
        log_decision_for(sch, rid, "success")


def test_priority_never_bypasses_hard_constraints():
    """优先级再高也不能绕过量化/算子/内存硬性约束，也不能换用不符变体。"""
    cap = DeviceCapability(
        device_id="cpu0", kind="cpu",
        supported_quantizations=frozenset({"fp32"}),  # 不支持 int8
        supported_operators=frozenset({"conv"}),
        total_memory_mb=100, max_concurrency=1,
    )
    device = Device(cap, max_queue_depth=4)
    registry = VariantRegistry()
    registry.register(ModelVariant("m-int8", "m", "int8", frozenset({"conv"}),
                                   memory_mb=16))
    sch = Scheduler([device], registry, BlockingRuntime(), queue_timeout_s=0.05)

    with pytest.raises(SchedulerError) as exc_info:
        sch.submit("vip", "m-int8", "payload-vip", priority=100)
    assert exc_info.value.code is ErrorCode.CAPABILITY_MISSING
    assert device.pending_count == 0
    assert device.active_count == 0
    log_decision_for(sch, "vip", ErrorCode.CAPABILITY_MISSING.value)
