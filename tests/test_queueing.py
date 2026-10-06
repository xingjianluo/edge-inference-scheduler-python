"""场景四：等待队列治理 —— 容量背压、取消/放弃、优先级准入与防饿死。"""

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
from conftest import log_decision

log = logging.getLogger("scenario.queueing")


def _build(*, memory_mb=100, max_concurrency=1, latency_s=0.3,
           queue_timeout_s=2.0, max_queue_size=8, aging_after_s=1.0):
    cap = DeviceCapability(
        device_id="cpu0", kind="cpu",
        supported_quantizations=frozenset({"fp32"}),
        supported_operators=frozenset({"conv"}),
        total_memory_mb=memory_mb, max_concurrency=max_concurrency,
    )
    device = Device(cap, max_queue_size=max_queue_size,
                    aging_after_s=aging_after_s)
    registry = VariantRegistry()
    registry.register(ModelVariant("m-fp32", "m", "fp32", frozenset({"conv"}),
                                   memory_mb=memory_mb))  # 一个请求即占满内存
    runtime = SimulatedRuntime(latency_s=latency_s)
    sch = Scheduler([device], registry, runtime, queue_timeout_s=queue_timeout_s)
    return sch, device, runtime


def _wait_for(cond, timeout=2.0):
    """轮询等待条件成立，避免依赖固定 sleep 的时序假设。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.005)
    return False


class Submitter:
    """在后台线程提交请求，收集结果/错误，保证时序可控。"""

    def __init__(self, sch):
        self._sch = sch
        self.results = {}
        self.errors = {}
        self._threads = []

    def submit(self, rid, variant="m-fp32", priority=0):
        def work():
            try:
                self.results[rid] = self._sch.submit(
                    rid, variant, f"payload-{rid}", priority=priority)
            except SchedulerError as e:
                self.errors[rid] = e.code
        t = threading.Thread(target=work)
        t.start()
        self._threads.append(t)

    def join_all(self):
        for t in self._threads:
            t.join()


def test_queue_full_backpressure_distinct_from_resource_exhausted():
    """队列满 -> QUEUE_FULL 立即拒绝；队列有位置但资源腾不出 -> RESOURCE_EXHAUSTED。"""
    # 场景 A：等待队列容量 1，r1 占用、r2 排队，r3 到顶被立即拒绝
    sch, device, _runtime = _build(max_queue_size=1, latency_s=0.4)
    sub = Submitter(sch)
    sub.submit("r1")
    assert _wait_for(lambda: device.active_count == 1)
    sub.submit("r2")
    assert _wait_for(lambda: device.pending_count == 1)

    t0 = time.monotonic()
    with pytest.raises(SchedulerError) as exc_info:
        sch.submit("r3", "m-fp32", "payload-r3")
    elapsed = time.monotonic() - t0
    assert exc_info.value.code is ErrorCode.QUEUE_FULL
    assert elapsed < 0.2  # 背压是立即拒绝，不排队等待
    log_decision(sch, "r3", exc_info.value.code.value)
    sub.join_all()

    # 场景 B：队列有位置，但排队超时资源仍腾不出 -> RESOURCE_EXHAUSTED
    sch2, device2, _ = _build(max_queue_size=8, latency_s=0.3,
                              queue_timeout_s=0.05)
    sub2 = Submitter(sch2)
    sub2.submit("r1")
    assert _wait_for(lambda: device2.active_count == 1)
    sub2.submit("r2")
    sub2.join_all()
    assert sub2.errors["r2"] is ErrorCode.RESOURCE_EXHAUSTED
    log_decision(sch2, "r2", sub2.errors["r2"].value)

    # 两种拒绝原因可单独断言、互不相同
    assert ErrorCode.QUEUE_FULL is not ErrorCode.RESOURCE_EXHAUSTED
    log.info("backpressure(QUEUE_FULL) != transient(RESOURCE_EXHAUSTED): "
             "asserted separately")


def test_cancelled_waiter_never_acquires_nor_produces_result():
    """等待中的请求被取消：彻底出队，资源空出来也不会被它占用，不产出结果。"""
    sch, device, runtime = _build(latency_s=0.3, queue_timeout_s=5.0)
    sub = Submitter(sch)
    sub.submit("r1")
    assert _wait_for(lambda: device.active_count == 1)
    sub.submit("r2")
    assert _wait_for(lambda: device.pending_count == 1)

    assert sch.cancel("r2") is True
    assert sch.cancel("r2") is False  # 已取消，重复取消找不到等待者
    sub.join_all()

    assert sub.errors["r2"] is ErrorCode.REQUEST_CANCELLED
    assert "r2" not in sub.results
    # r2 从未进入运行时：没有产出任何结果
    assert all(rid != "r2" for rid, _, _ in runtime.call_log)
    log_decision(sch, "r2", sub.errors["r2"].value)

    # 资源空出来后由新请求占用，而不是被已取消的 r2 占用
    result = sch.submit("r3", "m-fp32", "payload-r3")
    assert result.request_id == "r3"
    log_decision(sch, "r3", "success")
    # 设备状态最终一致：无泄漏、无残留等待者
    assert device.used_memory_mb == 0
    assert device.active_count == 0
    assert device.pending_count == 0


def test_priority_admission_order_then_fifo_within_same_priority():
    """资源紧张时高优先级先拿到资源；同一优先级严格先到先得。"""
    sch, device, runtime = _build(latency_s=0.25, queue_timeout_s=5.0)
    sub = Submitter(sch)
    sub.submit("r1")  # 占用者
    assert _wait_for(lambda: device.active_count == 1)

    # 依次入队：低、高、中；每起一个线程都等它真正入队，保证入队顺序确定
    sub.submit("r-low", priority=0)
    assert _wait_for(lambda: device.pending_count == 1)
    sub.submit("r-high", priority=10)
    assert _wait_for(lambda: device.pending_count == 2)
    sub.submit("r-mid", priority=5)
    assert _wait_for(lambda: device.pending_count == 3)
    log.info("pending order before release: %s", device.pending_ids())

    sub.join_all()
    assert not sub.errors
    # 准入顺序：占用者 -> 高 -> 中 -> 低
    order = [rid for rid, _, _ in runtime.call_log]
    assert order == ["r1", "r-high", "r-mid", "r-low"]
    for rid in ["r-high", "r-mid", "r-low"]:
        log_decision(sch, rid, "success")

    # 同优先级：FIFO
    sch2, device2, runtime2 = _build(latency_s=0.2, queue_timeout_s=5.0)
    sub2 = Submitter(sch2)
    sub2.submit("r1")
    assert _wait_for(lambda: device2.active_count == 1)
    for i, rid in enumerate(["r-a", "r-b", "r-c"]):
        sub2.submit(rid, priority=3)
        assert _wait_for(lambda n=i + 1: device2.pending_count == n)
    sub2.join_all()
    order2 = [rid for rid, _, _ in runtime2.call_log]
    assert order2 == ["r1", "r-a", "r-b", "r-c"]
    log.info("same-priority FIFO order: %s", order2)


def test_low_priority_not_starved_by_aging():
    """低优先级等待超过 aging_after_s 后自动升级，先于后来的高优先级拿到资源。"""
    sch, device, runtime = _build(latency_s=0.4, queue_timeout_s=5.0,
                                  aging_after_s=0.15)
    sub = Submitter(sch)
    sub.submit("r1")  # 占用者，0.4s 后释放
    assert _wait_for(lambda: device.active_count == 1)
    sub.submit("r-low", priority=0)
    assert _wait_for(lambda: device.pending_count == 1)

    time.sleep(0.3)  # r-low 已等待 0.3s >= aging_after_s，被提升到最高优先段
    sub.submit("r-high", priority=10)  # 此时入队的高优先级尚未老化
    assert _wait_for(lambda: device.pending_count == 2)

    sub.join_all()
    assert not sub.errors
    order = [rid for rid, _, _ in runtime.call_log]
    # r-low 因老化升级排在 r-high 之前：低优先级不会被无限插队
    assert order == ["r1", "r-low", "r-high"]
    log.info("aging promoted r-low ahead of later high-priority: %s", order)
    for rid in ["r-low", "r-high"]:
        log_decision(sch, rid, "success")


def test_priority_never_bypasses_hard_constraints():
    """优先级再高也不能绕过硬性匹配：不支持的量化仍然被拒绝，不变体替换。"""
    cap = DeviceCapability(
        device_id="cpu0", kind="cpu",
        supported_quantizations=frozenset({"fp32"}),
        supported_operators=frozenset({"conv"}),
        total_memory_mb=512, max_concurrency=1,
    )
    registry = VariantRegistry()
    registry.register(ModelVariant("m-int8", "m", "int8", frozenset({"conv"}),
                                   memory_mb=64))
    sch = Scheduler([Device(cap)], registry, SimulatedRuntime())

    with pytest.raises(SchedulerError) as exc_info:
        sch.submit("r-prio", "m-int8", "payload", priority=1000)
    assert exc_info.value.code is ErrorCode.CAPABILITY_MISSING
    log_decision(sch, "r-prio", exc_info.value.code.value)
    # 运行时从未被调用：没有静默换用任何变体
    assert sch._runtime.call_log == []
