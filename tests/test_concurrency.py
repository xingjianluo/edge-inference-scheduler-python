"""场景二：同一设备上的并发争抢，排队/拒绝确定且状态最终一致。"""

import logging
import threading

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

log = logging.getLogger("scenario.concurrency")


class CountingRuntime(SimulatedRuntime):
    """记录运行中最大并发数，验证设备并发上限从未被突破。"""

    def __init__(self, latency_s):
        super().__init__(latency_s=latency_s)
        self._lock = threading.Lock()
        self._inflight = 0
        self.max_inflight = 0

    def infer(self, device, variant, request_id, payload, timeout_s):
        with self._lock:
            self._inflight += 1
            self.max_inflight = max(self.max_inflight, self._inflight)
        try:
            return super().infer(device, variant, request_id, payload, timeout_s)
        finally:
            with self._lock:
                self._inflight -= 1


def _build(memory_mb, max_concurrency, latency_s, queue_timeout_s):
    cap = DeviceCapability(
        device_id="cpu0", kind="cpu",
        supported_quantizations=frozenset({"fp32"}),
        supported_operators=frozenset({"conv"}),
        total_memory_mb=memory_mb, max_concurrency=max_concurrency,
    )
    device = Device(cap)
    registry = VariantRegistry()
    registry.register(ModelVariant("m-fp32", "m", "fp32", frozenset({"conv"}),
                                   memory_mb=memory_mb))  # 一个请求即占满内存
    runtime = CountingRuntime(latency_s)
    sch = Scheduler([device], registry, runtime, queue_timeout_s=queue_timeout_s)
    return sch, device, runtime


def _run_concurrent(sch, request_ids):
    results, errors = {}, {}

    def work(rid):
        try:
            results[rid] = sch.submit(rid, "m-fp32", f"payload-{rid}")
        except SchedulerError as e:
            errors[rid] = e.code

    threads = [threading.Thread(target=work, args=(r,)) for r in request_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, errors


def test_contention_queues_then_rejects():
    """内存仅够一个请求：并发 3 个请求，1 个成功，其余排队超时后被确定性拒绝。"""
    sch, device, runtime = _build(memory_mb=100, max_concurrency=2,
                                  latency_s=0.3, queue_timeout_s=0.05)
    results, errors = _run_concurrent(sch, ["r1", "r2", "r3"])

    assert len(results) == 1
    assert set(errors.values()) == {ErrorCode.RESOURCE_EXHAUSTED}
    assert len(results) + len(errors) == 3
    # 并发上限从未被突破
    assert runtime.max_inflight == 1
    # 结果只返回给对应请求，互不污染
    for rid, res in results.items():
        assert res.request_id == rid
        assert f"payload-{rid}" in res.output
    # 设备状态最终一致：无泄漏、无重复占用
    assert device.used_memory_mb == 0
    assert device.active_count == 0
    for rid in ["r1", "r2", "r3"]:
        normalized = "success" if rid in results else errors[rid].value
        log_decision(sch, rid, normalized)
    log.info("final device state: used=%dMB active=%d",
             device.used_memory_mb, device.active_count)


def test_contention_queue_succeeds_with_longer_wait():
    """排队超时时间足够时，两个请求串行成功，不重复占用资源。"""
    sch, device, runtime = _build(memory_mb=100, max_concurrency=2,
                                  latency_s=0.1, queue_timeout_s=2.0)
    results, errors = _run_concurrent(sch, ["r1", "r2"])

    assert not errors
    assert len(results) == 2
    assert runtime.max_inflight == 1  # 内存约束使二者串行
    assert {r.output for r in results.values()} == {
        f"output[r1]:payload-r1:m-fp32@cpu0",
        f"output[r2]:payload-r2:m-fp32@cpu0",
    }
    assert device.used_memory_mb == 0
    assert device.active_count == 0
    for rid in ["r1", "r2"]:
        log_decision(sch, rid, "success")
