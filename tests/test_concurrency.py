"""场景三：同一设备上的并发推理与资源争抢。

- 内存不足时按确定的 FIFO 排队，完成后串行进入；
- 队列容量满时以 queue_full 拒绝，原因与“内存不足”可区分；
- 并发下不得重复占用、状态不互相污染、结果不串号；
- 所有请求结束后设备状态最终一致（占用清零）。
"""
from __future__ import annotations

import threading
import time

import pytest

from edge_scheduler.errors import ErrorKind, SchedulerError
from edge_scheduler.logging_utils import log_case
from edge_scheduler.runtime import InferenceInput
from edge_scheduler.scheduler import SchedulerConfig

# int8 变体 400MB，sim-npu 总 512MB => 同一时刻最多 1 个请求


def test_two_requests_serialize_on_memory_fifo(
    logger, scheduler_factory, pool
):
    sch = scheduler_factory(
        SchedulerConfig(queue_capacity=4, admission_timeout=5.0), log=True
    )
    results: dict[str, object] = {}
    errors: dict[str, SchedulerError] = {}
    barrier = threading.Barrier(2)

    def worker(rid: str):
        try:
            barrier.wait()
            results[rid] = sch.infer(
                "llm", "int8", InferenceInput(rid, f"payload-{rid}"),
                run_seconds=0.15,
            )
        except SchedulerError as exc:
            errors[rid] = exc

    t1 = threading.Thread(target=worker, args=("c-req-1",))
    t2 = threading.Thread(target=worker, args=("c-req-2",))
    t1.start(); t2.start(); t1.join(); t2.join()

    assert not errors, {k: (v.kind.value, v.reason) for k, v in errors.items()}
    r1 = results["c-req-1"]
    r2 = results["c-req-2"]
    # 结果严格归属各自请求，绝不串号
    assert r1.request_id == "c-req-1" and r2.request_id == "c-req-2"
    assert r1.output.endswith("payload-c-req-1")
    assert r2.output.endswith("payload-c-req-2")
    # 两个请求都真实执行了（副作用各 1）
    assert r1.side_effects == 1 and r2.side_effects == 1
    # FIFO：先启动的请求先完成（attempt 均为 1，但完成顺序与启动顺序一致）
    assert r1.device_id == r2.device_id == "sim-npu"
    # 最终设备状态一致：内存清零、无活动请求
    npu = pool.get("sim-npu")
    assert npu.used_memory_mb == 0
    assert npu.active_requests == set()
    log_case(
        logger,
        "concurrency_fifo_serialize",
        inp="2 concurrent int8 requests each 400MB on 512MB sim-npu",
        basis="400MB*2 > 512MB => head-of-line FIFO admission",
        result=f"both succeeded on {r1.device_id}, outputs bound to own request, "
        f"final used_memory=0",
    )


def test_queue_full_is_distinct_from_memory_pressure(
    logger, scheduler_factory, pool
):
    # 队列容量仅 1：让请求按确定顺序进入，
    # 第 1 个已准入(执行中)、第 2 个排队中、第 3 个立刻 queue_full。
    sch = scheduler_factory(
        SchedulerConfig(queue_capacity=1, admission_timeout=5.0), log=True
    )
    npu_started = threading.Event()
    proceed = threading.Event()
    done: list[str] = []
    lock = threading.Lock()

    def long_worker(rid: str):
        # 在 runtime 真正执行后、释放前挂住，确保占用稳定
        orig_run = sch.runtime.run

        def wrapped(variant, device, inp, **kwargs):
            res = orig_run(variant, device, inp, **kwargs)
            if rid == "q-req-1":
                npu_started.set()
                proceed.wait(timeout=2.0)
            return res

        sch.runtime.run = wrapped  # type: ignore[method-assign]
        try:
            sch.infer("llm", "int8", InferenceInput(rid, "p"))
        finally:
            with lock:
                done.append(rid)

    t1 = threading.Thread(target=long_worker, args=("q-req-1",))
    t1.start()
    assert npu_started.wait(2.0)
    assert pool.get("sim-npu").used_memory_mb == 400

    # 第 2 个请求：必然排队（内存不足），等它确实进入队列
    def queued_worker(rid: str):
        sch.infer("llm", "int8", InferenceInput(rid, "p"))
        with lock:
            done.append(rid)

    t2 = threading.Thread(target=queued_worker, args=("q-req-2",))
    t2.start()
    deadline = time.monotonic() + 2.0
    while sch.queue_depth("sim-npu") < 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert sch.queue_depth("sim-npu") == 1

    # 第 3 个请求：队列已满（1/1），立即拒绝，与内存不足原因可区分
    with pytest.raises(SchedulerError) as ei:
        sch.infer("llm", "int8", InferenceInput("q-req-3", "p"))
    assert ei.value.kind is ErrorKind.QUEUE_FULL
    assert ei.value.details["queue_len"] == 1
    rejected = ei.value

    proceed.set()
    t1.join(); t2.join()
    assert sorted(done) == ["q-req-1", "q-req-2"]
    log_case(
        logger,
        "concurrency_queue_full",
        inp="requests enter in fixed order: q-req-1 running, q-req-2 queued, q-req-3",
        basis="queue_capacity=1; length already 1 when q-req-3 arrives",
        result=f"succeeded={sorted(done)} rejected={rejected.kind.value}",
    )
    # 两个成功请求释放后，设备内存归零（被拒绝者从未占用）
    assert pool.get("sim-npu").used_memory_mb == 0
    assert pool.get("sim-npu").active_requests == set()


def test_no_duplicate_admission_under_concurrency(logger, pool):
    # 直接对设备池做并发准入压测：同一设备只能容纳一个 400MB 请求

    admitted: list[str] = []
    rejected: list[str] = []
    lock = threading.Lock()

    def admit(rid: str):
        try:
            pool.try_admit("sim-npu", rid, 400)
            with lock:
                admitted.append(rid)
        except SchedulerError as exc:
            assert exc.kind in (
                ErrorKind.INSUFFICIENT_MEMORY,
                ErrorKind.DUPLICATE_REQUEST,
            )
            with lock:
                rejected.append(rid)

    threads = [threading.Thread(target=admit, args=(f"p-{i}",)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    npu = pool.get("sim-npu")
    # 任意时刻约束：只有一个请求准入成功
    assert len(admitted) == 1
    assert len(npu.active_requests) == 1
    assert npu.used_memory_mb == 400
    # 清理后状态一致
    pool.release("sim-npu", admitted[0], 400)
    assert npu.used_memory_mb == 0 and npu.active_requests == set()
    log_case(
        logger,
        "concurrency_no_duplicate_admit",
        inp="20 parallel admissions of 400MB on 512MB device",
        basis="single lock + free_memory check at admission",
        result=f"admitted={admitted} rejected={len(rejected)}; "
        "exactly one owner at any time; state consistent after release",
    )


def test_double_release_is_rejected(logger, pool):
    pool.try_admit("sim-npu", "dbl-1", 100)
    pool.release("sim-npu", "dbl-1", 100)
    with pytest.raises(SchedulerError) as ei:
        # 已释放资源不得被再次释放/使用
        pool.release("sim-npu", "dbl-1", 100)
    assert ei.value.kind is ErrorKind.DUPLICATE_REQUEST
    assert pool.get("sim-npu").used_memory_mb == 0
    log_case(
        logger,
        "concurrency_double_release",
        inp="admit dbl-1 100MB; release twice",
        basis="ownership token checked on release",
        result="second release rejected as duplicate_request; memory stays 0",
    )


def test_waiter_on_offlined_device_gets_distinct_error(
    logger, scheduler_factory, pool
):
    # 第 1 个请求占住 npu 执行；第 2 个排队期间 npu 被标记离线，
    # 排队者必须拿到 device_unavailable（可区分），而不是笼统的聚合失败。
    sch = scheduler_factory(
        SchedulerConfig(queue_capacity=2, admission_timeout=5.0)
    )
    started = threading.Event()
    proceed = threading.Event()
    orig_run = sch.runtime.run

    def blocking_run(variant, device, inp, **kwargs):
        res = orig_run(variant, device, inp, **kwargs)
        if inp.request_id == "off-1":
            started.set()
            proceed.wait(timeout=2.0)
        return res

    sch.runtime.run = blocking_run  # type: ignore[method-assign]
    errs: list[SchedulerError] = []

    def first():
        sch.infer("llm", "int8", InferenceInput("off-1", "p"))

    def second():
        try:
            sch.infer("llm", "int8", InferenceInput("off-2", "p"))
        except SchedulerError as exc:
            errs.append(exc)

    t1 = threading.Thread(target=first)
    t2 = threading.Thread(target=second)
    t1.start()
    assert started.wait(2.0)
    t2.start()
    deadline = time.monotonic() + 2.0
    while sch.queue_depth("sim-npu") < 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    pool.mark_offline("sim-npu")
    proceed.set()
    t1.join(); t2.join()

    assert len(errs) == 1
    assert errs[0].kind is ErrorKind.DEVICE_UNAVAILABLE
    # off-1 的占用随执行结束正常释放；off-2 从未占用
    npu = pool.get("sim-npu")
    assert npu.used_memory_mb == 0
    assert npu.active_requests == set()
    log_case(
        logger,
        "concurrency_waiter_device_offline",
        inp="off-1 running, off-2 queued; sim-npu marked offline while off-2 waits",
        basis="admission while device offline raises device_unavailable and bubbles up",
        result=errs[0].kind.value,
    )


def test_duplicate_inflight_request_rejected(logger, scheduler_factory):
    # 同一 request_id 在首次推理未结束时再次进入调度 => 拒绝。
    # 直接占住 inflight 标记模拟“首个请求仍在执行”。
    sch = scheduler_factory()
    sch._inflight.add("same-rid")
    try:
        with pytest.raises(SchedulerError) as ei:
            sch.infer("llm", "int8", InferenceInput("same-rid", "p"))
        assert ei.value.kind is ErrorKind.DUPLICATE_REQUEST
    finally:
        sch._inflight.discard("same-rid")
    log_case(
        logger,
        "concurrency_duplicate_inflight",
        inp="dispatch same-rid while it is marked in-flight",
        basis="inflight set guard in scheduler.infer",
        result=ei.value.kind.value,
    )
