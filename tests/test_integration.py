"""端到端集成：探测 → 注册匹配 → 并发排队 → 失败降级 → 状态一致。"""
from __future__ import annotations

import threading

from edge_scheduler.devices import DevicePool, LocalProbe
from edge_scheduler.errors import ErrorKind, SchedulerError
from edge_scheduler.logging_utils import log_case
from edge_scheduler.runtime import FailureSpec, InferenceInput, SimulatedRuntime
from edge_scheduler.scheduler import EdgeScheduler, SchedulerConfig
from edge_scheduler.variants import FallbackChain, ModelVariant, VariantRegistry


def _build():
    reg = VariantRegistry()
    reg.register(
        ModelVariant("detector", "int8", frozenset({"int8"}),
                     frozenset({"matmul", "rope"}), 400)
    )
    reg.register(
        ModelVariant("detector", "fp32", frozenset({"fp32"}),
                     frozenset({"matmul"}), 128)
    )
    reg.register_chain(FallbackChain("detector", ("int8", "fp32")))
    pool = DevicePool(LocalProbe().detect())
    sch = EdgeScheduler(
        reg, pool, SimulatedRuntime(),
        SchedulerConfig(queue_capacity=4, admission_timeout=5.0),
    )
    return reg, pool, sch


def test_end_to_end_mixed_workload(logger):
    _reg, pool, sch = _build()
    outcomes: dict[str, object] = {}
    lock = threading.Lock()

    # 2 个请求争抢只能容纳 1 个 int8(400MB/512MB) 的 npu：FIFO 串行
    # 第 1 个首次超时，走后备链到 fp32(local-cpu)；第 2 个直接 int8 成功
    def req(rid: str, fail: bool):
        try:
            res = sch.infer(
                "detector",
                "int8",
                InferenceInput(rid, f"img-{rid}"),
                failure=FailureSpec(ErrorKind.INFERENCE_TIMEOUT) if fail else None,
                use_chain=True,
            )
            with lock:
                outcomes[rid] = res
        except SchedulerError as exc:
            with lock:
                outcomes[rid] = exc

    t1 = threading.Thread(target=req, args=("e2e-1", True))
    t2 = threading.Thread(target=req, args=("e2e-2", False))
    t1.start(); t2.start(); t1.join(); t2.join()

    r1 = outcomes["e2e-1"]
    r2 = outcomes["e2e-2"]
    assert not isinstance(r1, SchedulerError), r1
    assert not isinstance(r2, SchedulerError), r2
    # 结果归属正确，不串号
    assert r1.request_id == "e2e-1" and r1.output.endswith("img-e2e-1")
    assert r2.request_id == "e2e-2" and r2.output.endswith("img-e2e-2")
    # r1 超时降级到 fp32，r2 严格占 int8 npu（具体先后不影响最终落点判定：
    # int8 只有 npu；fp32 兜底在任意设备）
    assert r2.variant_id == "int8" and r2.device_id == "sim-npu"
    assert r1.variant_id == "fp32"
    # 每个请求恰好一次真实副作用
    assert sch.runtime.side_effect_count("e2e-1") == 1
    assert sch.runtime.side_effect_count("e2e-2") == 1

    # 最终所有设备占用归零、请求集合清空
    for st in pool.list_devices():
        assert st.used_memory_mb == 0, st.capability.device_id
        assert st.active_requests == set()

    log_case(
        logger,
        "end_to_end",
        inp="2 detector requests: e2e-1(timeout+chain), e2e-2(int8)",
        basis=(
            "int8 400MB serialized on sim-npu(512MB); "
            "e2e-1 attempt1 timeout -> fp32 fallback; e2e-2 int8 on npu"
        ),
        result={
            "e2e-1": f"{r1.variant_id}@{r1.device_id}",
            "e2e-2": f"{r2.variant_id}@{r2.device_id}",
            "final_memory": {s.capability.device_id: s.used_memory_mb
                             for s in pool.list_devices()},
        },
    )
