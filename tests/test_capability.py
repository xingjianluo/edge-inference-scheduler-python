"""场景一：能力缺失。

设备不满足变体的硬性约束（量化精度/算子集）时必须拒绝，
拒绝原因可区分（缺精度 vs 缺算子），设备状态保持一致。
"""
from __future__ import annotations

from edge_scheduler.devices import LocalProbe
from edge_scheduler.errors import ErrorKind, SchedulerError
from edge_scheduler.logging_utils import log_case
from edge_scheduler.variants import check_compatibility


def test_local_probe_is_deterministic_and_offline_free():
    a = LocalProbe().detect()
    b = LocalProbe().detect()
    assert [(c.device_id, c.dtypes, c.ops, c.memory_mb) for c in a] == [
        (c.device_id, c.dtypes, c.ops, c.memory_mb) for c in b
    ]
    cpu = next(c for c in a if c.device_id == "local-cpu")
    assert cpu.is_simulated is False
    assert "fp32" in cpu.dtypes


def test_int8_variant_rejected_on_devices_missing_dtype_and_ops(logger, pool):
    caps = [s.capability for s in pool.list_devices()]
    from tests.conftest import make_variant

    variant = make_variant("int8", {"int8"}, {"matmul", "rope"}, 400)
    results = {}
    for cap in caps:
        results[cap.device_id] = check_compatibility(variant, cap)

    # sim-npu 满足全部硬约束
    assert results["sim-npu"] is None
    # local-cpu 同时缺 int8 精度与 rope 算子，原因可区分
    cpu_miss = results["local-cpu"]
    assert cpu_miss is not None
    assert cpu_miss.missing_dtypes == frozenset({"int8"})
    assert cpu_miss.missing_ops == frozenset({"rope"})
    # sim-gpu 缺 int8 与 rope
    gpu_miss = results["sim-gpu"]
    assert gpu_miss.missing_dtypes == frozenset({"int8"})
    assert "rope" in gpu_miss.missing_ops

    log_case(
        logger,
        "capability_missing",
        inp="variant=int8 needs dtypes={int8}, ops={matmul,rope}",
        basis={
            dev: (None if m is None else list(m.reasons))
            for dev, m in results.items()
        },
        result="only sim-npu compatible; others rejected with distinct reasons",
    )


def test_only_missing_dtype_is_distinguishable_from_missing_op(logger, pool):
    from tests.conftest import make_variant

    caps = {s.capability.device_id: s.capability for s in pool.list_devices()}
    gpu = caps["sim-gpu"]
    # 算子都支持，但精度要求 fp16 之外的 bf16（仅缺精度）
    v_dtype = make_variant("bf16", {"bf16"}, {"matmul"}, 100)
    miss = check_compatibility(v_dtype, gpu)
    assert miss is not None and miss.missing_dtypes == frozenset({"bf16"})
    assert miss.missing_ops == frozenset()

    # 精度支持 fp16，但要求 gpu 没有的 rope 算子（仅缺算子）
    v_op = make_variant("fp16-rope", {"fp16"}, {"rope"}, 100)
    miss2 = check_compatibility(v_op, gpu)
    assert miss2 is not None and miss2.missing_ops == frozenset({"rope"})
    assert miss2.missing_dtypes == frozenset()

    log_case(
        logger,
        "capability_reason_distinct",
        inp="two variants: need bf16 / need rope on sim-gpu",
        basis={
            "bf16": list(miss.reasons),
            "fp16-rope": list(miss2.reasons),
        },
        result="missing_dtypes and missing_ops reported separately",
    )


def test_scheduler_rejects_and_reports_each_device_reason(
    logger, registry, pool, scheduler_factory
):
    from edge_scheduler.runtime import InferenceInput

    # 构造一个没有任何设备能承载的变体（int4 精度 + 专有算子），不放入后备链
    from edge_scheduler.variants import ModelVariant

    ghost = ModelVariant(
        model="llm",
        variant_id="int4-ghost",
        required_dtypes=frozenset({"int4"}),
        required_ops=frozenset({"quantized_matmul", "ghost_op"}),
        memory_mb=10,
    )
    registry.register(ghost)
    sch = scheduler_factory(log=True)
    try:
        sch.infer("llm", "int4-ghost", InferenceInput("req-cap", "x"))
        raise AssertionError("应当被拒绝")
    except SchedulerError as exc:
        assert exc.kind is ErrorKind.FALLBACK_EXHAUSTED
        kinds = {a["kind"] for a in exc.details["attempts"]}
        assert kinds == {ErrorKind.CAPABILITY_MISSING.value}
        # 三台设备都给出原因，且都点明 int4 缺失
        assert len(exc.details["attempts"]) == 3
        assert all("missing_dtypes=['int4']" in " ".join(a["reasons"])
                   for a in exc.details["attempts"])
        log_case(
            logger,
            "capability_missing_scheduler",
            inp="model=llm variant=int4-ghost request=req-cap",
            basis=exc.details["attempts"],
            result=f"{exc.kind.value}: {exc.reason}",
        )

    # 拒绝后设备状态最终一致：无占用、全部在线
    for st in pool.list_devices():
        assert st.used_memory_mb == 0
        assert st.active_requests == set()
