"""场景二：变体不兼容。

- 指定变体在全部设备上不满足硬约束 => VARIANT 调度失败，
  归一化为 FALLBACK_EXHAUSTED 且每步原因是 capability_missing；
- 默认严格模式：即便模型注册了其它变体，也不得静默换用；
- 只有调用方显式 use_chain=True 时，才按后备链确定顺序降级。
"""
from __future__ import annotations

import pytest

from edge_scheduler.errors import ErrorKind, SchedulerError
from edge_scheduler.logging_utils import log_case
from edge_scheduler.runtime import InferenceInput
from edge_scheduler.variants import ModelVariant


def test_int8_variant_only_lands_on_npu(logger, scheduler_factory):
    sch = scheduler_factory()
    res = sch.infer("llm", "int8", InferenceInput("req-v1", "p"))
    assert res.device_id == "sim-npu"
    assert res.variant_id == "int8"
    log_case(
        logger,
        "variant_match_ok",
        inp="request=req-v1 preferred=int8",
        basis="int8 needs {int8}+rope; only sim-npu satisfies",
        result=f"dispatched {res.variant_id} -> {res.device_id}",
    )


def test_strict_mode_does_not_silently_switch_variant(
    logger, registry, pool, scheduler_factory
):
    # 注册一个任何设备都不支持的 int4 变体
    registry.register(
        ModelVariant(
            model="llm",
            variant_id="int4",
            required_dtypes=frozenset({"int4"}),
            required_ops=frozenset({"matmul"}),
            memory_mb=10,
        )
    )
    sch = scheduler_factory(log=True)
    with pytest.raises(SchedulerError) as ei:
        sch.infer("llm", "int4", InferenceInput("req-v2", "p"))
    err = ei.value
    assert err.kind is ErrorKind.FALLBACK_EXHAUSTED
    attempted = {(a["variant_id"], a["device_id"]) for a in err.details["attempts"]}
    # 严格模式只尝试了 int4，没有偷偷换成 int8/fp16/fp32
    assert {v for v, _ in attempted} == {"int4"}
    assert all(
        a["kind"] == ErrorKind.CAPABILITY_MISSING.value
        for a in err.details["attempts"]
    )
    log_case(
        logger,
        "variant_incompatible_no_silent_switch",
        inp="request=req-v2 preferred=int4",
        basis=err.details["attempts"],
        result=f"rejected {err.kind.value}; variants actually tried={sorted({v for v,_ in attempted})}",
    )
    # 设备状态一致：没有任何残留占用
    assert all(s.used_memory_mb == 0 and s.active_requests == set()
               for s in pool.list_devices())


def test_explicit_chain_falls_back_in_deterministic_order(
    logger, scheduler_factory, pool
):
    # 指定 int8，但显式允许后备链；int8 第一次超时 => 必须降级到链中下一可用 fp16
    from edge_scheduler.runtime import FailureSpec

    sch = scheduler_factory(log=True)
    res = sch.infer(
        "llm",
        "int8",
        InferenceInput("req-v3", "p"),
        failure=FailureSpec(ErrorKind.INFERENCE_TIMEOUT, frozenset({1})),
        use_chain=True,
    )
    # 后备链 int8 -> fp16 -> fp32：int8 失败后落到 sim-gpu 的 fp16
    assert res.variant_id == "fp16"
    assert res.device_id == "sim-gpu"
    assert res.attempt == 2
    log_case(
        logger,
        "variant_explicit_fallback",
        inp="request=req-v3 preferred=int8 use_chain=True fail=timeout@attempt1",
        basis="chain order int8(npu) -> fp16(gpu) -> fp32(any)",
        result=f"dispatched {res.variant_id} -> {res.device_id} attempt={res.attempt}",
    )
    # npu / gpu 占用都已释放
    assert pool.get("sim-npu").used_memory_mb == 0
    assert pool.get("sim-gpu").used_memory_mb == 0


def test_preferred_variant_missing_raises_distinct_error(
    logger, scheduler_factory
):
    sch = scheduler_factory()
    with pytest.raises(SchedulerError) as ei:
        sch.infer("llm", "does-not-exist", InferenceInput("req-v4", "p"))
    assert ei.value.kind is ErrorKind.VARIANT_INCOMPATIBLE
    assert ei.value.details["variant_id"] == "does-not-exist"
    log_case(
        logger,
        "variant_not_registered",
        inp="request=req-v4 preferred=does-not-exist",
        basis="registry lookup",
        result=ei.value.kind.value,
    )
